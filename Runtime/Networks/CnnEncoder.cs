using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// A minimal pure-C# CNN encoder: one or more strided-conv+ReLU layers followed by
/// a linear projection that produces a fixed-size embedding vector.
///
/// Input layout: row-major HWC float[] (normalized pixels from ObservationBuffer.AddImage).
/// Output: float[OutputSize] embedding for merging into the shared MLP trunk.
///
/// All arithmetic intentionally avoids unsafe code and external dependencies so it works
/// on every platform Godot supports.
/// </summary>
internal sealed class CnnEncoder
{
    private readonly ConvLayer[] _convLayers;
    private readonly LinearLayer _projection;
    private readonly int _inputWidth;
    private readonly int _inputHeight;
    private readonly int _inputChannels;

    public int OutputSize => _projection.OutputSize;

    // ── Constructor ───────────────────────────────────────────────────────────

    public CnnEncoder(int width, int height, int channels, RLCnnEncoderDef def)
    {
        _inputWidth    = width;
        _inputHeight   = height;
        _inputChannels = channels;

        var filterCounts = def.FilterCounts;
        var kernelSizes  = def.KernelSizes;
        var strides      = def.Strides;

        if (filterCounts.Length == 0 || filterCounts.Length != kernelSizes.Length || filterCounts.Length != strides.Length)
            throw new ArgumentException("[CnnEncoder] FilterCounts, KernelSizes, and Strides must be non-empty and the same length.");

        var rng = new Random(42);
        _convLayers = new ConvLayer[filterCounts.Length];

        var prevChannels = channels;
        var prevH = height;
        var prevW = width;
        for (var i = 0; i < filterCounts.Length; i++)
        {
            var outC = filterCounts[i];
            var k    = kernelSizes[i];
            var s    = strides[i];
            _convLayers[i] = new ConvLayer(prevH, prevW, prevChannels, outC, k, s, rng);
            prevH = _convLayers[i].OutputHeight;
            prevW = _convLayers[i].OutputWidth;
            prevChannels = outC;
        }

        var flatSize = prevH * prevW * prevChannels;
        _projection = new LinearLayer(flatSize, def.OutputSize, rng);
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Single-sample forward pass. Caches intermediate activations for backprop.
    /// </summary>
    public float[] Forward(float[] input)
    {
        var x = input;
        foreach (var conv in _convLayers)
            x = conv.Forward(x);
        return _projection.Forward(x);
    }

    // ── Gradient accumulation (batch update) ─────────────────────────────────

    public CnnGradientBuffer CreateGradientBuffer()
    {
        var convBuffers = new ConvGradientBuffer[_convLayers.Length];
        for (var i = 0; i < _convLayers.Length; i++)
            convBuffers[i] = _convLayers[i].CreateGradientBuffer();
        return new CnnGradientBuffer(convBuffers, _projection.CreateGradientBuffer());
    }

    /// <summary>
    /// Accumulates gradients into <paramref name="buffer"/> given the embedding-space gradient
    /// <paramref name="outputGrad"/>. Returns the input-space gradient (pixel gradients, rarely used).
    /// </summary>
    public float[] AccumulateGradients(float[] outputGrad, CnnGradientBuffer buffer)
    {
        var projInputGrad = _projection.AccumulateGradients(outputGrad, buffer.ProjectionGrad);

        var grad = projInputGrad;
        for (var i = _convLayers.Length - 1; i >= 0; i--)
            grad = _convLayers[i].AccumulateGradients(grad, buffer.ConvGrads[i]);

        return grad;
    }

    /// <summary>Applies accumulated gradients with global grad scale (already incorporates clip norm).</summary>
    public void ApplyGradients(CnnGradientBuffer buffer, float learningRate, float gradScale)
    {
        foreach (var (layer, grad) in _convLayers.Zip(buffer.ConvGrads))
            layer.ApplyGradients(grad, learningRate, gradScale);
        _projection.ApplyGradients(buffer.ProjectionGrad, learningRate, gradScale);
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add(_convLayers.Length);
        foreach (var conv in _convLayers)
            conv.AppendSerialized(weights, shapes);
        _projection.AppendSerialized(weights, shapes);
    }

    public void LoadSerialized(IReadOnlyList<float> weights, ref int wi, IReadOnlyList<int> shapes, ref int si)
    {
        var layerCount = shapes[si++];
        for (var i = 0; i < layerCount; i++)
            _convLayers[i].LoadSerialized(weights, ref wi, shapes, ref si);
        _projection.LoadSerialized(weights, ref wi, shapes, ref si);
    }

    // ── Copy weights ──────────────────────────────────────────────────────────

    public void CopyWeightsTo(CnnEncoder other)
    {
        for (var i = 0; i < _convLayers.Length; i++)
            _convLayers[i].CopyTo(other._convLayers[i]);
        _projection.CopyTo(other._projection);
    }

    public void LoadWeightsFrom(CnnEncoder other)
    {
        for (var i = 0; i < _convLayers.Length; i++)
            _convLayers[i].CopyTo(_convLayers[i]); // same-to-self is a no-op; below we pull from other
        for (var i = 0; i < _convLayers.Length; i++)
            other._convLayers[i].CopyTo(_convLayers[i]);
        other._projection.CopyTo(_projection);
    }

    // ── GradNorm helper ───────────────────────────────────────────────────────

    public float GradNormSquared(CnnGradientBuffer buffer)
    {
        var sum = 0f;
        foreach (var (_, g) in _convLayers.Zip(buffer.ConvGrads))
        {
            foreach (var v in g.FilterGrads) sum += v * v;
            foreach (var v in g.BiasGrads)   sum += v * v;
        }
        foreach (var v in buffer.ProjectionGrad.WeightGrads) sum += v * v;
        foreach (var v in buffer.ProjectionGrad.BiasGrads)   sum += v * v;
        return sum;
    }

    // ── Inner types ──────────────────────────────────────────────────────────

    private sealed class ConvLayer
    {
        // Weight layout: [outC, kernelH, kernelW, inC]
        private readonly float[] _filters;
        private readonly float[] _biases;
        private readonly int _inH, _inW, _inC;
        private readonly int _outC, _kernelH, _kernelW, _stride;
        public  readonly int OutputHeight;
        public  readonly int OutputWidth;

        // Adam moments
        private readonly float[] _wm, _wv, _bm, _bv;
        private float _b1t = 1f, _b2t = 1f;
        private const float B1 = 0.9f, B2 = 0.999f, Eps = 1e-8f;

        // Forward cache
        private float[]? _lastInput;
        private float[]? _lastPreAct;

        public ConvLayer(int inH, int inW, int inC, int outC, int kernel, int stride, Random rng)
        {
            _inH     = inH;  _inW    = inW;  _inC   = inC;
            _outC    = outC; _kernelH = kernel; _kernelW = kernel; _stride = stride;
            OutputHeight = (inH - kernel) / stride + 1;
            OutputWidth  = (inW - kernel) / stride + 1;

            var filterSize = outC * kernel * kernel * inC;
            _filters = new float[filterSize];
            _biases  = new float[outC];
            _wm = new float[filterSize]; _wv = new float[filterSize];
            _bm = new float[outC];       _bv = new float[outC];

            // He initialisation
            var scale = MathF.Sqrt(2f / (kernel * kernel * inC));
            for (var i = 0; i < filterSize; i++)
                _filters[i] = (float)(rng.NextGaussian() * scale);
        }

        public float[] Forward(float[] input)
        {
            _lastInput = input;
            var outH = OutputHeight;
            var outW = OutputWidth;
            var preAct  = new float[outH * outW * _outC];
            var output  = new float[outH * outW * _outC];

            for (var oc = 0; oc < _outC; oc++)
            for (var oh = 0; oh < outH; oh++)
            for (var ow = 0; ow < outW; ow++)
            {
                var sum = _biases[oc];
                for (var ic = 0; ic < _inC; ic++)
                for (var kh = 0; kh < _kernelH; kh++)
                for (var kw = 0; kw < _kernelW; kw++)
                {
                    var ih = oh * _stride + kh;
                    var iw = ow * _stride + kw;
                    sum += input[ih * _inW * _inC + iw * _inC + ic]
                         * _filters[oc * _kernelH * _kernelW * _inC + kh * _kernelW * _inC + kw * _inC + ic];
                }
                var idx = oh * outW * _outC + ow * _outC + oc;
                preAct[idx] = sum;
                output[idx] = MathF.Max(0f, sum); // ReLU
            }

            _lastPreAct = preAct;
            return output;
        }

        public ConvGradientBuffer CreateGradientBuffer() =>
            new ConvGradientBuffer(_filters.Length, _outC);

        public float[] AccumulateGradients(float[] outputGrad, ConvGradientBuffer buffer)
        {
            var outH     = OutputHeight;
            var outW     = OutputWidth;
            var inputGrad = new float[_inH * _inW * _inC];

            for (var oc = 0; oc < _outC; oc++)
            for (var oh = 0; oh < outH; oh++)
            for (var ow = 0; ow < outW; ow++)
            {
                var outIdx = oh * outW * _outC + ow * _outC + oc;
                // ReLU backward: gate by pre-activation sign
                var reluGrad = _lastPreAct![outIdx] > 0f ? outputGrad[outIdx] : 0f;

                buffer.BiasGrads[oc] += reluGrad;

                for (var ic = 0; ic < _inC; ic++)
                for (var kh = 0; kh < _kernelH; kh++)
                for (var kw = 0; kw < _kernelW; kw++)
                {
                    var ih = oh * _stride + kh;
                    var iw = ow * _stride + kw;
                    var wIdx   = oc * _kernelH * _kernelW * _inC + kh * _kernelW * _inC + kw * _inC + ic;
                    var inIdx  = ih * _inW * _inC + iw * _inC + ic;
                    buffer.FilterGrads[wIdx] += reluGrad * _lastInput![inIdx];
                    inputGrad[inIdx]         += reluGrad * _filters[wIdx];
                }
            }

            return inputGrad;
        }

        public void ApplyGradients(ConvGradientBuffer buffer, float lr, float scale)
        {
            _b1t *= B1; _b2t *= B2;
            var lrCorrected = lr * MathF.Sqrt(1f - _b2t) / (1f - _b1t);

            for (var i = 0; i < _filters.Length; i++)
            {
                var g = buffer.FilterGrads[i] * scale;
                _wm[i] = B1 * _wm[i] + (1f - B1) * g;
                _wv[i] = B2 * _wv[i] + (1f - B2) * g * g;
                _filters[i] -= lrCorrected * _wm[i] / (MathF.Sqrt(_wv[i]) + Eps);
            }

            for (var i = 0; i < _outC; i++)
            {
                var g = buffer.BiasGrads[i] * scale;
                _bm[i] = B1 * _bm[i] + (1f - B1) * g;
                _bv[i] = B2 * _bv[i] + (1f - B2) * g * g;
                _biases[i] -= lrCorrected * _bm[i] / (MathF.Sqrt(_bv[i]) + Eps);
            }
        }

        public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
        {
            shapes.Add(_outC); shapes.Add(_kernelH); shapes.Add(_kernelW); shapes.Add(_inC); shapes.Add(_stride);
            foreach (var w in _filters) weights.Add(w);
            foreach (var b in _biases)  weights.Add(b);
        }

        public void LoadSerialized(IReadOnlyList<float> weights, ref int wi, IReadOnlyList<int> shapes, ref int si)
        {
            si += 5; // skip shape descriptor (already encoded in the constructor)
            for (var i = 0; i < _filters.Length; i++) _filters[i] = weights[wi++];
            for (var i = 0; i < _biases.Length;  i++) _biases[i]  = weights[wi++];
        }

        public void CopyTo(ConvLayer other)
        {
            Array.Copy(_filters, other._filters, _filters.Length);
            Array.Copy(_biases,  other._biases,  _biases.Length);
        }
    }

    private sealed class LinearLayer
    {
        private readonly float[] _weights;
        private readonly float[] _biases;
        private readonly int _inSize;
        public  readonly int OutputSize;

        private readonly float[] _wm, _wv, _bm, _bv;
        private float _b1t = 1f, _b2t = 1f;
        private const float B1 = 0.9f, B2 = 0.999f, Eps = 1e-8f;

        private float[]? _lastInput;

        public LinearLayer(int inputSize, int outputSize, Random rng)
        {
            _inSize    = inputSize;
            OutputSize = outputSize;
            _weights = new float[inputSize * outputSize];
            _biases  = new float[outputSize];
            _wm = new float[_weights.Length]; _wv = new float[_weights.Length];
            _bm = new float[outputSize];      _bv = new float[outputSize];

            var scale = MathF.Sqrt(2f / inputSize);
            for (var i = 0; i < _weights.Length; i++)
                _weights[i] = (float)(rng.NextGaussian() * scale);
        }

        public float[] Forward(float[] input)
        {
            _lastInput = input;
            var output = new float[OutputSize];
            for (var o = 0; o < OutputSize; o++)
            {
                var sum = _biases[o];
                for (var i = 0; i < _inSize; i++)
                    sum += input[i] * _weights[o * _inSize + i];
                output[o] = MathF.Max(0f, sum); // ReLU on projection
            }
            return output;
        }

        public LinearGradientBuffer CreateGradientBuffer() => new LinearGradientBuffer(_weights.Length, OutputSize);

        public float[] AccumulateGradients(float[] outputGrad, LinearGradientBuffer buffer)
        {
            var inputGrad = new float[_inSize];
            for (var o = 0; o < OutputSize; o++)
            {
                var g = outputGrad[o]; // no activation gating needed since network applies no further activation
                buffer.BiasGrads[o] += g;
                for (var i = 0; i < _inSize; i++)
                {
                    buffer.WeightGrads[o * _inSize + i] += g * _lastInput![i];
                    inputGrad[i]                        += g * _weights[o * _inSize + i];
                }
            }
            return inputGrad;
        }

        public void ApplyGradients(LinearGradientBuffer buffer, float lr, float scale)
        {
            _b1t *= B1; _b2t *= B2;
            var lrCorrected = lr * MathF.Sqrt(1f - _b2t) / (1f - _b1t);

            for (var i = 0; i < _weights.Length; i++)
            {
                var g = buffer.WeightGrads[i] * scale;
                _wm[i] = B1 * _wm[i] + (1f - B1) * g;
                _wv[i] = B2 * _wv[i] + (1f - B2) * g * g;
                _weights[i] -= lrCorrected * _wm[i] / (MathF.Sqrt(_wv[i]) + Eps);
            }

            for (var i = 0; i < OutputSize; i++)
            {
                var g = buffer.BiasGrads[i] * scale;
                _bm[i] = B1 * _bm[i] + (1f - B1) * g;
                _bv[i] = B2 * _bv[i] + (1f - B2) * g * g;
                _biases[i] -= lrCorrected * _bm[i] / (MathF.Sqrt(_bv[i]) + Eps);
            }
        }

        public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
        {
            shapes.Add(_inSize); shapes.Add(OutputSize);
            foreach (var w in _weights) weights.Add(w);
            foreach (var b in _biases)  weights.Add(b);
        }

        public void LoadSerialized(IReadOnlyList<float> weights, ref int wi, IReadOnlyList<int> shapes, ref int si)
        {
            si += 2; // skip descriptor
            for (var i = 0; i < _weights.Length; i++) _weights[i] = weights[wi++];
            for (var i = 0; i < _biases.Length;  i++) _biases[i]  = weights[wi++];
        }

        public void CopyTo(LinearLayer other)
        {
            Array.Copy(_weights, other._weights, _weights.Length);
            Array.Copy(_biases,  other._biases,  _biases.Length);
        }
    }
}

// ── Gradient buffer types ─────────────────────────────────────────────────────

internal sealed class ConvGradientBuffer
{
    public float[] FilterGrads { get; }
    public float[] BiasGrads   { get; }

    public ConvGradientBuffer(int filterCount, int biasCount)
    {
        FilterGrads = new float[filterCount];
        BiasGrads   = new float[biasCount];
    }
}

internal sealed class LinearGradientBuffer
{
    public float[] WeightGrads { get; }
    public float[] BiasGrads   { get; }

    public LinearGradientBuffer(int weightCount, int biasCount)
    {
        WeightGrads = new float[weightCount];
        BiasGrads   = new float[biasCount];
    }
}

internal sealed class CnnGradientBuffer
{
    public ConvGradientBuffer[]  ConvGrads      { get; }
    public LinearGradientBuffer  ProjectionGrad { get; }

    public CnnGradientBuffer(ConvGradientBuffer[] convGrads, LinearGradientBuffer projGrad)
    {
        ConvGrads      = convGrads;
        ProjectionGrad = projGrad;
    }
}

// ── Extension helper ──────────────────────────────────────────────────────────

internal static class RandomExtensions
{
    public static double NextGaussian(this Random rng)
    {
        // Box-Muller
        var u1 = Math.Max(rng.NextDouble(), 1e-10);
        var u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
