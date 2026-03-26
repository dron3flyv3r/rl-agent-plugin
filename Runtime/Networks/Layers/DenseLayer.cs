using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Gradient accumulation buffer — weight gradients + bias gradients for one layer.
/// Used in batch update mode: accumulate across samples, then call ApplyGradients once.
/// </summary>
internal sealed class GradientBuffer
{
    public GradientBuffer(int weightCount, int biasCount)
    {
        WeightGradients = new float[weightCount];
        BiasGradients   = new float[biasCount];
    }

    public float[] WeightGradients { get; }
    public float[] BiasGradients   { get; }

    public float SumSquares()
    {
        var sum = 0f;
        foreach (var g in WeightGradients) sum += g * g;
        foreach (var g in BiasGradients)   sum += g * g;
        return sum;
    }
}

/// <summary>
/// Fully-connected (dense / linear) layer with optional activation.
/// Supports Adam, SGD, and None (frozen — no weight updates, used for SAC target networks).
/// </summary>
internal sealed class DenseLayer : NetworkLayer
{
    private const float AdamBeta1   = 0.9f;
    private const float AdamBeta2   = 0.999f;
    private const float AdamEpsilon = 1e-8f;

    private readonly int              _inputSize;
    private readonly int              _outputSize;
    private readonly RLActivationKind? _activation;
    private readonly RLOptimizerKind  _optimizer;
    private readonly float[]          _weights;
    private readonly float[]          _biases;

    // Adam moment vectors (null for SGD / None)
    private readonly float[]? _wm;
    private readonly float[]? _wv;
    private readonly float[]? _bm;
    private readonly float[]? _bv;

    // Iterative bias-correction accumulators: beta^t
    private float _adamB1Pow = 1f;
    private float _adamB2Pow = 1f;

    // Forward-pass cache (set by Forward; used by Backward / AccumulateGradients / ComputeInputGrad)
    private float[]? _lastInput;
    private float[]? _lastPreActivation;

    private readonly RandomNumberGenerator _rng = new();

    public override int InputSize  => _inputSize;
    public override int OutputSize => _outputSize;

    public DenseLayer(int inputSize, int outputSize, RLActivationKind? activation, RLOptimizerKind optimizer)
    {
        _inputSize  = inputSize;
        _outputSize = outputSize;
        _activation = activation;
        _optimizer  = optimizer;
        _weights    = new float[inputSize * outputSize];
        _biases     = new float[outputSize];

        _rng.Randomize();
        var scale = Mathf.Sqrt(2.0f / Mathf.Max(1, inputSize));
        for (var i = 0; i < _weights.Length; i++)
            _weights[i] = _rng.Randfn(0.0f, scale);

        if (optimizer == RLOptimizerKind.Adam)
        {
            _wm = new float[_weights.Length];
            _wv = new float[_weights.Length];
            _bm = new float[outputSize];
            _bv = new float[outputSize];
        }
    }

    // ── Forward ─────────────────────────────────────────────────────────────

    public override float[] Forward(float[] input, bool isTraining = false)
    {
        _lastInput = input;

        var preActivation = new float[_outputSize];
        for (var oi = 0; oi < _outputSize; oi++)
        {
            var sum = _biases[oi];
            var wBase = oi * _inputSize;
            for (var ii = 0; ii < _inputSize; ii++)
                sum += input[ii] * _weights[wBase + ii];
            preActivation[oi] = sum;
        }

        _lastPreActivation = preActivation;

        var activated = preActivation.ToArray();
        if (_activation.HasValue)
            for (var i = 0; i < activated.Length; i++)
                activated[i] = Activate(activated[i], _activation.Value);

        return activated;
    }

    public override VectorBatch ForwardBatch(VectorBatch input)
    {
        if (input.VectorSize != _inputSize)
            throw new ArgumentException($"Expected vector size {_inputSize}, got {input.VectorSize}.");

        var output = new VectorBatch(input.BatchSize, _outputSize);
        for (var b = 0; b < input.BatchSize; b++)
        {
            var inOff  = b * _inputSize;
            var outOff = b * _outputSize;
            for (var oi = 0; oi < _outputSize; oi++)
            {
                var sum   = _biases[oi];
                var wBase = oi * _inputSize;
                for (var ii = 0; ii < _inputSize; ii++)
                    sum += input.Data[inOff + ii] * _weights[wBase + ii];

                output.Data[outOff + oi] = _activation.HasValue ? Activate(sum, _activation.Value) : sum;
            }
        }

        return output;
    }

    // ── Backward (single-sample, immediate weight update) ────────────────────

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
    {
        var input        = _lastInput        ?? throw new InvalidOperationException("Backward called before Forward.");
        var preActivation = _lastPreActivation;

        var localGrad = ComputeLocalGradient(outputGrad, preActivation);
        var inputGrad = ComputeInputGradient(localGrad);
        ApplyWeightUpdate(input, localGrad, learningRate, gradScale);
        return inputGrad;
    }

    // ── Batch accumulate + apply ─────────────────────────────────────────────

    public override GradientBuffer CreateGradientBuffer() =>
        new(_weights.Length, _biases.Length);

    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
    {
        var input         = _lastInput        ?? throw new InvalidOperationException("AccumulateGradients called before Forward.");
        var preActivation = _lastPreActivation;

        var localGrad = ComputeLocalGradient(outputGrad, preActivation);
        var inputGrad = new float[_inputSize];

        for (var oi = 0; oi < _outputSize; oi++)
        {
            for (var ii = 0; ii < _inputSize; ii++)
            {
                var wi = oi * _inputSize + ii;
                inputGrad[ii]                  += _weights[wi] * localGrad[oi];
                buffer.WeightGradients[wi]     += localGrad[oi] * input[ii];
            }

            buffer.BiasGradients[oi] += localGrad[oi];
        }

        return inputGrad;
    }

    public override void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale)
    {
        if (_optimizer == RLOptimizerKind.None) return;

        if (_optimizer == RLOptimizerKind.Adam)
        {
            _adamB1Pow *= AdamBeta1;
            _adamB2Pow *= AdamBeta2;
            var b1Corr = 1f - _adamB1Pow;
            var b2Corr = 1f - _adamB2Pow;

            for (var oi = 0; oi < _outputSize; oi++)
            {
                for (var ii = 0; ii < _inputSize; ii++)
                {
                    var wi = oi * _inputSize + ii;
                    var g  = buffer.WeightGradients[wi] * gradScale;
                    _wm![wi] = AdamBeta1 * _wm[wi] + (1f - AdamBeta1) * g;
                    _wv![wi] = AdamBeta2 * _wv[wi] + (1f - AdamBeta2) * g * g;
                    _weights[wi] -= learningRate * (_wm[wi] / b1Corr) / (Mathf.Sqrt(_wv![wi] / b2Corr) + AdamEpsilon);
                }

                var bg = buffer.BiasGradients[oi] * gradScale;
                _bm![oi] = AdamBeta1 * _bm[oi] + (1f - AdamBeta1) * bg;
                _bv![oi] = AdamBeta2 * _bv[oi] + (1f - AdamBeta2) * bg * bg;
                _biases[oi] -= learningRate * (_bm[oi] / b1Corr) / (Mathf.Sqrt(_bv![oi] / b2Corr) + AdamEpsilon);
            }
        }
        else // SGD
        {
            for (var oi = 0; oi < _outputSize; oi++)
            {
                for (var ii = 0; ii < _inputSize; ii++)
                {
                    var wi = oi * _inputSize + ii;
                    _weights[wi] -= learningRate * buffer.WeightGradients[wi] * gradScale;
                }

                _biases[oi] -= learningRate * buffer.BiasGradients[oi] * gradScale;
            }
        }
    }

    // ── Frozen gradient (SAC dQ/da) ──────────────────────────────────────────

    public override float[] ComputeInputGrad(float[] outputGrad)
    {
        var preActivation = _lastPreActivation;
        var localGrad     = ComputeLocalGradient(outputGrad, preActivation);
        return ComputeInputGradient(localGrad);
    }

    // ── Target-network copy ──────────────────────────────────────────────────

    public override void CopyFrom(NetworkLayer source)
    {
        if (source is not DenseLayer dl)
            throw new InvalidOperationException("CopyFrom: source must be a DenseLayer.");
        Array.Copy(dl._weights, _weights, _weights.Length);
        Array.Copy(dl._biases,  _biases,  _biases.Length);
    }

    public override void SoftUpdateFrom(NetworkLayer source, float tau)
    {
        if (source is not DenseLayer dl)
            throw new InvalidOperationException("SoftUpdateFrom: source must be a DenseLayer.");
        for (var i = 0; i < _weights.Length; i++)
            _weights[i] = tau * dl._weights[i] + (1f - tau) * _weights[i];
        for (var i = 0; i < _biases.Length; i++)
            _biases[i] = tau * dl._biases[i] + (1f - tau) * _biases[i];
    }

    // ── Serialization ────────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add((int)RLLayerKind.Dense);
        shapes.Add(_inputSize);
        shapes.Add(_outputSize);
        shapes.Add(_activation.HasValue ? (int)_activation.Value + 1 : 0);
        foreach (var w in _weights) weights.Add(w);
        foreach (var b in _biases)  weights.Add(b);
    }

    public override void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false)
    {
        if (!isLegacy)
        {
            var typeCode = shapes[si++];
            if (typeCode != (int)RLLayerKind.Dense)
                throw new InvalidOperationException($"Expected Dense layer type ({(int)RLLayerKind.Dense}), got {typeCode}.");
        }

        var serializedIn   = shapes[si++];
        var serializedOut  = shapes[si++];
        var serializedAct  = shapes[si++];

        if (serializedIn != _inputSize || serializedOut != _outputSize)
            throw new InvalidOperationException("Checkpoint layer shape does not match the active network.");

        var expectedAct = _activation.HasValue ? (int)_activation.Value + 1 : 0;
        if (serializedAct != expectedAct)
            throw new InvalidOperationException("Checkpoint activation does not match the active network.");

        for (var i = 0; i < _weights.Length; i++) _weights[i] = weights[wi++];
        for (var i = 0; i < _biases.Length;  i++) _biases[i]  = weights[wi++];

        // Reset Adam moments so they warm up from the new weights
        if (_optimizer == RLOptimizerKind.Adam)
        {
            Array.Clear(_wm!, 0, _wm!.Length);
            Array.Clear(_wv!, 0, _wv!.Length);
            Array.Clear(_bm!, 0, _bm!.Length);
            Array.Clear(_bv!, 0, _bv!.Length);
            _adamB1Pow = 1f;
            _adamB2Pow = 1f;
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private float[] ComputeLocalGradient(float[] outputGrad, float[]? preActivation)
    {
        if (!_activation.HasValue || preActivation is null)
            return outputGrad.ToArray();

        var local = outputGrad.ToArray();
        for (var i = 0; i < local.Length; i++)
            local[i] *= ActivateDerivative(preActivation[i], _activation.Value);
        return local;
    }

    private float[] ComputeInputGradient(float[] localGrad)
    {
        var inputGrad = new float[_inputSize];
        for (var oi = 0; oi < _outputSize; oi++)
            for (var ii = 0; ii < _inputSize; ii++)
                inputGrad[ii] += _weights[oi * _inputSize + ii] * localGrad[oi];
        return inputGrad;
    }

    private void ApplyWeightUpdate(float[] input, float[] localGrad, float learningRate, float gradScale)
    {
        if (_optimizer == RLOptimizerKind.None) return;

        if (_optimizer == RLOptimizerKind.Adam)
        {
            _adamB1Pow *= AdamBeta1;
            _adamB2Pow *= AdamBeta2;
            var b1Corr = 1f - _adamB1Pow;
            var b2Corr = 1f - _adamB2Pow;

            for (var oi = 0; oi < _outputSize; oi++)
            {
                for (var ii = 0; ii < _inputSize; ii++)
                {
                    var wi = oi * _inputSize + ii;
                    var g  = localGrad[oi] * input[ii] * gradScale;
                    _wm![wi] = AdamBeta1 * _wm[wi] + (1f - AdamBeta1) * g;
                    _wv![wi] = AdamBeta2 * _wv[wi] + (1f - AdamBeta2) * g * g;
                    _weights[wi] -= learningRate * (_wm[wi] / b1Corr) / (Mathf.Sqrt(_wv![wi] / b2Corr) + AdamEpsilon);
                }

                var bg = localGrad[oi] * gradScale;
                _bm![oi] = AdamBeta1 * _bm[oi] + (1f - AdamBeta1) * bg;
                _bv![oi] = AdamBeta2 * _bv[oi] + (1f - AdamBeta2) * bg * bg;
                _biases[oi] -= learningRate * (_bm[oi] / b1Corr) / (Mathf.Sqrt(_bv![oi] / b2Corr) + AdamEpsilon);
            }
        }
        else // SGD
        {
            for (var oi = 0; oi < _outputSize; oi++)
            {
                for (var ii = 0; ii < _inputSize; ii++)
                    _weights[oi * _inputSize + ii] -= learningRate * localGrad[oi] * input[ii] * gradScale;
                _biases[oi] -= learningRate * localGrad[oi] * gradScale;
            }
        }
    }

    private static float Activate(float value, RLActivationKind activation) =>
        activation == RLActivationKind.Relu ? Mathf.Max(0.0f, value) : Mathf.Tanh(value);

    private static float ActivateDerivative(float preActivation, RLActivationKind activation)
    {
        if (activation == RLActivationKind.Relu)
            return preActivation > 0.0f ? 1.0f : 0.0f;
        var t = Mathf.Tanh(preActivation);
        return 1.0f - t * t;
    }
}
