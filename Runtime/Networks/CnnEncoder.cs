using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// CPU CNN encoder — wraps the native <c>RlCnnEncoder</c> GDExtension object.
/// Implements <see cref="IEncoder"/> so it is interchangeable with the future
/// <c>GpuCnnEncoder</c> (Vulkan compute) without changing any training code.
/// </summary>
internal sealed class CnnEncoder : IEncoder
{
    private readonly GodotObject _native;
    public  int  OutputSize             { get; }
    public  bool SupportsBatchedTraining => false;

    public CnnEncoder(int width, int height, int channels, RLCnnEncoderDef def)
    {
        _native = ClassDB.Instantiate("RlCnnEncoder").AsGodotObject()
                  ?? throw new InvalidOperationException(
                      "[CnnEncoder] Failed to instantiate RlCnnEncoder. " +
                      "Make sure the native library is built and " +
                      "rl_cnn.gdextension is in the project.");

        _native.Call(RlCnnEncoderNativeFunctions.Initialize,
            width, height, channels,
            new Godot.Collections.Array<int>(def.FilterCounts),
            new Godot.Collections.Array<int>(def.KernelSizes),
            new Godot.Collections.Array<int>(def.Strides),
            def.OutputSize);

        OutputSize = def.OutputSize;
    }

    // ── IEncoder: single-sample inference ────────────────────────────────────

    public float[] Forward(float[] input)
        => (float[])_native.Call(RlCnnEncoderNativeFunctions.Forward, (Variant)input);

    public void ForwardBatch(float[] inputBatch, int batchSize, float[] outputBatch)
    {
        var inputSize = inputBatch.Length / Math.Max(1, batchSize);
        if (inputBatch.Length != batchSize * inputSize)
            throw new ArgumentException("[CnnEncoder] ForwardBatch input size is not divisible by batchSize.", nameof(inputBatch));
        if (outputBatch.Length != batchSize * OutputSize)
            throw new ArgumentException("[CnnEncoder] ForwardBatch output buffer has the wrong size.", nameof(outputBatch));

        var input = new float[inputSize];
        for (var b = 0; b < batchSize; b++)
        {
            Array.Copy(inputBatch, b * inputSize, input, 0, inputSize);
            var output = Forward(input);
            Array.Copy(output, 0, outputBatch, b * OutputSize, OutputSize);
        }
    }

    // ── IEncoder: per-sample gradient accumulation ───────────────────────────

    public ICnnGradientToken CreateGradientToken()
        => new CnnGradientToken(_native.Call(RlCnnEncoderNativeFunctions.CreateGradientBuffer));

    public float[] AccumulateGradients(float[] outputGrad, ICnnGradientToken token)
        => (float[])_native.Call(RlCnnEncoderNativeFunctions.AccumulateGradients,
               (Variant)outputGrad,
               ((CnnGradientToken)token).Data);

    public void AccumulateGradientsBatch(float[] outputGradBatch, int batchSize, ICnnGradientToken token)
    {
        // CnnEncoder.SupportsBatchedTraining is false, so PolicyValueNetwork never calls this
        // path — it uses the per-sample AccumulateGradients route instead.  A naive loop here
        // would be incorrect because each AccumulateGradients call relies on activations cached
        // by the immediately preceding Forward, but we don't have the input batch to re-run it.
        throw new NotSupportedException(
            "[CnnEncoder] AccumulateGradientsBatch is not supported on the CPU encoder. " +
            "Check SupportsBatchedTraining before calling the batched backward path.");
    }

    public void ApplyGradients(ICnnGradientToken token, float learningRate, float gradScale)
        => _native.Call(RlCnnEncoderNativeFunctions.ApplyGradients,
               ((CnnGradientToken)token).Data, learningRate, gradScale);

    public float GradNormSquared(ICnnGradientToken token)
        => (float)_native.Call(RlCnnEncoderNativeFunctions.GradNormSquared,
               ((CnnGradientToken)token).Data);

    internal float[] DebugReadGradientBuffer(ICnnGradientToken token)
        => (float[])((CnnGradientToken)token).Data;

    // ── IEncoder: serialization ───────────────────────────────────────────────

    public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        var w = (float[])_native.Call(RlCnnEncoderNativeFunctions.GetWeights);
        var s = (int[])  _native.Call(RlCnnEncoderNativeFunctions.GetShapes);
        foreach (var v in w) weights.Add(v);
        foreach (var v in s) shapes.Add(v);
    }

    public void LoadSerialized(IReadOnlyList<float> weights, ref int wi,
                               IReadOnlyList<int>   shapes,  ref int si)
    {
        // shapes descriptor layout (matches get_shapes / AppendSerialized):
        //   [num_conv_layers, outC, kH, kW, inC, stride, ... (×num_conv), inSize, outSize]
        int nConv   = shapes[si];
        int nShapes = 1 + nConv * 5 + 2;

        int nWeights = 0;
        for (var c = 0; c < nConv; c++)
        {
            int outC = shapes[si + 1 + c * 5 + 0];
            int kH   = shapes[si + 1 + c * 5 + 1];
            int kW   = shapes[si + 1 + c * 5 + 2];
            int inC  = shapes[si + 1 + c * 5 + 3];
            nWeights += outC * kH * kW * inC + outC;
        }
        int projIn  = shapes[si + 1 + nConv * 5];
        int projOut = shapes[si + 1 + nConv * 5 + 1];
        nWeights += projIn * projOut + projOut;

        var wSlice = new float[nWeights];
        for (var i = 0; i < nWeights; i++) wSlice[i] = weights[wi++];

        var sSlice = new int[nShapes];
        for (var i = 0; i < nShapes; i++) sSlice[i] = shapes[si++];

        _native.Call(RlCnnEncoderNativeFunctions.SetWeights, (Variant)wSlice, (Variant)sSlice);
    }

    // ── IEncoder: weight copy ─────────────────────────────────────────────────

    public void CopyWeightsTo(IEncoder other)
    {
        if (ReferenceEquals(this, other))
            return;

        if (other is CnnEncoder cpu)
        {
            // Fast path: both native — direct native call, no intermediate allocation.
            var weights = _native.Call(RlCnnEncoderNativeFunctions.GetWeights);
            var shapes  = _native.Call(RlCnnEncoderNativeFunctions.GetShapes);
            cpu._native.Call(RlCnnEncoderNativeFunctions.SetWeights, weights, shapes);
            return;
        }

        // Cross-device path (e.g. CPU → GPU): round-trip through the serialization format.
        var w = new List<float>();
        var s = new List<int>();
        AppendSerialized(w, s);
        var wi = 0;
        var si = 0;
        other.LoadSerialized(w, ref wi, s, ref si);
    }
}

// ── Gradient token ────────────────────────────────────────────────────────────

/// <summary>
/// CPU implementation of <see cref="ICnnGradientToken"/>.
/// Wraps the flat <c>float[]</c> gradient buffer managed by the native CNN encoder.
/// </summary>
internal sealed class CnnGradientToken : ICnnGradientToken
{
    internal Variant Data { get; }
    internal CnnGradientToken(Variant data) => Data = data;
}
