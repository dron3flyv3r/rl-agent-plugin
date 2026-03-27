using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Wraps the native <c>RlCnnEncoder</c> GDExtension object.
/// The public API is identical to the old pure-C# implementation so nothing
/// else in the codebase needs to change.
/// </summary>
internal sealed class CnnEncoder
{
    private readonly GodotObject _native;
    public  int OutputSize { get; }

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

    // ── Forward ───────────────────────────────────────────────────────────────

    public float[] Forward(float[] input)
        => (float[])_native.Call(RlCnnEncoderNativeFunctions.Forward, (Variant)input);

    // ── Gradient accumulation ─────────────────────────────────────────────────

    public CnnGradientBuffer CreateGradientBuffer()
        => new CnnGradientBuffer((float[])_native.Call(RlCnnEncoderNativeFunctions.CreateGradientBuffer));

    public float[] AccumulateGradients(float[] outputGrad, CnnGradientBuffer buffer)
        => (float[])_native.Call(RlCnnEncoderNativeFunctions.AccumulateGradients, (Variant)outputGrad, (Variant)buffer.Data);

    public void ApplyGradients(CnnGradientBuffer buffer, float learningRate, float gradScale)
        => _native.Call(RlCnnEncoderNativeFunctions.ApplyGradients, (Variant)buffer.Data, learningRate, gradScale);

    // ── Gradient norm (for gradient clipping) ────────────────────────────────

    public float GradNormSquared(CnnGradientBuffer buffer)
        => (float)_native.Call(RlCnnEncoderNativeFunctions.GradNormSquared, (Variant)buffer.Data);

    // ── Serialization ─────────────────────────────────────────────────────────

    public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        var w = (float[])_native.Call(RlCnnEncoderNativeFunctions.GetWeights);
        var s = (int[])  _native.Call(RlCnnEncoderNativeFunctions.GetShapes);
        foreach (var v in w) weights.Add(v);
        foreach (var v in s) shapes.Add(v);
    }

    public void LoadSerialized(IReadOnlyList<float> weights, ref int wi,
                               IReadOnlyList<int> shapes, ref int si)
    {
        // shapes descriptor layout (matches get_shapes / AppendSerialized):
        //   [num_conv_layers, outC, kH, kW, inC, stride, ... (×num_conv), inSize, outSize]
        int nConv   = shapes[si];
        int nShapes = 1 + nConv * 5 + 2; // total shape ints for this encoder

        // Count weights from shape descriptor without calling get_weights.
        int nWeights = 0;
        for (var c = 0; c < nConv; c++)
        {
            int outC = shapes[si + 1 + c * 5 + 0];
            int kH   = shapes[si + 1 + c * 5 + 1];
            int kW   = shapes[si + 1 + c * 5 + 2];
            int inC  = shapes[si + 1 + c * 5 + 3];
            // stride is [4], not needed for weight count
            nWeights += outC * kH * kW * inC + outC; // filters + biases
        }
        int projIn  = shapes[si + 1 + nConv * 5];
        int projOut = shapes[si + 1 + nConv * 5 + 1];
        nWeights += projIn * projOut + projOut; // weights + biases

        var wSlice = new float[nWeights];
        for (var i = 0; i < nWeights; i++) wSlice[i] = weights[wi++];

        var sSlice = new int[nShapes];
        for (var i = 0; i < nShapes; i++) sSlice[i] = shapes[si++];

        _native.Call("set_weights", (Variant)wSlice, (Variant)sSlice);
    }

    // ── Weight copy (SAC target network sync) ─────────────────────────────────

    public void CopyWeightsTo(CnnEncoder other)
    {
        var weights = _native.Call(RlCnnEncoderNativeFunctions.GetWeights);
        var shapes  = _native.Call(RlCnnEncoderNativeFunctions.GetShapes);
        other._native.Call(RlCnnEncoderNativeFunctions.SetWeights, weights, shapes);
    }

    public void LoadWeightsFrom(CnnEncoder other) => other.CopyWeightsTo(this);
}

// ── Gradient buffer ───────────────────────────────────────────────────────────

/// <summary>
/// Wraps the flat float[] gradient buffer managed by the native CNN encoder.
/// </summary>
internal sealed class CnnGradientBuffer
{
    internal float[] Data { get; }
    internal CnnGradientBuffer(float[] data) => Data = data;
}
