using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// C# wrapper around the native <c>RlLayerNormLayer</c> GDExtension object.
/// Implements <see cref="NetworkLayer"/> so it drops in wherever a
/// <see cref="LayerNormLayer"/> is used.
/// Uses plain SGD for gamma/beta (no Adam), matching the C# implementation.
/// </summary>
internal sealed class NativeLayerNormLayer : NetworkLayer, INativeLayer
{
    private readonly GodotObject _native;
    private readonly int         _size;

    public override int InputSize  => _size;
    public override int OutputSize => _size;

    public NativeLayerNormLayer(int size)
    {
        _size = size;

        _native = ClassDB.Instantiate("RlLayerNormLayer").AsGodotObject()
                  ?? throw new InvalidOperationException(
                      "[NativeLayerNormLayer] Failed to instantiate RlLayerNormLayer. " +
                      "Make sure the native library is built and " +
                      "rl_cnn.gdextension is loaded in the project.");

        _native.Call(RlLayerNormLayerNativeFunctions.Initialize, size);
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    public override float[] Forward(float[] input, bool isTraining = false)
        => (float[])_native.Call(RlLayerNormLayerNativeFunctions.Forward, (Variant)input);

    public override VectorBatch ForwardBatch(VectorBatch input)
    {
        var flat = (float[])_native.Call(RlLayerNormLayerNativeFunctions.ForwardBatch,
                       (Variant)input.Data, input.BatchSize);
        var batch = new VectorBatch(input.BatchSize, _size);
        Array.Copy(flat, batch.Data, flat.Length);
        return batch;
    }

    // ── Single-sample backward (immediate SGD update) ─────────────────────────

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
        => (float[])_native.Call(RlLayerNormLayerNativeFunctions.Backward,
               (Variant)outputGrad, learningRate, gradScale);

    // ── Batch accumulate + apply ──────────────────────────────────────────────

    public override GradientBuffer CreateGradientBuffer()
    {
        var buf = new NativeGradientBuffer(this);
        buf.NativeData = _native.Call(RlLayerNormLayerNativeFunctions.CreateGradientBuffer);
        return buf;
    }

    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
    {
        var nb = (NativeGradientBuffer)buffer;
        var result = (Godot.Collections.Array)_native.Call(
            RlLayerNormLayerNativeFunctions.AccumulateGradientsWithBuffer,
            (Variant)outputGrad,
            nb.NativeData);
        var inputGrad = (float[])result[0];
        var updatedBuffer = (float[])result[1];
        nb.NativeData = (Variant)updatedBuffer;
        return inputGrad;
    }

    public override void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale)
    {
        var nb = (NativeGradientBuffer)buffer;
        _native.Call(RlLayerNormLayerNativeFunctions.ApplyGradients,
            nb.NativeData, learningRate, gradScale);
    }

    // ── Frozen gradient ───────────────────────────────────────────────────────

    public override float[] ComputeInputGrad(float[] outputGrad)
        => (float[])_native.Call(RlLayerNormLayerNativeFunctions.ComputeInputGrad,
               (Variant)outputGrad);

    // ── INativeLayer ──────────────────────────────────────────────────────────

    public float GradNormSquared(Variant nativeBuffer)
        => (float)_native.Call(RlLayerNormLayerNativeFunctions.GradNormSquared, nativeBuffer);

    // ── Target-network copy ───────────────────────────────────────────────────

    public override void CopyFrom(NetworkLayer source)
    {
        switch (source)
        {
            case NativeLayerNormLayer nln:
                _native.Call(RlLayerNormLayerNativeFunctions.CopyWeightsFrom, nln._native);
                return;

            case LayerNormLayer ln:
                var w = new List<float>(); var s = new List<int>();
                ln.AppendSerialized(w, s);
                int wi = 0, si = 0;
                LoadSerialized(w, ref wi, s, ref si);
                return;

            default:
                throw new InvalidOperationException(
                    $"[NativeLayerNormLayer] CopyFrom: unsupported source type {source.GetType().Name}.");
        }
    }

    public override void SoftUpdateFrom(NetworkLayer source, float tau)
    {
        switch (source)
        {
            case NativeLayerNormLayer nln:
                _native.Call(RlLayerNormLayerNativeFunctions.SoftUpdateFrom, nln._native, tau);
                return;

            case LayerNormLayer ln:
                var temp = new NativeLayerNormLayer(_size);
                var w = new List<float>(); var s = new List<int>();
                ln.AppendSerialized(w, s);
                int wi = 0, si = 0;
                temp.LoadSerialized(w, ref wi, s, ref si);
                _native.Call(RlLayerNormLayerNativeFunctions.SoftUpdateFrom, temp._native, tau);
                return;

            default:
                throw new InvalidOperationException(
                    $"[NativeLayerNormLayer] SoftUpdateFrom: unsupported source type {source.GetType().Name}.");
        }
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        var w = (float[])_native.Call(RlLayerNormLayerNativeFunctions.GetWeights);
        var s = (int[])  _native.Call(RlLayerNormLayerNativeFunctions.GetShapes);
        foreach (var v in w) weights.Add(v);
        foreach (var v in s) shapes.Add(v);
    }

    public override void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false)
    {
        if (!isLegacy)
        {
            var typeCode = shapes[si++];
            if (typeCode != (int)RLLayerKind.LayerNorm)
                throw new InvalidOperationException(
                    $"[NativeLayerNormLayer] Expected LayerNorm type ({(int)RLLayerKind.LayerNorm}), got {typeCode}.");
        }

        var serializedSize = shapes[si++];
        if (serializedSize != _size)
            throw new InvalidOperationException(
                $"[NativeLayerNormLayer] Checkpoint size {serializedSize} does not match network size {_size}.");

        int wCount = 2 * _size;  // gamma + beta
        var wSlice = new float[wCount];
        for (var i = 0; i < wCount; i++) wSlice[i] = weights[wi++];

        _native.Call(RlLayerNormLayerNativeFunctions.SetWeights,
            (Variant)wSlice,
            (Variant)new int[] { 2, _size });
    }
}
