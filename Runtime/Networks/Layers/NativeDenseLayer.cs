using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// C# wrapper around the native <c>RlDenseLayer</c> GDExtension object.
/// Implements <see cref="NetworkLayer"/> so it drops in transparently wherever
/// a <see cref="DenseLayer"/> is used, with AVX2-accelerated forward/backward passes.
///
/// Activation code passed to native (matches DenseLayer.AppendSerialized):
///   0 = None,  1 = Tanh,  2 = ReLU
/// Optimizer code (matches RLOptimizerKind):
///   0 = Adam,  1 = SGD,  -1 = None
/// </summary>
internal sealed class NativeDenseLayer : NetworkLayer, INativeLayer
{
    private readonly GodotObject _native;
    private readonly int         _inputSize;
    private readonly int         _outputSize;
    private readonly int         _activationCode;  // 0=None, 1=Tanh, 2=ReLU

    public override int InputSize  => _inputSize;
    public override int OutputSize => _outputSize;

    public NativeDenseLayer(int inputSize, int outputSize,
                             RLActivationKind? activation, RLOptimizerKind optimizer)
    {
        _inputSize      = inputSize;
        _outputSize     = outputSize;
        _activationCode = activation.HasValue ? (int)activation.Value + 1 : 0;

        _native = ClassDB.Instantiate("RlDenseLayer").AsGodotObject()
                  ?? throw new InvalidOperationException(
                      "[NativeDenseLayer] Failed to instantiate RlDenseLayer. " +
                      "Make sure the native library is built and " +
                      "rl_cnn.gdextension is loaded in the project.");

        _native.Call(RlDenseLayerNativeFunctions.Initialize,
            inputSize, outputSize, _activationCode, (int)optimizer);
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    public override float[] Forward(float[] input, bool isTraining = false)
        => (float[])_native.Call(RlDenseLayerNativeFunctions.Forward, (Variant)input);

    public override VectorBatch ForwardBatch(VectorBatch input)
    {
        var flat = (float[])_native.Call(RlDenseLayerNativeFunctions.ForwardBatch,
                       (Variant)input.Data, input.BatchSize);
        var batch = new VectorBatch(input.BatchSize, _outputSize);
        Array.Copy(flat, batch.Data, flat.Length);
        return batch;
    }

    // ── Single-sample backward (immediate weight update) ─────────────────────

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
        => (float[])_native.Call(RlDenseLayerNativeFunctions.Backward,
               (Variant)outputGrad, learningRate, gradScale);

    // ── Batch accumulate + apply ──────────────────────────────────────────────

    public override GradientBuffer CreateGradientBuffer()
    {
        var buf = new NativeGradientBuffer(this);
        buf.NativeData = _native.Call(RlDenseLayerNativeFunctions.CreateGradientBuffer);
        return buf;
    }

    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
    {
        var nb = (NativeGradientBuffer)buffer;
        // Use explicit return payload so updated buffer survives Variant marshaling.
        var result = (Godot.Collections.Array)_native.Call(
            RlDenseLayerNativeFunctions.AccumulateGradientsWithBuffer,
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
        _native.Call(RlDenseLayerNativeFunctions.ApplyGradients,
            nb.NativeData, learningRate, gradScale);
    }

    // ── Frozen gradient (SAC dQ/da) ───────────────────────────────────────────

    public override float[] ComputeInputGrad(float[] outputGrad)
        => (float[])_native.Call(RlDenseLayerNativeFunctions.ComputeInputGrad,
               (Variant)outputGrad);

    // ── INativeLayer: gradient norm for global clipping ───────────────────────

    public float GradNormSquared(Variant nativeBuffer)
        => (float)_native.Call(RlDenseLayerNativeFunctions.GradNormSquared, nativeBuffer);

    // ── Target-network copy ───────────────────────────────────────────────────

    public override void CopyFrom(NetworkLayer source)
    {
        switch (source)
        {
            case NativeDenseLayer ndl:
                // Fast path: both native — direct native call, no float[] allocation.
                _native.Call(RlDenseLayerNativeFunctions.CopyWeightsFrom, ndl._native);
                return;

            case DenseLayer dl:
                // Cross-backend: serialize out of C# layer, load into native.
                var w = new List<float>(); var s = new List<int>();
                dl.AppendSerialized(w, s);
                int wi = 0, si = 0;
                LoadSerialized(w, ref wi, s, ref si);
                return;

            default:
                throw new InvalidOperationException(
                    $"[NativeDenseLayer] CopyFrom: unsupported source type {source.GetType().Name}.");
        }
    }

    public override void SoftUpdateFrom(NetworkLayer source, float tau)
    {
        switch (source)
        {
            case NativeDenseLayer ndl:
                _native.Call(RlDenseLayerNativeFunctions.SoftUpdateFrom, ndl._native, tau);
                return;

            case DenseLayer dl:
                // Cross-backend: load C# weights into a temp native layer, polyak from there.
                var temp = new NativeDenseLayer(_inputSize, _outputSize, null, RLOptimizerKind.None);
                var w = new List<float>(); var s = new List<int>();
                dl.AppendSerialized(w, s);
                int wi = 0, si = 0;
                temp.LoadSerialized(w, ref wi, s, ref si);
                _native.Call(RlDenseLayerNativeFunctions.SoftUpdateFrom, temp._native, tau);
                return;

            default:
                throw new InvalidOperationException(
                    $"[NativeDenseLayer] SoftUpdateFrom: unsupported source type {source.GetType().Name}.");
        }
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        var w = (float[])_native.Call(RlDenseLayerNativeFunctions.GetWeights);
        var s = (int[])  _native.Call(RlDenseLayerNativeFunctions.GetShapes);
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
            if (typeCode != (int)RLLayerKind.Dense)
                throw new InvalidOperationException(
                    $"[NativeDenseLayer] Expected Dense layer type ({(int)RLLayerKind.Dense}), got {typeCode}.");
        }

        var serializedIn   = shapes[si++];
        var serializedOut  = shapes[si++];
        var serializedAct  = shapes[si++];

        if (serializedIn != _inputSize || serializedOut != _outputSize)
            throw new InvalidOperationException(
                "[NativeDenseLayer] Checkpoint layer shape does not match the active network.");
        if (serializedAct != _activationCode)
            throw new InvalidOperationException(
                "[NativeDenseLayer] Checkpoint activation does not match the active network.");

        int wCount = _inputSize * _outputSize + _outputSize;
        var wSlice = new float[wCount];
        for (var i = 0; i < wCount; i++) wSlice[i] = weights[wi++];

        _native.Call(RlDenseLayerNativeFunctions.SetWeights,
            (Variant)wSlice,
            (Variant)new int[] { 0, _inputSize, _outputSize, _activationCode });
    }
}
