using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// C# wrapper around the native <c>RlGruLayer</c> GDExtension object.
///
/// Single-step inference:
///   <see cref="ForwardRecurrent"/> — reads h from <see cref="RecurrentState"/>,
///   runs one GRU step, updates state in-place, returns h output.
///
/// Training (BPTT) is invoked through <see cref="AccumulateSequenceGradients"/>.
///
/// Serialization kind: <see cref="RLLayerKind.Gru"/> = 5.
/// </summary>
internal sealed class NativeGruLayer : NetworkLayer, INativeLayer
{
    private readonly GodotObject _native;
    private readonly int         _inputSize;
    private readonly int         _hiddenSize;
    private readonly float       _gradClipNorm;

    public override int InputSize  => _inputSize;
    public override int OutputSize => _hiddenSize;
    public override bool IsRecurrent => true;

    public NativeGruLayer(int inputSize, int hiddenSize, RLOptimizerKind optimizer, float gradClipNorm = 1.0f)
    {
        _inputSize  = inputSize;
        _hiddenSize = hiddenSize;
        _gradClipNorm = gradClipNorm;

        _native = ClassDB.Instantiate("RlGruLayer").AsGodotObject()
                  ?? throw new InvalidOperationException(
                      "[NativeGruLayer] Failed to instantiate RlGruLayer. " +
                      "Ensure the native library is built and rl_cnn.gdextension is loaded.");

        _native.Call(RlGruLayerNativeFunctions.Initialize, inputSize, hiddenSize, (int)optimizer);
    }

    // ── Single-step recurrent forward ────────────────────────────────────────

    public override float[] ForwardRecurrent(float[] input, RecurrentState state)
    {
        var result = (float[])_native.Call(RlGruLayerNativeFunctions.Forward,
            (Variant)input, (Variant)state.H);
        Array.Copy(result, state.H, _hiddenSize);
        return state.H;
    }

    public override void ResetState(RecurrentState state)
        => Array.Clear(state.H, 0, state.H.Length);

    // ── Standard forward (stateless, zero h) ──────────────────────────────────

    public override float[] Forward(float[] input, bool isTraining = false)
    {
        var zeros = new float[_hiddenSize];
        return (float[])_native.Call(RlGruLayerNativeFunctions.Forward,
            (Variant)input, (Variant)zeros);
    }

    public override VectorBatch ForwardBatch(VectorBatch input)
    {
        var output = new VectorBatch(input.BatchSize, _hiddenSize);
        var zeros  = new float[_hiddenSize];
        for (int b = 0; b < input.BatchSize; ++b)
        {
            var x = input.CopyRow(b);
            var h = (float[])_native.Call(RlGruLayerNativeFunctions.Forward,
                (Variant)x, (Variant)zeros);
            Array.Copy(h, 0, output.Data, b * _hiddenSize, _hiddenSize);
        }
        return output;
    }

    // ── Backward (no-op; sequence BPTT only) ─────────────────────────────────

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
        => new float[_inputSize];

    // ── Batch accumulate + apply ──────────────────────────────────────────────

    public override GradientBuffer CreateGradientBuffer()
    {
        var buf = new NativeGradientBuffer(this);
        buf.NativeData = _native.Call(RlGruLayerNativeFunctions.CreateGradientBuffer);
        return buf;
    }

    /// <summary>Runs BPTT over the cached sequence.</summary>
    public float[] AccumulateSequenceGradients(float[] seqHGrads, int seqLen,
                                                NativeGradientBuffer buffer,
                                                float[] h0)
    {
        var result = (Godot.Collections.Array)_native.Call(
            RlGruLayerNativeFunctions.AccumulateSequenceGradients,
            (Variant)seqHGrads, seqLen, buffer.NativeData, (Variant)h0);
        var inputGrads    = (float[])result[0];
        var updatedBuffer = (float[])result[1];
        buffer.NativeData = (Variant)updatedBuffer;
        return inputGrads;
    }

    /// <summary>Runs sequence forward and caches result for BPTT.</summary>
    public float[] ForwardSequence(float[] flatInputs, int seqLen, float[] h0)
        => (float[])_native.Call(RlGruLayerNativeFunctions.ForwardSequence,
               (Variant)flatInputs, seqLen, (Variant)h0);

    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
        => new float[_inputSize];

    public override void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale)
    {
        var nb = (NativeGradientBuffer)buffer;
        _native.Call(RlGruLayerNativeFunctions.ApplyGradients,
            nb.NativeData, learningRate, gradScale, _gradClipNorm);
    }

    // ── Frozen gradient ───────────────────────────────────────────────────────

    public override float[] ComputeInputGrad(float[] outputGrad)
        => new float[_inputSize];

    // ── INativeLayer ──────────────────────────────────────────────────────────

    public float GradNormSquared(Variant nativeBuffer)
        => (float)_native.Call(RlGruLayerNativeFunctions.GradNormSquared, nativeBuffer);

    // ── Target-network copy ───────────────────────────────────────────────────

    public override void CopyFrom(NetworkLayer source)
    {
        if (source is NativeGruLayer other)
            _native.Call(RlGruLayerNativeFunctions.CopyWeightsFrom, other._native);
        else
            throw new InvalidOperationException(
                $"[NativeGruLayer] CopyFrom: unsupported source type {source.GetType().Name}.");
    }

    public override void SoftUpdateFrom(NetworkLayer source, float tau)
    {
        if (source is NativeGruLayer other)
            _native.Call(RlGruLayerNativeFunctions.SoftUpdateFrom, other._native, tau);
        else
            throw new InvalidOperationException(
                $"[NativeGruLayer] SoftUpdateFrom: unsupported source type {source.GetType().Name}.");
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        var w = (float[])_native.Call(RlGruLayerNativeFunctions.GetWeights);
        var s = (int[])  _native.Call(RlGruLayerNativeFunctions.GetShapes);
        foreach (var v in w) weights.Add(v);
        foreach (var v in s) shapes.Add(v);
    }

    public override void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false)
    {
        var typeCode = shapes[si++];
        if (typeCode != (int)RLLayerKind.Gru)
            throw new InvalidOperationException(
                $"[NativeGruLayer] Expected Gru layer type ({(int)RLLayerKind.Gru}), got {typeCode}.");

        var savedIn     = shapes[si++];
        var savedHidden = shapes[si++];
        if (savedIn != _inputSize || savedHidden != _hiddenSize)
            throw new InvalidOperationException(
                "[NativeGruLayer] Checkpoint layer shape does not match the active network.");

        // Weight count: 3*h*in + 3*h*h + 3*h
        int wCount = 3 * _hiddenSize * _inputSize + 3 * _hiddenSize * _hiddenSize + 3 * _hiddenSize;
        var wSlice = new float[wCount];
        for (var i = 0; i < wCount; i++) wSlice[i] = weights[wi++];

        _native.Call(RlGruLayerNativeFunctions.SetWeights,
            (Variant)wSlice,
            (Variant)new int[] { (int)RLLayerKind.Gru, _inputSize, _hiddenSize });
    }
}
