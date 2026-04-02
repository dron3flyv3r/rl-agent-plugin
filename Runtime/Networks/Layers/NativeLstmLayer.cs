using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// C# wrapper around the native <c>RlLstmLayer</c> GDExtension object.
///
/// Single-step inference:
///   <see cref="ForwardRecurrent"/> — reads h/c from <see cref="RecurrentState"/>,
///   runs one LSTM step, updates state in-place, returns h output.
///
/// Training (BPTT) is invoked through the sequence-level API exposed on
/// <see cref="GradientBuffer"/> via <see cref="AccumulateSequenceGradients"/>.
///
/// Serialization kind: <see cref="RLLayerKind.Lstm"/> = 4.
/// </summary>
internal sealed class NativeLstmLayer : NetworkLayer, INativeLayer
{
    private readonly GodotObject _native;
    private readonly int         _inputSize;
    private readonly int         _hiddenSize;
    private readonly float       _gradClipNorm;

    public override int InputSize  => _inputSize;
    public override int OutputSize => _hiddenSize;
    public override bool IsRecurrent => true;

    public NativeLstmLayer(int inputSize, int hiddenSize, RLOptimizerKind optimizer, float gradClipNorm = 1.0f)
    {
        _inputSize  = inputSize;
        _hiddenSize = hiddenSize;
        _gradClipNorm = gradClipNorm;

        _native = ClassDB.Instantiate("RlLstmLayer").AsGodotObject()
                  ?? throw new InvalidOperationException(
                      "[NativeLstmLayer] Failed to instantiate RlLstmLayer. " +
                      "Ensure the native library is built and rl_cnn.gdextension is loaded.");

        _native.Call(RlLstmLayerNativeFunctions.Initialize, inputSize, hiddenSize, (int)optimizer);
    }

    // ── Single-step recurrent forward ────────────────────────────────────────

    public override float[] ForwardRecurrent(float[] input, RecurrentState state)
    {
        var h = state.H;
        var c = state.C ?? new float[_hiddenSize];

        var result = (float[])_native.Call(RlLstmLayerNativeFunctions.Forward,
            (Variant)input, (Variant)h, (Variant)c);

        // result = [h_next | c_next]
        Array.Copy(result, 0,            state.H, 0, _hiddenSize);
        if (state.C != null)
            Array.Copy(result, _hiddenSize, state.C, 0, _hiddenSize);
        else
            state.C = result[_hiddenSize..];

        return state.H;
    }

    public override void ResetState(RecurrentState state)
    {
        Array.Clear(state.H, 0, state.H.Length);
        if (state.C != null) Array.Clear(state.C, 0, state.C.Length);
    }

    // ── Standard (non-recurrent) forward — delegates to recurrent with temp state ──

    public override float[] Forward(float[] input, bool isTraining = false)
    {
        // Stateless single-step: use zero h/c
        var zeros = new float[_hiddenSize];
        var result = (float[])_native.Call(RlLstmLayerNativeFunctions.Forward,
            (Variant)input, (Variant)zeros, (Variant)zeros);
        return result[.._hiddenSize];
    }

    public override VectorBatch ForwardBatch(VectorBatch input)
    {
        // Batch inference with stateless zero-initialised h/c (one step per sample)
        var output = new VectorBatch(input.BatchSize, _hiddenSize);
        var zeros  = new float[_hiddenSize];
        for (int b = 0; b < input.BatchSize; ++b)
        {
            var x = input.CopyRow(b);
            var result = (float[])_native.Call(RlLstmLayerNativeFunctions.Forward,
                (Variant)x, (Variant)zeros, (Variant)zeros);
            Array.Copy(result, 0, output.Data, b * _hiddenSize, _hiddenSize);
        }
        return output;
    }

    // ── Single-sample backward (not used for recurrent; sequence BPTT is used) ──

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
        => new float[_inputSize];   // no-op; training goes through sequence API

    // ── Batch accumulate + apply (sequence BPTT) ─────────────────────────────

    public override GradientBuffer CreateGradientBuffer()
    {
        var buf = new NativeGradientBuffer(this);
        buf.NativeData = _native.Call(RlLstmLayerNativeFunctions.CreateGradientBuffer);
        return buf;
    }

    /// <summary>
    /// Runs BPTT over the sequence cached by the most recent <c>forward_sequence</c>
    /// call (triggered externally by the training loop).
    /// </summary>
    public float[] AccumulateSequenceGradients(float[] seqHGrads, int seqLen,
                                                NativeGradientBuffer buffer,
                                                float[] h0, float[] c0)
    {
        var result = (Godot.Collections.Array)_native.Call(
            RlLstmLayerNativeFunctions.AccumulateSequenceGradients,
            (Variant)seqHGrads, buffer.NativeData, (Variant)h0, (Variant)c0);
        var inputGrads    = (float[])result[0];
        var updatedBuffer = (float[])result[1];
        buffer.NativeData = (Variant)updatedBuffer;
        return inputGrads;
    }

    /// <summary>Runs sequence forward and caches result for BPTT.</summary>
    public float[] ForwardSequence(float[] flatInputs, int seqLen, float[] h0, float[] c0)
        => (float[])_native.Call(RlLstmLayerNativeFunctions.ForwardSequence,
               (Variant)flatInputs, seqLen, (Variant)h0, (Variant)c0);

    // AccumulateGradients for the non-sequence path (required by abstract base)
    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
        => new float[_inputSize];   // no-op; only sequence BPTT is supported

    public override void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale)
    {
        var nb = (NativeGradientBuffer)buffer;
        _native.Call(RlLstmLayerNativeFunctions.ApplyGradients,
            nb.NativeData, learningRate, gradScale, _gradClipNorm);
    }

    // ── Frozen gradient (SAC dQ/da) ───────────────────────────────────────────

    public override float[] ComputeInputGrad(float[] outputGrad)
        => new float[_inputSize];   // recurrent layers are not used in frozen SAC critic path

    // ── INativeLayer ──────────────────────────────────────────────────────────

    public float GradNormSquared(Variant nativeBuffer)
        => (float)_native.Call(RlLstmLayerNativeFunctions.GradNormSquared, nativeBuffer);

    // ── Target-network copy ───────────────────────────────────────────────────

    public override void CopyFrom(NetworkLayer source)
    {
        if (source is NativeLstmLayer other)
            _native.Call(RlLstmLayerNativeFunctions.CopyWeightsFrom, other._native);
        else
            throw new InvalidOperationException(
                $"[NativeLstmLayer] CopyFrom: unsupported source type {source.GetType().Name}.");
    }

    public override void SoftUpdateFrom(NetworkLayer source, float tau)
    {
        if (source is NativeLstmLayer other)
            _native.Call(RlLstmLayerNativeFunctions.SoftUpdateFrom, other._native, tau);
        else
            throw new InvalidOperationException(
                $"[NativeLstmLayer] SoftUpdateFrom: unsupported source type {source.GetType().Name}.");
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        var w = (float[])_native.Call(RlLstmLayerNativeFunctions.GetWeights);
        var s = (int[])  _native.Call(RlLstmLayerNativeFunctions.GetShapes);
        foreach (var v in w) weights.Add(v);
        foreach (var v in s) shapes.Add(v);
    }

    public override void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false)
    {
        var typeCode = shapes[si++];
        if (typeCode != (int)RLLayerKind.Lstm)
            throw new InvalidOperationException(
                $"[NativeLstmLayer] Expected Lstm layer type ({(int)RLLayerKind.Lstm}), got {typeCode}.");

        var savedIn     = shapes[si++];
        var savedHidden = shapes[si++];
        if (savedIn != _inputSize || savedHidden != _hiddenSize)
            throw new InvalidOperationException(
                "[NativeLstmLayer] Checkpoint layer shape does not match the active network.");

        // Weight count: 4*h*in + 4*h*h + 4*h
        int wCount = 4 * _hiddenSize * _inputSize + 4 * _hiddenSize * _hiddenSize + 4 * _hiddenSize;
        var wSlice = new float[wCount];
        for (var i = 0; i < wCount; i++) wSlice[i] = weights[wi++];

        _native.Call(RlLstmLayerNativeFunctions.SetWeights,
            (Variant)wSlice,
            (Variant)new int[] { (int)RLLayerKind.Lstm, _inputSize, _hiddenSize });
    }
}
