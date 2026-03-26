using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Dropout layer — randomly zeroes outputs with probability <c>rate</c> during training,
/// scaling survivors by 1/(1−rate) (inverted dropout). Pure passthrough in inference mode.
/// No learnable parameters; <c>SoftUpdateFrom</c> and <c>CopyFrom</c> are no-ops.
/// </summary>
internal sealed class DropoutLayer : NetworkLayer
{
    private readonly int   _size;
    private readonly float _rate;
    private readonly float _invKeep; // 1f / (1f - rate)
    private readonly Random _rng = new();

    // Stored during Forward(isTraining=true) for use in Backward / AccumulateGradients
    private bool[]? _mask;

    public override int InputSize  => _size;
    public override int OutputSize => _size;

    public DropoutLayer(int size, float rate)
    {
        _size    = size;
        _rate    = Math.Clamp(rate, 0f, 1f);
        _invKeep = _rate < 1f ? 1f / (1f - _rate) : 0f;
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    public override float[] Forward(float[] input, bool isTraining = false)
    {
        if (!isTraining)
        {
            _mask = null;
            return input; // passthrough — no copy needed (callers treat output as read-only)
        }

        _mask = new bool[_size];
        var output = new float[_size];
        for (var i = 0; i < _size; i++)
        {
            _mask[i]  = _rng.NextSingle() >= _rate;
            output[i] = _mask[i] ? input[i] * _invKeep : 0f;
        }

        return output;
    }

    public override VectorBatch ForwardBatch(VectorBatch input) => input; // inference-only, passthrough

    // ── Single-sample backward (immediate weight update) ──────────────────

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
        => ApplyMask(outputGrad);

    // ── Batch accumulate + apply ──────────────────────────────────────────

    public override GradientBuffer CreateGradientBuffer() => new(0, 0);

    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
        => ApplyMask(outputGrad);

    public override void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale) { }

    // ── Frozen gradient ───────────────────────────────────────────────────

    public override float[] ComputeInputGrad(float[] outputGrad) => ApplyMask(outputGrad);

    // ── Serialization ─────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add((int)RLLayerKind.Dropout);
        shapes.Add(_size);
        shapes.Add(BitConverter.SingleToInt32Bits(_rate));
        // No learnable weights
    }

    public override void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false)
    {
        if (!isLegacy)
        {
            var typeCode = shapes[si++];
            if (typeCode != (int)RLLayerKind.Dropout)
                throw new InvalidOperationException($"Expected Dropout layer type ({(int)RLLayerKind.Dropout}), got {typeCode}.");
        }
        si++; // size
        si++; // rate bits
        // No weights to read
    }

    // ── Private helpers ───────────────────────────────────────────────────

    private float[] ApplyMask(float[] grad)
    {
        if (_mask is null)
            return grad; // inference passthrough

        var result = new float[_size];
        for (var i = 0; i < _size; i++)
            result[i] = _mask[i] ? grad[i] * _invKeep : 0f;
        return result;
    }
}
