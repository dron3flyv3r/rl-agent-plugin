using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Flatten layer — complete passthrough in Stage 1.
/// In Stage 2 this will reshape [C, H, W] image tensors into a flat feature vector.
/// No learnable parameters.
/// </summary>
internal sealed class FlattenLayer : NetworkLayer
{
    private readonly int _size;

    public override int InputSize  => _size;
    public override int OutputSize => _size;

    public FlattenLayer(int size) => _size = size;

    // ── Forward ──────────────────────────────────────────────────────────────

    public override float[] Forward(float[] input, bool isTraining = false) => input;

    public override VectorBatch ForwardBatch(VectorBatch input) => input;

    // ── Backward / gradient ───────────────────────────────────────────────

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
        => outputGrad;

    public override GradientBuffer CreateGradientBuffer() => new(0, 0);

    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
        => outputGrad;

    public override void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale) { }

    public override float[] ComputeInputGrad(float[] outputGrad) => outputGrad;

    // ── Serialization ─────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add((int)RLLayerKind.Flatten);
        shapes.Add(_size);
    }

    public override void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false)
    {
        if (!isLegacy)
        {
            var typeCode = shapes[si++];
            if (typeCode != (int)RLLayerKind.Flatten)
                throw new InvalidOperationException($"Expected Flatten layer type ({(int)RLLayerKind.Flatten}), got {typeCode}.");
        }
        si++; // size
    }
}
