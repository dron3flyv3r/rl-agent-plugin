using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Abstract base for all neural-network layers.
///
/// Layers cache their own forward-pass state internally so callers never need
/// to thread cache structs through the backward pass.
///
/// To implement a custom layer:
///   1. Subclass NetworkLayer.
///   2. Override all abstract members.
///   3. Override SoftUpdateFrom / CopyFrom if the layer has learnable parameters.
///   4. Pair with a [GlobalClass] RLLayerDef subclass for Inspector exposure.
/// </summary>
internal abstract class NetworkLayer
{
    public abstract int InputSize  { get; }
    public abstract int OutputSize { get; }

    // ── Forward ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Single-sample forward pass. Caches state needed for Backward / AccumulateGradients.
    /// isTraining enables stochastic layers (Dropout). Default false = inference mode.
    /// </summary>
    public abstract float[] Forward(float[] input, bool isTraining = false);

    /// <summary>
    /// Batch inference forward pass. No gradient caching. No stochastic behaviour.
    /// </summary>
    public abstract VectorBatch ForwardBatch(VectorBatch input);

    // ── Single-sample backward (immediate weight update) ────────────────────

    /// <summary>
    /// Backpropagates outputGrad through the layer, updates weights, and returns
    /// the input-space gradient. Uses state cached by the most recent Forward call.
    /// gradScale is applied to weight updates only (not to the returned input gradient).
    /// </summary>
    public abstract float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f);

    // ── Batch accumulate + apply ─────────────────────────────────────────────

    /// <summary>Creates a zeroed gradient buffer sized for this layer.</summary>
    public abstract GradientBuffer CreateGradientBuffer();

    /// <summary>
    /// Accumulates gradients into buffer and returns the input-space gradient.
    /// Does NOT update weights. Uses state cached by the most recent Forward call.
    /// </summary>
    public abstract float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer);

    /// <summary>Applies accumulated gradients in buffer to layer weights.</summary>
    public abstract void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale);

    // ── Frozen gradient (SAC dQ/da) ──────────────────────────────────────────

    /// <summary>
    /// Returns the input-space gradient WITHOUT updating any weights.
    /// Uses state cached by the most recent Forward call.
    /// </summary>
    public abstract float[] ComputeInputGrad(float[] outputGrad);

    // ── Target-network copy (SAC) ────────────────────────────────────────────

    /// <summary>
    /// Polyak-averages weights from source: θ = τ·θ_src + (1−τ)·θ.
    /// Default: no-op for parameter-free layers.
    /// </summary>
    public virtual void SoftUpdateFrom(NetworkLayer source, float tau) { }

    /// <summary>
    /// Hard-copies weights from source. Default: no-op for parameter-free layers.
    /// </summary>
    public virtual void CopyFrom(NetworkLayer source) { }

    // ── Serialization ────────────────────────────────────────────────────────

    /// <summary>
    /// Appends this layer's shape descriptor (ints) and weight values (floats)
    /// to the provided lists. The first int written must be the RLLayerKind code.
    /// </summary>
    public abstract void AppendSerialized(ICollection<float> weights, ICollection<int> shapes);

    /// <summary>
    /// Reads this layer's data from the checkpoint arrays starting at the given offsets.
    /// Advances wi and si past the data consumed.
    /// isLegacy = true → v2 format (3-int Dense descriptor, no type prefix).
    /// </summary>
    public abstract void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false);
}
