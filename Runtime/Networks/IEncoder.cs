using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Common interface for all CNN encoder implementations.
/// The CPU path is <see cref="CnnEncoder"/> (C++ GDExtension).
/// The GPU path will be <c>GpuCnnEncoder</c> (Vulkan compute via RenderingDevice).
///
/// <see cref="PolicyValueNetwork"/> holds an <c>IEncoder?</c> per image stream so
/// the two implementations are interchangeable without touching any training code.
/// </summary>
internal interface IEncoder
{
    /// <summary>Size of the embedding vector produced by <see cref="Forward"/>.</summary>
    int OutputSize { get; }

    /// <summary>
    /// True when this encoder supports a full-batch forward+backward dispatch
    /// (e.g. GPU path). When false, <see cref="PolicyValueNetwork"/> falls back to
    /// the per-sample accumulation loop using <see cref="CreateGradientToken"/> etc.
    /// </summary>
    bool SupportsBatchedTraining { get; }

    // ── Single-sample inference (rollout collection, evaluation) ─────────────

    /// <summary>Encodes a single observation and returns an embedding of length <see cref="OutputSize"/>.</summary>
    float[] Forward(float[] input);

    /// <summary>
    /// Encodes a contiguous batch of observations laid out as
    /// <c>[sample0_input..., sample1_input..., ...]</c> into
    /// <paramref name="outputBatch"/> laid out the same way by sample.
    /// </summary>
    void ForwardBatch(float[] inputBatch, int batchSize, float[] outputBatch);

    // ── Per-sample gradient accumulation (CPU training path) ─────────────────

    /// <summary>Allocates a zeroed gradient accumulation token for one mini-batch.</summary>
    ICnnGradientToken CreateGradientToken();

    /// <summary>
    /// Backpropagates <paramref name="outputGrad"/> through the encoder and accumulates
    /// parameter gradients into <paramref name="token"/>.
    /// Returns the gradient with respect to the encoder input (pixel space).
    /// </summary>
    float[] AccumulateGradients(float[] outputGrad, ICnnGradientToken token);

    /// <summary>
    /// Batched version of <see cref="AccumulateGradients"/> for encoders that can
    /// retain forward caches for an entire mini-batch on-device. The output gradients
    /// are laid out as <c>[sample0_outputGrad..., sample1_outputGrad..., ...]</c>.
    /// </summary>
    void AccumulateGradientsBatch(float[] outputGradBatch, int batchSize, ICnnGradientToken token);

    /// <summary>Applies one Adam step using the accumulated gradients in <paramref name="token"/>.</summary>
    void ApplyGradients(ICnnGradientToken token, float learningRate, float gradScale);

    /// <summary>Returns the squared L2 norm of the accumulated gradients (used for global gradient clipping).</summary>
    float GradNormSquared(ICnnGradientToken token);

    // ── Serialization ─────────────────────────────────────────────────────────

    /// <summary>Appends all weights and shape descriptors to the provided collections (checkpoint save).</summary>
    void AppendSerialized(ICollection<float> weights, ICollection<int> shapes);

    /// <summary>Restores weights from a flat weight/shape buffer at the given offsets (checkpoint load).</summary>
    void LoadSerialized(IReadOnlyList<float> weights, ref int wi,
                        IReadOnlyList<int>   shapes,  ref int si);

    // ── Weight copy (SAC target network sync) ─────────────────────────────────

    /// <summary>Copies all weights from this encoder into <paramref name="other"/> (same architecture required).</summary>
    void CopyWeightsTo(IEncoder other);
}

/// <summary>
/// Opaque handle to a gradient accumulation buffer owned by an <see cref="IEncoder"/>.
/// The CPU implementation wraps a <c>float[]</c>; the GPU implementation will hold GPU buffer RIDs.
/// </summary>
internal interface ICnnGradientToken { }
