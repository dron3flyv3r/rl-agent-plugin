using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Interface implemented by native layer wrappers to expose gradient-norm
/// computation over the native buffer without marshalling.
/// </summary>
internal interface INativeLayer
{
    float GradNormSquared(Variant nativeBuffer);
}

/// <summary>
/// Gradient buffer backed by a native GDExtension <c>PackedFloat32Array</c>.
/// The base-class <see cref="GradientBuffer"/> float[] arrays are zero-length
/// placeholders — all data lives in <see cref="NativeData"/>.
///
/// Passed directly to native <c>accumulate_gradients</c> and
/// <c>apply_gradients</c> calls without copying, matching the pattern used
/// by <see cref="CnnGradientToken"/>.
/// </summary>
internal sealed class NativeGradientBuffer : GradientBuffer
{
    private readonly INativeLayer _owner;

    /// <summary>
    /// The native gradient buffer Variant (holds a PackedFloat32Array).
    /// Passed directly to GDExtension calls — mutations made by
    /// <c>accumulate_gradients</c> persist in this reference.
    /// </summary>
    public Variant NativeData { get; set; }

    public NativeGradientBuffer(INativeLayer owner) : base(0, 0)
        => _owner = owner;

    /// <summary>
    /// Delegates to native <c>grad_norm_squared</c> so
    /// <see cref="PolicyValueNetwork"/> global-norm clipping works correctly.
    /// </summary>
    public override float SumSquares() => _owner.GradNormSquared(NativeData);
}
