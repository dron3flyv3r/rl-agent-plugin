using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Abstract base for background fill strategies applied to <see cref="RLCameraSensor2D"/>
/// when the camera viewport extends beyond scene content.
///
/// When a <c>Background</c> is configured on the sensor, the viewport renders with
/// <c>TransparentBg = true</c>. Any pixel with <c>alpha == 0</c> is considered empty
/// (out-of-bounds) and this strategy is responsible for filling it.
///
/// The image passed to <see cref="Apply"/> is always <see cref="Image.Format.Rgba8"/>.
/// The returned image must also be <see cref="Image.Format.Rgba8"/> — the sensor
/// will convert to RGB after the background pass, before the transform pipeline runs.
///
/// To create a custom background:
///   1. Subclass <c>RLCameraBackground</c> and mark it <c>[GlobalClass]</c>.
///   2. Add <c>[Export]</c> properties for any user-configurable parameters.
///   3. Implement <see cref="Apply"/> to fill transparent pixels and return the result.
/// </summary>
[GlobalClass]
[Tool]
public abstract partial class RLCameraBackground : Resource
{
    /// <summary>
    /// Fill transparent (alpha == 0) pixels in <paramref name="image"/> and return
    /// the result. The implementation may mutate and return the same instance or
    /// allocate and return a new one. The image is always <see cref="Image.Format.Rgba8"/>.
    /// </summary>
    public abstract Image Apply(Image image);
}
