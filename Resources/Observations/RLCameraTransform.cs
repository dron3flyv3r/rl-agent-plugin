using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Abstract base for image-processing transforms applied to the
/// <see cref="RLCameraSensor2D"/> output before it is fed to the network.
///
/// Transforms are executed in array order. Each transform receives the
/// <see cref="Image"/> produced by the previous one and may return the same
/// instance (mutated) or a new one.
///
/// To create a custom transform:
///   1. Subclass <c>RLCameraTransform</c> and mark it <c>[GlobalClass]</c> so
///      it appears in the Inspector resource picker.
///   2. Add <c>[Export]</c> properties for any user-configurable parameters.
///   3. Implement <c>Apply</c> to perform the pixel operation.
///   4. Override <c>ComputeOutputSize</c> if the transform changes image dimensions.
///   5. Override <c>ComputeOutputChannels</c> if the transform changes channel count.
/// </summary>
[GlobalClass]
[Tool]
public abstract partial class RLCameraTransform : Resource
{
    /// <summary>
    /// Apply the transform to <paramref name="image"/> and return the result.
    /// The implementation may mutate and return the same <see cref="Image"/>
    /// instance or allocate and return a new one.
    /// </summary>
    public abstract Image Apply(Image image);

    /// <summary>
    /// Compute the output spatial size this transform produces for a given input size.
    /// Default: pass-through (output == input).
    /// </summary>
    public virtual Vector2I ComputeOutputSize(Vector2I inputSize) => inputSize;

    /// <summary>
    /// Compute the output channel count this transform produces for a given input channel count.
    /// Default: pass-through (output == input).
    /// </summary>
    public virtual int ComputeOutputChannels(int inputChannels) => inputChannels;
}
