using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Converts the image to single-channel grayscale (<see cref="Image.Format.L8"/>).
/// After this transform <c>OutputChannels</c> becomes 1.
///
/// Place this early in the pipeline — transforming to grayscale before
/// <see cref="RLCropTransform"/> reduces the number of bytes copied during the crop.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLGrayscaleTransform : RLCameraTransform
{
    public override Image Apply(Image image)
    {
        image.Convert(Image.Format.L8);
        return image;
    }

    public override int ComputeOutputChannels(int inputChannels) => 1;
}
