using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Scales the image to <see cref="TargetSize"/> using the chosen interpolation filter.
/// Useful for downsampling a large render to the resolution actually fed to the network.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLResizeTransform : RLCameraTransform
{
    /// <summary>Output width and height in pixels.</summary>
    [Export]
    public Vector2I TargetSize
    {
        get => _targetSize;
        set { _targetSize = new Vector2I(Mathf.Max(1, value.X), Mathf.Max(1, value.Y)); EmitChanged(); }
    }

    /// <summary>Interpolation filter used when resizing.</summary>
    [Export]
    public Image.Interpolation Interpolation
    {
        get => _interpolation;
        set { _interpolation = value; EmitChanged(); }
    }

    private Vector2I            _targetSize     = new(64, 64);
    private Image.Interpolation _interpolation  = Image.Interpolation.Bilinear;

    public override Image Apply(Image image)
    {
        image.Resize(_targetSize.X, _targetSize.Y, _interpolation);
        return image;
    }

    public override Vector2I ComputeOutputSize(Vector2I inputSize) => _targetSize;
}
