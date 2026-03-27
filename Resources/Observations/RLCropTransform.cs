using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Crops a rectangular region out of the image.
/// <see cref="Offset"/> is clamped so the crop window never exceeds the image boundary.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLCropTransform : RLCameraTransform
{
    /// <summary>Top-left pixel offset of the crop window. (0, 0) = top-left corner.</summary>
    [Export]
    public Vector2I Offset
    {
        get => _offset;
        set { _offset = value; EmitChanged(); }
    }

    /// <summary>Width and height of the output image in pixels.</summary>
    [Export]
    public Vector2I Size
    {
        get => _size;
        set { _size = new Vector2I(Mathf.Max(1, value.X), Mathf.Max(1, value.Y)); EmitChanged(); }
    }

    private Vector2I _offset = Vector2I.Zero;
    private Vector2I _size   = new(64, 64);

    public override Image Apply(Image image)
    {
        var clampedOffset = new Vector2I(
            Mathf.Clamp(_offset.X, 0, Mathf.Max(0, image.GetWidth()  - _size.X)),
            Mathf.Clamp(_offset.Y, 0, Mathf.Max(0, image.GetHeight() - _size.Y)));
        return image.GetRegion(new Rect2I(clampedOffset, _size));
    }

    public override Vector2I ComputeOutputSize(Vector2I inputSize) => _size;
}
