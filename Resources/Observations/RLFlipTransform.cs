using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Flips the image horizontally, vertically, or both.
/// Does not change image dimensions or channel count.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLFlipTransform : RLCameraTransform
{
    /// <summary>Mirror the image left-to-right.</summary>
    [Export]
    public bool Horizontal
    {
        get => _horizontal;
        set { _horizontal = value; EmitChanged(); }
    }

    /// <summary>Mirror the image top-to-bottom.</summary>
    [Export]
    public bool Vertical
    {
        get => _vertical;
        set { _vertical = value; EmitChanged(); }
    }

    private bool _horizontal = false;
    private bool _vertical   = false;

    public override Image Apply(Image image)
    {
        if (_horizontal) image.FlipX();
        if (_vertical)   image.FlipY();
        return image;
    }

    // ComputeOutputSize and ComputeOutputChannels use base defaults (pass-through).
}
