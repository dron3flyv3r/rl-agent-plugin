using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Fills out-of-bounds (transparent) pixels with a single solid <see cref="FillColor"/>.
/// This is the simplest background strategy — use it when you want a neutral,
/// consistent colour outside the scene boundary (e.g. black, grey, or a chroma-key colour).
/// </summary>
[GlobalClass]
[Tool]
public partial class RLSolidBackground : RLCameraBackground
{
    /// <summary>The colour used to fill any pixel with alpha == 0.</summary>
    [Export]
    public Color FillColor { get; set; } = Colors.Black;

    public override Image Apply(Image image)
    {
        var width  = image.GetWidth();
        var height = image.GetHeight();
        var fill   = FillColor;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                if (image.GetPixel(x, y).A == 0f)
                    image.SetPixel(x, y, fill);
            }
        }

        return image;
    }
}
