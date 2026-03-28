using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Fills out-of-bounds (transparent) pixels by mirroring the scene content at each edge.
/// When the camera extends beyond the scene boundary, the pixels are reflected inward —
/// like placing a mirror at the edge.
///
/// Optional post-processing:
/// <list type="bullet">
///   <item><see cref="BlurRadius"/> — box-blur passes applied to filled pixels to soften the seam.</item>
///   <item><see cref="BlendWidth"/> — feather zone in pixels where mirrored content fades into the scene.</item>
/// </list>
/// </summary>
[GlobalClass]
[Tool]
public partial class RLMirrorBackground : RLCameraBackground
{
    /// <summary>
    /// Number of box-blur passes applied to the filled area after mirroring.
    /// 0 = no blur (hard mirror). Higher values produce a softer, more blended result.
    /// </summary>
    [Export(PropertyHint.Range, "0,8,1")]
    public int BlurRadius { get; set; } = 0;

    /// <summary>
    /// Width in pixels of the blend zone at the seam between real and mirrored content.
    /// Within this zone the mirrored pixel is alpha-blended toward the scene pixel.
    /// 0 = hard seam.
    /// </summary>
    [Export(PropertyHint.Range, "0,32,1")]
    public int BlendWidth { get; set; } = 0;

    public override Image Apply(Image image)
    {
        var w = image.GetWidth();
        var h = image.GetHeight();

        // First pass: fill transparent pixels by reflecting coordinates into valid content.
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                if (image.GetPixel(x, y).A > 0f) continue;

                var sx = MirrorCoord(x, w);
                var sy = MirrorCoord(y, h);
                image.SetPixel(x, y, image.GetPixel(sx, sy) with { A = 1f });
            }
        }

        // Optional blur over the whole image to soften the mirror seam.
        for (var pass = 0; pass < BlurRadius; pass++)
            image = BoxBlur(image, w, h);

        // Optional blend zone: re-read original alpha (all 1 now) so we feather by distance
        // from the nearest opaque edge pixel.
        if (BlendWidth > 0)
            ApplyBlendZone(image, w, h);

        return image;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>Reflects <paramref name="coord"/> back into [0, size) using mirror addressing.</summary>
    private static int MirrorCoord(int coord, int size)
    {
        if (size <= 1) return 0;
        coord = ((coord % (2 * size)) + 2 * size) % (2 * size);
        return coord < size ? coord : 2 * size - 1 - coord;
    }

    private static Image BoxBlur(Image src, int w, int h)
    {
        var dst = Image.CreateEmpty(w, h, false, Image.Format.Rgba8);

        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var r = 0f; var g = 0f; var b = 0f; var a = 0f;
                var count = 0;
                for (var ky = -1; ky <= 1; ky++)
                {
                    for (var kx = -1; kx <= 1; kx++)
                    {
                        var nx = Mathf.Clamp(x + kx, 0, w - 1);
                        var ny = Mathf.Clamp(y + ky, 0, h - 1);
                        var c  = src.GetPixel(nx, ny);
                        r += c.R; g += c.G; b += c.B; a += c.A;
                        count++;
                    }
                }
                dst.SetPixel(x, y, new Color(r / count, g / count, b / count, a / count));
            }
        }

        return dst;
    }

    private void ApplyBlendZone(Image image, int w, int h)
    {
        // Build a distance-from-edge map (distance to nearest originally-opaque pixel).
        // We approximate this by scanning inward from borders.
        // Pixels within BlendWidth of the border get their fill darkened toward transparent.
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var distToEdge = Mathf.Min(
                    Mathf.Min(x, w - 1 - x),
                    Mathf.Min(y, h - 1 - y));

                if (distToEdge >= BlendWidth) continue;

                var t   = distToEdge / (float)BlendWidth; // 0 at border, 1 at blend boundary
                var col = image.GetPixel(x, y);
                image.SetPixel(x, y, col with { A = t });
            }
        }
    }
}
