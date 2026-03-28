using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Fills out-of-bounds (transparent) pixels by clamping to the nearest edge pixel of the
/// scene content — like the "extend" or "clamp" mode in texture samplers.
///
/// This is useful when the scene has a consistent border colour and you want the background
/// to continue that colour naturally without a hard cut.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLExtendBackground : RLCameraBackground
{
    public override Image Apply(Image image)
    {
        var w = image.GetWidth();
        var h = image.GetHeight();

        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                if (image.GetPixel(x, y).A > 0f) continue;

                var sx = ClampToNearestOpaque(image, x, y, w, h);
                image.SetPixel(x, y, image.GetPixel(sx.X, sx.Y) with { A = 1f });
            }
        }

        return image;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the coordinate of the nearest opaque pixel by clamping the out-of-bounds
    /// pixel toward the image centre along each axis independently.
    /// </summary>
    private static Vector2I ClampToNearestOpaque(Image image, int x, int y, int w, int h)
    {
        // Walk inward from each edge until we find an opaque pixel on the same row/column.
        var cx = x;
        var cy = y;

        // Clamp x into the opaque column range.
        if (cx < 0) cx = 0;
        else if (cx >= w) cx = w - 1;

        // If that column pixel is still transparent, scan horizontally toward centre.
        if (image.GetPixel(cx, Mathf.Clamp(cy, 0, h - 1)).A == 0f)
        {
            var mid = w / 2;
            cx = cx < mid ? ScanRight(image, 0, Mathf.Clamp(cy, 0, h - 1), w)
                          : ScanLeft(image, w - 1, Mathf.Clamp(cy, 0, h - 1));
        }

        // Clamp y.
        if (cy < 0) cy = 0;
        else if (cy >= h) cy = h - 1;

        if (image.GetPixel(Mathf.Clamp(cx, 0, w - 1), cy).A == 0f)
        {
            var mid = h / 2;
            cy = cy < mid ? ScanDown(image, Mathf.Clamp(cx, 0, w - 1), 0, h)
                          : ScanUp(image, Mathf.Clamp(cx, 0, w - 1), h - 1);
        }

        return new Vector2I(Mathf.Clamp(cx, 0, w - 1), Mathf.Clamp(cy, 0, h - 1));
    }

    private static int ScanRight(Image img, int startX, int row, int w)
    {
        for (var x = startX; x < w; x++)
            if (img.GetPixel(x, row).A > 0f) return x;
        return 0;
    }

    private static int ScanLeft(Image img, int startX, int row)
    {
        for (var x = startX; x >= 0; x--)
            if (img.GetPixel(x, row).A > 0f) return x;
        return startX;
    }

    private static int ScanDown(Image img, int col, int startY, int h)
    {
        for (var y = startY; y < h; y++)
            if (img.GetPixel(col, y).A > 0f) return y;
        return 0;
    }

    private static int ScanUp(Image img, int col, int startY)
    {
        for (var y = startY; y >= 0; y--)
            if (img.GetPixel(col, y).A > 0f) return y;
        return startY;
    }
}
