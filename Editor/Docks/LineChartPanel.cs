using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Editor;

/// <summary>
/// A self-contained line chart control for the RL Training Dashboard.
/// Supports multiple series, optional EMA smoothing overlay, gradient fill under curve,
/// automatic downsampling, and interactive scroll/zoom via mouse wheel.
/// Right-click opens a context menu for view reset and smoothing options.
/// </summary>
[Tool]
public partial class LineChartPanel : Control
{
    public readonly record struct SeriesEntry(string Label, Color LineColor, List<float> Points);

    private readonly List<SeriesEntry> _series = new();

    public string ChartTitle { get; set; } = "";
    public bool ShowSmoothed { get; set; } = true;
    public float SmoothAlpha { get; set; } = 0.08f;

    /// <summary>
    /// When > 0, clamps the x-axis extent to this many points so that a longer
    /// compare series does not widen the view beyond the primary run's length.
    /// Set this to the primary run's episode count before adding compare series.
    /// </summary>
    public int AnchorLength { get; set; } = 0;

    // ── Layout constants ────────────────────────────────────────────────────
    private const float TitleH = 24f;
    private const float LeftMargin = 54f;
    private const float BottomMargin = 22f;
    private const float RightMargin = 12f;
    private const float TopPad = 2f;
    private const int MaxDrawPoints = 600;
    private const int GridLines = 5;

    // ── View / scroll state ─────────────────────────────────────────────────
    private const int DefaultWindow = 2500;
    private int _viewWindow = DefaultWindow;   // number of data points in the visible window
    private int _viewOffset = 0;     // points scrolled back from the right edge (0 = live tail)
    private const int MinWindow = 20;

    // ── Colours ─────────────────────────────────────────────────────────────
    private static readonly Color CBg = new(0.12f, 0.12f, 0.12f);
    private static readonly Color CBorder = new(0.28f, 0.28f, 0.28f);
    private static readonly Color CPlotBg = new(0.085f, 0.085f, 0.085f);
    private static readonly Color CGrid = new(0.195f, 0.195f, 0.195f);
    private static readonly Color CAxisLabel = new(0.48f, 0.48f, 0.48f);
    private static readonly Color CTitle = new(0.88f, 0.88f, 0.88f);
    private static readonly Color CLegend = new(0.70f, 0.70f, 0.70f);
    private static readonly Color CNoData = new(0.38f, 0.38f, 0.38f);

    // ── Hover / crosshair state ──────────────────────────────────────────────
    private bool _mouseInside = false;

    // ── Context menu ─────────────────────────────────────────────────────────
    private PopupMenu _contextMenu = null!;
    private const int CmdResetView    = 0;
    private const int CmdResetZoom    = 1;
    private const int CmdToggleSmooth = 2;
    private const int CmdSmoothFast   = 3;
    private const int CmdSmoothMedium = 4;
    private const int CmdSmoothSlow   = 5;

    private static float Ui(float value) => EditorUiScale.Px(value);

    private static int Ui(int value) => EditorUiScale.Px(value);

    // ── Public API ──────────────────────────────────────────────────────────
    public void ClearSeries()
    {
        _series.Clear();
        _viewOffset = 0;
        QueueRedraw();
    }

    public void UpdateSeries(string label, Color color, IEnumerable<float> data)
    {
        var list = data.ToList();
        var idx = _series.FindIndex(s => s.Label == label);
        if (idx >= 0)
            _series[idx] = new SeriesEntry(label, color, list);
        else
            _series.Add(new SeriesEntry(label, color, list));

        // Keep offset clamped when new data arrives.
        ClampOffset();
        QueueRedraw();
    }

    public void RemoveSeries(string label)
    {
        var idx = _series.FindIndex(s => s.Label == label);
        if (idx >= 0) _series.RemoveAt(idx);
        QueueRedraw();
    }

    // ── Input / lifecycle ────────────────────────────────────────────────────
    public override void _Ready()
    {
        MouseFilter = MouseFilterEnum.Stop;
        MouseEntered += () => { _mouseInside = true;  QueueRedraw(); };
        MouseExited  += () => { _mouseInside = false; QueueRedraw(); };

        _contextMenu = new PopupMenu();
        AddChild(_contextMenu);
        _contextMenu.AddItem("Reset View",              CmdResetView);
        _contextMenu.AddItem("Reset Zoom",              CmdResetZoom);
        _contextMenu.AddSeparator();
        _contextMenu.AddCheckItem("Show Smoothing",     CmdToggleSmooth);
        _contextMenu.AddSeparator();
        _contextMenu.AddItem("Smooth: Fast   (α = 0.20)", CmdSmoothFast);
        _contextMenu.AddItem("Smooth: Medium (α = 0.08)", CmdSmoothMedium);
        _contextMenu.AddItem("Smooth: Slow   (α = 0.04)", CmdSmoothSlow);
        _contextMenu.IdPressed += OnContextMenuIdPressed;
    }

    public override void _GuiInput(InputEvent @event)
    {
        if (@event is InputEventMouseMotion)
        {
            QueueRedraw();
            return; // don't consume — just refresh the crosshair
        }

        if (@event is not InputEventMouseButton { Pressed: true } btn) return;

        var ctrl = Input.IsKeyPressed(Key.Ctrl);

        switch (btn.ButtonIndex)
        {
            case MouseButton.Right:
                OpenContextMenu(btn.GlobalPosition);
                AcceptEvent();
                break;
            case MouseButton.WheelUp when ctrl:
                ZoomIn();
                AcceptEvent();
                break;
            case MouseButton.WheelDown when ctrl:
                ZoomOut();
                AcceptEvent();
                break;
            case MouseButton.WheelUp:
                PanLeft();
                AcceptEvent();
                break;
            case MouseButton.WheelDown:
                PanRight();
                AcceptEvent();
                break;
        }
    }

    private void OpenContextMenu(Vector2 globalPos)
    {
        _contextMenu.SetItemChecked(_contextMenu.GetItemIndex(CmdToggleSmooth), ShowSmoothed);
        // Convert canvas coords → OS screen coords so the popup appears at the cursor.
        var screenPos = (Vector2I)(GetViewport().GetScreenTransform() * globalPos)
                      + GetWindow().Position;
        _contextMenu.Position = screenPos;
        _contextMenu.ResetSize();
        _contextMenu.Popup();
    }

    private void OnContextMenuIdPressed(long id)
    {
        switch ((int)id)
        {
            case CmdResetView:
                _viewOffset = 0;
                QueueRedraw();
                break;
            case CmdResetZoom:
                _viewWindow = AnchorLength > 0 ? AnchorLength : DefaultWindow;
                ClampOffset();
                QueueRedraw();
                break;
            case CmdToggleSmooth:
                ShowSmoothed = !ShowSmoothed;
                QueueRedraw();
                break;
            case CmdSmoothFast:
                SmoothAlpha = 0.20f;
                ShowSmoothed = true;
                QueueRedraw();
                break;
            case CmdSmoothMedium:
                SmoothAlpha = 0.08f;
                ShowSmoothed = true;
                QueueRedraw();
                break;
            case CmdSmoothSlow:
                SmoothAlpha = 0.04f;
                ShowSmoothed = true;
                QueueRedraw();
                break;
        }
    }

    private void PanLeft()
    {
        var step = Math.Max(1, _viewWindow / 10);
        _viewOffset += step;
        ClampOffset();
        QueueRedraw();
    }

    private void PanRight()
    {
        var step = Math.Max(1, _viewWindow / 10);
        _viewOffset = Math.Max(0, _viewOffset - step);
        QueueRedraw();
    }

    private void ZoomIn()
    {
        _viewWindow = Math.Max(MinWindow, (int)(_viewWindow * 0.8f));
        ClampOffset();
        QueueRedraw();
    }

    private void ZoomOut()
    {
        _viewWindow = (int)(_viewWindow * 1.25f);
        ClampOffset();
        QueueRedraw();
    }

    private void ClampOffset()
    {
        var maxPoints = _series.Count > 0 ? _series.Max(s => s.Points.Count) : 0;
        var window = Math.Min(_viewWindow, maxPoints);
        _viewOffset = Math.Max(0, Math.Min(_viewOffset, maxPoints - window));
    }

    /// <summary>
    /// Computes the shared global view range in episode-index space.
    /// All series are aligned by episode 0 on the left; the longest series defines the x-axis extent.
    /// </summary>
    /// <summary>
    /// Snaps the zoom window to AnchorLength (primary run) and resets the pan offset.
    /// Call this once when a compare run is selected so the user starts with a clean view
    /// of the primary run; they can then zoom out further to reveal the compare run.
    /// </summary>
    public void ResetZoomToAnchor()
    {
        _viewWindow = AnchorLength > 0 ? AnchorLength : DefaultWindow;
        _viewOffset = 0;
        ClampOffset();
        QueueRedraw();
    }

    private (int globalStart, int globalEnd, int window) ComputeGlobalView()
    {
        // Always use the true maximum so a longer compare series is reachable by zooming out.
        var maxPoints = _series.Count > 0 ? _series.Max(s => s.Points.Count) : 0;
        if (maxPoints == 0) return (0, 0, 0);
        var window = Math.Min(_viewWindow, maxPoints);
        var globalEnd = maxPoints - Math.Max(0, Math.Min(_viewOffset, maxPoints - window));
        var globalStart = Math.Max(0, globalEnd - window);
        return (globalStart, globalEnd, window);
    }

    // ── Drawing ─────────────────────────────────────────────────────────────
    public override void _Draw()
    {
        try
        {
            var size = Size;
            if (size.X < 12 || size.Y < 12) return;

            var titleH = Ui(TitleH);
            var leftMargin = Ui(LeftMargin);
            var bottomMargin = Ui(BottomMargin);
            var rightMargin = Ui(RightMargin);
            var topPad = Ui(TopPad);
            var borderWidth = Math.Max(1f, Ui(1f));
            var gridWidth = Math.Max(1f, Ui(1f));
            var rawWidth = ShowSmoothed ? Ui(0.9f) : Ui(1.7f);
            var smoothedWidth = Ui(2.6f);
            var legendSwatchW = Ui(14f);
            var legendSwatchH = Ui(4f);
            var legendGap = Ui(18f);
            var legendAdvance = Ui(36f);
            var axisHintWidth = Ui(48);
            var currentValueWidth = Ui(60);
            var tooltipPad = Ui(5f);
            var tooltipOffset = Ui(10f);
            var tooltipLift = Ui(6f);
            var tooltipDrop = Ui(8f);
            var crosshairRadius = Ui(3.5f);

            DrawRect(new Rect2(Vector2.Zero, size), CBg, filled: true);
            DrawRect(new Rect2(Vector2.Zero, size), CBorder, filled: false, width: borderWidth);

            var font = GetThemeDefaultFont();
            var fs = Mathf.Clamp((int)GetThemeDefaultFontSize(), Ui(8), Ui(15));

            DrawString(font, new Vector2(leftMargin, titleH - Ui(7f)), ChartTitle,
                HorizontalAlignment.Left, -1, fs, CTitle);

            var plot = new Rect2(
                leftMargin,
                titleH + topPad,
                size.X - leftMargin - rightMargin,
                size.Y - titleH - topPad - bottomMargin);

            if (plot.Size.X < 4 || plot.Size.Y < 4) return;

            DrawRect(plot, CPlotBg, filled: true);

            // ── Global view range (shared across all series for correct episode alignment) ──
            var (globalStart, globalEnd, viewWindow) = ComputeGlobalView();

            float gMin = float.MaxValue, gMax = float.MinValue;
            foreach (var s in _series)
            {
                if (s.Points.Count == 0) continue;
                var (slice, _) = GetViewSlice(s.Points, globalStart, globalEnd);
                if (slice.Count == 0) continue;
                gMin = Math.Min(gMin, slice.Min());
                gMax = Math.Max(gMax, slice.Max());
            }

            if (gMin == float.MaxValue)
            {
                var cx = plot.Position.X + plot.Size.X * 0.5f;
                var cy = plot.Position.Y + plot.Size.Y * 0.5f;
                DrawString(font, new Vector2(cx - Ui(32f), cy + fs * 0.4f), "No data yet",
                    HorizontalAlignment.Left, -1, fs - 1, CNoData);
                return;
            }

            if (Math.Abs(gMax - gMin) < 1e-6f) { gMin -= 0.5f; gMax += 0.5f; }
            float range = gMax - gMin;
            gMin -= range * 0.06f;
            gMax += range * 0.06f;
            range = gMax - gMin;

            for (int gi = 0; gi <= GridLines; gi++)
            {
                float t = (float)gi / GridLines;
                float gy = plot.Position.Y + plot.Size.Y * (1f - t);
                DrawLine(new Vector2(plot.Position.X, gy),
                         new Vector2(plot.Position.X + plot.Size.X, gy),
                         CGrid, gridWidth);
                float axVal = gMin + range * t;
                DrawString(font, new Vector2(Ui(2f), gy + fs * 0.38f),
                    FormatAxisValue(axVal),
                    HorizontalAlignment.Left, leftMargin - Ui(6f), fs - 3, CAxisLabel);
            }

            float rawAlpha = ShowSmoothed ? 0.18f : 1.0f;

            foreach (var s in _series)
            {
                if (s.Points.Count < 2) continue;
                var (slice, startIdx) = GetViewSlice(s.Points, globalStart, globalEnd);
                if (slice.Count < 2) continue;
                var pts = BuildPoints(plot, Downsample(slice, MaxDrawPoints), gMin, range, startIdx, globalStart, viewWindow, slice.Count);
                DrawFill(plot, pts, s.LineColor, ShowSmoothed ? 0.08f : 0.20f);
                var lineColor = new Color(s.LineColor.R, s.LineColor.G, s.LineColor.B, rawAlpha);
                DrawPolyline(pts, lineColor, rawWidth, antialiased: true);
            }

            if (ShowSmoothed)
            {
                foreach (var s in _series)
                {
                    if (s.Points.Count < 12) continue;
                    var (slice, startIdx) = GetViewSlice(s.Points, globalStart, globalEnd);
                    if (slice.Count < 12) continue;
                    var smoothed = Ema(Downsample(slice, MaxDrawPoints), SmoothAlpha);
                    var smPts = BuildPoints(plot, smoothed, gMin, range, startIdx, globalStart, viewWindow, slice.Count);
                    DrawPolyline(smPts, GetSmoothedColor(s.LineColor), smoothedWidth, antialiased: true);
                }
            }

            int totalPts = _series.Count > 0 ? _series.Max(s => s.Points.Count) : 0;
            if (globalEnd > globalStart + 1)
            {
                float labelY = plot.Position.Y + plot.Size.Y + bottomMargin - Ui(6f);
                DrawString(font, new Vector2(plot.Position.X, labelY),
                    (globalStart + 1).ToString(), HorizontalAlignment.Left, Ui(40), fs - 3, CAxisLabel);
                DrawString(font, new Vector2(plot.Position.X + plot.Size.X - axisHintWidth, labelY),
                    globalEnd.ToString(), HorizontalAlignment.Left, axisHintWidth, fs - 3, CAxisLabel);

                if (totalPts > _viewWindow || _viewOffset > 0)
                {
                    var hint = _viewOffset == 0
                        ? $"[{_viewWindow} pts  Ctrl+scroll=zoom]"
                        : $"[scroll to pan  Ctrl+scroll=zoom]";
                    DrawString(font, new Vector2(plot.Position.X + plot.Size.X * 0.5f - Ui(60f), labelY),
                        hint, HorizontalAlignment.Left, -1, fs - 4, new Color(0.38f, 0.38f, 0.38f));
                }
            }

            float lx = plot.Position.X + Ui(6f);
            float ly = plot.Position.Y + Ui(6f);
            foreach (var s in _series)
            {
                DrawRect(new Rect2(lx, ly + Ui(1f), legendSwatchW, legendSwatchH), s.LineColor, filled: true);
                DrawString(font, new Vector2(lx + legendGap, ly + fs * 0.68f), s.Label,
                    HorizontalAlignment.Left, -1, fs - 3, CLegend);
                lx += font.GetStringSize(s.Label, HorizontalAlignment.Left, -1, fs - 3).X + legendAdvance;
            }

            if (_series.Count > 0 && _series[0].Points.Count > 0)
            {
                var cur = FormatAxisValue(_series[0].Points[^1]);
                DrawString(font, new Vector2(size.X - rightMargin - Ui(58f), titleH - Ui(7f)),
                    cur, HorizontalAlignment.Right, currentValueWidth, fs, _series[0].LineColor);
            }

            if (_mouseInside)
            {
                var mp = GetLocalMousePosition();
                if (plot.HasPoint(mp) && _series.Count > 0)
                {
                    float tx = (mp.X - plot.Position.X) / plot.Size.X;

                    // Compute the global episode index under the cursor.
                    int snapEpisode = globalStart + (int)Math.Round(tx * Math.Max(viewWindow - 1, 1));
                    snapEpisode = Math.Clamp(snapEpisode, globalStart, globalEnd - 1);
                    float snapX = plot.Position.X + (float)(snapEpisode - globalStart) / Math.Max(viewWindow - 1, 1) * plot.Size.X;

                    var snapPoints = new List<(string Label, Color LineColor, float Val, float SnapY)>();

                    foreach (var s in _series)
                    {
                        if (s.Points.Count < 2) continue;
                        var (slice, startIdx) = GetViewSlice(s.Points, globalStart, globalEnd);
                        if (slice.Count < 2) continue;

                        // Map global episode to an index within this series' slice.
                        int sliceIdx = snapEpisode - startIdx;
                        if (sliceIdx < 0 || sliceIdx >= slice.Count) continue;

                        var ds = Downsample(slice, MaxDrawPoints);
                        var display = ShowSmoothed && ds.Count >= 12 ? Ema(ds, SmoothAlpha) : ds;

                        int dsIdx = Math.Clamp(
                            (int)Math.Round((float)sliceIdx / Math.Max(slice.Count - 1, 1) * (display.Count - 1)),
                            0, display.Count - 1);

                        float val = display[dsIdx];
                        float ty = Math.Clamp((val - gMin) / range, 0f, 1f);
                        float sy = plot.Position.Y + plot.Size.Y * (1f - ty);

                        snapPoints.Add((s.Label, s.LineColor, val, sy));
                    }

                    if (snapPoints.Count > 0)
                    {
                        float primaryY = snapPoints[0].SnapY;
                        var crossColor = new Color(0.70f, 0.70f, 0.70f, 0.28f);
                        DrawLine(new Vector2(plot.Position.X, primaryY),
                                 new Vector2(plot.Position.X + plot.Size.X, primaryY), crossColor, gridWidth);
                        DrawLine(new Vector2(snapX, plot.Position.Y),
                                 new Vector2(snapX, plot.Position.Y + plot.Size.Y), crossColor, gridWidth);

                        foreach (var sp in snapPoints)
                            DrawCircle(new Vector2(snapX, sp.SnapY), crosshairRadius,
                                new Color(sp.LineColor.R, sp.LineColor.G, sp.LineColor.B, 0.9f));

                        var lines = new List<(string Text, Color Col)>
                        {
                            ($"ep {snapEpisode + 1}", new Color(0.65f, 0.65f, 0.65f))
                        };
                        foreach (var sp in snapPoints)
                            lines.Add(($"{sp.Label}: {FormatAxisValue(sp.Val)}", sp.LineColor));

                        float lineH = fs + 3f;
                        float maxW = 0f;
                        foreach (var (text, _) in lines)
                            maxW = Math.Max(maxW, font.GetStringSize(text, HorizontalAlignment.Left, -1, fs - 2).X);
                        float bw = maxW + tooltipPad * 2f;
                        float bh = lines.Count * lineH + tooltipPad * 2f;

                        float bx = snapX + tooltipOffset;
                        float by = primaryY - bh - tooltipLift;
                        if (bx + bw > plot.Position.X + plot.Size.X) bx = snapX - bw - tooltipOffset;
                        if (by < plot.Position.Y) by = primaryY + tooltipDrop;

                        DrawRect(new Rect2(bx, by, bw, bh),
                            new Color(0.10f, 0.10f, 0.10f, 0.92f), filled: true);
                        DrawRect(new Rect2(bx, by, bw, bh),
                            new Color(0.40f, 0.40f, 0.40f, 0.75f), filled: false, width: borderWidth);
                        for (int li = 0; li < lines.Count; li++)
                            DrawString(font, new Vector2(bx + tooltipPad, by + tooltipPad + (li + 0.82f) * lineH),
                                lines[li].Text, HorizontalAlignment.Left, -1, fs - 2, lines[li].Col);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            GD.PushError($"[LineChartPanel] Draw failed for '{ChartTitle}': {ex.Message}");
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the slice of this series that falls within [globalStart, globalEnd),
    /// preserving episode-index alignment across series of different lengths.
    /// </summary>
    private static (List<float> slice, int startIdx) GetViewSlice(List<float> points, int globalStart, int globalEnd)
    {
        if (points.Count == 0) return (new List<float>(), 0);
        var start = Math.Min(globalStart, points.Count);
        var end   = Math.Min(globalEnd,   points.Count);
        if (end <= start) return (new List<float>(), start);
        return (points.GetRange(start, end - start), start);
    }

    /// <summary>
    /// Maps data points to screen coordinates. data may be downsampled, so originalSliceCount
    /// gives the true episode span the data covers. seriesStart/globalStart/viewWindow place
    /// that span correctly in the shared x-axis window.
    /// </summary>
    private static Vector2[] BuildPoints(Rect2 area, List<float> data, float min, float range,
        int seriesStart, int globalStart, int viewWindow, int originalSliceCount)
    {
        var pts = new Vector2[data.Count];
        float denom = Math.Max(viewWindow - 1, 1);
        for (int i = 0; i < data.Count; i++)
        {
            // Map downsampled index back to episode offset within the original slice.
            float episodeOffset = data.Count <= 1 ? 0f
                : (float)i / (data.Count - 1) * (originalSliceCount - 1);
            float tx = (seriesStart - globalStart + episodeOffset) / denom;
            float ty = Math.Clamp((data[i] - min) / range, 0f, 1f);
            pts[i] = new Vector2(
                area.Position.X + tx * area.Size.X,
                area.Position.Y + area.Size.Y * (1f - ty));
        }
        return pts;
    }

    /// <summary>
    /// Draws a gradient fill under the curve using per-segment trapezoid quads.
    /// Each segment is a convex quad so there are no polygon self-intersection artifacts.
    /// The fill fades from semi-opaque at the data line to transparent at the baseline.
    /// </summary>
    private void DrawFill(Rect2 area, Vector2[] pts, Color color, float topAlpha)
    {
        if (pts.Length < 2) return;
        float baseY = area.Position.Y + area.Size.Y;
        var topColor = new Color(color.R, color.G, color.B, topAlpha);
        var botColor = new Color(color.R, color.G, color.B, 0.00f);

        for (int i = 0; i < pts.Length - 1; i++)
        {
            DrawPolygon(
                new[] { pts[i], pts[i + 1], new Vector2(pts[i + 1].X, baseY), new Vector2(pts[i].X, baseY) },
                new[] { topColor, topColor, botColor, botColor }
            );
        }
    }

    private static Color GetSmoothedColor(Color baseColor)
    {
        const float lift = 0.38f;
        return new Color(
            baseColor.R + (1f - baseColor.R) * lift,
            baseColor.G + (1f - baseColor.G) * lift,
            baseColor.B + (1f - baseColor.B) * lift,
            0.96f);
    }

    private static List<float> Ema(List<float> data, float alpha)
    {
        var result = new List<float>(data.Count);
        float v = data[0];
        foreach (var d in data)
        {
            v = alpha * d + (1f - alpha) * v;
            result.Add(v);
        }
        return result;
    }

    private static List<float> Downsample(List<float> data, int max)
    {
        if (data.Count <= max) return data;
        var result = new List<float>(max);
        float step = (float)(data.Count - 1) / (max - 1);
        for (int i = 0; i < max; i++)
            result.Add(data[(int)Math.Round(i * step)]);
        return result;
    }

    private static string FormatAxisValue(float v)
    {
        float abs = Math.Abs(v);
        if (abs >= 10000f) return v.ToString("F0");
        if (abs >= 100f)   return v.ToString("F1");
        if (abs >= 1f)     return v.ToString("F2");
        return v.ToString("F3");
    }
}
