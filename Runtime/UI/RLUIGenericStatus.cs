using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Drop-in CanvasLayer overlay that shows live NEAT training stats and a
/// visualisation of the all-time champion genome.
///
/// Usage:
///   1. Add RLUIGenericStatus as a child of any node in your game scene.
///   2. Tune StatsWidth/Height, NetworkWidth/Height, and Colors in the Inspector.
///   3. Move the panel by adjusting the CanvasLayer's built-in Offset property.
///
/// The node auto-discovers the TrainingBootstrap by walking up the scene tree.
/// During normal play (no bootstrap) it shows "Waiting for training…".
/// In the Editor it displays a static preview so you can size and position it.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLUIGenericStatus : CanvasLayer
{
    // ── Shared layout constants (read by StatusDrawControl) ───────────────────

    internal const float HeaderH  = 36f;
    internal const float InnerPad = 12f;
    internal const float DividerW = 1f;

    // ── Exports: Layout ───────────────────────────────────────────────────────

    private float _statsWidth    = 200f;
    private float _statsHeight   = 280f;
    private float _networkWidth  = 320f;
    private float _networkHeight = 280f;

    [ExportGroup("Layout")]

    [Export]
    public float StatsWidth
    {
        get => _statsWidth;
        set { _statsWidth = value; _canvas?.UpdateSize(); }
    }

    [Export]
    public float StatsHeight
    {
        get => _statsHeight;
        set { _statsHeight = value; _canvas?.UpdateSize(); }
    }

    [Export]
    public float NetworkWidth
    {
        get => _networkWidth;
        set { _networkWidth = value; _canvas?.UpdateSize(); }
    }

    [Export]
    public float NetworkHeight
    {
        get => _networkHeight;
        set { _networkHeight = value; _canvas?.UpdateSize(); }
    }

    // ── Exports: Colors ───────────────────────────────────────────────────────

    [ExportGroup("Colors")]
    [Export] public Color AccentColor     { get; set; } = new(0.961f, 0.620f, 0.043f);        // #F59E0B amber
    [Export] public Color MeanColor       { get; set; } = new(0.984f, 0.443f, 0.522f);        // #FB7185 rose
    [Export] public Color BackgroundColor { get; set; } = new(0.110f, 0.078f, 0.063f, 0.90f); // #1C1410
    [Export] public Color HeaderColor     { get; set; } = new(0.145f, 0.110f, 0.078f, 1.00f); // #251C14
    [Export] public Color TextPrimary     { get; set; } = new(0.996f, 0.953f, 0.773f);        // #FEF3C7
    [Export] public Color TextSecondary   { get; set; } = new(0.659f, 0.565f, 0.439f);        // #A89070
    [Export] public Color InputNodeColor  { get; set; } = new(0.376f, 0.647f, 0.980f);        // #60A5FA
    [Export] public Color HiddenNodeColor { get; set; } = new(0.471f, 0.443f, 0.424f);        // #78716C

    // ── Internal live state (read by StatusDrawControl) ───────────────────────

    internal NeatGenomeSnapshot?            LatestSnapshot;
    internal readonly List<(float B, float M)> History = new(52);

    // ── Private ───────────────────────────────────────────────────────────────

    private StatusDrawControl? _canvas;
    private INeatDataFeed?     _feed;
    private int                _lastGeneration = -1;

    // ── Godot lifecycle ───────────────────────────────────────────────────────

    public override void _Ready()
    {
        Layer = 100;

        // On hot-reload Godot restores the child via the parameterless constructor
        // (_o = null). Reuse it and wire the owner; otherwise create fresh.
        _canvas = null;
        foreach (var child in GetChildren())
        {
            if (child is StatusDrawControl existing) { _canvas = existing; break; }
        }

        if (_canvas == null)
        {
            _canvas = new StatusDrawControl(this);
            _canvas.MouseFilter = Control.MouseFilterEnum.Ignore;
            AddChild(_canvas);
        }
        else
        {
            _canvas.SetOwner(this);
        }

        _canvas.UpdateSize();

        if (!Engine.IsEditorHint())
            Callable.From(FindFeedProvider).CallDeferred();
    }

    public override void _Process(double delta)
    {
        if (Engine.IsEditorHint())
        {
            _canvas?.QueueRedraw();
            return;
        }

        if (_feed == null) return;

        var snap = _feed.GetSnapshot();
        LatestSnapshot = snap;

        // Append a history entry once per generation
        if (snap.Generation != _lastGeneration)
        {
            _lastGeneration = snap.Generation;
            if (History.Count >= 50) History.RemoveAt(0);
            History.Add((snap.BestFitness, snap.MeanFitness));
        }

        _canvas?.QueueRedraw();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private void FindFeedProvider()
    {
        var current = (Node?)GetParent();
        while (current != null)
        {
            if (current is TrainingBootstrap bootstrap)
            {
                _feed = bootstrap.GetNeatDataFeed();
                return;
            }
            current = current.GetParent();
        }
    }
}

// ── Draw control ──────────────────────────────────────────────────────────────

/// <summary>Single Control child that renders everything via Godot's draw API.</summary>
internal sealed partial class StatusDrawControl : Control
{
    // Layout
    private const float AccentStripeW = 4f;
    private const float SectionGap   = 10f;  // horizontal breathing room around the divider
    private const float NodeR         = 7f;
    private const float SparkH        = 58f;
    private const float RowH          = 21f;
    private const int   FontSz        = 13;
    private const int   FontSzSm      = 11;
    private const int   CornerR       = 12;

    // null! — set via constructor or SetOwner(); drawing methods are only reached
    // from _Draw which guards _o != null, so no dereferencing of null occurs.
    private RLUIGenericStatus _o = null!;

    // Cached StyleBoxFlats — BgColor refreshed per _Draw call, never reallocated
    private readonly StyleBoxFlat _panelStyle  = MakeRoundBox(CornerR);
    private readonly StyleBoxFlat _headerStyle = MakeRoundBox(CornerR, topOnly: true);

    /// <summary>Required by Godot's source generator.</summary>
    public StatusDrawControl() { }

    public StatusDrawControl(RLUIGenericStatus owner) { _o = owner; }

    /// <summary>
    /// Called by RLUIGenericStatus._Ready to wire up the owner after Godot
    /// restores this node from scene state using the parameterless constructor.
    /// </summary>
    internal void SetOwner(RLUIGenericStatus owner) => _o = owner;

    internal void UpdateSize()
    {
        if (_o == null) return;
        float bodyH = Math.Max(_o.StatsHeight, _o.NetworkHeight);
        Size = new Vector2(
            _o.StatsWidth + SectionGap * 2 + RLUIGenericStatus.DividerW + _o.NetworkWidth,
            RLUIGenericStatus.HeaderH + bodyH);
    }

    // ── _Draw entry ───────────────────────────────────────────────────────────

    public override void _Draw()
    {
        if (_o is null) return;  // parameterless instance before SetOwner — skip
        if (Engine.IsEditorHint())
        {
            DrawAll(BuildPreviewSnapshot(), BuildPreviewHistory(), preview: true);
            return;
        }

        if (_o.LatestSnapshot != null)
            DrawAll(_o.LatestSnapshot, _o.History, preview: false);
        else
            DrawWaiting();
    }

    // ── Top-level draw ────────────────────────────────────────────────────────

    private void DrawAll(NeatGenomeSnapshot snap, List<(float B, float M)> history, bool preview)
    {
        float sw = _o.StatsWidth;
        float nw = _o.NetworkWidth;
        float sh = _o.StatsHeight;
        float nh = _o.NetworkHeight;
        float dw = RLUIGenericStatus.DividerW;
        float hh = RLUIGenericStatus.HeaderH;

        // ── Backgrounds
        _panelStyle.BgColor = _o.BackgroundColor;
        DrawStyleBox(_panelStyle, new Rect2(Vector2.Zero, Size));

        _headerStyle.BgColor = _o.HeaderColor;
        DrawStyleBox(_headerStyle, new Rect2(0, 0, Size.X, hh));

        // ── Accent stripe
        DrawRect(new Rect2(0, 0, AccentStripeW, hh), _o.AccentColor);

        // ── Header text
        string tag   = preview ? "  [PREVIEW]" : "";
        string title = $"NEAT  ·  Gen {snap.Generation}  ·  Best {snap.BestFitness:F2}  ·  Alive {snap.AliveCount}/{snap.PopulationSize}{tag}";
        var font = ThemeDB.FallbackFont;
        DrawString(font,
            new Vector2(AccentStripeW + 10f, hh * 0.5f + FontSz * 0.36f),
            title, HorizontalAlignment.Left, -1, FontSz, _o.TextPrimary);

        // ── Section divider (centred in the gap)
        float divX = sw + SectionGap;
        var divColor = new Color(
            _o.BackgroundColor.R + 0.07f,
            _o.BackgroundColor.G + 0.05f,
            _o.BackgroundColor.B + 0.04f, 1f);
        DrawRect(new Rect2(divX, hh, dw, Size.Y - hh), divColor);

        // ── Sections
        DrawStatsSection(new Rect2(0,               hh, sw, sh), snap, history);
        DrawNetworkSection(new Rect2(divX + dw + SectionGap, hh, nw, nh), snap);
    }

    private void DrawWaiting()
    {
        _panelStyle.BgColor = _o.BackgroundColor;
        DrawStyleBox(_panelStyle, new Rect2(Vector2.Zero, Size));

        _headerStyle.BgColor = _o.HeaderColor;
        DrawStyleBox(_headerStyle, new Rect2(0, 0, Size.X, RLUIGenericStatus.HeaderH));

        DrawRect(new Rect2(0, 0, AccentStripeW, RLUIGenericStatus.HeaderH), _o.AccentColor);

        DrawString(ThemeDB.FallbackFont,
            new Vector2(AccentStripeW + 10f, RLUIGenericStatus.HeaderH * 0.5f + FontSz * 0.36f),
            "NEAT  ·  Waiting for training to start…",
            HorizontalAlignment.Left, -1, FontSz, _o.TextSecondary);
    }

    // ── Stats section ─────────────────────────────────────────────────────────

    private void DrawStatsSection(Rect2 rect, NeatGenomeSnapshot snap, List<(float B, float M)> history)
    {
        var font = ThemeDB.FallbackFont;
        float ip = RLUIGenericStatus.InnerPad;
        float x  = rect.Position.X + ip;
        float y  = rect.Position.Y + ip;

        // Section label
        DrawString(font, new Vector2(x, y + FontSzSm),
            "STATS", HorizontalAlignment.Left, -1, FontSzSm, _o.TextSecondary);
        y += FontSzSm + 10f;

        // Stat rows
        float rightX = rect.Position.X + rect.Size.X - ip;
        void Row(string label, string value, Color valColor)
        {
            DrawString(font, new Vector2(x,      y + FontSz), label,
                HorizontalAlignment.Left,  -1, FontSz, _o.TextSecondary);
            DrawString(font, new Vector2(rightX, y + FontSz), value,
                HorizontalAlignment.Right, -1, FontSz, valColor);
            y += RowH;
        }

        Row("Best fitness",  $"{snap.BestFitness:F2}",                           _o.AccentColor);
        Row("Mean fitness",  $"{snap.MeanFitness:F2}",                           _o.MeanColor);
        Row("Alive",        $"{snap.AliveCount} / {snap.PopulationSize}",   _o.TextPrimary);
        Row("Species",      $"{snap.SpeciesCount}",                              _o.TextPrimary);
        Row("Generation",   $"{snap.Generation}",                                _o.TextPrimary);

        // Sparkline area starts from a fixed offset from the bottom of the section
        float sparkBottom = rect.Position.Y + rect.Size.Y - ip;
        float sparkTop    = sparkBottom - SparkH;

        // Separator
        var sepColor = new Color(_o.TextSecondary.R, _o.TextSecondary.G, _o.TextSecondary.B, 0.25f);
        DrawLine(new Vector2(x, sparkTop - 10f), new Vector2(rightX, sparkTop - 10f), sepColor, 1f, true);

        // Legend
        float legY = sparkTop - 3f;
        DrawCircle(new Vector2(x + 5f,  legY), 3.5f, _o.AccentColor);
        DrawString(font, new Vector2(x + 13f, legY + FontSzSm * 0.38f),
            "Best", HorizontalAlignment.Left, -1, FontSzSm, _o.TextSecondary);
        DrawCircle(new Vector2(x + 52f, legY), 3.5f, _o.MeanColor);
        DrawString(font, new Vector2(x + 60f, legY + FontSzSm * 0.38f),
            "Mean", HorizontalAlignment.Left, -1, FontSzSm, _o.TextSecondary);

        // Sparkline chart
        DrawSparkline(new Rect2(x, sparkTop + 4f, rightX - x, sparkBottom - sparkTop - 4f), history);
    }

    private void DrawSparkline(Rect2 rect, List<(float B, float M)> history)
    {
        if (history.Count < 2) return;

        float minV = float.MaxValue, maxV = float.MinValue;
        foreach (var (b, m) in history)
        {
            if (b < minV) minV = b; if (b > maxV) maxV = b;
            if (m < minV) minV = m; if (m > maxV) maxV = m;
        }
        float range = maxV - minV;
        if (range < 0.001f) range = 1f;

        int n = history.Count;
        var bestPts = new Vector2[n];
        var meanPts = new Vector2[n];
        var fillPts = new Vector2[n + 2];

        for (int i = 0; i < n; i++)
        {
            float px = rect.Position.X + i / (float)(n - 1) * rect.Size.X;
            bestPts[i] = new Vector2(px, rect.Position.Y + rect.Size.Y
                - (history[i].B - minV) / range * rect.Size.Y);
            meanPts[i] = new Vector2(px, rect.Position.Y + rect.Size.Y
                - (history[i].M - minV) / range * rect.Size.Y);
        }

        // Filled area under best curve
        Array.Copy(bestPts, fillPts, n);
        fillPts[n]     = new Vector2(rect.Position.X + rect.Size.X, rect.Position.Y + rect.Size.Y);
        fillPts[n + 1] = new Vector2(rect.Position.X,               rect.Position.Y + rect.Size.Y);
        var fillColors = new Color[] { new(_o.AccentColor.R, _o.AccentColor.G, _o.AccentColor.B, 0.13f) };
        DrawPolygon(fillPts, fillColors);

        // Lines
        DrawPolyline(bestPts, _o.AccentColor, 2f, true);
        DrawPolyline(meanPts, new Color(_o.MeanColor.R, _o.MeanColor.G, _o.MeanColor.B, 0.80f), 1.5f, true);
    }

    // ── Network section ───────────────────────────────────────────────────────

    private void DrawNetworkSection(Rect2 rect, NeatGenomeSnapshot snap)
    {
        var font = ThemeDB.FallbackFont;
        float ip = RLUIGenericStatus.InnerPad;
        float x  = rect.Position.X + ip;
        float y  = rect.Position.Y + ip;

        // Section label
        DrawString(font, new Vector2(x, y + FontSzSm),
            "BEST NETWORK", HorizontalAlignment.Left, -1, FontSzSm, _o.TextSecondary);

        // Legend strip at bottom
        float legendH  = FontSzSm + 14f;
        float legendY  = rect.Position.Y + rect.Size.Y - ip;
        float colW     = rect.Size.X / 3f;

        void LegNode(float lx, Color c, string label)
        {
            DrawCircle(new Vector2(lx + 5f, legendY), 4f, c);
            DrawString(font, new Vector2(lx + 14f, legendY + FontSzSm * 0.38f),
                label, HorizontalAlignment.Left, -1, FontSzSm, _o.TextSecondary);
        }
        LegNode(rect.Position.X + ip,            _o.InputNodeColor,  "inputs");
        LegNode(rect.Position.X + ip + colW,     _o.HiddenNodeColor, "hidden");
        LegNode(rect.Position.X + ip + colW * 2, _o.AccentColor,     "outputs");

        // Network draw area
        var netArea = new Rect2(
            x,
            y + FontSzSm + 8f,
            rect.Size.X - ip * 2f,
            rect.Size.Y - ip * 2f - FontSzSm - 8f - legendH);

        var nodePos = ComputeLayout(snap, netArea);

        // Max weight for normalisation
        float maxW = 0f;
        foreach (var c in snap.Connections)
            if (c.Enabled && MathF.Abs(c.Weight) > maxW) maxW = MathF.Abs(c.Weight);
        if (maxW < 0.001f) maxW = 1f;

        // Draw connections — weakest first so strong ones render on top
        var enabledConns = snap.Connections
            .Where(c => c.Enabled)
            .OrderBy(c => MathF.Abs(c.Weight));

        foreach (var conn in enabledConns)
        {
            if (!nodePos.TryGetValue(conn.InNode,  out var from)) continue;
            if (!nodePos.TryGetValue(conn.OutNode, out var to))   continue;

            float t        = MathF.Abs(conn.Weight) / maxW;
            float lineW    = 1f + 2.5f * t;
            var   baseCol  = conn.Weight >= 0f ? _o.AccentColor : _o.MeanColor;
            var   lineCol  = new Color(baseCol.R, baseCol.G, baseCol.B, 0.30f + 0.50f * t);
            DrawLine(from, to, lineCol, lineW, true);
        }

        // Draw nodes
        foreach (var node in snap.Nodes)
        {
            if (!nodePos.TryGetValue(node.Id, out var pos)) continue;

            Color fill = node.Role switch
            {
                UINodeRole.Input  => _o.InputNodeColor,
                UINodeRole.Bias   => new Color(_o.InputNodeColor.R, _o.InputNodeColor.G, _o.InputNodeColor.B, 0.55f),
                UINodeRole.Output => _o.AccentColor,
                _                 => _o.HiddenNodeColor,
            };

            var border = new Color(
                Math.Min(fill.R + 0.25f, 1f),
                Math.Min(fill.G + 0.25f, 1f),
                Math.Min(fill.B + 0.25f, 1f),
                fill.A);

            DrawCircle(pos, NodeR + 1.8f, border);
            DrawCircle(pos, NodeR,         fill);
        }
    }

    // ── Network layout ────────────────────────────────────────────────────────

    private static Dictionary<int, Vector2> ComputeLayout(NeatGenomeSnapshot snap, Rect2 area)
    {
        // Assign each node a column index
        var col = new Dictionary<int, int>();
        foreach (var n in snap.Nodes)
        {
            col[n.Id] = n.Role switch
            {
                UINodeRole.Input  => 0,
                UINodeRole.Bias   => 0,
                UINodeRole.Output => int.MaxValue,  // resolved below
                _                 => 1,             // hidden: propagated below
            };
        }

        // BFS propagation: push hidden nodes right past their predecessors
        for (int pass = 0; pass < 20; pass++)
        {
            bool changed = false;
            foreach (var c in snap.Connections)
            {
                if (!c.Enabled) continue;
                if (!col.TryGetValue(c.InNode,  out int inC) || inC == int.MaxValue) continue;
                if (!col.TryGetValue(c.OutNode, out int outC) || outC == int.MaxValue) continue;
                if (outC <= inC)
                {
                    col[c.OutNode] = inC + 1;
                    changed = true;
                }
            }
            if (!changed) break;
        }

        // Finalise output column
        int maxHidden  = col.Values.Where(v => v != int.MaxValue).DefaultIfEmpty(0).Max();
        int outputCol  = maxHidden + 1;
        int totalCols  = outputCol + 1;
        foreach (var n in snap.Nodes)
            if (n.Role == UINodeRole.Output) col[n.Id] = outputCol;

        // Group nodes by column, then assign positions
        var byCol = snap.Nodes
            .GroupBy(n => col.TryGetValue(n.Id, out int c) ? c : 0)
            .ToDictionary(g => g.Key, g => g.ToList());

        var pos = new Dictionary<int, Vector2>();
        float colStep = totalCols <= 1 ? 0f : area.Size.X / (totalCols - 1);

        foreach (var (colIdx, nodes) in byCol)
        {
            float px      = area.Position.X + colIdx * colStep;
            float rowStep = area.Size.Y / (nodes.Count + 1);
            for (int i = 0; i < nodes.Count; i++)
                pos[nodes[i].Id] = new Vector2(px, area.Position.Y + rowStep * (i + 1));
        }

        return pos;
    }

    // ── Editor preview data ───────────────────────────────────────────────────

    private static NeatGenomeSnapshot BuildPreviewSnapshot() => new()
    {
        Nodes = new List<UINodeInfo>
        {
            new(0,  UINodeRole.Input),  new(1,  UINodeRole.Input),
            new(2,  UINodeRole.Input),  new(3,  UINodeRole.Input),
            new(4,  UINodeRole.Input),  new(5,  UINodeRole.Input),
            new(6,  UINodeRole.Bias),
            new(7,  UINodeRole.Hidden), new(8,  UINodeRole.Hidden),
            new(9,  UINodeRole.Hidden),
            new(10, UINodeRole.Output),
        },
        Connections = new List<UIConnectionInfo>
        {
            new(0, 7,  0.80f, true),  new(1, 7, -0.50f, true),
            new(2, 8,  0.90f, true),  new(3, 8,  0.30f, true),
            new(4, 9, -0.70f, true),  new(5, 9,  0.60f, true),
            new(6, 7,  0.20f, true),  new(7, 9,  0.40f, true),
            new(7, 10, 1.20f, true),  new(8, 10,-0.40f, true),
            new(9, 10, 0.90f, true),
        },
        Generation     = 42,
        BestFitness    = 12.40f,
        MeanFitness    = 8.21f,
        PopulationSize = 20,
        AliveCount     = 13,
        SpeciesCount   = 5,
        InputCount     = 6,
        OutputCount    = 1,
    };

    private static List<(float B, float M)> BuildPreviewHistory()
    {
        var h = new List<(float, float)>(30);
        for (int i = 0; i < 30; i++)
        {
            float t    = i / 29f;
            float best = MathF.Sin(t * MathF.PI * 1.5f) * 4f + t * 12f;
            float mean = best * 0.65f + MathF.Sin(t * MathF.PI * 2.3f) * 0.8f;
            h.Add((best, mean));
        }
        return h;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static StyleBoxFlat MakeRoundBox(int radius, bool topOnly = false)
    {
        var sb = new StyleBoxFlat();
        sb.SetCornerRadiusAll(radius);
        sb.CornerDetail = 16;
        if (topOnly) { sb.CornerRadiusBottomLeft = 0; sb.CornerRadiusBottomRight = 0; }
        return sb;
    }
}
