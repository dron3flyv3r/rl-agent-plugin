using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// In-game debug overlay that shows observations, actions, and reward signals
/// for a selected agent. Created by RLAcademy when EnableSpyOverlay is true.
///
/// Human-mode and Inference-mode agents are listed by default. Quick test mode
/// can opt into Train-mode agents as well.
/// Press Tab / Shift+Tab to cycle through agents.
/// </summary>
public partial class RLAgentSpyOverlay : CanvasLayer
{
    private RLAcademy _academy = null!;
    private readonly List<IRLAgent> _agents = new();
    private int _pinnedIndex;
    private SpyDrawPanel _panel = null!;
    private bool _includeTrainAgents;

    internal void Initialize(RLAcademy academy, bool includeTrainAgents = false)
    {
        _academy = academy;
        _includeTrainAgents = includeTrainAgents;
        Layer = 128;
    }

    public override void _Ready()
    {
        _panel = new SpyDrawPanel();
        _panel.MouseFilter = Control.MouseFilterEnum.Ignore;
        _panel.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        AddChild(_panel);
    }

    public override void _PhysicsProcess(double delta)
    {
        _agents.Clear();
        foreach (var agent in _academy.GetAgents())
        {
            if (agent.ControlMode is RLAgentControlMode.Human or RLAgentControlMode.Inference
                || (_includeTrainAgents && agent.ControlMode is RLAgentControlMode.Train or RLAgentControlMode.Auto))
                _agents.Add(agent);
        }

        if (_agents.Count == 0)
        {
            _panel.ClearSnapshot();
            return;
        }

        _pinnedIndex = Mathf.Clamp(_pinnedIndex, 0, _agents.Count - 1);
        var pinned = _agents[_pinnedIndex];

        // Human-mode agents are not stepped by the academy, so collect their observations here.
        if (pinned.ControlMode == RLAgentControlMode.Human)
            pinned.CollectObservationArray();

        _panel.UpdateSnapshot(pinned, _pinnedIndex, _agents.Count);
    }

    public override void _Input(InputEvent @event)
    {
        if (@event is not InputEventKey { Pressed: true, Echo: false } key) return;
        if (key.Keycode != Key.Tab || _agents.Count <= 1) return;

        _pinnedIndex = key.ShiftPressed
            ? (_pinnedIndex - 1 + _agents.Count) % _agents.Count
            : (_pinnedIndex + 1) % _agents.Count;
    }
}

// ── Internal draw panel ──────────────────────────────────────────────────────

internal sealed partial class SpyDrawPanel : Control
{
    private const int FontSize = 13;
    private const int LineHeight = FontSize + 3;
    private const float Padding = 10f;
    private const float PanelWidth = 310f;
    private const int MaxObsLines = 20;

    private static readonly Color BgColor     = new(0f,    0f,    0f,    0.78f);
    private static readonly Color BorderColor = new(0.3f,  0.7f,  1f,    0.55f);
    private static readonly Color HeaderColor = new(0.35f, 0.8f,  1f,    1f);
    private static readonly Color SepColor    = new(0.45f, 0.45f, 0.45f, 0.8f);
    private static readonly Color WhiteColor  = Colors.White;
    private static readonly Color GrayColor   = new(0.55f, 0.55f, 0.55f, 1f);
    private static readonly Color PosColor    = new(0.4f,  1f,    0.5f,  1f);
    private static readonly Color NegColor    = new(1f,    0.4f,  0.4f,  1f);
    private static readonly Color YellowColor = new(1f,    0.9f,  0.35f, 1f);

    private readonly List<(string text, Color color)> _lines = new();
    private bool _hasSnapshot;

    internal void UpdateSnapshot(IRLAgent agent, int index, int total)
    {
        _lines.Clear();
        _hasSnapshot = true;

        // ── Header
        var modeStr = agent.ControlMode switch
        {
            RLAgentControlMode.Human => "Human",
            RLAgentControlMode.Train => "Train",
            _ => "Inference",
        };
        var agentSuffix = total > 1 ? $"  [{index + 1}/{total}]" : "";
        _lines.Add(($"[RLSpy]  {agent.AsNode().Name}  ({modeStr}){agentSuffix}", HeaderColor));
        _lines.Add(($"Steps: {agent.EpisodeSteps}   Episode Reward: {agent.EpisodeReward:F3}", WhiteColor));

        // ── Observations
        _lines.Add(("──── Observations ──────────────", SepColor));
        AppendObservations(agent);

        // ── Action
        _lines.Add(("──── Action ────────────────────", SepColor));
        AppendAction(agent);

        // ── Reward
        _lines.Add(("──── Last Step Reward ──────────", SepColor));
        AppendReward(agent);

        // ── Footer hint
        if (total > 1)
            _lines.Add(("  Tab / Shift+Tab: cycle agent", GrayColor));

        QueueRedraw();
    }

    internal void ClearSnapshot()
    {
        if (!_hasSnapshot) return;
        _hasSnapshot = false;
        _lines.Clear();
        QueueRedraw();
    }

    public override void _Draw()
    {
        if (!_hasSnapshot || _lines.Count == 0) return;

        var font = ThemeDB.FallbackFont;
        var totalHeight = _lines.Count * LineHeight + Padding * 2;
        var rect = new Rect2(8, 8, PanelWidth + Padding * 2, totalHeight);

        DrawRect(rect, BgColor, true);
        DrawRect(rect, BorderColor, false, 1f);

        var x = rect.Position.X + Padding;
        var y = rect.Position.Y + Padding + FontSize;

        foreach (var (text, color) in _lines)
        {
            DrawString(font, new Vector2(x, y), text,
                HorizontalAlignment.Left, PanelWidth, FontSize, color);
            y += LineHeight;
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private void AppendObservations(IRLAgent agent)
    {
        var obs = agent.GetLastObservation();
        var segments = agent.LastObservationSegments;
        var linesAdded = 0;

        if (segments.Count > 0)
        {
            foreach (var seg in segments)
            {
                for (var i = 0; i < seg.Length; i++)
                {
                    if (linesAdded >= MaxObsLines)
                    {
                        var remaining = obs.Length - (seg.StartIndex + i);
                        _lines.Add(($"  … {remaining} more value(s)", GrayColor));
                        return;
                    }

                    var idx = seg.StartIndex + i;
                    var val = idx < obs.Length ? obs[idx] : 0f;
                    string label;
                    if (seg.DebugLabels is { } dl && i < dl.Length)
                        label = $"{seg.Name}.{dl[i]}";
                    else
                        label = seg.Length == 1 ? seg.Name : $"{seg.Name}[{i}]";

                    _lines.Add(($"  {label}: {val:F3}", ValColor(val)));
                    linesAdded++;
                }
            }

            // Any trailing values not covered by named segments
            var coveredEnd = segments[segments.Count - 1].StartIndex + segments[segments.Count - 1].Length;
            for (var i = coveredEnd; i < obs.Length; i++)
            {
                if (linesAdded >= MaxObsLines)
                {
                    _lines.Add(($"  … {obs.Length - i} more value(s)", GrayColor));
                    return;
                }
                _lines.Add(($"  obs[{i}]: {obs[i]:F3}", ValColor(obs[i])));
                linesAdded++;
            }
        }
        else
        {
            for (var i = 0; i < obs.Length; i++)
            {
                if (linesAdded >= MaxObsLines)
                {
                    _lines.Add(($"  … {obs.Length - i} more value(s)", GrayColor));
                    return;
                }
                _lines.Add(($"  obs[{i}]: {obs[i]:F3}", ValColor(obs[i])));
                linesAdded++;
            }
        }

        if (obs.Length == 0)
            _lines.Add(("  (no observations collected yet)", GrayColor));
    }

    private void AppendAction(IRLAgent agent)
    {
        if (agent.CurrentActionIndex >= 0)
        {
            var labels = agent.GetDiscreteActionLabels();
            var label = agent.CurrentActionIndex < labels.Length
                ? labels[agent.CurrentActionIndex]
                : agent.CurrentActionIndex.ToString();
            _lines.Add(($"  Discrete: {agent.CurrentActionIndex}  ({label})", YellowColor));
        }
        else if (agent.CurrentContinuousActions.Length > 0)
        {
            for (var i = 0; i < agent.CurrentContinuousActions.Length; i++)
            {
                var v = agent.CurrentContinuousActions[i];
                _lines.Add(($"  cont[{i}]: {v:F3}", ValColor(v)));
            }
        }
        else
        {
            _lines.Add(("  (none)", GrayColor));
        }
    }

    private void AppendReward(IRLAgent agent)
    {
        // Inference agents: reward is consumed each step, so LastStepReward reflects the previous step.
        // Human agents: nobody consumes pending reward, so show the live pending total instead.
        var isInference = agent.ControlMode == RLAgentControlMode.Inference;
        var breakdown = isInference ? agent.GetLastStepRewardBreakdown() : agent.GetPendingRewardBreakdown();
        var total = isInference ? agent.LastStepReward : agent.PendingReward;

        if (breakdown.Count > 0)
        {
            foreach (var (tag, amount) in breakdown)
            {
                var col = amount > 0 ? PosColor : amount < 0 ? NegColor : WhiteColor;
                _lines.Add(($"  {tag}: {amount:+0.000;-0.000;0.000}", col));
            }
            var totalCol = total > 0 ? PosColor : total < 0 ? NegColor : WhiteColor;
            _lines.Add(($"  Total: {total:+0.000;-0.000;0.000}", totalCol));
        }
        else if (total != 0f)
        {
            var col = total > 0 ? PosColor : NegColor;
            _lines.Add(($"  {total:+0.000;-0.000}", col));
        }
        else
        {
            _lines.Add(("  (no reward this step)", GrayColor));
        }
    }

    private static Color ValColor(float v) =>
        v > 0.1f ? PosColor : v < -0.1f ? NegColor : WhiteColor;
}
