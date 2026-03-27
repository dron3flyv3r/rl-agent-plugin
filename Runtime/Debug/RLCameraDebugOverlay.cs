using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Debug overlay that shows the live output of every <see cref="RLCameraSensor2D"/> found
/// on any agent in the academy. Works both in normal play and during training.
///
/// Created by <see cref="RLAcademy"/> when <c>EnableCameraDebug</c> is true.
/// Each sensor gets a labelled panel in the top-right corner of the screen.
/// </summary>
public partial class RLCameraDebugOverlay : CanvasLayer
{
    private RLAcademy _academy = null!;

    // One entry per discovered sensor.
    private sealed class SensorPanel
    {
        public RLCameraSensor2D Sensor;
        public PanelContainer   Panel;
        public TextureRect      Rect;
        public Label            Label;

        public SensorPanel(RLCameraSensor2D sensor, PanelContainer panel, TextureRect rect, Label label)
        {
            Sensor = sensor;
            Panel  = panel;
            Rect   = rect;
            Label  = label;
        }
    }

    private readonly List<SensorPanel> _panels = new();
    private HBoxContainer?             _container;

    internal void Initialize(RLAcademy academy)
    {
        _academy = academy;
        Layer    = 127;
    }

    public override void _Ready()
    {
        // Anchor row to top-right.
        var anchor = new Control();
        anchor.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.TopRight);
        anchor.MouseFilter = Control.MouseFilterEnum.Ignore;
        AddChild(anchor);

        _container = new HBoxContainer();
        _container.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.TopRight);
        _container.GrowHorizontal = Control.GrowDirection.Begin;
        _container.AddThemeConstantOverride("separation", 6);
        _container.MouseFilter = Control.MouseFilterEnum.Ignore;
        AddChild(_container);
    }

    public override void _PhysicsProcess(double delta)
    {
        RefreshSensors();
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private void RefreshSensors()
    {
        // Collect all camera sensors currently attached to any agent.
        var found = new List<RLCameraSensor2D>();
        foreach (var agent in _academy.GetAgents())
        {
            foreach (var child in (agent as Node)!.GetChildren())
            {
                CollectSensors(child, found);
            }
        }

        // Add panels for newly discovered sensors.
        foreach (var sensor in found)
        {
            if (_panels.Exists(p => p.Sensor == sensor)) continue;
            AddPanel(sensor);
        }

        // Remove panels for sensors that are gone.
        for (var i = _panels.Count - 1; i >= 0; i--)
        {
            if (!IsInstanceValid(_panels[i].Sensor) || !found.Contains(_panels[i].Sensor))
            {
                _panels[i].Panel.QueueFree();
                _panels.RemoveAt(i);
            }
        }

        // Update label text in case settings changed.
        foreach (var entry in _panels)
        {
            var s = entry.Sensor;
            entry.Label.Text = $"{s.Name}  {s.OutputWidth}×{s.OutputHeight}  {(s.OutputChannels == 1 ? "Gray" : "RGB")}";
        }
    }

    private void AddPanel(RLCameraSensor2D sensor)
    {
        if (_container is null) return;

        var panel = new PanelContainer();
        panel.MouseFilter = Control.MouseFilterEnum.Ignore;
        _container.AddChild(panel);

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 2);
        vbox.MouseFilter = Control.MouseFilterEnum.Ignore;
        panel.AddChild(vbox);

        var label = new Label();
        label.AddThemeFontSizeOverride("font_size", 10);
        label.MouseFilter = Control.MouseFilterEnum.Ignore;
        vbox.AddChild(label);

        var rect = new TextureRect
        {
            CustomMinimumSize = new Vector2(128, 128),
            ExpandMode        = TextureRect.ExpandModeEnum.IgnoreSize,
            StretchMode       = TextureRect.StretchModeEnum.Scale,
            Texture           = sensor.ViewportTexture,
            MouseFilter       = Control.MouseFilterEnum.Ignore,
        };
        vbox.AddChild(rect);

        _panels.Add(new SensorPanel(sensor, panel, rect, label));
    }

    private static void CollectSensors(Node node, List<RLCameraSensor2D> result)
    {
        if (node is RLCameraSensor2D sensor)
            result.Add(sensor);

        foreach (var child in node.GetChildren())
            CollectSensors(child, result);
    }
}
