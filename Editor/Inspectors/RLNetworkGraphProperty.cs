using System;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Custom <see cref="EditorProperty"/> for <see cref="RLNetworkGraph.TrunkLayers"/>.
/// Each layer is an inline-editable row with a ⠿ grip handle. Dragging a row
/// live-reorders the list as the cursor moves; releasing commits one undo step.
/// Right-clicking the grip opens a context menu.
/// </summary>
[Tool]
public partial class RLNetworkGraphProperty : EditorProperty
{
    private RLNetworkGraph? _graph;
    private VBoxContainer   _rowContainer = null!;
    private bool _updating;

    // ── Drag state ────────────────────────────────────────────────────────────
    private bool _dragging;
    private int  _dragCurrentIndex; // where the dragged layer sits right now

    public RLNetworkGraphProperty()
    {
        var root = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        AddChild(root);
        SetBottomEditor(root);

        _rowContainer = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _rowContainer.AddThemeConstantOverride("separation", 3);
        root.AddChild(_rowContainer);

        var addBtn = new MenuButton
        {
            Text                = "+ Add Layer",
            SizeFlagsHorizontal = SizeFlags.ShrinkBegin,
            TooltipText         = "Append a new layer to the network.",
        };
        PopulateAddMenu(addBtn.GetPopup());
        root.AddChild(addBtn);

        // Required so _Input fires even when the mouse is outside this control.
        SetProcessInput(true);
    }

    // ── Inspector sync ───────────────────────────────────────────────────────

    public override void _UpdateProperty()
    {
        _graph = GetEditedObject() as RLNetworkGraph;
        _updating = true;
        RebuildRows();
        _updating = false;
    }

    // ── Global input (drag tracking) ─────────────────────────────────────────

    public override void _Input(InputEvent @event)
    {
        if (!_dragging) return;

        if (@event is InputEventMouseMotion)
        {
            HandleDragMotion();
        }
        else if (@event is InputEventMouseButton { ButtonIndex: MouseButton.Left, Pressed: false })
        {
            FinalizeDrag();
            GetViewport().SetInputAsHandled();
        }
    }

    // ── Drag logic ────────────────────────────────────────────────────────────

    internal void StartDrag(int layerIndex)
    {
        if (_graph == null) return;
        _dragging          = true;
        _dragCurrentIndex  = layerIndex;
        _updating = true;
        RebuildRows(_dragCurrentIndex);
        _updating = false;
    }

    private void HandleDragMotion()
    {
        if (_graph == null) return;

        float mouseY      = _rowContainer.GetLocalMousePosition().Y;
        int   targetIndex = ComputeTargetIndex(mouseY);
        if (targetIndex == _dragCurrentIndex) return;

        // Move the layer in the graph array live.
        var layer = _graph.TrunkLayers[_dragCurrentIndex];
        _graph.TrunkLayers.RemoveAt(_dragCurrentIndex);
        _graph.TrunkLayers.Insert(targetIndex, layer);
        _dragCurrentIndex = targetIndex;

        _updating = true;
        RebuildRows(_dragCurrentIndex);
        _updating = false;
    }

    private int ComputeTargetIndex(float mouseY)
    {
        int count = _rowContainer.GetChildCount();
        for (int i = 0; i < count; i++)
        {
            if (_rowContainer.GetChild(i) is RLNetworkGraphLayerRow row)
            {
                if (mouseY < row.Position.Y + row.Size.Y / 2f)
                    return i;
            }
        }
        return Math.Max(0, count - 1);
    }

    private void FinalizeDrag()
    {
        _dragging = false;
        if (_graph == null) return;
        // Emit once so the editor records a single undo step.
        EmitChanged(GetEditedProperty(), _graph.TrunkLayers);
        _updating = true;
        RebuildRows();
        _updating = false;
    }

    // ── Row building ─────────────────────────────────────────────────────────

    private void RebuildRows(int highlightIndex = -1)
    {
        foreach (var child in _rowContainer.GetChildren())
            child.QueueFree();

        if (_graph == null) return;

        for (int i = 0; i < _graph.TrunkLayers.Count; i++)
        {
            if (_graph.TrunkLayers[i] is RLLayerDef layerDef)
            {
                var row = CreateRow(i, layerDef);
                if (i == highlightIndex)
                    ApplyDragStyle(row);
                _rowContainer.AddChild(row);
            }
        }
    }

    private RLNetworkGraphLayerRow CreateRow(int index, RLLayerDef layer)
    {
        var typeName = GetTypeName(layer);
        var row      = new RLNetworkGraphLayerRow(index, this);
        ApplyRowStyle(row, index);

        var flow = row.FlowContent;

        flow.AddChild(new RLNetworkGraphGripHandle(index, typeName, this));

        flow.AddChild(new Label
        {
            Text              = typeName,
            CustomMinimumSize = new Vector2(65, 0),
            VerticalAlignment = VerticalAlignment.Center,
        });

        AppendLayerControls(flow, layer);
        return row;
    }

    private void AppendLayerControls(FlowContainer flow, RLLayerDef layer)
    {
        switch (layer)
        {
            case RLDenseLayerDef dense:
                flow.AddChild(MakePair("Size",
                    MakeIntSpinBox(dense.Size, 1, 4096,
                        v => ApplyLayerEdit(dense, () => dense.Size = v))));
                flow.AddChild(MakeActivationDropdown(dense.Activation,
                    v => ApplyLayerEdit(dense, () => dense.Activation = v)));
                break;

            case RLDropoutLayerDef dropout:
                flow.AddChild(MakePair("Rate",
                    MakeFloatSpinBox(dropout.Rate, 0f, 0.99f, 0.01f,
                        v => ApplyLayerEdit(dropout, () => dropout.Rate = v))));
                break;

            case RLLstmLayerDef lstm:
                flow.AddChild(MakePair("Hidden",
                    MakeIntSpinBox(lstm.HiddenSize, 1, 4096,
                        v => ApplyLayerEdit(lstm, () => lstm.HiddenSize = v))));
                flow.AddChild(MakePair("Clip",
                    MakeFloatSpinBox(lstm.GradClipNorm, 0f, 100f, 0.1f,
                        v => ApplyLayerEdit(lstm, () => lstm.GradClipNorm = v))));
                break;

            case RLGruLayerDef gru:
                flow.AddChild(MakePair("Hidden",
                    MakeIntSpinBox(gru.HiddenSize, 1, 4096,
                        v => ApplyLayerEdit(gru, () => gru.HiddenSize = v))));
                flow.AddChild(MakePair("Clip",
                    MakeFloatSpinBox(gru.GradClipNorm, 0f, 100f, 0.1f,
                        v => ApplyLayerEdit(gru, () => gru.GradClipNorm = v))));
                break;

            // LayerNorm and Flatten have no configurable parameters.
        }
    }

    // ── Control factories ────────────────────────────────────────────────────

    private static HBoxContainer MakePair(string labelText, Control input)
    {
        var hbox = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        hbox.AddThemeConstantOverride("separation", 4);
        hbox.AddChild(new Label
        {
            Text              = labelText,
            VerticalAlignment = VerticalAlignment.Center,
        });
        hbox.AddChild(input);
        return hbox;
    }

    private static SpinBox MakeIntSpinBox(int value, int min, int max, Action<int> onChange)
    {
        var spin = new SpinBox
        {
            MinValue            = min,
            MaxValue            = max,
            Step                = 1,
            Value               = value,
            CustomMinimumSize   = new Vector2(60, 0),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        spin.ValueChanged += v => onChange((int)v);
        return spin;
    }

    private static SpinBox MakeFloatSpinBox(float value, float min, float max, float step,
                                             Action<float> onChange)
    {
        var spin = new SpinBox
        {
            MinValue            = min,
            MaxValue            = max,
            Step                = step,
            Value               = value,
            CustomMinimumSize   = new Vector2(60, 0),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        spin.ValueChanged += v => onChange((float)v);
        return spin;
    }

    private static OptionButton MakeActivationDropdown(RLActivationKind current,
                                                        Action<RLActivationKind> onChange)
    {
        var btn = new OptionButton
        {
            CustomMinimumSize   = new Vector2(60, 0),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        btn.AddItem("Tanh", (int)RLActivationKind.Tanh);
        btn.AddItem("ReLU", (int)RLActivationKind.Relu);
        btn.Select(btn.GetItemIndex((int)current));
        btn.ItemSelected += idx => onChange((RLActivationKind)btn.GetItemId((int)idx));
        return btn;
    }

    // ── Add-layer menu ───────────────────────────────────────────────────────

    private void PopulateAddMenu(PopupMenu popup)
    {
        popup.AddItem("Dense",     0);
        popup.AddItem("Dropout",   1);
        popup.AddItem("LayerNorm", 2);
        popup.AddItem("Flatten",   3);
        popup.AddSeparator();
        popup.AddItem("LSTM",      4);
        popup.AddItem("GRU",       5);
        popup.IdPressed += AddLayer;
    }

    // ── Context menu ─────────────────────────────────────────────────────────

    internal void ShowContextMenu(int layerIndex)
    {
        var popup = new PopupMenu();
        popup.AddItem("Move Up",   0);
        popup.AddItem("Move Down", 1);
        popup.AddSeparator();
        popup.AddItem("Delete",    2);

        popup.SetItemDisabled(0, layerIndex == 0);
        popup.SetItemDisabled(1, _graph == null || layerIndex >= _graph.TrunkLayers.Count - 1);

        popup.IdPressed += id =>
        {
            switch ((int)id)
            {
                case 0: MoveLayer(layerIndex, layerIndex - 1); break;
                case 1: MoveLayer(layerIndex, layerIndex + 1); break;
                case 2: RemoveLayer(layerIndex);               break;
            }
        };

        AddChild(popup);
        popup.Position = DisplayServer.MouseGetPosition();
        popup.Popup();
        popup.PopupHide += popup.QueueFree;
    }

    // ── Mutations ────────────────────────────────────────────────────────────

    private void AddLayer(long typeId)
    {
        if (_graph == null) return;
        RLLayerDef newLayer = (int)typeId switch
        {
            0 => new RLDenseLayerDef(),
            1 => new RLDropoutLayerDef(),
            2 => new RLLayerNormDef(),
            3 => new RLFlattenLayerDef(),
            4 => new RLLstmLayerDef(),
            5 => new RLGruLayerDef(),
            _ => new RLDenseLayerDef(),
        };
        _graph.TrunkLayers.Add(newLayer);
        Commit();
    }

    private void RemoveLayer(int index)
    {
        if (_graph == null || index < 0 || index >= _graph.TrunkLayers.Count) return;
        _graph.TrunkLayers.RemoveAt(index);
        Commit();
    }

    internal void MoveLayer(int fromIndex, int toIndex)
    {
        if (_graph == null) return;
        if (fromIndex < 0 || fromIndex >= _graph.TrunkLayers.Count) return;
        if (toIndex < 0 || toIndex >= _graph.TrunkLayers.Count) return;
        if (fromIndex == toIndex) return;
        var layer = _graph.TrunkLayers[fromIndex];
        _graph.TrunkLayers.RemoveAt(fromIndex);
        _graph.TrunkLayers.Insert(toIndex, layer);
        Commit();
    }

    private void ApplyLayerEdit(RLLayerDef layer, Action edit)
    {
        if (_updating || _graph == null) return;
        edit();
        _graph.EmitChanged();
    }

    private void Commit()
    {
        if (_updating || _graph == null) return;
        EmitChanged(GetEditedProperty(), _graph.TrunkLayers);
        CallDeferred(nameof(RebuildRowsDeferred));
    }

    private void RebuildRowsDeferred()
    {
        if (!IsInsideTree()) return;
        _updating = true;
        RebuildRows(_dragging ? _dragCurrentIndex : -1);
        _updating = false;
    }

    // ── Helpers / styling ────────────────────────────────────────────────────

    internal static string GetTypeName(RLLayerDef layer) => layer switch
    {
        RLDenseLayerDef   => "Dense",
        RLDropoutLayerDef => "Dropout",
        RLLayerNormDef    => "LayerNorm",
        RLFlattenLayerDef => "Flatten",
        RLLstmLayerDef    => "LSTM",
        RLGruLayerDef     => "GRU",
        _                 => layer.GetType().Name,
    };

    private static void ApplyRowStyle(PanelContainer panel, int rowIndex)
    {
        var theme     = EditorInterface.Singleton.GetEditorTheme();
        var styleName = rowIndex % 2 == 0 ? "sub_inspector_bg0" : "sub_inspector_bg1";

        StyleBoxFlat? flat = theme.HasStylebox(styleName, "EditorStyles")
            ? theme.GetStylebox(styleName, "EditorStyles").Duplicate() as StyleBoxFlat
            : null;

        if (flat == null)
        {
            var baseColor = theme.GetColor("base_color", "Editor");
            flat = new StyleBoxFlat
            {
                BgColor             = rowIndex % 2 == 0 ? baseColor.Lightened(0.04f) : baseColor,
                ContentMarginLeft   = 4,
                ContentMarginRight  = 4,
                ContentMarginTop    = 2,
                ContentMarginBottom = 2,
            };
        }

        const int Radius = 4;
        flat.CornerRadiusTopLeft     = Radius;
        flat.CornerRadiusTopRight    = Radius;
        flat.CornerRadiusBottomLeft  = Radius;
        flat.CornerRadiusBottomRight = Radius;
        panel.AddThemeStyleboxOverride("panel", flat);
    }

    private static void ApplyDragStyle(RLNetworkGraphLayerRow row)
    {
        var accentColor = EditorInterface.Singleton.GetEditorTheme()
                                         .GetColor("accent_color", "Editor");
        var style = new StyleBoxFlat
        {
            BgColor                  = new Color(accentColor.R, accentColor.G, accentColor.B, 0.2f),
            BorderColor              = accentColor,
            BorderWidthLeft          = 2, BorderWidthRight  = 2,
            BorderWidthTop           = 2, BorderWidthBottom = 2,
            CornerRadiusTopLeft      = 4, CornerRadiusTopRight    = 4,
            CornerRadiusBottomLeft   = 4, CornerRadiusBottomRight = 4,
            ContentMarginLeft        = 4, ContentMarginRight  = 4,
            ContentMarginTop         = 2, ContentMarginBottom  = 2,
        };
        row.AddThemeStyleboxOverride("panel", style);
    }
}

// ── Top-level helper types ────────────────────────────────────────────────────
// Top-level (non-nested) so Godot's C# source generator reliably processes
// the virtual-method overrides.

/// <summary>
/// A themed row for one layer. Children are added to <see cref="FlowContent"/>
/// so they wrap when the inspector panel is narrow.
/// </summary>
[Tool]
internal partial class RLNetworkGraphLayerRow : PanelContainer
{
    public FlowContainer FlowContent { get; }

    public RLNetworkGraphLayerRow(int layerIndex, RLNetworkGraphProperty owner)
    {
        SizeFlagsHorizontal = SizeFlags.ExpandFill;

        var margin = new MarginContainer();
        margin.AddThemeConstantOverride("margin_top",    2);
        margin.AddThemeConstantOverride("margin_bottom", 2);
        margin.AddThemeConstantOverride("margin_left",   4);
        margin.AddThemeConstantOverride("margin_right",  4);
        AddChild(margin);

        FlowContent = new FlowContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        FlowContent.AddThemeConstantOverride("h_separation", 6);
        FlowContent.AddThemeConstantOverride("v_separation", 4);
        margin.AddChild(FlowContent);
    }
}

/// <summary>
/// Grip handle on the left of each row.
/// Left-press initiates live drag; right-click opens the context menu.
/// </summary>
[Tool]
internal partial class RLNetworkGraphGripHandle : Control
{
    private readonly int    _layerIndex;
    private readonly string _typeName;
    private readonly RLNetworkGraphProperty _owner;

    public RLNetworkGraphGripHandle(int layerIndex, string typeName, RLNetworkGraphProperty owner)
    {
        _layerIndex = layerIndex;
        _typeName   = typeName;
        _owner      = owner;

        CustomMinimumSize       = new Vector2(18, 24);
        TooltipText             = "Drag to reorder  ·  Right-click for options";
        MouseDefaultCursorShape = CursorShape.Drag;

        AddChild(new Label
        {
            Text                = "⠿",
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment   = VerticalAlignment.Center,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical   = SizeFlags.ExpandFill,
            MouseFilter         = MouseFilterEnum.Ignore,
        });
    }

    public override void _GuiInput(InputEvent @event)
    {
        if (@event is not InputEventMouseButton mb) return;

        if (mb.ButtonIndex == MouseButton.Left && mb.Pressed)
        {
            _owner.StartDrag(_layerIndex);
            AcceptEvent();
        }
        else if (mb.ButtonIndex == MouseButton.Right && mb.Pressed)
        {
            _owner.ShowContextMenu(_layerIndex);
            AcceptEvent();
        }
    }
}
