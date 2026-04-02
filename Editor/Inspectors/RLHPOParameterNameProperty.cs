using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Custom inspector property for <see cref="RLHPOParameter.ParameterName"/>.
/// Shows a categorised dropdown populated by reflection from <see cref="RLTrainerConfig"/>
/// properties tagged with <see cref="HpoGroupAttribute"/>.
/// A "— Custom —" entry at the bottom lets users type any string manually.
/// </summary>
[Tool]
public partial class RLHPOParameterNameProperty : EditorProperty
{
    // ── Desired group display order ──────────────────────────────────────────
    private static readonly string[] GroupOrder = { "General", "PPO / A2C", "SAC", "DQN", "MCTS" };

    // ── Reflected param list (built once, shared across all instances) ────────
    private static readonly System.Lazy<List<(string Group, List<string> Names)>> LazyGroups
        = new(BuildGroups);

    // ── Controls ────────────────────────────────────────────────────────────
    private readonly OptionButton _dropdown;
    private readonly LineEdit _customEdit;

    // ── Index maps ─────────────────────────────────────────────────────────
    // OptionButton.ItemSelected and Select(idx) both use the non-separator item
    // index from OptionButton's own items[] array — separators are NOT counted.
    // GetItemId(idx) however delegates to the underlying PopupMenu which DOES
    // count separators, so we avoid it entirely and use these maps instead.
    private readonly Dictionary<string, int> _paramToSelectIdx  = []; // name  → selectIdx
    private readonly Dictionary<int, string> _selectIdxToParam  = []; // selectIdx → name
    private int _customSelectIdx;

    private bool _updatingControl;

    public RLHPOParameterNameProperty()
    {
        var vbox = new VBoxContainer();
        AddChild(vbox);

        _dropdown = new OptionButton
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            ClipText = true,
            TooltipText = "Select a hyperparameter from RLTrainerConfig, or choose \"— Custom —\" to enter a name manually."
        };
        vbox.AddChild(_dropdown);

        _customEdit = new LineEdit
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            PlaceholderText = "Enter parameter name…",
            Visible = false
        };
        vbox.AddChild(_customEdit);

        BuildDropdown();

        _dropdown.GetPopup().MaxSize = new Vector2I(0, 350);

        _dropdown.ItemSelected += OnDropdownItemSelected;
        _customEdit.TextSubmitted += OnCustomTextSubmitted;
        _customEdit.FocusExited += OnCustomFocusExited;
    }

    // ── Dropdown population ──────────────────────────────────────────────────

    private void BuildDropdown()
    {
        int selectIdx = 0;

        foreach (var (group, names) in LazyGroups.Value)
        {
            _dropdown.AddSeparator(group); // does NOT increment selectIdx
            foreach (var name in names)
            {
                _dropdown.AddItem(name);
                _paramToSelectIdx[name]     = selectIdx;
                _selectIdxToParam[selectIdx] = name;
                selectIdx++;
            }
        }

        _dropdown.AddSeparator(string.Empty);
        _customSelectIdx = selectIdx;
        _dropdown.AddItem("— Custom —");
    }

    private static List<(string Group, List<string> Names)> BuildGroups()
    {
        Dictionary<string, List<string>> dict = [];

        foreach (var prop in typeof(RLTrainerConfig)
                     .GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            var attr = prop.GetCustomAttribute<HpoGroupAttribute>();
            if (attr is null) continue;
            if (!dict.ContainsKey(attr.Group))
                dict[attr.Group] = new List<string>();
            dict[attr.Group].Add(prop.Name);
        }

        // Emit groups in the desired display order; append unknown groups at the end.
        var result = new List<(string, List<string>)>();
        foreach (var group in GroupOrder)
            if (dict.TryGetValue(group, out var names))
                result.Add((group, names));
        foreach (var (key, val) in dict)
            if (!GroupOrder.Contains(key))
                result.Add((key, val));

        return result;
    }

    // ── Inspector ↔ control sync ─────────────────────────────────────────────

    public override void _UpdateProperty()
    {
        var value = (string?)GetEditedObject().Get(GetEditedProperty()) ?? string.Empty;

        _updatingControl = true;
        if (_paramToSelectIdx.TryGetValue(value, out int idx))
        {
            _dropdown.Select(idx);
            _customEdit.Visible = false;
        }
        else
        {
            _dropdown.Select(_customSelectIdx);
            _customEdit.Text = value;
            _customEdit.Visible = true;
        }
        _updatingControl = false;
    }

    // ── Signal handlers ──────────────────────────────────────────────────────

    private void OnDropdownItemSelected(long selectIndex)
    {
        if (_updatingControl) return;

        // ItemSelected gives the non-separator item index (OptionButton.items[]).
        // We use our own map instead of GetItemId(), which incorrectly counts
        // separators when delegating to the underlying PopupMenu.
        if (_selectIdxToParam.TryGetValue((int)selectIndex, out var paramName))
        {
            _customEdit.Visible = false;
            EmitChanged(GetEditedProperty(), paramName);
        }
        else // "— Custom —" was selected
        {
            _customEdit.Visible = true;
            _customEdit.GrabFocus();
        }
    }

    private void OnCustomTextSubmitted(string text)
    {
        if (_updatingControl) return;
        CommitCustomValue(text);
    }

    private void OnCustomFocusExited()
    {
        if (_updatingControl) return;
        CommitCustomValue(_customEdit.Text);
    }

    private void CommitCustomValue(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return;
        EmitChanged(GetEditedProperty(), value.Trim());
    }
}
