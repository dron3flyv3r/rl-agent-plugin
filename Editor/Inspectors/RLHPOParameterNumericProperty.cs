using System.Globalization;
using Godot;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Custom numeric editor for HPO parameter bounds.
/// Avoids Godot's built-in float parsing noise when entering small values in scientific notation.
/// </summary>
[Tool]
public partial class RLHPOParameterNumericProperty : EditorProperty
{
    private readonly LineEdit _edit;
    private bool _updatingControl;

    public RLHPOParameterNumericProperty(string placeholderText)
    {
        _edit = new LineEdit
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            PlaceholderText = placeholderText
        };
        AddChild(_edit);

        _edit.TextSubmitted += OnTextSubmitted;
        _edit.FocusExited += OnFocusExited;
    }

    public override void _UpdateProperty()
    {
        Variant value = GetEditedObject().Get(GetEditedProperty());
        float numericValue = value.VariantType switch
        {
            Variant.Type.Float => (float)(double)value,
            Variant.Type.Int => (int)value,
            _ => 0f,
        };

        _updatingControl = true;
        _edit.Text = FormatNumeric(numericValue);
        _updatingControl = false;
    }

    private void OnTextSubmitted(string text)
    {
        if (_updatingControl) return;
        Commit(text);
    }

    private void OnFocusExited()
    {
        if (_updatingControl) return;
        Commit(_edit.Text);
    }

    private void Commit(string text)
    {
        var trimmed = text.Trim();
        if (string.IsNullOrEmpty(trimmed))
        {
            _UpdateProperty();
            return;
        }

        if (!TryParseNumeric(trimmed, out float parsed))
        {
            GD.PushWarning($"[HPO] Invalid numeric value '{trimmed}' for {GetEditedProperty()}.");
            _UpdateProperty();
            return;
        }

        EmitChanged(GetEditedProperty(), parsed);
    }

    private static bool TryParseNumeric(string text, out float value)
    {
        if (float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            return true;

        if (float.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out value))
            return true;

        value = 0f;
        return false;
    }

    private static string FormatNumeric(float value)
    {
        if (value == 0f)
            return "0";

        var text = value.ToString("0.############################", CultureInfo.InvariantCulture);
        return text.Contains('.')
            ? text.TrimEnd('0').TrimEnd('.')
            : text;
    }
}
