using Godot;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Replaces the plain string editor for <see cref="RLHPOParameter.ParameterName"/>
/// with a categorised dropdown built from <see cref="Runtime.RLTrainerConfig"/> properties.
/// </summary>
[Tool]
public partial class RLHPOParameterInspectorPlugin : EditorInspectorPlugin
{
    public override bool _CanHandle(GodotObject @object)
        => @object is RLHPOParameter;

    public override bool _ParseProperty(
        GodotObject @object,
        Variant.Type type,
        string name,
        PropertyHint hintType,
        string hintString,
        PropertyUsageFlags usageFlags,
        bool wide)
    {
        if (name == nameof(RLHPOParameter.ParameterName))
        {
            AddPropertyEditor(name, new RLHPOParameterNameProperty());
            return true; // suppress the default LineEdit
        }
        if (name == nameof(RLHPOParameter.Low))
        {
            AddPropertyEditor(name, new RLHPOParameterNumericProperty("e.g. 0.0001"));
            return true;
        }
        if (name == nameof(RLHPOParameter.High))
        {
            AddPropertyEditor(name, new RLHPOParameterNumericProperty("e.g. 0.003"));
            return true;
        }
        return false;
    }
}
