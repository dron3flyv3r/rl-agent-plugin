using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Replaces the default array editor for <see cref="RLNetworkGraph.TrunkLayers"/>
/// with a compact inline list where each layer's properties are editable directly
/// in the row. Supports drag-to-reorder and right-click context menu.
/// </summary>
[Tool]
public partial class RLNetworkGraphInspectorPlugin : EditorInspectorPlugin
{
    public override bool _CanHandle(GodotObject @object)
        => @object is RLNetworkGraph;

    public override bool _ParseProperty(
        GodotObject @object,
        Variant.Type type,
        string name,
        PropertyHint hintType,
        string hintString,
        PropertyUsageFlags usageFlags,
        bool wide)
    {
        if (name == nameof(RLNetworkGraph.TrunkLayers))
        {
            AddPropertyEditor(name, new RLNetworkGraphProperty());
            return true; // suppress the default array editor
        }
        return false;
    }
}
