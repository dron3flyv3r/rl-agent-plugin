using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Written by the editor when the user starts a recording session.
/// Read by <see cref="RecordingBootstrap"/> on game startup.
/// </summary>
public sealed class RecordingLaunchManifest
{
    public const string ActiveManifestPath = "user://rl-agent-plugin/recording_manifest.json";

    /// <summary>res:// path to the game scene to load for recording.</summary>
    public string ScenePath { get; set; } = string.Empty;

    /// <summary>
    /// Node path to the RLAcademy within the loaded scene.
    /// Empty string triggers depth-first traversal fallback.
    /// </summary>
    public string AcademyNodePath { get; set; } = string.Empty;

    /// <summary>Absolute path where the .rldem output file will be written.</summary>
    public string OutputFilePath { get; set; } = string.Empty;

    /// <summary>
    /// Policy group ID to record. Empty string means record all agents regardless of group.
    /// </summary>
    public string AgentGroupId { get; set; } = string.Empty;

    /// <summary>Initial simulation time scale applied when the recording session starts.</summary>
    public float TimeScale { get; set; } = 1.0f;

    /// <summary>
    /// When true, RecordingBootstrap calls HandleScriptedInput() instead of HandleHumanInput()
    /// so the agent drives itself via its OnScriptedInput() heuristic override.
    /// </summary>
    public bool ScriptMode { get; set; } = false;

    public Error SaveToUserStorage()
    {
        var dirError = DirAccess.MakeDirRecursiveAbsolute(
            ProjectSettings.GlobalizePath("user://rl-agent-plugin"));
        if (dirError != Error.Ok) return dirError;

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Write);
        if (file is null) return FileAccess.GetOpenError();

        file.StoreString(Json.Stringify(new Godot.Collections.Dictionary
        {
            { nameof(ScenePath), ScenePath },
            { nameof(AcademyNodePath), AcademyNodePath },
            { nameof(OutputFilePath), OutputFilePath },
            { nameof(AgentGroupId), AgentGroupId },
            { nameof(TimeScale), TimeScale },
            { nameof(ScriptMode), ScriptMode },
        }, "\t"));

        return Error.Ok;
    }

    public static RecordingLaunchManifest? LoadFromUserStorage()
    {
        if (!FileAccess.FileExists(ActiveManifestPath)) return null;

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Read);
        if (file is null) return null;

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary) return null;

        var d = parsed.AsGodotDictionary();
        return new RecordingLaunchManifest
        {
            ScenePath = ReadString(d, nameof(ScenePath)),
            AcademyNodePath = ReadString(d, nameof(AcademyNodePath)),
            OutputFilePath = ReadString(d, nameof(OutputFilePath)),
            AgentGroupId = ReadString(d, nameof(AgentGroupId)),
            TimeScale = d.ContainsKey(nameof(TimeScale)) ? (float)d[nameof(TimeScale)].AsDouble() : 1.0f,
            ScriptMode = d.ContainsKey(nameof(ScriptMode)) && d[nameof(ScriptMode)].AsBool(),
        };
    }

    private static string ReadString(Godot.Collections.Dictionary d, string key)
        => d.ContainsKey(key) ? d[key].ToString() : string.Empty;
}
