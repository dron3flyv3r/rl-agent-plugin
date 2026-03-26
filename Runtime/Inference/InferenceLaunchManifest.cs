using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Written by the editor "Run Inference" button and read by <see cref="InferenceBootstrap"/>.
/// Carries just enough information to launch the training scene in inference mode.
/// </summary>
public sealed class InferenceLaunchManifest
{
    public const string ActiveManifestPath = "user://rl_agent_plugin/inference_manifest.json";

    /// <summary>res:// path to the training scene to load.</summary>
    public string ScenePath { get; set; } = string.Empty;

    /// <summary>Path to the RLAcademy node relative to the scene root.</summary>
    public string AcademyNodePath { get; set; } = string.Empty;

    /// <summary>Simulation speed forwarded to RLAcademy.</summary>
    public float SimulationSpeed { get; set; } = 1.0f;

    /// <summary>Action-repeat forwarded to RLAcademy.</summary>
    public int ActionRepeat { get; set; } = 1;

    public Error SaveToUserStorage()
    {
        var dir = ActiveManifestPath.GetBaseDir();
        if (!string.IsNullOrEmpty(dir))
            DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(dir));

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Write);
        if (file is null) return FileAccess.GetOpenError();

        var data = new Godot.Collections.Dictionary
        {
            { "scene_path",       ScenePath },
            { "academy_path",     AcademyNodePath },
            { "simulation_speed", SimulationSpeed },
            { "action_repeat",    ActionRepeat },
        };
        file.StoreString(Json.Stringify(data));
        return Error.Ok;
    }

    public static InferenceLaunchManifest? LoadFromUserStorage()
    {
        if (!FileAccess.FileExists(ActiveManifestPath)) return null;

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Read);
        if (file is null) return null;

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary) return null;

        var d = parsed.AsGodotDictionary();
        return new InferenceLaunchManifest
        {
            ScenePath       = d.ContainsKey("scene_path")       ? d["scene_path"].AsString()        : string.Empty,
            AcademyNodePath = d.ContainsKey("academy_path")     ? d["academy_path"].AsString()      : string.Empty,
            SimulationSpeed = d.ContainsKey("simulation_speed") ? d["simulation_speed"].AsSingle()  : 1.0f,
            ActionRepeat    = d.ContainsKey("action_repeat")    ? (int)d["action_repeat"].AsInt64() : 1,
        };
    }
}
