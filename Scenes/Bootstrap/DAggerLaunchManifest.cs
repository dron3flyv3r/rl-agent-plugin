using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Written by the editor when the user starts a DAgger aggregation round.
/// Read by <see cref="DAggerBootstrap"/> on game startup.
/// </summary>
public sealed class DAggerLaunchManifest
{
    public const string ActiveManifestPath = "user://rl-agent-plugin/dagger_manifest.json";

    public string ScenePath { get; set; } = string.Empty;
    public string AcademyNodePath { get; set; } = string.Empty;
    public string OutputFilePath { get; set; } = string.Empty;
    public string AgentGroupId { get; set; } = string.Empty;
    public string SeedDatasetPath { get; set; } = string.Empty;
    public string LearnerCheckpointPath { get; set; } = string.Empty;
    public string NetworkGraphPath { get; set; } = string.Empty;
    public int AdditionalFrames { get; set; } = 2048;

    /// <summary>
    /// Base decay factor for the DAgger beta mixing policy.
    /// In round <c>i</c> (1-indexed) the effective mixing probability is
    /// <c>beta^i</c>: that fraction of steps are driven by the expert,
    /// the rest by the learner. Set to 1.0 to always use the expert
    /// (pure BC-style data), or 0.0 to always use the learner.
    /// </summary>
    public float MixingBeta { get; set; } = 0.5f;

    /// <summary>
    /// 1-indexed round number used to compute the effective beta (<c>MixingBeta^RoundIndex</c>).
    /// Manual DAgger always passes 1; Auto DAgger passes the current loop round.
    /// </summary>
    public int RoundIndex { get; set; } = 1;

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
            { nameof(SeedDatasetPath), SeedDatasetPath },
            { nameof(LearnerCheckpointPath), LearnerCheckpointPath },
            { nameof(NetworkGraphPath), NetworkGraphPath },
            { nameof(AdditionalFrames), AdditionalFrames },
            { nameof(MixingBeta), MixingBeta },
            { nameof(RoundIndex), RoundIndex },
        }, "\t"));

        return Error.Ok;
    }

    public static DAggerLaunchManifest? LoadFromUserStorage()
    {
        if (!FileAccess.FileExists(ActiveManifestPath)) return null;

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Read);
        if (file is null) return null;

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary) return null;

        var d = parsed.AsGodotDictionary();
        return new DAggerLaunchManifest
        {
            ScenePath = ReadString(d, nameof(ScenePath)),
            AcademyNodePath = ReadString(d, nameof(AcademyNodePath)),
            OutputFilePath = ReadString(d, nameof(OutputFilePath)),
            AgentGroupId = ReadString(d, nameof(AgentGroupId)),
            SeedDatasetPath = ReadString(d, nameof(SeedDatasetPath)),
            LearnerCheckpointPath = ReadString(d, nameof(LearnerCheckpointPath)),
            NetworkGraphPath = ReadString(d, nameof(NetworkGraphPath)),
            AdditionalFrames = d.ContainsKey(nameof(AdditionalFrames))
                ? d[nameof(AdditionalFrames)].AsInt32()
                : 2048,
            MixingBeta = d.ContainsKey(nameof(MixingBeta))
                ? (float)d[nameof(MixingBeta)].AsDouble()
                : 0.5f,
            RoundIndex = d.ContainsKey(nameof(RoundIndex))
                ? d[nameof(RoundIndex)].AsInt32()
                : 1,
        };
    }

    private static string ReadString(Godot.Collections.Dictionary d, string key)
        => d.ContainsKey(key) ? d[key].ToString() : string.Empty;
}
