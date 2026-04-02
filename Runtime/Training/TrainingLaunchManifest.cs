using System;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class TrainingLaunchManifest
{
    public const string ActiveManifestPath = "user://rl-agent-plugin/active_manifest.json";

    public string ScenePath { get; set; } = string.Empty;
    public string AcademyNodePath { get; set; } = string.Empty;
    public string RunId { get; set; } = string.Empty;
    public string RunDirectory { get; set; } = string.Empty;
    public string TrainingConfigPath { get; set; } = string.Empty;
    public string NetworkConfigPath { get; set; } = string.Empty;
    public string CheckpointPath { get; set; } = string.Empty;
    public string MetricsPath { get; set; } = string.Empty;
    public string StatusPath { get; set; } = string.Empty;
    public int CheckpointInterval { get; set; } = 10;
    public float SimulationSpeed { get; set; } = 1.0f;
    public int ActionRepeat { get; set; } = 1;
    public int BatchSize { get; set; } = 1;
    public bool QuickTestMode { get; set; }
    public int QuickTestEpisodeLimit { get; set; } = 5;
    public bool QuickTestShowSpyOverlay { get; set; }
    /// <summary>
    /// When set by the HPO orchestrator, TrainingBootstrap applies the JSON key-value
    /// overrides from this path on top of the resolved RLTrainerConfig.
    /// Empty string means no HPO override.
    /// </summary>
    public string HpoOverridePath { get; set; } = string.Empty;
    /// <summary>
    /// Optional session heartbeat file written by the HPO master/orchestrator.
    /// Trial subprocesses require this heartbeat to stay fresh while they run.
    /// </summary>
    public string HpoMasterHeartbeatPath { get; set; } = string.Empty;
    /// <summary>
    /// Session token that must match the contents of <see cref="HpoMasterHeartbeatPath"/>.
    /// Prevents stale or unrelated heartbeat files from keeping orphaned trials alive.
    /// </summary>
    public string HpoMasterHeartbeatToken { get; set; } = string.Empty;

    public static TrainingLaunchManifest CreateDefault() => new();

    public Error SaveToUserStorage()
    {
        var directoryError = EnsureParentDirectory(ActiveManifestPath);
        if (directoryError != Error.Ok)
        {
            return directoryError;
        }

        if (!string.IsNullOrWhiteSpace(RunDirectory))
        {
            var runDirectoryError = DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(RunDirectory));
            if (runDirectoryError != Error.Ok)
            {
                return runDirectoryError;
            }
        }

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            return FileAccess.GetOpenError();
        }

        file.StoreString(Json.Stringify(ToDictionary(), "\t"));
        return Error.Ok;
    }

    public static TrainingLaunchManifest? LoadFromUserStorage()
    {
        if (!FileAccess.FileExists(ActiveManifestPath))
        {
            return null;
        }

        using var file = FileAccess.Open(ActiveManifestPath, FileAccess.ModeFlags.Read);
        if (file is null)
        {
            return null;
        }

        var parsedManifest = Json.ParseString(file.GetAsText());
        if (parsedManifest.VariantType != Variant.Type.Dictionary)
        {
            return null;
        }

        var data = parsedManifest.AsGodotDictionary();
        return new TrainingLaunchManifest
        {
            ScenePath = ReadString(data, nameof(ScenePath)),
            AcademyNodePath = ReadString(data, nameof(AcademyNodePath)),
            RunId = ReadString(data, nameof(RunId)),
            RunDirectory = ReadString(data, nameof(RunDirectory)),
            TrainingConfigPath = ReadString(data, nameof(TrainingConfigPath)),
            NetworkConfigPath = ReadString(data, nameof(NetworkConfigPath)),
            CheckpointPath = ReadString(data, nameof(CheckpointPath)),
            MetricsPath = ReadString(data, nameof(MetricsPath)),
            StatusPath = ReadString(data, nameof(StatusPath)),
            CheckpointInterval = ReadInt(data, nameof(CheckpointInterval), 10),
            SimulationSpeed = ReadFloat(data, nameof(SimulationSpeed), 1.0f),
            ActionRepeat = ReadInt(data, nameof(ActionRepeat), 1),
            BatchSize = ReadInt(data, nameof(BatchSize), 1),
            QuickTestMode = ReadBool(data, nameof(QuickTestMode)),
            QuickTestEpisodeLimit = ReadInt(data, nameof(QuickTestEpisodeLimit), 5),
            QuickTestShowSpyOverlay = ReadBool(data, nameof(QuickTestShowSpyOverlay)),
            HpoOverridePath = ReadString(data, nameof(HpoOverridePath)),
            HpoMasterHeartbeatPath = ReadString(data, nameof(HpoMasterHeartbeatPath)),
            HpoMasterHeartbeatToken = ReadString(data, nameof(HpoMasterHeartbeatToken)),
        };
    }

    private Godot.Collections.Dictionary ToDictionary()
    {
        return new Godot.Collections.Dictionary
        {
            { nameof(ScenePath), ScenePath },
            { nameof(AcademyNodePath), AcademyNodePath },
            { nameof(RunId), RunId },
            { nameof(RunDirectory), RunDirectory },
            { nameof(TrainingConfigPath), TrainingConfigPath },
            { nameof(NetworkConfigPath), NetworkConfigPath },
            { nameof(CheckpointPath), CheckpointPath },
            { nameof(MetricsPath), MetricsPath },
            { nameof(StatusPath), StatusPath },
            { nameof(CheckpointInterval), CheckpointInterval },
            { nameof(SimulationSpeed), SimulationSpeed },
            { nameof(ActionRepeat), ActionRepeat },
            { nameof(BatchSize), BatchSize },
            { nameof(QuickTestMode), QuickTestMode },
            { nameof(QuickTestEpisodeLimit), QuickTestEpisodeLimit },
            { nameof(QuickTestShowSpyOverlay), QuickTestShowSpyOverlay },
            { nameof(HpoOverridePath), HpoOverridePath },
            { nameof(HpoMasterHeartbeatPath), HpoMasterHeartbeatPath },
            { nameof(HpoMasterHeartbeatToken), HpoMasterHeartbeatToken },
        };
    }

    private static Error EnsureParentDirectory(string filePath)
    {
        var normalizedPath = filePath.Replace('\\', '/');
        var lastSlash = normalizedPath.LastIndexOf('/');
        if (lastSlash < 0)
        {
            return Error.Ok;
        }

        var directoryPath = normalizedPath[..lastSlash];
        return DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(directoryPath));
    }

    private static string ReadString(Godot.Collections.Dictionary dictionary, string key)
    {
        return dictionary.ContainsKey(key) ? dictionary[key].ToString() : string.Empty;
    }

    private static int ReadInt(Godot.Collections.Dictionary dictionary, string key, int defaultValue)
    {
        if (!dictionary.ContainsKey(key))
        {
            return defaultValue;
        }

        var value = dictionary[key];
        return value.VariantType == Variant.Type.Int ? (int)value : defaultValue;
    }

    private static bool ReadBool(Godot.Collections.Dictionary dictionary, string key)
    {
        if (!dictionary.ContainsKey(key))
        {
            return false;
        }

        var value = dictionary[key];
        return value.VariantType == Variant.Type.Bool && (bool)value;
    }

    private static float ReadFloat(Godot.Collections.Dictionary dictionary, string key, float defaultValue)
    {
        if (!dictionary.ContainsKey(key))
        {
            return defaultValue;
        }

        var value = dictionary[key];
        return value.VariantType switch
        {
            Variant.Type.Float => (float)(double)value,
            Variant.Type.Int => (int)value,
            _ => defaultValue,
        };
    }
}
