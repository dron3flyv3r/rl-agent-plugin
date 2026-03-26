using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLCheckpoint : Resource
{
    public const int CurrentFormatVersion = 4;
    public const string PpoAlgorithm = "PPO";
    public const string SacAlgorithm = "SAC";

    [Export] public int FormatVersion { get; set; } = CurrentFormatVersion;
    [Export] public string RunId { get; set; } = string.Empty;
    [Export] public long TotalSteps { get; set; }
    [Export] public long EpisodeCount { get; set; }
    [Export] public long UpdateCount { get; set; }
    [Export] public float RewardSnapshot { get; set; }
    [Export] public string Algorithm { get; set; } = PpoAlgorithm;
    [Export] public int ObservationSize { get; set; }
    [Export] public int DiscreteActionCount { get; set; }
    [Export] public int ContinuousActionDimensions { get; set; }
    [Export] public float[] WeightBuffer { get; set; } = Array.Empty<float>();
    [Export] public int[] LayerShapeBuffer { get; set; } = Array.Empty<int>();

    public List<RLCheckpointLayer> NetworkLayers { get; set; } = new();
    public string NetworkOptimizer { get; set; } = "adam";
    public Dictionary<string, string[]> DiscreteActionLabels { get; set; } = new(StringComparer.Ordinal);
    public Dictionary<string, RLContinuousActionRange> ContinuousActionRanges { get; set; } = new(StringComparer.Ordinal);
    public Dictionary<string, float> Hyperparams { get; set; } = new(StringComparer.Ordinal);

    /// <summary>
    /// Serializes the checkpoint to a JSON file at the given Godot path (supports user://).
    /// More reliable than ResourceSaver.Save() for programmatically-created C# resources.
    /// </summary>
    public static Error SaveToFile(RLCheckpoint checkpoint, string path)
    {
        checkpoint.FormatVersion = CurrentFormatVersion;

        var dir = path.GetBaseDir();
        if (!string.IsNullOrEmpty(dir))
            DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(dir));

        using var file = FileAccess.Open(path, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            var err = FileAccess.GetOpenError();
            GD.PushError($"[RLCheckpoint] Failed to open '{path}' for writing: {err}");
            return err;
        }

        var weightArray = new Godot.Collections.Array();
        foreach (var w in checkpoint.WeightBuffer) weightArray.Add(Variant.From(w));

        var shapeArray = new Godot.Collections.Array();
        foreach (var s in checkpoint.LayerShapeBuffer) shapeArray.Add(Variant.From(s));

        var training = new Godot.Collections.Dictionary
        {
            { "total_steps",   checkpoint.TotalSteps   },
            { "episode_count", checkpoint.EpisodeCount },
            { "update_count",  checkpoint.UpdateCount  },
        };

        var data = new Godot.Collections.Dictionary
        {
            { "format_version", checkpoint.FormatVersion },
            { "run_id",         checkpoint.RunId         },
            { "training",       training                 },
            { "meta",           checkpoint.CreateMetadataDictionary() },
            { "weights",        weightArray              },
            { "shapes",         shapeArray               },
        };

        file.StoreString(Json.Stringify(data));
        return Error.Ok;
    }

    /// <summary>
    /// Loads a checkpoint previously saved with SaveToFile().
    /// </summary>
    public static RLCheckpoint? LoadFromFile(string path)
    {
        var resolvedPath = ResolvePath(path);
        if (!FileAccess.FileExists(resolvedPath))
        {
            GD.PushWarning($"[RLCheckpoint] File not found: {resolvedPath}");
            return null;
        }

        using var file = FileAccess.Open(resolvedPath, FileAccess.ModeFlags.Read);
        if (file is null)
        {
            GD.PushError($"[RLCheckpoint] Failed to open '{resolvedPath}' for reading: {FileAccess.GetOpenError()}");
            return null;
        }

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary)
        {
            GD.PushError($"[RLCheckpoint] Invalid JSON format in '{resolvedPath}'");
            return null;
        }

        var data = parsed.AsGodotDictionary();
        var weightArr = data.ContainsKey("weights") ? data["weights"].AsGodotArray() : new Godot.Collections.Array();
        var shapeArr  = data.ContainsKey("shapes")  ? data["shapes"].AsGodotArray()  : new Godot.Collections.Array();

        var weights = new float[weightArr.Count];
        for (var i = 0; i < weightArr.Count; i++) weights[i] = weightArr[i].AsSingle();

        var shapes = new int[shapeArr.Count];
        for (var i = 0; i < shapeArr.Count; i++) shapes[i] = (int)shapeArr[i].AsInt64();

        var checkpoint = new RLCheckpoint
        {
            FormatVersion    = data.ContainsKey("format_version") ? (int)data["format_version"].AsInt64() : 1,
            RunId            = data.ContainsKey("run_id")         ? data["run_id"].AsString()             : string.Empty,
            WeightBuffer     = weights,
            LayerShapeBuffer = shapes,
        };

        ReadTrainingStats(data, checkpoint);

        if (checkpoint.FormatVersion >= 2
            && data.ContainsKey("meta")
            && data["meta"].VariantType == Variant.Type.Dictionary)
        {
            checkpoint.ApplyMetadataDictionary(data["meta"].AsGodotDictionary());
        }
        else
        {
            checkpoint.PopulateLegacyMetadata();
        }

        return checkpoint;
    }

    /// <summary>
    /// Loads only the header and metadata fields from a checkpoint JSON file.
    /// WeightBuffer and LayerShapeBuffer will be empty — suitable for fast
    /// dashboard display without allocating large weight arrays.
    /// Returns null if the file cannot be read or parsed.
    /// </summary>
    public static RLCheckpoint? LoadMetadataOnly(string path)
    {
        var resolvedPath = ResolvePath(path);
        if (!FileAccess.FileExists(resolvedPath)) return null;

        try
        {
            using var file = FileAccess.Open(resolvedPath, FileAccess.ModeFlags.Read);
            if (file is null) return null;

            var parsed = Json.ParseString(file.GetAsText());
            if (parsed.VariantType != Variant.Type.Dictionary) return null;

            var data = parsed.AsGodotDictionary();
            var checkpoint = new RLCheckpoint
            {
                FormatVersion    = data.ContainsKey("format_version") ? (int)data["format_version"].AsInt64() : 1,
                RunId            = data.ContainsKey("run_id")         ? data["run_id"].AsString()             : string.Empty,
                WeightBuffer     = Array.Empty<float>(),
                LayerShapeBuffer = Array.Empty<int>(),
            };

            ReadTrainingStats(data, checkpoint);

            if (checkpoint.FormatVersion >= 2
                && data.ContainsKey("meta")
                && data["meta"].VariantType == Variant.Type.Dictionary)
            {
                checkpoint.ApplyMetadataDictionary(data["meta"].AsGodotDictionary());
            }
            else
            {
                checkpoint.PopulateLegacyMetadata();
            }

            return checkpoint;
        }
        catch
        {
            return null;
        }
    }

    public void PopulateLegacyMetadata()
    {
        FormatVersion          = 1;
        Algorithm              = PpoAlgorithm;
        ObservationSize        = LayerShapeBuffer.Length >= 3 ? LayerShapeBuffer[0] : 0;
        DiscreteActionCount    = DeriveLegacyDiscreteActionCount(LayerShapeBuffer);
        ContinuousActionDimensions = 0;
        NetworkLayers          = new List<RLCheckpointLayer>();
        NetworkOptimizer       = "adam";
        DiscreteActionLabels   = new Dictionary<string, string[]>(StringComparer.Ordinal);
        ContinuousActionRanges = new Dictionary<string, RLContinuousActionRange>(StringComparer.Ordinal);
        Hyperparams            = new Dictionary<string, float>(StringComparer.Ordinal);
    }

    internal string CreateMetadataJson()
    {
        return Json.Stringify(CreateMetadataDictionary());
    }

    internal void ApplyMetadataJson(string json)
    {
        var parsed = Json.ParseString(json);
        if (parsed.VariantType != Variant.Type.Dictionary)
            throw new InvalidOperationException("Checkpoint metadata JSON is not a dictionary.");

        ApplyMetadataDictionary(parsed.AsGodotDictionary());
    }

    internal Godot.Collections.Dictionary CreateMetadataDictionary()
    {
        // ── Network ──────────────────────────────────────────────────────────
        var layersArray = new Godot.Collections.Array();
        foreach (var layer in NetworkLayers)
        {
            var layerDict = new Godot.Collections.Dictionary { { "type", layer.Type } };
            switch (layer.Type)
            {
                case "dense":
                    layerDict["size"]       = layer.Size;
                    layerDict["activation"] = layer.Activation;
                    break;
                case "dropout":
                    layerDict["rate"] = layer.Rate;
                    break;
                // layer_norm and flatten carry no extra properties
            }
            layersArray.Add(layerDict);
        }

        var networkDict = new Godot.Collections.Dictionary
        {
            { "optimizer", NetworkOptimizer },
            { "layers",    layersArray      },
        };

        // ── Action space ─────────────────────────────────────────────────────
        var discreteLabels = new Godot.Collections.Dictionary();
        foreach (var (key, labels) in DiscreteActionLabels)
        {
            var labelArray = new Godot.Collections.Array();
            foreach (var label in labels) labelArray.Add(label);
            discreteLabels[key] = labelArray;
        }

        var continuousRanges = new Godot.Collections.Dictionary();
        foreach (var (key, range) in ContinuousActionRanges)
        {
            continuousRanges[key] = new Godot.Collections.Dictionary
            {
                { "dims", range.Dimensions },
                { "min",  range.Min        },
                { "max",  range.Max        },
            };
        }

        var actionSpaceDict = new Godot.Collections.Dictionary
        {
            {
                "discrete", new Godot.Collections.Dictionary
                {
                    { "count",  DiscreteActionCount },
                    { "labels", discreteLabels      },
                }
            },
            {
                "continuous", new Godot.Collections.Dictionary
                {
                    { "dims",   ContinuousActionDimensions },
                    { "ranges", continuousRanges           },
                }
            },
        };

        // ── Hyperparams ───────────────────────────────────────────────────────
        var hyperparams = new Godot.Collections.Dictionary();
        foreach (var (key, value) in Hyperparams) hyperparams[key] = value;

        return new Godot.Collections.Dictionary
        {
            { "algorithm",       Algorithm       },
            { "obs_size",        ObservationSize },
            { "reward_snapshot", RewardSnapshot  },
            { "network",         networkDict     },
            { "action_space",    actionSpaceDict },
            { "hyperparams",     hyperparams     },
        };
    }

    internal void ApplyMetadataDictionary(Godot.Collections.Dictionary metadata)
    {
        FormatVersion   = CurrentFormatVersion;
        Algorithm       = metadata.ContainsKey("algorithm")       ? metadata["algorithm"].AsString()       : PpoAlgorithm;
        ObservationSize = metadata.ContainsKey("obs_size")        ? (int)metadata["obs_size"].AsInt64()    : 0;
        RewardSnapshot  = metadata.ContainsKey("reward_snapshot") ? metadata["reward_snapshot"].AsSingle() : 0f;

        // ── Network (v4) ──────────────────────────────────────────────────────
        if (metadata.ContainsKey("network") && metadata["network"].VariantType == Variant.Type.Dictionary)
        {
            var network = metadata["network"].AsGodotDictionary();
            NetworkOptimizer = network.ContainsKey("optimizer") ? network["optimizer"].AsString() : "adam";
            NetworkLayers    = new List<RLCheckpointLayer>();

            if (network.ContainsKey("layers") && network["layers"].VariantType == Variant.Type.Array)
            {
                foreach (var layerVariant in network["layers"].AsGodotArray())
                {
                    if (layerVariant.VariantType != Variant.Type.Dictionary) continue;
                    var ld        = layerVariant.AsGodotDictionary();
                    var layerType = ld.ContainsKey("type") ? ld["type"].AsString() : "dense";
                    var layer     = new RLCheckpointLayer { Type = layerType };

                    switch (layerType)
                    {
                        case "dense":
                            layer.Size       = ld.ContainsKey("size")       ? (int)ld["size"].AsInt64()     : 0;
                            layer.Activation = ld.ContainsKey("activation") ? ld["activation"].AsString()   : "tanh";
                            break;
                        case "dropout":
                            layer.Rate = ld.ContainsKey("rate") ? ld["rate"].AsSingle() : 0.1f;
                            break;
                    }

                    NetworkLayers.Add(layer);
                }
            }
        }
        // ── Legacy network (v3): parallel int arrays ───────────────────────
        else if (metadata.ContainsKey("graph_layer_sizes"))
        {
            NetworkOptimizer = metadata.ContainsKey("graph_optimizer")
                ? OptimizerIntToString((int)metadata["graph_optimizer"].AsInt64())
                : "adam";

            var sizes       = ReadIntArray(metadata["graph_layer_sizes"].AsGodotArray());
            var activations = metadata.ContainsKey("graph_layer_activations")
                ? ReadIntArray(metadata["graph_layer_activations"].AsGodotArray())
                : Array.Empty<int>();

            NetworkLayers = new List<RLCheckpointLayer>(sizes.Length);
            for (var i = 0; i < sizes.Length; i++)
            {
                // size=0 / activation=-1 were sentinels for non-Dense layers in v3.
                // We can't reconstruct their exact type so they are skipped.
                if (sizes[i] <= 0) continue;
                var activation = activations.Length > i ? activations[i] : 0;
                if (activation < 0) continue;

                NetworkLayers.Add(new RLCheckpointLayer
                {
                    Type       = "dense",
                    Size       = sizes[i],
                    Activation = ActivationIntToString(activation),
                });
            }
        }
        // ── Very old (v1/v2): hidden_layer_sizes only ──────────────────────
        else if (metadata.ContainsKey("hidden_layer_sizes"))
        {
            NetworkOptimizer = "adam";
            var sizes = ReadIntArray(metadata["hidden_layer_sizes"].AsGodotArray());
            NetworkLayers = new List<RLCheckpointLayer>(sizes.Length);
            foreach (var size in sizes)
                NetworkLayers.Add(new RLCheckpointLayer { Type = "dense", Size = size, Activation = "tanh" });
        }
        else
        {
            NetworkOptimizer = "adam";
            NetworkLayers    = new List<RLCheckpointLayer>();
        }

        // ── Action space (v4) ─────────────────────────────────────────────────
        if (metadata.ContainsKey("action_space") && metadata["action_space"].VariantType == Variant.Type.Dictionary)
        {
            var actionSpace = metadata["action_space"].AsGodotDictionary();
            ReadDiscreteActionSpace(actionSpace);
            ReadContinuousActionSpace(actionSpace);
        }
        // ── Legacy action space (v3): flat fields ─────────────────────────────
        else
        {
            DiscreteActionCount    = metadata.ContainsKey("discrete_action_count")    ? (int)metadata["discrete_action_count"].AsInt64()    : 0;
            ContinuousActionDimensions = metadata.ContainsKey("continuous_action_dims") ? (int)metadata["continuous_action_dims"].AsInt64() : 0;

            DiscreteActionLabels = new Dictionary<string, string[]>(StringComparer.Ordinal);
            if (metadata.ContainsKey("discrete_action_labels") && metadata["discrete_action_labels"].VariantType == Variant.Type.Dictionary)
            {
                foreach (var key in metadata["discrete_action_labels"].AsGodotDictionary().Keys)
                    DiscreteActionLabels[key.AsString()] = ReadStringArray(metadata["discrete_action_labels"].AsGodotDictionary()[key].AsGodotArray());
            }

            ContinuousActionRanges = new Dictionary<string, RLContinuousActionRange>(StringComparer.Ordinal);
            if (metadata.ContainsKey("continuous_action_ranges") && metadata["continuous_action_ranges"].VariantType == Variant.Type.Dictionary)
            {
                foreach (var key in metadata["continuous_action_ranges"].AsGodotDictionary().Keys)
                    TryAddContinuousRange(ContinuousActionRanges, key.AsString(), metadata["continuous_action_ranges"].AsGodotDictionary()[key]);
            }
        }

        // ── Hyperparams (unchanged across versions) ────────────────────────────
        Hyperparams = new Dictionary<string, float>(StringComparer.Ordinal);
        if (metadata.ContainsKey("hyperparams") && metadata["hyperparams"].VariantType == Variant.Type.Dictionary)
        {
            foreach (var key in metadata["hyperparams"].AsGodotDictionary().Keys)
                Hyperparams[key.AsString()] = metadata["hyperparams"].AsGodotDictionary()[key].AsSingle();
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static void ReadTrainingStats(Godot.Collections.Dictionary data, RLCheckpoint checkpoint)
    {
        // v4: training sub-object
        if (data.ContainsKey("training") && data["training"].VariantType == Variant.Type.Dictionary)
        {
            var t = data["training"].AsGodotDictionary();
            checkpoint.TotalSteps   = t.ContainsKey("total_steps")   ? t["total_steps"].AsInt64()   : 0;
            checkpoint.EpisodeCount = t.ContainsKey("episode_count") ? t["episode_count"].AsInt64() : 0;
            checkpoint.UpdateCount  = t.ContainsKey("update_count")  ? t["update_count"].AsInt64()  : 0;
        }
        else
        {
            // v3: flat root fields
            checkpoint.TotalSteps   = data.ContainsKey("total_steps")   ? data["total_steps"].AsInt64()   : 0;
            checkpoint.EpisodeCount = data.ContainsKey("episode_count") ? data["episode_count"].AsInt64() : 0;
            checkpoint.UpdateCount  = data.ContainsKey("update_count")  ? data["update_count"].AsInt64()  : 0;
        }
    }

    private void ReadDiscreteActionSpace(Godot.Collections.Dictionary actionSpace)
    {
        DiscreteActionLabels = new Dictionary<string, string[]>(StringComparer.Ordinal);

        if (!actionSpace.ContainsKey("discrete") || actionSpace["discrete"].VariantType != Variant.Type.Dictionary)
        {
            DiscreteActionCount = 0;
            return;
        }

        var discrete = actionSpace["discrete"].AsGodotDictionary();
        DiscreteActionCount = discrete.ContainsKey("count") ? (int)discrete["count"].AsInt64() : 0;

        if (discrete.ContainsKey("labels") && discrete["labels"].VariantType == Variant.Type.Dictionary)
        {
            foreach (var key in discrete["labels"].AsGodotDictionary().Keys)
                DiscreteActionLabels[key.AsString()] = ReadStringArray(discrete["labels"].AsGodotDictionary()[key].AsGodotArray());
        }
    }

    private void ReadContinuousActionSpace(Godot.Collections.Dictionary actionSpace)
    {
        ContinuousActionRanges = new Dictionary<string, RLContinuousActionRange>(StringComparer.Ordinal);

        if (!actionSpace.ContainsKey("continuous") || actionSpace["continuous"].VariantType != Variant.Type.Dictionary)
        {
            ContinuousActionDimensions = 0;
            return;
        }

        var continuous = actionSpace["continuous"].AsGodotDictionary();
        ContinuousActionDimensions = continuous.ContainsKey("dims") ? (int)continuous["dims"].AsInt64() : 0;

        if (continuous.ContainsKey("ranges") && continuous["ranges"].VariantType == Variant.Type.Dictionary)
        {
            foreach (var key in continuous["ranges"].AsGodotDictionary().Keys)
                TryAddContinuousRange(ContinuousActionRanges, key.AsString(), continuous["ranges"].AsGodotDictionary()[key]);
        }
    }

    private static void TryAddContinuousRange(Dictionary<string, RLContinuousActionRange> target, string name, Variant rangeVariant)
    {
        if (rangeVariant.VariantType != Variant.Type.Dictionary) return;
        var d = rangeVariant.AsGodotDictionary();
        target[name] = new RLContinuousActionRange
        {
            Dimensions = d.ContainsKey("dims") ? (int)d["dims"].AsInt64() : 0,
            Min        = d.ContainsKey("min")  ? d["min"].AsSingle()      : -1f,
            Max        = d.ContainsKey("max")  ? d["max"].AsSingle()      : 1f,
        };
    }

    private static string ActivationIntToString(int activation) => activation switch
    {
        (int)RLActivationKind.Relu => "relu",
        _                          => "tanh",
    };

    private static string OptimizerIntToString(int optimizer) => optimizer switch
    {
        (int)RLOptimizerKind.Sgd  => "sgd",
        (int)RLOptimizerKind.None => "none",
        _                          => "adam",
    };

    private static string ResolvePath(string path)
    {
        return (path.StartsWith("res://", StringComparison.Ordinal) || path.StartsWith("user://", StringComparison.Ordinal))
            ? ProjectSettings.GlobalizePath(path)
            : path;
    }

    private static int DeriveLegacyDiscreteActionCount(IReadOnlyList<int> layerShapes)
    {
        if (layerShapes.Count < 6 || layerShapes.Count % 3 != 0) return 0;
        var layerCount = layerShapes.Count / 3;
        return layerShapes[(layerCount - 2) * 3 + 1];
    }

    private static int[] ReadIntArray(Godot.Collections.Array values)
    {
        var result = new int[values.Count];
        for (var i = 0; i < values.Count; i++) result[i] = (int)values[i].AsInt64();
        return result;
    }

    private static string[] ReadStringArray(Godot.Collections.Array values)
    {
        var result = new string[values.Count];
        for (var i = 0; i < values.Count; i++) result[i] = values[i].AsString();
        return result;
    }
}
