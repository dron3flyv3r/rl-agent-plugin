using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Lightweight metadata record for a single history checkpoint entry.
/// No weights are loaded.
/// </summary>
public sealed class CheckpointHistoryEntry
{
    public string AbsolutePath     { get; init; } = string.Empty;
    public string PolicyGroupId    { get; init; } = string.Empty;
    public long   UpdateCount      { get; init; }
    public long   TotalSteps       { get; init; }
    public long   EpisodeCount     { get; init; }
    public string Algorithm        { get; init; } = string.Empty;
    public float  RewardSnapshot   { get; init; }
    public bool   IsSelfPlayFrozen { get; init; }
}

public static class CheckpointRegistry
{
    private const string RunsRoot = "res://RL-Agent-Training/runs";

    public static List<string> ListCheckpointPaths()
    {
        var results = new List<string>();
        var runsDir = DirAccess.Open(RunsRoot);
        if (runsDir is null)
        {
            return results;
        }

        runsDir.ListDirBegin();
        while (true)
        {
            var name = runsDir.GetNext();
            if (string.IsNullOrEmpty(name))
            {
                break;
            }

            if (!runsDir.CurrentIsDir() || name.StartsWith("."))
            {
                continue;
            }

            foreach (var checkpointPath in ListRunCheckpoints($"{RunsRoot}/{name}"))
            {
                results.Add(checkpointPath);
            }
        }

        runsDir.ListDirEnd();
        results.Sort((left, right) => string.CompareOrdinal(right, left));
        return results;
    }

    private static IEnumerable<string> ListRunCheckpoints(string runDirectory)
    {
        var runDir = DirAccess.Open(runDirectory);
        if (runDir is null)
        {
            yield break;
        }

        runDir.ListDirBegin();
        while (true)
        {
            var entryName = runDir.GetNext();
            if (string.IsNullOrEmpty(entryName))
            {
                break;
            }

            if (runDir.CurrentIsDir() || entryName.StartsWith(".") || !entryName.EndsWith(".json"))
            {
                continue;
            }

            yield return $"{runDirectory}/{entryName}";
        }

        runDir.ListDirEnd();
    }

    public static string GetLatestCheckpointPath(string? groupId = null)
    {
        var checkpoints = ListCheckpointPaths();
        if (checkpoints.Count == 0)
        {
            return string.Empty;
        }

        if (string.IsNullOrWhiteSpace(groupId))
        {
            return checkpoints[0];
        }

        var safeGroupId = RLPolicyGroupBindingResolver.MakeSafeGroupId(groupId);
        var fileName = $"checkpoint__{safeGroupId}.json";
        foreach (var checkpointPath in checkpoints)
        {
            if (string.Equals(System.IO.Path.GetFileName(checkpointPath), fileName, StringComparison.OrdinalIgnoreCase))
            {
                return checkpointPath;
            }
        }

        return string.Empty;
    }

    public static string ResolveCheckpointPath(string preferredPath, string? groupId = null)
    {
        if (!string.IsNullOrWhiteSpace(preferredPath) && FileAccess.FileExists(preferredPath))
        {
            return preferredPath;
        }

        return GetLatestCheckpointPath(groupId);
    }

    /// <summary>
    /// Lists all history checkpoint entries for the given run directory (absolute path).
    /// Reads sidecar .meta.json files where available (fast path); falls back to
    /// LoadMetadataOnly() for full checkpoint JSONs without a sidecar (older runs).
    /// Also includes self-play frozen snapshots from selfplay/ subdirectories.
    /// Results are sorted by (PolicyGroupId, UpdateCount) ascending.
    /// </summary>
    public static List<CheckpointHistoryEntry> ListHistoryEntries(string runDirAbsPath)
    {
        var results = new List<CheckpointHistoryEntry>();

        // Regular history snapshots from history/
        var historyDir = System.IO.Path.Combine(runDirAbsPath, "history");
        if (System.IO.Directory.Exists(historyDir))
        {
            foreach (var file in System.IO.Directory.GetFiles(historyDir, "*.meta.json"))
            {
                var entry = ParseMetaFile(file, isSelfPlayFrozen: false);
                if (entry is not null) results.Add(entry);
            }

            // Fallback for full checkpoint JSONs that have no sidecar (older runs / manual copies)
            foreach (var file in System.IO.Directory.GetFiles(historyDir, "checkpoint__*.json"))
            {
                var sidecar = file.Replace(".json", ".meta.json", StringComparison.Ordinal);
                if (System.IO.File.Exists(sidecar)) continue; // already handled via sidecar
                var entry = ParseFullCheckpoint(file, isSelfPlayFrozen: false, overrideGroupId: null);
                if (entry is not null) results.Add(entry);
            }
        }

        // Self-play frozen snapshots from selfplay/*/
        var selfplayDir = System.IO.Path.Combine(runDirAbsPath, "selfplay");
        if (System.IO.Directory.Exists(selfplayDir))
        {
            foreach (var groupDir in System.IO.Directory.GetDirectories(selfplayDir))
            {
                var groupId = System.IO.Path.GetFileName(groupDir);
                foreach (var file in System.IO.Directory.GetFiles(groupDir, "opponent__u*.json"))
                {
                    var entry = ParseFullCheckpoint(file, isSelfPlayFrozen: true, overrideGroupId: groupId);
                    if (entry is not null) results.Add(entry);
                }
            }
        }

        results.Sort((a, b) =>
        {
            var groupCmp = string.Compare(a.PolicyGroupId, b.PolicyGroupId, StringComparison.Ordinal);
            return groupCmp != 0 ? groupCmp : a.UpdateCount.CompareTo(b.UpdateCount);
        });
        return results;
    }

    private static CheckpointHistoryEntry? ParseMetaFile(string absPath, bool isSelfPlayFrozen)
    {
        try
        {
            var content = System.IO.File.ReadAllText(absPath);
            var parsed  = Json.ParseString(content);
            if (parsed.VariantType != Variant.Type.Dictionary) return null;

            var d       = parsed.AsGodotDictionary();
            var groupId = ExtractGroupIdFromPath(absPath);

            // v4: training sub-object; v3: flat root fields
            long updateCount, totalSteps, episodeCount;
            if (d.ContainsKey("training") && d["training"].VariantType == Variant.Type.Dictionary)
            {
                var t        = d["training"].AsGodotDictionary();
                totalSteps   = t.ContainsKey("total_steps")   ? t["total_steps"].AsInt64()   : 0L;
                episodeCount = t.ContainsKey("episode_count") ? t["episode_count"].AsInt64() : 0L;
                updateCount  = t.ContainsKey("update_count")  ? t["update_count"].AsInt64()  : 0L;
            }
            else
            {
                totalSteps   = d.ContainsKey("total_steps")   ? d["total_steps"].AsInt64()   : 0L;
                episodeCount = d.ContainsKey("episode_count") ? d["episode_count"].AsInt64() : 0L;
                updateCount  = d.ContainsKey("update_count")  ? d["update_count"].AsInt64()  : 0L;
            }
            var algorithm    = RLCheckpoint.PpoAlgorithm;
            var reward       = 0f;

            if (d.ContainsKey("meta") && d["meta"].VariantType == Variant.Type.Dictionary)
            {
                var meta  = d["meta"].AsGodotDictionary();
                algorithm = meta.ContainsKey("algorithm")       ? meta["algorithm"].AsString()       : RLCheckpoint.PpoAlgorithm;
                reward    = meta.ContainsKey("reward_snapshot") ? meta["reward_snapshot"].AsSingle() : 0f;
            }

            return new CheckpointHistoryEntry
            {
                AbsolutePath     = absPath.Replace(".meta.json", ".json", StringComparison.Ordinal),
                PolicyGroupId    = groupId,
                UpdateCount      = updateCount,
                TotalSteps       = totalSteps,
                EpisodeCount     = episodeCount,
                Algorithm        = algorithm,
                RewardSnapshot   = reward,
                IsSelfPlayFrozen = isSelfPlayFrozen,
            };
        }
        catch
        {
            return null;
        }
    }

    private static CheckpointHistoryEntry? ParseFullCheckpoint(
        string absPath,
        bool isSelfPlayFrozen,
        string? overrideGroupId)
    {
        try
        {
            var cp = RLCheckpoint.LoadMetadataOnly(absPath);
            if (cp is null) return null;

            return new CheckpointHistoryEntry
            {
                AbsolutePath     = absPath,
                PolicyGroupId    = overrideGroupId ?? ExtractGroupIdFromPath(absPath),
                UpdateCount      = cp.UpdateCount,
                TotalSteps       = cp.TotalSteps,
                EpisodeCount     = cp.EpisodeCount,
                Algorithm        = cp.Algorithm,
                RewardSnapshot   = cp.RewardSnapshot,
                IsSelfPlayFrozen = isSelfPlayFrozen,
            };
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Extracts the safe group id from filenames like:
    ///   checkpoint__{groupId}__u000010.json
    ///   checkpoint__{groupId}__u000010.meta.json
    /// Returns "unknown" if the pattern does not match.
    /// </summary>
    private static string ExtractGroupIdFromPath(string absPath)
    {
        var name = System.IO.Path.GetFileNameWithoutExtension(absPath);
        // Strip second extension (.meta stays after GetFileNameWithoutExtension on .meta.json)
        if (name.EndsWith(".meta", StringComparison.Ordinal))
            name = name[..^5];

        const string prefix = "checkpoint__";
        if (!name.StartsWith(prefix, StringComparison.Ordinal)) return "unknown";

        var body    = name[prefix.Length..];
        var lastSep = body.LastIndexOf("__", StringComparison.Ordinal);
        return lastSep > 0 ? body[..lastSep] : body;
    }
}
