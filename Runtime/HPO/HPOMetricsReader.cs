using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Reads metrics JSONL files produced by <see cref="RunMetricsWriter"/> to extract
/// an objective value and the latest total-steps count for a running trial.
/// </summary>
public static class HPOMetricsReader
{
    /// <summary>
    /// Returns the objective value and total steps for a trial, or null if the
    /// metrics file does not yet exist or has no usable entries.
    /// </summary>
    public static (float objective, long totalSteps)? TryReadObjective(
        string runDirectory,
        RLHPOStudy study)
    {
        var objectiveConfig = study.ObjectiveConfig;
        if (objectiveConfig is null)
            return null;

        var configuredSources = objectiveConfig.Sources
            .Where(s => s is not null && !string.IsNullOrWhiteSpace(s.PolicyGroup))
            .ToList();
        if (configuredSources.Count == 0)
            return null;

        return TryReadAggregatedObjective(runDirectory, objectiveConfig, configuredSources);
    }

    private static (float objective, long totalSteps)? TryReadAggregatedObjective(
        string runDirectory,
        RLHPOObjectiveConfig objectiveConfig,
        List<RLHPOObjectiveSource> sources)
    {
        var values = new List<(float Objective, long Steps, float Weight)>(sources.Count);
        foreach (var source in sources)
        {
            string? metricsPath = FindMetricsPath(runDirectory, source.PolicyGroup);
            if (metricsPath is null || !FileAccess.FileExists(metricsPath))
                return null;

            var reading = TryReadObjectiveFromMetricsPath(metricsPath, objectiveConfig, source);
            if (!reading.HasValue)
                return null;

            values.Add((reading.Value.objective, reading.Value.totalSteps, Math.Max(0f, source.Weight)));
        }

        if (values.Count == 0)
            return null;

        float objective = objectiveConfig.Aggregation switch
        {
            RLHPOObjectiveAggregation.Min => values.Min(v => v.Objective),
            RLHPOObjectiveAggregation.Max => values.Max(v => v.Objective),
            RLHPOObjectiveAggregation.WeightedMean => WeightedMean(values),
            _ => values.Average(v => v.Objective),
        };

        long totalSteps = values.Max(v => v.Steps);
        return (objective, totalSteps);
    }

    private static float WeightedMean(List<(float Objective, long Steps, float Weight)> values)
    {
        float totalWeight = values.Sum(v => v.Weight);
        if (totalWeight <= 1e-6f)
            return values.Average(v => v.Objective);

        float weightedSum = 0f;
        foreach (var value in values)
            weightedSum += value.Objective * value.Weight;
        return weightedSum / totalWeight;
    }

    private static (float objective, long totalSteps)? TryReadObjectiveFromMetricsPath(
        string metricsPath,
        RLHPOObjectiveConfig objectiveConfig,
        RLHPOObjectiveSource source)
    {
        var entries = ReadLastNEntries(metricsPath, objectiveConfig.EvaluationWindow);
        if (entries.Count == 0)
            return null;

        string jsonlKey = ObjectiveKey(source);
        float sum = 0f;
        int count = 0;
        long maxSteps = 0;

        foreach (var entry in entries)
        {
            maxSteps = Math.Max(maxSteps, ReadLong(entry, "total_steps"));

            if (entry.ContainsKey(jsonlKey)
                && entry[jsonlKey].VariantType is Variant.Type.Float or Variant.Type.Int)
            {
                sum += (float)(double)entry[jsonlKey];
                count++;
            }
        }

        if (count == 0)
            return null;

        return (sum / count, maxSteps);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static string? FindMetricsPath(string runDirectory, string policyGroup)
    {
        if (!string.IsNullOrWhiteSpace(policyGroup))
        {
            string specific = $"{runDirectory}/metrics__{SanitizeGroupId(policyGroup)}.jsonl";
            if (FileAccess.FileExists(specific))
                return specific;
        }

        var absDir = ProjectSettings.GlobalizePath(runDirectory);
        if (!DirAccess.DirExistsAbsolute(absDir))
            return null;

        using var dir = DirAccess.Open(runDirectory);
        if (dir is null)
            return null;

        dir.ListDirBegin();
        string fileName;
        while ((fileName = dir.GetNext()) != "")
        {
            if (!dir.CurrentIsDir() && fileName.StartsWith("metrics__", StringComparison.Ordinal)
                && fileName.EndsWith(".jsonl", StringComparison.Ordinal))
            {
                var candidate = $"{runDirectory}/{fileName}";
                if (string.IsNullOrWhiteSpace(policyGroup) || MetricsFileMatchesPolicyGroup(candidate, policyGroup))
                {
                    dir.ListDirEnd();
                    return candidate;
                }
            }
        }
        dir.ListDirEnd();
        return null;
    }

    private static bool MetricsFileMatchesPolicyGroup(string resPath, string policyGroup)
    {
        using var file = FileAccess.Open(resPath, FileAccess.ModeFlags.Read);
        if (file is null)
            return false;

        int inspected = 0;
        while (!file.EofReached() && inspected < 8)
        {
            var line = file.GetLine().Trim();
            if (string.IsNullOrEmpty(line))
                continue;

            inspected++;
            var parsed = Json.ParseString(line);
            if (parsed.VariantType != Variant.Type.Dictionary)
                continue;

            var entry = parsed.AsGodotDictionary();
            if (entry.ContainsKey("policy_group")
                && string.Equals(entry["policy_group"].ToString(), policyGroup, StringComparison.Ordinal))
                return true;
        }

        return false;
    }

    private static List<Godot.Collections.Dictionary> ReadLastNEntries(string resPath, int n)
    {
        using var file = FileAccess.Open(resPath, FileAccess.ModeFlags.Read);
        if (file is null)
            return new List<Godot.Collections.Dictionary>();

        // Buffer the last n lines
        var lines = new Queue<string>(n + 1);
        while (!file.EofReached())
        {
            var line = file.GetLine().Trim();
            if (string.IsNullOrEmpty(line)) continue;
            if (lines.Count >= n) lines.Dequeue();
            lines.Enqueue(line);
        }

        var result = new List<Godot.Collections.Dictionary>(lines.Count);
        foreach (var line in lines)
        {
            var parsed = Json.ParseString(line);
            if (parsed.VariantType == Variant.Type.Dictionary)
                result.Add(parsed.AsGodotDictionary());
        }
        return result;
    }

    private static string ObjectiveKey(RLHPOObjectiveSource source) => source.Metric switch
    {
        RLHPOObjectiveMetric.MeanEpisodeReward => "episode_reward",
        RLHPOObjectiveMetric.MeanEpisodeLength => "episode_length",
        RLHPOObjectiveMetric.PolicyLoss        => "policy_loss",
        RLHPOObjectiveMetric.ValueLoss         => "value_loss",
        RLHPOObjectiveMetric.Custom            => source.CustomMetricKey,
        _                                       => "episode_reward",
    };

    private static string SanitizeGroupId(string groupId) =>
        System.Text.RegularExpressions.Regex.Replace(groupId, @"[^a-zA-Z0-9_\-]", "_");

    private static long ReadLong(Godot.Collections.Dictionary entry, string key)
    {
        if (!entry.ContainsKey(key))
            return 0L;

        var value = entry[key];
        return value.VariantType switch
        {
            Variant.Type.Int => value.AsInt64(),
            Variant.Type.Float => (long)value.AsDouble(),
            _ => 0L,
        };
    }
}
