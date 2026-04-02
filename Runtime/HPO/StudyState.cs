using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Persisted record of a single HPO trial. Serialised to / from JSON.
/// </summary>
public sealed class TrialRecord
{
    public int TrialId { get; set; }
    public Dictionary<string, float> Parameters { get; set; } = new();
    public RLHPOTrialState State { get; set; }
    public float? ObjectiveValue { get; set; }
    public string RunId { get; set; } = "";
    public string RunDirectory { get; set; } = "";
    public long TotalSteps { get; set; }
    public long StartedAtTicks { get; set; }
    public long CompletedAtTicks { get; set; }
}

/// <summary>
/// Full persistent state for an <see cref="RLHPOStudy"/>. Written to
/// <c>res://RL-Agent-Training/hpo/{StudyName}/study_state.json</c> after every trial.
/// </summary>
public sealed class StudyState
{
    private static readonly Regex InvalidObjectiveNumberRegex = new(
        @"(?<prefix>""(?:best_objective|objective)""\s*:\s*)(?<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?inf(?:inity)?|nan)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public string StudyName { get; set; } = "";
    public string OwnerRunId { get; set; } = "";
    public RLHPODirection Direction { get; set; }
    public List<TrialRecord> Trials { get; set; } = new();
    public int? BestTrialId { get; set; }
    public float? BestObjectiveValue { get; set; }

    // ── Mutation ──────────────────────────────────────────────────────────

    /// <summary>Recomputes <see cref="BestTrialId"/> from completed trials.</summary>
    public void UpdateBest()
    {
        BestTrialId = null;
        BestObjectiveValue = null;

        foreach (var trial in Trials)
        {
            if (trial.State != RLHPOTrialState.Complete || trial.ObjectiveValue is null)
                continue;

            var value = trial.ObjectiveValue.Value;
            if (float.IsNaN(value) || float.IsInfinity(value))
                continue;
            var isBetter = BestObjectiveValue is null
                || (Direction == RLHPODirection.Maximize && value > BestObjectiveValue.Value)
                || (Direction == RLHPODirection.Minimize && value < BestObjectiveValue.Value);

            if (isBetter)
            {
                BestObjectiveValue = value;
                BestTrialId = trial.TrialId;
            }
        }
    }

    /// <summary>
    /// Converts interrupted pending/running trials from a previous editor session
    /// into failed records so a resumed study can continue scheduling new trials.
    /// </summary>
    public bool RecoverInterruptedTrials()
    {
        bool changed = false;
        long now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        foreach (var trial in Trials)
        {
            if (trial.State is not (RLHPOTrialState.Pending or RLHPOTrialState.Running))
                continue;

            trial.State = RLHPOTrialState.Failed;
            if (trial.CompletedAtTicks == 0)
                trial.CompletedAtTicks = now;
            changed = true;
        }

        if (changed)
            UpdateBest();

        return changed;
    }

    // ── Serialisation ─────────────────────────────────────────────────────

    public void SaveToResPath(string resPath)
    {
        var absDir = System.IO.Path.GetDirectoryName(ProjectSettings.GlobalizePath(resPath));
        if (!string.IsNullOrEmpty(absDir))
            DirAccess.MakeDirRecursiveAbsolute(absDir);

        using var file = FileAccess.Open(resPath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            GD.PushWarning($"[HPO] Could not write study state to '{resPath}'.");
            return;
        }
        file.StoreString(Json.Stringify(ToGodotDictionary(), "\t"));
    }

    public static StudyState LoadOrCreate(string resPath, RLHPOStudy study)
    {
        if (FileAccess.FileExists(resPath))
        {
            using var file = FileAccess.Open(resPath, FileAccess.ModeFlags.Read);
            if (file is not null)
            {
                var parsed = ParseSanitizedJson(file.GetAsText());
                if (parsed.VariantType == Variant.Type.Dictionary)
                {
                    var loaded = FromGodotDictionary(parsed.AsGodotDictionary());
                    if (loaded is not null)
                        return loaded;
                }
            }
        }

        return new StudyState { StudyName = study.StudyName, Direction = study.Direction };
    }

    // ── Godot Dictionary serialisation ────────────────────────────────────

    private Godot.Collections.Dictionary ToGodotDictionary()
    {
        var trialsArray = new Godot.Collections.Array();
        foreach (var t in Trials)
            trialsArray.Add(TrialToDict(t));

        var d = new Godot.Collections.Dictionary
        {
            { "study_name",           StudyName },
            { "owner_run_id",         OwnerRunId },
            { "direction",            Direction.ToString() },
            { "trials",               trialsArray },
        };
        if (BestTrialId.HasValue)   d["best_trial_id"]       = BestTrialId.Value;
        if (BestObjectiveValue.HasValue && !float.IsNaN(BestObjectiveValue.Value) && !float.IsInfinity(BestObjectiveValue.Value))
            d["best_objective"] = BestObjectiveValue.Value;
        return d;
    }

    private static Godot.Collections.Dictionary TrialToDict(TrialRecord t)
    {
        var paramsDict = new Godot.Collections.Dictionary();
        foreach (var (k, v) in t.Parameters)
            paramsDict[k] = v;

        var d = new Godot.Collections.Dictionary
        {
            { "trial_id",    t.TrialId },
            { "state",       t.State.ToString() },
            { "run_id",      t.RunId },
            { "run_dir",     t.RunDirectory },
            { "total_steps", t.TotalSteps },
            { "started_at",  t.StartedAtTicks },
            { "completed_at",t.CompletedAtTicks },
            { "params",      paramsDict },
        };
        if (t.ObjectiveValue.HasValue && !float.IsNaN(t.ObjectiveValue.Value) && !float.IsInfinity(t.ObjectiveValue.Value))
            d["objective"] = t.ObjectiveValue.Value;
        return d;
    }

    private static StudyState? FromGodotDictionary(Godot.Collections.Dictionary d)
    {
        var state = new StudyState
        {
            StudyName = d.ContainsKey("study_name") ? d["study_name"].ToString() : "",
            OwnerRunId = d.ContainsKey("owner_run_id") ? d["owner_run_id"].ToString() : "",
            Direction = d.ContainsKey("direction") && d["direction"].ToString() == "Minimize"
                            ? RLHPODirection.Minimize
                            : RLHPODirection.Maximize,
        };

        if (d.ContainsKey("best_trial_id") && d["best_trial_id"].VariantType == Variant.Type.Int)
            state.BestTrialId = (int)d["best_trial_id"];
        if (d.ContainsKey("best_objective") && d["best_objective"].VariantType is Variant.Type.Float or Variant.Type.Int)
            state.BestObjectiveValue = (float)(double)d["best_objective"];

        if (!d.ContainsKey("trials") || d["trials"].VariantType != Variant.Type.Array)
            return state;

        foreach (var item in d["trials"].AsGodotArray())
        {
            if (item.VariantType != Variant.Type.Dictionary)
                continue;
            var td = item.AsGodotDictionary();
            var rec = new TrialRecord
            {
                TrialId          = td.ContainsKey("trial_id")    ? (int)td["trial_id"]                  : 0,
                RunId            = td.ContainsKey("run_id")      ? td["run_id"].ToString()               : "",
                RunDirectory     = td.ContainsKey("run_dir")     ? td["run_dir"].ToString()              : "",
                TotalSteps       = td.ContainsKey("total_steps") ? DictGetLong(td, "total_steps")       : 0L,
                StartedAtTicks   = td.ContainsKey("started_at")  ? DictGetLong(td, "started_at")        : 0L,
                CompletedAtTicks = td.ContainsKey("completed_at")? DictGetLong(td, "completed_at")      : 0L,
            };

            if (td.ContainsKey("state"))
                rec.State = td["state"].ToString() switch
                {
                    "Running"  => RLHPOTrialState.Running,
                    "Complete" => RLHPOTrialState.Complete,
                    "Pruned"   => RLHPOTrialState.Pruned,
                    "Failed"   => RLHPOTrialState.Failed,
                    _          => RLHPOTrialState.Pending,
                };

            if (td.ContainsKey("objective") && td["objective"].VariantType is Variant.Type.Float or Variant.Type.Int)
                rec.ObjectiveValue = (float)(double)td["objective"];

            if (td.ContainsKey("params") && td["params"].VariantType == Variant.Type.Dictionary)
            {
                foreach (var (pk, pv) in td["params"].AsGodotDictionary())
                {
                    if (pv.VariantType is Variant.Type.Float or Variant.Type.Int)
                        rec.Parameters[pk.ToString()] = (float)(double)pv;
                }
            }

            state.Trials.Add(rec);
        }

        return state;
    }

    public static Variant ParseSanitizedJson(string content)
    {
        return Json.ParseString(SanitizeInvalidObjectiveNumbers(content));
    }

    public static string SanitizeInvalidObjectiveNumbers(string content)
    {
        if (string.IsNullOrEmpty(content))
            return content;

        return InvalidObjectiveNumberRegex.Replace(content, match =>
        {
            var prefix = match.Groups["prefix"].Value;
            var value = match.Groups["value"].Value;
            return TryParseFiniteFloat(value, out _) ? match.Value : $"{prefix}null";
        });
    }

    private static long DictGetLong(Godot.Collections.Dictionary d, string key)
    {
        if (!d.ContainsKey(key)) return 0L;
        var value = d[key];
        return value.VariantType == Variant.Type.Int   ? value.AsInt64()
             : value.VariantType == Variant.Type.Float ? (long)value.AsDouble()
             : 0L;
    }

    private static bool TryParseFiniteFloat(string text, out float value)
    {
        if (!float.TryParse(text, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out value))
            return false;

        return !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
