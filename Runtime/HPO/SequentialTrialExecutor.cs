using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Executes HPO trials one at a time by launching a headless Godot subprocess for
/// each trial and polling its status and metrics files until completion or pruning.
/// <para>
/// Each subprocess receives <c>-- --rl-hpo-trial &lt;id&gt;</c> so that
/// <c>TrainingBootstrap</c> knows to run as a plain training trial and ignore the
/// embedded <c>RLHPOOrchestrator</c>.
/// </para>
/// </summary>
public sealed class SequentialTrialExecutor : ITrialExecutor
{
    private const string OverridePath = "user://rl-agent-plugin/hpo_override.json";
    private const string BootstrapScenePath = "res://addons/rl-agent-plugin/Scenes/Bootstrap/TrainingBootstrap.tscn";

    public async Task<TrialResult> ExecuteTrial(
        TrialRecord trial,
        RLHPOStudy study,
        TrainingLaunchManifest masterManifest,
        IHPOPruner pruner,
        IReadOnlyList<TrialRecord> history,
        Node owner)
    {
        var startedAt = DateTime.UtcNow;

        // ── 1. Write hyperparameter overrides ───────────────────────────
        HPOConfigApplicator.WriteOverrideFile(OverridePath, trial.Parameters);

        // ── 2. Build trial manifest ──────────────────────────────────────
        var runConfig = study.BaseRunConfig;
        var trialManifest = new TrainingLaunchManifest
        {
            ScenePath        = masterManifest.ScenePath,
            AcademyNodePath  = masterManifest.AcademyNodePath,
            RunId            = trial.RunId,
            RunDirectory     = trial.RunDirectory,
            StatusPath       = $"{trial.RunDirectory}/status.json",
            MetricsPath      = $"{trial.RunDirectory}/metrics__default.jsonl",
            SimulationSpeed  = runConfig?.SimulationSpeed ?? masterManifest.SimulationSpeed,
            BatchSize        = runConfig?.BatchSize       ?? masterManifest.BatchSize,
            ActionRepeat     = runConfig?.ActionRepeat    ?? masterManifest.ActionRepeat,
            CheckpointInterval = runConfig?.CheckpointInterval ?? masterManifest.CheckpointInterval,
            HpoOverridePath  = OverridePath,
            HpoMasterHeartbeatPath = masterManifest.HpoMasterHeartbeatPath,
            HpoMasterHeartbeatToken = masterManifest.HpoMasterHeartbeatToken,
        };

        var manifestError = trialManifest.SaveToUserStorage();
        if (manifestError != Error.Ok)
        {
            GD.PushError($"[HPO] Failed to save manifest for trial {trial.TrialId}: {manifestError}");
            return new TrialResult { Pruned = false, ObjectiveValue = float.NaN };
        }

        // ── 3. Launch headless subprocess ────────────────────────────────
        // Pass -- --rl-hpo-trial <id> so TrainingBootstrap skips HPO detection
        // HPO trials must enter via TrainingBootstrap so the subprocess loads the
        // freshly written manifest, instantiates the actual training scene from
        // manifest.ScenePath, and applies the per-trial override file.
        var args = new[]
        {
            "--headless",
            "--path", ProjectSettings.GlobalizePath("res://"),
            BootstrapScenePath,
            "--",
            "--rl-hpo-trial", trial.TrialId.ToString(),
            "--hpo-master-pid", System.Diagnostics.Process.GetCurrentProcess().Id.ToString(),
        };

        long pid = OS.CreateProcess(OS.GetExecutablePath(), args);
        if (pid <= 0)
        {
            GD.PushError($"[HPO] Failed to launch subprocess for trial {trial.TrialId}.");
            return new TrialResult { Pruned = false, ObjectiveValue = float.NaN };
        }

        GD.Print($"[HPO] Trial {trial.TrialId} subprocess PID={pid}");

        // ── 4. Poll until done / pruned ──────────────────────────────────
        float bestIntermediate = study.Direction == RLHPODirection.Maximize
            ? float.NegativeInfinity
            : float.PositiveInfinity;
        long latestSteps = 0;

        while (OS.IsProcessRunning((int)pid))
        {
            await owner.ToSignal(
                owner.GetTree().CreateTimer(study.PollIntervalSeconds),
                SceneTreeTimer.SignalName.Timeout);

            var reading = HPOMetricsReader.TryReadObjective(trial.RunDirectory, study);
            if (reading.HasValue)
            {
                latestSteps = reading.Value.totalSteps;
                bool better = study.Direction == RLHPODirection.Maximize
                    ? reading.Value.objective > bestIntermediate
                    : reading.Value.objective < bestIntermediate;
                if (better) bestIntermediate = reading.Value.objective;
            }

            latestSteps = Math.Max(latestSteps, ReadLatestStepsFromStatus(trialManifest.StatusPath));

            var elapsedSeconds = (DateTime.UtcNow - startedAt).TotalSeconds;
            var hitStepBudget = study.MaxTrialSteps > 0 && latestSteps >= study.MaxTrialSteps;
            var hitTimeBudget = study.MaxTrialSeconds > 0.0 && elapsedSeconds >= study.MaxTrialSeconds;
            if (hitStepBudget || hitTimeBudget)
            {
                var reason = hitStepBudget
                    ? $"step budget {study.MaxTrialSteps}"
                    : $"time budget {study.MaxTrialSeconds:0.#}s";
                GD.Print($"[HPO] Stopping trial {trial.TrialId} after reaching {reason}.");
                OS.Kill((int)pid);

                var capped = HPOMetricsReader.TryReadObjective(trial.RunDirectory, study);
                var cappedObjective = capped?.objective ?? bestIntermediate;
                var cappedSteps = capped?.totalSteps ?? latestSteps;

                if (!IsFinite(cappedObjective))
                {
                    GD.PushWarning($"[HPO] Trial {trial.TrialId} hit the budget before producing a valid objective.");
                    return new TrialResult
                    {
                        Pruned = false,
                        ObjectiveValue = float.NaN,
                        TotalSteps = cappedSteps,
                    };
                }

                return new TrialResult
                {
                    Pruned = false,
                    ObjectiveValue = cappedObjective,
                    TotalSteps = cappedSteps,
                };
            }

            if (!float.IsInfinity(bestIntermediate) && !float.IsNaN(bestIntermediate))
            {
                var completedOnly = history.Where(t => t.State == RLHPOTrialState.Complete).ToList();
                if (pruner.ShouldPrune(latestSteps, bestIntermediate, study, completedOnly))
                {
                    GD.Print($"[HPO] Pruning trial {trial.TrialId} at step {latestSteps}.");
                    OS.Kill((int)pid);
                    return new TrialResult { Pruned = true, StepsAtPrune = latestSteps };
                }
            }
        }

        // ── 5. Final objective ───────────────────────────────────────────
        var final = HPOMetricsReader.TryReadObjective(trial.RunDirectory, study);
        float finalObjective = final?.objective ?? bestIntermediate;
        long  finalSteps     = final?.totalSteps ?? latestSteps;

        if (float.IsNaN(finalObjective) || float.IsInfinity(finalObjective))
        {
            GD.PushWarning($"[HPO] Trial {trial.TrialId} produced no valid objective.");
            return new TrialResult { Pruned = false, ObjectiveValue = float.NaN, TotalSteps = finalSteps };
        }

        return new TrialResult { Pruned = false, ObjectiveValue = finalObjective, TotalSteps = finalSteps };
    }

    private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    private static long ReadLatestStepsFromStatus(string statusResPath)
    {
        if (!FileAccess.FileExists(statusResPath))
            return 0L;

        using var file = FileAccess.Open(statusResPath, FileAccess.ModeFlags.Read);
        if (file is null)
            return 0L;

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary)
            return 0L;

        var data = parsed.AsGodotDictionary();
        if (!data.ContainsKey("total_steps"))
            return 0L;

        var totalSteps = data["total_steps"];
        return totalSteps.VariantType switch
        {
            Variant.Type.Int => totalSteps.AsInt64(),
            Variant.Type.Float => (long)totalSteps.AsDouble(),
            _ => 0L,
        };
    }
}
