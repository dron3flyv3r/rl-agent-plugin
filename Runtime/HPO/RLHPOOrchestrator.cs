using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Place this node as a direct child of your <c>RLAcademy</c> node to enable
/// automated hyperparameter optimisation. Assign an <see cref="RLHPOStudy"/>
/// resource in the inspector and the training system handles the rest.
/// <para>
/// When <c>TrainingBootstrap</c> detects this node it runs the HPO loop instead
/// of a single training run. Each trial is a headless subprocess of the same
/// scene; the scene itself is never modified.
/// </para>
/// </summary>
public partial class RLHPOOrchestrator : Node
{
    [Export] public RLHPOStudy? Study { get; set; }

    private StudyState _state = null!;
    private IHPOSampler _sampler = null!;
    private IHPOPruner _pruner = null!;
    private ITrialExecutor _executor = null!;
    private TrainingLaunchManifest _manifest = null!;
    private string _studyDir = "";
    private string _stateResPath = "";
    private Node _timerOwner = null!;
    private string _masterHeartbeatResPath = "";
    private string _masterHeartbeatToken = "";
    private bool _masterHeartbeatRunning;
    private const float MasterHeartbeatIntervalSeconds = 1.0f;

    // ── Activation (called by TrainingBootstrap, not by Godot) ───────────

    /// <summary>
    /// Called by <c>TrainingBootstrap</c> when it detects this node as a child of
    /// the academy. Starts the HPO loop asynchronously; the caller should return
    /// immediately after this to avoid starting a normal training run.
    /// </summary>
    internal void Activate(TrainingLaunchManifest manifest, Node timerOwner)
    {
        _timerOwner = timerOwner;
        _manifest   = manifest;

        if (Study is null)
        {
            GD.PushError("[HPO] RLHPOOrchestrator has no RLHPOStudy assigned.");
            GetTree().Quit(1);
            return;
        }

        if (Study.SearchSpace is null || Study.SearchSpace.Count == 0)
        {
            GD.PushError("[HPO] RLHPOStudy.SearchSpace is empty — add at least one RLHPOParameter.");
            GetTree().Quit(1);
            return;
        }

        if (Study.ObjectiveConfig is null)
        {
            GD.PushError("[HPO] RLHPOStudy.ObjectiveConfig is missing.");
            GetTree().Quit(1);
            return;
        }

        var objectiveSources = Study.ObjectiveConfig.Sources
            .Where(s => s is not null && !string.IsNullOrWhiteSpace(s.PolicyGroup))
            .ToList();
        if (objectiveSources.Count == 0)
        {
            GD.PushError("[HPO] RLHPOStudy.ObjectiveConfig.Sources is empty — add at least one objective source.");
            GetTree().Quit(1);
            return;
        }

        if (objectiveSources.Any(s => s.Metric == RLHPOObjectiveMetric.Custom && string.IsNullOrWhiteSpace(s.CustomMetricKey)))
        {
            GD.PushError("[HPO] HPO objective source uses Custom metric but has no CustomMetricKey.");
            GetTree().Quit(1);
            return;
        }

        _studyDir     = $"res://RL-Agent-Training/hpo/{_manifest.RunId}/{Study.StudyName}";
        _stateResPath = $"{_studyDir}/study_state.json";
        DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(_studyDir));

        _masterHeartbeatResPath = $"{_studyDir}/master_heartbeat.json";
        _masterHeartbeatToken = Guid.NewGuid().ToString("N");
        _manifest.HpoMasterHeartbeatPath = _masterHeartbeatResPath;
        _manifest.HpoMasterHeartbeatToken = _masterHeartbeatToken;
        StartMasterHeartbeat();

        _state = StudyState.LoadOrCreate(_stateResPath, Study);
        bool stateChanged = !string.Equals(_state.OwnerRunId, _manifest.RunId, StringComparison.Ordinal);
        _state.OwnerRunId = _manifest.RunId;
        if (_state.RecoverInterruptedTrials())
            stateChanged = true;
        if (stateChanged)
            SaveState();

        _sampler = Study.SamplerKind == RLHPOSamplerKind.TPE
            ? new TPESampler()
            : new RandomSampler();

        _pruner = Study.PrunerKind switch
        {
            RLHPOPrunerKind.Median            => new MedianPruner(),
            RLHPOPrunerKind.SuccessiveHalving => new SuccessiveHalvingPruner(),
            _                                  => new NoPruner(),
        };

        _executor = new SequentialTrialExecutor();

        int doneCount = _state.Trials.Count(t =>
            t.State is RLHPOTrialState.Complete or RLHPOTrialState.Pruned or RLHPOTrialState.Failed);

        GD.Print($"[HPO] Study '{Study.StudyName}' — {doneCount}/{Study.TrialBudget} trials done.");
        if (_state.BestTrialId.HasValue)
            GD.Print($"[HPO] Best so far: trial {_state.BestTrialId} | objective={_state.BestObjectiveValue:F4}");

        _ = RunLoop();
    }

    public override void _ExitTree()
    {
        _masterHeartbeatRunning = false;
    }

    // ── Main loop ─────────────────────────────────────────────────────────

    private async Task RunLoop()
    {
        try
        {
            int doneCount = _state.Trials.Count(t =>
                t.State is RLHPOTrialState.Complete or RLHPOTrialState.Pruned or RLHPOTrialState.Failed);

            while (doneCount < Study!.TrialBudget)
            {
                int trialId   = _state.Trials.Count;
                long timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                string runId  = $"{Study.StudyName}_trial_{trialId}_{timestamp}";
                string runDir = $"res://RL-Agent-Training/runs/{runId}";

                var completedTrials = _state.Trials
                    .Where(t => t.State == RLHPOTrialState.Complete)
                    .ToList();
                var parameters = _sampler.Suggest(completedTrials, Study);

                GD.Print($"[HPO] Starting trial {trialId}/{Study.TrialBudget}: {FormatParams(parameters)}");

                var trial = new TrialRecord
                {
                    TrialId        = trialId,
                    Parameters     = parameters,
                    State          = RLHPOTrialState.Running,
                    RunId          = runId,
                    RunDirectory   = runDir,
                    StartedAtTicks = timestamp,
                };
                _state.Trials.Add(trial);
                SaveState();

                TrialResult result;
                try
                {
                    result = await _executor.ExecuteTrial(
                        trial, Study, _manifest, _pruner, _state.Trials.AsReadOnly(), _timerOwner);
                }
                catch (Exception ex)
                {
                    GD.PushError($"[HPO] Trial {trialId} threw an exception: {ex}");
                    trial.State = RLHPOTrialState.Failed;
                    SaveState();
                    doneCount++;
                    continue;
                }

                trial.CompletedAtTicks = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                trial.TotalSteps       = result.TotalSteps;

                if (result.Pruned)
                {
                    trial.State = RLHPOTrialState.Pruned;
                    GD.Print($"[HPO] Trial {trialId} pruned at step {result.StepsAtPrune}.");
                }
                else if (float.IsNaN(result.ObjectiveValue))
                {
                    trial.State = RLHPOTrialState.Failed;
                    GD.PushWarning($"[HPO] Trial {trialId} failed (no objective).");
                }
                else
                {
                    trial.State          = RLHPOTrialState.Complete;
                    trial.ObjectiveValue = result.ObjectiveValue;
                    GD.Print($"[HPO] Trial {trialId} complete — objective={result.ObjectiveValue:F4}");
                }

                _state.UpdateBest();
                SaveState();
                doneCount++;
            }

            GD.Print($"[HPO] Study '{Study.StudyName}' finished ({doneCount} trials).");
            if (_state.BestTrialId.HasValue)
            {
                var best = _state.Trials[_state.BestTrialId.Value];
                GD.Print($"[HPO] Best trial: {_state.BestTrialId} | objective={_state.BestObjectiveValue:F4}");
                GD.Print($"[HPO] Best parameters: {FormatParams(best.Parameters)}");
            }

            GetTree().Quit();
        }
        catch (Exception ex)
        {
            GD.PushError($"[HPO] RunLoop failed: {ex}");
            if (IsInsideTree())
                GetTree().Quit(1);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private void SaveState() => _state.SaveToResPath(_stateResPath);

    private void StartMasterHeartbeat()
    {
        if (_masterHeartbeatRunning || string.IsNullOrWhiteSpace(_masterHeartbeatResPath))
            return;

        WriteMasterHeartbeat();
        _masterHeartbeatRunning = true;
        RunMasterHeartbeatLoop();
    }

    private async void RunMasterHeartbeatLoop()
    {
        while (_masterHeartbeatRunning && IsInsideTree())
        {
            await _timerOwner.ToSignal(
                _timerOwner.GetTree().CreateTimer(MasterHeartbeatIntervalSeconds),
                SceneTreeTimer.SignalName.Timeout);

            if (!_masterHeartbeatRunning || !IsInsideTree())
                return;

            WriteMasterHeartbeat();
        }
    }

    private void WriteMasterHeartbeat()
    {
        var absDir = System.IO.Path.GetDirectoryName(ProjectSettings.GlobalizePath(_masterHeartbeatResPath));
        if (!string.IsNullOrWhiteSpace(absDir))
            DirAccess.MakeDirRecursiveAbsolute(absDir);

        using var file = FileAccess.Open(_masterHeartbeatResPath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            GD.PushWarning($"[HPO] Could not write master heartbeat to '{_masterHeartbeatResPath}'.");
            return;
        }

        var payload = new Godot.Collections.Dictionary
        {
            { "token", _masterHeartbeatToken },
            { "run_id", _manifest.RunId },
            { "updated_at_unix_ms", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() },
        };
        file.StoreString(Json.Stringify(payload));
    }

    private static string FormatParams(Dictionary<string, float> p) =>
        string.Join(", ", p.Select(kv => $"{kv.Key}={kv.Value:G4}"));
}
