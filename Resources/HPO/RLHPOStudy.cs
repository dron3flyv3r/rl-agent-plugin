using Godot;
using Godot.Collections;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin;

/// <summary>
/// Inspector-editable resource that defines a hyperparameter optimisation study.
/// Attach this to an <see cref="RLHPOOrchestrator"/> node to run automated
/// hyperparameter search over multiple training trials.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLHPOStudy : Resource
{
    // ── Identity ────────────────────────────────────────────────────────────

    /// <summary>Human-readable name. Used as a directory prefix for all trial runs.</summary>
    [Export] public string StudyName { get; set; } = "hpo_study";

    // ── Objective ───────────────────────────────────────────────────────────

    /// <summary>Whether to maximise or minimise the objective metric.</summary>
    [Export] public RLHPODirection Direction { get; set; } = RLHPODirection.Maximize;

    /// <summary>
    /// Objective definition used to score each trial from one or more policy-group metrics.
    /// </summary>
    [Export] public RLHPOObjectiveConfig? ObjectiveConfig { get; set; }

    // ── Budget ──────────────────────────────────────────────────────────────

    /// <summary>Total number of trials to run (including pruned ones).</summary>
    [Export(PropertyHint.Range, "1,1000")] public int TrialBudget { get; set; } = 20;

    /// <summary>
    /// Maximum environment steps a single trial is allowed to run before the
    /// executor stops it and records the latest available objective.
    /// Set to 0 to disable the hard step budget.
    /// </summary>
    [Export(PropertyHint.Range, "0,1000000000,1,or_greater")] public long MaxTrialSteps { get; set; } = 10_000;

    /// <summary>
    /// Maximum wall-clock seconds a single trial is allowed to run before the
    /// executor stops it and records the latest available objective.
    /// Set to 0 to disable the hard time budget.
    /// </summary>
    [Export(PropertyHint.Range, "0,86400,1,or_greater")] public double MaxTrialSeconds { get; set; } = 0.0;

    /// <summary>
    /// Maximum concurrent trials. Currently only 1 (sequential) is supported.
    /// Reserved for future parallel execution via <c>ParallelTrialExecutor</c>.
    /// </summary>
    [Export(PropertyHint.Range, "1,8")] public int MaxConcurrentTrials { get; set; } = 1;

    // ── Sampler ─────────────────────────────────────────────────────────────

    /// <summary>Algorithm used to suggest hyperparameter values for each trial.</summary>
    [Export] public RLHPOSamplerKind SamplerKind { get; set; } = RLHPOSamplerKind.TPE;

    // ── Pruner ──────────────────────────────────────────────────────────────

    /// <summary>Strategy for early-stopping underperforming trials.</summary>
    [Export] public RLHPOPrunerKind PrunerKind { get; set; } = RLHPOPrunerKind.Median;

    /// <summary>
    /// Minimum environment steps a trial must complete before pruning can occur.
    /// Set to the number of steps needed to get a reliable early signal.
    /// </summary>
    [Export] public long PruneAfterSteps { get; set; } = 10_000;

    /// <summary>
    /// How often (in seconds) the orchestrator polls a running trial for
    /// intermediate metrics to evaluate against the pruner.
    /// </summary>
    [Export(PropertyHint.Range, "1,60")] public float PollIntervalSeconds { get; set; } = 5f;

    // ── Search space ─────────────────────────────────────────────────────────

    /// <summary>
    /// List of hyperparameters to tune. Each entry names a property on
    /// <see cref="RLTrainerConfig"/> and specifies the sampling distribution.
    /// </summary>
    [Export] public Array<RLHPOParameter> SearchSpace { get; set; } = new();

    // ── Base configs ─────────────────────────────────────────────────────────

    /// <summary>
    /// Training config used as the base for every trial. The sampled hyperparameters
    /// are applied on top as overrides at startup. The resource itself is never mutated.
    /// </summary>
    [Export] public RLTrainingConfig? BaseTrainingConfig { get; set; }

    /// <summary>
    /// Optional run config shared by all trials (simulation speed, batch size, etc.).
    /// If null, the scene's academy run config is used.
    /// </summary>
    [Export] public RLRunConfig? BaseRunConfig { get; set; }

    // Note: scene path and academy node path are inferred automatically from the
    // training manifest when RLHPOOrchestrator is detected as a child of RLAcademy.
}
