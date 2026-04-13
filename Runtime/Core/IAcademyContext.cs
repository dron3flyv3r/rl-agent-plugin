using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Provides access to the live training state from within a custom <see cref="RLAcademy"/> subclass.
/// Obtained via <see cref="RLAcademy.OnTrainingInitialized"/> and valid for the entire training run.
/// <para>
/// Threading contract:
/// <list type="bullet">
///   <item><see cref="EstimateNextValues"/> and <see cref="SampleActions"/> are pure math — thread-safe.</item>
///   <item><see cref="RecordTransitionsAndReset"/> and <see cref="ApplyDecisions"/> touch the Godot scene tree — main thread only.</item>
///   <item><see cref="RunGroupDecisionPipeline"/> runs all four phases sequentially on the calling thread; safe when called from <c>_PhysicsProcess</c>.</item>
/// </list>
/// </para>
/// </summary>
public interface IAcademyContext
{
    // ── Counters ──────────────────────────────────────────────────────────────

    /// <summary>Total environment steps taken across all groups and batch instances.</summary>
    long TotalSteps { get; }

    /// <summary>Per-group cumulative episode completion counts.</summary>
    IReadOnlyDictionary<string, long> EpisodeCountByGroup { get; }

    // ── Discovery ─────────────────────────────────────────────────────────────

    /// <summary>All registered policy group IDs in insertion order.</summary>
    IReadOnlyList<string> GroupIds { get; }

    /// <summary>
    /// Returns the trainer for the given policy group, or <c>null</c> if the group is unknown.
    /// Tier 3: the caller has full <see cref="ITrainer"/> access — RecordTransition, TryUpdate, CreateCheckpoint, etc.
    /// </summary>
    ITrainer? GetTrainer(string groupId);

    /// <summary>
    /// Returns all agents currently enrolled in training for the given group,
    /// across all batch environment instances.
    /// </summary>
    IReadOnlyList<IRLAgent> GetGroupAgents(string groupId);

    // ── Training step execution ───────────────────────────────────────────────

    /// <summary>
    /// Runs the standard four-phase pipeline (A→B→C→D) for one group sequentially.
    /// Safe to call from the main thread inside <see cref="RLAcademy.TrainingStep"/>.
    /// </summary>
    void RunGroupDecisionPipeline(string groupId);

    /// <summary>
    /// Phase A — estimate bootstrap values for pending decisions.
    /// Thread-safe (pure math). Returns a token required by <see cref="RecordTransitionsAndReset"/>.
    /// </summary>
    PhaseAToken EstimateNextValues(string groupId);

    /// <summary>
    /// Phase B — record transitions, handle episode endings, reset done agents.
    /// <b>Main thread only</b> — mutates the Godot scene tree.
    /// Returns a token required by <see cref="SampleActions"/>.
    /// </summary>
    PhaseBToken RecordTransitionsAndReset(string groupId, PhaseAToken phaseA);

    /// <summary>
    /// Phase C — sample new actions from the trainer for all pending decisions.
    /// Thread-safe (pure math). Returns a token required by <see cref="ApplyDecisions"/>.
    /// </summary>
    PhaseCToken SampleActions(string groupId, PhaseBToken phaseB);

    /// <summary>
    /// Phase D — apply sampled decisions back to agents in the scene tree.
    /// <b>Main thread only</b> — mutates the Godot scene tree.
    /// </summary>
    void ApplyDecisions(string groupId, PhaseCToken phaseC);

    // ── Checkpointing ─────────────────────────────────────────────────────────

    /// <summary>
    /// Triggers a checkpoint write for all groups.
    /// Pass <c>forceWrite: true</c> to bypass the configured checkpoint interval.
    /// </summary>
    void TriggerCheckpoint(bool forceWrite = false);

    // ── Metrics ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Appends a custom scalar metric to the group's training log.
    /// The value appears in RLDash under <paramref name="metricKey"/>.
    /// </summary>
    void LogMetric(string groupId, string metricKey, float value);

    // ── Curriculum ────────────────────────────────────────────────────────────

    /// <summary>
    /// Sets curriculum progress on all academy instances (equivalent to calling
    /// <see cref="RLAcademy.SetCurriculumProgress"/> on each batch copy).
    /// </summary>
    void SetCurriculumProgress(float progress);

    // ── Control ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Requests graceful training termination. TrainingBootstrap will write the final
    /// checkpoint and status file, then quit at end of the current frame.
    /// </summary>
    void RequestStop(string reason = "Stopped by custom loop.");
}

// ── Opaque phase tokens ───────────────────────────────────────────────────────
// Distinct types enforce phase ordering at compile time: D cannot be called before C,
// C before B, or B before A.

/// <summary>Carries Phase A results. Pass to <see cref="IAcademyContext.RecordTransitionsAndReset"/>.</summary>
public sealed class PhaseAToken
{
    internal string GroupId { get; init; } = string.Empty;
    internal object Payload { get; init; } = null!;
}

/// <summary>Carries Phase B results. Pass to <see cref="IAcademyContext.SampleActions"/>.</summary>
public sealed class PhaseBToken
{
    internal string GroupId { get; init; } = string.Empty;
    internal object Payload { get; init; } = null!;
}

/// <summary>Carries Phase C results. Pass to <see cref="IAcademyContext.ApplyDecisions"/>.</summary>
public sealed class PhaseCToken
{
    internal string GroupId { get; init; } = string.Empty;
    internal object Payload { get; init; } = null!;
}
