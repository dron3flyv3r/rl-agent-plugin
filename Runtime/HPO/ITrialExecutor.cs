using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>Result returned by a trial executor when a trial completes or is pruned.</summary>
public sealed class TrialResult
{
    public bool Pruned { get; init; }
    public float ObjectiveValue { get; init; }
    public long TotalSteps { get; init; }
    public long StepsAtPrune { get; init; }
}

/// <summary>
/// Abstraction over trial execution strategy.
/// <para>Currently only <see cref="SequentialTrialExecutor"/> is implemented (one trial at a
/// time via a headless subprocess). A future <c>ParallelTrialExecutor</c> can launch
/// multiple subprocesses when <see cref="RLHPOStudy.MaxConcurrentTrials"/> &gt; 1.</para>
/// </summary>
public interface ITrialExecutor
{
    /// <summary>
    /// Run a single trial to completion (or prune it early).
    /// Implementations are expected to be async-friendly and non-blocking.
    /// </summary>
    /// <param name="trial">The trial record to execute (State must be Running).</param>
    /// <param name="study">Study configuration (poll interval, objective metric, etc.).</param>
    /// <param name="manifest">The active training manifest, used to resolve scene path and academy node path.</param>
    /// <param name="pruner">Pruner consulted after each metrics poll.</param>
    /// <param name="history">All trials so far (for the pruner's context).</param>
    /// <param name="owner">Godot node used to create timers for non-blocking waits.</param>
    Task<TrialResult> ExecuteTrial(
        TrialRecord trial,
        RLHPOStudy study,
        TrainingLaunchManifest manifest,
        IHPOPruner pruner,
        IReadOnlyList<TrialRecord> history,
        Node owner);
}
