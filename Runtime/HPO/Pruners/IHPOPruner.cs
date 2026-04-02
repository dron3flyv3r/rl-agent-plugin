using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Decides whether a running trial should be stopped early based on its
/// intermediate objective value relative to completed trials.
/// </summary>
public interface IHPOPruner
{
    /// <summary>
    /// Returns <c>true</c> if the trial should be pruned now.
    /// </summary>
    /// <param name="currentSteps">Environment steps completed so far in the trial.</param>
    /// <param name="currentObjective">Best objective value seen so far in the trial.</param>
    /// <param name="study">Study configuration (direction, prune threshold, etc.).</param>
    /// <param name="history">All previously completed trials.</param>
    bool ShouldPrune(
        long currentSteps,
        float currentObjective,
        RLHPOStudy study,
        IReadOnlyList<TrialRecord> history);
}

/// <summary>No-op pruner — never prunes a trial early.</summary>
public sealed class NoPruner : IHPOPruner
{
    public bool ShouldPrune(long currentSteps, float currentObjective,
        RLHPOStudy study, IReadOnlyList<TrialRecord> history) => false;
}
