using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Prunes a trial if its current objective is below (or above, for minimisation)
/// the median of all completed trials' final objective values.
/// <para>
/// Pruning is only activated after <see cref="RLHPOStudy.PruneAfterSteps"/> steps
/// and after at least <see cref="MinCompletedTrials"/> trials have finished.
/// </para>
/// </summary>
public sealed class MedianPruner : IHPOPruner
{
    /// <summary>Minimum number of completed trials before pruning is active.</summary>
    private int _minCompletedTrials = 3;

    public int MinCompletedTrials
    {
        get => _minCompletedTrials;
        set => _minCompletedTrials = value >= 1
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "MinCompletedTrials must be at least 1.");
    }

    public bool ShouldPrune(
        long currentSteps,
        float currentObjective,
        RLHPOStudy study,
        IReadOnlyList<TrialRecord> history)
    {
        if (currentSteps < study.PruneAfterSteps)
            return false;

        var completed = history
            .Where(t => t.State == RLHPOTrialState.Complete && t.ObjectiveValue.HasValue)
            .Select(t => t.ObjectiveValue!.Value)
            .ToList();

        if (completed.Count == 0 || completed.Count < MinCompletedTrials)
            return false;

        float median = Median(completed);
        return study.Direction == RLHPODirection.Maximize
            ? currentObjective < median
            : currentObjective > median;
    }

    private static float Median(List<float> values)
    {
        var sorted = values.OrderBy(v => v).ToList();
        int n = sorted.Count;
        return n % 2 == 1
            ? sorted[n / 2]
            : (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5f;
    }
}
