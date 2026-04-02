using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Successive Halving (SHA) pruner — at each defined "rung", keeps only the top
/// <c>1/η</c> fraction of trials and prunes the rest.
/// <para>
/// Rungs are expressed as fractions of <see cref="RLHPOStudy.PruneAfterSteps"/>.
/// For example, with <c>Eta=3</c> and <c>PruneAfterSteps=30_000</c>, rungs are
/// at steps 10_000, 20_000 and 30_000. A trial is pruned at a rung if its
/// current objective ranks in the bottom (1 − 1/η) fraction of all trials
/// that have also passed that rung.
/// </para>
/// </summary>
public sealed class SuccessiveHalvingPruner : IHPOPruner
{
    /// <summary>Halving factor. Each rung keeps 1/Eta fraction of trials.</summary>
    private int _eta = 3;

    public int Eta
    {
        get => _eta;
        set => _eta = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Eta must be greater than 0.");
    }

    /// <summary>Number of evaluation rungs evenly spaced up to PruneAfterSteps.</summary>
    private int _numRungs = 3;

    public int NumRungs
    {
        get => _numRungs;
        set => _numRungs = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "NumRungs must be greater than 0.");
    }

    /// <summary>Minimum number of trials that must have passed a rung before pruning.</summary>
    public int MinTrialsAtRung { get; set; } = 3;

    public bool ShouldPrune(
        long currentSteps,
        float currentObjective,
        RLHPOStudy study,
        IReadOnlyList<TrialRecord> history)
    {
        if (currentSteps < study.PruneAfterSteps / NumRungs)
            return false;

        // Find the highest rung the current trial has crossed
        long rungStep = RungStepForCurrentSteps(currentSteps, study.PruneAfterSteps);
        if (rungStep <= 0)
            return false;

        // Collect all completed trials' objective values (use final value as proxy)
        var trialsAtRung = history
            .Where(t => t.State == RLHPOTrialState.Complete
                     && t.ObjectiveValue.HasValue
                     && t.TotalSteps >= rungStep)
            .Select(t => t.ObjectiveValue!.Value)
            .ToList();

        if (trialsAtRung.Count < MinTrialsAtRung)
            return false;

        int keepCount = Math.Max(1, (int)Math.Ceiling(trialsAtRung.Count / (float)Eta));

        if (study.Direction == RLHPODirection.Maximize)
        {
            float threshold = trialsAtRung.OrderByDescending(v => v).ElementAt(keepCount - 1);
            return currentObjective < threshold;
        }
        else
        {
            float threshold = trialsAtRung.OrderBy(v => v).ElementAt(keepCount - 1);
            return currentObjective > threshold;
        }
    }

    private long RungStepForCurrentSteps(long currentSteps, long pruneAfterSteps)
    {
        long highestRung = 0;
        for (int i = 1; i <= NumRungs; i++)
        {
            long rung = pruneAfterSteps * i / NumRungs;
            if (currentSteps >= rung)
                highestRung = rung;
        }
        return highestRung;
    }
}
