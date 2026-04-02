using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Suggests a set of hyperparameter values for the next trial given the history
/// of already-completed trials.
/// </summary>
public interface IHPOSampler
{
    /// <summary>
    /// Returns a parameter name → value mapping for the next trial.
    /// Values for <see cref="RLHPOParameterKind.IntUniform"/> /
    /// <see cref="RLHPOParameterKind.IntLog"/> are rounded floats.
    /// </summary>
    Dictionary<string, float> Suggest(
        IReadOnlyList<TrialRecord> completedTrials,
        RLHPOStudy study);
}
