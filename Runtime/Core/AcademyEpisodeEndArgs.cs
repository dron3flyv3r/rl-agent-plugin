using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Data passed to <see cref="RLAcademy.OnEpisodeEnd"/> after every episode completion.
/// </summary>
public sealed class AcademyEpisodeEndArgs
{
    /// <summary>The agent whose episode just ended.</summary>
    public IRLAgent Agent { get; init; } = null!;

    /// <summary>Policy group ID this agent belongs to.</summary>
    public string GroupId { get; init; } = string.Empty;

    /// <summary>Display name for the policy group.</summary>
    public string GroupDisplayName { get; init; } = string.Empty;

    /// <summary>Total cumulative reward for the completed episode.</summary>
    public float EpisodeReward { get; init; }

    /// <summary>Number of environment steps in the completed episode.</summary>
    public int EpisodeSteps { get; init; }

    /// <summary>
    /// Per-signal reward breakdown. Empty if the agent does not use named reward signals.
    /// </summary>
    public IReadOnlyDictionary<string, float> RewardBreakdown { get; init; }
        = new Dictionary<string, float>();

    /// <summary>Running total steps across all agents and groups at episode-end time.</summary>
    public long TotalSteps { get; init; }

    /// <summary>Cumulative episode count for this group at episode-end time.</summary>
    public long GroupEpisodeCount { get; init; }

    /// <summary>Current curriculum progress in [0, 1]. Zero when curriculum is not configured.</summary>
    public float CurriculumProgress { get; init; }
}
