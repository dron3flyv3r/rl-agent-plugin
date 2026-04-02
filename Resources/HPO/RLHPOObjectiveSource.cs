using Godot;

namespace RlAgentPlugin;

/// <summary>
/// One explicit policy-group metric source used by an HPO objective config.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLHPOObjectiveSource : Resource
{
    /// <summary>
    /// Policy group id whose metrics file contributes to the trial objective.
    /// This must match the policy group's resolved id / metrics filename suffix.
    /// </summary>
    [Export] public string PolicyGroup { get; set; } = "";

    /// <summary>
    /// Metric extracted from this policy group's metrics JSONL.
    /// </summary>
    [Export] public RLHPOObjectiveMetric Metric { get; set; } = RLHPOObjectiveMetric.MeanEpisodeReward;

    /// <summary>
    /// JSONL key used when <see cref="Metric"/> is <see cref="RLHPOObjectiveMetric.Custom"/>.
    /// </summary>
    [Export] public string CustomMetricKey { get; set; } = "";

    /// <summary>
    /// Relative weight used when the parent objective config uses
    /// <see cref="RLHPOObjectiveAggregation.WeightedMean"/>.
    /// Ignored by the other aggregation modes.
    /// </summary>
    [Export(PropertyHint.Range, "0,1000,0.01,or_greater")] public float Weight { get; set; } = 1f;
}
