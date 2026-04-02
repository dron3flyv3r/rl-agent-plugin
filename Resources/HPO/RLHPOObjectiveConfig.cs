using Godot;
using Godot.Collections;

namespace RlAgentPlugin;

/// <summary>
/// Defines how an HPO trial is scored from one or more policy-group metrics.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLHPOObjectiveConfig : Resource
{
    /// <summary>
    /// Number of trailing JSONL entries averaged for each configured source.
    /// Shared across all sources to keep the objective easy to reason about.
    /// </summary>
    [Export(PropertyHint.Range, "1,500")] public int EvaluationWindow { get; set; } = 20;

    /// <summary>
    /// How multiple per-source objective values are combined into one scalar.
    /// </summary>
    [Export] public RLHPOObjectiveAggregation Aggregation { get; set; } = RLHPOObjectiveAggregation.Mean;

    /// <summary>
    /// Explicit metric sources that contribute to the study objective.
    /// Add one source for single-policy HPO, or multiple for self-play / multi-policy setups.
    /// </summary>
    [Export] public Array<RLHPOObjectiveSource> Sources { get; set; } = new();
}
