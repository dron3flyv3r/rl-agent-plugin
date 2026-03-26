using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Base class for user-defined training stopping conditions.
/// Subclass in GDScript or C# and assign to <see cref="RLStoppingConfig.CustomCondition"/>.
/// <para>
/// GDScript override signature:
/// <code>func should_stop(total_steps: int, total_episodes: int, elapsed_seconds: float, rolling_reward: float) -> bool</code>
/// </para>
/// </summary>
[GlobalClass]
[Tool]
public partial class RLCustomStoppingCondition : Resource
{
    /// <summary>
    /// Override this method to implement a custom stopping rule.
    /// Called once per physics frame after built-in conditions are evaluated.
    /// </summary>
    /// <param name="totalSteps">Current global step count.</param>
    /// <param name="totalEpisodes">Sum of completed episodes across all policy groups.</param>
    /// <param name="elapsedSeconds">Seconds elapsed since training started.</param>
    /// <param name="rollingReward">
    /// Current rolling average reward (NaN if <see cref="RLStoppingConfig.RewardThresholdWindow"/> is 0
    /// or the window is not yet full).
    /// </param>
    /// <returns>True to trigger a clean training stop.</returns>
    public virtual bool ShouldStop(long totalSteps, long totalEpisodes, double elapsedSeconds, float rollingReward)
        => false;
}

/// <summary>
/// Inspector-configurable stopping conditions for a training run.
/// Assign an instance to <see cref="RLRunConfig.StoppingConditions"/>.
/// All conditions with a nonzero value are active; leave a value at zero (or window at 0) to disable that condition.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLStoppingConfig : Resource
{
    /// <summary>
    /// How to combine multiple active conditions.
    /// <list type="bullet">
    ///   <item><b>Any</b> (default) — stop when any single condition fires (OR logic).</item>
    ///   <item><b>All</b> — stop only when every active condition fires simultaneously (AND logic).</item>
    /// </list>
    /// </summary>
    [Export] public RLStoppingCombineMode CombineMode { get; set; } = RLStoppingCombineMode.Any;

    /// <summary>Stop after this many global physics steps. Set to 0 to disable.</summary>
    [Export(PropertyHint.Range, "0,1000000000,1,or_greater")]
    public long MaxSteps { get; set; } = 0;

    /// <summary>Stop after this many seconds of real training time. Set to 0 to disable.</summary>
    [Export(PropertyHint.Range, "0,864000,1,or_greater")]
    public double MaxSeconds { get; set; } = 0.0;

    /// <summary>Stop after this many total episodes across all policy groups. Set to 0 to disable.</summary>
    [Export(PropertyHint.Range, "0,100000000,1,or_greater")]
    public long MaxEpisodes { get; set; } = 0;

    /// <summary>
    /// Stop when the rolling average episode reward meets or exceeds this value.
    /// Only evaluated when <see cref="RewardThresholdWindow"/> is greater than 0.
    /// </summary>
    [Export] public float RewardThreshold { get; set; } = 0f;

    /// <summary>
    /// Number of recent episodes used to compute the rolling reward average.
    /// Set to 0 to disable the reward threshold condition entirely.
    /// The condition does not fire until this many episodes have been completed.
    /// </summary>
    [Export(PropertyHint.Range, "0,10000,1,or_greater")]
    public int RewardThresholdWindow { get; set; } = 0;

    /// <summary>
    /// Policy group ID whose reward is tracked for the threshold check.
    /// Leave empty to average across all policy groups.
    /// </summary>
    [Export] public string RewardThresholdGroup { get; set; } = string.Empty;

    /// <summary>
    /// Optional user-defined stopping condition.
    /// Subclass <see cref="RLCustomStoppingCondition"/> and assign an instance here.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLCustomStoppingCondition))]
    public RLCustomStoppingCondition? CustomCondition { get; set; }
}
