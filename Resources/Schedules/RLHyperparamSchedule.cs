using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Progress snapshot passed to every schedule's Evaluate method.
/// </summary>
public readonly struct ScheduleContext
{
    /// <summary>Number of completed gradient updates for this policy group.</summary>
    public long UpdateCount  { get; init; }

    /// <summary>Total environment steps taken across all groups.</summary>
    public long TotalSteps   { get; init; }

    /// <summary>Number of completed episodes for this policy group.</summary>
    public long EpisodeCount { get; init; }
}

/// <summary>
/// Base class for all hyperparameter schedules.
/// Subclass this (and add [GlobalClass]) to create a custom schedule that
/// appears as a resource type in the Godot Inspector.
///
/// Built-in schedules: RLConstantSchedule, RLLinearSchedule,
/// RLExponentialSchedule, RLCosineSchedule.
///
/// Example custom schedule:
/// <code>
/// [GlobalClass]
/// public partial class MyWarmupSchedule : RLHyperparamSchedule
/// {
///     [Export] public float WarmupValue  { get; set; } = 0.0001f;
///     [Export] public float TargetValue  { get; set; } = 0.001f;
///     [Export] public int   WarmupUpdates { get; set; } = 100;
///
///     public override float Evaluate(ScheduleContext ctx)
///     {
///         if (ctx.UpdateCount >= WarmupUpdates) return TargetValue;
///         return Mathf.Lerp(WarmupValue, TargetValue, ctx.UpdateCount / (float)WarmupUpdates);
///     }
/// }
/// </code>
/// </summary>
[GlobalClass]
[Tool]
public abstract partial class RLHyperparamSchedule : Resource
{
    /// <summary>
    /// Returns the scheduled value at the current training progress.
    /// Called once per gradient update, before the trainer runs.
    /// </summary>
    public abstract float Evaluate(ScheduleContext ctx);
}
