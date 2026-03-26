using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Returns a fixed value regardless of training progress.
/// Useful as an explicit no-op schedule or when you want to override
/// the base hyperparameter value without using a decaying schedule.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLConstantSchedule : RLHyperparamSchedule
{
    [Export] public float Value { get; set; } = 0.001f;

    public override float Evaluate(ScheduleContext ctx) => Value;
}
