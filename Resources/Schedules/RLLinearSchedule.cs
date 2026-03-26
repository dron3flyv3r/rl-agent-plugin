using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Linearly interpolates from <see cref="StartValue"/> to <see cref="EndValue"/>
/// over <see cref="DurationUpdates"/> gradient updates, then holds at EndValue.
///
/// Common use: linear learning-rate decay, entropy decay.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLLinearSchedule : RLHyperparamSchedule
{
    /// <summary>Value at update 0.</summary>
    [Export] public float StartValue { get; set; } = 0.001f;

    /// <summary>Value once DurationUpdates have elapsed (held constant after).</summary>
    [Export] public float EndValue { get; set; } = 0.0001f;

    /// <summary>
    /// Number of gradient updates over which the transition occurs.
    /// Progress = UpdateCount / DurationUpdates, clamped to [0, 1].
    /// </summary>
    [Export] public int DurationUpdates { get; set; } = 1000;

    public override float Evaluate(ScheduleContext ctx)
    {
        if (DurationUpdates <= 0) return EndValue;
        var t = Mathf.Clamp(ctx.UpdateCount / (float)DurationUpdates, 0f, 1f);
        return Mathf.Lerp(StartValue, EndValue, t);
    }
}
