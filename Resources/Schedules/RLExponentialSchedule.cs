using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Decays from <see cref="StartValue"/> to <see cref="EndValue"/> following an
/// exponential curve over <see cref="DurationUpdates"/> gradient updates.
///
/// Formula: value = StartValue * (EndValue / StartValue) ^ (updateCount / DurationUpdates)
///
/// Common use: fast initial decay for learning rate or entropy, slower tail-off.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLExponentialSchedule : RLHyperparamSchedule
{
    /// <summary>Value at update 0.</summary>
    [Export] public float StartValue { get; set; } = 0.001f;

    /// <summary>Asymptotic target value (held constant after DurationUpdates).</summary>
    [Export] public float EndValue { get; set; } = 0.00001f;

    /// <summary>
    /// Number of gradient updates for the exponential transition.
    /// Progress = UpdateCount / DurationUpdates, clamped to [0, 1].
    /// </summary>
    [Export] public int DurationUpdates { get; set; } = 1000;

    public override float Evaluate(ScheduleContext ctx)
    {
        if (DurationUpdates <= 0) return EndValue;
        if (StartValue <= 0f || EndValue <= 0f) return EndValue;
        var t = Mathf.Clamp(ctx.UpdateCount / (float)DurationUpdates, 0f, 1f);
        return StartValue * Mathf.Pow(EndValue / StartValue, t);
    }
}
