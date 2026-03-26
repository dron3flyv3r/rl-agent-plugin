using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Anneals from <see cref="StartValue"/> to <see cref="EndValue"/> following a
/// cosine curve over <see cref="DurationUpdates"/> gradient updates.
///
/// Formula: value = EndValue + (StartValue - EndValue) * (1 + cos(π * t)) / 2
///
/// Starts fast, slows down near the end — popular for learning-rate warm restarts
/// and smooth entropy decay.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLCosineSchedule : RLHyperparamSchedule
{
    /// <summary>Value at update 0 (top of the cosine curve).</summary>
    [Export] public float StartValue { get; set; } = 0.001f;

    /// <summary>Value at DurationUpdates (bottom of the cosine curve, held constant after).</summary>
    [Export] public float EndValue { get; set; } = 0.0001f;

    /// <summary>
    /// Number of gradient updates for one half-period of the cosine.
    /// Progress = UpdateCount / DurationUpdates, clamped to [0, 1].
    /// </summary>
    [Export] public int DurationUpdates { get; set; } = 1000;

    public override float Evaluate(ScheduleContext ctx)
    {
        if (DurationUpdates <= 0) return EndValue;
        var t = Mathf.Clamp(ctx.UpdateCount / (float)DurationUpdates, 0f, 1f);
        return EndValue + (StartValue - EndValue) * (1f + Mathf.Cos(Mathf.Pi * t)) * 0.5f;
    }
}
