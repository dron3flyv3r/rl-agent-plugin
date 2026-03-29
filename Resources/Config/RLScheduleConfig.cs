using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Optional per-hyperparameter schedules.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Schedules"/>.
///
/// When a schedule is assigned it overrides the corresponding flat value on
/// each gradient update. Leave a slot null to keep the flat value constant.
///
/// Built-in types: RLConstantSchedule, RLLinearSchedule,
///                 RLExponentialSchedule, RLCosineSchedule.
/// Custom: subclass RLHyperparamSchedule and add [GlobalClass].
/// </summary>
[GlobalClass]
[Tool]
public partial class RLScheduleConfig : Resource
{
    /// <summary>Optional schedule for trainer learning rate.</summary>
    [Export] public RLHyperparamSchedule? LearningRate { get; set; }
    /// <summary>Optional schedule for PPO entropy bonus coefficient.</summary>
    [Export] public RLHyperparamSchedule? EntropyCoefficient { get; set; }
    /// <summary>Optional schedule for PPO clip epsilon.</summary>
    [Export] public RLHyperparamSchedule? ClipEpsilon { get; set; }
    /// <summary>Optional schedule for SAC alpha temperature.</summary>
    [Export] public RLHyperparamSchedule? SacAlpha { get; set; }

    /// <summary>
    /// Evaluates assigned schedules and writes resulting values into the trainer config.
    /// </summary>
    /// <param name="config">Mutable runtime trainer settings to override.</param>
    /// <param name="ctx">Current schedule evaluation context.</param>
    internal void ApplyTo(RLTrainerConfig config, ScheduleContext ctx)
    {
        if (LearningRate is not null) config.LearningRate = LearningRate.Evaluate(ctx);
        if (EntropyCoefficient is not null) config.EntropyCoefficient = EntropyCoefficient.Evaluate(ctx);
        if (ClipEpsilon is not null) config.ClipEpsilon = ClipEpsilon.Evaluate(ctx);
        if (SacAlpha is not null) config.SacInitAlpha = SacAlpha.Evaluate(ctx);
    }
}
