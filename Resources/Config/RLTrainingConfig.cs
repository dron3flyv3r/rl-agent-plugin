using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Top-level training configuration resource combining the algorithm definition
/// and optional hyperparameter schedules.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLTrainingConfig : Resource
{
    private const string AlgorithmType = nameof(RLAlgorithmConfig);

    /// <summary>
    /// Algorithm-specific training settings (PPO, SAC, or custom trainer config).
    /// </summary>
    [Export(PropertyHint.ResourceType, AlgorithmType)]
    public RLAlgorithmConfig? Algorithm { get; set; }

    /// <summary>
    /// Optional dynamic schedules that override selected hyperparameters over time.
    /// </summary>
    [Export]
    public RLScheduleConfig? Schedules { get; set; }

    /// <summary>
    /// Evaluates all non-null schedules and writes their results into <paramref name="config"/>.
    /// Called by TrainingBootstrap once per gradient update, before TryUpdate().
    /// </summary>
    /// <param name="config">Mutable runtime trainer settings to override.</param>
    /// <param name="ctx">Current schedule evaluation context.</param>
    internal void ApplySchedules(RLTrainerConfig config, ScheduleContext ctx)
    {
        Schedules?.ApplyTo(config, ctx);
    }

    /// <summary>
    /// Converts inspector configuration into normalized runtime trainer settings.
    /// </summary>
    /// <returns>
    /// A populated <see cref="RLTrainerConfig"/> when <see cref="Algorithm"/> is assigned;
    /// otherwise <see langword="null"/>.
    /// </returns>
    public RLTrainerConfig? ToTrainerConfig()
    {
        if (Algorithm is null) return null;
        var config = new RLTrainerConfig();
        Algorithm.ApplyTo(config);
        return config;
    }
}
