using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// SAC (Soft Actor-Critic) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLSACConfig : RLAlgorithmConfig
{
    /// <summary>Optimizer learning rate for policy and critic networks.</summary>
    [Export(PropertyHint.Range, "0.0001,1.0,0.0001")] public float LearningRate { get; set; } = 0.0003f;
    /// <summary>Discount factor for future rewards.</summary>
    [Export(PropertyHint.Range, "0.0001,1.0,0.01")] public float Gamma { get; set; } = 0.99f;
    /// <summary>Gradient norm clip value.</summary>
    [Export(PropertyHint.Range, "0.0,10.0,0.1")] public float MaxGradientNorm { get; set; } = 0.5f;
    /// <summary>Maximum replay buffer capacity in transitions.</summary>
    [Export(PropertyHint.Range, "1,1000000,1,or_greater")] public int ReplayBufferCapacity { get; set; } = 100_000;
    /// <summary>Number of replay samples used for each SAC update.</summary>
    [Export(PropertyHint.Range, "1,1024,1,or_greater")] public int BatchSize { get; set; } = 256;
    /// <summary>Minimum transitions to collect before training begins.</summary>
    [Export(PropertyHint.Range, "0,1000000,1,or_greater")] public int WarmupSteps { get; set; } = 1_000;
    /// <summary>Target-network Polyak averaging coefficient.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.001")] public float Tau { get; set; } = 0.005f;
    /// <summary>Initial entropy temperature (alpha).</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float InitAlpha { get; set; } = 0.2f;
    /// <summary>Automatically tune entropy temperature during training.</summary>
    [Export] public bool AutoTuneAlpha { get; set; } = true;
    /// <summary>Run one gradient update every N environment steps.</summary>
    [Export(PropertyHint.Range, "1,100,1,or_greater")] public int UpdateEverySteps { get; set; } = 1;
    /// <summary>
    /// Number of gradient updates performed per environment step (Updates-To-Data ratio).
    /// 0 (default) = auto: scales with the number of active data sources (1 master + N workers).
    /// Set to a fixed value (e.g. 4) to override for reproducible experiments.
    /// Higher values improve sample efficiency at the cost of more compute per step.
    /// </summary>
    [Export(PropertyHint.Range, "0,32,or_greater")] public int UpdatesPerStep { get; set; } = 0;
    /// <summary>
    /// Fraction of maximum entropy used as the discrete-action target entropy.
    /// 1.0 = fully random (uniform policy); 0.5 = half of maximum entropy.
    /// Lower values make the policy converge to more deterministic behaviour.
    /// Only used when the action space is discrete (ignored for continuous).
    /// </summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float TargetEntropyFraction { get; set; } = 0.5f;

    public override RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.SAC;

    /// <inheritdoc />
    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm = RLAlgorithmKind.SAC;
        config.LearningRate = LearningRate;
        config.Gamma = Gamma;
        config.MaxGradientNorm = MaxGradientNorm;
        config.ReplayBufferCapacity = ReplayBufferCapacity;
        config.SacBatchSize = BatchSize;
        config.SacWarmupSteps = WarmupSteps;
        config.SacTau = Tau;
        config.SacInitAlpha = InitAlpha;
        config.SacAutoTuneAlpha = AutoTuneAlpha;
        config.SacUpdateEverySteps = UpdateEverySteps;
        config.SacTargetEntropyFraction = TargetEntropyFraction;
        config.SacUpdatesPerStep = UpdatesPerStep;
        config.StatusWriteIntervalSteps = StatusWriteIntervalSteps;
    }
}
