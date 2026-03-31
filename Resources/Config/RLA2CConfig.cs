using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// A2C (Advantage Actor-Critic) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
///
/// A2C is an on-policy algorithm that supports both discrete and continuous action spaces.
/// It is similar to PPO but simpler: no clipping, no multiple epochs, single-pass update.
/// Use A2C when you want a lightweight policy-gradient baseline.
/// Use PPO when you need more stable training with larger rollouts.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLA2CConfig : RLAlgorithmConfig
{
    /// <summary>Number of environment steps collected before each A2C update.</summary>
    [Export(PropertyHint.Range, "1,4096,1,or_greater")] public int RolloutLength { get; set; } = 64;
    /// <summary>Optimizer learning rate.</summary>
    [Export(PropertyHint.Range, "0.0001,1.0,0.0001")] public float LearningRate { get; set; } = 0.0007f;
    /// <summary>Discount factor for future rewards.</summary>
    [Export(PropertyHint.Range, "0.0001,1.0,0.01")] public float Gamma { get; set; } = 0.99f;
    /// <summary>GAE lambda parameter controlling bias/variance trade-off (1.0 = full Monte Carlo).</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float GaeLambda { get; set; } = 1.0f;
    /// <summary>Gradient norm clip value.</summary>
    [Export(PropertyHint.Range, "0.0,10.0,0.1")] public float MaxGradientNorm { get; set; } = 0.5f;
    /// <summary>Scale factor applied to the value-function loss.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float ValueLossCoefficient { get; set; } = 0.5f;
    /// <summary>Entropy bonus coefficient for exploration.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float EntropyCoefficient { get; set; } = 0.01f;

    public override RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.A2C;
    public override bool SupportsDiscreteActions => true;
    public override bool SupportsContinuousActions => true;
    public override bool SupportsMultiAgent => true;
    public override bool IsOnPolicy => true;

    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm             = RLAlgorithmKind.A2C;
        config.RolloutLength         = RolloutLength;
        config.EpochsPerUpdate       = 1;                   // A2C always does a single pass
        config.PpoMiniBatchSize      = RolloutLength;       // full rollout as one batch
        config.LearningRate          = LearningRate;
        config.Gamma                 = Gamma;
        config.GaeLambda             = GaeLambda;
        config.ClipEpsilon           = float.MaxValue / 2f; // effectively no clipping
        config.MaxGradientNorm       = MaxGradientNorm;
        config.ValueLossCoefficient  = ValueLossCoefficient;
        config.UseValueClipping      = false;
        config.ValueClipEpsilon      = 0f;
        config.EntropyCoefficient    = EntropyCoefficient;
        config.StatusWriteIntervalSteps = StatusWriteIntervalSteps;
    }
}
