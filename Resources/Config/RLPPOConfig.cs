using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// PPO (Proximal Policy Optimization) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLPPOConfig : RLAlgorithmConfig
{
    /// <summary>Number of environment steps collected before each PPO update.</summary>
    [Export] public int   RolloutLength         { get; set; } = 256;
    /// <summary>How many optimization passes are run over each rollout.</summary>
    [Export] public int   EpochsPerUpdate        { get; set; } = 4;
    /// <summary>Mini-batch size used during PPO optimization epochs.</summary>
    [Export] public int   MiniBatchSize          { get; set; } = 64;
    /// <summary>Optimizer learning rate.</summary>
    [Export] public float LearningRate           { get; set; } = 0.0005f;
    /// <summary>Discount factor for future rewards.</summary>
    [Export] public float Gamma                  { get; set; } = 0.99f;
    /// <summary>GAE lambda parameter controlling bias/variance tradeoff.</summary>
    [Export] public float GaeLambda              { get; set; } = 0.95f;
    /// <summary>PPO policy clipping epsilon.</summary>
    [Export] public float ClipEpsilon            { get; set; } = 0.2f;
    /// <summary>Gradient norm clip value.</summary>
    [Export] public float MaxGradientNorm        { get; set; } = 0.5f;
    /// <summary>Scale factor applied to value-function loss.</summary>
    [Export] public float ValueLossCoefficient   { get; set; } = 0.5f;
    /// <summary>Enables clipping of value-function updates.</summary>
    [Export] public bool  UseValueClipping       { get; set; } = true;
    /// <summary>Value-function clipping epsilon (used only when value clipping is enabled).</summary>
    [Export] public float ValueClipEpsilon       { get; set; } = 0.2f;
    /// <summary>Entropy bonus coefficient for exploration.</summary>
    [Export] public float EntropyCoefficient     { get; set; } = 0.01f;

    public override RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.PPO;

    /// <inheritdoc />
    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm               = RLAlgorithmKind.PPO;
        config.RolloutLength           = RolloutLength;
        config.EpochsPerUpdate         = EpochsPerUpdate;
        config.PpoMiniBatchSize        = MiniBatchSize;
        config.LearningRate            = LearningRate;
        config.Gamma                   = Gamma;
        config.GaeLambda               = GaeLambda;
        config.ClipEpsilon             = ClipEpsilon;
        config.MaxGradientNorm         = MaxGradientNorm;
        config.ValueLossCoefficient    = ValueLossCoefficient;
        config.UseValueClipping        = UseValueClipping;
        config.ValueClipEpsilon        = ValueClipEpsilon;
        config.EntropyCoefficient      = EntropyCoefficient;
        config.StatusWriteIntervalSteps  = StatusWriteIntervalSteps;
    }
}
