using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// DQN (Deep Q-Network) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
///
/// DQN is an off-policy, value-based algorithm that supports <b>discrete action spaces only</b>.
/// It uses a replay buffer and a periodically updated target network for stable training.
/// Enable <see cref="UseDoubleDqn"/> (default: true) to reduce Q-value overestimation.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLDQNConfig : RLAlgorithmConfig
{
    /// <summary>Optimizer learning rate.</summary>
    [Export(PropertyHint.Range, "0.0001,1.0,0.0001")] public float LearningRate { get; set; } = 0.0005f;
    /// <summary>Discount factor for future rewards.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float Gamma { get; set; } = 0.99f;
    /// <summary>Gradient norm clip value (0 = disabled).</summary>
    [Export(PropertyHint.Range, "0.0,10.0,0.1")] public float MaxGradientNorm { get; set; } = 10f;
    /// <summary>Maximum replay buffer capacity in transitions.</summary>
    [Export(PropertyHint.Range, "1,1000000,1,or_greater")] public int ReplayBufferCapacity { get; set; } = 50_000;
    /// <summary>Number of transitions sampled per gradient update.</summary>
    [Export(PropertyHint.Range, "1,512,1,or_greater")] public int BatchSize { get; set; } = 64;
    /// <summary>Minimum transitions collected before training begins.</summary>
    [Export(PropertyHint.Range, "0,100000,1,or_greater")] public int WarmupSteps { get; set; } = 1_000;
    /// <summary>Initial epsilon for epsilon-greedy exploration (1.0 = fully random).</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float EpsilonStart { get; set; } = 1.0f;
    /// <summary>Final epsilon after decay is complete.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.001")] public float EpsilonEnd { get; set; } = 0.05f;
    /// <summary>Number of environment steps over which epsilon decays from start to end.</summary>
    [Export(PropertyHint.Range, "1,1000000,1,or_greater")] public int EpsilonDecaySteps { get; set; } = 50_000;
    /// <summary>
    /// How often (in environment steps) the target network is hard-copied from the online network.
    /// Lower values track the online network more closely but can destabilize training.
    /// </summary>
    [Export(PropertyHint.Range, "1,100000,1,or_greater")] public int TargetUpdateInterval { get; set; } = 1_000;
    /// <summary>
    /// Enable Double DQN (recommended).
    /// Uses the online network to select the action and the target network to evaluate it,
    /// reducing Q-value overestimation bias.
    /// </summary>
    [Export] public bool UseDoubleDqn { get; set; } = true;

    // ── Dyna-Q planning extension ─────────────────────────────────────────────
    /// <summary>
    /// Number of additional "imagined" Q-learning updates per real environment step,
    /// using the learned world model. 0 (default) disables Dyna-Q planning entirely.
    /// Higher values improve sample efficiency but increase compute per step.
    /// Recommended range: 5–50 for small environments.
    /// </summary>
    [Export(PropertyHint.Range, "0,100,1,or_greater")] public int DynaModelUpdatesPerStep { get; set; } = 0;
    /// <summary>Learning rate for the Dyna-Q world model (transition predictor).</summary>
    [Export(PropertyHint.Range, "0.0001,1.0,0.0001")] public float DynaModelLearningRate { get; set; } = 0.001f;

    public override RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.DQN;
    /// <summary>DQN only supports discrete action spaces.</summary>
    public override bool SupportsDiscreteActions => true;
    /// <summary>DQN does not support continuous action spaces. Use SAC or PPO instead.</summary>
    public override bool SupportsContinuousActions => false;
    public override bool SupportsMultiAgent => true;
    public override bool IsOnPolicy => false;

    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm              = RLAlgorithmKind.DQN;
        config.LearningRate           = LearningRate;
        config.Gamma                  = Gamma;
        config.MaxGradientNorm        = MaxGradientNorm;
        config.ReplayBufferCapacity   = ReplayBufferCapacity;
        config.DqnBatchSize           = BatchSize;
        config.DqnWarmupSteps         = WarmupSteps;
        config.DqnEpsilonStart        = EpsilonStart;
        config.DqnEpsilonEnd          = EpsilonEnd;
        config.DqnEpsilonDecaySteps   = EpsilonDecaySteps;
        config.DqnTargetUpdateInterval = TargetUpdateInterval;
        config.DqnUseDouble                = UseDoubleDqn;
        config.DynaModelUpdatesPerStep     = DynaModelUpdatesPerStep;
        config.DynaModelLearningRate       = DynaModelLearningRate;
        config.StatusWriteIntervalSteps    = StatusWriteIntervalSteps;
    }
}
