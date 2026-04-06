namespace RlAgentPlugin.Runtime;

/// <summary>
/// Normalized runtime trainer settings derived from <see cref="RLTrainingConfig"/>.
/// This is no longer an inspector-assigned resource.
/// </summary>
public partial class RLTrainerConfig
{
    // ── Algorithm selection ─────────────────────────────────────────────────
    public RLAlgorithmKind Algorithm { get; set; } = RLAlgorithmKind.Custom;
    /// <summary>
    /// Used only when <see cref="Algorithm"/> is <see cref="RLAlgorithmKind.Custom"/>.
    /// Must match the key passed to <see cref="TrainerFactory.Register"/>.
    /// </summary>
    public string CustomTrainerId { get; set; } = string.Empty;

    // ── PPO hyperparameters ─────────────────────────────────────────────────
    [HpoGroup("PPO / A2C")] public int RolloutLength { get; set; } = 256;
    [HpoGroup("PPO / A2C")] public int EpochsPerUpdate { get; set; } = 4;
    [HpoGroup("PPO / A2C")] public int PpoMiniBatchSize { get; set; } = 64;
    [HpoGroup("General")]   public float LearningRate { get; set; } = 0.0005f;
    [HpoGroup("General")]   public float Gamma { get; set; } = 0.99f;
    [HpoGroup("PPO / A2C")] public float GaeLambda { get; set; } = 0.95f;
    [HpoGroup("PPO / A2C")] public float ClipEpsilon { get; set; } = 0.2f;
    [HpoGroup("General")]   public float MaxGradientNorm { get; set; } = 0.5f;
    [HpoGroup("PPO / A2C")] public float ValueLossCoefficient { get; set; } = 0.5f;
    [HpoGroup("PPO / A2C")] public bool UseValueClipping { get; set; } = true;
    [HpoGroup("PPO / A2C")] public float ValueClipEpsilon { get; set; } = 0.2f;
    [HpoGroup("PPO / A2C")] public float EntropyCoefficient { get; set; } = 0.01f;
    [HpoGroup("General")]   public int StatusWriteIntervalSteps { get; set; } = 32;

    // ── Recurrent BPTT ────────────────────────────────────────────────────
    /// <summary>
    /// Truncated-BPTT sequence length for recurrent policies (LSTM/GRU).
    /// Transitions within an episode are grouped into subsequences of this length.
    /// Only used when the policy network contains recurrent trunk layers.
    /// </summary>
    [HpoGroup("PPO / A2C")] public int BpttLength { get; set; } = 16;

    // ── DQN hyperparameters (ignored by other algorithms) ─────────────────
    [HpoGroup("General")] public int ReplayBufferCapacity { get; set; } = 100_000;
    [HpoGroup("DQN")]     public int DqnBatchSize { get; set; } = 64;
    [HpoGroup("DQN")]     public int DqnWarmupSteps { get; set; } = 1_000;
    [HpoGroup("DQN")]     public float DqnEpsilonStart { get; set; } = 1.0f;
    [HpoGroup("DQN")]     public float DqnEpsilonEnd { get; set; } = 0.05f;
    [HpoGroup("DQN")]     public int DqnEpsilonDecaySteps { get; set; } = 50_000;
    [HpoGroup("DQN")]     public int DqnTargetUpdateInterval { get; set; } = 1_000;
    [HpoGroup("DQN")]     public bool DqnUseDouble { get; set; } = true;
    /// <summary>Dyna-Q: imagined model-based updates per real step (0 = disabled).</summary>
    [HpoGroup("DQN")] public int DynaModelUpdatesPerStep { get; set; } = 0;
    /// <summary>Dyna-Q: learning rate for the transition model.</summary>
    [HpoGroup("DQN")] public float DynaModelLearningRate { get; set; } = 0.001f;

    // ── MCTS hyperparameters ───────────────────────────────────────────────
    [HpoGroup("MCTS")] public int MctsNumSimulations { get; set; } = 50;
    [HpoGroup("MCTS")] public int MctsMaxSearchDepth { get; set; } = 20;
    [HpoGroup("MCTS")] public int MctsRolloutDepth { get; set; } = 10;
    [HpoGroup("MCTS")] public float MctsExplorationConstant { get; set; } = 1.414f;
    [HpoGroup("MCTS")] public float MctsGamma { get; set; } = 0.99f;

    // ── SAC hyperparameters (ignored by PPO) ───────────────────────────────
    [HpoGroup("SAC")] public int SacBatchSize { get; set; } = 256;
    [HpoGroup("SAC")] public int SacWarmupSteps { get; set; } = 1_000;
    [HpoGroup("SAC")] public float SacTau { get; set; } = 0.005f;
    [HpoGroup("SAC")] public float SacInitAlpha { get; set; } = 0.2f;
    [HpoGroup("SAC")] public bool SacAutoTuneAlpha { get; set; } = true;
    [HpoGroup("SAC")] public int SacUpdateEverySteps { get; set; } = 1;
    [HpoGroup("SAC")] public float SacTargetEntropyFraction { get; set; } = 0.5f;
    [HpoGroup("SAC")] public float SacContinuousTargetEntropyScale { get; set; } = 1.0f;
    [HpoGroup("SAC")] public bool SacUseContinuousTargetEntropyOverride { get; set; } = false;
    [HpoGroup("SAC")] public float SacContinuousTargetEntropyOverride { get; set; } = 0.0f;
    /// <summary>
    /// Gradient updates per environment step (UTD ratio).
    /// 0 = auto: equals the number of active data sources (master + connected workers).
    /// </summary>
    [HpoGroup("SAC")] public int SacUpdatesPerStep { get; set; } = 0;
}
