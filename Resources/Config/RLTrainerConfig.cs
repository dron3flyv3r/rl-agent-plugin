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
    public int RolloutLength { get; set; } = 256;
    public int EpochsPerUpdate { get; set; } = 4;
    public int PpoMiniBatchSize { get; set; } = 64;
    public float LearningRate { get; set; } = 0.0005f;
    public float Gamma { get; set; } = 0.99f;
    public float GaeLambda { get; set; } = 0.95f;
    public float ClipEpsilon { get; set; } = 0.2f;
    public float MaxGradientNorm { get; set; } = 0.5f;
    public float ValueLossCoefficient { get; set; } = 0.5f;
    public bool UseValueClipping { get; set; } = true;
    public float ValueClipEpsilon { get; set; } = 0.2f;
    public float EntropyCoefficient { get; set; } = 0.01f;
    public int StatusWriteIntervalSteps { get; set; } = 32;

    // ── DQN hyperparameters (ignored by other algorithms) ─────────────────
    public int DqnBatchSize { get; set; } = 64;
    public int DqnWarmupSteps { get; set; } = 1_000;
    public float DqnEpsilonStart { get; set; } = 1.0f;
    public float DqnEpsilonEnd { get; set; } = 0.05f;
    public int DqnEpsilonDecaySteps { get; set; } = 50_000;
    public int DqnTargetUpdateInterval { get; set; } = 1_000;
    public bool DqnUseDouble { get; set; } = true;
    /// <summary>Dyna-Q: imagined model-based updates per real step (0 = disabled).</summary>
    public int DynaModelUpdatesPerStep { get; set; } = 0;
    /// <summary>Dyna-Q: learning rate for the transition model.</summary>
    public float DynaModelLearningRate { get; set; } = 0.001f;

    // ── MCTS hyperparameters ───────────────────────────────────────────────
    public int MctsNumSimulations { get; set; } = 50;
    public int MctsMaxSearchDepth { get; set; } = 20;
    public int MctsRolloutDepth { get; set; } = 10;
    public float MctsExplorationConstant { get; set; } = 1.414f;
    public float MctsGamma { get; set; } = 0.99f;

    // ── SAC hyperparameters (ignored by PPO) ───────────────────────────────
    public int ReplayBufferCapacity { get; set; } = 100_000;
    public int SacBatchSize { get; set; } = 256;
    public int SacWarmupSteps { get; set; } = 1_000;
    public float SacTau { get; set; } = 0.005f;
    public float SacInitAlpha { get; set; } = 0.2f;
    public bool SacAutoTuneAlpha { get; set; } = true;
    public int SacUpdateEverySteps { get; set; } = 1;
    public float SacTargetEntropyFraction { get; set; } = 0.5f;
    /// <summary>
    /// Gradient updates per environment step (UTD ratio).
    /// 0 = auto: equals the number of active data sources (master + connected workers).
    /// </summary>
    public int SacUpdatesPerStep { get; set; } = 0;
}
