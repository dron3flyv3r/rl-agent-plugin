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
