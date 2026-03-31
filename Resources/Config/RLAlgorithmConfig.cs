using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Base class for algorithm configuration resources.
/// Assign an <see cref="RLPPOConfig"/> or <see cref="RLSACConfig"/> asset to
/// <see cref="RLTrainingConfig.Algorithm"/> in the Inspector.
///
/// To use a custom trainer: subclass this, add [GlobalClass], set
/// <see cref="RLTrainerConfig.Algorithm"/> to <see cref="RLAlgorithmKind.Custom"/>
/// and fill <see cref="RLTrainerConfig.CustomTrainerId"/> inside <see cref="ApplyTo"/>.
/// </summary>
[GlobalClass]
[Tool]
public abstract partial class RLAlgorithmConfig : Resource
{
    /// <summary>
    /// Frequency (in environment steps) for console/dashboard status writes during training.
    /// </summary>
    [Export(PropertyHint.Range, "1,1000,1,or_greater")] public int StatusWriteIntervalSteps  { get; set; } = 32;

    /// <summary>The algorithm this config represents. Implemented by each concrete subclass.</summary>
    public virtual RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.Custom;

    /// <summary>Whether this algorithm supports discrete (integer) action spaces.</summary>
    public virtual bool SupportsDiscreteActions => true;

    /// <summary>Whether this algorithm supports continuous (float) action spaces.</summary>
    public virtual bool SupportsContinuousActions => true;

    /// <summary>
    /// Whether this algorithm works correctly when multiple independent policy groups
    /// are trained in the same scene (e.g. competitive or cooperative multi-agent setups).
    /// All algorithms support shared-policy multi-agent (many agents, one policy group),
    /// but native cross-agent communication or centralized training requires specialized
    /// MARL algorithms (QMIX, MADDPG, etc.).
    /// </summary>
    public virtual bool SupportsMultiAgent => true;

    /// <summary>
    /// True for on-policy algorithms that collect a rollout, update, then discard it (e.g. PPO, A2C).
    /// False for off-policy algorithms that store experience in a replay buffer (e.g. SAC, DQN).
    /// </summary>
    public virtual bool IsOnPolicy => true;

    /// <summary>Writes all settings from this config into <paramref name="config"/>.</summary>
    internal abstract void ApplyTo(RLTrainerConfig config);
}
