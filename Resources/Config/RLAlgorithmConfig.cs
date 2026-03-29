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

    /// <summary>Writes all settings from this config into <paramref name="config"/>.</summary>
    internal abstract void ApplyTo(RLTrainerConfig config);
}
