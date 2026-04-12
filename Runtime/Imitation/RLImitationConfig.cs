namespace RlAgentPlugin.Runtime.Imitation;

/// <summary>
/// Hyperparameters for behavior cloning (BC) training.
/// Editor-only; not a Godot Resource.
/// </summary>
public sealed class RLImitationConfig
{
    /// <summary>Number of full passes over the dataset.</summary>
    public int Epochs { get; set; } = 20;

    /// <summary>Mini-batch size for each gradient step.</summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>Adam learning rate.</summary>
    public float LearningRate { get; set; } = 3e-4f;

    /// <summary>Maximum gradient norm for clipping. 0 disables clipping.</summary>
    public float MaxGradientNorm { get; set; } = 0.5f;

    /// <summary>Shuffle the dataset at the start of each epoch.</summary>
    public bool ShuffleEachEpoch { get; set; } = true;
}
