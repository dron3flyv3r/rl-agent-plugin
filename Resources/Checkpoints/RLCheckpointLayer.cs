namespace RlAgentPlugin.Runtime;

/// <summary>
/// Describes a single trunk layer as stored in checkpoint metadata.
/// Layer-specific properties are populated based on <see cref="Type"/>:
/// <list type="bullet">
///   <item><term>dense</term><description><see cref="Size"/> and <see cref="Activation"/></description></item>
///   <item><term>dropout</term><description><see cref="Rate"/></description></item>
///   <item><term>layer_norm</term><description>(no extra properties)</description></item>
///   <item><term>flatten</term><description>(no extra properties)</description></item>
/// </list>
/// </summary>
public sealed class RLCheckpointLayer
{
    /// <summary>Layer type: "dense", "dropout", "layer_norm", or "flatten".</summary>
    public string Type { get; set; } = "dense";

    /// <summary>Output size. Dense layers only.</summary>
    public int Size { get; set; }

    /// <summary>Activation function name: "tanh" or "relu". Dense layers only.</summary>
    public string Activation { get; set; } = "tanh";

    /// <summary>Drop probability in [0, 1). Dropout layers only.</summary>
    public float Rate { get; set; }
}
