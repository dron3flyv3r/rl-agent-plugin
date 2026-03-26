using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Defines the architecture of a convolutional encoder used for image observation streams.
/// Assign one of these to an <see cref="RLStreamEncoderConfig"/> to enable CNN processing
/// for the corresponding named image stream.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLCnnEncoderDef : Resource
{
    /// <summary>Number of output filters for each convolutional layer.</summary>
    [Export] public int[] FilterCounts { get; set; } = new[] { 32, 64 };

    /// <summary>Square kernel size (height == width) for each convolutional layer.</summary>
    [Export] public int[] KernelSizes { get; set; } = new[] { 3, 3 };

    /// <summary>Stride for each convolutional layer.</summary>
    [Export] public int[] Strides { get; set; } = new[] { 2, 2 };

    /// <summary>
    /// Dimensionality of the linear projection applied after flattening the final
    /// conv output. This is the size of the embedding fed into the shared trunk.
    /// </summary>
    [Export(PropertyHint.Range, "8,1024,8,or_greater")]
    public int OutputSize { get; set; } = 64;
}
