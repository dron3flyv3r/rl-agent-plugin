using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Defines the encoder architecture for a single observation stream.
/// Assign this resource directly to a sensor node (e.g. <see cref="RLCameraSensor2D.EncoderConfig"/>).
/// The sensor binds it to its stream automatically — no string name needed.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLStreamEncoderConfig : Resource
{
    /// <summary>
    /// CNN encoder definition for image streams.
    /// Leave null for vector streams or to use the default MLP path.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLCnnEncoderDef))]
    public RLCnnEncoderDef? CnnEncoder { get; set; }

    /// <summary>
    /// Optional separate MLP trunk to apply after the (optional) CNN projection,
    /// before merging this stream's embedding with others.
    /// Leave null to skip the per-stream MLP entirely.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLNetworkGraph))]
    public RLNetworkGraph? VectorEncoder { get; set; }
}
