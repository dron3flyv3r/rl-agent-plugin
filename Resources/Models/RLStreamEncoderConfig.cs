using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Associates a named observation stream with a specific encoder architecture.
/// Add one of these to <see cref="RLNetworkGraph.StreamEncoders"/> for each stream
/// that needs non-default processing.
/// <list type="bullet">
///   <item>For <b>image</b> streams: set <see cref="CnnEncoder"/>.</item>
///   <item>For <b>vector</b> streams with a custom sub-MLP: set <see cref="VectorEncoder"/>.</item>
///   <item>Leave both null to fall back to the graph's default trunk layers.</item>
/// </list>
/// </summary>
[GlobalClass]
[Tool]
public partial class RLStreamEncoderConfig : Resource
{
    /// <summary>
    /// Name of the observation stream this config applies to.
    /// Must match the <c>name</c> argument passed to
    /// <see cref="ObservationBuffer.AddVector"/> or <see cref="ObservationBuffer.AddImage"/>.
    /// </summary>
    [Export] public string StreamName { get; set; } = string.Empty;

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
