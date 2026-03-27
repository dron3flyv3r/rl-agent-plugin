namespace RlAgentPlugin.Runtime;

/// <summary>
/// Describes a single named observation stream within an <see cref="ObservationSpec"/>.
/// </summary>
public sealed record ObservationStreamSpec(
    /// <summary>Name of the stream (e.g. "position", "camera").</summary>
    string Name,
    /// <summary>Modality of this stream.</summary>
    ObservationStreamKind Kind,
    /// <summary>Number of floats this stream occupies in the flat observation array.</summary>
    int FlatSize,
    /// <summary>Image width in pixels (0 for Vector streams).</summary>
    int Width,
    /// <summary>Image height in pixels (0 for Vector streams).</summary>
    int Height,
    /// <summary>Number of image channels (0 for Vector streams).</summary>
    int Channels,
    /// <summary>
    /// Optional encoder config bound directly to this stream via its source sensor.
    /// When set, the network uses this config directly to build the stream's encoder.
    /// </summary>
    RLStreamEncoderConfig? EncoderConfig = null);
