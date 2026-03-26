using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Describes the complete observation schema for a policy group: one or more named
/// streams of possibly different modalities. The flat float[] observation array produced
/// by <see cref="ObservationBuffer"/> is the concatenation of all streams in order.
/// </summary>
public sealed class ObservationSpec
{
    public IReadOnlyList<ObservationStreamSpec> Streams { get; }

    /// <summary>Total number of floats across all streams (== observation array length).</summary>
    public int TotalSize { get; }

    public ObservationSpec(IReadOnlyList<ObservationStreamSpec> streams)
    {
        Streams = streams;
        var total = 0;
        foreach (var s in streams) total += s.FlatSize;
        TotalSize = total;
    }

    /// <summary>
    /// Creates a legacy single-stream spec from a flat observation size.
    /// Used as a backward-compatible fallback when agents call only <c>buffer.Add()</c>.
    /// </summary>
    public static ObservationSpec Flat(int size) =>
        new(new[] { new ObservationStreamSpec("obs", ObservationStreamKind.Vector, size, 0, 0, 0) });
}
