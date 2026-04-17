using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>Node type used in the public genome UI snapshot.</summary>
public enum UINodeRole { Input = 0, Bias = 1, Hidden = 2, Output = 3 }

/// <summary>Lightweight, allocation-free snapshot of a single genome node.</summary>
public readonly struct UINodeInfo
{
    public readonly int        Id;
    public readonly UINodeRole Role;
    public UINodeInfo(int id, UINodeRole role) { Id = id; Role = role; }
}

/// <summary>Lightweight, allocation-free snapshot of a single genome connection.</summary>
public readonly struct UIConnectionInfo
{
    public readonly int   InNode;
    public readonly int   OutNode;
    public readonly float Weight;
    public readonly bool  Enabled;
    public UIConnectionInfo(int inNode, int outNode, float weight, bool enabled)
    { InNode = inNode; OutNode = outNode; Weight = weight; Enabled = enabled; }
}

/// <summary>
/// Immutable snapshot of the all-time champion genome plus population-level statistics.
/// Created once per generation by <see cref="INeatDataFeed.GetSnapshot"/> and read
/// by the <see cref="RLUIGenericStatus"/> overlay on the main thread.
/// </summary>
public sealed class NeatGenomeSnapshot
{
    public List<UINodeInfo>       Nodes          { get; init; } = new();
    public List<UIConnectionInfo> Connections    { get; init; } = new();
    public int   Generation    { get; init; }
    public float BestFitness   { get; init; }
    public float MeanFitness   { get; init; }
    public int   PopulationSize { get; init; }
    public int   AliveCount    { get; init; }
    public int   SpeciesCount  { get; init; }
    public int   InputCount    { get; init; }
    public int   OutputCount   { get; init; }
}

/// <summary>
/// Implemented by <see cref="NeatTrainer"/>. Exposes a read-only window into the
/// current population so the <see cref="RLUIGenericStatus"/> overlay can render
/// live training data without reaching into trainer internals.
/// </summary>
public interface INeatDataFeed
{
    /// <summary>
    /// Returns a cached snapshot of the current population state.
    /// The snapshot is rebuilt only when the generation counter advances,
    /// so polling this every frame is allocation-free after the first call per generation.
    /// </summary>
    NeatGenomeSnapshot GetSnapshot();
}
