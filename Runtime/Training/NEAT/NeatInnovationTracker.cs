using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Tracks innovation numbers for NEAT structural mutations.
/// Per-trainer instance (not global static) to avoid cross-run contamination.
///
/// Within a generation, the same (inNode→outNode) pair receives the same innovation number
/// so that crossover can correctly align genes. The per-generation cache is cleared at the
/// start of each generation via <see cref="StartGeneration"/>.
/// </summary>
internal sealed class NeatInnovationTracker
{
    private int _nextInnovation;
    private int _nextNodeId;

    // Per-generation caches (cleared each generation)
    private readonly Dictionary<(int, int), int> _connectionCache = new();
    private readonly Dictionary<int, int> _nodeSplitCache = new(); // splitConnectionInnovation → new node id

    public int NextInnovation => _nextInnovation;
    public int NextNodeId => _nextNodeId;

    /// <param name="startNodeId">
    /// Should be set to inputCount + 1 (bias) + outputCount so new hidden node IDs
    /// don't collide with the initial fixed node IDs.
    /// </param>
    public NeatInnovationTracker(int initialInnovations, int startNodeId)
    {
        _nextInnovation = initialInnovations;
        _nextNodeId     = startNodeId;
    }

    /// <summary>
    /// Returns the innovation number for a connection from <paramref name="inNode"/> to
    /// <paramref name="outNode"/>. If this structural change has already happened in the
    /// current generation, returns the cached innovation number (same innovation =
    /// same gene for crossover alignment). Otherwise allocates a new one.
    /// </summary>
    public int GetOrCreateConnectionInnovation(int inNode, int outNode)
    {
        var key = (inNode, outNode);
        if (_connectionCache.TryGetValue(key, out int existing))
            return existing;

        int innov = _nextInnovation++;
        _connectionCache[key] = innov;
        return innov;
    }

    /// <summary>
    /// Returns the new hidden node ID created by splitting the connection with
    /// <paramref name="splitConnectionInnovation"/>. If the same connection was
    /// already split in this generation, returns the same node ID.
    /// </summary>
    public int GetOrCreateSplitNodeId(int splitConnectionInnovation)
    {
        if (_nodeSplitCache.TryGetValue(splitConnectionInnovation, out int existing))
            return existing;

        int nodeId = _nextNodeId++;
        _nodeSplitCache[splitConnectionInnovation] = nodeId;
        return nodeId;
    }

    /// <summary>
    /// Call at the start of each new generation to clear the per-generation caches.
    /// The global counters (_nextInnovation, _nextNodeId) keep incrementing.
    /// </summary>
    public void StartGeneration()
    {
        _connectionCache.Clear();
        _nodeSplitCache.Clear();
    }

    /// <summary>Serializes tracker state for checkpoint resumption.</summary>
    public (int nextInnovation, int nextNodeId) SaveState() =>
        (_nextInnovation, _nextNodeId);
}
