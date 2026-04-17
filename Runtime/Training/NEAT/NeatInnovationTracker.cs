using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Tracks innovation numbers for NEAT structural mutations.
/// Per-trainer instance (not global static) to avoid cross-run contamination.
///
/// The same structural mutation should keep the same historical marking across the
/// lifetime of the population so crossover can correctly align homologous genes.
/// This tracker therefore keeps a persistent mapping for connection innovations and
/// split-node IDs instead of resetting it each generation.
/// </summary>
internal sealed class NeatInnovationTracker
{
    private int _nextInnovation;
    private int _nextNodeId;

    // Historical innovation maps (kept for the lifetime of the trainer)
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
    /// Called at the start of each new generation.
    /// Historical innovation maps intentionally persist across generations so that
    /// independently rediscovered structures retain the same innovation numbers.
    /// </summary>
    public void StartGeneration()
    {
        // Intentionally left blank.
    }

    /// <summary>Serializes tracker state for checkpoint resumption.</summary>
    public (int nextInnovation, int nextNodeId) SaveState() =>
        (_nextInnovation, _nextNodeId);
}
