using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Executes a variable-topology feed-forward pass for a single NEAT genome.
/// One instance is held per agent slot in <see cref="NeatTrainer"/>.
///
/// The network is rebuilt automatically whenever the genome's <c>TopoDirty</c>
/// flag is set (structural mutations, crossover, or deserialisation).
///
/// Cyclic connections (which NEAT crossover can sometimes produce) are silently
/// excluded from the topological ordering — those nodes receive zero activation
/// rather than crashing.
/// </summary>
internal sealed class GenomeNetwork
{
    private readonly NeatGenome _genome;
    private readonly int _inputCount;
    private readonly int _outputCount;

    // Ordered input node IDs (position-stable)
    private int[] _inputNodeIds  = Array.Empty<int>();
    // Ordered output node IDs (position-stable)
    private int[] _outputNodeIds = Array.Empty<int>();
    // Bias node ID (-1 if absent)
    private int   _biasNodeId    = -1;

    // Activation buffer, indexed by node ID.  Sized to maxNodeId + 1.
    private float[] _activations = Array.Empty<float>();

    // Per-node incoming connection list (rebuilt with topology)
    // Key = node ID, Value = list of (inNodeId, weight)
    private Dictionary<int, List<(int inNode, float weight)>> _incoming = new();

    public GenomeNetwork(NeatGenome genome, int inputCount, int outputCount)
    {
        _genome      = genome;
        _inputCount  = inputCount;
        _outputCount = outputCount;
        RebuildTopology();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>
    /// Runs a forward pass on <paramref name="inputs"/> and returns the raw output
    /// activations. For discrete actions, apply softmax externally. For continuous
    /// actions, apply tanh externally.
    /// </summary>
    public float[] Forward(float[] inputs)
    {
        if (_genome.TopoDirty)
            RebuildTopology();

        // Zero-init activation buffer
        Array.Clear(_activations, 0, _activations.Length);

        // Assign inputs
        for (int i = 0; i < _inputCount && i < _inputNodeIds.Length; i++)
        {
            int id = _inputNodeIds[i];
            if (id < _activations.Length)
                _activations[id] = inputs[i];
        }

        // Bias node = 1.0
        if (_biasNodeId >= 0 && _biasNodeId < _activations.Length)
            _activations[_biasNodeId] = 1.0f;

        // Process hidden + output nodes in topological order
        foreach (int nodeId in _genome.TopoOrder)
        {
            var node = _genome.GetNodeById(nodeId);
            if (node is null) continue;

            float pre = node.Bias;
            if (_incoming.TryGetValue(nodeId, out var incoming))
            {
                foreach (var (inNode, weight) in incoming)
                {
                    if (inNode < _activations.Length)
                        pre += _activations[inNode] * weight;
                }
            }

            _activations[nodeId] = Activate(pre, node.Activation);
        }

        // Collect output activations
        var output = new float[_outputCount];
        for (int i = 0; i < _outputCount && i < _outputNodeIds.Length; i++)
        {
            int id = _outputNodeIds[i];
            if (id < _activations.Length)
                output[i] = _activations[id];
        }

        return output;
    }

    // ── Topology rebuild ────────────────────────────────────────────────────

    private void RebuildTopology()
    {
        // Ensure topo order is up-to-date
        if (_genome.TopoDirty)
            _genome.ComputeTopologicalOrder();

        // Collect nodes by role (preserve insertion order as canonical ordering)
        var inputs  = _genome.Nodes.Where(n => n.Role == NeatNodeRole.Input).ToList();
        var outputs = _genome.Nodes.Where(n => n.Role == NeatNodeRole.Output).ToList();
        var bias    = _genome.Nodes.FirstOrDefault(n => n.Role == NeatNodeRole.Bias);

        _inputNodeIds  = inputs.Select(n => n.Id).ToArray();
        _outputNodeIds = outputs.Select(n => n.Id).ToArray();
        _biasNodeId    = bias?.Id ?? -1;

        // Determine activation buffer size
        int maxId = -1;
        foreach (var n in _genome.Nodes)
            if (n.Id > maxId) maxId = n.Id;

        int bufferSize = maxId + 1;
        if (_activations.Length < bufferSize)
            _activations = new float[bufferSize];

        // Rebuild incoming connection lookup for enabled connections
        _incoming.Clear();
        foreach (var c in _genome.Connections)
        {
            if (!c.Enabled) continue;
            if (!_incoming.TryGetValue(c.OutNode, out var list))
            {
                list = new List<(int, float)>();
                _incoming[c.OutNode] = list;
            }
            list.Add((c.InNode, c.Weight));
        }
    }

    // ── Activation functions ────────────────────────────────────────────────

    private static float Activate(float x, RLActivationKind activation) => activation switch
    {
        RLActivationKind.Relu => x > 0f ? x : 0f,
        _                     => MathF.Tanh(x),   // Tanh is default
    };
}
