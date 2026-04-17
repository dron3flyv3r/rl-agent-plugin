using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

internal enum NeatNodeRole { Input = 0, Bias = 1, Hidden = 2, Output = 3 }

internal sealed class NeatNodeGene
{
    public int Id { get; init; }
    public NeatNodeRole Role { get; init; }
    public RLActivationKind Activation { get; set; }
    public float Bias { get; set; }

    public NeatNodeGene Clone() => new()
    {
        Id         = Id,
        Role       = Role,
        Activation = Activation,
        Bias       = Bias,
    };
}

internal sealed class NeatConnectionGene
{
    public int InNode { get; init; }
    public int OutNode { get; init; }
    public float Weight { get; set; }
    public bool Enabled { get; set; } = true;
    public int Innovation { get; init; }

    public NeatConnectionGene Clone() => new()
    {
        InNode     = InNode,
        OutNode    = OutNode,
        Weight     = Weight,
        Enabled    = Enabled,
        Innovation = Innovation,
    };
}

internal sealed class NeatGenome
{
    private static int _nextId = 0;

    public int GenomeId { get; init; }
    public List<NeatNodeGene> Nodes { get; init; } = new();
    public List<NeatConnectionGene> Connections { get; init; } = new();
    public float Fitness { get; set; }
    public float AdjustedFitness { get; set; }
    public int SpeciesId { get; set; } = -1;

    internal int[] TopoOrder = Array.Empty<int>();
    internal bool TopoDirty = true;

    // Cached lookups — rebuilt when TopoDirty
    private Dictionary<int, NeatNodeGene> _nodeById = new();

    public static NeatGenome Create() => new() { GenomeId = System.Threading.Interlocked.Increment(ref _nextId) };

    // ── Lookups ────────────────────────────────────────────────────────────

    public NeatNodeGene? GetNodeById(int id)
    {
        if (_nodeById.Count != Nodes.Count)
            RebuildNodeIndex();
        return _nodeById.TryGetValue(id, out var n) ? n : null;
    }

    private void RebuildNodeIndex()
    {
        _nodeById.Clear();
        foreach (var n in Nodes)
            _nodeById[n.Id] = n;
    }

    public void InvalidateCache()
    {
        TopoDirty = true;
        _nodeById.Clear();
    }

    // ── Deep copy ──────────────────────────────────────────────────────────

    public NeatGenome Clone()
    {
        var g = Create();
        g.SpeciesId = SpeciesId;
        g.Fitness   = Fitness;   // must be copied so AllTimeChampion tracking works
        foreach (var n in Nodes) g.Nodes.Add(n.Clone());
        foreach (var c in Connections) g.Connections.Add(c.Clone());
        g.TopoDirty = true;
        return g;
    }

    // ── Topological sort (Kahn's algorithm) ───────────────────────────────

    internal void ComputeTopologicalOrder()
    {
        // Build in-degree and adjacency for hidden+output nodes only,
        // following only enabled connections.
        var inDegree    = new Dictionary<int, int>();
        var adjacency   = new Dictionary<int, List<int>>();

        // Identify non-input node IDs
        var nonInputIds = new HashSet<int>();
        foreach (var n in Nodes)
        {
            if (n.Role == NeatNodeRole.Input || n.Role == NeatNodeRole.Bias) continue;
            nonInputIds.Add(n.Id);
            inDegree[n.Id]  = 0;
            adjacency[n.Id] = new List<int>();
        }

        foreach (var c in Connections)
        {
            if (!c.Enabled) continue;
            if (!nonInputIds.Contains(c.OutNode)) continue;

            // Only count in-degree from non-input/bias nodes.
            // Input and bias nodes are the implicit frontier — they are never enqueued
            // and their out-edges are handled by the activation assignment before the loop.
            // Counting their edges here would leave output nodes stuck with in-degree > 0
            // in a minimal (no-hidden) network, producing an empty TopoOrder.
            if (nonInputIds.Contains(c.InNode))
                inDegree[c.OutNode]++;

            if (!adjacency.ContainsKey(c.InNode))
                adjacency[c.InNode] = new List<int>();
            adjacency[c.InNode].Add(c.OutNode);
        }

        // Seed queue with zero in-degree non-input nodes
        var queue  = new Queue<int>();
        foreach (var (id, deg) in inDegree)
            if (deg == 0) queue.Enqueue(id);

        var order = new List<int>(nonInputIds.Count);
        while (queue.Count > 0)
        {
            int cur = queue.Dequeue();
            order.Add(cur);
            if (!adjacency.TryGetValue(cur, out var neighbours)) continue;
            foreach (int nb in neighbours)
            {
                if (!inDegree.ContainsKey(nb)) continue;
                if (--inDegree[nb] == 0)
                    queue.Enqueue(nb);
            }
        }

        // Nodes left in inDegree with count > 0 are in cycles — silently excluded.
        TopoOrder = order.ToArray();
        TopoDirty = false;
    }

    // ── Compatibility distance ─────────────────────────────────────────────
    // δ = (C1*E + C2*D) / N  +  C3 * W̄
    // E = excess genes, D = disjoint genes, W̄ = avg weight diff of matching genes
    // N = max(|genes1|, |genes2|), but clamped to 1 if both < 20

    public float CompatibilityDistance(NeatGenome other, float c1, float c2, float c3)
    {
        if (Connections.Count == 0 && other.Connections.Count == 0)
            return 0f;

        var thisMap  = BuildInnovationMap(Connections);
        var otherMap = BuildInnovationMap(other.Connections);

        int maxInnov1 = Connections.Count  > 0 ? Connections.Max(c => c.Innovation)  : 0;
        int maxInnov2 = other.Connections.Count > 0 ? other.Connections.Max(c => c.Innovation) : 0;
        int lowerMax  = Math.Min(maxInnov1, maxInnov2);

        int excess    = 0;
        int disjoint  = 0;
        float wDiff   = 0f;
        int matching  = 0;

        var allInnovations = thisMap.Keys.Union(otherMap.Keys);
        foreach (int innov in allInnovations)
        {
            bool inThis  = thisMap.ContainsKey(innov);
            bool inOther = otherMap.ContainsKey(innov);

            if (inThis && inOther)
            {
                wDiff   += MathF.Abs(thisMap[innov].Weight - otherMap[innov].Weight);
                matching++;
            }
            else if (innov > lowerMax)
            {
                excess++;
            }
            else
            {
                disjoint++;
            }
        }

        int n = Math.Max(1, Math.Max(Connections.Count, other.Connections.Count));

        float avgW = matching > 0 ? wDiff / matching : 0f;
        return (c1 * excess + c2 * disjoint) / n + c3 * avgW;
    }

    // ── Weight mutation ────────────────────────────────────────────────────

    public void MutateWeights(float perturbRate, float perturbScale, float resetScale, Random rng)
    {
        foreach (var c in Connections)
        {
            if (rng.NextSingle() < perturbRate)
                c.Weight = Clamp(c.Weight + SampleGaussian(rng) * perturbScale, -8f, 8f);
            else
                c.Weight = Clamp(SampleGaussian(rng) * resetScale, -8f, 8f);
        }

        foreach (var n in Nodes)
        {
            if (n.Role == NeatNodeRole.Input || n.Role == NeatNodeRole.Bias) continue;
            if (rng.NextSingle() < perturbRate)
                n.Bias = Clamp(n.Bias + SampleGaussian(rng) * perturbScale, -8f, 8f);
            else
                n.Bias = Clamp(SampleGaussian(rng) * resetScale, -8f, 8f);
        }
    }

    // ── Add-connection mutation ────────────────────────────────────────────
    // Returns true if a new connection was added.

    public bool MutateAddConnection(NeatInnovationTracker tracker, Random rng)
    {
        // Candidates: any (nodeA→nodeB) where A is not output, B is not input/bias,
        // and no existing connection (enabled or disabled) already exists.
        var existing = new HashSet<(int, int)>(
            Connections.Select(c => (c.InNode, c.OutNode)));

        var nonOutputIds = Nodes
            .Where(n => n.Role != NeatNodeRole.Output)
            .Select(n => n.Id)
            .ToList();
        var nonInputIds = Nodes
            .Where(n => n.Role != NeatNodeRole.Input && n.Role != NeatNodeRole.Bias)
            .Select(n => n.Id)
            .ToList();

        // Build list of valid pairs and shuffle
        var candidates = new List<(int, int)>();
        foreach (int a in nonOutputIds)
            foreach (int b in nonInputIds)
                if (a != b
                    && !existing.Contains((a, b))
                    && !WouldCreateCycle(a, b))
                    candidates.Add((a, b));

        if (candidates.Count == 0) return false;

        var (inId, outId) = candidates[rng.Next(candidates.Count)];
        int innov = tracker.GetOrCreateConnectionInnovation(inId, outId);
        Connections.Add(new NeatConnectionGene
        {
            InNode     = inId,
            OutNode    = outId,
            Weight     = SampleGaussian(rng),
            Enabled    = true,
            Innovation = innov,
        });
        TopoDirty = true;
        return true;
    }

    // ── Add-node mutation ─────────────────────────────────────────────────
    // Splits a random enabled connection: old_in → [new_node] → old_out
    // Old connection disabled; in→new gets weight 1.0; new→out gets old weight.

    public void MutateAddNode(NeatInnovationTracker tracker, RLActivationKind activation, Random rng)
    {
        var enabled = Connections.Where(c => c.Enabled).ToList();
        if (enabled.Count == 0) return;

        var split = enabled[rng.Next(enabled.Count)];
        int newNodeId   = tracker.GetOrCreateSplitNodeId(split.Innovation);
        int innov1      = tracker.GetOrCreateConnectionInnovation(split.InNode, newNodeId);
        int innov2      = tracker.GetOrCreateConnectionInnovation(newNodeId, split.OutNode);

        // This genome has already materialized the historical split for this connection.
        // Do not add duplicate nodes or duplicate innovations.
        if (Nodes.Any(n => n.Id == newNodeId)
            || Connections.Any(c => c.Innovation == innov1 || c.Innovation == innov2)
            || Connections.Any(c => c.InNode == split.InNode && c.OutNode == newNodeId)
            || Connections.Any(c => c.InNode == newNodeId && c.OutNode == split.OutNode))
        {
            return;
        }

        split.Enabled = false;

        Nodes.Add(new NeatNodeGene
        {
            Id         = newNodeId,
            Role       = NeatNodeRole.Hidden,
            Activation = activation,
            Bias       = 0f,
        });
        Connections.Add(new NeatConnectionGene
        {
            InNode     = split.InNode,
            OutNode    = newNodeId,
            Weight     = 1.0f,
            Enabled    = true,
            Innovation = innov1,
        });
        Connections.Add(new NeatConnectionGene
        {
            InNode     = newNodeId,
            OutNode    = split.OutNode,
            Weight     = split.Weight,
            Enabled    = true,
            Innovation = innov2,
        });

        InvalidateCache();
    }

    // ── Toggle-connection mutation ─────────────────────────────────────────

    public void MutateToggleConnection(Random rng)
    {
        if (Connections.Count == 0) return;
        var c = Connections[rng.Next(Connections.Count)];
        if (c.Enabled)
        {
            c.Enabled = false;
        }
        else if (!WouldCreateCycle(c.InNode, c.OutNode))
        {
            c.Enabled = true;
        }
        TopoDirty = true;
    }

    // ── Crossover (static factory) ─────────────────────────────────────────
    // fitter = more fit parent (excess/disjoint genes are inherited from it).
    // If fitness is equal, genes from weaker may also be included randomly.

    public static NeatGenome Crossover(NeatGenome fitter, NeatGenome weaker, bool equalFitness, Random rng)
    {
        var weakerMap = weaker.Connections.ToDictionary(c => c.Innovation);

        var childConnections = new List<NeatConnectionGene>();
        foreach (var fc in fitter.Connections)
        {
            NeatConnectionGene gene;
            if (weakerMap.TryGetValue(fc.Innovation, out var wc))
            {
                // Matching gene: pick from either parent randomly
                gene = (rng.NextSingle() < 0.5f ? fc : wc).Clone();
                // 75% chance child has gene disabled if either parent has it disabled
                if (!fc.Enabled || !wc.Enabled)
                    gene.Enabled = rng.NextSingle() >= 0.75f;
            }
            else
            {
                // Excess or disjoint — inherit from fitter
                gene = fc.Clone();
            }
            childConnections.Add(gene);
        }

        // If equal fitness, also randomly include weaker-only genes
        if (equalFitness)
        {
            var fitterInnovs = new HashSet<int>(fitter.Connections.Select(c => c.Innovation));
            foreach (var wc in weaker.Connections)
            {
                if (!fitterInnovs.Contains(wc.Innovation) && rng.NextSingle() < 0.5f)
                    childConnections.Add(wc.Clone());
            }
        }

        // Nodes: start with fitter's full node set, then ensure every connection endpoint
        // copied into the child is backed by a concrete node definition.
        var childNodes = fitter.Nodes.Select(n => n.Clone()).ToList();
        var childNodeIds = new HashSet<int>(childNodes.Select(n => n.Id));
        var fitterNodeMap = fitter.Nodes.ToDictionary(n => n.Id);
        var weakerNodeMap = weaker.Nodes.ToDictionary(n => n.Id);

        foreach (var nodeId in childConnections.SelectMany(c => new[] { c.InNode, c.OutNode }).Distinct())
        {
            if (childNodeIds.Contains(nodeId)) continue;

            if (fitterNodeMap.TryGetValue(nodeId, out var fitterNode))
            {
                childNodes.Add(fitterNode.Clone());
                childNodeIds.Add(nodeId);
                continue;
            }

            if (weakerNodeMap.TryGetValue(nodeId, out var weakerNode))
            {
                childNodes.Add(weakerNode.Clone());
                childNodeIds.Add(nodeId);
            }
        }

        childConnections = childConnections
            .GroupBy(c => c.Innovation)
            .Select(g => g.First())
            .ToList();

        var child = Create();
        child.Nodes.AddRange(childNodes);
        child.Connections.AddRange(childConnections);
        child.TopoDirty = true;
        return child;
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static Dictionary<int, NeatConnectionGene> BuildInnovationMap(IEnumerable<NeatConnectionGene> connections)
    {
        var map = new Dictionary<int, NeatConnectionGene>();
        foreach (var connection in connections)
        {
            if (!map.ContainsKey(connection.Innovation))
                map[connection.Innovation] = connection;
        }
        return map;
    }

    private bool WouldCreateCycle(int inNode, int outNode)
    {
        if (inNode == outNode) return true;
        return HasEnabledPath(outNode, inNode);
    }

    private bool HasEnabledPath(int startNode, int targetNode)
    {
        var stack = new Stack<int>();
        var visited = new HashSet<int>();
        stack.Push(startNode);

        while (stack.Count > 0)
        {
            int current = stack.Pop();
            if (!visited.Add(current)) continue;
            if (current == targetNode) return true;

            foreach (var connection in Connections)
            {
                if (!connection.Enabled || connection.InNode != current) continue;
                stack.Push(connection.OutNode);
            }
        }

        return false;
    }

    private static float SampleGaussian(Random rng)
    {
        // Box-Muller transform
        float u1 = Math.Max(rng.NextSingle(), 1e-10f);
        float u2 = rng.NextSingle();
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }

    private static float Clamp(float v, float lo, float hi) =>
        v < lo ? lo : v > hi ? hi : v;
}
