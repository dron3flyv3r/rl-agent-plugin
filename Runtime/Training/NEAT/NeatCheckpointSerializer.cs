using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Packs and unpacks NEAT genome(s) into the flat <see cref="RLCheckpoint"/>
/// WeightBuffer + LayerShapeBuffer arrays without adding any new fields to the
/// checkpoint resource.
///
/// LayerShapeBuffer layout (ints):
///   [0]  = FORMAT_MAGIC (= 1)
///   [1]  = inputCount
///   [2]  = outputCount
///   [3]  = nodeCount
///   [4]  = connectionCount
///   // Per node (3 ints):
///   [5 + i*3 + 0] = nodeId
///   [5 + i*3 + 1] = nodeRole  (0=Input, 1=Bias, 2=Hidden, 3=Output)
///   [5 + i*3 + 2] = activation (0=Tanh, 1=Relu)
///   // Per connection (4 ints):
///   [5 + nodeCount*3 + j*4 + 0] = inNodeId
///   [5 + nodeCount*3 + j*4 + 1] = outNodeId
///   [5 + nodeCount*3 + j*4 + 2] = innovation
///   [5 + nodeCount*3 + j*4 + 3] = enabled (1=enabled, 0=disabled)
///
/// WeightBuffer layout (floats):
///   [0 .. nodeCount-1]            = node biases  (same order as LayerShapeBuffer nodes)
///   [nodeCount .. nodeCount+connCount-1] = connection weights
///
/// Full-population mode (SaveFullPopulation=true):
///   Genomes are stored back-to-back. A per-genome header (5 ints) is prepended
///   to each genome block. Hyperparams["neat_save_mode"]=1, ["neat_pop_size"]=N.
/// </summary>
internal static class NeatCheckpointSerializer
{
    private const int FormatMagic = 1;

    // ── Serialize champion only ─────────────────────────────────────────────

    public static RLCheckpoint Serialize(
        NeatGenome champion,
        string groupId, long totalSteps, long episodeCount, long updateCount,
        RLNEATConfig config, int obsSize, int actionCount, bool isDiscrete,
        int generation, int speciesCount, float allTimeBestFitness,
        int nextInnovation, int nextNodeId)
    {
        var cp = new RLCheckpoint
        {
            Algorithm             = RLCheckpoint.NeatAlgorithm,
            RunId                 = groupId,
            TotalSteps            = totalSteps,
            EpisodeCount          = episodeCount,
            UpdateCount           = updateCount,
            ObservationSize       = obsSize,
            DiscreteActionCount   = isDiscrete ? actionCount : 0,
            ContinuousActionDimensions = isDiscrete ? 0 : actionCount,
            RewardSnapshot        = champion.Fitness,
        };

        PackGenome(champion, out var shapes, out var weights);
        cp.LayerShapeBuffer = shapes;
        cp.WeightBuffer     = weights;

        cp.Hyperparams = BuildHyperparams(config, generation, speciesCount,
            allTimeBestFitness, nextInnovation, nextNodeId, saveMode: 0, popSize: 1);
        return cp;
    }

    // ── Serialize full population ───────────────────────────────────────────

    public static RLCheckpoint SerializeFullPopulation(
        List<NeatGenome> genomes, NeatGenome champion,
        string groupId, long totalSteps, long episodeCount, long updateCount,
        RLNEATConfig config, int obsSize, int actionCount, bool isDiscrete,
        int generation, int speciesCount, float allTimeBestFitness,
        int nextInnovation, int nextNodeId)
    {
        var cp = new RLCheckpoint
        {
            Algorithm             = RLCheckpoint.NeatAlgorithm,
            RunId                 = groupId,
            TotalSteps            = totalSteps,
            EpisodeCount          = episodeCount,
            UpdateCount           = updateCount,
            ObservationSize       = obsSize,
            DiscreteActionCount   = isDiscrete ? actionCount : 0,
            ContinuousActionDimensions = isDiscrete ? 0 : actionCount,
            RewardSnapshot        = champion.Fitness,
        };

        var allShapes  = new List<int>();
        var allWeights = new List<float>();

        // Store champion index as the first entry so deserialization knows which is best
        int championIdx = genomes.IndexOf(champion);
        if (championIdx < 0) championIdx = 0;

        // Header: [FormatMagic=2, popSize, championIdx]
        allShapes.AddRange(new[] { 2, genomes.Count, championIdx });

        foreach (var g in genomes)
        {
            PackGenome(g, out var gShapes, out var gWeights);
            // Per-genome: [genomeLength in shapes, genomeLength in weights, ...data...]
            allShapes.Add(gShapes.Length);
            allShapes.AddRange(gShapes);
            allWeights.Add(gWeights.Length);
            allWeights.AddRange(gWeights);
        }

        cp.LayerShapeBuffer = allShapes.ToArray();
        cp.WeightBuffer     = allWeights.ToArray();

        cp.Hyperparams = BuildHyperparams(config, generation, speciesCount,
            allTimeBestFitness, nextInnovation, nextNodeId, saveMode: 1, popSize: genomes.Count);
        return cp;
    }

    // ── Deserialize champion genome ─────────────────────────────────────────

    public static NeatGenome DeserializeGenome(RLCheckpoint checkpoint)
    {
        var shapes  = checkpoint.LayerShapeBuffer;
        var weights = checkpoint.WeightBuffer;

        if (shapes is null || shapes.Length < 5)
            throw new InvalidOperationException("[NEAT] Checkpoint has no valid LayerShapeBuffer.");

        int magic = shapes[0];

        if (magic == 2) // full-population mode — extract champion
        {
            int championIdx = shapes[2];
            return DeserializeGenomeAtPopulationSlot(shapes, weights, championIdx);
        }

        // Champion-only mode
        return UnpackGenome(shapes, weights, 0, 0);
    }

    // ── Deserialize full population ─────────────────────────────────────────

    public static List<NeatGenome> DeserializePopulation(RLCheckpoint checkpoint)
    {
        var shapes  = checkpoint.LayerShapeBuffer;
        var weights = checkpoint.WeightBuffer;

        if (shapes is null || shapes.Length < 3 || shapes[0] != 2)
            throw new InvalidOperationException("[NEAT] Checkpoint does not contain a full population.");

        int popSize  = shapes[1];
        var genomes  = new List<NeatGenome>(popSize);
        int si       = 3; // skip [magic, popSize, championIdx]
        int wi       = 1; // skip first float which is unused padding in full-pop format
                          // Actually: first float is length of first genome's weight block, stored as int cast to float
                          // But we stored int as float — let's use the pattern we wrote above
        // Re-check our format: allWeights.Add(gWeights.Length) stores length as float
        // allWeights.AddRange(gWeights) stores actual weights
        // So wi starts at 0 and each genome starts with a float = (float)weightCount

        wi = 0;
        for (int i = 0; i < popSize; i++)
        {
            int shapeLen  = shapes[si];     si++;
            int weightLen = (int)weights[wi]; wi++;

            var gShapes  = shapes[si..(si + shapeLen)];   si += shapeLen;
            var gWeights = weights[wi..(wi + weightLen)];  wi += weightLen;

            genomes.Add(UnpackGenome(gShapes, gWeights, 0, 0));
        }

        return genomes;
    }

    /// <summary>Creates an inference policy wrapping the champion genome in the checkpoint.</summary>
    public static NeatInferencePolicy PolicyFromCheckpoint(RLCheckpoint checkpoint)
    {
        var genome = DeserializeGenome(checkpoint);
        bool isDiscrete = checkpoint.DiscreteActionCount > 0;
        int actionCount = isDiscrete ? checkpoint.DiscreteActionCount : checkpoint.ContinuousActionDimensions;
        return new NeatInferencePolicy(genome, actionCount, isDiscrete);
    }

    // ── Pack / Unpack a single genome ───────────────────────────────────────

    private static void PackGenome(NeatGenome g, out int[] shapes, out float[] weights)
    {
        int nodeCount = g.Nodes.Count;
        int connCount = g.Connections.Count;

        // LayerShapeBuffer: header(5) + nodes(3 each) + connections(4 each)
        shapes = new int[5 + nodeCount * 3 + connCount * 4];
        shapes[0] = FormatMagic;
        shapes[1] = g.Nodes.Count(n => n.Role == NeatNodeRole.Input);
        shapes[2] = g.Nodes.Count(n => n.Role == NeatNodeRole.Output);
        shapes[3] = nodeCount;
        shapes[4] = connCount;

        int si = 5;
        foreach (var n in g.Nodes)
        {
            shapes[si++] = n.Id;
            shapes[si++] = (int)n.Role;
            shapes[si++] = (int)n.Activation;
        }

        foreach (var c in g.Connections)
        {
            shapes[si++] = c.InNode;
            shapes[si++] = c.OutNode;
            shapes[si++] = c.Innovation;
            shapes[si++] = c.Enabled ? 1 : 0;
        }

        // WeightBuffer: biases then connection weights
        weights = new float[nodeCount + connCount];
        int wi = 0;
        foreach (var n in g.Nodes) weights[wi++] = n.Bias;
        foreach (var c in g.Connections) weights[wi++] = c.Weight;
    }

    private static NeatGenome UnpackGenome(int[] shapes, float[] weights, int shapeOffset, int weightOffset)
    {
        int si = shapeOffset;
        int wi = weightOffset;

        // shapes[si] = magic (already verified by caller if needed)
        si++;                      // skip magic
        si++;                      // skip inputCount (reconstructed from node roles)
        si++;                      // skip outputCount
        int nodeCount = shapes[si++];
        int connCount = shapes[si++];

        var g = NeatGenome.Create();

        for (int i = 0; i < nodeCount; i++)
        {
            int id         = shapes[si++];
            var role       = (NeatNodeRole)shapes[si++];
            var activation = (RLActivationKind)shapes[si++];
            float bias     = weights[wi++];
            g.Nodes.Add(new NeatNodeGene { Id = id, Role = role, Activation = activation, Bias = bias });
        }

        for (int j = 0; j < connCount; j++)
        {
            int inNode    = shapes[si++];
            int outNode   = shapes[si++];
            int innov     = shapes[si++];
            bool enabled  = shapes[si++] == 1;
            float weight  = weights[wi++];
            g.Connections.Add(new NeatConnectionGene
            {
                InNode     = inNode,
                OutNode    = outNode,
                Weight     = weight,
                Enabled    = enabled,
                Innovation = innov,
            });
        }

        g.TopoDirty = true;
        return g;
    }

    private static NeatGenome DeserializeGenomeAtPopulationSlot(int[] shapes, float[] weights, int slot)
    {
        int si = 3; // skip [magic=2, popSize, championIdx]
        int wi = 0;
        for (int i = 0; i <= slot; i++)
        {
            int shapeLen  = shapes[si];      si++;
            int weightLen = (int)weights[wi]; wi++;

            if (i == slot)
                return UnpackGenome(shapes[si..], weights[wi..], 0, 0);

            si += shapeLen;
            wi += weightLen;
        }
        throw new InvalidOperationException($"[NEAT] Slot {slot} not found in full-population checkpoint.");
    }

    // ── Hyperparams dictionary ──────────────────────────────────────────────

    private static System.Collections.Generic.Dictionary<string, float> BuildHyperparams(
        RLNEATConfig config, int generation, int speciesCount,
        float allTimeBestFitness, int nextInnovation, int nextNodeId,
        int saveMode, int popSize)
    {
        return new System.Collections.Generic.Dictionary<string, float>(StringComparer.Ordinal)
        {
            ["neat_save_mode"]             = saveMode,
            ["neat_pop_size"]              = popSize,
            ["neat_generation"]            = generation,
            ["neat_species_count"]         = speciesCount,
            ["neat_all_time_best_fitness"] = allTimeBestFitness,
            ["neat_next_innovation"]       = nextInnovation,
            ["neat_next_node_id"]          = nextNodeId,
            // Config snapshot
            ["neat_population_size"]       = config.PopulationSize,
            ["neat_episodes_per_genome"]   = config.EpisodesPerGenome,
            ["neat_compat_threshold"]      = config.CompatibilityThreshold,
            ["neat_weight_mut_rate"]       = config.WeightMutationRate,
            ["neat_add_conn_rate"]         = config.AddConnectionRate,
            ["neat_add_node_rate"]         = config.AddNodeRate,
        };
    }
}
