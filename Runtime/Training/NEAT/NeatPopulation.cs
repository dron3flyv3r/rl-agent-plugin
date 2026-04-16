using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Owns and evolves the full genome population for NEAT.
/// Called exclusively from <see cref="NeatTrainer.TryUpdate"/> on the main thread.
/// </summary>
internal sealed class NeatPopulation
{
    public List<NeatGenome> Genomes { get; private set; }
    public List<NeatSpecies> Species { get; } = new();
    public NeatInnovationTracker Innovation { get; }
    public int Generation { get; private set; }
    public NeatGenome? AllTimeChampion { get; private set; }

    private readonly int _inputCount;
    private readonly int _outputCount;
    private readonly RLActivationKind _hiddenActivation;
    private readonly Random _rng;

    // ── Construction ────────────────────────────────────────────────────────

    /// <summary>
    /// Creates the initial minimal population.
    /// Each genome starts with input nodes, one bias node, and output nodes.
    /// Connections from inputs (and bias) to outputs are added with probability
    /// <paramref name="density"/> (1.0 = fully connected).
    /// </summary>
    public NeatPopulation(
        int popSize, int inputCount, int outputCount,
        RLActivationKind hiddenActivation, float density, Random rng)
    {
        _inputCount       = inputCount;
        _outputCount      = outputCount;
        _hiddenActivation = hiddenActivation;
        _rng              = rng;

        // Initial node IDs:
        //   0 .. inputCount-1        → input nodes
        //   inputCount               → bias node
        //   inputCount+1 .. inputCount+outputCount  → output nodes
        // Next hidden node ID starts at inputCount + 1 + outputCount
        int biasId      = inputCount;
        int firstOutput = inputCount + 1;
        int firstHidden = inputCount + 1 + outputCount;

        Innovation = new NeatInnovationTracker(
            initialInnovations: 0,
            startNodeId: firstHidden);

        Genomes = new List<NeatGenome>(popSize);
        for (int i = 0; i < popSize; i++)
            Genomes.Add(CreateMinimalGenome(
                inputCount, biasId, firstOutput, outputCount, density));

        Speciate(0f, 1f, 0.4f, 3f);
    }

    // ── Full generational step ──────────────────────────────────────────────

    public void Advance(RLNEATConfig config)
    {
        Innovation.StartGeneration();

        ComputeAdjustedFitness();

        foreach (var s in Species)
        {
            s.UpdateStagnation();
            s.Age++;
        }

        // Update all-time champion
        var currentChampion = GetChampion();
        if (AllTimeChampion is null || currentChampion.Fitness > AllTimeChampion.Fitness)
            AllTimeChampion = currentChampion.Clone();

        // Remove stagnant species (but protect the all-time champion's species and young species)
        int championSpecies = currentChampion.SpeciesId;
        Species.RemoveAll(s =>
            s.StagnationCounter > config.StagnationLimit
            && s.Age > 2
            && s.SpeciesId != championSpecies
            && Species.Count > 2);

        // Produce new population
        Genomes = Reproduce(config);

        // Re-speciate the new generation
        Speciate(config.ExcessCoeff, config.DisjointCoeff, config.WeightDiffCoeff, config.CompatibilityThreshold);

        Generation++;
    }

    // ── Speciation ──────────────────────────────────────────────────────────

    public void Speciate(float c1, float c2, float c3, float threshold)
    {
        // Pick a random representative from each surviving species (from old members)
        foreach (var s in Species)
        {
            if (s.Members.Count > 0)
                s.Representative = s.Members[_rng.Next(s.Members.Count)];
            s.Members.Clear();
        }

        // Assign each genome to a compatible species
        foreach (var g in Genomes)
        {
            NeatSpecies? match = null;
            foreach (var s in Species)
            {
                if (g.CompatibilityDistance(s.Representative, c1, c2, c3) < threshold)
                {
                    match = s;
                    break;
                }
            }

            if (match is null)
            {
                // Create a new species with this genome as the representative
                match = NeatSpecies.Create(g);
                Species.Add(match);
            }

            g.SpeciesId = match.SpeciesId;
            match.Members.Add(g);
        }

        // Remove empty species
        Species.RemoveAll(s => s.Members.Count == 0);
    }

    // ── Fitness sharing ─────────────────────────────────────────────────────

    public void ComputeAdjustedFitness()
    {
        foreach (var s in Species)
        {
            int size = s.Members.Count;
            foreach (var g in s.Members)
                g.AdjustedFitness = size > 0 ? Math.Max(0f, g.Fitness) / size : 0f;
        }
    }

    // ── Champion ────────────────────────────────────────────────────────────

    public NeatGenome GetChampion() =>
        Genomes.Count == 0 ? Genomes[0] : Genomes.MaxBy(g => g.Fitness)!;

    // ── Reproduction ────────────────────────────────────────────────────────

    private List<NeatGenome> Reproduce(RLNEATConfig config)
    {
        int popSize = Genomes.Count;
        float totalAdjFitness = Species.Sum(s => s.SumAdjustedFitness());

        // Allocate offspring count per species
        var quotas = new Dictionary<int, int>();
        int allocated = 0;
        foreach (var s in Species)
        {
            float fraction = totalAdjFitness > 0f
                ? s.SumAdjustedFitness() / totalAdjFitness
                : 1f / Species.Count;
            int quota = (int)MathF.Round(fraction * popSize);
            quota = Math.Max(1, quota);   // at least 1
            quotas[s.SpeciesId] = quota;
            allocated += quota;
        }

        // Normalize to exactly popSize
        AdjustQuotas(quotas, popSize, allocated);

        // Build offspring list
        var offspring = new List<NeatGenome>(popSize);

        // Always carry the all-time champion unchanged at index 0
        if (AllTimeChampion is not null)
            offspring.Add(AllTimeChampion.Clone());

        foreach (var s in Species.OrderByDescending(s => s.AverageAdjustedFitness()))
        {
            if (!quotas.TryGetValue(s.SpeciesId, out int quota) || quota <= 0) continue;

            // Sort members by adjusted fitness descending
            var pool = s.Members.OrderByDescending(g => g.AdjustedFitness).ToList();

            // Elitism: copy top ElitismCount unchanged
            int elites = Math.Min(config.ElitismCount, Math.Min(quota, pool.Count));
            for (int i = 0; i < elites && offspring.Count < popSize; i++)
            {
                // Skip if it duplicates the champion we already added
                if (offspring.Count == 1 && pool[i].GenomeId == AllTimeChampion?.GenomeId)
                    continue;
                offspring.Add(pool[i].Clone());
            }

            // Breeding pool = top SurvivalThreshold fraction
            int breedingCount = Math.Max(1, (int)MathF.Ceiling(pool.Count * config.SurvivalThreshold));
            var breedingPool = pool.Take(breedingCount).ToList();

            // Fill remaining quota
            int remaining = quota - elites;
            for (int i = 0; i < remaining && offspring.Count < popSize; i++)
            {
                NeatGenome child;
                bool doCrossover = breedingPool.Count >= 2 && _rng.NextSingle() < config.CrossoverRate;
                if (doCrossover)
                {
                    var p1 = breedingPool[_rng.Next(breedingPool.Count)];
                    NeatGenome p2;
                    do { p2 = breedingPool[_rng.Next(breedingPool.Count)]; } while (p2 == p1 && breedingPool.Count > 1);

                    bool equalFitness = MathF.Abs(p1.AdjustedFitness - p2.AdjustedFitness) < 1e-6f;
                    var fitter  = p1.AdjustedFitness >= p2.AdjustedFitness ? p1 : p2;
                    var weaker  = fitter == p1 ? p2 : p1;
                    child = NeatGenome.Crossover(fitter, weaker, equalFitness, _rng);
                }
                else
                {
                    child = breedingPool[_rng.Next(breedingPool.Count)].Clone();
                }

                ApplyMutation(child, config);
                offspring.Add(child);
            }
        }

        // Pad with champion clones if we're short (edge case: very few species)
        while (offspring.Count < popSize)
        {
            var seed = GetChampion().Clone();
            ApplyMutation(seed, config);
            offspring.Add(seed);
        }

        return offspring.Take(popSize).ToList();
    }

    private void ApplyMutation(NeatGenome g, RLNEATConfig config)
    {
        if (_rng.NextSingle() < config.WeightMutationRate)
            g.MutateWeights(config.WeightPerturbRate, config.WeightPerturbScale, config.WeightResetScale, _rng);

        if (_rng.NextSingle() < config.AddNodeRate)
            g.MutateAddNode(Innovation, _hiddenActivation, _rng);

        if (_rng.NextSingle() < config.AddConnectionRate)
            g.MutateAddConnection(Innovation, _rng);

        if (_rng.NextSingle() < config.ToggleConnectionRate)
            g.MutateToggleConnection(_rng);
    }

    private static void AdjustQuotas(Dictionary<int, int> quotas, int target, int current)
    {
        while (current > target)
        {
            // Remove one from the species with the largest quota (that has > 1)
            int maxKey = quotas.OrderByDescending(kv => kv.Value).First().Key;
            if (quotas[maxKey] > 1) { quotas[maxKey]--; current--; }
            else break;
        }
        while (current < target)
        {
            int maxKey = quotas.OrderByDescending(kv => kv.Value).First().Key;
            quotas[maxKey]++; current++;
        }
    }

    // ── Minimal genome factory ───────────────────────────────────────────────

    private NeatGenome CreateMinimalGenome(
        int inputCount, int biasId, int firstOutputId,
        int outputCount, float density)
    {
        var g = NeatGenome.Create();

        // Input nodes
        for (int i = 0; i < inputCount; i++)
            g.Nodes.Add(new NeatNodeGene { Id = i, Role = NeatNodeRole.Input, Activation = RLActivationKind.Tanh });

        // Bias node
        g.Nodes.Add(new NeatNodeGene { Id = biasId, Role = NeatNodeRole.Bias, Activation = RLActivationKind.Tanh });

        // Output nodes
        for (int o = 0; o < outputCount; o++)
            g.Nodes.Add(new NeatNodeGene
            {
                Id         = firstOutputId + o,
                Role       = NeatNodeRole.Output,
                Activation = _hiddenActivation,
            });

        // Connect inputs + bias → outputs with probability density
        for (int outIdx = 0; outIdx < outputCount; outIdx++)
        {
            int outNodeId = firstOutputId + outIdx;

            for (int inIdx = 0; inIdx < inputCount; inIdx++)
            {
                if (_rng.NextSingle() > density) continue;
                int innov = Innovation.GetOrCreateConnectionInnovation(inIdx, outNodeId);
                g.Connections.Add(new NeatConnectionGene
                {
                    InNode     = inIdx,
                    OutNode    = outNodeId,
                    Weight     = (float)(_rng.NextDouble() * 2.0 - 1.0),
                    Enabled    = true,
                    Innovation = innov,
                });
            }

            // Bias → output
            if (_rng.NextSingle() <= density)
            {
                int innov = Innovation.GetOrCreateConnectionInnovation(biasId, outNodeId);
                g.Connections.Add(new NeatConnectionGene
                {
                    InNode     = biasId,
                    OutNode    = outNodeId,
                    Weight     = (float)(_rng.NextDouble() * 2.0 - 1.0),
                    Enabled    = true,
                    Innovation = innov,
                });
            }
        }

        g.TopoDirty = true;
        return g;
    }
}
