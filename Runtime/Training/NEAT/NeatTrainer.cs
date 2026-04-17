using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// NEAT trainer. Inherits population bookkeeping from <see cref="EvolutionaryTrainer"/>;
/// this class only contains NEAT-specific logic: genome networks, speciation, crossover,
/// checkpoint serialisation.
///
/// One <see cref="GenomeNetwork"/> per agent slot (GroupAgentSlot → genome index).
/// <see cref="TryUpdate"/> fires when all agents have completed
/// <see cref="RLNEATConfig.EpisodesPerGenome"/> episodes, then evolves the population.
/// </summary>
public sealed class NeatTrainer : EvolutionaryTrainer, INeatDataFeed
{
    // ── Self-registration ───────────────────────────────────────────────────
    static NeatTrainer()
    {
        TrainerFactory.Register("NEAT", config => new NeatTrainer(config));
    }

    // ── NEAT-specific state ─────────────────────────────────────────────────

    private readonly RLNEATConfig _neatConfig;
    private readonly NeatPopulation _population;
    private readonly int _obsSize;
    private readonly int _actionCount;
    private readonly bool _isDiscrete;
    private readonly Random _rng;

    private GenomeNetwork[] _networks;

    // ── INeatDataFeed ───────────────────────────────────────────────────────

    // Topology is cached per generation — only rebuilt when the champion changes.
    private List<UINodeInfo>?       _cachedNodes;
    private List<UIConnectionInfo>? _cachedConnections;
    private int                     _topoGeneration = -1;

    public NeatGenomeSnapshot GetSnapshot()
    {
        int gen = _population.Generation;

        // Rebuild node/connection lists only when the generation advances
        // (topology only changes between generations, not within one).
        if (_cachedNodes == null || gen != _topoGeneration)
        {
            var genome = _population.AllTimeChampion ?? _population.GetChampion();

            _cachedNodes = new List<UINodeInfo>(genome.Nodes.Count);
            foreach (var n in genome.Nodes)
                _cachedNodes.Add(new UINodeInfo(n.Id, (UINodeRole)(int)n.Role));

            _cachedConnections = new List<UIConnectionInfo>(genome.Connections.Count);
            foreach (var c in genome.Connections)
                _cachedConnections.Add(new UIConnectionInfo(c.InNode, c.OutNode, c.Weight, c.Enabled));

            _topoGeneration = gen;
        }

        // Live stats — read directly from the accumulators so they update
        // every frame during a running generation, not just at end-of-gen.
        int   budget    = EpisodesPerGenome();
        float best      = 0f;
        float mean      = 0f;
        int   alive     = 0;
        // Use the same divisor TryUpdate will use so the live display matches the
        // final assigned fitness. Fall back to 1 so a mid-episode (eps == 0)
        // genome still shows its accumulated reward instead of zero.
        float fitDivisor = Math.Max(1, budget);

        for (int i = 0; i < EffectivePopSize; i++)
        {
            int   eps     = EpisodeCounts[i];
            float fitness = FitnessAccum[i] / fitDivisor;
            if (fitness > best) best = fitness;
            mean += fitness;
            if (eps < budget) alive++;
        }
        if (EffectivePopSize > 0) mean /= EffectivePopSize;

        return new NeatGenomeSnapshot
        {
            Nodes          = _cachedNodes,
            Connections    = _cachedConnections!,
            Generation     = gen,
            BestFitness    = best,
            MeanFitness    = mean,
            PopulationSize = EffectivePopSize,
            AliveCount     = alive,
            SpeciesCount   = _population.Species.Count,
            InputCount     = _obsSize,
            OutputCount    = _actionCount,
        };
    }

    // ── Debug ────────────────────────────────────────────────────────────────

    private bool _batchSizeLogged;
    private int _sampleCallsThisGen;

    // ── Construction ────────────────────────────────────────────────────────

    public NeatTrainer(PolicyGroupConfig config)
    {
        _neatConfig = config.AlgorithmConfig as RLNEATConfig ?? new RLNEATConfig();

        if (config.DiscreteActionCount > 0 && config.ContinuousActionDimensions > 0)
            throw new InvalidOperationException(
                $"[NEAT] Mixed discrete+continuous action spaces are not supported. " +
                $"Group '{config.GroupId}' has both. Use separate policy groups.");

        if (config.DiscreteActionCount <= 0 && config.ContinuousActionDimensions <= 0)
            throw new InvalidOperationException(
                $"[NEAT] Group '{config.GroupId}' has no actions defined.");

        _isDiscrete  = config.DiscreteActionCount > 0;
        _actionCount = _isDiscrete ? config.DiscreteActionCount : config.ContinuousActionDimensions;
        _obsSize     = config.ObsSpec?.TotalSize ?? config.ObservationSize;
        _rng         = new Random();

        int popSize = _neatConfig.PopulationSize;

        _population = new NeatPopulation(
            popSize, _obsSize, _actionCount,
            _neatConfig.HiddenActivation, _neatConfig.InitialConnectionDensity, _rng);

        _networks = BuildNetworks(popSize);

        // Initialise base-class accumulators and episode budget
        InitAccumulators(popSize);
        SetEpisodesPerGenome(_neatConfig.EpisodesPerGenome);
    }

    // ── EvolutionaryTrainer: inference ──────────────────────────────────────

    public override PolicyDecision SampleAction(float[] observation)
    {
        return SampleForSlot(observation, 0);
    }

    public override PolicyDecision[] SampleActions(VectorBatch observations)
    {
        _sampleCallsThisGen++;
        var decisions = base.SampleActions(observations);

        if (!_batchSizeLogged)
        {
            _batchSizeLogged = true;
        }

        return decisions;
    }

    protected override PolicyDecision SampleForSlot(float[] observation, int slot)
    {
        if (slot >= EffectivePopSize) slot = 0;
        var output = _networks[slot].Forward(observation);
        var decision = ToDecision(output);
        return decision;
    }

    // ── ITrainer: training update ───────────────────────────────────────────

    public override TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (!IsGenerationComplete()) return null;

        // Assign fitness (mean reward per episode)
        for (int i = 0; i < EffectivePopSize; i++)
            _population.Genomes[i].Fitness =
                FitnessAccum[i] / Math.Max(1, _neatConfig.EpisodesPerGenome);

        // Debug: log per-genome fitness so we can see if genomes are differentiating.
        var fitnessStrs = new System.Text.StringBuilder();
        for (int i = 0; i < Math.Min(EffectivePopSize, 20); i++)
            fitnessStrs.Append($"  g{i}={_population.Genomes[i].Fitness:F2} (ep={EpisodeCounts[i]})");
        _batchSizeLogged = false;
        _sampleCallsThisGen = 0;

        float bestFitness = _population.Genomes.Max(g => g.Fitness);
        float meanFitness = _population.Genomes.Average(g => g.Fitness);
        float diversity   = SamplePopulationDiversity();

        _population.Advance(_neatConfig);
        _networks = BuildNetworks(EffectivePopSize);
        ResetAccumulators();

        var checkpoint = BuildCheckpoint(groupId, totalSteps, episodeCount, _population.Generation);

        return new TrainerUpdateStats
        {
            PolicyLoss   = bestFitness,
            ValueLoss    = meanFitness,
            Entropy      = diversity,
            ClipFraction = _population.Species.Count,
            Checkpoint   = checkpoint,
        };
    }

    // ── ITrainer: checkpoint ────────────────────────────────────────────────

    public override RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount) =>
        BuildCheckpoint(groupId, totalSteps, episodeCount, updateCount);

    public override IInferencePolicy SnapshotPolicyForEval() =>
        new NeatInferencePolicy(_population.GetChampion().Clone(), _actionCount, _isDiscrete);

    public override void LoadFromCheckpoint(RLCheckpoint checkpoint)
    {
        if (!string.Equals(checkpoint.Algorithm, RLCheckpoint.NeatAlgorithm,
                StringComparison.OrdinalIgnoreCase))
        {
            GD.PushWarning($"[NEAT] LoadFromCheckpoint: checkpoint algorithm is '{checkpoint.Algorithm}', expected 'NEAT'. Ignoring.");
            return;
        }

        bool hasFullPop = checkpoint.LayerShapeBuffer?.Length > 0
            && checkpoint.LayerShapeBuffer[0] == 2
            && _neatConfig.SaveFullPopulation;

        if (hasFullPop)
        {
            try
            {
                var savedGenomes = NeatCheckpointSerializer.DeserializePopulation(checkpoint);
                int count = Math.Min(savedGenomes.Count, EffectivePopSize);
                for (int i = 0; i < count; i++)
                    _population.Genomes[i] = savedGenomes[i];

                _population.Speciate(
                    _neatConfig.ExcessCoeff, _neatConfig.DisjointCoeff,
                    _neatConfig.WeightDiffCoeff, _neatConfig.CompatibilityThreshold);
            }
            catch (Exception ex)
            {
                GD.PushWarning($"[NEAT] Failed to deserialize full population: {ex.Message}. Falling back to champion warm-start.");
                WarmStartFromChampion(checkpoint);
            }
        }
        else
        {
            WarmStartFromChampion(checkpoint);
        }

        _networks = BuildNetworks(EffectivePopSize);
        ResetAccumulators();
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    private PolicyDecision ToDecision(float[] output)
    {
        if (_isDiscrete)
        {
            return new PolicyDecision
            {
                DiscreteAction = SelectDiscreteAction(output),
                Value          = 0f,
                LogProbability = 0f,
            };
        }

        var actions = new float[_actionCount];
        for (int i = 0; i < _actionCount; i++)
            actions[i] = i < output.Length ? MathF.Tanh(output[i]) : 0f;

        return new PolicyDecision
        {
            ContinuousActions = actions,
            Value             = 0f,
            LogProbability    = 0f,
        };
    }

    private static int SelectDiscreteAction(float[] logits)
    {
        if (logits.Length == 0) return 0;
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private GenomeNetwork[] BuildNetworks(int count)
    {
        var nets = new GenomeNetwork[count];
        for (int i = 0; i < count; i++)
            nets[i] = new GenomeNetwork(_population.Genomes[i], _obsSize, _actionCount);
        return nets;
    }

    private RLCheckpoint BuildCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        var champion = _population.GetChampion();
        var (nextInnov, nextNode) = _population.Innovation.SaveState();

        if (_neatConfig.SaveFullPopulation)
        {
            return NeatCheckpointSerializer.SerializeFullPopulation(
                _population.Genomes, champion,
                groupId, totalSteps, episodeCount, updateCount,
                _neatConfig, _obsSize, _actionCount, _isDiscrete,
                _population.Generation, _population.Species.Count,
                _population.AllTimeChampion?.Fitness ?? champion.Fitness,
                nextInnov, nextNode);
        }

        return NeatCheckpointSerializer.Serialize(
            champion,
            groupId, totalSteps, episodeCount, updateCount,
            _neatConfig, _obsSize, _actionCount, _isDiscrete,
            _population.Generation, _population.Species.Count,
            _population.AllTimeChampion?.Fitness ?? champion.Fitness,
            nextInnov, nextNode);
    }

    private void WarmStartFromChampion(RLCheckpoint checkpoint)
    {
        try
        {
            var seed = NeatCheckpointSerializer.DeserializeGenome(checkpoint);
            for (int i = 0; i < EffectivePopSize; i++)
            {
                _population.Genomes[i] = i == 0 ? seed : seed.Clone();
                if (i > 0)
                    _population.Genomes[i].MutateWeights(1f, 0.3f, 0.1f, _rng);
            }
            _population.Speciate(
                _neatConfig.ExcessCoeff, _neatConfig.DisjointCoeff,
                _neatConfig.WeightDiffCoeff, _neatConfig.CompatibilityThreshold);
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[NEAT] Failed to deserialize champion from checkpoint: {ex.Message}. Starting fresh.");
        }
    }

    private float SamplePopulationDiversity()
    {
        int n = _population.Genomes.Count;
        if (n < 2) return 0f;

        float c1 = _neatConfig.ExcessCoeff;
        float c2 = _neatConfig.DisjointCoeff;
        float c3 = _neatConfig.WeightDiffCoeff;

        int sampleCount = Math.Min(20, n * (n - 1) / 2);
        float sum = 0f;
        for (int s = 0; s < sampleCount; s++)
        {
            int a = _rng.Next(n);
            int b = _rng.Next(n - 1);
            if (b >= a) b++;
            sum += _population.Genomes[a].CompatibilityDistance(_population.Genomes[b], c1, c2, c3);
        }
        return sum / sampleCount;
    }
}
