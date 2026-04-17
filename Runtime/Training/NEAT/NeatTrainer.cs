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
    public static bool DebugGenerationStats { get; set; }

    /// <summary>
    /// When true, prints a detailed per-slot fitness table, reproduction breakdown,
    /// species composition, and champion network sample outputs after every generation.
    /// Enable from FlappyBirdController (or any scene script) to diagnose learning issues.
    /// </summary>
    public static bool DebugDeepStats { get; set; }

    private sealed class SlotGenomeSnapshot
    {
        public int Slot;
        public int GenomeId;
        public int SpeciesId;
        public float Fitness;
        public float AdjustedFitness;
        public int Nodes;
        public int EnabledConnections;
    }

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
            var genome = _population.GetPreferredChampion();

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
    private float _lastLoggedBestFitness = float.NaN;
    private float _lastLoggedMeanFitness = float.NaN;

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
        _batchSizeLogged = false;
        _sampleCallsThisGen = 0;

        int generation = _population.Generation;
        int speciesBefore = _population.Species.Count;
        var champion = _population.GetChampion();
        float bestFitness = _population.Genomes.Max(g => g.Fitness);
        float meanFitness = _population.Genomes.Average(g => g.Fitness);
        float worstFitness = _population.Genomes.Min(g => g.Fitness);
        float diversity   = SamplePopulationDiversity();
        float fitnessStdDev = ComputeFitnessStdDev(meanFitness);
        int distinctFitnesses = CountDistinctFitnesses();
        int enabledConnections = champion.Connections.Count(c => c.Enabled);

        // Capture per-slot fitness before Advance() replaces the genomes
        float[]? fitnessSnapshot = DebugDeepStats
            ? _population.Genomes.Select(g => g.Fitness).ToArray()
            : null;
        SlotGenomeSnapshot[]? slotSnapshots = DebugDeepStats
            ? _population.Genomes.Select((g, slot) => new SlotGenomeSnapshot
            {
                Slot = slot,
                GenomeId = g.GenomeId,
                SpeciesId = g.SpeciesId,
                Fitness = g.Fitness,
                AdjustedFitness = g.AdjustedFitness,
                Nodes = g.Nodes.Count,
                EnabledConnections = g.Connections.Count(c => c.Enabled),
            }).ToArray()
            : null;
        string? speciesBeforeSummary = DebugDeepStats ? BuildSpeciesFitnessSummary() : null;

        _population.Advance(_neatConfig);
        _networks = BuildNetworks(EffectivePopSize);
        ResetAccumulators();

        if (DebugGenerationStats)
        {
            GD.Print(
                $"[NEAT] gen={generation} best={bestFitness:F3} mean={meanFitness:F3} worst={worstFitness:F3} " +
                $"std={fitnessStdDev:F3} distinct={distinctFitnesses}/{EffectivePopSize} " +
                $"species={speciesBefore}->{_population.Species.Count} threshold={_population.CurrentThreshold:F2} diversity={diversity:F3} " +
                $"champion(nodes={champion.Nodes.Count}, enabled_conns={enabledConnections}) " +
                $"delta_best={FormatDelta(bestFitness, _lastLoggedBestFitness)} " +
                $"delta_mean={FormatDelta(meanFitness, _lastLoggedMeanFitness)}");

            _lastLoggedBestFitness = bestFitness;
            _lastLoggedMeanFitness = meanFitness;
        }

        if (DebugDeepStats)
            LogDeepStats(generation, bestFitness, fitnessSnapshot!, slotSnapshots!, speciesBeforeSummary!);

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
        new NeatInferencePolicy(_population.GetPreferredChampion().Clone(), _actionCount, _isDiscrete);

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
                _population.SeedChampionState(NeatCheckpointSerializer.DeserializeGenome(checkpoint), replaceAllTime: true);
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

    // ── Deep diagnostic logging ─────────────────────────────────────────────

    private void LogDeepStats(
        int generation,
        float evaluatedBest,
        float[] fitnessSnapshot,
        SlotGenomeSnapshot[] slotSnapshots,
        string speciesBeforeSummary)
    {
        var sb = new System.Text.StringBuilder();

        // 1. Champion preservation check
        var elite = _population.EliteChampion;
        var historical = _population.AllTimeChampion;
        float slot0Score = fitnessSnapshot.Length > 0 ? fitnessSnapshot[0] : float.NaN;
        sb.AppendLine(
            $"[NEAT/deep] gen={generation} | evaluated_best={evaluatedBest:F3}" +
            $"  elite_slot0_eval={slot0Score:F3}" +
            $"  elite_stored={elite?.Fitness:F3}" +
            $"  historical_best={historical?.Fitness:F3}" +
            $"  (nodes={elite?.Nodes.Count} conns={elite?.Connections.Count(c => c.Enabled)})");

        // 2. Reproduction breakdown
        sb.AppendLine(
            $"[NEAT/deep] reproduction | champion_slot=0  elites={_population.LastEliteCount}" +
            $"  crossover={_population.LastCrossoverCount}" +
            $"  clone+mutate={_population.LastMutationCount}");

        var sortedFitness = fitnessSnapshot.OrderBy(v => v).ToArray();
        sb.AppendLine(
            $"[NEAT/deep] fitness_dist |" +
            $" min={Percentile(sortedFitness, 0f):F3}" +
            $" p10={Percentile(sortedFitness, 0.10f):F3}" +
            $" p25={Percentile(sortedFitness, 0.25f):F3}" +
            $" p50={Percentile(sortedFitness, 0.50f):F3}" +
            $" p75={Percentile(sortedFitness, 0.75f):F3}" +
            $" p90={Percentile(sortedFitness, 0.90f):F3}" +
            $" max={Percentile(sortedFitness, 1f):F3}");

        sb.AppendLine($"[NEAT/deep] species_before | {speciesBeforeSummary}");

        // 3. Per-slot fitness — tabulate all slots using pre-Advance snapshot
        sb.Append("[NEAT/deep] per-slot fitness |");
        for (int i = 0; i < fitnessSnapshot.Length; i++)
        {
            var slot = slotSnapshots[i];
            sb.Append(
                $" [{i}]=fit:{fitnessSnapshot[i]:F2}/adj:{slot.AdjustedFitness:F2}/sp:{slot.SpeciesId}/g:{slot.GenomeId}/n:{slot.Nodes}/c:{slot.EnabledConnections}");
        }
        sb.AppendLine();

        sb.AppendLine($"[NEAT/deep] top_slots | {BuildRankedSlotSummary(slotSnapshots, take: 5, descending: true)}");
        sb.AppendLine($"[NEAT/deep] bottom_slots | {BuildRankedSlotSummary(slotSnapshots, take: 5, descending: false)}");

        // 4. Species composition (new generation — fitness is 0 until next evaluation)
        sb.Append("[NEAT/deep] species |");
        foreach (var s in _population.Species.OrderByDescending(s => s.Members.Count))
        {
            sb.Append($"  sp{s.SpeciesId}(n={s.Members.Count} stag={s.StagnationCounter} age={s.Age})");
        }
        sb.AppendLine();

        // 5. Network distinctness check — run the same test input through the first
        //    4 slots and check that outputs differ (proves slot→network mapping works).
        //    Test 1: neutral (all zeros).  Test 2: "bird below gap, falling" scenario.
        var neutralIn = new float[_obsSize];
        // Approximate: bird at bottom half (by=0.5), falling fast (vy=0.8),
        // pipe dead ahead (dx=0), gap above bird (gapOffset=-0.5).
        var scenarioIn = new float[] { 0.5f, 0.8f, 0f, -0.5f, 0.5f, -0.5f };
        if (scenarioIn.Length < _obsSize)
            scenarioIn = scenarioIn.Concat(new float[_obsSize - scenarioIn.Length]).ToArray();
        else if (scenarioIn.Length > _obsSize)
            Array.Resize(ref scenarioIn, _obsSize);

        sb.Append("[NEAT/deep] slot outputs (neutral | should-flap) |");
        bool allSameNeutral = true;
        float[]? prevNeutralOut = null;
        for (int i = 0; i < Math.Min(4, EffectivePopSize); i++)
        {
            var nOut = _networks[i].Forward(neutralIn);
            var sOut = _networks[i].Forward(scenarioIn);
            string nAction = nOut.Length > 1 && nOut[1] > nOut[0] ? "F" : "I";
            string sAction = sOut.Length > 1 && sOut[1] > sOut[0] ? "F" : "I";
            sb.Append($"  [{i}]={nAction}({(nOut.Length > 0 ? nOut.Max() : 0f):F2})|{sAction}({(sOut.Length > 0 ? sOut.Max() : 0f):F2})");

            if (prevNeutralOut != null && !nOut.SequenceEqual(prevNeutralOut)) allSameNeutral = false;
            prevNeutralOut = nOut;
        }
        if (allSameNeutral && EffectivePopSize > 1)
            sb.Append("  *** WARNING: first 4 slots have IDENTICAL outputs — slot/genome mapping may be broken ***");
        sb.AppendLine();

        GD.Print(sb.ToString().TrimEnd());
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
        var champion = _population.GetPreferredChampion();
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
            _population.SeedChampionState(seed, replaceAllTime: true);
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

    private float ComputeFitnessStdDev(float meanFitness)
    {
        if (_population.Genomes.Count == 0) return 0f;

        float sumSq = 0f;
        foreach (var genome in _population.Genomes)
        {
            float delta = genome.Fitness - meanFitness;
            sumSq += delta * delta;
        }

        return MathF.Sqrt(sumSq / _population.Genomes.Count);
    }

    private int CountDistinctFitnesses()
    {
        var distinct = new HashSet<int>();
        foreach (var genome in _population.Genomes)
            distinct.Add((int)MathF.Round(genome.Fitness * 1000f));
        return distinct.Count;
    }

    private static string FormatDelta(float current, float previous)
    {
        if (float.IsNaN(previous)) return "n/a";
        float delta = current - previous;
        return delta >= 0f ? $"+{delta:F3}" : $"{delta:F3}";
    }

    private string BuildSpeciesFitnessSummary()
    {
        if (_population.Species.Count == 0) return "none";

        return string.Join(" ",
            _population.Species
                .OrderByDescending(s => s.Members.Count)
                .Select(s =>
                {
                    float mean = s.Members.Count > 0 ? s.Members.Average(m => m.Fitness) : 0f;
                    float best = s.Members.Count > 0 ? s.Members.Max(m => m.Fitness) : 0f;
                    return $"sp{s.SpeciesId}(n={s.Members.Count},mean={mean:F3},best={best:F3},stag={s.StagnationCounter},age={s.Age})";
                }));
    }

    private static string BuildRankedSlotSummary(IEnumerable<SlotGenomeSnapshot> slots, int take, bool descending)
    {
        var ranked = descending
            ? slots.OrderByDescending(s => s.Fitness).ThenBy(s => s.Slot)
            : slots.OrderBy(s => s.Fitness).ThenBy(s => s.Slot);

        return string.Join(" ",
            ranked.Take(take)
                  .Select(s => $"slot={s.Slot},fit={s.Fitness:F3},adj={s.AdjustedFitness:F3},sp={s.SpeciesId},g={s.GenomeId},n={s.Nodes},c={s.EnabledConnections}"));
    }

    private static float Percentile(float[] sortedValues, float percentile)
    {
        if (sortedValues.Length == 0) return 0f;
        if (sortedValues.Length == 1) return sortedValues[0];

        float clamped = Math.Clamp(percentile, 0f, 1f);
        float index = clamped * (sortedValues.Length - 1);
        int lower = (int)MathF.Floor(index);
        int upper = (int)MathF.Ceiling(index);
        if (lower == upper) return sortedValues[lower];

        float blend = index - lower;
        return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * blend;
    }
}
