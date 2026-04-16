using System;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Abstract base class for all population-based evolutionary trainers.
///
/// Handles the boilerplate every evolutionary algorithm shares:
///   • Per-slot fitness accumulation and episode counting
///   • The generation barrier (wait until all slots complete their episode budget)
///   • Batch inference loop (SampleActions → SampleForSlot per slot)
///   • Population size validation and mismatch warning
///   • Default zero-value function (evolutionary algorithms have no critic)
///
/// Subclasses must implement:
///   • <see cref="SampleForSlot"/> — forward-pass for a single genome/individual
///   • <see cref="SampleAction"/> — single-agent fallback (usually calls SampleForSlot(obs, 0))
///   • <see cref="TryUpdate"/> — assign fitness, evolve, rebuild networks
///   • <see cref="CreateCheckpoint"/>
///   • <see cref="SnapshotPolicyForEval"/>
///   • <see cref="LoadFromCheckpoint"/>
///
/// Self-registration pattern (copy into each concrete subclass static constructor):
/// <code>
/// static MyEvolutionaryTrainer()
///     => TrainerFactory.Register("MY_ALGO", cfg => new MyEvolutionaryTrainer(cfg));
/// </code>
/// </summary>
public abstract class EvolutionaryTrainer : ITrainer
{
    // ── Protected population bookkeeping ─────────────────────────────────────

    /// <summary>Accumulated reward per agent slot for the current generation.</summary>
    protected float[] FitnessAccum  = Array.Empty<float>();

    /// <summary>Completed episodes per agent slot for the current generation.</summary>
    protected int[]   EpisodeCounts = Array.Empty<int>();

    /// <summary>
    /// Effective population size (clamped if agent count ≠ PopulationSize).
    /// Subclass constructors should set this before any inference calls.
    /// </summary>
    protected int EffectivePopSize;

    private bool _sizeValidated;

    // ── Abstract interface ────────────────────────────────────────────────────

    /// <summary>Run inference for a single agent slot using its individual's network/policy.</summary>
    protected abstract PolicyDecision SampleForSlot(float[] observation, int slot);

    // ITrainer members left to subclass
    public abstract PolicyDecision SampleAction(float[] observation);
    public abstract TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount);
    public abstract RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount);
    public abstract IInferencePolicy SnapshotPolicyForEval();
    public abstract void LoadFromCheckpoint(RLCheckpoint checkpoint);

    // ── ITrainer: batch inference (provided) ──────────────────────────────────

    public virtual PolicyDecision[] SampleActions(VectorBatch observations)
    {
        EnsureSizeValidated(observations.BatchSize);

        var decisions = new PolicyDecision[observations.BatchSize];
        for (int slot = 0; slot < observations.BatchSize; slot++)
            decisions[slot] = SampleForSlot(observations.CopyRow(slot), slot);
        return decisions;
    }

    // ── ITrainer: value function (provided — always zero) ─────────────────────

    /// <summary>Evolutionary algorithms have no value critic. Always returns 0.</summary>
    public float EstimateValue(float[] observation) => 0f;

    /// <summary>Evolutionary algorithms have no value critic. Returns a zero array.</summary>
    public float[] EstimateValues(VectorBatch observations) => new float[observations.BatchSize];

    // ── ITrainer: data collection (provided) ──────────────────────────────────

    public void RecordTransition(Transition t)
    {
        int slot = t.GroupAgentSlot;
        if (slot < 0 || slot >= EffectivePopSize) return;
        if (EpisodeCounts[slot] >= EpisodesPerGenome()) return;

        FitnessAccum[slot] += t.Reward;
        if (t.Done)
            EpisodeCounts[slot]++;
    }

    // ── Generation barrier helpers ────────────────────────────────────────────

    /// <summary>
    /// Returns true when every slot has completed its episode budget.
    /// Call this at the top of <see cref="TryUpdate"/> to implement the generation barrier.
    /// </summary>
    protected bool IsGenerationComplete()
    {
        int budget = EpisodesPerGenome();
        for (int i = 0; i < EffectivePopSize; i++)
            if (EpisodeCounts[i] < budget) return false;
        return true;
    }

    /// <summary>
    /// Zero out both accumulator arrays after evolution.
    /// Call this at the end of a successful <see cref="TryUpdate"/>.
    /// </summary>
    protected void ResetAccumulators()
    {
        Array.Clear(FitnessAccum,  0, FitnessAccum.Length);
        Array.Clear(EpisodeCounts, 0, EpisodeCounts.Length);
    }

    /// <summary>
    /// Initialise the per-slot arrays to <paramref name="popSize"/> elements.
    /// Call once from the subclass constructor after setting <see cref="EffectivePopSize"/>.
    /// </summary>
    protected void InitAccumulators(int popSize)
    {
        EffectivePopSize = popSize;
        FitnessAccum     = new float[popSize];
        EpisodeCounts    = new int[popSize];
    }

    // ── Size validation ───────────────────────────────────────────────────────

    /// <summary>
    /// Called on the first <see cref="SampleActions"/> batch. Warns and clamps
    /// <see cref="EffectivePopSize"/> if the agent count doesn't match.
    /// </summary>
    protected void EnsureSizeValidated(int batchSize)
    {
        if (_sizeValidated) return;
        _sizeValidated = true;

        if (batchSize == EffectivePopSize) return;

        GD.PushWarning(
            $"[{GetType().Name}] PopulationSize ({EffectivePopSize}) != agent count ({batchSize}). " +
            $"Clamping to {Math.Min(batchSize, EffectivePopSize)}. " +
            $"Set PopulationSize = number of agents in this policy group.");

        EffectivePopSize = Math.Min(batchSize, EffectivePopSize);
    }

    // ── Hook for episode budget ───────────────────────────────────────────────

    /// <summary>
    /// Returns the per-genome episode budget for the current config.
    /// Default reads from the algorithm config cast to <see cref="RLEvolutionaryConfig"/>.
    /// Override if your subclass stores this value directly.
    /// </summary>
    protected virtual int EpisodesPerGenome() => _episodesPerGenome;

    private int _episodesPerGenome = 1;

    /// <summary>
    /// Set the per-genome episode budget. Call from the subclass constructor
    /// after reading it from the algorithm config.
    /// </summary>
    protected void SetEpisodesPerGenome(int value) => _episodesPerGenome = Math.Max(1, value);
}
