using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Abstract base resource for all population-based evolutionary algorithm configs.
///
/// Subclass this to create a new evolutionary algorithm config (e.g. CMA-ES, OpenAI ES).
/// The two properties here are generic to every evolutionary algorithm:
///   - <see cref="PopulationSize"/> — how many individuals to evaluate per generation
///   - <see cref="EpisodesPerGenome"/> — how many episodes each individual runs before fitness is assigned
///
/// Algorithm-specific hyperparameters (mutation rates, speciation, etc.) belong in the subclass.
/// </summary>
[GlobalClass]
[Tool]
public abstract partial class RLEvolutionaryConfig : RLAlgorithmConfig
{
    // ── Population ────────────────────────────────────────────────────────────

    /// <summary>
    /// Number of individuals in the population.
    /// Must equal the number of agents assigned to this policy group in the scene
    /// (or the <see cref="RLAgentSpawner.TrainingCount"/> when using the spawner).
    /// </summary>
    [Export(PropertyHint.Range, "2,1000,1,or_greater")]
    public int PopulationSize { get; set; } = 50;

    /// <summary>
    /// Number of complete episodes each individual is evaluated before the generation
    /// ends and evolution runs. Higher values reduce fitness noise at the cost of
    /// wall-clock time per generation.
    /// </summary>
    [Export(PropertyHint.Range, "1,100,1,or_greater")]
    public int EpisodesPerGenome { get; set; } = 1;

    // ── RLAlgorithmConfig overrides ───────────────────────────────────────────

    /// <summary>All evolutionary algorithms register as Custom trainers.</summary>
    public override RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.Custom;

    // AlgorithmKind, SupportsDiscreteActions, SupportsContinuousActions, SupportsMultiAgent,
    // IsOnPolicy, and ApplyTo are left to each concrete subclass.
}
