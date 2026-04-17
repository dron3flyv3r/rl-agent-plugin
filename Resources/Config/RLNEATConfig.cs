using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// NEAT (NeuroEvolution of Augmenting Topologies) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
///
/// NEAT evolves both the weights and topology of neural networks via a population
/// of genomes. Each agent in the policy group evaluates one genome per episode.
///
/// IMPORTANT: <see cref="RLEvolutionaryConfig.PopulationSize"/> must equal the number of agents
/// assigned to this policy group in the scene.  The RLNetworkGraph resource is ignored —
/// NEAT constructs its own variable-topology networks from genome genes.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLNEATConfig : RLEvolutionaryConfig
{
    // ── Speciation ────────────────────────────────────────────────────────────

    /// <summary>
    /// Genomes with compatibility distance δ below this threshold are considered the same species.
    /// Increase to allow more diversity; decrease to merge species more aggressively.
    /// </summary>
    [Export(PropertyHint.Range, "0.1,10.0,0.1")] public float CompatibilityThreshold { get; set; } = 3.0f;

    /// <summary>Weight for excess genes in the compatibility distance formula (C1).</summary>
    [Export(PropertyHint.Range, "0.0,5.0,0.1")] public float ExcessCoeff { get; set; } = 1.0f;

    /// <summary>Weight for disjoint genes in the compatibility distance formula (C2).</summary>
    [Export(PropertyHint.Range, "0.0,5.0,0.1")] public float DisjointCoeff { get; set; } = 1.0f;

    /// <summary>Weight for average weight difference in the compatibility distance formula (C3).</summary>
    [Export(PropertyHint.Range, "0.0,5.0,0.1")] public float WeightDiffCoeff { get; set; } = 0.4f;

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// <summary>Probability that a genome's weights are mutated each generation.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float WeightMutationRate { get; set; } = 0.8f;

    /// <summary>
    /// Given weight mutation is active, probability of perturbing an existing weight
    /// (vs. replacing it with a random value).
    /// </summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float WeightPerturbRate { get; set; } = 0.9f;

    /// <summary>Standard deviation of Gaussian noise applied during weight perturbation.</summary>
    [Export(PropertyHint.Range, "0.001,2.0,0.01")] public float WeightPerturbScale { get; set; } = 0.1f;

    /// <summary>Standard deviation of Gaussian distribution used for full weight reset.</summary>
    [Export(PropertyHint.Range, "0.01,5.0,0.01")] public float WeightResetScale { get; set; } = 1.0f;

    /// <summary>Probability that a random new connection is added between two existing nodes.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.001")] public float AddConnectionRate { get; set; } = 0.05f;

    /// <summary>
    /// Probability that a random existing connection is split by inserting a new hidden node.
    /// </summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.001")] public float AddNodeRate { get; set; } = 0.03f;

    /// <summary>Probability that a random connection is toggled enabled/disabled.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.001")] public float ToggleConnectionRate { get; set; } = 0.01f;

    /// <summary>Activation function for newly-added hidden nodes.</summary>
    [Export] public RLActivationKind HiddenActivation { get; set; } = RLActivationKind.Tanh;

    // ── Reproduction ──────────────────────────────────────────────────────────

    /// <summary>
    /// Fraction of offspring produced via crossover (vs. clone + mutate).
    /// Requires at least 2 members in the species breeding pool.
    /// </summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float CrossoverRate { get; set; } = 0.75f;

    /// <summary>
    /// Number of top-performing genomes copied unchanged per species each generation (elitism).
    /// Only applied when the species quota is >= ElitismCount.
    /// </summary>
    [Export(PropertyHint.Range, "0,10,1")] public int ElitismCount { get; set; } = 1;

    /// <summary>
    /// Top fraction of each species allowed to act as breeding parents.
    /// E.g. 0.2 = only the best 20% can reproduce.
    /// </summary>
    [Export(PropertyHint.Range, "0.05,1.0,0.05")] public float SurvivalThreshold { get; set; } = 0.2f;

    /// <summary>
    /// Fraction of input→output connections present in the initial minimal population.
    /// 1.0 = fully connected; lower values = sparser starting topology.
    /// </summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.05")] public float InitialConnectionDensity { get; set; } = 1.0f;

    // ── Dynamic speciation ────────────────────────────────────────────────────

    /// <summary>
    /// When > 0, the compatibility threshold is automatically tuned each generation
    /// so that the number of species stays near this value.
    /// Recommended: roughly PopulationSize / 4 (e.g. 5 for 20 agents).
    /// Set to 0 to disable dynamic threshold and use the fixed <see cref="CompatibilityThreshold"/>.
    /// </summary>
    [Export(PropertyHint.Range, "0,50,1")] public int TargetSpeciesCount { get; set; } = 5;

    /// <summary>
    /// Amount by which the dynamic compatibility threshold is adjusted per generation
    /// when the species count is above or below <see cref="TargetSpeciesCount"/>.
    /// Smaller values = smoother but slower convergence.
    /// </summary>
    [Export(PropertyHint.Range, "0.01,1.0,0.01")] public float ThresholdAdjustRate { get; set; } = 0.1f;

    // ── Stagnation ────────────────────────────────────────────────────────────

    /// <summary>
    /// If a species shows no improvement for this many generations, it is removed
    /// (unless it contains the all-time champion or only 2 species remain).
    /// </summary>
    [Export(PropertyHint.Range, "1,200,1")] public int StagnationLimit { get; set; } = 25;

    // ── Checkpointing ─────────────────────────────────────────────────────────

    /// <summary>
    /// When true, the checkpoint stores all N genomes (true generation resume).
    /// When false (default), only the champion genome is saved; on resume, the
    /// population is warm-started as N mutated copies of the champion.
    /// </summary>
    [Export] public bool SaveFullPopulation { get; set; }

    // ── RLEvolutionaryConfig overrides ───────────────────────────────────────

    // AlgorithmKind => Custom is inherited from RLEvolutionaryConfig.
    public override bool SupportsDiscreteActions         => true;
    public override bool SupportsContinuousActions       => true;
    public override bool SupportsMultiAgent              => true;
    public override bool IsOnPolicy                      => true;

    /// <inheritdoc />
    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm            = RLAlgorithmKind.Custom;
        config.CustomTrainerId      = "NEAT";
        config.StatusWriteIntervalSteps = StatusWriteIntervalSteps;
        // NEAT does not use gradient-based optimisation; most RLTrainerConfig fields
        // are irrelevant and are left at their defaults.
    }
}
