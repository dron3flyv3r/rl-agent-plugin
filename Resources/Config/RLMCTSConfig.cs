using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// MCTS (Monte Carlo Tree Search) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
///
/// MCTS is a <b>pure planning</b> algorithm — it does not learn from experience.
/// At each decision point it runs <see cref="NumSimulations"/> simulated rollouts using your
/// <see cref="IEnvironmentModel"/> implementation to select the best action.
///
/// <para>
/// <b>Requirements:</b>
/// <list type="bullet">
/// <item><description>Discrete action spaces only.</description></item>
/// <item><description>You must implement <see cref="IEnvironmentModel"/> and call
/// <c>MctsTrainer.SetEnvironmentModel(this)</c> before training starts.</description></item>
/// </list>
/// </para>
/// </summary>
[GlobalClass]
[Tool]
public partial class RLMCTSConfig : RLAlgorithmConfig
{
    /// <summary>Number of simulated rollouts run per action decision. Higher = stronger but slower.</summary>
    [Export] public int NumSimulations { get; set; } = 50;

    /// <summary>Maximum depth for the selection phase of each simulation.</summary>
    [Export] public int MaxSearchDepth { get; set; } = 20;

    /// <summary>Depth of the random rollout used to evaluate a leaf node.</summary>
    [Export] public int RolloutDepth { get; set; } = 10;

    /// <summary>UCT exploration constant (c). Sqrt(2) ≈ 1.414 is the theoretical default.</summary>
    [Export] public float ExplorationConstant { get; set; } = 1.414f;

    /// <summary>Discount factor applied during simulated rollouts.</summary>
    [Export] public float Gamma { get; set; } = 0.99f;

    // ── Capability overrides ─────────────────────────────────────────────────

    public override RLAlgorithmKind AlgorithmKind  => RLAlgorithmKind.MCTS;
    public override bool SupportsDiscreteActions   => true;
    public override bool SupportsContinuousActions => false;
    public override bool IsOnPolicy                => true;  // no replay buffer
    public override bool SupportsMultiAgent        => true;  // each agent plans independently

    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm                 = RLAlgorithmKind.MCTS;
        config.MctsNumSimulations        = NumSimulations;
        config.MctsMaxSearchDepth        = MaxSearchDepth;
        config.MctsRolloutDepth          = RolloutDepth;
        config.MctsExplorationConstant   = ExplorationConstant;
        config.MctsGamma                 = Gamma;
        config.StatusWriteIntervalSteps  = StatusWriteIntervalSteps;
    }
}
