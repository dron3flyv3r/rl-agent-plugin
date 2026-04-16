using System.Collections.Generic;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

public enum WizardMode
{
    Fresh,
    Existing,
}

public enum WizardDimension
{
    TwoD,
    ThreeD,
}

public enum WizardTrainingMode
{
    SingleAgent,
    SharedPolicy,
    IndividualPolicies,
    SelfPlay,
}

public enum WizardAlgorithm
{
    PPO,
    SAC,
    DQN,
    A2C,
}

public enum WizardNetworkPreset
{
    Tiny,
    Small,
    Medium,
    Large,
}

/// <summary>
/// Holds all user choices made across wizard steps. Passed to the plugin's apply logic after the user clicks Apply.
/// </summary>
public sealed class RLSetupWizardState
{
    public WizardMode Mode { get; set; } = WizardMode.Fresh;
    /// <summary>Only relevant for Fresh mode — determines which agent node type is inserted.</summary>
    public WizardDimension Dimension { get; set; } = WizardDimension.TwoD;
    public WizardTrainingMode TrainingMode { get; set; } = WizardTrainingMode.SingleAgent;
    /// <summary>
    /// For Single/SharedPolicy: the selected character nodes (fresh) or agent nodes (existing).
    /// For SelfPlay: Group A nodes.
    /// </summary>
    public List<Node> GroupANodes { get; } = new();
    /// <summary>For SelfPlay only: Group B nodes.</summary>
    public List<Node> GroupBNodes { get; } = new();
    public WizardAlgorithm Algorithm { get; set; } = WizardAlgorithm.PPO;
    public WizardNetworkPreset NetworkPreset { get; set; } = WizardNetworkPreset.Small;
    public RLActivationKind Activation { get; set; } = RLActivationKind.Tanh;
    public RLOptimizerKind Optimizer { get; set; } = RLOptimizerKind.Adam;
    public int MaxEpisodeSteps { get; set; } = 0;
    public int ActionRepeat { get; set; } = 1;
    public float SimulationSpeed { get; set; } = 1.0f;
    public int CheckpointInterval { get; set; } = 10;
    public string RunPrefix { get; set; } = string.Empty;
}
