using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Defines one logical policy group, including identity, optional inference model,
/// and trainable network architecture.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLPolicyGroupConfig : Resource
{
    private Resource? _networkGraph = new RLNetworkGraph();

    /// <summary>
    /// Policy group identifier used to match agents via <c>IRLAgent.PolicyGroupId</c>.
    /// </summary>
    [Export] public string AgentId { get; set; } = string.Empty;
    /// <summary>
    /// Optional per-episode step cap for agents in this group (0 = unlimited).
    /// </summary>
    [Export(PropertyHint.Range, "0,1000,or_greater")] public int MaxEpisodeSteps { get; set; } = 0;
    /// <summary>
    /// Optional model path used for inference or warm-starting this policy group.
    /// </summary>
    [Export(PropertyHint.File, "*.rlmodel")] public string InferenceModelPath { get; set; } = string.Empty;

    [ExportGroup("Inference")]
    /// <summary>
    /// When enabled, actions are sampled from the policy distribution rather than
    /// taking the greedy/mean action. Adds natural behavioural variation between
    /// agents sharing the same model and helps verify that a policy genuinely
    /// learned rather than memorising a single deterministic path.
    /// </summary>
    [Export] public bool StochasticInference { get; set; } = false;

    [ExportGroup("Network")]
    /// <summary>
    /// Network architecture resource used when creating a trainable policy.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLNetworkGraph))]
    public Resource? NetworkGraph
    {
        get => _networkGraph;
        set => _networkGraph = value;
    }

    public RLNetworkGraph? ResolvedNetworkGraph => _networkGraph as RLNetworkGraph;
}
