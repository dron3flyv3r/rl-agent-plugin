using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Container resource for all configured self-play pairings in a training run.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLSelfPlayConfig : Resource
{
    /// <summary>
    /// List of policy-group matchups used to configure self-play training.
    /// </summary>
    [Export] public Godot.Collections.Array<RLPolicyPairingConfig> Pairings { get; set; } = new();
}
