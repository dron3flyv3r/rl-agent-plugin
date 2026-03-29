using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Defines a self-play matchup between two policy groups and how opponents are sampled.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLPolicyPairingConfig : Resource
{
    private Resource? _groupA;
    private Resource? _groupB;

    /// <summary>
    /// Optional stable ID for this matchup; auto-generated from group IDs when blank.
    /// </summary>
    [Export] public string PairingId { get; set; } = string.Empty;

    [ExportGroup("Groups")]
    /// <summary>First policy group participating in this matchup.</summary>
    [Export(PropertyHint.ResourceType, nameof(RLPolicyGroupConfig))]
    public Resource? GroupA
    {
        get => _groupA;
        set => _groupA = value;
    }

    /// <summary>Second policy group participating in this matchup.</summary>
    [Export(PropertyHint.ResourceType, nameof(RLPolicyGroupConfig))]
    public Resource? GroupB
    {
        get => _groupB;
        set => _groupB = value;
    }

    [ExportGroup("Training")]
    /// <summary>Whether Group A is trainable in this matchup.</summary>
    [Export] public bool TrainGroupA { get; set; } = true;
    /// <summary>Whether Group B is trainable in this matchup.</summary>
    [Export] public bool TrainGroupB { get; set; } = true;
    /// <summary>
    /// Probability of sampling historical opponents instead of current latest policy.
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.01")] public float HistoricalOpponentRate { get; set; } = 0.5f;
    /// <summary>
    /// Save a frozen opponent snapshot every N checkpoint writes.
    /// </summary>
    [Export(PropertyHint.Range, "1,100000,1,or_greater")] public int FrozenCheckpointInterval { get; set; } = 10;

    [ExportGroup("PFSP")]
    /// <summary>Maximum number of historical opponent snapshots to keep per group.</summary>
    [Export(PropertyHint.Range, "0,100,or_greater")] public int   MaxPoolSize  { get; set; } = 20;
    /// <summary>Enable Prioritized Fictitious Self-Play sampling.</summary>
    [Export] public bool  PfspEnabled  { get; set; } = true;
    /// <summary>PFSP weighting exponent; larger values bias harder opponents.</summary>
    [Export(PropertyHint.Range, "0.0,10.0,0.1")] public float PfspAlpha    { get; set; } = 4.0f;
    /// <summary>Minimum win-rate threshold considered "solved" for PFSP weighting.</summary>
    [Export(PropertyHint.Range, "0.0,1.0,0.01")] public float WinThreshold { get; set; } = 0.0f;

    public RLPolicyGroupConfig? ResolvedGroupA => _groupA as RLPolicyGroupConfig;
    public RLPolicyGroupConfig? ResolvedGroupB => _groupB as RLPolicyGroupConfig;
}
