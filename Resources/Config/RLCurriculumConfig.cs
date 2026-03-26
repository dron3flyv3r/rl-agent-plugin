using Godot;

namespace RlAgentPlugin.Runtime;

public enum RLCurriculumMode
{
    /// <summary>Advance curriculum based on global training step progress.</summary>
    StepBased = 0,
    /// <summary>Adapt curriculum by recent episode success-rate.</summary>
    SuccessRate = 1,
}

/// <summary>
/// Curriculum progression settings used to control environment difficulty over time.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLCurriculumConfig : Resource
{
    /// <summary>
    /// Curriculum progression strategy: step-based or success-rate adaptive.
    /// </summary>
    [Export] public RLCurriculumMode Mode { get; set; } = RLCurriculumMode.StepBased;
    /// <summary>
    /// Total combined steps to reach full curriculum progress (step-based mode).
    /// </summary>
    [Export(PropertyHint.Range, "0,10000000,1,or_greater")] public long MaxSteps { get; set; } = 0;
    /// <summary>
    /// Number of recent episodes used to compute success rate (success-rate mode).
    /// </summary>
    [Export(PropertyHint.Range, "1,10000,1,or_greater")] public int SuccessWindowEpisodes { get; set; } = 25;
    /// <summary>
    /// Episode reward threshold counted as a success in success-rate mode.
    /// </summary>
    [Export] public float SuccessRewardThreshold { get; set; } = 1.0f;
    /// <summary>
    /// Promote difficulty when success rate is at or above this value.
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.01")] public float PromoteThreshold { get; set; } = 0.8f;
    /// <summary>
    /// Demote difficulty when success rate is at or below this value.
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.01")] public float DemoteThreshold { get; set; } = 0.2f;
    /// <summary>
    /// Curriculum progress increment when promotion criteria is met.
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.01")] public float ProgressStepUp { get; set; } = 0.1f;
    /// <summary>
    /// Curriculum progress decrement when demotion criteria is met.
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.01")] public float ProgressStepDown { get; set; } = 0.1f;
    /// <summary>
    /// Require a full success-rate window before adapting difficulty.
    /// </summary>
    [Export] public bool RequireFullWindow { get; set; } = true;
    /// <summary>
    /// Editor/debug override for curriculum progress (0..1), mainly for local testing.
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.01")] public float DebugProgress { get; set; } = 0f;
}
