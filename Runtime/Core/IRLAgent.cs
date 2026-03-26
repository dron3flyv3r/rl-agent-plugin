using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Common interface implemented by both RLAgent2D and RLAgent3D.
/// Allows the training/inference framework to work with agents
/// regardless of whether they live in a 2D or 3D scene.
/// </summary>
public interface IRLAgent
{
    /// <summary>Returns the underlying Node (this). Lets framework code call Node-level APIs such as GetPathTo.</summary>
    Node AsNode();

    // ── Control ──────────────────────────────────────────────────────────────
    RLAgentControlMode ControlMode { get; set; }
    RLPolicyGroupConfig PolicyGroupConfig { get; }

    // ── Episode state ─────────────────────────────────────────────────────────
    int EpisodeSteps { get; }
    float EpisodeReward { get; }
    int CurrentActionIndex { get; }
    float[] CurrentContinuousActions { get; }
    bool IsDone { get; }
    float PendingReward { get; }
    float LastStepReward { get; }
    IReadOnlyList<ObservationSegment> LastObservationSegments { get; }

    // ── Actions ───────────────────────────────────────────────────────────────
    void ApplyAction(int action);
    void ApplyAction(float[] continuousActions);
    int GetDiscreteActionCount();
    int GetContinuousActionDimensions();
    bool SupportsOnlyDiscreteActions();
    string[] GetDiscreteActionLabels();
    RLActionDefinition[] GetActionSpace();

    // ── Observations ──────────────────────────────────────────────────────────
    float[] GetLastObservation();

    // ── Framework-internal ── observation spec ────────────────────────────────
    /// <summary>
    /// Probes the agent's observation collection once and returns the resulting
    /// <see cref="ObservationSpec"/>. The result is cached; subsequent calls return the
    /// same object. Used by the training bootstrap to build per-group network specs.
    /// </summary>
    ObservationSpec CollectObservationSpec();

    // ── Reward queries ────────────────────────────────────────────────────────
    IReadOnlyDictionary<string, float> GetPendingRewardBreakdown();
    IReadOnlyDictionary<string, float> GetLastStepRewardBreakdown();
    IReadOnlyDictionary<string, float> GetEpisodeRewardBreakdown();

    // ── Episode management ────────────────────────────────────────────────────
    void EndEpisode();
    void ResetEpisode();
    void NotifyCurriculumProgress(float progress);
    bool HasReachedEpisodeLimit();
    void AccumulateReward(float reward, IReadOnlyDictionary<string, float>? rewardBreakdown = null);

    // ── Config ────────────────────────────────────────────────────────────────
    string GetInferenceModelPath();
    string GetPolicyAgentId();

    // ── Framework-internal ────────────────────────────────────────────────────
    float[] CollectObservationArray();
    void TickStep();
    void HandleHumanInput();
    float ConsumePendingReward();
    Dictionary<string, float> ConsumePendingRewardBreakdown();
    bool ConsumeDonePending();
}
