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
    bool HasReachedEpisodeLimit();
    void AccumulateReward(float reward, IReadOnlyDictionary<string, float>? rewardBreakdown = null);

    // ── Config ────────────────────────────────────────────────────────────────
    string GetInferenceModelPath();
    string GetPolicyAgentId();

    // ── Recurrent state (optional) ────────────────────────────────────────────
    /// <summary>
    /// Per-agent recurrent hidden state, maintained by the training bootstrap.
    /// Null for feedforward agents. Reset to null (zeroed on next inference) when
    /// the episode ends (IsDone == true).
    /// </summary>
    RecurrentState? RecurrentHiddenState { get; set; }

    // ── Recording ─────────────────────────────────────────────────────────────
    /// <summary>
    /// When >= 0, the player/character script should call ApplyAction with this value
    /// instead of reading keyboard input, then reset to -1. Used by RecordingBootstrap
    /// in step mode so the dock can inject specific discrete actions.
    /// </summary>
    int PendingStepAction { get; set; }

    // ── Framework-internal ────────────────────────────────────────────────────
    float[] CollectObservationArray();
    void TickStep();
    void HandleHumanInput();
    /// <summary>
    /// Called by RecordingBootstrap in Script mode each physics step.
    /// Override in your agent to implement a heuristic / scripted policy
    /// (call ApplyAction inside the override).
    /// </summary>
    void HandleScriptedInput();
    float ConsumePendingReward();
    Dictionary<string, float> ConsumePendingRewardBreakdown();
    bool ConsumeDonePending();
}
