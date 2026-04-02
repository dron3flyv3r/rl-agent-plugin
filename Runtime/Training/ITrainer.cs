using System;
using Godot;

namespace RlAgentPlugin.Runtime;

public enum RLAlgorithmKind
{
    PPO = 0,
    SAC = 1,
    /// <summary>
    /// Deep Q-Network. Off-policy, discrete actions only.
    /// Set <see cref="RLDQNConfig.UseDoubleDqn"/> to true to enable Double DQN.
    /// </summary>
    DQN = 2,
    /// <summary>
    /// Advantage Actor-Critic. On-policy, discrete and continuous actions.
    /// Simpler than PPO (no clipping), suitable as a baseline.
    /// </summary>
    A2C = 3,
    /// <summary>
    /// Monte Carlo Tree Search. Pure planning — no learning.
    /// Requires a registered <see cref="IEnvironmentModel"/> via <c>MctsTrainer.SetEnvironmentModel</c>.
    /// Discrete action spaces only.
    /// </summary>
    MCTS = 4,
    /// <summary>
    /// Use a custom trainer registered via <see cref="TrainerFactory.Register"/>.
    /// Set <see cref="PolicyGroupConfig.CustomTrainerId"/> to the registered key.
    /// </summary>
    Custom = 99,
}

public sealed class PolicyDecision
{
    /// <summary>Sampled discrete action index, or -1 if continuous-only.</summary>
    public int DiscreteAction { get; init; } = -1;
    /// <summary>Sampled continuous action vector (empty for discrete-only).</summary>
    public float[] ContinuousActions { get; init; } = Array.Empty<float>();
    public RecurrentState? RecurrentState { get; init; }
    public float LogProbability { get; init; }
    public float Value { get; init; }
    public float Entropy { get; init; }
}

public sealed class Transition
{
    public float[] Observation { get; init; } = Array.Empty<float>();
    /// <summary>Taken discrete action index, or -1 if continuous.</summary>
    public int DiscreteAction { get; init; } = -1;
    public float[] ContinuousActions { get; init; } = Array.Empty<float>();
    public float Reward { get; init; }
    public bool Done { get; init; }
    /// <summary>Next observation (used by SAC for target Q computation).</summary>
    public float[] NextObservation { get; init; } = Array.Empty<float>();
    /// <summary>Log probability of the taken action (used by PPO).</summary>
    public float OldLogProbability { get; init; }
    /// <summary>Value estimate at this state (used by PPO for GAE).</summary>
    public float Value { get; init; }
    /// <summary>Value estimate at next state (used by PPO for GAE).</summary>
    public float NextValue { get; init; }
    /// <summary>
    /// Recurrent hidden state snapshot at the START of this step (h vector).
    /// Null for feedforward agents or the initial step of a new episode.
    /// </summary>
    public float[]? HiddenState { get; init; }
    /// <summary>
    /// LSTM cell state snapshot at the START of this step (c vector).
    /// Null for GRU agents, feedforward agents, or the initial step.
    /// </summary>
    public float[]? CellState { get; init; }
    /// <summary>
    /// Index of this agent within its policy group's decision batch.
    /// Used to group transitions into per-agent sequences for recurrent BPTT.
    /// Defaults to 0 (single-agent groups).
    /// </summary>
    public int GroupAgentSlot { get; init; }
}

public sealed class TrainerUpdateStats
{
    public float PolicyLoss { get; init; }
    public float ValueLoss { get; init; }
    public float Entropy { get; init; }
    public float ClipFraction { get; init; }
    public float? SacAlpha { get; init; }
    public RLCheckpoint Checkpoint { get; init; } = new();
}

public sealed class PolicyGroupConfig
{
    public string GroupId { get; init; } = string.Empty;
    public string RunId { get; init; } = string.Empty;
    public RLAlgorithmKind Algorithm { get; init; } = RLAlgorithmKind.Custom;
    /// <summary>Key passed to <see cref="TrainerFactory.Register"/> when Algorithm is Custom.</summary>
    public string CustomTrainerId { get; init; } = string.Empty;
    public RLPolicyGroupConfig? SharedPolicy { get; init; }
    public RLTrainerConfig TrainerConfig { get; init; } = new();
    public RLNetworkGraph NetworkGraph { get; init; } = new();
    public RLActionDefinition[] ActionDefinitions { get; init; } = Array.Empty<RLActionDefinition>();
    public int ObservationSize { get; init; }
    public int DiscreteActionCount { get; init; }
    public int ContinuousActionDimensions { get; init; }
    /// <summary>
    /// Multi-stream observation spec. When non-null, trainers use the spec-aware
    /// <see cref="PolicyValueNetwork"/> constructor instead of the flat int path.
    /// Null means legacy flat observations.
    /// </summary>
    public ObservationSpec? ObsSpec { get; init; }
    public string CheckpointPath { get; init; } = string.Empty;
    public string MetricsPath { get; init; } = string.Empty;
}

/// <summary>
/// Optional extension for trainers that support offloading gradient updates to a background thread.
/// <para>
/// Contract: call <see cref="TryScheduleBackgroundUpdate"/> after the rollout buffer is full;
/// call <see cref="TryPollResult"/> at the start of the next frame to pick up the result.
/// Do not call <see cref="TryScheduleBackgroundUpdate"/> again until <see cref="TryPollResult"/>
/// has returned a non-null value (i.e. the previous result has been applied).
/// </para>
/// <para>
/// Only works for trainers whose <c>TryUpdate</c> logic is pure math (no Godot API calls).
/// Custom <see cref="ITrainer"/> implementations that call Godot scene tree API inside
/// <c>TryUpdate</c> must NOT implement this interface.
/// </para>
/// </summary>
public interface IAsyncTrainer : ITrainer
{
    /// <summary>
    /// Snapshots the current network weights and queues a background Task to run backprop.
    /// Returns <c>true</c> if a job was queued, <c>false</c> if the buffer isn't full yet or
    /// a job is already in flight / awaiting poll.
    /// </summary>
    /// <param name="maxTransitions">
    /// Maximum number of transitions to include in this update's batch.
    /// Transitions beyond the cap are discarded when the snapshot is taken.
    /// Defaults to <see cref="int.MaxValue"/> (no cap).
    /// </param>
    bool TryScheduleBackgroundUpdate(string groupId, long totalSteps, long episodeCount, int maxTransitions = int.MaxValue);

    /// <summary>
    /// Returns training stats if the background job completed since the last poll; <c>null</c>
    /// otherwise. Applying the trained weights to the live network happens on this call (main thread).
    /// </summary>
    TrainerUpdateStats? TryPollResult(string groupId, long totalSteps, long episodeCount);

    /// <summary>
    /// Blocks until any in-flight background job completes and applies its result. Returns the
    /// stats, or <c>null</c> if there was nothing pending. Call from <c>_ExitTree</c> for a
    /// clean shutdown.
    /// </summary>
    TrainerUpdateStats? FlushPendingUpdate(string groupId, long totalSteps, long episodeCount);
}

public interface ITrainer
{
    PolicyDecision SampleAction(float[] observation);
    PolicyDecision[] SampleActions(VectorBatch observations);
    /// <summary>Returns a value estimate (PPO: value head; SAC: returns 0).</summary>
    float EstimateValue(float[] observation);
    float[] EstimateValues(VectorBatch observations);
    void RecordTransition(Transition transition);
    TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount);
    RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount);

    // ── Recurrent policy support ──────────────────────────────────────────────

    /// <summary>
    /// True when the trainer's policy network contains recurrent trunk layers (LSTM/GRU).
    /// When true, <see cref="TrainingBootstrap"/> calls <see cref="SampleActionRecurrent"/>
    /// per-agent instead of batch <see cref="SampleActions"/>.
    /// </summary>
    bool HasRecurrentPolicy => false;

    /// <summary>
    /// Returns a zero-initialised <see cref="RecurrentState"/> compatible with this trainer's
    /// recurrent network. Returns null for feedforward trainers.
    /// </summary>
    RecurrentState? CreateZeroRecurrentState() => null;

    /// <summary>
    /// Single-step recurrent inference. Updates <paramref name="state"/> in-place and returns
    /// the policy decision. For feedforward trainers falls back to <see cref="SampleAction"/>.
    /// </summary>
    PolicyDecision SampleActionRecurrent(float[] observation, RecurrentState state)
        => SampleAction(observation);

    /// <summary>
    /// Value estimate for the given observation and recurrent state without mutating the state.
    /// Feedforward trainers fall back to <see cref="EstimateValue"/>.
    /// </summary>
    float EstimateValueRecurrent(float[] observation, RecurrentState state)
        => EstimateValue(observation);

    /// <summary>
    /// Returns a greedy/deterministic inference policy backed by a weight snapshot of the
    /// current network. The returned policy is fully independent of the trainer — safe to
    /// use from the main thread while training continues.
    /// No optimizer state, replay buffer, or rollout buffer is touched.
    /// </summary>
    IInferencePolicy SnapshotPolicyForEval();

    /// <summary>
    /// Loads network weights from a checkpoint into the live trainer network.
    /// Call this immediately after trainer construction to resume a previous training run.
    /// The optimizer state and replay/rollout buffers are not affected.
    /// Custom trainers that do not override this will log a warning and start with fresh weights.
    /// </summary>
    void LoadFromCheckpoint(RLCheckpoint checkpoint)
    {
        GD.PushWarning($"[RL Resume] Trainer '{GetType().Name}' does not implement LoadFromCheckpoint — resuming with fresh weights.");
    }
}
