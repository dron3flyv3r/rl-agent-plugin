using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// DQN (Deep Q-Network) trainer. Supports standard DQN and Double DQN.
/// <para>
/// <b>Action space:</b> Discrete only. Continuous actions are not supported.
/// </para>
/// <para>
/// <b>Policy:</b> Off-policy with experience replay. Epsilon-greedy exploration
/// decays from <c>EpsilonStart</c> to <c>EpsilonEnd</c> over <c>EpsilonDecaySteps</c>.
/// </para>
/// <para>
/// <b>Target network:</b> Hard-copied from the online network every <c>TargetUpdateInterval</c> steps.
/// Double DQN decouples action selection and evaluation to reduce overestimation.
/// </para>
/// </summary>
public sealed class DqnTrainer : ITrainer, IAsyncTrainer
{
    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _trainerConfig;
    private readonly DqnNetwork _network;
    private readonly ReplayBuffer<Transition> _buffer;
    private readonly Random _rng;
    private readonly int _actionCount;

    private long _totalStepsSeen;
    private long _nextTargetUpdate;

    // ── Dyna-Q world model (null when DynaModelUpdatesPerStep == 0) ───────────
    private readonly DynaWorldModel? _dynaModel;

    // ── Async gradient update state ───────────────────────────────────────────
    private DqnNetwork? _shadowNetwork;
    private Task<DqnAsyncResult>? _pendingUpdate;

    private sealed class DqnAsyncResult
    {
        public float MeanLoss { get; init; }
        public RLCheckpoint Checkpoint { get; init; } = new();
    }

    public DqnTrainer(PolicyGroupConfig config)
    {
        if (config.DiscreteActionCount <= 0)
            throw new InvalidOperationException(
                $"[DQN] Only discrete action spaces are supported. " +
                $"Group '{config.GroupId}' has no discrete actions. Use SAC or PPO for continuous control.");

        if (config.ContinuousActionDimensions > 0)
            throw new InvalidOperationException(
                $"[DQN] Continuous actions are not supported. " +
                $"Group '{config.GroupId}' has continuous actions. Use SAC or PPO instead.");

        _config        = config;
        _trainerConfig = config.TrainerConfig;
        _actionCount   = config.DiscreteActionCount;
        _rng           = new Random();

        var obsSize = config.ObsSpec?.TotalSize ?? config.ObservationSize;
        _network = new DqnNetwork(obsSize, _actionCount, config.NetworkGraph);
        _buffer  = new ReplayBuffer<Transition>(_trainerConfig.ReplayBufferCapacity);

        _nextTargetUpdate = _trainerConfig.DqnTargetUpdateInterval;

        if (_trainerConfig.DynaModelUpdatesPerStep > 0)
            _dynaModel = new DynaWorldModel(obsSize, _actionCount);

        GD.Print($"[DQN] Group '{config.GroupId}': obs={obsSize}, actions={_actionCount}, " +
                 $"double={_trainerConfig.DqnUseDouble}, buffer={_trainerConfig.ReplayBufferCapacity}" +
                 (_trainerConfig.DynaModelUpdatesPerStep > 0 ? $", dyna={_trainerConfig.DynaModelUpdatesPerStep}" : string.Empty));
    }

    // ── Current epsilon (decays from EpsilonStart → EpsilonEnd) ──────────────

    private float CurrentEpsilon =>
        _trainerConfig.DqnEpsilonEnd +
        (_trainerConfig.DqnEpsilonStart - _trainerConfig.DqnEpsilonEnd) *
        Math.Max(0f, (_trainerConfig.DqnEpsilonDecaySteps - _totalStepsSeen) /
                     (float)Math.Max(1, _trainerConfig.DqnEpsilonDecaySteps));

    // ── ITrainer ──────────────────────────────────────────────────────────────

    public PolicyDecision SampleAction(float[] observation)
    {
        if (_rng.NextDouble() < CurrentEpsilon)
        {
            // Random exploration.
            return new PolicyDecision { DiscreteAction = _rng.Next(_actionCount) };
        }

        var qValues = _network.GetOnlineQValues(observation);
        return new PolicyDecision { DiscreteAction = ArgMax(qValues) };
    }

    public PolicyDecision[] SampleActions(VectorBatch observations)
    {
        var decisions = new PolicyDecision[observations.BatchSize];
        var epsilon   = CurrentEpsilon;
        for (var b = 0; b < observations.BatchSize; b++)
        {
            var obs = observations.CopyRow(b);
            if (_rng.NextDouble() < epsilon)
                decisions[b] = new PolicyDecision { DiscreteAction = _rng.Next(_actionCount) };
            else
            {
                var qValues = _network.GetOnlineQValues(obs);
                decisions[b] = new PolicyDecision { DiscreteAction = ArgMax(qValues) };
            }
        }
        return decisions;
    }

    /// <summary>DQN has no separate value head; returns the maximum Q-value as a proxy.</summary>
    public float EstimateValue(float[] observation)
    {
        var qValues = _network.GetOnlineQValues(observation);
        return qValues[ArgMax(qValues)];
    }

    public float[] EstimateValues(VectorBatch observations)
    {
        var values = new float[observations.BatchSize];
        for (var b = 0; b < observations.BatchSize; b++)
        {
            var q = _network.GetOnlineQValues(observations.CopyRow(b));
            values[b] = q[ArgMax(q)];
        }
        return values;
    }

    public void RecordTransition(Transition transition)
    {
        if (!IsFiniteTransition(transition)) return;
        _buffer.Add(transition);
        _totalStepsSeen++;
    }

    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_buffer.Count < _trainerConfig.DqnWarmupSteps)
            return null;

        var batch = _buffer.SampleBatch(_trainerConfig.DqnBatchSize, _rng);
        var loss  = _network.TrainBatch(
            batch,
            _trainerConfig.Gamma,
            _trainerConfig.LearningRate,
            _trainerConfig.MaxGradientNorm,
            _trainerConfig.DqnUseDouble);

        // ── Dyna-Q: train world model and run imagined Q-learning updates ─────
        if (_dynaModel is not null && _trainerConfig.DynaModelUpdatesPerStep > 0)
        {
            _dynaModel.TrainBatch(batch, _trainerConfig.DynaModelLearningRate);

            // Generate imagined transitions by querying the model with random past (obs, action) pairs.
            var imagined = new Transition[_trainerConfig.DynaModelUpdatesPerStep];
            for (var i = 0; i < imagined.Length; i++)
            {
                var real  = batch[_rng.Next(batch.Length)];
                var act   = _rng.Next(_actionCount);
                var (nextObs, reward) = _dynaModel.Predict(real.Observation, act);
                // Use a heuristic "done" signal: episode ends if imagined reward is very negative.
                imagined[i] = new Transition
                {
                    Observation     = real.Observation,
                    DiscreteAction  = act,
                    Reward          = reward,
                    NextObservation = nextObs,
                    Done            = false,
                };
            }

            _network.TrainBatch(
                imagined,
                _trainerConfig.Gamma,
                _trainerConfig.LearningRate,
                _trainerConfig.MaxGradientNorm,
                _trainerConfig.DqnUseDouble);
        }

        // Hard-copy online → target every N steps.
        if (_totalStepsSeen >= _nextTargetUpdate)
        {
            _network.HardCopyToTarget();
            _nextTargetUpdate = _totalStepsSeen + _trainerConfig.DqnTargetUpdateInterval;
        }

        return new TrainerUpdateStats
        {
            PolicyLoss   = loss,
            ValueLoss    = 0f,
            Entropy      = CurrentEpsilon,   // report epsilon as a proxy for exploration rate
            ClipFraction = 0f,
            Checkpoint   = CreateCheckpoint(groupId, totalSteps, episodeCount, 0),
        };
    }

    public RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        return CheckpointMetadataBuilder.Apply(
            _network.SaveCheckpoint(groupId, totalSteps, episodeCount, updateCount),
            _config);
    }

    public IInferencePolicy SnapshotPolicyForEval()
    {
        var checkpoint = CreateCheckpoint(_config.GroupId, 0, 0, 0);
        return InferencePolicyFactory.Create(checkpoint);
    }

    public void LoadFromCheckpoint(RLCheckpoint checkpoint) => _network.LoadCheckpoint(checkpoint);

    // ── IAsyncTrainer ─────────────────────────────────────────────────────────

    public bool TryScheduleBackgroundUpdate(string groupId, long totalSteps, long episodeCount, int maxTransitions = int.MaxValue)
    {
        if (_pendingUpdate is not null) return false;
        if (_buffer.Count < _trainerConfig.DqnWarmupSteps) return false;

        var batch = _buffer.SampleBatch(_trainerConfig.DqnBatchSize, _rng);

        // Lazy-create shadow network with the same architecture.
        _shadowNetwork ??= new DqnNetwork(
            _config.ObsSpec?.TotalSize ?? _config.ObservationSize,
            _actionCount,
            _config.NetworkGraph);

        var shadowCheckpoint = _network.SaveCheckpoint(string.Empty, 0, 0, 0);
        _shadowNetwork.LoadCheckpoint(shadowCheckpoint);

        var shadow     = _shadowNetwork;
        var config     = _trainerConfig;
        var stepsSeen  = _totalStepsSeen;
        var nextTarget = _nextTargetUpdate;

        _pendingUpdate = Task.Run(() =>
        {
            var loss = shadow.TrainBatch(batch, config.Gamma, config.LearningRate,
                                         config.MaxGradientNorm, config.DqnUseDouble);
            if (stepsSeen >= nextTarget)
                shadow.HardCopyToTarget();

            return new DqnAsyncResult { MeanLoss = loss };
        });

        return true;
    }

    public TrainerUpdateStats? TryPollResult(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null || !_pendingUpdate.IsCompleted) return null;

        if (_pendingUpdate.IsFaulted)
        {
            var msg = _pendingUpdate.Exception?.GetBaseException().Message ?? "unknown";
            _pendingUpdate = null;
            GD.PushError($"[DQN] Background training task faulted — skipping update. {msg}");
            return null;
        }

        var result = _pendingUpdate.Result;
        _pendingUpdate = null;

        // Apply shadow weights back to live network (must be on main thread).
        var loaded = _shadowNetwork!.SaveCheckpoint(string.Empty, 0, 0, 0);
        _network.LoadCheckpoint(loaded);

        if (_totalStepsSeen >= _nextTargetUpdate)
            _nextTargetUpdate = _totalStepsSeen + _trainerConfig.DqnTargetUpdateInterval;

        return new TrainerUpdateStats
        {
            PolicyLoss   = result.MeanLoss,
            ValueLoss    = 0f,
            Entropy      = CurrentEpsilon,
            ClipFraction = 0f,
            Checkpoint   = CreateCheckpoint(groupId, totalSteps, episodeCount, 0),
        };
    }

    public TrainerUpdateStats? FlushPendingUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null) return null;
        _pendingUpdate.Wait();
        return TryPollResult(groupId, totalSteps, episodeCount);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private static int ArgMax(float[] values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static bool IsFiniteTransition(Transition t)
    {
        if (!float.IsFinite(t.Reward)) return false;
        foreach (var f in t.Observation)     if (!float.IsFinite(f)) return false;
        foreach (var f in t.NextObservation) if (!float.IsFinite(f)) return false;
        return true;
    }
}
