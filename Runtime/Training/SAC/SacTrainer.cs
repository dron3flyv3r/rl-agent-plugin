using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class SacTrainer : ITrainer, IAsyncTrainer, IDistributedTrainer
{
    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _trainerConfig;
    private readonly SacNetwork _network;
    private readonly ReplayBuffer<Transition> _buffer;
    private readonly Random _rng;
    private readonly bool _isContinuous;

    private float _logAlpha;
    private readonly float _targetEntropy;
    private long _totalStepsSeen;
    // Threshold-based cadence: train when _totalStepsSeen >= this value.
    // Avoids the fragile modulo check that can skip multiples when N agents
    // make decisions per frame and N doesn't divide SacUpdateEverySteps.
    private long _nextTrainingStep;

    // ── UTD (Updates-To-Data ratio) ───────────────────────────────────────────
    /// <summary>
    /// Number of active data sources feeding this trainer (master=1 + connected workers).
    /// Set by <see cref="DistributedMaster"/> each tick; defaults to 1 for standalone mode.
    /// Used to compute the auto UTD ratio when <see cref="RLTrainerConfig.SacUpdatesPerStep"/> is 0.
    /// </summary>
    public int DataSources { get; set; } = 1;

    /// <summary>
    /// Number of parallel environments per process (RunConfig.BatchSize).
    /// Set by <see cref="TrainingBootstrap"/> at startup; defaults to 1.
    /// Scales the auto UTD ratio so gradient updates match the full data rate.
    /// </summary>
    public int EnvBatchSize { get; set; } = 1;

    // Auto UTD cap: prevents the background job from becoming too compute-heavy
    // when large BatchSizes or many workers would otherwise push UTD very high.
    private const int AutoUtdCap = 8;

    private int ResolveUtd()
    {
        if (_trainerConfig.SacUpdatesPerStep > 0)
            return _trainerConfig.SacUpdatesPerStep;
        // Auto: one gradient update per new transition across all data sources,
        // capped to avoid excessive compute per physics step.
        return Math.Clamp(DataSources * EnvBatchSize, 1, AutoUtdCap);
    }

    // ── Async gradient update state ───────────────────────────────────────────
    private SacNetwork? _shadowNetwork;
    private Task<SacAsyncResult>? _pendingUpdate;

    // ── Distributed staging buffer ────────────────────────────────────────────
    // Transitions accumulated since last worker send.
    // Capped at 2 × SacBatchSize to bound memory when not consumed by a master.
    private readonly Queue<Transition> _stagingTransitions = new();
    // Distributed workers never train locally, so retaining a full replay buffer there is pure
    // memory waste. The master still keeps its replay buffer and receives streamed rollouts.
    public bool StoreTransitionsInReplayBuffer { get; set; } = true;

    public SacTrainer(PolicyGroupConfig config)
    {
        if (config.DiscreteActionCount > 0)
            throw new InvalidOperationException(
                $"[SAC] Discrete action spaces are not supported by SacTrainer. " +
                $"Convert the action space for group '{config.GroupId}' to continuous-only actions. " +
                $"Use PPO if you need discrete actions.");

        _config = config;
        _trainerConfig = config.TrainerConfig;
        _isContinuous = config.ContinuousActionDimensions > 0 && config.DiscreteActionCount == 0;
        _rng = new Random();

        // Note: SacNetwork does not yet support per-stream CNN encoders.
        // When ObsSpec is provided, we use TotalSize so image pixels are treated as raw floats.
        var sacObsSize = config.ObsSpec?.TotalSize ?? config.ObservationSize;
        _network = new SacNetwork(
            sacObsSize,
            _isContinuous ? config.ContinuousActionDimensions : config.DiscreteActionCount,
            _isContinuous,
            config.NetworkGraph,
            config.TrainerConfig.LearningRate);

        _buffer = new ReplayBuffer<Transition>(config.TrainerConfig.ReplayBufferCapacity);

        _logAlpha = MathF.Log(Math.Max(config.TrainerConfig.SacInitAlpha, 1e-8f));

        // Target entropy:
        //   continuous → action_dims × scale, or exact override when configured
        //   discrete   → fraction × log(|A|)  where fraction ∈ (0,1]
        //     fraction=1.0 = uniform random (max entropy); fraction=0.5 = half of max.
        //     Using log(|A|) directly (fraction=1) keeps the policy fully random forever.
        _targetEntropy = _isContinuous
            ? ResolveContinuousTargetEntropy(config)
            : config.TrainerConfig.SacTargetEntropyFraction * MathF.Log(config.DiscreteActionCount);
    }

    public PolicyDecision SampleAction(float[] observation)
    {
        if (_isContinuous)
        {
            var (action, logProb, _, _) = _network.SampleContinuousAction(observation, _rng);
            return new PolicyDecision
            {
                DiscreteAction = -1,
                ContinuousActions = action,
                LogProbability = logProb,
                Value = 0f,
                Entropy = -logProb,
            };
        }
        else
        {
            var (action, logProb, entropy) = _network.SampleDiscreteAction(observation, _rng);
            return new PolicyDecision
            {
                DiscreteAction = action,
                ContinuousActions = Array.Empty<float>(),
                LogProbability = logProb,
                Value = 0f,
                Entropy = entropy,
            };
        }
    }

    public PolicyDecision[] SampleActions(VectorBatch observations)
    {
        var decisions = new PolicyDecision[observations.BatchSize];
        if (_isContinuous)
        {
            var batch = _network.SampleContinuousActions(observations, _rng);
            for (var batchIndex = 0; batchIndex < observations.BatchSize; batchIndex++)
            {
                decisions[batchIndex] = new PolicyDecision
                {
                    DiscreteAction = -1,
                    ContinuousActions = batch.Actions[batchIndex],
                    LogProbability = batch.LogProbabilities[batchIndex],
                    Value = 0f,
                    Entropy = -batch.LogProbabilities[batchIndex],
                };
            }

            return decisions;
        }

        var discreteBatch = _network.SampleDiscreteActions(observations, _rng);
        for (var batchIndex = 0; batchIndex < observations.BatchSize; batchIndex++)
        {
            decisions[batchIndex] = new PolicyDecision
            {
                DiscreteAction = discreteBatch.Actions[batchIndex],
                ContinuousActions = Array.Empty<float>(),
                LogProbability = discreteBatch.LogProbabilities[batchIndex],
                Value = 0f,
                Entropy = discreteBatch.Entropies[batchIndex],
            };
        }

        return decisions;
    }

    public float EstimateValue(float[] observation) => 0f;

    public float[] EstimateValues(VectorBatch observations) => new float[observations.BatchSize];

    public void RecordTransition(Transition transition)
    {
        // Guard against NaN/infinity from physics explosions corrupting the replay buffer.
        if (!IsFiniteTransition(transition)) return;

        if (StoreTransitionsInReplayBuffer)
        {
            _buffer.Add(transition);
            _totalStepsSeen++;
        }
        _stagingTransitions.Enqueue(transition);
        // Cap staging buffer so it doesn't grow forever in standalone mode
        while (_stagingTransitions.Count > _trainerConfig.SacBatchSize * 2)
            _stagingTransitions.Dequeue();
    }

    private static bool IsFiniteTransition(Transition t)
    {
        if (!float.IsFinite(t.Reward)) return false;
        foreach (var f in t.Observation)    if (!float.IsFinite(f)) return false;
        foreach (var f in t.NextObservation) if (!float.IsFinite(f)) return false;
        return true;
    }

    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_buffer.Count < _trainerConfig.SacWarmupSteps)
            return null;

        if (_totalStepsSeen < _nextTrainingStep)
            return null;
        _nextTrainingStep = _totalStepsSeen + Math.Max(1, _trainerConfig.SacUpdateEverySteps);

        var utd = ResolveUtd();
        var totalPolicyLoss = 0f;
        var totalValueLoss  = 0f;
        var totalEntropy    = 0f;

        for (var u = 0; u < utd; u++)
        {
            var batch = _buffer.SampleBatch(_trainerConfig.SacBatchSize, _rng);
            var alpha = MathF.Exp(_logAlpha);

            var (policyLossSum, valueLossSum, entropySum) = _isContinuous
                ? BatchUpdateContinuous(_network, batch, alpha, _trainerConfig, _rng)
                : BatchUpdateDiscrete(_network, batch, alpha, _trainerConfig);

            if (_trainerConfig.SacAutoTuneAlpha && batch.Length > 0)
            {
                var meanEntropy = entropySum / batch.Length;
                _logAlpha -= _trainerConfig.LearningRate * (meanEntropy - _targetEntropy);
                _logAlpha = Math.Clamp(_logAlpha, -20f, 4f);
            }

            _network.SoftUpdateTargets(_trainerConfig.SacTau);

            var n = Math.Max(1, batch.Length);
            totalPolicyLoss += policyLossSum / n;
            totalValueLoss  += valueLossSum  / n;
            totalEntropy    += entropySum    / n;
        }

        return new TrainerUpdateStats
        {
            PolicyLoss   = totalPolicyLoss / utd,
            ValueLoss    = totalValueLoss  / utd,
            Entropy      = totalEntropy    / utd,
            ClipFraction = 0f,
            SacAlpha     = MathF.Exp(_logAlpha),
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
        if (_pendingUpdate is not null)
            return false;

        if (_buffer.Count < _trainerConfig.SacWarmupSteps)
            return false;

        if (_totalStepsSeen < _nextTrainingStep)
            return false;
        _nextTrainingStep = _totalStepsSeen + Math.Max(1, _trainerConfig.SacUpdateEverySteps);

        // Pre-sample the batch on the main thread (buffer access is main-thread-safe here).
        var batch = _buffer.SampleBatch(_trainerConfig.SacBatchSize, _rng);

        // Lazy-create shadow with identical architecture.
        _shadowNetwork ??= new SacNetwork(
            _config.ObservationSize,
            _isContinuous ? _config.ContinuousActionDimensions : _config.DiscreteActionCount,
            _isContinuous,
            _config.NetworkGraph,
            _trainerConfig.LearningRate);

        // Copy live weights into shadow so the background job works on an isolated copy.
        _network.CopyWeightsTo(_shadowNetwork);

        var shadow = _shadowNetwork;
        var config = _trainerConfig;
        var isContinuous = _isContinuous;
        var logAlpha = _logAlpha;
        var targetEntropy = _targetEntropy;
        var utd = ResolveUtd();
        var rng = new Random(); // dedicated RNG — System.Random is not thread-safe

        _pendingUpdate = Task.Run(() => RunBackgroundUpdate(shadow, batch, config, isContinuous, logAlpha, targetEntropy, utd, rng));
        return true;
    }

    public TrainerUpdateStats? TryPollResult(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null || !_pendingUpdate.IsCompleted)
            return null;

        if (_pendingUpdate.IsFaulted)
        {
            // Clear the stuck task so training can resume on the next eligible step.
            // (Re-throwing would leave _trainingInProgress dirty in the master, blocking forever.)
            var msg = _pendingUpdate.Exception?.GetBaseException().Message ?? "unknown";
            _pendingUpdate = null;
            Godot.GD.PushError($"[SAC] Background training task faulted — skipping update. {msg}");
            return null;
        }

        var result = _pendingUpdate.Result;
        _pendingUpdate = null;

        // Apply trained shadow weights back to the live network (main thread only).
        _network.LoadWeightsFrom(_shadowNetwork!);
        _logAlpha = result.NewLogAlpha;

        return new TrainerUpdateStats
        {
            PolicyLoss = result.PolicyLoss,
            ValueLoss = result.ValueLoss,
            Entropy = result.Entropy,
            ClipFraction = 0f,
            SacAlpha = MathF.Exp(_logAlpha),
            Checkpoint = CreateCheckpoint(groupId, totalSteps, episodeCount, 0),
        };
    }

    public TrainerUpdateStats? FlushPendingUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null)
            return null;
        _pendingUpdate.Wait();
        return TryPollResult(groupId, totalSteps, episodeCount);
    }

    // ── Background job ────────────────────────────────────────────────────────

    private static SacAsyncResult RunBackgroundUpdate(
        SacNetwork network,
        Transition[] batch,
        RLTrainerConfig config,
        bool isContinuous,
        float logAlpha,
        float targetEntropy,
        int utd,
        Random rng)
    {
        var totalPolicyLoss = 0f;
        var totalValueLoss  = 0f;
        var totalEntropy    = 0f;

        for (var u = 0; u < utd; u++)
        {
            // First UTD step uses the pre-sampled batch; subsequent steps re-use the same
            // batch (background thread cannot safely access the replay buffer). This gives
            // meaningful multi-step gradient updates without requiring thread-safe sampling.
            var alpha = MathF.Exp(logAlpha);

            var (policyLossSum, valueLossSum, entropySum) = isContinuous
                ? BatchUpdateContinuous(network, batch, alpha, config, rng)
                : BatchUpdateDiscrete(network, batch, alpha, config);

            if (config.SacAutoTuneAlpha && batch.Length > 0)
            {
                var meanEntropy = entropySum / batch.Length;
                logAlpha -= config.LearningRate * (meanEntropy - targetEntropy);
                logAlpha = Math.Clamp(logAlpha, -20f, 4f);
            }

            network.SoftUpdateTargets(config.SacTau);

            var n = Math.Max(1, batch.Length);
            totalPolicyLoss += policyLossSum / n;
            totalValueLoss  += valueLossSum  / n;
            totalEntropy    += entropySum    / n;
        }

        return new SacAsyncResult
        {
            PolicyLoss  = totalPolicyLoss / utd,
            ValueLoss   = totalValueLoss  / utd,
            Entropy     = totalEntropy    / utd,
            NewLogAlpha = logAlpha,
        };
    }

    private static float ResolveContinuousTargetEntropy(PolicyGroupConfig config)
    {
        var trainer = config.TrainerConfig;
        if (trainer.SacUseContinuousTargetEntropyOverride)
            return Math.Max(0f, trainer.SacContinuousTargetEntropyOverride);

        return Math.Max(0f, config.ContinuousActionDimensions * trainer.SacContinuousTargetEntropyScale);
    }

    // ── Discrete SAC batch update ─────────────────────────────────────────────

    /// <summary>
    /// Computes all Bellman targets for the batch (read-only pass), then performs a single
    /// Adam step on Q1, Q2, and the actor — one weight update per network per batch call.
    /// </summary>
    private static (float policyLossSum, float valueLossSum, float entropySum) BatchUpdateDiscrete(
        SacNetwork network, Transition[] batch, float alpha, RLTrainerConfig config)
    {
        var qSamples  = new List<(float[] obs, int action, float target)>(batch.Length);
        var actorObs  = new List<float[]>(batch.Length);
        var valueLoss = 0f;
        var entropy   = 0f;

        // Phase 1 — Bellman targets (no weight updates)
        foreach (var t in batch)
        {
            var (nextQ1, nextQ2)          = network.GetDiscreteTargetQValues(t.NextObservation);
            var (nextProbs, nextLogProbs) = network.GetDiscreteActorProbs(t.NextObservation);

            var nextV = 0f;
            for (var a = 0; a < nextQ1.Length; a++)
                nextV += nextProbs[a] * (Math.Min(nextQ1[a], nextQ2[a]) - alpha * nextLogProbs[a]);

            var y = t.Reward + config.Gamma * (t.Done ? 0f : 1f) * nextV;

            var (q1, q2) = network.GetDiscreteQValues(t.Observation);
            valueLoss += MathF.Abs(q1[t.DiscreteAction] - y) + MathF.Abs(q2[t.DiscreteAction] - y);

            var (probs, logProbs) = network.GetDiscreteActorProbs(t.Observation);
            for (var a = 0; a < probs.Length; a++) entropy -= probs[a] * logProbs[a];

            qSamples.Add((t.Observation, t.DiscreteAction, y));
            actorObs.Add(t.Observation);
        }

        // Phase 2 — single gradient step per network
        network.BatchUpdateQ1Discrete(qSamples, config.LearningRate);
        network.BatchUpdateQ2Discrete(qSamples, config.LearningRate);
        network.BatchUpdateActorDiscrete(actorObs, alpha, config.LearningRate);

        return (policyLossSum: -entropy, valueLossSum: valueLoss, entropySum: entropy);
    }

    // ── Continuous SAC batch update ───────────────────────────────────────────

    /// <summary>
    /// Computes all Bellman targets and samples fresh actor actions for the batch, then
    /// performs a single Adam step on Q1, Q2, and the actor.
    /// </summary>
    private static (float policyLossSum, float valueLossSum, float entropySum) BatchUpdateContinuous(
        SacNetwork network, Transition[] batch, float alpha, RLTrainerConfig config, Random rng)
    {
        var qSamples     = new List<(float[] obs, float[] action, float target)>(batch.Length);
        var actorSamples = new List<(float[] obs, float[] action, float[] eps, float[] u)>(batch.Length);
        var valueLoss    = 0f;
        var entropy      = 0f;

        // Phase 1 — Bellman targets + fresh action samples (no weight updates)
        foreach (var t in batch)
        {
            var (nextAction, nextLogProb, _, _) = network.SampleContinuousAction(t.NextObservation, rng);
            var (nextQ1t, nextQ2t)              = network.GetContinuousTargetQValues(t.NextObservation, nextAction);
            var y = t.Reward + config.Gamma * (t.Done ? 0f : 1f) *
                    (Math.Min(nextQ1t, nextQ2t) - alpha * nextLogProb);

            var (q1, q2) = network.GetContinuousQValues(t.Observation, t.ContinuousActions);
            valueLoss   += MathF.Abs(q1 - y) + MathF.Abs(q2 - y);

            var (action, logProb, eps, u) = network.SampleContinuousAction(t.Observation, rng);
            entropy += -logProb;

            qSamples.Add((t.Observation, t.ContinuousActions, y));
            actorSamples.Add((t.Observation, action, eps, u));
        }

        // Phase 2 — single gradient step per network
        network.BatchUpdateQ1Continuous(qSamples, config.LearningRate);
        network.BatchUpdateQ2Continuous(qSamples, config.LearningRate);
        network.BatchUpdateActorContinuous(actorSamples, alpha, config.LearningRate);

        return (policyLossSum: entropy, valueLossSum: valueLoss, entropySum: entropy);
    }

    // ── IDistributedTrainer ───────────────────────────────────────────────────

    public bool IsOffPolicy => true;

    public bool IsRolloutReady => _stagingTransitions.Count >= _trainerConfig.SacBatchSize;

    public byte[] ExportAndClearRollout()
    {
        var batch = new List<DistributedTransition>(_stagingTransitions.Count);
        while (_stagingTransitions.TryDequeue(out var t))
        {
            batch.Add(new DistributedTransition
            {
                Observation       = t.Observation,
                DiscreteAction    = t.DiscreteAction,
                ContinuousActions = t.ContinuousActions,
                Reward            = t.Reward,
                Done              = t.Done,
                NextObservation   = t.NextObservation,
                OldLogProbability = 0f,
                Value             = 0f,
                NextValue         = 0f,
            });
        }
        return DistributedProtocol.SerializeRollout(batch);
    }

    public void InjectRollout(byte[] data)
    {
        foreach (var t in DistributedProtocol.DeserializeRollout(data))
        {
            _buffer.Add(new Transition
            {
                Observation       = t.Observation,
                DiscreteAction    = t.DiscreteAction,
                ContinuousActions = t.ContinuousActions,
                Reward            = t.Reward,
                Done              = t.Done,
                NextObservation   = t.NextObservation,
            });
            // Count injected worker transitions toward training cadence so the
            // _nextTrainingStep threshold advances correctly in distributed mode.
            // Without this, _totalStepsSeen stays at 0 on a master with no local
            // agents and TryUpdate permanently blocks after the first update.
            _totalStepsSeen++;
        }
    }

    public byte[] ExportWeights()
    {
        var cp = _network.SaveCheckpoint("_dist_", 0, 0, 0);
        return DistributedProtocol.SerializeWeights(cp.WeightBuffer, cp.LayerShapeBuffer);
    }

    public void ImportWeights(byte[] data)
    {
        var (weights, shapes) = DistributedProtocol.DeserializeWeights(data);
        _network.LoadCheckpoint(new RLCheckpoint { WeightBuffer = weights, LayerShapeBuffer = shapes });
    }

    // ── Private types ─────────────────────────────────────────────────────────

    private sealed class SacAsyncResult
    {
        public float PolicyLoss { get; init; }
        public float ValueLoss { get; init; }
        public float Entropy { get; init; }
        public float NewLogAlpha { get; init; }
    }
}
