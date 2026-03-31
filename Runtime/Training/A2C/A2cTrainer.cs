using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// A2C (Advantage Actor-Critic) trainer. On-policy, single-pass update without PPO clipping.
/// <para>
/// <b>Action space:</b> Discrete or continuous (same as PPO, not both simultaneously).
/// </para>
/// <para>
/// <b>Policy:</b> On-policy rollout. Advantages estimated using GAE (lambda, gamma).
/// A single gradient step over the entire rollout — no epochs, no minibatches.
/// </para>
/// <para>
/// Internally reuses <see cref="PolicyValueNetwork"/> with <c>ClipEpsilon</c> set to a very large
/// value so the clipped surrogate never activates, giving vanilla actor-critic gradients.
/// </para>
/// </summary>
public sealed class A2cTrainer : ITrainer, IAsyncTrainer
{
    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _trainerConfig;
    private readonly PolicyValueNetwork _network;
    private readonly List<A2cTransition> _transitions = new();
    private readonly RandomNumberGenerator _rng = new();

    // ── Async gradient update state ───────────────────────────────────────────
    private PolicyValueNetwork? _shadowNetwork;
    private Task<A2cAsyncResult>? _pendingUpdate;

    private sealed class A2cAsyncResult
    {
        public float PolicyLoss { get; init; }
        public float ValueLoss { get; init; }
        public float Entropy { get; init; }
        public RLCheckpoint? Checkpoint { get; init; }
    }

    private sealed class A2cTransition
    {
        public float[] Observation { get; init; } = Array.Empty<float>();
        public int Action { get; init; }
        public float[] ContinuousActions { get; init; } = Array.Empty<float>();
        public float Reward { get; init; }
        public bool Done { get; init; }
        public float OldLogProbability { get; init; }
        public float Value { get; init; }
        public float NextValue { get; init; }
    }

    public A2cTrainer(PolicyGroupConfig config)
    {
        if (config.DiscreteActionCount > 0 && config.ContinuousActionDimensions > 0)
            throw new InvalidOperationException(
                $"[A2C] Mixed discrete+continuous action spaces are not supported. " +
                $"Group '{config.GroupId}' has both. Use a separate policy group for each.");

        if (config.DiscreteActionCount <= 0 && config.ContinuousActionDimensions <= 0)
            throw new InvalidOperationException(
                $"[A2C] Group '{config.GroupId}' has no actions defined.");

        _config        = config;
        _trainerConfig = config.TrainerConfig;
        _rng.Randomize();

        _network = config.ObsSpec is not null
            ? new PolicyValueNetwork(
                config.ObsSpec,
                config.DiscreteActionCount,
                config.ContinuousActionDimensions,
                config.NetworkGraph)
            : new PolicyValueNetwork(
                config.ObservationSize,
                config.DiscreteActionCount,
                config.ContinuousActionDimensions,
                config.NetworkGraph);
    }

    // ── ITrainer ──────────────────────────────────────────────────────────────

    public PolicyDecision SampleAction(float[] observation)
    {
        var inference = _network.Infer(observation);

        if (_config.ContinuousActionDimensions > 0)
            return SampleContinuousDecision(inference.Logits, inference.Value);

        var probs      = Softmax(inference.Logits);
        var action     = SampleFromProbs(probs);
        var logProb    = Mathf.Log(Math.Max(1e-6f, probs[action]));
        return new PolicyDecision
        {
            DiscreteAction  = action,
            Value           = inference.Value,
            LogProbability  = logProb,
            Entropy         = CalcEntropy(probs),
        };
    }

    public PolicyDecision[] SampleActions(VectorBatch observations)
    {
        var inference = _network.InferBatch(observations);
        var decisions = new PolicyDecision[observations.BatchSize];

        if (_config.ContinuousActionDimensions > 0)
        {
            for (var b = 0; b < observations.BatchSize; b++)
                decisions[b] = SampleContinuousDecision(inference.Logits.CopyRow(b), inference.Values[b]);
            return decisions;
        }

        for (var b = 0; b < observations.BatchSize; b++)
        {
            var probs   = Softmax(inference.Logits.CopyRow(b));
            var action  = SampleFromProbs(probs);
            var logProb = Mathf.Log(Math.Max(1e-6f, probs[action]));
            decisions[b] = new PolicyDecision
            {
                DiscreteAction  = action,
                Value           = inference.Values[b],
                LogProbability  = logProb,
                Entropy         = CalcEntropy(probs),
            };
        }
        return decisions;
    }

    public float EstimateValue(float[] observation) => _network.Infer(observation).Value;

    public float[] EstimateValues(VectorBatch observations) => _network.InferBatch(observations).Values;

    public void RecordTransition(Transition t)
    {
        _transitions.Add(new A2cTransition
        {
            Observation       = t.Observation.ToArray(),
            Action            = t.DiscreteAction,
            ContinuousActions = t.ContinuousActions.Length > 0
                ? (float[])t.ContinuousActions.Clone()
                : Array.Empty<float>(),
            Reward            = t.Reward,
            Done              = t.Done,
            OldLogProbability = t.OldLogProbability,
            Value             = t.Value,
            NextValue         = t.NextValue,
        });
    }

    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_transitions.Count < _trainerConfig.RolloutLength)
            return null;

        var samples = BuildSamples(_transitions, _trainerConfig);
        NormalizeAdvantages(samples);

        // Single-pass over the full rollout (EpochsPerUpdate=1, MiniBatchSize=rollout.Count).
        var config = BuildA2CUpdateConfig(samples.Count);
        var result = _network.ApplyGradients(samples, config);

        _transitions.Clear();

        var statsCheckpoint = CheckpointMetadataBuilder.Apply(
            _network.SaveCheckpoint(groupId, totalSteps, episodeCount, 0), _config);

        return new TrainerUpdateStats
        {
            PolicyLoss   = result.PolicyLoss,
            ValueLoss    = result.ValueLoss,
            Entropy      = result.Entropy,
            ClipFraction = 0f,
            Checkpoint   = statsCheckpoint,
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
        if (_transitions.Count < _trainerConfig.RolloutLength) return false;

        var transitions = new List<A2cTransition>(_transitions);
        _transitions.Clear();

        _shadowNetwork ??= _config.ObsSpec is not null
            ? new PolicyValueNetwork(_config.ObsSpec, _config.DiscreteActionCount,
                                     _config.ContinuousActionDimensions, _config.NetworkGraph)
            : new PolicyValueNetwork(_config.ObservationSize, _config.DiscreteActionCount,
                                     _config.ContinuousActionDimensions, _config.NetworkGraph);

        var shadowCheckpoint = _network.SaveCheckpoint(string.Empty, 0, 0, 0);
        _shadowNetwork.LoadCheckpoint(shadowCheckpoint);

        var shadow = _shadowNetwork;
        var config = _trainerConfig;

        _pendingUpdate = Task.Run(() =>
        {
            var samples = BuildSamples(transitions, config);
            NormalizeAdvantages(samples);
            var updateConfig = BuildA2CUpdateConfig(samples.Count);
            var stats = shadow.ApplyGradients(samples, updateConfig);
            var cp = shadow.SaveCheckpoint("_a2c_trained_", 0, 0, 0);
            return new A2cAsyncResult
            {
                PolicyLoss = stats.PolicyLoss,
                ValueLoss  = stats.ValueLoss,
                Entropy    = stats.Entropy,
                Checkpoint = cp,
            };
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
            GD.PushError($"[A2C] Background training task faulted — skipping update. {msg}");
            return null;
        }

        var result = _pendingUpdate.Result;
        _pendingUpdate = null;

        if (result.Checkpoint is not null)
            _network.LoadCheckpoint(result.Checkpoint);

        return new TrainerUpdateStats
        {
            PolicyLoss   = result.PolicyLoss,
            ValueLoss    = result.ValueLoss,
            Entropy      = result.Entropy,
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

    /// <summary>
    /// Builds a <see cref="RLTrainerConfig"/> with A2C semantics:
    /// one epoch, full-batch minibatch, and infinite clip epsilon (no clipping).
    /// </summary>
    private RLTrainerConfig BuildA2CUpdateConfig(int sampleCount) => new()
    {
        Algorithm            = RLAlgorithmKind.A2C,
        LearningRate         = _trainerConfig.LearningRate,
        Gamma                = _trainerConfig.Gamma,
        GaeLambda            = _trainerConfig.GaeLambda,
        ClipEpsilon          = float.MaxValue / 2f,   // effectively disabled
        EpochsPerUpdate      = 1,
        PpoMiniBatchSize     = sampleCount,            // single batch = whole rollout
        MaxGradientNorm      = _trainerConfig.MaxGradientNorm,
        ValueLossCoefficient = _trainerConfig.ValueLossCoefficient,
        UseValueClipping     = false,
        ValueClipEpsilon     = 0f,
        EntropyCoefficient   = _trainerConfig.EntropyCoefficient,
    };

    private static List<TrainingSample> BuildSamples(
        IReadOnlyList<A2cTransition> transitions,
        RLTrainerConfig config)
    {
        var count      = transitions.Count;
        var advantages = new float[count];
        var returns    = new float[count];
        var nextAdv    = 0f;

        for (var i = count - 1; i >= 0; i--)
        {
            var t    = transitions[i];
            var mask = t.Done ? 0f : 1f;
            var delta = t.Reward + config.Gamma * t.NextValue * mask - t.Value;
            nextAdv        = delta + config.Gamma * config.GaeLambda * mask * nextAdv;
            advantages[i]  = nextAdv;
            returns[i]     = t.Value + advantages[i];
        }

        var samples = new List<TrainingSample>(count);
        for (var i = 0; i < count; i++)
        {
            var t = transitions[i];
            samples.Add(new TrainingSample
            {
                Observation       = t.Observation,
                Action            = t.Action,
                ContinuousActions = t.ContinuousActions,
                Return            = returns[i],
                Advantage         = advantages[i],
                OldLogProbability = t.OldLogProbability,
                ValueEstimate     = t.Value,
            });
        }
        return samples;
    }

    private static void NormalizeAdvantages(IList<TrainingSample> samples)
    {
        var mean     = samples.Average(s => s.Advantage);
        var variance = samples.Average(s => { var d = s.Advantage - mean; return d * d; });
        var std      = Mathf.Sqrt((float)variance + 1e-8f);
        foreach (var s in samples) s.Advantage = (s.Advantage - (float)mean) / std;
    }

    private PolicyDecision SampleContinuousDecision(float[] actorOut, float value)
    {
        var D       = _config.ContinuousActionDimensions;
        var action  = new float[D];
        var logProb = 0f;
        var entropy = 0f;

        for (var d = 0; d < D; d++)
        {
            var mean   = actorOut[d];
            var logStd = Math.Clamp(actorOut[D + d], -20f, 2f);
            var std    = MathF.Exp(logStd);
            var eps    = SampleNormal();
            var u      = mean + std * eps;
            var a      = MathF.Tanh(u);
            action[d]  = a;
            logProb   += -0.5f * eps * eps - logStd
                         - 0.5f * MathF.Log(2f * MathF.PI)
                         - MathF.Log(1f - a * a + 1e-6f);
            entropy   += 0.5f * (1f + MathF.Log(2f * MathF.PI)) + logStd;
        }

        return new PolicyDecision
        {
            DiscreteAction  = -1,
            ContinuousActions = action,
            Value           = value,
            LogProbability  = logProb,
            Entropy         = entropy,
        };
    }

    private float SampleNormal()
    {
        var u1 = Math.Max(_rng.Randf(), 1e-10f);
        var u2 = _rng.Randf();
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.Pi * u2);
    }

    private int SampleFromProbs(float[] probs)
    {
        var roll = _rng.Randf();
        var cum  = 0f;
        for (var i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (roll <= cum) return i;
        }
        return probs.Length - 1;
    }

    private static float[] Softmax(float[] logits)
    {
        var max   = logits.Max();
        var probs = new float[logits.Length];
        var total = 0f;
        for (var i = 0; i < logits.Length; i++) { probs[i] = Mathf.Exp(logits[i] - max); total += probs[i]; }
        for (var i = 0; i < probs.Length; i++)  probs[i] /= total;
        return probs;
    }

    private static float CalcEntropy(float[] probs)
    {
        var h = 0f;
        foreach (var p in probs) if (p > 1e-6f) h -= p * Mathf.Log(p);
        return h;
    }
}
