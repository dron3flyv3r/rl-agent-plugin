using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class PpoTrainer : ITrainer, IAsyncTrainer, IDistributedTrainer
{
    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _trainerConfig;
    private readonly PolicyValueNetwork _network;
    private readonly List<PpoTransition> _transitions = new();
    private readonly RandomNumberGenerator _rng = new();

    // ── Async gradient update state ───────────────────────────────────────────
    private PolicyValueNetwork? _shadowNetwork;
    private Task<PpoAsyncResult>? _pendingUpdate;

    public PpoTrainer(PolicyGroupConfig config)
    {
        _config = config;
        _trainerConfig = config.TrainerConfig;
        _network = new PolicyValueNetwork(
            config.ObservationSize,
            config.DiscreteActionCount,
            config.ContinuousActionDimensions,
            config.NetworkGraph);
        _rng.Randomize();
    }

    public int TransitionCount => _transitions.Count;

    public PolicyDecision SampleAction(float[] observation)
    {
        var inference = _network.Infer(observation);

        if (_config.ContinuousActionDimensions > 0)
            return SampleContinuousDecision(inference.Logits, inference.Value);

        var probabilities = Softmax(inference.Logits);
        var sampledAction = SampleFromProbabilities(probabilities);
        var logProbability = Mathf.Log(Math.Max(1e-6f, probabilities[sampledAction]));
        return new PolicyDecision
        {
            DiscreteAction = sampledAction,
            Value = inference.Value,
            LogProbability = logProbability,
            Entropy = CalculateEntropy(probabilities),
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

        for (var batchIndex = 0; batchIndex < observations.BatchSize; batchIndex++)
        {
            var probabilities = Softmax(inference.Logits.CopyRow(batchIndex));
            var sampledAction = SampleFromProbabilities(probabilities);
            var logProbability = Mathf.Log(Math.Max(1e-6f, probabilities[sampledAction]));
            decisions[batchIndex] = new PolicyDecision
            {
                DiscreteAction = sampledAction,
                Value = inference.Values[batchIndex],
                LogProbability = logProbability,
                Entropy = CalculateEntropy(probabilities),
            };
        }

        return decisions;
    }

    private PolicyDecision SampleContinuousDecision(float[] actorOut, float value)
    {
        var D = _config.ContinuousActionDimensions;
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
            DiscreteAction = -1,
            ContinuousActions = action,
            Value = value,
            LogProbability = logProb,
            Entropy = entropy,
        };
    }

    public float EstimateValue(float[] observation)
    {
        return _network.Infer(observation).Value;
    }

    public float[] EstimateValues(VectorBatch observations)
    {
        return _network.InferBatch(observations).Values;
    }

    public void RecordTransition(Transition t)
    {
        _transitions.Add(new PpoTransition
        {
            Observation = t.Observation.ToArray(),
            Action = t.DiscreteAction,
            ContinuousActions = t.ContinuousActions.Length > 0
                ? (float[])t.ContinuousActions.Clone()
                : Array.Empty<float>(),
            Reward = t.Reward,
            Done = t.Done,
            OldLogProbability = t.OldLogProbability,
            Value = t.Value,
            NextValue = t.NextValue,
        });
    }

    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_transitions.Count < _trainerConfig.RolloutLength)
        {
            return null;
        }

        var samples = BuildTrainingSamples();
        NormalizeAdvantages(samples);
        var miniBatchSize = Math.Clamp(_trainerConfig.PpoMiniBatchSize, 1, samples.Count);
        var policyLoss = 0.0f;
        var valueLoss = 0.0f;
        var entropy = 0.0f;
        var clipFraction = 0.0f;
        var processedSamples = 0;

        for (var epoch = 0; epoch < _trainerConfig.EpochsPerUpdate; epoch++)
        {
            Shuffle(samples);
            for (var start = 0; start < samples.Count; start += miniBatchSize)
            {
                var count = Math.Min(miniBatchSize, samples.Count - start);
                var batchSamples = new TrainingSample[count];
                for (var index = 0; index < count; index++)
                {
                    batchSamples[index] = samples[start + index];
                }

                var updateStats = _network.ApplyGradients(batchSamples, _trainerConfig);
                policyLoss += updateStats.PolicyLoss * count;
                valueLoss += updateStats.ValueLoss * count;
                entropy += updateStats.Entropy * count;
                clipFraction += updateStats.ClipFraction * count;
                processedSamples += count;
            }
        }

        _transitions.Clear();
        var normalizer = Math.Max(1, processedSamples);

        return new TrainerUpdateStats
        {
            PolicyLoss = policyLoss / normalizer,
            ValueLoss = valueLoss / normalizer,
            Entropy = entropy / normalizer,
            ClipFraction = clipFraction / normalizer,
            Checkpoint = CreateCheckpoint(groupId, totalSteps, episodeCount, 0),
        };
    }

    public RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        return CheckpointMetadataBuilder.Apply(
            _network.SaveCheckpoint(groupId, totalSteps, episodeCount, updateCount),
            _config);
    }

    // ── IAsyncTrainer ─────────────────────────────────────────────────────────

    public bool TryScheduleBackgroundUpdate(string groupId, long totalSteps, long episodeCount, int maxTransitions = int.MaxValue)
    {
        // Reject if a job is running or a completed result hasn't been polled yet.
        if (_pendingUpdate is not null)
            return false;

        if (_transitions.Count < _trainerConfig.RolloutLength)
            return false;

        // Snapshot transitions (capped) and clear the live buffer so the main thread can keep filling it.
        var count = Math.Min(_transitions.Count, maxTransitions);
        var transitions = _transitions.GetRange(0, count);
        _transitions.Clear();

        // Lazy-create shadow network with identical architecture.
        _shadowNetwork ??= new PolicyValueNetwork(
            _config.ObservationSize,
            _config.DiscreteActionCount,
            _config.ContinuousActionDimensions,
            _config.NetworkGraph);

        // Copy live weights into shadow so backprop works on an isolated copy.
        _network.CopyWeightsTo(_shadowNetwork);

        var shadow = _shadowNetwork;
        var config = _trainerConfig;
        _pendingUpdate = Task.Run(() => RunBackgroundUpdate(shadow, transitions, config));
        return true;
    }

    public TrainerUpdateStats? TryPollResult(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null || !_pendingUpdate.IsCompleted)
            return null;

        var result = _pendingUpdate.Result;
        _pendingUpdate = null;

        // Apply the trained shadow weights back to the live network (main thread only).
        _network.LoadWeightsFrom(_shadowNetwork!);

        return new TrainerUpdateStats
        {
            PolicyLoss = result.PolicyLoss,
            ValueLoss = result.ValueLoss,
            Entropy = result.Entropy,
            ClipFraction = result.ClipFraction,
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

    private static PpoAsyncResult RunBackgroundUpdate(
        PolicyValueNetwork network,
        List<PpoTransition> transitions,
        RLTrainerConfig config)
    {
        var samples = BuildTrainingSamplesFrom(transitions, config);
        NormalizeAdvantages(samples);
        var miniBatchSize = Math.Clamp(config.PpoMiniBatchSize, 1, samples.Count);
        var policyLoss = 0f;
        var valueLoss = 0f;
        var entropy = 0f;
        var clipFraction = 0f;
        var processedSamples = 0;
        var rng = new Random();

        for (var epoch = 0; epoch < config.EpochsPerUpdate; epoch++)
        {
            ShuffleWithRandom(samples, rng);
            for (var start = 0; start < samples.Count; start += miniBatchSize)
            {
                var count = Math.Min(miniBatchSize, samples.Count - start);
                var batchSamples = new TrainingSample[count];
                for (var i = 0; i < count; i++)
                    batchSamples[i] = samples[start + i];

                var stats = network.ApplyGradients(batchSamples, config);
                policyLoss += stats.PolicyLoss * count;
                valueLoss += stats.ValueLoss * count;
                entropy += stats.Entropy * count;
                clipFraction += stats.ClipFraction * count;
                processedSamples += count;
            }
        }

        var norm = Math.Max(1, processedSamples);
        return new PpoAsyncResult
        {
            PolicyLoss = policyLoss / norm,
            ValueLoss = valueLoss / norm,
            Entropy = entropy / norm,
            ClipFraction = clipFraction / norm,
        };
    }

    // ── Training sample construction ──────────────────────────────────────────

    private List<TrainingSample> BuildTrainingSamples() =>
        BuildTrainingSamplesFrom(_transitions, _trainerConfig);

    private static List<TrainingSample> BuildTrainingSamplesFrom(
        IReadOnlyList<PpoTransition> transitions, RLTrainerConfig config)
    {
        var samples = new List<TrainingSample>(transitions.Count);
        var advantages = new float[transitions.Count];
        var returns = new float[transitions.Count];
        var nextAdvantage = 0.0f;

        for (var index = transitions.Count - 1; index >= 0; index--)
        {
            var transition = transitions[index];
            var mask = transition.Done ? 0.0f : 1.0f;
            var delta = transition.Reward + (config.Gamma * transition.NextValue * mask) - transition.Value;
            nextAdvantage = delta + (config.Gamma * config.GaeLambda * mask * nextAdvantage);
            advantages[index] = nextAdvantage;
            returns[index] = transition.Value + advantages[index];
        }

        for (var index = 0; index < transitions.Count; index++)
        {
            var transition = transitions[index];
            samples.Add(new TrainingSample
            {
                Observation = transition.Observation,
                Action = transition.Action,
                ContinuousActions = transition.ContinuousActions,
                Return = returns[index],
                Advantage = advantages[index],
                OldLogProbability = transition.OldLogProbability,
                ValueEstimate = transition.Value,
            });
        }

        return samples;
    }

    private static void NormalizeAdvantages(IList<TrainingSample> samples)
    {
        var mean = samples.Average(sample => sample.Advantage);
        var variance = samples.Average(sample =>
        {
            var diff = sample.Advantage - mean;
            return diff * diff;
        });

        var stdDev = Mathf.Sqrt((float)variance + 1e-8f);
        foreach (var sample in samples)
        {
            sample.Advantage = (sample.Advantage - (float)mean) / stdDev;
        }
    }

    private float SampleNormal()
    {
        // Box-Muller transform
        var u1 = Math.Max(_rng.Randf(), 1e-10f);
        var u2 = _rng.Randf();
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.Pi * u2);
    }

    private int SampleFromProbabilities(float[] probabilities)
    {
        var roll = _rng.Randf();
        var cumulative = 0.0f;
        for (var index = 0; index < probabilities.Length; index++)
        {
            cumulative += probabilities[index];
            if (roll <= cumulative)
            {
                return index;
            }
        }

        return probabilities.Length - 1;
    }

    private void Shuffle(IList<TrainingSample> samples)
    {
        for (var index = samples.Count - 1; index > 0; index--)
        {
            var swapIndex = _rng.RandiRange(0, index);
            (samples[index], samples[swapIndex]) = (samples[swapIndex], samples[index]);
        }
    }

    private static void ShuffleWithRandom(IList<TrainingSample> samples, Random rng)
    {
        for (var index = samples.Count - 1; index > 0; index--)
        {
            var swapIndex = rng.Next(0, index + 1);
            (samples[index], samples[swapIndex]) = (samples[swapIndex], samples[index]);
        }
    }

    private static float[] Softmax(IReadOnlyList<float> logits)
    {
        var maxLogit = logits.Max();
        var probabilities = new float[logits.Count];
        var total = 0.0f;
        for (var index = 0; index < logits.Count; index++)
        {
            probabilities[index] = Mathf.Exp(logits[index] - maxLogit);
            total += probabilities[index];
        }

        for (var index = 0; index < probabilities.Length; index++)
        {
            probabilities[index] /= total;
        }

        return probabilities;
    }

    private static float CalculateEntropy(IReadOnlyList<float> probabilities)
    {
        var entropy = 0.0f;
        foreach (var probability in probabilities)
        {
            if (probability <= 1e-6f)
            {
                continue;
            }

            entropy -= probability * Mathf.Log(probability);
        }

        return entropy;
    }

    // ── IDistributedTrainer ───────────────────────────────────────────────────

    public bool IsOffPolicy => false;

    public bool IsRolloutReady => _transitions.Count >= _trainerConfig.RolloutLength;

    public byte[] ExportAndClearRollout()
    {
        var snapshot = _transitions
            .Select(t => new DistributedTransition
            {
                Observation       = t.Observation,
                DiscreteAction    = t.Action,
                ContinuousActions = t.ContinuousActions,
                Reward            = t.Reward,
                Done              = t.Done,
                NextObservation   = Array.Empty<float>(),
                OldLogProbability = t.OldLogProbability,
                Value             = t.Value,
                NextValue         = t.NextValue,
            })
            .ToList();
        _transitions.Clear();
        return DistributedProtocol.SerializeRollout(snapshot);
    }

    public void InjectRollout(byte[] data)
    {
        foreach (var t in DistributedProtocol.DeserializeRollout(data))
        {
            _transitions.Add(new PpoTransition
            {
                Observation       = t.Observation,
                Action            = t.DiscreteAction,
                ContinuousActions = t.ContinuousActions,
                Reward            = t.Reward,
                Done              = t.Done,
                OldLogProbability = t.OldLogProbability,
                Value             = t.Value,
                NextValue         = t.NextValue,
            });
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
        _network.LoadCheckpoint(new RLCheckpoint
        {
            WeightBuffer     = weights,
            LayerShapeBuffer = shapes,
        });
    }

    // ── Internal types ────────────────────────────────────────────────────────

    private sealed class PpoAsyncResult
    {
        public float PolicyLoss { get; init; }
        public float ValueLoss { get; init; }
        public float Entropy { get; init; }
        public float ClipFraction { get; init; }
    }

    private sealed class PpoTransition
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
}

public sealed class TrainingSample
{
    public float[] Observation { get; init; } = Array.Empty<float>();
    public int Action { get; init; }
    public float[] ContinuousActions { get; init; } = Array.Empty<float>();
    public float Return { get; init; }
    public float Advantage { get; set; }
    public float OldLogProbability { get; init; }
    public float ValueEstimate { get; init; }
}
