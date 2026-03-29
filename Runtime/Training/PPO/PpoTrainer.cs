using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class PpoTrainer : ITrainer, IAsyncTrainer, IDistributedTrainer
{
    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _trainerConfig;
    private readonly PolicyValueNetwork _network;
    private readonly bool _useGpuTrainingNetwork;
    private readonly List<PpoTransition> _transitions = new();
    private readonly RandomNumberGenerator _rng = new();

    // ── Async gradient update state ───────────────────────────────────────────
    private PolicyValueNetwork? _shadowNetwork;
    private Task<PpoAsyncResult>? _pendingUpdate;

    // Dedicated GPU training thread — created once, reuses a single GpuCnnEncoder
    // for the lifetime of the trainer.  GpuDevice is thread-bound, so we cannot
    // share it across thread-pool threads; a dedicated thread sidesteps that.
    private BlockingCollection<GpuTrainingJob>? _gpuJobQueue;
    private Thread?                             _gpuThread;

    private sealed class GpuTrainingJob
    {
        public required List<TrainingSample>             Samples;
        public required int                              MiniBatchSize;
        public required RLTrainerConfig                  Config;
        public required RLCheckpoint                     BaseCheckpoint;
        public required TaskCompletionSource<PpoAsyncResult> Result;
    }

    public PpoTrainer(PolicyGroupConfig config)
    {
        _config = config;
        _trainerConfig = config.TrainerConfig;
        var hasImageStreams = HasImageStreams(config);
        _useGpuTrainingNetwork = hasImageStreams && GpuDevice.IsAvailable();
        _network = CreatePolicyNetwork(config, preferGpuImageEncoders: false);

        if (hasImageStreams)
        {
            if (_useGpuTrainingNetwork)
            {
                GD.Print($"[PPO] Group '{config.GroupId}': using GPU image encoder for training updates; rollout inference remains on CPU.");
            }
            else
            {
                GD.Print($"[PPO] Group '{config.GroupId}': GPU image encoder unavailable; training updates will run on CPU.");
            }
        }

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

        var tUpdate = RLProfiler.Begin();
        var samples = BuildTrainingSamples();
        NormalizeAdvantages(samples);
        var miniBatchSize = Math.Clamp(_trainerConfig.PpoMiniBatchSize, 1, samples.Count);
        var result = _useGpuTrainingNetwork
            ? RunSynchronousGpuUpdate(samples, miniBatchSize)
            : RunTrainingEpochs(_network, samples, _trainerConfig, miniBatchSize, new Random(unchecked((int)_rng.Randi())));

        _transitions.Clear();
        RLCheckpoint? statsCheckpoint = null;
        if (result.Checkpoint is not null)
        {
            _network.LoadCheckpoint(result.Checkpoint);
            statsCheckpoint = PrepareStatsCheckpoint(result.Checkpoint, groupId, totalSteps, episodeCount);
        }
        RLProfiler.End("PPO.TryUpdate", tUpdate);

        return new TrainerUpdateStats
        {
            PolicyLoss = result.PolicyLoss,
            ValueLoss = result.ValueLoss,
            Entropy = result.Entropy,
            ClipFraction = result.ClipFraction,
            Checkpoint = statsCheckpoint ?? new RLCheckpoint(),
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
        // Create a checkpoint with current weights and reconstruct an inference policy from it.
        var checkpoint = CreateCheckpoint(_config.GroupId, 0, 0, 0);
        return InferencePolicyFactory.Create(checkpoint);
    }

    public void LoadFromCheckpoint(RLCheckpoint checkpoint) => _network.LoadCheckpoint(checkpoint);

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

        if (_useGpuTrainingNetwork)
        {
            // Build samples on the calling thread; the GPU thread only trains.
            var samples       = BuildTrainingSamplesFrom(transitions, _trainerConfig);
            NormalizeAdvantages(samples);
            var miniBatchSize = Math.Clamp(_trainerConfig.PpoMiniBatchSize, 1, samples.Count);

            EnsureGpuTrainingThread();
            var tcs = new TaskCompletionSource<PpoAsyncResult>(TaskCreationOptions.RunContinuationsAsynchronously);
            _gpuJobQueue!.Add(new GpuTrainingJob
            {
                Samples        = samples,
                MiniBatchSize  = miniBatchSize,
                Config         = _trainerConfig,
                BaseCheckpoint = _network.SaveCheckpoint("_ppo_bg_src_", 0, 0, 0),
                Result         = tcs,
            });
            _pendingUpdate = tcs.Task;
            return true;
        }

        // Lazy-create CPU shadow network with identical architecture.
        _shadowNetwork ??= CreatePolicyNetwork(_config, preferGpuImageEncoders: false);

        // Copy live weights into shadow so backprop works on an isolated copy.
        _network.CopyWeightsTo(_shadowNetwork);

        var shadow = _shadowNetwork;
        var shadowConfig = _trainerConfig;
        _pendingUpdate = Task.Run(() => RunBackgroundUpdate(shadow, transitions, shadowConfig));
        return true;
    }

    public TrainerUpdateStats? TryPollResult(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null || !_pendingUpdate.IsCompleted)
            return null;

        if (_pendingUpdate.IsFaulted)
        {
            var msg = _pendingUpdate.Exception?.GetBaseException().Message
                      ?? _pendingUpdate.Exception?.Message
                      ?? "unknown error";
            GD.PushError($"[PPO] Background training task faulted — skipping update. {msg}");
            _pendingUpdate = null;
            return null;
        }

        var result = _pendingUpdate.Result;
        _pendingUpdate = null;

        RLCheckpoint? statsCheckpoint = null;
        if (result.Checkpoint is not null)
        {
            _network.LoadCheckpoint(result.Checkpoint);
            statsCheckpoint = PrepareStatsCheckpoint(result.Checkpoint, groupId, totalSteps, episodeCount);
        }
        else
        {
            // Apply the trained CPU shadow weights back to the live network (main thread only).
            _network.LoadWeightsFrom(_shadowNetwork!);
        }

        return new TrainerUpdateStats
        {
            PolicyLoss = result.PolicyLoss,
            ValueLoss = result.ValueLoss,
            Entropy = result.Entropy,
            ClipFraction = result.ClipFraction,
            Checkpoint = statsCheckpoint ?? new RLCheckpoint(),
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
        var tBg = RLProfiler.Begin();
        var samples = BuildTrainingSamplesFrom(transitions, config);
        NormalizeAdvantages(samples);
        var miniBatchSize = Math.Clamp(config.PpoMiniBatchSize, 1, samples.Count);
        var result = RunTrainingEpochs(network, samples, config, miniBatchSize, new Random());
        RLProfiler.End("PPO.BackgroundUpdate", tBg);
        return result;
    }

    /// <summary>
    /// Lazy-creates the dedicated GPU training thread + job queue.
    /// The thread owns the GpuCnnEncoder for its entire lifetime,
    /// satisfying the GpuDevice thread-affinity requirement.
    /// </summary>
    private void EnsureGpuTrainingThread()
    {
        if (_gpuJobQueue is not null)
            return;

        _gpuJobQueue = new BlockingCollection<GpuTrainingJob>(boundedCapacity: 1);
        _gpuThread   = new Thread(GpuTrainingThreadProc)
        {
            IsBackground = true,
            Name         = $"PPO GPU Training [{_config.GroupId}]",
        };
        _gpuThread.Start();
    }

    private void GpuTrainingThreadProc()
    {
        using var network = CreatePolicyNetwork(_config, preferGpuImageEncoders: true);
        GD.Print($"[PPO] GPU training network ready on dedicated thread for group '{_config.GroupId}'.");

        foreach (var job in _gpuJobQueue!.GetConsumingEnumerable())
        {
            try
            {
                network.LoadCheckpoint(job.BaseCheckpoint);
                var tBg    = RLProfiler.Begin();
                var result = RunTrainingEpochs(network, job.Samples, job.Config,
                                               job.MiniBatchSize, new Random(), captureCheckpoint: true);
                RLProfiler.End("PPO.BackgroundUpdate", tBg);
                job.Result.SetResult(result);
            }
            catch (Exception ex)
            {
                job.Result.SetException(ex);
            }
        }
    }

    private PpoAsyncResult RunSynchronousGpuUpdate(List<TrainingSample> samples, int miniBatchSize)
    {
        // Route through the persistent GPU thread to avoid repeated shader compilation.
        EnsureGpuTrainingThread();
        var tcs = new TaskCompletionSource<PpoAsyncResult>(TaskCreationOptions.RunContinuationsAsynchronously);
        _gpuJobQueue!.Add(new GpuTrainingJob
        {
            Samples        = samples,
            MiniBatchSize  = miniBatchSize,
            Config         = _trainerConfig,
            BaseCheckpoint = _network.SaveCheckpoint("_ppo_sync_src_", 0, 0, 0),
            Result         = tcs,
        });
        return tcs.Task.GetAwaiter().GetResult(); // block main thread — same as before
    }

    private static PpoAsyncResult RunTrainingEpochs(
        PolicyValueNetwork network,
        List<TrainingSample> samples,
        RLTrainerConfig config,
        int miniBatchSize,
        Random rng,
        bool captureCheckpoint = false)
    {
        var policyLoss = 0f;
        var valueLoss = 0f;
        var entropy = 0f;
        var clipFraction = 0f;
        var processedSamples = 0;

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
            Checkpoint = captureCheckpoint
                ? network.SaveCheckpoint("_ppo_trained_", 0, 0, 0)
                : null,
        };
    }

    private static PolicyValueNetwork CreatePolicyNetwork(PolicyGroupConfig config, bool preferGpuImageEncoders)
    {
        return config.ObsSpec is not null
            ? new PolicyValueNetwork(
                config.ObsSpec,
                config.DiscreteActionCount,
                config.ContinuousActionDimensions,
                config.NetworkGraph,
                preferGpuImageEncoders)
            : new PolicyValueNetwork(
                config.ObservationSize,
                config.DiscreteActionCount,
                config.ContinuousActionDimensions,
                config.NetworkGraph);
    }

    private static bool ShouldUseGpuTrainingNetwork(PolicyGroupConfig config)
    {
        return HasImageStreams(config) && GpuDevice.IsAvailable();
    }

    private static bool HasImageStreams(PolicyGroupConfig config)
    {
        if (config.ObsSpec is null)
            return false;

        foreach (var stream in config.ObsSpec.Streams)
            if (stream.Kind == ObservationStreamKind.Image)
                return true;

        return false;
    }

    private RLCheckpoint PrepareStatsCheckpoint(RLCheckpoint checkpoint, string groupId, long totalSteps, long episodeCount)
    {
        checkpoint.RunId = groupId;
        checkpoint.TotalSteps = totalSteps;
        checkpoint.EpisodeCount = episodeCount;
        checkpoint.UpdateCount = 0;
        return CheckpointMetadataBuilder.Apply(checkpoint, _config);
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
        public RLCheckpoint? Checkpoint { get; init; }
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
