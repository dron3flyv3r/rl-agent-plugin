using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

internal sealed class PolicyValueNetwork : IDisposable
{
    private readonly NetworkLayer[] _trunkLayers;
    private readonly DenseLayer _policyHead;
    private readonly DenseLayer _valueHead;
    private readonly int _continuousActionDims;

    // ── Multi-stream support ──────────────────────────────────────────────────
    // Non-null when constructed with an ObservationSpec that has >1 streams or
    // contains Image streams. In that case Infer() routes input through per-stream
    // encoders first, then concatenates embeddings before the shared trunk.
    private readonly StreamEncoder[]? _streamEncoders;
    private readonly ObservationSpec? _observationSpec;

    /// <summary>Discrete-action constructor (existing behaviour).</summary>
    public PolicyValueNetwork(int observationSize, int actionCount, RLNetworkGraph graph)
        : this(observationSize, actionCount, 0, graph) { }

    /// <summary>
    /// Unified constructor (flat observation, legacy path).
    /// Pass <paramref name="continuousActionDims"/> &gt; 0 for a continuous Gaussian policy
    /// (policy head outputs [mean_0..D-1, logStd_0..D-1]).
    /// Pass <paramref name="actionCount"/> &gt; 0 for a discrete softmax policy.
    /// Exactly one of the two should be non-zero.
    /// </summary>
    public PolicyValueNetwork(int observationSize, int actionCount, int continuousActionDims, RLNetworkGraph graph)
    {
        _continuousActionDims = continuousActionDims;
        _trunkLayers = graph.BuildTrunkLayers(observationSize);
        var lastSize = graph.OutputSize(observationSize);
        var policyOutSize = continuousActionDims > 0 ? continuousActionDims * 2 : actionCount;
        _policyHead = new DenseLayer(lastSize, policyOutSize, null, graph.Optimizer);
        _valueHead  = new DenseLayer(lastSize, 1,             null, graph.Optimizer);
    }

    /// <summary>
    /// Multi-stream constructor. Builds per-stream encoders based on
    /// <paramref name="spec"/> and the <see cref="RLStreamEncoderConfig"/> entries
    /// in <paramref name="graph"/>, then merges all embeddings before the shared trunk.
    /// Agents that only used legacy <c>buffer.Add()</c> produce a single-stream flat spec
    /// and fall back to the original single-trunk path transparently.
    /// </summary>
    public PolicyValueNetwork(
        ObservationSpec spec,
        int actionCount,
        int continuousActionDims,
        RLNetworkGraph graph,
        bool preferGpuImageEncoders = false)
    {
        _continuousActionDims = continuousActionDims;
        _observationSpec = spec;

        // Build per-stream encoders and compute merged embedding size.
        var encoders = new StreamEncoder[spec.Streams.Count];
        var mergedSize = 0;
        for (var i = 0; i < spec.Streams.Count; i++)
        {
            var stream = spec.Streams[i];
            var streamCfg = stream.EncoderConfig;

            IEncoder? cnn = null;
            NetworkLayer[]? vectorLayers = null;
            int encoderOutputSize;

            if (stream.Kind == ObservationStreamKind.Image)
            {
                var cnnDef = streamCfg?.CnnEncoder ?? new RLCnnEncoderDef();
                cnn = CreateImageEncoder(
                    stream.Width,
                    stream.Height,
                    stream.Channels,
                    cnnDef,
                    preferGpuImageEncoders);
                var cnnOut = cnn.OutputSize;
                // Optional per-stream vector encoder after CNN projection.
                if (streamCfg?.VectorEncoder is { } vecGraph && vecGraph.TrunkLayers.Count > 0)
                {
                    vectorLayers = vecGraph.BuildTrunkLayers(cnnOut, graph.Optimizer);
                    encoderOutputSize = vecGraph.OutputSize(cnnOut);
                }
                else
                {
                    encoderOutputSize = cnnOut;
                }
            }
            else
            {
                // Vector stream: optional per-stream MLP, otherwise pass-through.
                if (streamCfg?.VectorEncoder is { } vecGraph && vecGraph.TrunkLayers.Count > 0)
                {
                    vectorLayers = vecGraph.BuildTrunkLayers(stream.FlatSize, graph.Optimizer);
                    encoderOutputSize = vecGraph.OutputSize(stream.FlatSize);
                }
                else
                {
                    encoderOutputSize = stream.FlatSize;
                }
            }

            encoders[i]  = new StreamEncoder(i, stream, cnn, vectorLayers, encoderOutputSize);
            mergedSize  += encoderOutputSize;
        }

        _streamEncoders = encoders;

        // Shared trunk and heads take the concatenated embedding as input.
        _trunkLayers = graph.BuildTrunkLayers(mergedSize);
        var lastSize      = graph.OutputSize(mergedSize);
        var policyOutSize = continuousActionDims > 0 ? continuousActionDims * 2 : actionCount;
        _policyHead = new DenseLayer(lastSize, policyOutSize, null, graph.Optimizer);
        _valueHead  = new DenseLayer(lastSize, 1,             null, graph.Optimizer);
    }

    public NetworkInference Infer(float[] observation)
    {
        var t = RLProfiler.Begin();
        var x = _streamEncoders is not null ? EncodeStreams(observation) : observation;
        foreach (var layer in _trunkLayers)
            x = layer.Forward(x);

        var logits = _policyHead.Forward(x);
        var value  = _valueHead.Forward(x);
        RLProfiler.End("Infer", t);

        return new NetworkInference { Logits = logits, Value = value[0] };
    }

    /// <summary>Routes a flat observation through per-stream encoders and concatenates embeddings.</summary>
    private float[] EncodeStreams(float[] observation, int batchIndex = -1, float[][]? batchedCnnOutputs = null)
    {
        var encoders = _streamEncoders!;
        var spec     = _observationSpec!;
        var mergedSize = 0;
        foreach (var enc in encoders) mergedSize += enc.OutputSize;

        var merged  = new float[mergedSize];
        var offset  = 0; // offset into the flat observation
        var outIdx  = 0; // offset into merged embedding

        for (var i = 0; i < encoders.Length; i++)
        {
            var enc    = encoders[i];
            var stream = spec.Streams[i];
            var slice  = observation.AsSpan(offset, stream.FlatSize).ToArray();
            offset += stream.FlatSize;

            float[] embedding;
            if (enc.Cnn is not null)
            {
                if (batchIndex >= 0 &&
                    batchedCnnOutputs is not null &&
                    enc.Cnn.SupportsBatchedTraining &&
                    batchedCnnOutputs[i] is { } outputBatch)
                {
                    embedding = new float[enc.Cnn.OutputSize];
                    Array.Copy(outputBatch, batchIndex * embedding.Length, embedding, 0, embedding.Length);
                }
                else
                {
                    var tc = RLProfiler.Begin();
                    embedding = enc.Cnn.Forward(slice);
                    RLProfiler.End("CNN.Encode", tc);
                }
            }
            else
            {
                embedding = slice;
            }

            if (enc.VectorLayers is not null)
            {
                foreach (var layer in enc.VectorLayers)
                    embedding = layer.Forward(embedding);
            }

            Array.Copy(embedding, 0, merged, outIdx, embedding.Length);
            outIdx += enc.OutputSize;
        }

        return merged;
    }

    private NetworkInference InferPrepared(float[] observation, int batchIndex, float[][]? batchedCnnOutputs)
    {
        var x = EncodeStreams(observation, batchIndex, batchedCnnOutputs);
        foreach (var layer in _trunkLayers)
            x = layer.Forward(x);

        var logits = _policyHead.Forward(x);
        var value  = _valueHead.Forward(x);
        return new NetworkInference { Logits = logits, Value = value[0] };
    }

    /// <summary>
    /// Runs <see cref="EncodeStreams"/> on every row of <paramref name="observations"/> and
    /// returns a new <see cref="VectorBatch"/> of the merged embedding size.
    /// </summary>
    private VectorBatch EncodeStreamsBatch(VectorBatch observations)
    {
        var encoders   = _streamEncoders!;
        var mergedSize = 0;
        foreach (var enc in encoders) mergedSize += enc.OutputSize;

        var encoded = new VectorBatch(observations.BatchSize, mergedSize);
        for (var b = 0; b < observations.BatchSize; b++)
        {
            var row = observations.CopyRow(b);
            encoded.SetRow(b, EncodeStreams(row));
        }
        return encoded;
    }

    public BatchNetworkInference InferBatch(VectorBatch observations)
    {
        var t = RLProfiler.Begin();
        var trunkOutput = _streamEncoders is not null
            ? EncodeStreamsBatch(observations)
            : observations;
        foreach (var layer in _trunkLayers)
            trunkOutput = layer.ForwardBatch(trunkOutput);

        var logits     = _policyHead.ForwardBatch(trunkOutput);
        var valueBatch = _valueHead.ForwardBatch(trunkOutput);
        var values     = new float[observations.BatchSize];
        for (var b = 0; b < observations.BatchSize; b++)
            values[b] = valueBatch.Get(b, 0);
        RLProfiler.End("InferBatch", t);

        return new BatchNetworkInference { Logits = logits, Values = values };
    }

    public PpoBatchUpdateStats ApplyGradients(IReadOnlyList<TrainingSample> samples, RLTrainerConfig config)
    {
        if (samples.Count == 0)
            return new PpoBatchUpdateStats();

        var tGrad = RLProfiler.Begin();

        var trunkGradients = new GradientBuffer[_trunkLayers.Length];
        for (var i = 0; i < _trunkLayers.Length; i++)
            trunkGradients[i] = _trunkLayers[i].CreateGradientBuffer();

        // Create fresh gradient tokens for stream encoders.
        ICnnGradientToken[]? cnnGrads      = null;
        GradientBuffer[][]?  vecLayerGrads = null;
        if (_streamEncoders is not null)
        {
            cnnGrads      = new ICnnGradientToken[_streamEncoders.Length];
            vecLayerGrads = new GradientBuffer[_streamEncoders.Length][];
            for (var ei = 0; ei < _streamEncoders.Length; ei++)
            {
                if (_streamEncoders[ei].Cnn is not null)
                    cnnGrads[ei] = _streamEncoders[ei].Cnn!.CreateGradientToken();
                vecLayerGrads[ei] = _streamEncoders[ei].CreateVectorGradBuffers() ?? System.Array.Empty<GradientBuffer>();
            }
        }

        float[][]? batchedCnnOutputs = null;
        float[][]? batchedCnnOutputGrads = null;
        if (_streamEncoders is not null && _observationSpec is not null)
        {
            var tBatchForward = RLProfiler.Begin();
            batchedCnnOutputs = new float[_streamEncoders.Length][];
            batchedCnnOutputGrads = new float[_streamEncoders.Length][];
            var streamOffsets = BuildStreamOffsets(_observationSpec);
            var batchedEncoderCount = 0;

            for (var ei = 0; ei < _streamEncoders.Length; ei++)
            {
                var enc = _streamEncoders[ei];
                if (enc.Cnn is null || !enc.Cnn.SupportsBatchedTraining)
                    continue;

                batchedEncoderCount++;

                var stream = _observationSpec.Streams[ei];
                var inputBatch = new float[samples.Count * stream.FlatSize];
                for (var sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
                {
                    Array.Copy(
                        samples[sampleIndex].Observation,
                        streamOffsets[ei],
                        inputBatch,
                        sampleIndex * stream.FlatSize,
                        stream.FlatSize);
                }

                var outputBatch = new float[samples.Count * enc.Cnn.OutputSize];
                enc.Cnn.ForwardBatch(inputBatch, samples.Count, outputBatch);
                batchedCnnOutputs[ei] = outputBatch;
                batchedCnnOutputGrads[ei] = new float[samples.Count * enc.Cnn.OutputSize];
            }
            RLProfiler.End("ApplyGradients.CnnBatchForward", tBatchForward);
            if (batchedEncoderCount == 0)
                GD.Print("[PolicyValueNetwork] ApplyGradients: no batched CNN encoders active for this network.");
        }

        var policyGradients = _policyHead.CreateGradientBuffer();
        var valueGradients  = _valueHead.CreateGradientBuffer();

        var totalPolicyLoss = 0f;
        var totalValueLoss  = 0f;
        var totalEntropy    = 0f;
        var clipCount       = 0;

        var tSampleLoop = RLProfiler.Begin();
        for (var sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            var sample = samples[sampleIndex];
            var inference = _streamEncoders is not null
                ? InferPrepared(sample.Observation, sampleIndex, batchedCnnOutputs)
                : Infer(sample.Observation);
            float[] logitsGradient;

            if (_continuousActionDims > 0)
            {
                // ── Continuous Gaussian policy (tanh-squashed) ─────────────────────
                var D = _continuousActionDims;
                var actorOut = inference.Logits; // [mean_0..D-1, logStd_0..D-1]
                var newLogProb = 0f;
                var entropy = 0f;

                // Pre-compute per-dimension values; reused for both loss and gradient.
                var eps = new float[D];
                var std = new float[D];
                for (var d = 0; d < D; d++)
                {
                    var a      = sample.ContinuousActions[d];
                    var u      = Atanh(a);
                    var lStd   = Math.Clamp(actorOut[D + d], -20f, 2f);
                    std[d]     = MathF.Exp(lStd);
                    eps[d]     = (u - actorOut[d]) / std[d];
                    newLogProb += -0.5f * eps[d] * eps[d] - lStd
                                  - 0.5f * MathF.Log(2f * MathF.PI)
                                  - MathF.Log(1f - a * a + 1e-6f);
                    // Gaussian entropy per dim: 0.5*(1+log(2π)) + logStd
                    entropy += 0.5f * (1f + MathF.Log(2f * MathF.PI)) + lStd;
                }

                totalEntropy += entropy;

                var ratio          = MathF.Exp(newLogProb - sample.OldLogProbability);
                var clippedRatio   = Math.Clamp(ratio, 1f - config.ClipEpsilon, 1f + config.ClipEpsilon);
                var unclipped      = ratio * sample.Advantage;
                var clipped        = clippedRatio * sample.Advantage;
                totalPolicyLoss   += -Math.Min(unclipped, clipped);
                if (MathF.Abs(ratio - 1f) > config.ClipEpsilon) clipCount++;

                logitsGradient = new float[D * 2];
                if (unclipped <= clipped) // unclipped objective is smaller → gradient is non-zero
                {
                    // Gradient of -(L_PPO + α*H) w.r.t. [mean_d, logStd_d]:
                    //   ∂(-L_PPO)/∂mean_d    = -ratio*A * (eps_d / std_d)
                    //   ∂(-L_PPO)/∂logStd_d  = -ratio*A * (eps_d² - 1)
                    //   ∂(-α*H) /∂logStd_d   = -α               (H = Σ[logStd + const])
                    var policyScale = ratio * sample.Advantage;
                    for (var d = 0; d < D; d++)
                    {
                        logitsGradient[d]     = -policyScale * (eps[d] / std[d]);
                        logitsGradient[D + d] = -policyScale * (eps[d] * eps[d] - 1f)
                                                - config.EntropyCoefficient;
                    }
                }
                else if (config.EntropyCoefficient > 0f)
                {
                    // Clipped: only entropy gradient applies.
                    for (var d = 0; d < D; d++)
                        logitsGradient[D + d] = -config.EntropyCoefficient;
                }
            }
            else
            {
                // ── Discrete softmax policy (existing path) ────────────────────────
                var probs = Softmax(inference.Logits);
                var actionProbability = Math.Clamp(probs[sample.Action], 1e-6f, 1.0f);
                var logProbability = Mathf.Log(actionProbability);
                var ratio = Mathf.Exp(logProbability - sample.OldLogProbability);
                var clippedRatio = Math.Clamp(ratio, 1.0f - config.ClipEpsilon, 1.0f + config.ClipEpsilon);
                var unclippedObjective = ratio * sample.Advantage;
                var clippedObjective = clippedRatio * sample.Advantage;
                totalPolicyLoss += -Math.Min(unclippedObjective, clippedObjective);

                if (Mathf.Abs(ratio - 1.0f) > config.ClipEpsilon) clipCount++;

                logitsGradient = new float[probs.Length];
                if (unclippedObjective <= clippedObjective)
                {
                    for (var index = 0; index < probs.Length; index++)
                        logitsGradient[index] = ratio * probs[index] * sample.Advantage;
                    logitsGradient[sample.Action] -= ratio * sample.Advantage;
                }

                var entropy = 0f;
                foreach (var probability in probs)
                {
                    if (probability > 1e-6f) entropy -= probability * Mathf.Log(probability);
                }

                totalEntropy += entropy;
                if (config.EntropyCoefficient > 0f)
                {
                    for (var j = 0; j < logitsGradient.Length; j++)
                    {
                        var logPj = probs[j] > 1e-6f ? Mathf.Log(probs[j]) : Mathf.Log(1e-6f);
                        logitsGradient[j] += config.EntropyCoefficient * probs[j] * (entropy + logPj);
                    }
                }
            }

            var valuePrediction = inference.Value;
            var valueError = valuePrediction - sample.Return;
            var valueLoss = valueError * valueError;
            var valueGradientScalar = valueError;

            if (config.UseValueClipping && config.ValueClipEpsilon > 0f)
            {
                var clippedValue = sample.ValueEstimate
                    + Math.Clamp(valuePrediction - sample.ValueEstimate, -config.ValueClipEpsilon, config.ValueClipEpsilon);
                var clippedError = clippedValue - sample.Return;
                var clippedValueLoss = clippedError * clippedError;
                if (clippedValueLoss > valueLoss)
                {
                    valueLoss = clippedValueLoss;
                    if (Mathf.Abs(valuePrediction - sample.ValueEstimate) > config.ValueClipEpsilon)
                        valueGradientScalar = 0f;
                }
            }

            totalValueLoss += valueLoss;
            var valueGradient = new[] { config.ValueLossCoefficient * valueGradientScalar };

            // Infer() already cached state in each layer; AccumulateGradients uses it.
            var trunkGradientFromPolicy = _policyHead.AccumulateGradients(logitsGradient, policyGradients);
            var trunkGradientFromValue  = _valueHead.AccumulateGradients(valueGradient,   valueGradients);

            var trunkGradient = new float[trunkGradientFromPolicy.Length];
            for (var index = 0; index < trunkGradient.Length; index++)
                trunkGradient[index] = trunkGradientFromPolicy[index] + trunkGradientFromValue[index];

            for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
                trunkGradient = _trunkLayers[layerIndex].AccumulateGradients(trunkGradient, trunkGradients[layerIndex]);

            // Backprop into per-stream encoders if present.
            if (_streamEncoders is not null)
            {
                var outIdx = 0;
                for (var ei = 0; ei < _streamEncoders.Length; ei++)
                {
                    var enc     = _streamEncoders[ei];
                    var encSize = enc.OutputSize;
                    var encGrad = trunkGradient.AsSpan(outIdx, encSize).ToArray();
                    outIdx += encSize;

                    // Backprop through per-stream vector layers (in reverse).
                    if (enc.VectorLayers is not null)
                    {
                        for (var li = enc.VectorLayers.Length - 1; li >= 0; li--)
                            encGrad = enc.VectorLayers[li].AccumulateGradients(encGrad, vecLayerGrads![ei][li]);
                    }

                    if (enc.Cnn is not null)
                    {
                        if (enc.Cnn.SupportsBatchedTraining &&
                            batchedCnnOutputGrads is not null &&
                            batchedCnnOutputGrads[ei] is { } gradBatch)
                        {
                            Array.Copy(encGrad, 0, gradBatch, sampleIndex * encGrad.Length, encGrad.Length);
                        }
                        else
                        {
                            enc.Cnn.AccumulateGradients(encGrad, cnnGrads![ei]);
                        }
                    }
                }
            }
        }
        RLProfiler.End("ApplyGradients.SampleLoop", tSampleLoop);

        if (batchedCnnOutputGrads is not null && cnnGrads is not null && _streamEncoders is not null)
        {
            var tBatchBackward = RLProfiler.Begin();
            for (var ei = 0; ei < _streamEncoders.Length; ei++)
            {
                var enc = _streamEncoders[ei];
                if (enc.Cnn is not null &&
                    enc.Cnn.SupportsBatchedTraining &&
                    batchedCnnOutputGrads[ei] is { } gradBatch)
                {
                    enc.Cnn.AccumulateGradientsBatch(gradBatch, samples.Count, cnnGrads[ei]);
                }
            }
            RLProfiler.End("ApplyGradients.CnnBatchBackward", tBatchBackward);
        }

        var tGradNorm = RLProfiler.Begin();
        var globalNormSquared = policyGradients.SumSquares() + valueGradients.SumSquares();
        foreach (var g in trunkGradients) globalNormSquared += g.SumSquares();

        // Include CNN/stream grad norms for global clip.
        if (cnnGrads is not null && _streamEncoders is not null)
            for (var ei = 0; ei < _streamEncoders.Length; ei++)
                if (_streamEncoders[ei].Cnn is not null)
                    globalNormSquared += _streamEncoders[ei].Cnn!.GradNormSquared(cnnGrads[ei]);

        var gradientScale = 1f / samples.Count;
        if (config.MaxGradientNorm > 0f)
        {
            var averageNorm = Mathf.Sqrt(globalNormSquared) * gradientScale;
            if (averageNorm > config.MaxGradientNorm)
                gradientScale *= config.MaxGradientNorm / averageNorm;
        }
        RLProfiler.End("ApplyGradients.GradNorm", tGradNorm);

        var tApplyCpu = RLProfiler.Begin();
        _policyHead.ApplyGradients(policyGradients, config.LearningRate, gradientScale);
        _valueHead.ApplyGradients(valueGradients,   config.LearningRate, gradientScale);
        for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
            _trunkLayers[layerIndex].ApplyGradients(trunkGradients[layerIndex], config.LearningRate, gradientScale);
        RLProfiler.End("ApplyGradients.ApplyCpuLayers", tApplyCpu);

        // Apply stream encoder gradients.
        if (cnnGrads is not null && _streamEncoders is not null)
        {
            var tApplyEncoders = RLProfiler.Begin();
            for (var ei = 0; ei < _streamEncoders.Length; ei++)
            {
                var enc = _streamEncoders[ei];
                if (enc.Cnn is not null)
                    enc.Cnn.ApplyGradients(cnnGrads[ei], config.LearningRate, gradientScale);
                if (enc.VectorLayers is not null)
                    for (var li = 0; li < enc.VectorLayers.Length; li++)
                        enc.VectorLayers[li].ApplyGradients(vecLayerGrads![ei][li], config.LearningRate, gradientScale);
            }
            RLProfiler.End("ApplyGradients.ApplyEncoders", tApplyEncoders);
        }

        RLProfiler.End("ApplyGradients", tGrad);
        return new PpoBatchUpdateStats
        {
            PolicyLoss    = totalPolicyLoss / samples.Count,
            ValueLoss     = totalValueLoss  / samples.Count,
            Entropy       = totalEntropy    / samples.Count,
            ClipFraction  = (float)clipCount / samples.Count,
        };
    }

    public RLCheckpoint SaveCheckpoint(string runId, long totalSteps, long episodeCount, long updateCount)
    {
        var weights = new List<float>();
        var shapes  = new List<int>();

        if (_streamEncoders is not null)
        {
            shapes.Add(1); // marker: multi-stream checkpoint
            foreach (var enc in _streamEncoders)
            {
                enc.Cnn?.AppendSerialized(weights, shapes);
                if (enc.VectorLayers is not null)
                    foreach (var layer in enc.VectorLayers)
                        layer.AppendSerialized(weights, shapes);
            }
        }
        else
        {
            shapes.Add(0); // marker: flat checkpoint
        }

        foreach (var layer in _trunkLayers) layer.AppendSerialized(weights, shapes);
        _policyHead.AppendSerialized(weights, shapes);
        _valueHead.AppendSerialized(weights, shapes);

        return new RLCheckpoint
        {
            RunId        = runId,
            TotalSteps   = totalSteps,
            EpisodeCount = episodeCount,
            UpdateCount  = updateCount,
            WeightBuffer = weights.ToArray(),
            LayerShapeBuffer = shapes.ToArray(),
        };
    }

    /// <summary>Copies weights from this network into <paramref name="other"/> (same architecture required).</summary>
    internal void CopyWeightsTo(PolicyValueNetwork other)
    {
        if (_streamEncoders is not null && other._streamEncoders is not null)
            for (var i = 0; i < _streamEncoders.Length; i++)
                _streamEncoders[i].CopyWeightsTo(other._streamEncoders[i]);

        for (var i = 0; i < _trunkLayers.Length; i++)
            other._trunkLayers[i].CopyFrom(_trunkLayers[i]);
        other._policyHead.CopyFrom(_policyHead);
        other._valueHead.CopyFrom(_valueHead);
    }

    /// <summary>Overwrites this network's weights from <paramref name="other"/> (same architecture required).</summary>
    internal void LoadWeightsFrom(PolicyValueNetwork other)
    {
        if (other._streamEncoders is not null && _streamEncoders is not null)
            for (var i = 0; i < _streamEncoders.Length; i++)
                other._streamEncoders[i].CopyWeightsTo(_streamEncoders[i]);

        for (var i = 0; i < _trunkLayers.Length; i++)
            _trunkLayers[i].CopyFrom(other._trunkLayers[i]);
        _policyHead.CopyFrom(other._policyHead);
        _valueHead.CopyFrom(other._valueHead);
    }

    public void Dispose()
    {
        if (_streamEncoders is null)
            return;

        foreach (var encoder in _streamEncoders)
            if (encoder.Cnn is IDisposable disposable)
                disposable.Dispose();
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var wi       = 0;
        var si       = 0;
        var isLegacy = checkpoint.FormatVersion < RLCheckpoint.CurrentFormatVersion;

        // Read multi-stream marker (only present in format version 6+).
        var hasStreamEncoders = false;
        if (!isLegacy && checkpoint.LayerShapeBuffer.Length > 0)
        {
            hasStreamEncoders = checkpoint.LayerShapeBuffer[si++] == 1;
        }

        if (hasStreamEncoders && _streamEncoders is not null)
        {
            foreach (var enc in _streamEncoders)
            {
                enc.Cnn?.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);
                if (enc.VectorLayers is not null)
                    foreach (var layer in enc.VectorLayers)
                        layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
            }
        }
        else if (!isLegacy)
        {
            // flat checkpoint — skip the marker we already read
        }

        foreach (var layer in _trunkLayers)
            layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);

        _policyHead.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        _valueHead.LoadSerialized(checkpoint.WeightBuffer,  ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
    }

    /// <summary>
    /// Deterministic continuous action: tanh(mean).
    /// Only valid when this network was built with continuousActionDims &gt; 0.
    /// </summary>
    public float[] SelectDeterministicContinuousAction(float[] observation)
    {
        var actorOut = Infer(observation).Logits;
        var D = _continuousActionDims;
        var action = new float[D];
        for (var i = 0; i < D; i++)
            action[i] = MathF.Tanh(actorOut[i]);
        return action;
    }

    public int SelectGreedyAction(float[] observation)
    {
        var logits = Infer(observation).Logits;
        var bestIndex = 0;
        var bestValue = logits[0];
        for (var index = 1; index < logits.Length; index++)
        {
            if (logits[index] > bestValue)
            {
                bestValue = logits[index];
                bestIndex = index;
            }
        }

        return bestIndex;
    }

    /// <summary>
    /// Stochastic discrete action: sample from softmax(logits).
    /// </summary>
    public int SelectStochasticAction(float[] observation, Random rng)
    {
        var logits = Infer(observation).Logits;
        var probs = Softmax(logits);
        var roll = rng.NextSingle();
        var cum = 0f;
        for (var i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (roll <= cum) return i;
        }
        return probs.Length - 1;
    }

    /// <summary>
    /// Stochastic continuous action: sample from N(mean, exp(logStd)) then tanh-squash.
    /// Only valid when this network was built with continuousActionDims > 0.
    /// </summary>
    public float[] SelectStochasticContinuousAction(float[] observation, Random rng)
    {
        var actorOut = Infer(observation).Logits;
        var D = _continuousActionDims;
        var action = new float[D];
        for (var i = 0; i < D; i++)
        {
            var mean   = actorOut[i];
            var logStd = Math.Clamp(actorOut[D + i], -20f, 2f);
            var std    = MathF.Exp(logStd);
            var u1     = Math.Max(rng.NextSingle(), 1e-10f);
            var u2     = rng.NextSingle();
            var eps    = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
            action[i]  = MathF.Tanh(mean + std * eps);
        }
        return action;
    }

    public int[] SelectGreedyActions(VectorBatch observations)
    {
        var inference = InferBatch(observations);
        var actions = new int[observations.BatchSize];
        for (var b = 0; b < observations.BatchSize; b++)
        {
            var bestIndex = 0;
            var bestValue = inference.Logits.Get(b, 0);
            for (var a = 1; a < inference.Logits.VectorSize; a++)
            {
                var logit = inference.Logits.Get(b, a);
                if (logit > bestValue) { bestValue = logit; bestIndex = a; }
            }

            actions[b] = bestIndex;
        }

        return actions;
    }

    /// <summary>Inverse tanh, clamped to avoid ±∞ at the boundaries.</summary>
    private static float Atanh(float x)
    {
        x = Math.Clamp(x, -1f + 1e-6f, 1f - 1e-6f);
        return 0.5f * MathF.Log((1f + x) / (1f - x));
    }

    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var expValues = new float[logits.Length];
        var total = 0.0f;
        for (var index = 0; index < logits.Length; index++)
        {
            expValues[index] = Mathf.Exp(logits[index] - maxLogit);
            total += expValues[index];
        }

        for (var index = 0; index < expValues.Length; index++)
            expValues[index] /= total;

        return expValues;
    }

    private static int[] BuildStreamOffsets(ObservationSpec spec)
    {
        var offsets = new int[spec.Streams.Count];
        var offset = 0;
        for (var i = 0; i < spec.Streams.Count; i++)
        {
            offsets[i] = offset;
            offset += spec.Streams[i].FlatSize;
        }
        return offsets;
    }

    private static IEncoder CreateImageEncoder(
        int width,
        int height,
        int channels,
        RLCnnEncoderDef def,
        bool preferGpuImageEncoders)
    {
        if (preferGpuImageEncoders && GpuDevice.IsAvailable())
        {
            try
            {
                GD.Print($"[PolicyValueNetwork] Using GpuCnnEncoder for image stream {width}x{height}x{channels}.");
                return new GpuCnnEncoder(width, height, channels, def);
            }
            catch (Exception ex)
            {
                var message =
                    $"[PolicyValueNetwork] GpuCnnEncoder init failed for image stream {width}x{height}x{channels}: {ex}";
                GD.PushError(message);
                throw new InvalidOperationException(message, ex);
            }
        }

        if (preferGpuImageEncoders)
            GD.PushWarning($"[PolicyValueNetwork] Vulkan unavailable — falling back to CnnEncoder for image stream {width}x{height}x{channels}.");
        return new CnnEncoder(width, height, channels, def);
    }

    // ── StreamEncoder helper ─────────────────────────────────────────────────

    private sealed class StreamEncoder
    {
        public readonly int                StreamIndex;
        public readonly ObservationStreamSpec Stream;
        public readonly IEncoder?          Cnn;
        public readonly NetworkLayer[]?    VectorLayers;
        public readonly int                OutputSize;

        public StreamEncoder(
            int index,
            ObservationStreamSpec stream,
            IEncoder? cnn,
            NetworkLayer[]? vectorLayers,
            int outputSize)
        {
            StreamIndex  = index;
            Stream       = stream;
            Cnn          = cnn;
            VectorLayers = vectorLayers;
            OutputSize   = outputSize;
        }

        public GradientBuffer[]? CreateVectorGradBuffers() =>
            VectorLayers is not null
                ? System.Array.ConvertAll(VectorLayers, l => l.CreateGradientBuffer())
                : null;

        public void CopyWeightsTo(StreamEncoder other)
        {
            Cnn?.CopyWeightsTo(other.Cnn!);
            if (VectorLayers is not null && other.VectorLayers is not null)
                for (var i = 0; i < VectorLayers.Length; i++)
                    other.VectorLayers[i].CopyFrom(VectorLayers[i]);
        }
    }

    internal sealed class NetworkInference
    {
        public float[] Logits { get; init; } = Array.Empty<float>();
        public float Value { get; init; }
    }

    internal sealed class BatchNetworkInference
    {
        public VectorBatch Logits { get; init; } = new(1, 1);
        public float[] Values { get; init; } = Array.Empty<float>();
    }

    internal sealed class PpoBatchUpdateStats
    {
        public float PolicyLoss   { get; init; }
        public float ValueLoss    { get; init; }
        public float Entropy      { get; init; }
        public float ClipFraction { get; init; }
    }
}
