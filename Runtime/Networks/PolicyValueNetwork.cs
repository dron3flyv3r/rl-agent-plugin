using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

internal sealed class PolicyValueNetwork
{
    private readonly NetworkLayer[] _trunkLayers;
    private readonly DenseLayer _policyHead;
    private readonly DenseLayer _valueHead;
    private readonly int _continuousActionDims;

    /// <summary>Discrete-action constructor (existing behaviour).</summary>
    public PolicyValueNetwork(int observationSize, int actionCount, RLNetworkGraph graph)
        : this(observationSize, actionCount, 0, graph) { }

    /// <summary>
    /// Unified constructor.
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
        // Continuous: [mean_0..D-1, logStd_0..D-1] = 2*D outputs
        // Discrete:   actionCount logits
        var policyOutSize = continuousActionDims > 0 ? continuousActionDims * 2 : actionCount;
        _policyHead = new DenseLayer(lastSize, policyOutSize, null, graph.Optimizer);
        _valueHead  = new DenseLayer(lastSize, 1,             null, graph.Optimizer);
    }

    public NetworkInference Infer(float[] observation)
    {
        var x = observation;
        foreach (var layer in _trunkLayers)
            x = layer.Forward(x);

        var logits = _policyHead.Forward(x);
        var value  = _valueHead.Forward(x);

        return new NetworkInference { Logits = logits, Value = value[0] };
    }

    public BatchNetworkInference InferBatch(VectorBatch observations)
    {
        var trunkOutput = observations;
        foreach (var layer in _trunkLayers)
            trunkOutput = layer.ForwardBatch(trunkOutput);

        var logits     = _policyHead.ForwardBatch(trunkOutput);
        var valueBatch = _valueHead.ForwardBatch(trunkOutput);
        var values     = new float[observations.BatchSize];
        for (var b = 0; b < observations.BatchSize; b++)
            values[b] = valueBatch.Get(b, 0);

        return new BatchNetworkInference { Logits = logits, Values = values };
    }

    public PpoBatchUpdateStats ApplyGradients(IReadOnlyList<TrainingSample> samples, RLTrainerConfig config)
    {
        if (samples.Count == 0)
            return new PpoBatchUpdateStats();

        var trunkGradients = new GradientBuffer[_trunkLayers.Length];
        for (var i = 0; i < _trunkLayers.Length; i++)
            trunkGradients[i] = _trunkLayers[i].CreateGradientBuffer();

        var policyGradients = _policyHead.CreateGradientBuffer();
        var valueGradients  = _valueHead.CreateGradientBuffer();

        var totalPolicyLoss = 0f;
        var totalValueLoss  = 0f;
        var totalEntropy    = 0f;
        var clipCount       = 0;

        foreach (var sample in samples)
        {
            var inference = Infer(sample.Observation);
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
        }

        var globalNormSquared = policyGradients.SumSquares() + valueGradients.SumSquares();
        foreach (var g in trunkGradients) globalNormSquared += g.SumSquares();

        var gradientScale = 1f / samples.Count;
        if (config.MaxGradientNorm > 0f)
        {
            var averageNorm = Mathf.Sqrt(globalNormSquared) * gradientScale;
            if (averageNorm > config.MaxGradientNorm)
                gradientScale *= config.MaxGradientNorm / averageNorm;
        }

        _policyHead.ApplyGradients(policyGradients, config.LearningRate, gradientScale);
        _valueHead.ApplyGradients(valueGradients,   config.LearningRate, gradientScale);
        for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
            _trunkLayers[layerIndex].ApplyGradients(trunkGradients[layerIndex], config.LearningRate, gradientScale);

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
        for (var i = 0; i < _trunkLayers.Length; i++)
            other._trunkLayers[i].CopyFrom(_trunkLayers[i]);
        other._policyHead.CopyFrom(_policyHead);
        other._valueHead.CopyFrom(_valueHead);
    }

    /// <summary>Overwrites this network's weights from <paramref name="other"/> (same architecture required).</summary>
    internal void LoadWeightsFrom(PolicyValueNetwork other)
    {
        for (var i = 0; i < _trunkLayers.Length; i++)
            _trunkLayers[i].CopyFrom(other._trunkLayers[i]);
        _policyHead.CopyFrom(other._policyHead);
        _valueHead.CopyFrom(other._valueHead);
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var wi       = 0;
        var si       = 0;
        var isLegacy = checkpoint.FormatVersion < RLCheckpoint.CurrentFormatVersion;

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
