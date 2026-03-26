using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// SAC neural networks: actor + twin Q-networks + twin target Q-networks.
/// Supports both discrete actions (Q per action) and continuous actions (Q from obs+action).
/// </summary>
internal sealed class SacNetwork
{
    private readonly int _obsSize;
    private readonly int _actionDim;

    // Actor layers (trunk + head)
    private readonly NetworkLayer[] _actorTrunk;
    private readonly NetworkLayer   _actorHead;

    // Twin Q-networks
    private readonly NetworkLayer[] _q1Trunk;
    private readonly NetworkLayer   _q1Head;
    private readonly NetworkLayer[] _q2Trunk;
    private readonly NetworkLayer   _q2Head;

    // Target Q-networks (frozen — weights maintained via soft/hard copy only)
    private readonly NetworkLayer[] _q1TargetTrunk;
    private readonly NetworkLayer   _q1TargetHead;
    private readonly NetworkLayer[] _q2TargetTrunk;
    private readonly NetworkLayer   _q2TargetHead;

    public SacNetwork(int obsSize, int actionDim, bool isContinuous, RLNetworkGraph graph, float learningRate)
    {
        _obsSize   = obsSize;
        _actionDim = actionDim;

        _actorTrunk = graph.BuildTrunkLayers(obsSize);
        var actorTrunkOut = graph.OutputSize(obsSize);
        // Discrete: logits over actions; Continuous: [mean_0..mean_D, log_std_0..log_std_D]
        var actorOutSize = isContinuous ? actionDim * 2 : actionDim;
        _actorHead = new DenseLayer(actorTrunkOut, actorOutSize, null, graph.Optimizer);

        var qInputSize = isContinuous ? obsSize + actionDim : obsSize;
        var qOutSize   = isContinuous ? 1 : actionDim;
        var qTrunkOut  = graph.OutputSize(qInputSize);

        _q1Trunk = graph.BuildTrunkLayers(qInputSize);
        _q1Head  = new DenseLayer(qTrunkOut, qOutSize, null, graph.Optimizer);

        _q2Trunk = graph.BuildTrunkLayers(qInputSize);
        _q2Head  = new DenseLayer(qTrunkOut, qOutSize, null, graph.Optimizer);

        // Target Q-networks — RLOptimizerKind.None: no moment vectors, no weight updates
        _q1TargetTrunk = graph.BuildTrunkLayers(qInputSize, RLOptimizerKind.None);
        _q1TargetHead  = new DenseLayer(qTrunkOut, qOutSize, null, RLOptimizerKind.None);

        _q2TargetTrunk = graph.BuildTrunkLayers(qInputSize, RLOptimizerKind.None);
        _q2TargetHead  = new DenseLayer(qTrunkOut, qOutSize, null, RLOptimizerKind.None);

        HardCopyToTargets();
    }

    // ── Discrete action methods ──────────────────────────────────────────────

    public (int action, float logProb, float entropy) SampleDiscreteAction(float[] obs, Random rng)
        => SampleDiscreteActionFromLogits(ForwardActor(obs), rng);

    public DiscreteActionBatch SampleDiscreteActions(VectorBatch observations, Random rng)
    {
        var logitsBatch = ForwardActorBatch(observations);
        var actions     = new int[observations.BatchSize];
        var logProbs    = new float[observations.BatchSize];
        var entropies   = new float[observations.BatchSize];
        for (var b = 0; b < observations.BatchSize; b++)
        {
            var (action, logProb, entropy) = SampleDiscreteActionFromLogits(logitsBatch.CopyRow(b), rng);
            actions[b]   = action;
            logProbs[b]  = logProb;
            entropies[b] = entropy;
        }

        return new DiscreteActionBatch { Actions = actions, LogProbabilities = logProbs, Entropies = entropies };
    }

    private static (int action, float logProb, float entropy) SampleDiscreteActionFromLogits(float[] logits, Random rng)
    {
        var probs  = Softmax(logits);
        var action = SampleFromProbs(probs, rng);
        var logProb = MathF.Log(Math.Max(probs[action], 1e-8f));
        var entropy = 0f;
        foreach (var p in probs)
        {
            if (p > 1e-8f) entropy -= p * MathF.Log(p);
        }

        return (action, logProb, entropy);
    }

    public int GreedyDiscreteAction(float[] obs)
    {
        var logits = ForwardActor(obs);
        var best = 0;
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best]) best = i;
        }

        return best;
    }

    public (float[] probs, float[] logProbs) GetDiscreteActorProbs(float[] obs)
    {
        var logits = ForwardActor(obs);
        var probs  = Softmax(logits);
        var logProbs = new float[probs.Length];
        for (var i = 0; i < probs.Length; i++)
            logProbs[i] = probs[i] > 1e-8f ? MathF.Log(probs[i]) : MathF.Log(1e-8f);
        return (probs, logProbs);
    }

    public (float[] q1, float[] q2) GetDiscreteQValues(float[] obs)
        => (ForwardQ(_q1Trunk, _q1Head, obs), ForwardQ(_q2Trunk, _q2Head, obs));

    public (float[] q1Target, float[] q2Target) GetDiscreteTargetQValues(float[] obs)
        => (ForwardQ(_q1TargetTrunk, _q1TargetHead, obs), ForwardQ(_q2TargetTrunk, _q2TargetHead, obs));

    // ── Continuous action methods ────────────────────────────────────────────

    public (float[] action, float logProb, float[] eps, float[] u) SampleContinuousAction(float[] obs, Random rng)
        => SampleContinuousActionFromActorOutput(ForwardActor(obs), rng);

    public ContinuousActionBatch SampleContinuousActions(VectorBatch observations, Random rng)
    {
        var actorBatch = ForwardActorBatch(observations);
        var actions    = new float[observations.BatchSize][];
        var logProbs   = new float[observations.BatchSize];
        var epsilons   = new float[observations.BatchSize][];
        var preSquash  = new float[observations.BatchSize][];
        for (var b = 0; b < observations.BatchSize; b++)
        {
            var (action, logProb, eps, u) = SampleContinuousActionFromActorOutput(actorBatch.CopyRow(b), rng);
            actions[b]   = action;
            logProbs[b]  = logProb;
            epsilons[b]  = eps;
            preSquash[b] = u;
        }

        return new ContinuousActionBatch
        {
            Actions = actions, LogProbabilities = logProbs,
            Epsilons = epsilons, PreSquashActions = preSquash,
        };
    }

    private (float[] action, float logProb, float[] eps, float[] u) SampleContinuousActionFromActorOutput(float[] actorOut, Random rng)
    {
        var mean   = actorOut[.._actionDim];
        var logStd = new float[_actionDim];
        for (var i = 0; i < _actionDim; i++)
            logStd[i] = Math.Clamp(actorOut[_actionDim + i], -20f, 2f);

        var eps     = new float[_actionDim];
        var u       = new float[_actionDim];
        var action  = new float[_actionDim];
        var logProb = 0f;
        for (var i = 0; i < _actionDim; i++)
        {
            eps[i] = SampleNormal(rng);
            var std = MathF.Exp(logStd[i]);
            u[i]   = mean[i] + std * eps[i];
            action[i] = MathF.Tanh(u[i]);
            var squash = 1f - action[i] * action[i];
            logProb += -0.5f * eps[i] * eps[i] - logStd[i] - MathF.Log(squash + 1e-6f);
        }

        logProb -= _actionDim * 0.5f * MathF.Log(2f * MathF.PI);
        return (action, logProb, eps, u);
    }

    public float[] DeterministicContinuousAction(float[] obs)
    {
        var actorOut = ForwardActor(obs);
        var action = new float[_actionDim];
        for (var i = 0; i < _actionDim; i++) action[i] = MathF.Tanh(actorOut[i]);
        return action;
    }

    public (float q1, float q2) GetContinuousQValues(float[] obs, float[] action)
    {
        var input = Concat(obs, action);
        return (ForwardQ(_q1Trunk, _q1Head, input)[0], ForwardQ(_q2Trunk, _q2Head, input)[0]);
    }

    public (float q1Target, float q2Target) GetContinuousTargetQValues(float[] obs, float[] action)
    {
        var input = Concat(obs, action);
        return (ForwardQ(_q1TargetTrunk, _q1TargetHead, input)[0], ForwardQ(_q2TargetTrunk, _q2TargetHead, input)[0]);
    }

    // ── Gradient updates ─────────────────────────────────────────────────────

    // ── Batch Q update — discrete ─────────────────────────────────────────────

    /// <summary>
    /// Performs a single Adam step on Q1 using the provided (obs, action, target) batch.
    /// One gradient accumulation pass over all samples, then one weight update.
    /// </summary>
    public void BatchUpdateQ1Discrete(
        IReadOnlyList<(float[] obs, int action, float target)> samples, float learningRate)
        => BatchUpdateQNetworkDiscrete(_q1Trunk, _q1Head, samples, learningRate);

    /// <summary>Performs a single Adam step on Q2 using the provided (obs, action, target) batch.</summary>
    public void BatchUpdateQ2Discrete(
        IReadOnlyList<(float[] obs, int action, float target)> samples, float learningRate)
        => BatchUpdateQNetworkDiscrete(_q2Trunk, _q2Head, samples, learningRate);

    private void BatchUpdateQNetworkDiscrete(
        NetworkLayer[] trunk, NetworkLayer head,
        IReadOnlyList<(float[] obs, int action, float target)> samples,
        float learningRate)
    {
        var n = samples.Count;
        if (n == 0) return;

        var trunkBufs = new GradientBuffer[trunk.Length];
        for (var i = 0; i < trunk.Length; i++) trunkBufs[i] = trunk[i].CreateGradientBuffer();
        var headBuf = head.CreateGradientBuffer();

        foreach (var (obs, action, target) in samples)
        {
            var q        = ForwardQ(trunk, head, obs);   // caches trunk/head state
            var outGrad  = new float[q.Length];
            outGrad[action] = q[action] - target;

            var grad = head.AccumulateGradients(outGrad, headBuf);
            for (var i = trunk.Length - 1; i >= 0; i--)
                grad = trunk[i].AccumulateGradients(grad, trunkBufs[i]);
        }

        var scale = 1f / n;
        head.ApplyGradients(headBuf, learningRate, scale);
        for (var i = 0; i < trunk.Length; i++)
            trunk[i].ApplyGradients(trunkBufs[i], learningRate, scale);
    }

    /// <summary>Performs a single Adam step on the actor using a batch of observations.</summary>
    public void BatchUpdateActorDiscrete(
        IReadOnlyList<float[]> observations, float alpha, float learningRate)
    {
        var n = observations.Count;
        if (n == 0) return;

        var trunkBufs = new GradientBuffer[_actorTrunk.Length];
        for (var i = 0; i < _actorTrunk.Length; i++) trunkBufs[i] = _actorTrunk[i].CreateGradientBuffer();
        var headBuf = _actorHead.CreateGradientBuffer();

        foreach (var obs in observations)
        {
            // ForwardActorFull caches actor trunk/head state for AccumulateGradients.
            // GetDiscreteQValues forwards through _q1Trunk/_q2Trunk (separate layer instances —
            // does NOT overwrite the actor cache).
            var logits = ForwardActorFull(obs);
            var probs  = Softmax(logits);
            var (q1, q2) = GetDiscreteQValues(obs);

            var logProbClipped = new float[probs.Length];
            for (var i = 0; i < probs.Length; i++)
                logProbClipped[i] = probs[i] > 1e-8f ? MathF.Log(probs[i]) : MathF.Log(1e-8f);

            var f    = new float[probs.Length];
            var eFpi = 0f;
            for (var i = 0; i < probs.Length; i++)
            {
                f[i]  = Math.Min(q1[i], q2[i]) - alpha * logProbClipped[i];
                eFpi += probs[i] * f[i];
            }

            var grad = new float[probs.Length];
            for (var j = 0; j < probs.Length; j++)
                grad[j] = -probs[j] * (f[j] - eFpi);

            var inputGrad = _actorHead.AccumulateGradients(grad, headBuf);
            for (var i = _actorTrunk.Length - 1; i >= 0; i--)
                inputGrad = _actorTrunk[i].AccumulateGradients(inputGrad, trunkBufs[i]);
        }

        var scale = 1f / n;
        _actorHead.ApplyGradients(headBuf, learningRate, scale);
        for (var i = 0; i < _actorTrunk.Length; i++)
            _actorTrunk[i].ApplyGradients(trunkBufs[i], learningRate, scale);
    }

    // ── Batch Q update — continuous ───────────────────────────────────────────

    /// <summary>Performs a single Adam step on Q1 using the provided (obs, action, target) batch.</summary>
    public void BatchUpdateQ1Continuous(
        IReadOnlyList<(float[] obs, float[] action, float target)> samples, float learningRate)
        => BatchUpdateQNetworkContinuous(_q1Trunk, _q1Head, samples, learningRate);

    /// <summary>Performs a single Adam step on Q2 using the provided (obs, action, target) batch.</summary>
    public void BatchUpdateQ2Continuous(
        IReadOnlyList<(float[] obs, float[] action, float target)> samples, float learningRate)
        => BatchUpdateQNetworkContinuous(_q2Trunk, _q2Head, samples, learningRate);

    private void BatchUpdateQNetworkContinuous(
        NetworkLayer[] trunk, NetworkLayer head,
        IReadOnlyList<(float[] obs, float[] action, float target)> samples,
        float learningRate)
    {
        var n = samples.Count;
        if (n == 0) return;

        var trunkBufs = new GradientBuffer[trunk.Length];
        for (var i = 0; i < trunk.Length; i++) trunkBufs[i] = trunk[i].CreateGradientBuffer();
        var headBuf = head.CreateGradientBuffer();

        foreach (var (obs, action, target) in samples)
        {
            var input   = Concat(obs, action);
            var q       = ForwardQ(trunk, head, input);
            var outGrad = new[] { q[0] - target };

            var grad = head.AccumulateGradients(outGrad, headBuf);
            for (var i = trunk.Length - 1; i >= 0; i--)
                grad = trunk[i].AccumulateGradients(grad, trunkBufs[i]);
        }

        var scale = 1f / n;
        head.ApplyGradients(headBuf, learningRate, scale);
        for (var i = 0; i < trunk.Length; i++)
            trunk[i].ApplyGradients(trunkBufs[i], learningRate, scale);
    }

    /// <summary>
    /// Performs a single Adam step on the actor for continuous actions.
    /// Each sample must supply a freshly-reparameterized action (from SampleContinuousAction).
    /// </summary>
    public void BatchUpdateActorContinuous(
        IReadOnlyList<(float[] obs, float[] action, float[] eps, float[] u)> samples,
        float alpha, float learningRate)
    {
        var n = samples.Count;
        if (n == 0) return;

        var trunkBufs = new GradientBuffer[_actorTrunk.Length];
        for (var i = 0; i < _actorTrunk.Length; i++) trunkBufs[i] = _actorTrunk[i].CreateGradientBuffer();
        var headBuf = _actorHead.CreateGradientBuffer();

        foreach (var (obs, action, eps, u) in samples)
        {
            var input = Concat(obs, action);

            // Forward Q1 and Q2 (caches their states; actor cache is separate).
            var q1 = ForwardQ(_q1Trunk, _q1Head, input)[0];
            var q2 = ForwardQ(_q2Trunk, _q2Head, input)[0];

            // dQ/da from the more pessimistic Q network (cache still fresh).
            var qInputGrad = q1 <= q2
                ? ComputeQInputGradOnly(_q1Trunk, _q1Head, new[] { 1f })
                : ComputeQInputGradOnly(_q2Trunk, _q2Head, new[] { 1f });
            var dQdAction = qInputGrad[_obsSize..];

            // ForwardActorFull caches actor state for AccumulateGradients.
            var actorOut  = ForwardActorFull(obs);
            var actorGrad = new float[_actionDim * 2];

            for (var d = 0; d < _actionDim; d++)
            {
                var a      = action[d];
                var squash = 1f - a * a;
                var logStd = Math.Clamp(actorOut[_actionDim + d], -20f, 2f);
                var std    = MathF.Exp(logStd);

                actorGrad[d] = squash * (-dQdAction[d] + 2f * alpha * a / (squash + 1e-6f));
                actorGrad[_actionDim + d] = -dQdAction[d] * squash * std * eps[d]
                    - alpha * (1f - 2f * a * squash * std * eps[d] / (squash + 1e-6f));
            }

            var inputGrad = _actorHead.AccumulateGradients(actorGrad, headBuf);
            for (var i = _actorTrunk.Length - 1; i >= 0; i--)
                inputGrad = _actorTrunk[i].AccumulateGradients(inputGrad, trunkBufs[i]);
        }

        var scale = 1f / n;
        _actorHead.ApplyGradients(headBuf, learningRate, scale);
        for (var i = 0; i < _actorTrunk.Length; i++)
            _actorTrunk[i].ApplyGradients(trunkBufs[i], learningRate, scale);
    }

    public void SoftUpdateTargets(float tau)
    {
        for (var i = 0; i < _q1Trunk.Length; i++)
        {
            _q1TargetTrunk[i].SoftUpdateFrom(_q1Trunk[i], tau);
            _q2TargetTrunk[i].SoftUpdateFrom(_q2Trunk[i], tau);
        }

        _q1TargetHead.SoftUpdateFrom(_q1Head, tau);
        _q2TargetHead.SoftUpdateFrom(_q2Head, tau);
    }

    // ── Checkpoint ───────────────────────────────────────────────────────────

    /// <summary>Copies all sub-network weights from this into <paramref name="other"/> (identical architecture required).</summary>
    internal void CopyWeightsTo(SacNetwork other)
    {
        for (var i = 0; i < _actorTrunk.Length; i++) other._actorTrunk[i].CopyFrom(_actorTrunk[i]);
        other._actorHead.CopyFrom(_actorHead);
        for (var i = 0; i < _q1Trunk.Length; i++) other._q1Trunk[i].CopyFrom(_q1Trunk[i]);
        other._q1Head.CopyFrom(_q1Head);
        for (var i = 0; i < _q2Trunk.Length; i++) other._q2Trunk[i].CopyFrom(_q2Trunk[i]);
        other._q2Head.CopyFrom(_q2Head);
        for (var i = 0; i < _q1TargetTrunk.Length; i++) other._q1TargetTrunk[i].CopyFrom(_q1TargetTrunk[i]);
        other._q1TargetHead.CopyFrom(_q1TargetHead);
        for (var i = 0; i < _q2TargetTrunk.Length; i++) other._q2TargetTrunk[i].CopyFrom(_q2TargetTrunk[i]);
        other._q2TargetHead.CopyFrom(_q2TargetHead);
    }

    /// <summary>Overwrites this network's weights from <paramref name="other"/> (identical architecture required).</summary>
    internal void LoadWeightsFrom(SacNetwork other) => other.CopyWeightsTo(this);

    public RLCheckpoint SaveCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        var weights = new List<float>();
        var shapes  = new List<int>();

        foreach (var layer in _actorTrunk) layer.AppendSerialized(weights, shapes);
        _actorHead.AppendSerialized(weights, shapes);
        foreach (var layer in _q1Trunk) layer.AppendSerialized(weights, shapes);
        _q1Head.AppendSerialized(weights, shapes);
        foreach (var layer in _q2Trunk) layer.AppendSerialized(weights, shapes);
        _q2Head.AppendSerialized(weights, shapes);

        return new RLCheckpoint
        {
            RunId        = groupId,
            TotalSteps   = totalSteps,
            EpisodeCount = episodeCount,
            UpdateCount  = updateCount,
            WeightBuffer = weights.ToArray(),
            LayerShapeBuffer = shapes.ToArray(),
        };
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var wi       = 0;
        var si       = 0;
        var isLegacy = checkpoint.FormatVersion < RLCheckpoint.CurrentFormatVersion;

        foreach (var layer in _actorTrunk) layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        _actorHead.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        foreach (var layer in _q1Trunk) layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        _q1Head.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        foreach (var layer in _q2Trunk) layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        _q2Head.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);

        HardCopyToTargets();
    }

    public void LoadActorCheckpoint(RLCheckpoint checkpoint)
    {
        var wi       = 0;
        var si       = 0;
        var isLegacy = checkpoint.FormatVersion < RLCheckpoint.CurrentFormatVersion;

        foreach (var layer in _actorTrunk)
            layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        _actorHead.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>Inference-only forward through the actor. Caches state for BackwardActorFromLogits.</summary>
    private float[] ForwardActor(float[] obs)
    {
        var x = obs;
        foreach (var layer in _actorTrunk) x = layer.Forward(x);
        return _actorHead.Forward(x);
    }

    private VectorBatch ForwardActorBatch(VectorBatch observations)
    {
        var x = observations;
        foreach (var layer in _actorTrunk) x = layer.ForwardBatch(x);
        return _actorHead.ForwardBatch(x);
    }

    /// <summary>
    /// Forwards through the actor (same as ForwardActor), caching state in each layer.
    /// Call BackwardActorFromLogits immediately after to apply the gradient.
    /// </summary>
    private float[] ForwardActorFull(float[] obs)
    {
        var x = obs;
        foreach (var layer in _actorTrunk) x = layer.Forward(x);
        return _actorHead.Forward(x);
    }

    /// <summary>Forwards through a Q network, caching state in each layer.</summary>
    private static float[] ForwardQ(NetworkLayer[] trunk, NetworkLayer head, float[] input)
    {
        var x = input;
        foreach (var layer in trunk) x = layer.Forward(x);
        return head.Forward(x);
    }

    /// <summary>
    /// Propagates the output gradient backward through a Q network WITHOUT updating any weights.
    /// Used for actor reparameterization (dQ/da). Forward must have already been called on the
    /// same trunk/head pair to populate the per-layer cache.
    /// </summary>
    private static float[] ComputeQInputGradOnly(NetworkLayer[] trunk, NetworkLayer head, float[] outputGrad)
    {
        var grad = head.ComputeInputGrad(outputGrad);
        for (var i = trunk.Length - 1; i >= 0; i--)
            grad = trunk[i].ComputeInputGrad(grad);
        return grad;
    }

    private void HardCopyToTargets()
    {
        for (var i = 0; i < _q1Trunk.Length; i++)
        {
            _q1TargetTrunk[i].CopyFrom(_q1Trunk[i]);
            _q2TargetTrunk[i].CopyFrom(_q2Trunk[i]);
        }

        _q1TargetHead.CopyFrom(_q1Head);
        _q2TargetHead.CopyFrom(_q2Head);
    }

    private static float[] Softmax(float[] logits)
    {
        var max   = logits.Max();
        var probs = new float[logits.Length];
        var total = 0f;
        for (var i = 0; i < logits.Length; i++) { probs[i] = MathF.Exp(logits[i] - max); total += probs[i]; }
        for (var i = 0; i < probs.Length; i++) probs[i] /= total;
        return probs;
    }

    private static int SampleFromProbs(float[] probs, Random rng)
    {
        var roll = rng.NextSingle();
        var cum  = 0f;
        for (var i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (roll <= cum) return i;
        }

        return probs.Length - 1;
    }

    private static float SampleNormal(Random rng)
    {
        var u1 = Math.Max(rng.NextSingle(), 1e-10f);
        var u2 = rng.NextSingle();
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }

    private static float[] Concat(float[] a, float[] b)
    {
        var result = new float[a.Length + b.Length];
        Array.Copy(a, 0, result, 0, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }

    public sealed class DiscreteActionBatch
    {
        public int[] Actions { get; init; } = Array.Empty<int>();
        public float[] LogProbabilities { get; init; } = Array.Empty<float>();
        public float[] Entropies { get; init; } = Array.Empty<float>();
    }

    public sealed class ContinuousActionBatch
    {
        public float[][] Actions { get; init; } = Array.Empty<float[]>();
        public float[] LogProbabilities { get; init; } = Array.Empty<float>();
        public float[][] Epsilons { get; init; } = Array.Empty<float[]>();
        public float[][] PreSquashActions { get; init; } = Array.Empty<float[]>();
    }
}
