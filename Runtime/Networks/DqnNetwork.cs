using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// DQN neural network: an online Q-network and a periodically hard-copied target Q-network.
/// Accepts flat observations and outputs one Q-value per discrete action.
/// </summary>
internal sealed class DqnNetwork
{
    private readonly int _obsSize;
    private readonly int _actionCount;

    // Online Q-network — trained every update step.
    private readonly NetworkLayer[] _onlineTrunk;
    private readonly NetworkLayer _onlineHead;

    // Target Q-network — weights are frozen and hard-copied from online periodically.
    private readonly NetworkLayer[] _targetTrunk;
    private readonly NetworkLayer _targetHead;

    public DqnNetwork(int obsSize, int actionCount, RLNetworkGraph graph)
    {
        _obsSize     = obsSize;
        _actionCount = actionCount;
        var useNativeLayers = graph.ResolveNativeLayerBackend();

        var trunkOut = graph.OutputSize(obsSize);

        _onlineTrunk = graph.BuildTrunkLayers(obsSize, null, useNativeLayers);
        _onlineHead = useNativeLayers
            ? new NativeDenseLayer(trunkOut, actionCount, null, graph.Optimizer)
            : new DenseLayer(trunkOut, actionCount, null, graph.Optimizer);

        _targetTrunk = graph.BuildTrunkLayers(obsSize, RLOptimizerKind.None, useNativeLayers);
        _targetHead = useNativeLayers
            ? new NativeDenseLayer(trunkOut, actionCount, null, RLOptimizerKind.None)
            : new DenseLayer(trunkOut, actionCount, null, RLOptimizerKind.None);

        // Target starts as a copy of online.
        HardCopyToTarget();
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// <summary>Returns Q-values from the online network (used for action selection and DDQN).</summary>
    public float[] GetOnlineQValues(float[] obs) => ForwardQ(_onlineTrunk, (DenseLayer)_onlineHead, obs);

    /// <summary>Returns Q-values from the frozen target network (used for Bellman targets).</summary>
    public float[] GetTargetQValues(float[] obs) => ForwardQ(_targetTrunk, (DenseLayer)_targetHead, obs);

    // ── Target network management ────────────────────────────────────────────

    /// <summary>Hard-copies all online weights to the target network.</summary>
    public void HardCopyToTarget()
    {
        for (var i = 0; i < _onlineTrunk.Length; i++)
            _targetTrunk[i].CopyFrom(_onlineTrunk[i]);
        _targetHead.CopyFrom(_onlineHead);
    }

    // ── Training ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs one gradient update step on the online Q-network using a batch of transitions.
    /// Returns the mean TD error (mean squared Bellman residual across the batch).
    /// </summary>
    /// <param name="batch">Sampled transitions.</param>
    /// <param name="gamma">Discount factor.</param>
    /// <param name="learningRate">Optimizer learning rate.</param>
    /// <param name="maxGradNorm">Gradient norm clip (0 = disabled).</param>
    /// <param name="useDoubleDqn">True for Double DQN target computation.</param>
    public float TrainBatch(
        Transition[] batch,
        float gamma,
        float learningRate,
        float maxGradNorm,
        bool useDoubleDqn)
    {
        var n = batch.Length;
        if (n == 0) return 0f;

        // Pre-compute Bellman targets (no gradient through target network).
        var targets = new float[n];
        for (var i = 0; i < n; i++)
        {
            var t = batch[i];
            var targetQ = GetTargetQValues(t.NextObservation);
            float maxFutureQ;
            if (useDoubleDqn)
            {
                // Double DQN: action selection by online, evaluation by target.
                var onlineNextQ = GetOnlineQValues(t.NextObservation);
                maxFutureQ = targetQ[ArgMax(onlineNextQ)];
            }
            else
            {
                maxFutureQ = targetQ[ArgMax(targetQ)];
            }
            targets[i] = t.Reward + (t.Done ? 0f : gamma * maxFutureQ);
        }

        // Build gradient buffers for the online network.
        var trunkBufs = new GradientBuffer[_onlineTrunk.Length];
        for (var i = 0; i < _onlineTrunk.Length; i++)
            trunkBufs[i] = _onlineTrunk[i].CreateGradientBuffer();
        var headBuf = _onlineHead.CreateGradientBuffer();

        var totalLoss = 0f;

        for (var i = 0; i < n; i++)
        {
            var t = batch[i];
            // Forward the online network (caches state for backprop).
            var trunkOut = RunTrunk(_onlineTrunk, t.Observation, isTraining: true);
            var qValues  = _onlineHead.Forward(trunkOut, isTraining: true);

            // Huber-style gradient: clip large errors to ±1 (robust to outliers).
            var error = qValues[t.DiscreteAction] - targets[i];
            var clipped = Math.Clamp(error, -1f, 1f);
            totalLoss += 0.5f * error * error;

            // Gradient is non-zero only at the taken action (1/n for mean over batch).
            var outGrad = new float[_actionCount];
            outGrad[t.DiscreteAction] = clipped / n;

            var grad = _onlineHead.AccumulateGradients(outGrad, headBuf);
            for (var li = _onlineTrunk.Length - 1; li >= 0; li--)
                grad = _onlineTrunk[li].AccumulateGradients(grad, trunkBufs[li]);
        }

        // Gradient norm clipping.
        float gradScale = 1f;
        if (maxGradNorm > 0f)
        {
            var normSq = headBuf.SumSquares();
            foreach (var buf in trunkBufs) normSq += buf.SumSquares();
            var norm = MathF.Sqrt(normSq);
            if (norm > maxGradNorm) gradScale = maxGradNorm / norm;
        }

        _onlineHead.ApplyGradients(headBuf, learningRate, gradScale);
        for (var i = 0; i < _onlineTrunk.Length; i++)
            _onlineTrunk[i].ApplyGradients(trunkBufs[i], learningRate, gradScale);

        return totalLoss / n;
    }

    // ── Serialization ────────────────────────────────────────────────────────

    public RLCheckpoint SaveCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        var weights = new List<float>();
        var shapes  = new List<int>();

        foreach (var layer in _onlineTrunk) layer.AppendSerialized(weights, shapes);
        _onlineHead.AppendSerialized(weights, shapes);

        return new RLCheckpoint
        {
            RunId            = groupId,
            TotalSteps       = totalSteps,
            EpisodeCount     = episodeCount,
            UpdateCount      = updateCount,
            WeightBuffer     = weights.ToArray(),
            LayerShapeBuffer = shapes.ToArray(),
        };
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var wi = 0;
        var si = 0;

        foreach (var layer in _onlineTrunk)
            layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);
        _onlineHead.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);

        // Keep target network in sync after loading.
        HardCopyToTarget();
    }

    // Clone is not exposed; inference snapshots are created through DqnInferencePolicy + LoadCheckpoint.

    // ── Private helpers ──────────────────────────────────────────────────────

    private static float[] RunTrunk(NetworkLayer[] trunk, float[] input, bool isTraining)
    {
        var x = input;
        foreach (var layer in trunk)
            x = layer.Forward(x, isTraining);
        return x;
    }

    private static float[] ForwardQ(NetworkLayer[] trunk, DenseLayer head, float[] obs)
    {
        var x = obs;
        foreach (var layer in trunk)
            x = layer.Forward(x);
        return head.Forward(x);
    }

    private static int ArgMax(float[] values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
