using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Dyna-Q world model: learns to predict (nextObservation, reward) from (observation, action).
/// Used by <see cref="DqnTrainer"/> to generate imagined transitions for additional Q-learning updates.
///
/// Architecture: a small MLP mapping [obs; action_onehot] → [nextObs; reward].
/// Trained with MSE loss after each real environment step.
/// </summary>
internal sealed class DynaWorldModel
{
    private readonly int _obsSize;
    private readonly int _actionCount;
    private readonly NetworkLayer[] _trunk;
    private readonly DenseLayer _head;   // outputs: [nextObs (obsSize) | reward (1)]

    private const int HiddenSize = 64;

    public DynaWorldModel(int obsSize, int actionCount)
    {
        _obsSize     = obsSize;
        _actionCount = actionCount;

        var inputSize  = obsSize + actionCount;  // obs concatenated with one-hot action
        var outputSize = obsSize + 1;            // predicted nextObs + predicted reward

        // Two hidden layers with ReLU.
        _trunk = new NetworkLayer[]
        {
            new DenseLayer(inputSize,  HiddenSize, RLActivationKind.Relu, RLOptimizerKind.Adam),
            new DenseLayer(HiddenSize, HiddenSize, RLActivationKind.Relu, RLOptimizerKind.Adam),
        };
        _head = new DenseLayer(HiddenSize, outputSize, null, RLOptimizerKind.Adam);
    }

    /// <summary>Predicts (nextObservation, reward) for a given (observation, action) pair.</summary>
    public (float[] nextObs, float reward) Predict(float[] obs, int action)
    {
        var input  = BuildInput(obs, action);
        var output = RunForward(input);
        var nextObs = output[..^1];     // first obsSize values
        var reward  = output[^1];       // last value is reward
        return (nextObs, reward);
    }

    /// <summary>
    /// Trains the model on a batch of real transitions (one gradient step per call).
    /// </summary>
    public void TrainBatch(Transition[] batch, float learningRate)
    {
        var n = batch.Length;
        if (n == 0) return;

        var trunkBufs = new GradientBuffer[_trunk.Length];
        for (var i = 0; i < _trunk.Length; i++) trunkBufs[i] = _trunk[i].CreateGradientBuffer();
        var headBuf = _head.CreateGradientBuffer();

        var outputSize = _obsSize + 1;

        foreach (var t in batch)
        {
            var input  = BuildInput(t.Observation, t.DiscreteAction);
            var output = RunForwardTraining(input);

            // Target: actual nextObs + actual reward.
            var outGrad = new float[outputSize];
            for (var j = 0; j < _obsSize; j++)
                outGrad[j] = (output[j] - t.NextObservation[j]) / n;
            outGrad[_obsSize] = (output[_obsSize] - t.Reward) / n;

            var grad = _head.AccumulateGradients(outGrad, headBuf);
            for (var li = _trunk.Length - 1; li >= 0; li--)
                grad = _trunk[li].AccumulateGradients(grad, trunkBufs[li]);
        }

        _head.ApplyGradients(headBuf, learningRate, 1f);
        for (var i = 0; i < _trunk.Length; i++)
            _trunk[i].ApplyGradients(trunkBufs[i], learningRate, 1f);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private float[] BuildInput(float[] obs, int action)
    {
        var input = new float[_obsSize + _actionCount];
        Array.Copy(obs, input, _obsSize);
        if (action >= 0 && action < _actionCount)
            input[_obsSize + action] = 1f;
        return input;
    }

    private float[] RunForward(float[] input)
    {
        var x = input;
        foreach (var layer in _trunk) x = layer.Forward(x);
        return _head.Forward(x);
    }

    private float[] RunForwardTraining(float[] input)
    {
        var x = input;
        foreach (var layer in _trunk) x = layer.Forward(x, isTraining: true);
        return _head.Forward(x, isTraining: true);
    }
}
