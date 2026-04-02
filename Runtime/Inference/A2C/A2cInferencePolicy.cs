using System;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Inference policy for A2C checkpoints.
/// Shares the same network architecture as PPO (<see cref="PolicyValueNetwork"/>), so
/// A2C checkpoints can be loaded by <see cref="PpoInferencePolicy"/> directly — this class
/// is a thin alias that makes the algorithm name explicit.
/// </summary>
public sealed class A2cInferencePolicy : IInferencePolicy
{
    private readonly PolicyValueNetwork _network;
    private readonly int _continuousActionDims;
    private readonly bool _stochastic;
    private readonly Random _rng = new();

    public A2cInferencePolicy(
        int observationSize,
        int actionCount,
        int continuousActionDims,
        RLNetworkGraph graph,
        bool stochastic = false)
    {
        if (actionCount <= 0 && continuousActionDims <= 0)
            throw new ArgumentException("A2C inference requires at least one action dimension.");

        _continuousActionDims = continuousActionDims;
        _stochastic           = stochastic;
        _network              = new PolicyValueNetwork(observationSize, actionCount, continuousActionDims, graph);
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        _network.LoadCheckpoint(checkpoint);
    }

    public RecurrentState? CreateZeroRecurrentState()
        => _network.HasRecurrentTrunk ? _network.CreateZeroRecurrentState() : null;

    public PolicyDecision Predict(float[] observation)
    {
        if (_continuousActionDims > 0)
        {
            return new PolicyDecision
            {
                ContinuousActions = _stochastic
                    ? _network.SelectStochasticContinuousAction(observation, _rng)
                    : _network.SelectDeterministicContinuousAction(observation),
            };
        }

        return new PolicyDecision
        {
            DiscreteAction = _stochastic
                ? _network.SelectStochasticAction(observation, _rng)
                : _network.SelectGreedyAction(observation),
        };
    }

    public PolicyDecision PredictRecurrent(float[] observation, RecurrentState state)
    {
        var inference = _network.InferRecurrent(observation, state);
        if (_continuousActionDims > 0)
        {
            return new PolicyDecision
            {
                ContinuousActions = _stochastic
                    ? SampleStochasticContinuousAction(inference.Logits)
                    : SelectDeterministicContinuousAction(inference.Logits),
                RecurrentState = inference.State,
            };
        }

        return new PolicyDecision
        {
            DiscreteAction = _stochastic
                ? SelectStochasticAction(inference.Logits)
                : SelectGreedyAction(inference.Logits),
            RecurrentState = inference.State,
        };
    }

    private float[] SelectDeterministicContinuousAction(float[] actorOut)
    {
        var action = new float[_continuousActionDims];
        for (var i = 0; i < _continuousActionDims; i++)
            action[i] = MathF.Tanh(actorOut[i]);
        return action;
    }

    private float[] SampleStochasticContinuousAction(float[] actorOut)
    {
        var action = new float[_continuousActionDims];
        for (var i = 0; i < _continuousActionDims; i++)
        {
            var mean   = actorOut[i];
            var logStd = Math.Clamp(actorOut[_continuousActionDims + i], -20f, 2f);
            var std    = MathF.Exp(logStd);
            var u1     = Math.Max(_rng.NextSingle(), 1e-10f);
            var u2     = _rng.NextSingle();
            var eps    = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
            action[i]  = MathF.Tanh(mean + std * eps);
        }
        return action;
    }

    private int SelectGreedyAction(float[] logits)
    {
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

    private int SelectStochasticAction(float[] logits)
    {
        var probs = Softmax(logits);
        var roll = _rng.NextSingle();
        var cum = 0f;
        for (var i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (roll <= cum) return i;
        }
        return probs.Length - 1;
    }

    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var probabilities = new float[logits.Length];
        var total = 0.0f;
        for (var index = 0; index < logits.Length; index++)
        {
            probabilities[index] = MathF.Exp(logits[index] - maxLogit);
            total += probabilities[index];
        }

        for (var index = 0; index < probabilities.Length; index++)
            probabilities[index] /= total;

        return probabilities;
    }
}
