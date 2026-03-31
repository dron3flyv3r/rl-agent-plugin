using System;

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
}
