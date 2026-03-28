using System;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// PPO inference policy for agents that use multi-stream or image observations.
/// Uses the spec-aware <see cref="PolicyValueNetwork"/> constructor to route each
/// observation stream through its dedicated encoder before the shared trunk.
/// </summary>
public sealed class PpoInferencePolicyMultiStream : IInferencePolicy
{
    private readonly PolicyValueNetwork _network;
    private readonly int _continuousActionDims;
    private readonly bool _stochastic;
    private readonly Random _rng = new();

    public PpoInferencePolicyMultiStream(
        ObservationSpec spec,
        int actionCount,
        int continuousActionDims,
        RLNetworkGraph graph,
        bool stochastic = false)
    {
        if (actionCount <= 0 && continuousActionDims <= 0)
            throw new ArgumentException("PPO inference requires at least one discrete or continuous action dimension.");

        _continuousActionDims = continuousActionDims;
        _stochastic = stochastic;
        _network = new PolicyValueNetwork(spec, actionCount, continuousActionDims, graph);
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
