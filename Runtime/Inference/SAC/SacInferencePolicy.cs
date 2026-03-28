using System;

namespace RlAgentPlugin.Runtime;

public sealed class SacInferencePolicy : IInferencePolicy
{
    private readonly SacNetwork _network;
    private readonly bool _stochastic;
    private readonly Random _rng = new();

    public SacInferencePolicy(int observationSize, int actionDimensions, bool isContinuous, RLNetworkGraph graph, bool stochastic = false)
    {
        if (actionDimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(actionDimensions), "SAC inference requires at least one action dimension.");

        if (!isContinuous)
            throw new InvalidOperationException(
                "SAC inference does not support discrete action spaces. Convert the action space to continuous-only actions.");

        _stochastic = stochastic;
        _network = new SacNetwork(observationSize, actionDimensions, isContinuous, graph, 0f);
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        _network.LoadActorCheckpoint(checkpoint);
    }

    public PolicyDecision Predict(float[] observation)
    {
        var actions = _stochastic
            ? _network.SampleContinuousAction(observation, _rng).action
            : _network.DeterministicContinuousAction(observation);
        return new PolicyDecision { ContinuousActions = actions };
    }
}
