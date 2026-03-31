using System;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Inference policy for DQN checkpoints.
/// Greedy (deterministic): always selects the action with the highest Q-value.
/// Stochastic: uses epsilon-greedy with a fixed epsilon.
/// </summary>
public sealed class DqnInferencePolicy : IInferencePolicy
{
    private readonly DqnNetwork _network;
    private readonly bool _stochastic;
    private readonly Random _rng;

    /// <param name="observationSize">Flat observation vector length.</param>
    /// <param name="actionCount">Number of discrete actions.</param>
    /// <param name="graph">Network architecture.</param>
    /// <param name="stochastic">If true, uses epsilon=0.05 greedy; otherwise fully greedy.</param>
    public DqnInferencePolicy(int observationSize, int actionCount, RLNetworkGraph graph, bool stochastic = false)
    {
        _network    = new DqnNetwork(observationSize, actionCount, graph);
        _stochastic = stochastic;
        _rng        = new Random();
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        _network.LoadCheckpoint(checkpoint);
    }

    public PolicyDecision Predict(float[] observation)
    {
        // Epsilon-greedy with ε=0.05 during stochastic inference; purely greedy otherwise.
        if (_stochastic && _rng.NextDouble() < 0.05)
        {
            var qAll = _network.GetOnlineQValues(observation);
            return new PolicyDecision { DiscreteAction = _rng.Next(qAll.Length) };
        }

        var qValues = _network.GetOnlineQValues(observation);
        return new PolicyDecision { DiscreteAction = ArgMax(qValues) };
    }

    private static int ArgMax(float[] values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
