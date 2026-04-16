using System;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Inference-mode policy for NEAT checkpoints. Wraps a single champion
/// <see cref="GenomeNetwork"/> and implements <see cref="IInferencePolicy"/>.
///
/// Created by <see cref="NeatCheckpointSerializer.PolicyFromCheckpoint"/> or
/// by <see cref="NeatTrainer.SnapshotPolicyForEval"/>.
/// </summary>
public sealed class NeatInferencePolicy : IInferencePolicy
{
    private GenomeNetwork _network;
    private readonly int _actionCount;
    private readonly bool _isDiscrete;
    private readonly Random _rng = new();

    internal NeatInferencePolicy(NeatGenome genome, int actionCount, bool isDiscrete)
    {
        _network     = new GenomeNetwork(genome, ComputeInputCount(genome), actionCount);
        _actionCount = actionCount;
        _isDiscrete  = isDiscrete;
    }

    // ── IInferencePolicy ────────────────────────────────────────────────────

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var genome = NeatCheckpointSerializer.DeserializeGenome(checkpoint);
        bool isDiscrete = checkpoint.DiscreteActionCount > 0;
        int actionCount = isDiscrete ? checkpoint.DiscreteActionCount : checkpoint.ContinuousActionDimensions;
        _network = new GenomeNetwork(genome, ComputeInputCount(genome), actionCount);
    }

    public PolicyDecision Predict(float[] observation)
    {
        var output = _network.Forward(observation);

        if (_isDiscrete)
        {
            return new PolicyDecision
            {
                DiscreteAction = SelectGreedyAction(output),
            };
        }

        var actions = new float[_actionCount];
        for (int i = 0; i < _actionCount; i++)
            actions[i] = i < output.Length ? MathF.Tanh(output[i]) : 0f;

        return new PolicyDecision { ContinuousActions = actions };
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static int ComputeInputCount(NeatGenome genome)
    {
        int count = 0;
        foreach (var n in genome.Nodes)
            if (n.Role == NeatNodeRole.Input) count++;
        return count;
    }

    private static int SelectGreedyAction(float[] logits)
    {
        if (logits.Length == 0) return 0;
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
