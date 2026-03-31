using System;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Inference policy for MCTS checkpoints.
/// Runs UCT search using the registered <see cref="IEnvironmentModel"/> to select actions.
/// <para>
/// MCTS has no learned weights, so <see cref="LoadCheckpoint"/> is a no-op.
/// Ensure <see cref="MctsTrainer.SetEnvironmentModel"/> has been called before use.
/// </para>
/// </summary>
public sealed class MctsInferencePolicy : IInferencePolicy
{
    private readonly MctsSearch _search;

    public MctsInferencePolicy(
        IEnvironmentModel model,
        int actionCount,
        int numSimulations,
        int maxDepth,
        int rolloutDepth,
        float explorationConstant,
        float gamma)
    {
        _search = new MctsSearch(
            model:               model,
            actionCount:         actionCount,
            numSimulations:      numSimulations,
            maxDepth:            maxDepth,
            rolloutDepth:        rolloutDepth,
            explorationConstant: explorationConstant,
            gamma:               gamma);
    }

    /// <summary>
    /// Creates an MCTS inference policy using the currently registered environment model.
    /// Requires <see cref="MctsTrainer.SetEnvironmentModel"/> to have been called beforehand.
    /// Config parameters are read from the checkpoint hyperparams when available, otherwise
    /// use safe defaults.
    /// </summary>
    public static MctsInferencePolicy FromCheckpoint(RLCheckpoint checkpoint)
    {
        var model = MctsTrainer.GetRegisteredModel();
        if (model == null)
            throw new InvalidOperationException(
                "[MCTS] No environment model is registered. " +
                "Call MctsTrainer.SetEnvironmentModel(this) before loading an MCTS checkpoint.");

        checkpoint.Hyperparams.TryGetValue("mcts_num_simulations",     out var numSims);
        checkpoint.Hyperparams.TryGetValue("mcts_max_search_depth",    out var maxDepth);
        checkpoint.Hyperparams.TryGetValue("mcts_rollout_depth",       out var rolloutDepth);
        checkpoint.Hyperparams.TryGetValue("mcts_exploration_constant", out var c);
        checkpoint.Hyperparams.TryGetValue("gamma",                    out var gamma);

        return new MctsInferencePolicy(
            model:               model,
            actionCount:         checkpoint.DiscreteActionCount,
            numSimulations:      numSims      > 0 ? (int)numSims      : 50,
            maxDepth:            maxDepth     > 0 ? (int)maxDepth     : 20,
            rolloutDepth:        rolloutDepth > 0 ? (int)rolloutDepth : 10,
            explorationConstant: c            > 0 ? c                 : 1.414f,
            gamma:               gamma        > 0 ? gamma             : 0.99f);
    }

    /// <summary>MCTS has no learned weights — this is a no-op.</summary>
    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        GD.Print("[MCTS] LoadCheckpoint called on inference policy — MCTS has no learned weights.");
    }

    public PolicyDecision Predict(float[] observation)
    {
        var action = _search.Search(observation);
        return new PolicyDecision
        {
            DiscreteAction = action,
            LogProbability = 0f,
            Value          = 0f,
            Entropy        = 0f,
        };
    }
}
