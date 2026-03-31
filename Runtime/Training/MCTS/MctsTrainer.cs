using System;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// MCTS (Monte Carlo Tree Search) trainer.
/// <para>
/// This is a <b>pure planning</b> algorithm — it does not learn from experience.
/// Instead, at each decision point it runs UCT tree search using the registered
/// <see cref="IEnvironmentModel"/> to select the best action.
/// </para>
/// <para>
/// <b>Setup:</b> Call <see cref="SetEnvironmentModel"/> once before training starts
/// (e.g., in your scene's <c>_Ready</c>):
/// <code>MctsTrainer.SetEnvironmentModel(this);</code>
/// </para>
/// <para>
/// <b>Action space:</b> Discrete only.
/// </para>
/// </summary>
public sealed class MctsTrainer : ITrainer
{
    // ── Static model registration ────────────────────────────────────────────

    private static IEnvironmentModel? _registeredModel;

    /// <summary>
    /// Registers the environment model used by all MCTS trainers in this session.
    /// Call this once from your scene's <c>_Ready</c> method before training starts.
    /// </summary>
    public static void SetEnvironmentModel(IEnvironmentModel model)
    {
        _registeredModel = model ?? throw new ArgumentNullException(nameof(model));
        GD.Print("[MCTS] Environment model registered: ", model.GetType().Name);
    }

    /// <summary>Returns the currently registered environment model, or null if none is registered.</summary>
    public static IEnvironmentModel? GetRegisteredModel() => _registeredModel;

    // ── Instance state ───────────────────────────────────────────────────────

    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _mctsConfig;
    private readonly int _actionCount;
    private readonly MctsSearch _search;
    private readonly Random _rng;

    private long _totalStepsSeen;

    public MctsTrainer(PolicyGroupConfig config)
    {
        if (config.DiscreteActionCount <= 0)
            throw new InvalidOperationException(
                $"[MCTS] Only discrete action spaces are supported. " +
                $"Group '{config.GroupId}' has no discrete actions.");

        if (config.ContinuousActionDimensions > 0)
            throw new InvalidOperationException(
                $"[MCTS] Continuous actions are not supported. " +
                $"Group '{config.GroupId}' has continuous actions.");

        if (_registeredModel == null)
            throw new InvalidOperationException(
                "[MCTS] No environment model is registered. " +
                "Call MctsTrainer.SetEnvironmentModel(this) from your scene before training starts.");

        _config      = config;
        _mctsConfig  = config.TrainerConfig;
        _actionCount = config.DiscreteActionCount;
        _rng         = new Random();

        _search = new MctsSearch(
            model:               _registeredModel,
            actionCount:         _actionCount,
            numSimulations:      _mctsConfig.MctsNumSimulations,
            maxDepth:            _mctsConfig.MctsMaxSearchDepth,
            rolloutDepth:        _mctsConfig.MctsRolloutDepth,
            explorationConstant: _mctsConfig.MctsExplorationConstant,
            gamma:               _mctsConfig.MctsGamma,
            rng:                 _rng);
    }

    // ── ITrainer ─────────────────────────────────────────────────────────────

    public PolicyDecision SampleAction(float[] observation)
    {
        var action = _search.Search(observation);
        return new PolicyDecision
        {
            DiscreteAction   = action,
            LogProbability   = 0f,
            Value            = 0f,
            Entropy          = 0f,
        };
    }

    public PolicyDecision[] SampleActions(VectorBatch observations)
    {
        var results = new PolicyDecision[observations.BatchSize];
        for (var i = 0; i < observations.BatchSize; i++)
            results[i] = SampleAction(observations.CopyRow(i));
        return results;
    }

    /// <summary>MCTS has no value network; returns 0.</summary>
    public float EstimateValue(float[] observation) => 0f;

    public float[] EstimateValues(VectorBatch observations)
        => new float[observations.BatchSize];

    /// <summary>MCTS is a pure planning algorithm — transitions are not stored.</summary>
    public void RecordTransition(Transition transition)
    {
        _totalStepsSeen++;
    }

    /// <summary>MCTS has no learning phase; always returns null.</summary>
    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
        => null;

    public RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        // MCTS has no learned weights — checkpoint stores config/hyperparams only.
        var checkpoint = new RLCheckpoint
        {
            RunId            = groupId,
            TotalSteps       = totalSteps,
            EpisodeCount     = episodeCount,
            UpdateCount      = updateCount,
            WeightBuffer     = Array.Empty<float>(),
            LayerShapeBuffer = Array.Empty<int>(),
        };
        return CheckpointMetadataBuilder.Apply(checkpoint, _config);
    }

    public IInferencePolicy SnapshotPolicyForEval()
    {
        if (_registeredModel == null)
            throw new InvalidOperationException("[MCTS] No environment model registered.");

        return new MctsInferencePolicy(
            _registeredModel,
            _actionCount,
            _mctsConfig.MctsNumSimulations,
            _mctsConfig.MctsMaxSearchDepth,
            _mctsConfig.MctsRolloutDepth,
            _mctsConfig.MctsExplorationConstant,
            _mctsConfig.MctsGamma);
    }

    public void LoadFromCheckpoint(RLCheckpoint checkpoint)
    {
        // No weights to load; log informational message.
        GD.Print($"[MCTS] LoadFromCheckpoint called — MCTS has no learned weights. Resuming planning from fresh tree.");
    }
}
