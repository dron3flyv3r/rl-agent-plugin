using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAcademy : Node
{
    private RLTrainingConfig? _trainingConfig;

    [ExportGroup("Configuration")]
    /// <summary>
    /// Training algorithm and schedule resource used when this academy participates in training.
    /// </summary>
    [Export]
    public RLTrainingConfig? TrainingConfig
    {
        get => _trainingConfig;
        set { _trainingConfig = value; UpdateConfigurationWarnings(); }
    }
    /// <summary>
    /// Global episode length cap used when agents and policy groups do not override it (0 = no cap).
    /// </summary>
    [Export(PropertyHint.Range, "0,100000,1,or_greater")] public int MaxEpisodeSteps { get; set; } = 0;

    [ExportGroup("Run")]
    /// <summary>
    /// Run-level simulation settings (batching, time scale, checkpoints, threading).
    /// </summary>
    [Export] public RLRunConfig? RunConfig { get; set; }

    /// <summary>
    /// Optional.  When set, training runs in distributed mode: this process acts as the master
    /// trainer while headless worker processes collect rollouts in parallel.
    /// Leave null for the standard single-process training mode.
    /// </summary>
    [Export] public RLDistributedConfig? DistributedConfig { get; set; }

    [ExportGroup("Curriculum")]
    /// <summary>
    /// Optional curriculum settings for dynamic difficulty progression.
    /// </summary>
    [Export] public RLCurriculumConfig? Curriculum { get; set; }

    [ExportGroup("Self-Play")]
    /// <summary>
    /// Optional self-play pairing configuration for multi-policy training.
    /// </summary>
    [Export] public RLSelfPlayConfig? SelfPlay { get; set; }

    [ExportGroup("Debug")]
    /// <summary>Show the observation/reward/action spy overlay when running outside of training. Press Tab to cycle agents.</summary>
    [Export] public bool EnableSpyOverlay { get; set; } = false;
    /// <summary>Show a live preview panel of every RLCameraSensor2D found on any agent. Works both in play mode and during training.</summary>
    [Export] public bool EnableCameraDebug { get; set; } = false;

    // Delegating properties — not exported; satisfy TrainingBootstrap C# property access.
    public string RunPrefix           => RunConfig?.RunPrefix           ?? string.Empty;
    public float  SimulationSpeed     => RunConfig?.SimulationSpeed     ?? 1.0f;
    public int    BatchSize           => RunConfig?.BatchSize           ?? 1;
    public int    ActionRepeat        => RunConfig?.ActionRepeat        ?? 1;
    public int    CheckpointInterval  => RunConfig?.CheckpointInterval  ?? 10;
    public bool   ShowBatchGrid         => RunConfig?.ShowBatchGrid         ?? false;
    public bool   AsyncGradientUpdates  => RunConfig?.AsyncGradientUpdates  ?? false;
    public bool   ParallelPolicyGroups  => RunConfig?.ParallelPolicyGroups  ?? false;
    public RLAsyncRolloutPolicy AsyncRolloutPolicy => RunConfig?.AsyncRolloutPolicy ?? RLAsyncRolloutPolicy.Pause;
    public RLCurriculumMode CurriculumMode => Curriculum?.Mode ?? RLCurriculumMode.StepBased;
    public long   MaxCurriculumSteps    => Curriculum?.MaxSteps             ?? 0;
    public float  DebugCurriculumProgress => Curriculum?.DebugProgress  ?? 0f;

    public override Variant _Get(StringName property)
    {
        return (string)property switch
        {
            "RunPrefix"               => RunPrefix,
            "SimulationSpeed"         => SimulationSpeed,
            "BatchSize"               => BatchSize,
            "ActionRepeat"            => ActionRepeat,
            "CheckpointInterval"      => CheckpointInterval,
            "ShowBatchGrid"           => ShowBatchGrid,
            "CurriculumMode"         => (int)CurriculumMode,
            "MaxCurriculumSteps"      => MaxCurriculumSteps,
            "DebugCurriculumProgress" => DebugCurriculumProgress,
            _                         => base._Get(property),
        };
    }

    public bool InferenceActive { get; private set; }
    /// <summary>
    /// Current curriculum progress value in the range [0, 1].
    /// </summary>
    public float CurriculumProgress { get; private set; }

    /// <summary>
    /// Sets curriculum progress and immediately notifies every discovered agent.
    /// </summary>
    /// <param name="progress">Desired curriculum progress in [0, 1]; clamped automatically.</param>
    public void SetCurriculumProgress(float progress)
    {
        CurriculumProgress = Mathf.Clamp(progress, 0f, 1f);
        foreach (var agent in GetAgents())
            agent.NotifyCurriculumProgress(CurriculumProgress);
    }

    public override string[] _GetConfigurationWarnings()
    {
        var warnings = new List<string>();
        if (TrainingConfig is null)
            warnings.Add("TrainingConfig is not assigned. Assign an RLTrainingConfig resource.");
        else if (TrainingConfig.Algorithm is null)
            warnings.Add("RLTrainingConfig.Algorithm is not assigned. Assign an RLAlgorithmConfig resource.");
        return warnings.ToArray();
    }

    private RLAgentSpyOverlay?    _spyOverlay;
    private RLCameraDebugOverlay? _cameraDebugOverlay;
    private readonly Dictionary<IRLAgent, IInferencePolicy> _agentInferencePolicies = new();
    private readonly Dictionary<IRLAgent, string> _agentObservationGroups = new();
    private readonly Dictionary<IRLAgent, int> _inferenceStepCounters = new();
    private readonly Dictionary<string, string> _observationGroupDisplayNames = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _validatedObservationSizesByGroup = new(StringComparer.Ordinal);
    private List<IRLAgent> _humanAgents = new();
    public bool HumanModeActive { get; private set; }

    public override void _Ready()
    {
        AddToGroup("rl-agent-plugin_academy");
        TryInitializeInference();
        TryInitializeHumanMode();

        if (!IsInsideTrainingBootstrap() && DebugCurriculumProgress > 0f)
            SetCurriculumProgress(DebugCurriculumProgress);

        if (!IsInsideTrainingBootstrap() && EnableSpyOverlay)
        {
            _spyOverlay = new RLAgentSpyOverlay();
            _spyOverlay.Initialize(this);
            AddChild(_spyOverlay);
        }

        if (EnableCameraDebug)
        {
            _cameraDebugOverlay = new RLCameraDebugOverlay();
            _cameraDebugOverlay.Initialize(this);
            AddChild(_cameraDebugOverlay);
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (HumanModeActive)
        {
            foreach (var agent in _humanAgents)
            {
                agent.HandleHumanInput();
                agent.TickStep();
                var stepReward = agent.ConsumePendingReward();
                var stepBreakdown = agent.ConsumePendingRewardBreakdown();
                agent.AccumulateReward(stepReward, stepBreakdown.Count > 0 ? stepBreakdown : null);
                if (agent.ConsumeDonePending() || agent.HasReachedEpisodeLimit())
                    agent.ResetEpisode();
            }
        }

        if (!InferenceActive)
        {
            return;
        }

        var repeat = Math.Max(1, ActionRepeat);

        foreach (var pair in _agentInferencePolicies)
        {
            var agent = pair.Key;
            var policy = pair.Value;

            agent.TickStep();
            var stepReward = agent.ConsumePendingReward();
            var stepBreakdown = agent.ConsumePendingRewardBreakdown();
            agent.AccumulateReward(stepReward, stepBreakdown.Count > 0 ? stepBreakdown : null);
            if (agent.ConsumeDonePending() || agent.HasReachedEpisodeLimit())
            {
                agent.ResetEpisode();
                _inferenceStepCounters[agent] = 0;
            }

            _inferenceStepCounters.TryGetValue(agent, out var stepCount);
            stepCount++;
            _inferenceStepCounters[agent] = stepCount;

            if (stepCount >= repeat)
            {
                _inferenceStepCounters[agent] = 0;
                var observation = CollectValidatedObservation(agent);
                if (observation.Length == 0)
                {
                    continue;
                }

                var decision = policy.Predict(observation);
                if (decision.DiscreteAction >= 0)
                {
                    agent.ApplyAction(decision.DiscreteAction);
                }
                else if (decision.ContinuousActions.Length > 0)
                {
                    agent.ApplyAction(decision.ContinuousActions);
                }
            }
            else
            {
                // Re-apply current action so physics-driven movement continues each tick.
                if (agent.CurrentActionIndex >= 0)
                {
                    agent.ApplyAction(agent.CurrentActionIndex);
                }
                else if (agent.CurrentContinuousActions.Length > 0)
                {
                    agent.ApplyAction(agent.CurrentContinuousActions);
                }
            }
        }
    }

    /// <summary>
    /// Returns all agents under the academy's scene root, regardless of control mode.
    /// </summary>
    public IReadOnlyList<IRLAgent> GetAgents()
    {
        var agents = new List<IRLAgent>();
        var sceneRoot = ResolveSceneRoot();
        CollectAgents(sceneRoot, agents);
        return agents;
    }

    /// <summary>
    /// Returns agents matching a specific control mode, including Auto-mode agents when relevant.
    /// </summary>
    /// <param name="controlMode">Requested control mode filter.</param>
    public IReadOnlyList<IRLAgent> GetAgents(RLAgentControlMode controlMode)
    {
        var agents = new List<IRLAgent>();
        foreach (var agent in GetAgents())
        {
            if (agent.ControlMode == controlMode)
            {
                agents.Add(agent);
            }
            // Auto agents join Train groups during training and Inference groups during normal play.
            else if (agent.ControlMode == RLAgentControlMode.Auto
                     && (controlMode == RLAgentControlMode.Train || controlMode == RLAgentControlMode.Inference))
            {
                agents.Add(agent);
            }
        }

        return agents;
    }

    /// <summary>
    /// Forces all discovered agents to reset their current episodes.
    /// </summary>
    public void ResetAllAgents()
    {
        foreach (var agent in GetAgents())
        {
            agent.ResetEpisode();
        }
    }

    /// <summary>
    /// Returns a copy of configured self-play pairings, or an empty list when self-play is disabled.
    /// </summary>
    public List<RLPolicyPairingConfig> GetResolvedSelfPlayPairings()
        => SelfPlay is not null
            ? new List<RLPolicyPairingConfig>(SelfPlay.Pairings)
            : new List<RLPolicyPairingConfig>();

    /// <summary>
    /// Infers observation sizes for agents in the academy scene.
    /// </summary>
    /// <param name="controlMode">
    /// Optional control-mode filter. When null, includes all discovered agents.
    /// </param>
    /// <param name="resetEpisodes">
    /// Whether to reset episodes during probing to ensure clean observation collection.
    /// </param>
    /// <returns>Inference result containing per-group observation-size metadata.</returns>
    public ObservationSizeInferenceResult InferObservationSizes(RLAgentControlMode? controlMode = null, bool resetEpisodes = true)
    {
        var agents = controlMode.HasValue ? GetAgents(controlMode.Value) : GetAgents();
        return ObservationSizeInference.Infer(ResolveSceneRoot(), agents, resetEpisodes);
    }

    private static void CollectAgents(Node node, ICollection<IRLAgent> agents)
    {
        if (node is IRLAgent agent)
        {
            agents.Add(agent);
        }

        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
            {
                CollectAgents(childNode, agents);
            }
        }
    }

    private bool IsInsideTrainingBootstrap()
    {
        var current = GetParent();
        while (current is not null)
        {
            if (current is TrainingBootstrap) return true;
            current = current.GetParent();
        }
        return false;
    }

    private Node ResolveSceneRoot()
    {
        var current = this as Node;
        while (current.GetParent() is Node parent
               && parent is not TrainingBootstrap
               && parent is not InferenceBootstrap
               && parent is not SubViewport)
        {
            current = parent;
        }

        return current;
    }

    private void TryInitializeInference()
    {
        if (IsInsideTrainingBootstrap())
        {
            return;
        }

        var fallbackNetworkGraph = RLNetworkGraph.CreateDefault();
        var agents = GetAgents();
        if (agents.Count == 0)
        {
            return;
        }

        // Warn about any agents left in Train mode when running outside of the training bootstrap.
        // Auto mode is intentionally excluded — it's designed to work in both contexts.
        foreach (var agent in agents)
        {
            if (agent.ControlMode == RLAgentControlMode.Train)
            {
                GD.PushWarning(
                    $"[RLAcademy] Agent '{agent.AsNode().Name}' is set to Train mode, but the game was " +
                    "started normally. Training will not run. Use the 'Start Training' button in " +
                    "the RL Agent Plugin dock, or switch the agent to Inference or Auto mode.");
            }
        }

        // GetAgents(Inference) includes Auto-mode agents so their observation sizes are inferred too.
        var observationInference = InferObservationSizes(RLAgentControlMode.Inference);
        foreach (var error in observationInference.Errors)
        {
            GD.PushError($"[RLAcademy] {error}");
        }

        foreach (var agent in agents)
        {
            if (agent.ControlMode != RLAgentControlMode.Inference
                && agent.ControlMode != RLAgentControlMode.Auto)
            {
                continue;
            }

            observationInference.AgentBindings.TryGetValue(agent, out var binding);
            var modelPath = agent.GetInferenceModelPath();
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                if (agent.ControlMode == RLAgentControlMode.Inference)
                {
                    GD.PushWarning($"[RLAcademy] Agent '{agent.AsNode().Name}' is in Inference mode but has no .rlmodel assigned.");
                }

                continue;
            }

            if (!modelPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase))
            {
                GD.PushWarning(
                    $"[RLAcademy] Agent '{agent.AsNode().Name}' has an unsupported inference model path '{modelPath}'. " +
                    "Inference requires a .rlmodel file.");
                continue;
            }

            if (!FileAccess.FileExists(modelPath))
            {
                GD.PushWarning($"[RLAcademy] .rlmodel not found for agent '{agent.AsNode().Name}': {modelPath}");
                continue;
            }

            var checkpoint = RLModelLoader.LoadFromFile(modelPath);
            if (checkpoint is null)
            {
                GD.PushWarning($"[RLAcademy] Could not load inference model for agent '{agent.AsNode().Name}'.");
                continue;
            }

            var obsSize = checkpoint.ObservationSize;
            var actionCount = checkpoint.DiscreteActionCount > 0
                ? checkpoint.DiscreteActionCount
                : checkpoint.ContinuousActionDimensions;
            if (binding is not null)
            {
                _agentObservationGroups[agent] = binding.BindingKey;
                _observationGroupDisplayNames[binding.BindingKey] = binding.DisplayName;
                if (observationInference.GroupSizes.TryGetValue(binding.BindingKey, out var groupObservationSize))
                {
                    _validatedObservationSizesByGroup[binding.BindingKey] = groupObservationSize;
                }
            }

            var agentObsSize = observationInference.AgentSizes.TryGetValue(agent, out var inferredObservationSize)
                ? inferredObservationSize
                : 0;
            var agentDiscreteCount = agent.GetDiscreteActionCount();
            var agentContinuousDims = agent.GetContinuousActionDimensions();

            if (agentObsSize <= 0)
            {
                GD.PushError($"[RLAcademy] Could not infer observations for agent '{agent.AsNode().Name}'.");
                continue;
            }

            if (obsSize != agentObsSize)
            {
                GD.PushError(
                    $"[RLAcademy] Model observation size {obsSize} does not match agent '{agent.AsNode().Name}' " +
                    $"observation size {agentObsSize}.");
                continue;
            }

            if (checkpoint.ContinuousActionDimensions > 0 && checkpoint.ContinuousActionDimensions != agentContinuousDims)
            {
                GD.PushError(
                    $"[RLAcademy] Continuous action mismatch for '{agent.AsNode().Name}': " +
                    $"model={checkpoint.ContinuousActionDimensions}, agent={agentContinuousDims}.");
                continue;
            }

            if (checkpoint.DiscreteActionCount > 0 && checkpoint.DiscreteActionCount != agentDiscreteCount)
            {
                GD.PushError(
                    $"[RLAcademy] Discrete action mismatch for '{agent.AsNode().Name}': " +
                    $"model={checkpoint.DiscreteActionCount}, agent={agentDiscreteCount}.");
                continue;
            }

            if (obsSize <= 0 || actionCount <= 0)
            {
                GD.PushWarning($"[RLAcademy] Inference model for agent '{agent.AsNode().Name}' has invalid dimensions (obs={obsSize}, actions={actionCount}).");
                continue;
            }

            try
            {
                var policy = InferencePolicyFactory.Create(checkpoint, fallbackNetworkGraph, agent.PolicyGroupConfig.StochasticInference);
                policy.LoadCheckpoint(checkpoint);
                _agentInferencePolicies[agent] = policy;
                _inferenceStepCounters[agent] = 0;
                GD.Print(
                    $"[RLAcademy] Loaded {checkpoint.Algorithm} inference model for '{agent.AsNode().Name}' " +
                    $"(obs={obsSize}, actions={actionCount}).");
            }
            catch (Exception ex)
            {
                GD.PushError($"[RLAcademy] Failed to load inference model for agent '{agent.AsNode().Name}': {ex.Message} — " +
                             "Verify that the model metadata matches the active agent.");
            }
        }

        InferenceActive = _agentInferencePolicies.Count > 0;
    }

    public RLTrainerConfig? ResolveTrainerConfig()
    {
        return TrainingConfig?.ToTrainerConfig();
    }

    private float[] CollectValidatedObservation(IRLAgent agent)
    {
        var observation = agent.CollectObservationArray();
        ValidateGroupObservationSize(agent, observation.Length);
        return observation;
    }

    private void TryInitializeHumanMode()
    {
        if (IsInsideTrainingBootstrap()) return;

        _humanAgents = new List<IRLAgent>();
        foreach (var agent in GetAgents())
        {
            if (agent.ControlMode == RLAgentControlMode.Human)
                _humanAgents.Add(agent);
        }

        HumanModeActive = _humanAgents.Count > 0;
        if (!HumanModeActive) return;

        foreach (var agent in _humanAgents)
        {
            agent.ResetEpisode();
            GD.Print($"[RLAcademy] Human mode active for '{agent.AsNode().Name}'.");
        }
    }

    private void ValidateGroupObservationSize(IRLAgent agent, int observationSize)
    {
        if (!_agentObservationGroups.TryGetValue(agent, out var groupKey))
        {
            return;
        }

        if (!_validatedObservationSizesByGroup.TryGetValue(groupKey, out var expectedSize))
        {
            _validatedObservationSizesByGroup[groupKey] = observationSize;
            return;
        }

        if (expectedSize == observationSize)
        {
            return;
        }

        var displayName = _observationGroupDisplayNames.TryGetValue(groupKey, out var name)
            ? name
            : groupKey;
        GD.PushError(
            $"[RLAcademy] Agent '{agent.AsNode().Name}' in policy group '{displayName}' emitted {observationSize} observations, " +
            $"expected {expectedSize}. Observation size must remain stable for the whole group.");
    }
}
