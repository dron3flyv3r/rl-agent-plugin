using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAgent2D : Node2D, IRLAgent
{
    private ActionSpaceBuilder? _explicitActionSpace;
    private bool _explicitActionSpaceResolved;

    private readonly ObservationBuffer _observationBuffer = new();
    private float[] _lastObservation = Array.Empty<float>();
    private ObservationSegment[] _lastObservationSegments = Array.Empty<ObservationSegment>();
    private int? _validatedObservationSize;
    private float _pendingReward;
    private readonly Dictionary<string, float> _pendingRewardComponents = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _episodeRewardComponents = new(StringComparer.Ordinal);
    private Dictionary<string, float> _lastStepRewardBreakdown = new(StringComparer.Ordinal);
    private bool _isDone;
    private bool _donePending;
    private RLAgentControlMode _controlMode = RLAgentControlMode.Auto;

    [ExportGroup("Control")]
    [Export]
    public RLAgentControlMode ControlMode
    {
        get => _controlMode;
        set { _controlMode = value; UpdateConfigurationWarnings(); }
    }

    [Export] public RLPolicyGroupConfig PolicyGroupConfig { get; set; } = new RLPolicyGroupConfig();

    public int EpisodeSteps { get; private set; }
    public float EpisodeReward { get; private set; }
    public int CurrentActionIndex { get; private set; }
    public float[] CurrentContinuousActions { get; private set; } = Array.Empty<float>();
    public IReadOnlyList<ObservationSegment> LastObservationSegments => _lastObservationSegments;

    /// <summary>Reward accumulated since the last ConsumePendingReward call. Useful for human-mode agents where the training loop never consumes rewards.</summary>
    public float PendingReward => _pendingReward;

    /// <summary>Reward amount consumed in the most recent ConsumePendingReward call (i.e. the last completed step's reward).</summary>
    public float LastStepReward { get; private set; }

    /// <summary>Returns a snapshot of pending reward components without consuming them.</summary>
    public IReadOnlyDictionary<string, float> GetPendingRewardBreakdown() =>
        new Dictionary<string, float>(_pendingRewardComponents, StringComparer.Ordinal);

    /// <summary>Returns the reward component breakdown from the most recently consumed step.</summary>
    public IReadOnlyDictionary<string, float> GetLastStepRewardBreakdown() => _lastStepRewardBreakdown;

    public override void _Ready()
    {
        AddToGroup("rl_agent_plugin_agent");
    }

    // ── New API ───────────────────────────────────────────────────────────────

    /// <summary>Override to fill the observation buffer each step.</summary>
    public virtual void CollectObservations(ObservationBuffer buffer) { }

    /// <summary>
    /// Called each physics step. Override to compute rewards and call
    /// EndEpisode() when the episode is complete.
    /// </summary>
    public virtual void OnStep() { }

    /// <summary>Called at the start of every episode. Override to reset scene state.</summary>
    public virtual void OnEpisodeBegin() { }

    /// <summary>Called before each new episode during training when MaxCurriculumSteps > 0. progress is in [0, 1].</summary>
    public virtual void OnTrainingProgress(float progress) { }
    void IRLAgent.NotifyCurriculumProgress(float progress) => OnTrainingProgress(progress);

    /// <summary>Called each physics step when ControlMode is Human. Override to read player input.</summary>
    protected virtual void OnHumanInput() { }
    void IRLAgent.HandleHumanInput() => OnHumanInput();

    /// <summary>Accumulate reward for the current step.</summary>
    protected void AddReward(float reward) => _pendingReward += reward;

    /// <summary>Accumulate a named reward component for the current step.</summary>
    protected void AddReward(float reward, string tag)
    {
        _pendingReward += reward;
        AddRewardComponent(_pendingRewardComponents, tag, reward);
    }

    /// <summary>Replace the current pending reward with a specific value.</summary>
    protected void SetReward(float reward)
    {
        _pendingReward = reward;
        _pendingRewardComponents.Clear();
    }

    /// <summary>Replace the current pending reward with a single named reward component.</summary>
    protected void SetReward(float reward, string tag)
    {
        _pendingReward = reward;
        _pendingRewardComponents.Clear();
        AddRewardComponent(_pendingRewardComponents, tag, reward);
    }

    /// <summary>Signal that the episode has ended. Can be called from external scripts (e.g. a game controller).</summary>
    public void EndEpisode()
    {
        IsDone = true;
        _donePending = true;
    }

    /// <summary>When true, this agent instance is terminal and receives only no-op actions until ResetEpisode().</summary>
    public bool IsDone
    {
        get => _isDone;
        set
        {
            _isDone = value;
            if (_isDone)
            {
                ApplyNoOpAction();
            }
        }
    }

    // ── Action API ────────────────────────────────────────────────────────────

    public virtual void DefineActions(ActionSpaceBuilder builder) { }

    protected virtual void OnActionsReceived(ActionBuffer actions) { }

    public virtual RLActionDefinition[] GetActionSpace()
    {
        return ResolveExplicitActionSpace()?.Build() ?? Array.Empty<RLActionDefinition>();
    }

    public virtual string[] GetDiscreteActionLabels()
    {
        return ResolveExplicitActionSpace()?.BuildDiscreteActionLabels() ?? Array.Empty<string>();
    }

    public int GetDiscreteActionCount()
    {
        return ResolveExplicitActionSpace()?.GetDiscreteActionCount() ?? 0;
    }

    public bool SupportsOnlyDiscreteActions()
    {
        return ResolveExplicitActionSpace()?.SupportsOnlyDiscreteActions() ?? false;
    }

    public virtual void ApplyAction(int action)
    {
        if (IsDone)
        {
            ApplyNoOpAction();
            return;
        }

        var explicitActionSpace = ResolveExplicitActionSpace();
        var discreteActionCount = GetDiscreteActionCount();
        if (discreteActionCount > 0 && (action < 0 || action >= discreteActionCount))
        {
            return;
        }

        CurrentActionIndex = action;
        CurrentContinuousActions = Array.Empty<float>();
        if (explicitActionSpace is not null)
        {
            OnActionsReceived(explicitActionSpace.CreateDiscreteActionBuffer(action));
        }
    }

    public virtual void ApplyAction(float[] continuousActions)
    {
        if (IsDone)
        {
            ApplyNoOpAction();
            return;
        }

        CurrentActionIndex = -1;
        CurrentContinuousActions = continuousActions is null ? Array.Empty<float>() : (float[])continuousActions.Clone();
        var explicitActionSpace = ResolveExplicitActionSpace();
        if (explicitActionSpace is not null)
        {
            OnActionsReceived(explicitActionSpace.CreateContinuousActionBuffer(CurrentContinuousActions));
        }
    }

    public int GetContinuousActionDimensions()
    {
        return ResolveExplicitActionSpace()?.GetContinuousActionDimensions() ?? 0;
    }

    public void ApplyNoOpAction()
    {
        if (GetDiscreteActionCount() > 0)
        {
            CurrentActionIndex = 0;
            CurrentContinuousActions = Array.Empty<float>();
            var explicitActionSpace = ResolveExplicitActionSpace();
            if (explicitActionSpace is not null)
            {
                OnActionsReceived(explicitActionSpace.CreateDiscreteActionBuffer(0));
            }

            return;
        }

        var continuousDims = GetContinuousActionDimensions();
        CurrentActionIndex = -1;
        CurrentContinuousActions = continuousDims > 0
            ? new float[continuousDims]
            : Array.Empty<float>();
        var actionSpace = ResolveExplicitActionSpace();
        if (actionSpace is not null && continuousDims > 0)
        {
            OnActionsReceived(actionSpace.CreateContinuousActionBuffer(CurrentContinuousActions));
        }
    }

    public virtual void ResetEpisode()
    {
        EpisodeSteps = 0;
        EpisodeReward = 0.0f;
        _episodeRewardComponents.Clear();
        CurrentActionIndex = GetDiscreteActionCount() > 0 ? 0 : -1;
        CurrentContinuousActions = Array.Empty<float>();
        _pendingReward = 0f;
        _pendingRewardComponents.Clear();
        LastStepReward = 0f;
        _lastStepRewardBreakdown = new Dictionary<string, float>(StringComparer.Ordinal);
        _isDone = false;
        _donePending = false;
        OnEpisodeBegin();
        if (GetDiscreteActionCount() > 0)
        {
            ApplyAction(CurrentActionIndex);
        }
    }

    public void AccumulateReward(float reward, IReadOnlyDictionary<string, float>? rewardBreakdown = null)
    {
        EpisodeReward += reward;
        EpisodeSteps += 1;

        if (rewardBreakdown is null)
        {
            return;
        }

        foreach (var (tag, amount) in rewardBreakdown)
        {
            AddRewardComponent(_episodeRewardComponents, tag, amount);
        }
    }

    public bool HasReachedEpisodeLimit()
    {
        var maxEpisodeSteps = GetEffectiveMaxEpisodeSteps();
        return maxEpisodeSteps > 0 && EpisodeSteps >= maxEpisodeSteps;
    }

    public string GetInferenceModelPath()
    {
        return NormalizeResourcePath(PolicyGroupConfig.InferenceModelPath);
    }

    public string GetPolicyAgentId()
    {
        return string.IsNullOrWhiteSpace(PolicyGroupConfig.AgentId)
            ? Name
            : PolicyGroupConfig.AgentId.Trim();
    }

    public int GetExpectedObservationSize()
    {
        return ((IRLAgent)this).CollectObservationArray().Length;
    }

    public float[] GetLastObservation()
    {
        return (float[])_lastObservation.Clone();
    }

    public override string[] _GetConfigurationWarnings()
    {
        var warnings = new List<string>();
        if (ControlMode == RLAgentControlMode.Inference && string.IsNullOrWhiteSpace(GetInferenceModelPath()))
        {
            warnings.Add("ControlMode is Inference but no inference model path is set on PolicyGroupConfig.");
        }
        else if (ControlMode == RLAgentControlMode.Auto && string.IsNullOrWhiteSpace(GetInferenceModelPath()))
        {
            warnings.Add("ControlMode is Auto with no inference model path set. " +
                         "The agent will train normally, but will be skipped during normal play (Run Project / Run Inference) " +
                         "unless a checkpoint already exists in the default run directory.");
        }
        return warnings.ToArray();
    }

    public int GetEffectiveMaxEpisodeSteps()
    {
        if (PolicyGroupConfig.MaxEpisodeSteps > 0)
            return PolicyGroupConfig.MaxEpisodeSteps;

        var academy = ResolveAcademy();
        if (academy is null)
            GD.PushWarning($"[RL] {Name}: no RLAcademy found in scene — MaxEpisodeSteps will be 0 (episodes never timeout). Add an RLAcademy node to the scene.");
        return academy?.MaxEpisodeSteps ?? 0;
    }

    // ── Framework-internal ────────────────────────────────────────────────────

    public Node AsNode() => this;

    /// <summary>Called by the training/inference loop each physics step.</summary>
    void IRLAgent.TickStep() => OnStep();

    /// <summary>Returns accumulated reward and clears the pending buffer. Also updates LastStepReward.</summary>
    float IRLAgent.ConsumePendingReward()
    {
        var reward = _pendingReward;
        LastStepReward = reward;
        _pendingReward = 0f;
        return reward;
    }

    Dictionary<string, float> IRLAgent.ConsumePendingRewardBreakdown()
    {
        var breakdown = new Dictionary<string, float>(_pendingRewardComponents, StringComparer.Ordinal);
        _lastStepRewardBreakdown = breakdown;
        _pendingRewardComponents.Clear();
        return breakdown;
    }

    /// <summary>Returns the done flag and clears it.</summary>
    bool IRLAgent.ConsumeDonePending()
    {
        var done = _donePending;
        _donePending = false;
        return done;
    }

    float[] IRLAgent.CollectObservationArray()
    {
        _observationBuffer.Clear();
        CollectObservations(_observationBuffer);

        var observation = _observationBuffer.ToArray();
        _lastObservation = observation;
        _lastObservationSegments = _observationBuffer.GetSegmentsSnapshot();
        if (_validatedObservationSize is null)
        {
            _validatedObservationSize = observation.Length;
        }
        else if (_validatedObservationSize.Value != observation.Length)
        {
            GD.PushError(
                $"[RLAgent2D] Agent '{Name}' changed observation size from {_validatedObservationSize.Value} to {observation.Length}. " +
                "Observation size must remain stable across steps and episodes.");
        }

        return observation;
    }

    public IReadOnlyDictionary<string, float> GetEpisodeRewardBreakdown()
    {
        return new Dictionary<string, float>(_episodeRewardComponents, StringComparer.Ordinal);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private ActionSpaceBuilder? ResolveExplicitActionSpace()
    {
        if (_explicitActionSpaceResolved)
        {
            return _explicitActionSpace;
        }

        var builder = new ActionSpaceBuilder();
        DefineActions(builder);
        _explicitActionSpace = builder.HasActions ? builder : null;
        _explicitActionSpaceResolved = true;
        return _explicitActionSpace;
    }

    private static void AddRewardComponent(IDictionary<string, float> target, string tag, float amount)
    {
        if (string.IsNullOrWhiteSpace(tag))
        {
            return;
        }

        target.TryGetValue(tag, out var current);
        target[tag] = current + amount;
    }

    private static string NormalizeResourcePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        if (path.StartsWith("uid://", StringComparison.Ordinal))
        {
            var resolvedPath = ResourceUid.EnsurePath(path);
            if (!string.IsNullOrWhiteSpace(resolvedPath))
            {
                return resolvedPath;
            }
        }

        return path;
    }

    private RLAcademy? _cachedAcademy;

    private RLAcademy? ResolveAcademy()
    {
        if (_cachedAcademy is not null) return _cachedAcademy;

        // Fast path: walk up the ancestor chain.
        Node? current = this;
        while (current is not null)
        {
            if (current is RLAcademy a) { _cachedAcademy = a; return a; }
            current = current.GetParent();
        }

        // Fallback: scene-wide group search (handles sibling academy).
        if (IsInsideTree())
        {
            foreach (var node in GetTree().GetNodesInGroup("rl_agent_plugin_academy"))
                if (node is RLAcademy a) { _cachedAcademy = a; return a; }
        }

        return null;
    }
}
