using System;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Spawns agent scenes at runtime, switching count based on training vs. inference mode.
///
/// <b>Training mode</b> (launched via the plugin's Run Training button):
///   Spawns <see cref="TrainingCount"/> instances.
///
/// <b>Inference / standalone mode</b> (Run Project or Run Inference):
///   Spawns <see cref="InferenceCount"/> instances (default 1).
///
/// <b>Editor preview</b>:
///   Renders one instance of <see cref="AgentScene"/> at design time so you can
///   position and style agents without running the scene.
///   The preview is not saved with the scene.
///
/// Usage:
///   1. Add an RLAgentSpawner node as a sibling of your Academy.
///   2. Assign your agent <see cref="PackedScene"/> (the template).
///   3. Optionally set <see cref="SpawnPosition"/> to offset the spawn origin.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLAgentSpawner : Node
{
    // ── Exports ───────────────────────────────────────────────────────────────

    private PackedScene? _agentScene;
    private bool _overrideControlMode;
    private RLAgentControlMode _controlMode = RLAgentControlMode.Auto;
    private int _trainingCount;
    private int _inferenceCount = 1;
    private Vector2 _spawnPosition = Vector2.Zero;

    /// <summary>The scene to instantiate. Changing this updates the editor preview.</summary>
    [Export]
    public PackedScene? AgentScene
    {
        get => _agentScene;
        set
        {
            if (ReferenceEquals(_agentScene, value)) return;

            UnwatchAgentScene();
            _agentScene = value;
            WatchAgentScene();

            if (Engine.IsEditorHint() && IsInsideTree())
                RefreshEditorPreview();
        }
    }

    /// <summary>
    /// When enabled, all spawned <see cref="IRLAgent"/> nodes receive
    /// <see cref="ControlMode"/> regardless of what is set in the PackedScene.
    /// </summary>
    [Export]
    public bool OverrideControlMode
    {
        get => _overrideControlMode;
        set
        {
            _overrideControlMode = value;
            if (Engine.IsEditorHint() && IsInsideTree())
                RefreshEditorPreview();
        }
    }

    /// <summary>
    /// Control mode applied when <see cref="OverrideControlMode"/> is enabled.
    /// </summary>
    [Export]
    public RLAgentControlMode ControlMode
    {
        get => _controlMode;
        set
        {
            _controlMode = value;
            if (Engine.IsEditorHint() && IsInsideTree())
                RefreshEditorPreview();
        }
    }

    /// <summary>Number of agents to spawn during training.</summary>
    [Export(PropertyHint.Range, "0,1000,1,or_greater")]
    public int TrainingCount
    {
        get => _trainingCount;
        set
        {
            _trainingCount = value;
            if (Engine.IsEditorHint() && IsInsideTree())
                RefreshEditorPreview();
        }
    }

    /// <summary>Number of agents to spawn in inference / standalone mode.</summary>
    [Export(PropertyHint.Range, "1,32,1,or_greater")]
    public int InferenceCount
    {
        get => _inferenceCount;
        set
        {
            _inferenceCount = value;
            if (Engine.IsEditorHint() && IsInsideTree())
                RefreshEditorPreview();
        }
    }

    /// <summary>
    /// Local-space offset applied to each spawned instance's position (2D or 3D).
    /// Useful when the spawner node is not at the spawn origin.
    /// All instances spawn at the same position — your agent scene's own
    /// _Ready / OnEpisodeBegin is responsible for placing them individually.
    /// </summary>
    [Export]
    public Vector2 SpawnPosition
    {
        get => _spawnPosition;
        set
        {
            _spawnPosition = value;
            if (Engine.IsEditorHint() && IsInsideTree())
                RefreshEditorPreview();
        }
    }

    // ── Private state ─────────────────────────────────────────────────────────

    private Node? _previewInstance;
    private readonly Callable _agentSceneChangedCallable;
    private PackedScene? _watchedAgentScene;

    public RLAgentSpawner()
    {
        _agentSceneChangedCallable = Callable.From(OnAgentSceneResourceChanged);
    }

    // ── Godot lifecycle ───────────────────────────────────────────────────────

    public override void _Ready()
    {
        WatchAgentScene();

        if (Engine.IsEditorHint())
        {
            RefreshEditorPreview();
            return;
        }

        // Runtime: remove any lingering editor preview first (shouldn't exist, but safety)
        ClearPreview();

        SpawnAgents();
    }

    public override void _ExitTree()
    {
        ClearPreview();
        UnwatchAgentScene();
    }

    // ── Editor preview ────────────────────────────────────────────────────────

    private void RefreshEditorPreview()
    {
        ClearPreview();

        if (_agentScene == null) return;

        try
        {
            _previewInstance = _agentScene.Instantiate();

            // Apply spawn position offset if this is a 2D node
            if (_previewInstance is Node2D n2d)
                n2d.Position = SpawnPosition;

            ApplyControlModeOverride(_previewInstance);

            AddChild(_previewInstance);
            // NOT setting Owner → preview is transient, not saved to the .tscn file
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLAgentSpawner] Could not instantiate AgentScene for preview: {ex.Message}");
            _previewInstance = null;
        }
    }

    private void ClearPreview()
    {
        if (_previewInstance != null && IsInstanceValid(_previewInstance))
        {
            if (_previewInstance.GetParent() == this)
                RemoveChild(_previewInstance);

            _previewInstance.QueueFree();
            _previewInstance = null;
        }
    }

    private void WatchAgentScene()
    {
        if (_agentScene == null) return;

        if (_watchedAgentScene == _agentScene) return;

        UnwatchAgentScene();
        _watchedAgentScene = _agentScene;

        if (!_watchedAgentScene.IsConnected("changed", _agentSceneChangedCallable))
            _watchedAgentScene.Connect("changed", _agentSceneChangedCallable);
    }

    private void UnwatchAgentScene()
    {
        if (_watchedAgentScene == null) return;

        if (IsInstanceValid(_watchedAgentScene)
            && _watchedAgentScene.IsConnected("changed", _agentSceneChangedCallable))
        {
            _watchedAgentScene.Disconnect("changed", _agentSceneChangedCallable);
        }

        _watchedAgentScene = null;
    }

    private void OnAgentSceneResourceChanged()
    {
        if (Engine.IsEditorHint() && IsInsideTree())
            RefreshEditorPreview();
    }

    // ── Runtime spawning ──────────────────────────────────────────────────────

    private void SpawnAgents()
    {
        if (_agentScene == null)
        {
            GD.PushWarning("[RLAgentSpawner] AgentScene is not assigned — no agents spawned.");
            return;
        }

        int count = IsTrainingMode() ? ResolveTrainingCount() : InferenceCount;

        for (int i = 0; i < count; i++)
        {
            var instance = _agentScene.Instantiate();

            if (instance is Node2D n2d)
                n2d.Position = SpawnPosition;

            ApplyControlModeOverride(instance);

            AddChild(instance);
        }

        GD.Print($"[RLAgentSpawner] Spawned {count} agent(s) " +
                 $"({(IsTrainingMode() ? "training" : "inference")} mode).");
    }

    // ── Mode detection ────────────────────────────────────────────────────────

    private bool IsTrainingMode()
    {
        // Walk up the scene tree looking for a TrainingBootstrap — same logic as RLAcademy
        var current = GetParent();
        while (current is not null)
        {
            if (current is TrainingBootstrap) return true;
            current = current.GetParent();
        }
        return false;
    }

    // ── Training count resolution ─────────────────────────────────────────────

    private int ResolveTrainingCount()
    {
        if (TrainingCount > 0) return TrainingCount;

        GD.PushWarning(
            "[RLAgentSpawner] TrainingCount is 0. Falling back to InferenceCount. Set TrainingCount explicitly.");
        return InferenceCount;
    }

    private void ApplyControlModeOverride(Node root)
    {
        if (!OverrideControlMode) return;

        if (root is IRLAgent agent)
            agent.ControlMode = ControlMode;

        foreach (Node child in root.GetChildren())
            ApplyControlModeOverride(child);
    }
}
