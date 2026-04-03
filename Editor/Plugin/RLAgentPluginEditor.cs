using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Godot;
using RlAgentPlugin.Editor;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin;

[Tool]
public partial class RLAgentPluginEditor : EditorPlugin
{
    private const string AgentScriptPath = "res://addons/rl-agent-plugin/Runtime/Agents/RLAgent2D.cs";
    private const string Agent3DScriptPath = "res://addons/rl-agent-plugin/Runtime/Agents/RLAgent3D.cs";
    private const string AcademyScriptPath = "res://addons/rl-agent-plugin/Runtime/Core/RLAcademy.cs";
    private const string HpoOrchestratorScriptPath = "res://addons/rl-agent-plugin/Runtime/HPO/RLHPOOrchestrator.cs";
    private const string DenseLayerDefScriptPath = "res://addons/rl-agent-plugin/Resources/Models/RLDenseLayerDef.cs";
    private const string DropoutLayerDefScriptPath = "res://addons/rl-agent-plugin/Resources/Models/RLDropoutLayerDef.cs";
    private const string FlattenLayerDefScriptPath = "res://addons/rl-agent-plugin/Resources/Models/RLFlattenLayerDef.cs";
    private const string LayerNormDefScriptPath = "res://addons/rl-agent-plugin/Resources/Models/RLLayerNormDef.cs";
    private const string LstmLayerDefScriptPath = "res://addons/rl-agent-plugin/Resources/Models/RLLstmLayerDef.cs";
    private const string GruLayerDefScriptPath = "res://addons/rl-agent-plugin/Resources/Models/RLGruLayerDef.cs";
    private const int QuickTestEpisodeLimit = 5;
    private static readonly string[] RequiredGlobalScriptClasses =
    {
        nameof(RLPolicyPairingConfig),
        nameof(RLNetworkGraph),
        nameof(RLDenseLayerDef),
        nameof(RLDropoutLayerDef),
        nameof(RLFlattenLayerDef),
        nameof(RLLayerNormDef),
        nameof(RLLstmLayerDef),
        nameof(RLGruLayerDef),
        nameof(RLConstantSchedule),
        nameof(RLLinearSchedule),
        nameof(RLExponentialSchedule),
        nameof(RLCosineSchedule),
    };
    private static readonly Lazy<Dictionary<string, Type>> ScriptTypeIndex = new(BuildScriptTypeIndex);

    private RLModelFormatLoader? _rlModelFormatLoader;
    private EditorDock? _setupEditorDock;
    private RLSetupDock? _setupDock;
    private RLDashboard? _dashboard;
    private Button? _startTrainingButton;
    private Button? _stopTrainingButton;
    private Button? _runInferenceButton;
    private RLHPOParameterInspectorPlugin? _hpoInspectorPlugin;
    private RLNetworkGraphInspectorPlugin? _networkGraphInspectorPlugin;
    private TrainingSceneValidation? _lastValidation;
    private bool _launchedTrainingRun;
    private bool _launchedQuickTestRun;
    private string _activeStatusPath = string.Empty;
    private string _lastAutoScenePath = string.Empty;
    private string _lastValidationSignature = string.Empty;

    public override void _EnterTree()
    {
        _rlModelFormatLoader = new RLModelFormatLoader();
        ResourceLoader.AddResourceFormatLoader(_rlModelFormatLoader, true);

        _setupDock = new RLSetupDock();
        _setupDock.StartTrainingRequested += OnStartTrainingRequested;
        _setupDock.StopTrainingRequested += OnStopTrainingRequested;
        _setupDock.QuickTestRequested += OnQuickTestRequested;
        _setupDock.ValidateSceneRequested += OnValidateSceneRequested;

        _setupEditorDock = new EditorDock
        {
            Title = "RL Setup",
            DefaultSlot = EditorDock.DockSlot.RightUl,
        };
        _setupEditorDock.AddChild(_setupDock);

        AddDock(_setupEditorDock);

        _startTrainingButton = new Button { Text = "Start Training", TooltipText = "Launch the configured training run." };
        _startTrainingButton.Pressed += OnStartTrainingRequested;
        AddControlToContainer(CustomControlContainer.Toolbar, _startTrainingButton);

        _stopTrainingButton = new Button { Text = "Stop Training", TooltipText = "Stop the active training run." };
        _stopTrainingButton.Pressed += OnStopTrainingRequested;
        AddControlToContainer(CustomControlContainer.Toolbar, _stopTrainingButton);

        _runInferenceButton = new Button { Text = "Run Inference", TooltipText = "Run the scene with trained agents in inference mode." };
        _runInferenceButton.Pressed += OnRunInferenceRequested;
        AddControlToContainer(CustomControlContainer.Toolbar, _runInferenceButton);

        _dashboard = new RLDashboard { Name = "RLDash" };
        EditorInterface.Singleton.GetEditorMainScreen().AddChild(_dashboard);
        _MakeVisible(false);

        _hpoInspectorPlugin = new RLHPOParameterInspectorPlugin();
        AddInspectorPlugin(_hpoInspectorPlugin);

        _networkGraphInspectorPlugin = new RLNetworkGraphInspectorPlugin();
        AddInspectorPlugin(_networkGraphInspectorPlugin);

        RegisterCustomTypes();
        CallDeferred(nameof(EnsureProjectScriptClassesAreFresh));
        SetProcess(true);
        RefreshValidationFromActiveScene();
    }

    public override void _ExitTree()
    {
        SetProcess(false);
        UnregisterCustomTypes();

        if (_hpoInspectorPlugin is not null)
        {
            RemoveInspectorPlugin(_hpoInspectorPlugin);
            _hpoInspectorPlugin = null;
        }

        if (_networkGraphInspectorPlugin is not null)
        {
            RemoveInspectorPlugin(_networkGraphInspectorPlugin);
            _networkGraphInspectorPlugin = null;
        }

        if (_rlModelFormatLoader is not null)
        {
            ResourceLoader.RemoveResourceFormatLoader(_rlModelFormatLoader);
            _rlModelFormatLoader = null;
        }

        if (_startTrainingButton is not null)
        {
            _startTrainingButton.Pressed -= OnStartTrainingRequested;
            RemoveControlFromContainer(CustomControlContainer.Toolbar, _startTrainingButton);
            _startTrainingButton.QueueFree();
            _startTrainingButton = null;
        }

        if (_stopTrainingButton is not null)
        {
            _stopTrainingButton.Pressed -= OnStopTrainingRequested;
            RemoveControlFromContainer(CustomControlContainer.Toolbar, _stopTrainingButton);
            _stopTrainingButton.QueueFree();
            _stopTrainingButton = null;
        }

        if (_runInferenceButton is not null)
        {
            _runInferenceButton.Pressed -= OnRunInferenceRequested;
            RemoveControlFromContainer(CustomControlContainer.Toolbar, _runInferenceButton);
            _runInferenceButton.QueueFree();
            _runInferenceButton = null;
        }

        if (_setupDock is not null)
        {
            _setupDock.StartTrainingRequested -= OnStartTrainingRequested;
            _setupDock.StopTrainingRequested -= OnStopTrainingRequested;
            _setupDock.QuickTestRequested -= OnQuickTestRequested;
            _setupDock.ValidateSceneRequested -= OnValidateSceneRequested;
            _setupDock = null;
        }

        if (_setupEditorDock is not null)
        {
            RemoveDock(_setupEditorDock);
            _setupEditorDock.QueueFree();
            _setupEditorDock = null;
        }

        if (_dashboard is not null)
        {
            _dashboard.QueueFree();
            _dashboard = null;
        }
    }

    public override bool _HasMainScreen() => true;

    public override string _GetPluginName() => "RLDash";

    public override void _MakeVisible(bool visible)
    {
        if (_dashboard is not null)
            _dashboard.Visible = visible;
    }

    public override bool _Build()
    {
        // Training validation runs inside OnStartTrainingRequested(), not here.
        // Blocking the build would prevent normal "Run Project" from working.
        EnsureProjectScriptClassesAreFresh();
        return true;
    }

    public override void _Process(double delta)
    {
        if (_setupDock is null)
        {
            return;
        }

        var isPlaying = EditorInterface.Singleton.IsPlayingScene();
        var currentScenePath = ResolveTrainingScenePath();
        if (!isPlaying)
        {
            if (_launchedQuickTestRun)
            {
                _setupDock.SetLaunchStatus(ReadCompletedQuickTestStatus(_activeStatusPath));
                _launchedQuickTestRun = false;
                _activeStatusPath = string.Empty;
            }

            _launchedTrainingRun = false;
        }

        var hasScenePath = !string.IsNullOrWhiteSpace(currentScenePath);
        var validationBlocked = _lastValidation is { IsValid: false };
        var launchInProgress = _launchedTrainingRun || _launchedQuickTestRun;
        var startTrainingTooltip = !hasScenePath
            ? "Open a scene or set a main scene first."
            : validationBlocked && _lastValidation!.Errors.Count > 0
                ? _lastValidation.Errors[0]
                : "Launch the configured training run.";
        var quickTestTooltip = !hasScenePath
            ? "Open a scene or set a main scene first."
            : validationBlocked && _lastValidation!.Errors.Count > 0
                ? _lastValidation.Errors[0]
                : $"Run a short smoke test capped at {QuickTestEpisodeLimit} episodes.";
        var stopTooltip = launchInProgress
            ? "Stop the active RL launch."
            : "No RL launch is currently running.";
        var validateTooltip = !hasScenePath
            ? "Open a scene or set a main scene first."
            : isPlaying
            ? "Scene validation is disabled while a launch is running."
            : "Run scene validation now.";

        if (_startTrainingButton is not null)
        {
            _startTrainingButton.Disabled = !hasScenePath || launchInProgress || validationBlocked;
            _startTrainingButton.TooltipText = startTrainingTooltip;
        }

        if (_stopTrainingButton is not null)
        {
            _stopTrainingButton.Disabled = !launchInProgress;
            _stopTrainingButton.TooltipText = stopTooltip;
        }

        if (_runInferenceButton is not null)
        {
            var hasInferenceAgents = _lastValidation is { InferenceAgentCount: > 0 };
            _runInferenceButton.Disabled = isPlaying || !hasInferenceAgents;
            _runInferenceButton.TooltipText = hasInferenceAgents
                ? "Launch the scene with trained agents running in inference mode."
                : "Run Inference — assign a valid .rlmodel to at least one Inference or Auto agent.";
        }

        _setupDock.SetActionStates(
            canStartTraining: hasScenePath && !launchInProgress && !validationBlocked,
            startTrainingTooltip: startTrainingTooltip,
            canStop: launchInProgress,
            stopTooltip: stopTooltip,
            canQuickTest: hasScenePath && !launchInProgress && !validationBlocked,
            quickTestTooltip: quickTestTooltip,
            canValidateScene: hasScenePath && !isPlaying,
            validateSceneTooltip: validateTooltip);

        if (isPlaying)
        {
            return;
        }

        // Auto-refresh validation when the edited scene changes.
        if (currentScenePath != _lastAutoScenePath)
        {
            _lastAutoScenePath = currentScenePath;
            RefreshValidationFromActiveScene();
            return;
        }

        var currentSignature = BuildValidationSignature(EditorInterface.Singleton.GetEditedSceneRoot());
        if (currentSignature != _lastValidationSignature)
        {
            RefreshValidationFromActiveScene();
        }
    }

    private void EnsureProjectScriptClassesAreFresh()
    {
        var fileSystem = EditorInterface.Singleton.GetResourceFilesystem();
        if (fileSystem.IsScanning())
        {
            return;
        }

        var missingRequiredClass = false;
        foreach (var className in RequiredGlobalScriptClasses)
        {
            if (HasGlobalScriptClass(className))
            {
                continue;
            }

            missingRequiredClass = true;
            break;
        }

        var staleLegacyClass = HasGlobalScriptClass("RLNetworkConfig");
        if (!missingRequiredClass && !staleLegacyClass)
        {
            return;
        }

        fileSystem.ScanSources();
    }

    private void RegisterCustomTypes()
    {
        var agentScript = GD.Load<Script>(AgentScriptPath);
        var agent3DScript = GD.Load<Script>(Agent3DScriptPath);
        var academyScript = GD.Load<Script>(AcademyScriptPath);
        var hpoOrchestratorScript = GD.Load<Script>(HpoOrchestratorScriptPath);
        var denseLayerDefScript = GD.Load<Script>(DenseLayerDefScriptPath);
        var dropoutLayerDefScript = GD.Load<Script>(DropoutLayerDefScriptPath);
        var flattenLayerDefScript = GD.Load<Script>(FlattenLayerDefScriptPath);
        var layerNormDefScript = GD.Load<Script>(LayerNormDefScriptPath);
        var lstmLayerDefScript = GD.Load<Script>(LstmLayerDefScriptPath);
        var gruLayerDefScript = GD.Load<Script>(GruLayerDefScriptPath);
        if (agentScript is not null)
        {
            AddCustomType(nameof(RLAgent2D), nameof(Node2D), agentScript, null);
        }

        if (agent3DScript is not null)
        {
            AddCustomType(nameof(RLAgent3D), nameof(Node3D), agent3DScript, null);
        }

        if (academyScript is not null)
        {
            AddCustomType(nameof(RLAcademy), nameof(Node), academyScript, null);
        }

        if (hpoOrchestratorScript is not null)
        {
            AddCustomType(nameof(RLHPOOrchestrator), nameof(Node), hpoOrchestratorScript, null);
        }

        if (denseLayerDefScript is not null)
        {
            AddCustomType(nameof(RLDenseLayerDef), nameof(Resource), denseLayerDefScript, null);
        }

        if (dropoutLayerDefScript is not null)
        {
            AddCustomType(nameof(RLDropoutLayerDef), nameof(Resource), dropoutLayerDefScript, null);
        }

        if (flattenLayerDefScript is not null)
        {
            AddCustomType(nameof(RLFlattenLayerDef), nameof(Resource), flattenLayerDefScript, null);
        }

        if (layerNormDefScript is not null)
        {
            AddCustomType(nameof(RLLayerNormDef), nameof(Resource), layerNormDefScript, null);
        }

        if (lstmLayerDefScript is not null)
        {
            AddCustomType(nameof(RLLstmLayerDef), nameof(Resource), lstmLayerDefScript, null);
        }

        if (gruLayerDefScript is not null)
        {
            AddCustomType(nameof(RLGruLayerDef), nameof(Resource), gruLayerDefScript, null);
        }
    }

    private void UnregisterCustomTypes()
    {
        RemoveCustomType(nameof(RLAgent2D));
        RemoveCustomType(nameof(RLAgent3D));
        RemoveCustomType(nameof(RLAcademy));
        RemoveCustomType(nameof(RLHPOOrchestrator));
        RemoveCustomType(nameof(RLDenseLayerDef));
        RemoveCustomType(nameof(RLDropoutLayerDef));
        RemoveCustomType(nameof(RLFlattenLayerDef));
        RemoveCustomType(nameof(RLLayerNormDef));
        RemoveCustomType(nameof(RLLstmLayerDef));
        RemoveCustomType(nameof(RLGruLayerDef));
    }

    private void RefreshValidationFromActiveScene()
    {
        if (_setupDock is null)
        {
            return;
        }

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _lastValidation = null;
            _lastValidationSignature = string.Empty;
            _setupDock.SetScenePath(string.Empty);
            _setupDock.SetValidationSummary("No active scene. Open a scene or set a main scene in Project Settings.");
            _setupDock.SetConfigSummary(string.Empty, string.Empty, string.Empty);
            return;
        }

        _setupDock.SetScenePath(scenePath);
        var validation = ValidateSceneSafely(scenePath, "auto validation");
        _lastValidationSignature = BuildValidationSignature(EditorInterface.Singleton.GetEditedSceneRoot());
        UpdateValidationUi(validation);
    }

    private void OnStartTrainingRequested()
    {
        if (_setupDock is null)
        {
            return;
        }

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock.SetLaunchStatus("No active scene or main scene configured.");
            return;
        }

        var validation = ValidateSceneSafely(scenePath, "training launch");
        UpdateValidationUi(validation);
        if (!validation.IsValid)
        {
            _setupDock.SetLaunchStatus("Training launch blocked by scene validation errors.");
            return;
        }

        var manifest = TrainingLaunchManifest.CreateDefault();
        var runPrefix = SanitizeRunPrefix(validation.RunPrefix);
        var runIdPrefix = string.IsNullOrWhiteSpace(runPrefix) ? "run" : runPrefix;
        var checkpointFileName = string.IsNullOrWhiteSpace(runPrefix) ? "checkpoint.json" : $"{runPrefix}_checkpoint.json";
        manifest.ScenePath = scenePath;
        manifest.AcademyNodePath = validation.AcademyPath;
        manifest.RunId = $"{runIdPrefix}_{BuildRunTimestampSuffix()}";
        manifest.RunDirectory = $"res://RL-Agent-Training/runs/{manifest.RunId}";
        manifest.CheckpointPath = $"{manifest.RunDirectory}/{checkpointFileName}";
        manifest.TrainingConfigPath = validation.TrainingConfigPath;
        manifest.NetworkConfigPath = validation.NetworkConfigPath;
        manifest.MetricsPath = $"{manifest.RunDirectory}/metrics.jsonl";
        manifest.StatusPath = $"{manifest.RunDirectory}/status.json";
        manifest.CheckpointInterval = validation.CheckpointInterval;
        manifest.SimulationSpeed = validation.SimulationSpeed;
        manifest.ActionRepeat = validation.ActionRepeat;
        manifest.BatchSize = Math.Max(1, validation.BatchSize);

        var writeError = manifest.SaveToUserStorage();
        if (writeError != Error.Ok)
        {
            _setupDock.SetLaunchStatus($"Failed to write training manifest: {writeError}");
            return;
        }

        _setupDock.SetLaunchStatus($"Launching {manifest.RunId}\n{ProjectSettings.GlobalizePath(manifest.RunDirectory)}");

        // Write initial meta.json so the dashboard can show policy-group names on export.
        WriteRunMeta(manifest.RunId, validation.ExportNames, validation.ExportGroups, validation.HasCurriculum);

        var bootstrapScene = "res://addons/rl-agent-plugin/Scenes/Bootstrap/TrainingBootstrap.tscn";
        EditorInterface.Singleton.PlayCustomScene(bootstrapScene);
        _launchedTrainingRun = true;
        _launchedQuickTestRun = false;
        _activeStatusPath = manifest.StatusPath;

        // Defer the dashboard handoff until after the editor finishes switching scenes.
        CallDeferred(nameof(NotifyDashboardTrainingStartedDeferred), manifest.RunId);
    }

    private void OnQuickTestRequested()
    {
        if (_setupDock is null)
        {
            return;
        }

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock.SetLaunchStatus("No active scene or main scene configured.");
            return;
        }

        var validation = ValidateSceneSafely(scenePath, "quick test launch");
        UpdateValidationUi(validation);
        if (!validation.IsValid)
        {
            _setupDock.SetLaunchStatus("Quick test blocked by scene validation errors.");
            return;
        }

        if (validation.HasSelfPlayPairings)
        {
            _setupDock.SetLaunchStatus(
                "Quick test does not support self-play scenes yet because it forces BatchSize = 1. " +
                "Use Start Training for self-play setups.");
            return;
        }

        var manifest = TrainingLaunchManifest.CreateDefault();
        var runPrefixBase = SanitizeRunPrefix(validation.RunPrefix);
        var runPrefix = string.IsNullOrWhiteSpace(runPrefixBase)
            ? "quick_test"
            : $"{runPrefixBase}_quick_test";
        manifest.ScenePath = scenePath;
        manifest.AcademyNodePath = validation.AcademyPath;
        manifest.RunId = $"{runPrefix}_{BuildRunTimestampSuffix()}";
        manifest.RunDirectory = "user://rl-agent-plugin/quick_test";
        manifest.CheckpointPath = string.Empty;
        manifest.TrainingConfigPath = validation.TrainingConfigPath;
        manifest.NetworkConfigPath = validation.NetworkConfigPath;
        manifest.MetricsPath = string.Empty;
        manifest.StatusPath = $"{manifest.RunDirectory}/status.json";
        manifest.CheckpointInterval = validation.CheckpointInterval;
        manifest.SimulationSpeed = 1.0f;
        manifest.ActionRepeat = validation.ActionRepeat;
        manifest.BatchSize = 1;
        manifest.QuickTestMode = true;
        manifest.QuickTestEpisodeLimit = QuickTestEpisodeLimit;
        manifest.QuickTestShowSpyOverlay = true;

        var writeError = manifest.SaveToUserStorage();
        if (writeError != Error.Ok)
        {
            _setupDock.SetLaunchStatus($"Failed to write quick-test manifest: {writeError}");
            return;
        }

        _activeStatusPath = manifest.StatusPath;
        _setupDock.SetLaunchStatus(
            $"Launching quick test: {QuickTestEpisodeLimit} episodes, batch=1, speed=1.0. Spy overlay enabled.");

        var bootstrapScene = "res://addons/rl-agent-plugin/Scenes/Bootstrap/TrainingBootstrap.tscn";
        EditorInterface.Singleton.PlayCustomScene(bootstrapScene);
        _launchedQuickTestRun = true;
        _launchedTrainingRun = false;
    }

    private void OnRunInferenceRequested()
    {
        if (_setupDock is null) return;

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock.SetLaunchStatus("No active scene or main scene configured.");
            return;
        }

        var validation = ValidateSceneSafely(scenePath, "inference launch");

        // We only need an academy and at least one inference-capable agent; full training validation is not required.
        if (string.IsNullOrWhiteSpace(validation.AcademyPath))
        {
            _setupDock.SetLaunchStatus("Inference launch failed: no RLAcademy found in the scene.");
            return;
        }

        if (validation.InferenceAgentCount == 0)
        {
            _setupDock.SetLaunchStatus("Inference launch failed: no Inference or Auto agents with a valid .rlmodel were found.");
            return;
        }

        var manifest = new InferenceLaunchManifest
        {
            ScenePath       = scenePath,
            AcademyNodePath = validation.AcademyPath,
            SimulationSpeed = 1.0f,
            ActionRepeat    = validation.ActionRepeat,
        };

        var writeError = manifest.SaveToUserStorage();
        if (writeError != Error.Ok)
        {
            _setupDock.SetLaunchStatus($"Failed to write inference manifest: {writeError}");
            return;
        }

        _setupDock.SetLaunchStatus($"Launching inference mode for {validation.InferenceAgentCount} configured agent(s).");

        var inferenceBootstrapScene = "res://addons/rl-agent-plugin/Scenes/Bootstrap/InferenceBootstrap.tscn";
        EditorInterface.Singleton.PlayCustomScene(inferenceBootstrapScene);
    }

    private void OnValidateSceneRequested()
    {
        if (_setupDock is null)
        {
            return;
        }

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock.SetLaunchStatus("No active scene or main scene configured.");
            return;
        }

        var validation = ValidateSceneSafely(scenePath, "manual validation");
        _lastValidationSignature = BuildValidationSignature(EditorInterface.Singleton.GetEditedSceneRoot());
        UpdateValidationUi(validation);
        _setupDock.SetLaunchStatus(validation.IsValid
            ? "Scene validation passed."
            : $"Scene validation failed: {validation.Errors[0]}");
    }

    private static void WriteRunMeta(string runId, IReadOnlyList<string> agentNames, IReadOnlyList<string> agentGroups, bool hasCurriculum)
    {
        try
        {
            var runDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{runId}");
            System.IO.Directory.CreateDirectory(runDirAbs);

            var agentArr = new Godot.Collections.Array();
            foreach (var name in agentNames) agentArr.Add(Variant.From(name));

            var groupArr = new Godot.Collections.Array();
            foreach (var g in agentGroups) groupArr.Add(Variant.From(g));

            var d = new Godot.Collections.Dictionary
            {
                { "display_name",  "" },
                { "agent_names",   agentArr },
                { "agent_groups",  groupArr },
                { "has_curriculum", hasCurriculum },
            };

            System.IO.File.WriteAllText(
                System.IO.Path.Combine(runDirAbs, "meta.json"),
                Json.Stringify(d));
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Failed to write meta.json: {ex.Message}");
        }
    }

    private static string BuildRunTimestampSuffix()
    {
        var rounded = (long)Math.Round(Time.GetUnixTimeFromSystem(), 0, MidpointRounding.AwayFromZero);
        return rounded.ToString();
    }

    private void OnStopTrainingRequested()
    {
        if (!_launchedTrainingRun && !_launchedQuickTestRun)
        {
            return;
        }

        MarkActiveRunStopped();
        EditorInterface.Singleton.StopPlayingScene();
        _dashboard?.OnTrainingStopped();
        var stoppedQuickTest = _launchedQuickTestRun;
        _launchedTrainingRun = false;
        _launchedQuickTestRun = false;
        _activeStatusPath = string.Empty;
        _setupDock?.SetLaunchStatus(stoppedQuickTest ? "Quick test stopped." : "Training run stopped.");
    }

    private void MarkActiveRunStopped()
    {
        if (string.IsNullOrWhiteSpace(_activeStatusPath))
        {
            return;
        }

        try
        {
            var absolutePath = ProjectSettings.GlobalizePath(_activeStatusPath);
            if (!System.IO.File.Exists(absolutePath))
            {
                return;
            }

            var variant = Json.ParseString(System.IO.File.ReadAllText(absolutePath));
            if (variant.VariantType != Variant.Type.Dictionary)
            {
                return;
            }

            var payload = variant.AsGodotDictionary();
            payload["status"] = "stopped";
            payload["message"] = "Training stopped by editor.";
            System.IO.File.WriteAllText(absolutePath, Json.Stringify(payload));
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Failed to mark run as stopped: {ex.Message}");
        }
    }

    private void UpdateValidationUi(TrainingSceneValidation validation)
    {
        _lastValidation = validation;
        _setupDock?.SetValidationSummary(validation.BuildSummary(), validation.IsValid);
        _setupDock?.SetConfigSummary(
            validation.TrainingConfigPath,
            validation.NetworkConfigPath,
            validation.InferenceAgentCount > 0 ? $"{validation.InferenceAgentCount} agent(s) configured" : string.Empty);
    }

    private void NotifyDashboardTrainingStartedDeferred(string runId)
    {
        if (string.IsNullOrWhiteSpace(runId))
        {
            return;
        }

        _dashboard?.OnTrainingStarted(runId);
    }

    private static string ResolveTrainingScenePath()
    {
        // Prefer the currently open/edited scene.
        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (editedRoot is not null && !string.IsNullOrWhiteSpace(editedRoot.SceneFilePath))
        {
            return editedRoot.SceneFilePath;
        }

        // Fall back to the project's configured main scene.
        var mainScene = ProjectSettings.GetSetting("application/run/main_scene").ToString();
        return mainScene ?? string.Empty;
    }

    private static TrainingSceneValidation ValidateSceneSafely(string scenePath, string operation)
    {
        try
        {
            var validation = ValidateScene(scenePath);
            LogValidationMessages(operation, validation);
            return validation;
        }
        catch (Exception exception)
        {
            return BuildValidationCrashResult(scenePath, operation, exception);
        }
    }

    private static TrainingSceneValidation ValidateScene(string scenePath)
    {
        var validation = new TrainingSceneValidation
        {
            ScenePath = scenePath,
        };

        var packedScene = GD.Load<PackedScene>(scenePath);
        if (packedScene is null)
        {
            validation.Errors.Add($"Could not load scene: {scenePath}");
            return validation;
        }

        var root = packedScene.Instantiate();
        try
        {
            Node? academy = null;
            var curriculumConsumerCount = 0;
            // groupId → list of agent nodes
            var agentsByGroup = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<Node>>();
            var groupBindings = new System.Collections.Generic.Dictionary<string, ResolvedPolicyGroupBinding>();

            Traverse(root, node =>
            {
                if (IsAcademyNode(node))
                {
                    if (academy is null)
                    {
                        academy = node;
                    }
                    else
                    {
                        validation.Errors.Add("More than one RLAcademy was found. Only one academy is supported.");
                    }
                }

                if (IsAgentNode(node))
                {
                    validation.AgentNames.Add(node.Name.ToString());

                    var controlMode = ReadAgentControlMode(node);
                    var binding = RLPolicyGroupBindingResolver.Resolve(root, node);
                    validation.AgentGroups.Add(binding?.SafeGroupId ?? string.Empty);

                    if (controlMode == RLAgentControlMode.Train || controlMode == RLAgentControlMode.Auto)
                    {
                        validation.TrainAgentCount += 1;

                        if (binding is null)
                        {
                            var modeLabel = controlMode == RLAgentControlMode.Auto ? "Auto" : "Train";
                            validation.Errors.Add($"Agent '{root.GetPathTo(node)}' is in {modeLabel} mode but has no PolicyGroupConfig assigned.");
                        }
                        else
                        {
                            if (!agentsByGroup.ContainsKey(binding.BindingKey))
                            {
                                agentsByGroup[binding.BindingKey] = new System.Collections.Generic.List<Node>();
                                groupBindings[binding.BindingKey] = binding;
                            }

                            agentsByGroup[binding.BindingKey].Add(node);
                        }
                    }

                    if (controlMode == RLAgentControlMode.Inference)
                    {
                        if (ValidateInferenceModel(node, root, validation, requireConfiguredModel: true))
                        {
                            validation.InferenceAgentCount += 1;
                        }
                    }
                    else if (controlMode == RLAgentControlMode.Auto)
                    {
                        if (ValidateInferenceModel(node, root, validation, requireConfiguredModel: false))
                        {
                            validation.InferenceAgentCount += 1;
                        }
                    }
                }

                if (IsCurriculumConsumerNode(node))
                {
                    curriculumConsumerCount += 1;
                }
            });

            if (academy is null)
            {
                validation.Errors.Add("No RLAcademy node was found in the selected scene.");
            }
            else
            {
                validation.AcademyPath = root.GetPathTo(academy).ToString();
                var trainingConfigRes = ReadResourceProperty(academy, "TrainingConfig") as RLTrainingConfig;
                var runConfig = ReadResourceProperty(academy, "RunConfig");

                validation.TrainingConfigPath = trainingConfigRes?.ResourcePath ?? string.Empty;
                validation.NetworkConfigPath = validation.TrainingConfigPath;
                validation.RunPrefix = ReadStringProperty(runConfig, "RunPrefix");
                validation.CheckpointInterval = ReadIntProperty(runConfig, "CheckpointInterval", 10);
                validation.SimulationSpeed = ReadFloatProperty(runConfig, "SimulationSpeed", 1.0f);
                validation.ActionRepeat = ReadIntProperty(runConfig, "ActionRepeat", 1);
                validation.BatchSize = ReadIntProperty(runConfig, "BatchSize", 1);
                validation.EnableSpyOverlay = ReadBoolProperty(academy, "EnableSpyOverlay");
                validation.HasCurriculum = ReadResourceProperty(academy, "Curriculum") is not null;

                if (validation.HasCurriculum && curriculumConsumerCount == 0)
                {
                    validation.Errors.Add(
                        "RLAcademy has Curriculum assigned, but no scene node implements IRLCurriculumConsumer.");
                }

                if (trainingConfigRes is null)
                {
                    validation.Errors.Add("RLAcademy is missing an RLTrainingConfig resource.");
                }

                if (trainingConfigRes is not null && trainingConfigRes.Algorithm is null)
                {
                    validation.Errors.Add("RLAcademy.TrainingConfig has no Algorithm assigned.");
                }

                if (validation.TrainAgentCount == 0)
                {
                    validation.Errors.Add("No Train or Auto mode agents were found in the selected scene.");
                }

                var resolvedTrainerConfig = trainingConfigRes?.ToTrainerConfig();
                var algorithm = resolvedTrainerConfig?.Algorithm ?? RLAlgorithmKind.Custom;

                var typedTrainAgents = new List<IRLAgent>();
                foreach (var groupNodes in agentsByGroup.Values)
                {
                    foreach (var groupNode in groupNodes)
                    {
                        if (groupNode is IRLAgent typedAgent)
                        {
                            typedTrainAgents.Add(typedAgent);
                        }
                    }
                }

                var typedAcademy = academy as RLAcademy;
                var observationInference = typedAcademy is not null
                    ? typedAcademy.InferObservationSizes(RLAgentControlMode.Train)
                    : ObservationSizeInference.Infer(root, typedTrainAgents);
                foreach (var error in observationInference.Errors)
                {
                    validation.Errors.Add(error);
                }

                var groupSummaryByBindingKey = new Dictionary<string, PolicyGroupSummary>(StringComparer.Ordinal);

                // Per-group validation
                foreach (var (groupId, groupNodes) in agentsByGroup)
                {
                    var binding = groupBindings[groupId];
                    var firstNode = groupNodes[0];
                    var firstObservationSize = observationInference.GroupSizes.TryGetValue(binding.BindingKey, out var inferredObservationSize)
                        ? inferredObservationSize
                        : ReadAgentObservationSize(firstNode, root, validation);
                    var firstActionCount = ReadAgentActionCount(firstNode);
                    var firstIsDiscrete = SupportsOnlyDiscreteActions(firstNode);
                    var firstContinuousDims = ReadAgentContinuousDims(firstNode);

                    var groupSummary = new PolicyGroupSummary
                    {
                        GroupId = binding.DisplayName,
                        AgentCount = groupNodes.Count,
                        ObservationSize = firstObservationSize,
                        ActionCount = firstActionCount,
                        IsContinuous = !firstIsDiscrete,
                        ContinuousActionDimensions = firstContinuousDims,
                        UsesExplicitConfig = binding.UsesExplicitConfig,
                        PolicyConfigPath = binding.ConfigPath,
                        SelfPlay = false,
                        HistoricalOpponentRate = 0f,
                        FrozenCheckpointInterval = 10,
                    };
                    foreach (var node in groupNodes)
                    {
                        groupSummary.AgentPaths.Add(root.GetPathTo(node).ToString());
                    }

                    validation.PolicyGroups.Add(groupSummary);
                    groupSummaryByBindingKey[groupId] = groupSummary;

                    validation.ExportNames.Add(ResolvePolicyExportName(binding));
                    validation.ExportGroups.Add(binding.SafeGroupId);

                    if (algorithm == RLAlgorithmKind.PPO && firstActionCount == 0 && firstIsDiscrete)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': define at least one discrete action.");
                    }

                    // Skip when firstObservationSize == -1 (cast unavailable; not a real empty obs vector).
                    if (firstObservationSize == 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': could not infer a non-zero observation size.");
                    }

                    if (algorithm == RLAlgorithmKind.DQN && firstContinuousDims > 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': DQN supports discrete actions only; this group has continuous action dimensions.");
                    }

                    if (algorithm == RLAlgorithmKind.DQN && firstActionCount <= 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': DQN requires at least one discrete action.");
                    }

                    if (algorithm == RLAlgorithmKind.SAC
                        && firstActionCount >= 0
                        && firstContinuousDims >= 0
                        && firstActionCount > 0
                        && firstContinuousDims > 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': SAC does not support mixing discrete and continuous actions.");
                    }

                    if (algorithm == RLAlgorithmKind.SAC
                        && firstActionCount >= 0
                        && firstContinuousDims >= 0
                        && firstActionCount > 0
                        && firstContinuousDims <= 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': SAC currently supports continuous-only actions; this group is discrete.");
                    }

                    if (algorithm == RLAlgorithmKind.SAC
                        && firstActionCount >= 0
                        && firstContinuousDims >= 0
                        && firstActionCount <= 0
                        && firstContinuousDims <= 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': SAC requires at least one continuous action dimension.");
                    }

                    if (algorithm == RLAlgorithmKind.MCTS && firstContinuousDims > 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': MCTS supports discrete actions only; this group has continuous action dimensions.");
                    }

                    if (algorithm == RLAlgorithmKind.MCTS && firstActionCount <= 0)
                    {
                        validation.Errors.Add($"Group '{groupSummary.GroupId}': MCTS requires at least one discrete action.");
                    }

                    // Custom: id must be provided (action-space validation is the trainer's responsibility)
                    if (algorithm == RLAlgorithmKind.Custom)
                    {
                        var customId = resolvedTrainerConfig?.CustomTrainerId ?? string.Empty;
                        if (string.IsNullOrWhiteSpace(customId))
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': Algorithm is Custom but CustomTrainerId is not set.");
                    }

                    ValidateRecurrentSupport(binding, groupSummary, algorithm, validation);

                    // All agents in group must be consistent
                    foreach (var node in groupNodes)
                    {
                        var nodePath = root.GetPathTo(node).ToString();
                        var observationSize = node is IRLAgent typedAgent
                            && observationInference.AgentSizes.TryGetValue(typedAgent, out var inferredAgentSize)
                                ? inferredAgentSize
                                : ReadAgentObservationSize(node, root, validation);
                        var actionCount = ReadAgentActionCount(node);
                        var isDiscrete = SupportsOnlyDiscreteActions(node);
                        if (isDiscrete != firstIsDiscrete)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {nodePath}: all agents in a group must use the same action type (discrete vs continuous).");
                        }

                        if (firstObservationSize >= 0 && observationSize >= 0 && observationSize != firstObservationSize)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {nodePath}: all agents must emit the same observation vector length.");
                        }

                        if (algorithm == RLAlgorithmKind.PPO && isDiscrete && firstActionCount >= 0 && actionCount >= 0 && actionCount != firstActionCount)
                        {
                            validation.Errors.Add($"Group '{groupSummary.GroupId}': {nodePath}: all agents must share the same discrete action count.");
                        }
                    }
                }

                var configuredPairings = typedAcademy?.GetResolvedSelfPlayPairings() ?? new List<RLPolicyPairingConfig>();
                validation.HasSelfPlayPairings = configuredPairings.Count > 0;
                var requiredBatchCopies = ValidateSelfPlayPairings(configuredPairings, groupBindings, groupSummaryByBindingKey, validation);
                if (validation.BatchSize < requiredBatchCopies)
                {
                    validation.Errors.Add(
                        $"BatchSize must be at least {requiredBatchCopies} to support the configured self-play rival groups.");
                }

                // For backward compat, set ExpectedActionCount from first group
                if (agentsByGroup.Count > 0)
                {
                    validation.ExpectedActionCount = validation.PolicyGroups.Count > 0
                        ? validation.PolicyGroups[0].ActionCount
                        : 0;
                }
            }

            validation.IsValid = validation.Errors.Count == 0;
            return validation;
        }
        finally
        {
            root.QueueFree();
        }
    }

    private static void Traverse(Node node, System.Action<Node> visitor)
    {
        visitor(node);
        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
            {
                Traverse(childNode, visitor);
            }
        }
    }

    private static bool IsAcademyNode(Node node)
    {
        if (node is RLAcademy)
        {
            return true;
        }

        var managedType = ResolveManagedScriptType(node);
        if (managedType is not null && typeof(RLAcademy).IsAssignableFrom(managedType))
        {
            return true;
        }

        return ScriptInheritsPath(GetNodeScript(node), AcademyScriptPath);
    }

    private static bool IsAgentNode(Node node)
    {
        if (node is IRLAgent)
        {
            return true;
        }

        var managedType = ResolveManagedScriptType(node);
        if (managedType is not null && typeof(IRLAgent).IsAssignableFrom(managedType))
        {
            return true;
        }

        return ScriptInheritsPath(GetNodeScript(node), AgentScriptPath)
            || ScriptInheritsPath(GetNodeScript(node), Agent3DScriptPath);
    }

    private static bool IsCurriculumConsumerNode(Node node)
    {
        if (node is IRLCurriculumConsumer)
        {
            return true;
        }

        if (IsAgentNode(node))
        {
            return true;
        }

        var managedType = ResolveManagedScriptType(node);
        if (managedType is not null && typeof(IRLCurriculumConsumer).IsAssignableFrom(managedType))
        {
            return true;
        }

        return node.HasMethod(nameof(IRLCurriculumConsumer.NotifyCurriculumProgress));
    }

    private static int ReadAgentActionCount(Node node)
    {
        if (node is IRLAgent agent)
            return agent.GetDiscreteActionCount();
        // C# cast unavailable in editor context (e.g. assembly not yet fully loaded).
        // Return -1 so callers can skip checks rather than produce false-positive errors.
        return -1;
    }

    private static int ReadAgentContinuousDims(Node node)
    {
        if (node is IRLAgent agent)
            return agent.GetContinuousActionDimensions();
        // C# cast unavailable in editor context (e.g. assembly not yet fully loaded).
        // Return -1 so callers can skip checks rather than produce false-positive errors.
        return -1;
    }

    private static int ReadAgentObservationSize(Node node, Node root, TrainingSceneValidation validation)
    {
        if (node is not IRLAgent agent)
        {
            // -1 = unknown (cast unavailable). 0 = cast succeeded but returned empty obs.
            return -1;
        }

        try
        {
            if (!ObservationSizeInference.TryInferAgentObservationSize(agent, out var observationSize, out var error))
            {
                validation.Errors.Add($"Agent '{root.GetPathTo(node)}': observation inference failed: {error}");
                return 0;
            }

            return observationSize;
        }
        catch (Exception exception)
        {
            validation.Errors.Add($"Agent '{root.GetPathTo(node)}': observation inference failed: {exception.Message}");
            return 0;
        }
    }

    private static bool SupportsOnlyDiscreteActions(Node node)
    {
        if (node is IRLAgent agent)
            return agent.SupportsOnlyDiscreteActions();
        // Cast unavailable: assume discrete to avoid false-positive "PPO requires discrete-only" errors.
        return IsAgentNode(node);
    }

    private static bool ValidateInferenceModel(Node node, Node root, TrainingSceneValidation validation, bool requireConfiguredModel)
    {
        var errorCount = validation.Errors.Count;
        var nodePath = root.GetPathTo(node).ToString();

        // Try typed C# access first; fall back to Godot property access when the managed type
        // cannot be resolved from the editor plugin context (packedScene.Instantiate() returns
        // plain GodotObjects when the game assembly isn't yet visible to the plugin assembly).
        string modelPath;
        int discreteCount;
        int continuousDims;
        var hasTypedAgent = node is IRLAgent;

        if (hasTypedAgent)
        {
            var agent = (IRLAgent)node;
            modelPath = agent.GetInferenceModelPath();
            discreteCount = agent.GetDiscreteActionCount();
            continuousDims = agent.GetContinuousActionDimensions();
        }
        else
        {
            // C# cast unavailable — read InferenceModelPath via the PolicyGroupConfig sub-resource.
            var policyGroupConfig = ReadResourceProperty(node, "PolicyGroupConfig");
            var rawPath = ReadStringProperty(policyGroupConfig, "InferenceModelPath");
            // Resolve UID paths the same way RLAgent2D.NormalizeResourcePath does.
            if (rawPath.StartsWith("uid://", StringComparison.Ordinal))
            {
                var resolved = ResourceUid.EnsurePath(rawPath);
                rawPath = !string.IsNullOrWhiteSpace(resolved) ? resolved : rawPath;
            }
            modelPath = rawPath;
            // Action-space dimensions are unknown without the typed interface.
            // Use -1 as "unknown" so the checks below are skipped rather than producing false positives.
            discreteCount = -1;
            continuousDims = -1;
        }

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            if (requireConfiguredModel)
            {
                validation.Errors.Add($"Agent '{nodePath}' is in Inference mode but has no .rlmodel path.");
            }

            return false;
        }

        if (!modelPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase))
        {
            if (requireConfiguredModel)
                validation.Errors.Add($"Agent '{nodePath}': inference model path must point to a .rlmodel file.");
            return false;
        }

        var checkpoint = RLModelLoader.LoadFromFile(modelPath);

        if (checkpoint is null)
        {
            if (requireConfiguredModel)
                validation.Errors.Add($"Agent '{nodePath}': failed to load inference model '{modelPath}'.");
            return false;
        }

        // For Auto-mode agents the model is optional — only add blocking errors when the model
        // is explicitly required (Inference mode). Mismatch errors on an optional model should
        // not prevent training from starting.
        if (!requireConfiguredModel)
            return true;

        var observationSize = hasTypedAgent ? ReadAgentObservationSize(node, root, validation) : -1;

        if (observationSize > 0 && checkpoint.ObservationSize > 0 && checkpoint.ObservationSize != observationSize)
        {
            validation.Errors.Add(
                $"Agent '{nodePath}': model observation size {checkpoint.ObservationSize} " +
                $"does not match agent observation size {observationSize}.");
        }

        if (discreteCount >= 0 && checkpoint.DiscreteActionCount > 0 && checkpoint.DiscreteActionCount != discreteCount)
        {
            validation.Errors.Add(
                $"Agent '{nodePath}': model discrete action count {checkpoint.DiscreteActionCount} " +
                $"does not match agent count {discreteCount}.");
        }

        if (continuousDims >= 0 && checkpoint.ContinuousActionDimensions > 0 && checkpoint.ContinuousActionDimensions != continuousDims)
        {
            validation.Errors.Add(
                $"Agent '{nodePath}': model continuous action dims {checkpoint.ContinuousActionDimensions} " +
                $"does not match agent dims {continuousDims}.");
        }

        return validation.Errors.Count == errorCount;
    }

    private static RLAgentControlMode ReadAgentControlMode(Node node)
    {
        if (node is IRLAgent agent)
        {
            return agent.ControlMode;
        }

        var variant = node.Get("ControlMode");
        return variant.VariantType == Variant.Type.Int
            ? (RLAgentControlMode)(int)variant
            : RLAgentControlMode.Train;
    }

    private static Resource? ReadResourceProperty(Node node, string propertyName)
    {
        var variant = node.Get(propertyName);
        return variant.VariantType == Variant.Type.Object ? variant.AsGodotObject() as Resource : null;
    }

    private static string ResolveGroupKeyForPolicyConfig(
        IReadOnlyDictionary<string, ResolvedPolicyGroupBinding> groupBindings,
        RLPolicyGroupConfig config)
    {
        foreach (var (groupId, binding) in groupBindings)
        {
            if (ReferenceEquals(binding.Config, config))
            {
                return groupId;
            }
        }

        if (!string.IsNullOrWhiteSpace(config.ResourcePath))
        {
            foreach (var (groupId, binding) in groupBindings)
            {
                if (string.Equals(binding.ConfigPath, config.ResourcePath, StringComparison.Ordinal))
                {
                    return groupId;
                }
            }
        }

        return string.Empty;
    }

    private static int ValidateSelfPlayPairings(
        IReadOnlyList<RLPolicyPairingConfig> configuredPairings,
        IReadOnlyDictionary<string, ResolvedPolicyGroupBinding> groupBindings,
        IDictionary<string, PolicyGroupSummary> groupSummaryByBindingKey,
        TrainingSceneValidation validation)
    {
        var pairedGroups = new HashSet<string>(StringComparer.Ordinal);
        var learnerCountsByPair = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var pairing in configuredPairings)
        {
            var groupA = pairing.ResolvedGroupA;
            var groupB = pairing.ResolvedGroupB;
            var groupAId = groupA is null ? string.Empty : ResolveGroupKeyForPolicyConfig(groupBindings, groupA);
            var groupBId = groupB is null ? string.Empty : ResolveGroupKeyForPolicyConfig(groupBindings, groupB);
            var pairingName = ResolvePairingDisplayName(pairing);

            if (string.IsNullOrWhiteSpace(groupAId) || string.IsNullOrWhiteSpace(groupBId))
            {
                validation.Errors.Add($"Pairing '{pairingName}' must reference two train-mode policy groups used in this scene.");
                continue;
            }

            if (string.Equals(groupAId, groupBId, StringComparison.Ordinal))
            {
                validation.Errors.Add($"Pairing '{pairingName}' cannot pair a group against itself.");
                continue;
            }

            if (!pairing.TrainGroupA && !pairing.TrainGroupB)
            {
                validation.Errors.Add($"Pairing '{pairingName}' must train at least one side.");
                continue;
            }

            if (!pairedGroups.Add(groupAId) || !pairedGroups.Add(groupBId))
            {
                validation.Errors.Add(
                    $"Pairing '{pairingName}' reuses a policy group that is already part of another pairing. v1 supports disjoint 2-group pairings only.");
                continue;
            }

            var opponentDisplayName = groupBindings[groupBId].DisplayName;
            if (pairing.TrainGroupA && groupSummaryByBindingKey.TryGetValue(groupAId, out var groupASummary))
            {
                groupASummary.SelfPlay = true;
                groupASummary.OpponentGroupId = opponentDisplayName;
                groupASummary.HistoricalOpponentRate = pairing.HistoricalOpponentRate;
                groupASummary.FrozenCheckpointInterval = pairing.FrozenCheckpointInterval;
            }

            var groupADisplayName = groupBindings[groupAId].DisplayName;
            if (pairing.TrainGroupB && groupSummaryByBindingKey.TryGetValue(groupBId, out var groupBSummary))
            {
                groupBSummary.SelfPlay = true;
                groupBSummary.OpponentGroupId = groupADisplayName;
                groupBSummary.HistoricalOpponentRate = pairing.HistoricalOpponentRate;
                groupBSummary.FrozenCheckpointInterval = pairing.FrozenCheckpointInterval;
            }

            var pairKey = string.CompareOrdinal(groupAId, groupBId) <= 0
                ? $"{groupAId}|{groupBId}"
                : $"{groupBId}|{groupAId}";
            learnerCountsByPair[pairKey] = (pairing.TrainGroupA ? 1 : 0) + (pairing.TrainGroupB ? 1 : 0);
        }

        return learnerCountsByPair.Count == 0 ? 1 : learnerCountsByPair.Values.Max();
    }

    private static void ValidateRecurrentSupport(
        ResolvedPolicyGroupBinding binding,
        PolicyGroupSummary groupSummary,
        RLAlgorithmKind algorithm,
        TrainingSceneValidation validation)
    {
        var graph = binding.Config?.ResolvedNetworkGraph;
        if (graph is null)
            return;

        var recurrentLayerCount = CountRecurrentLayers(graph);
        if (recurrentLayerCount == 0)
            return;

        if (!NativeLayerSupport.IsAvailable)
        {
            validation.Errors.Add(
                $"Group '{groupSummary.GroupId}': LSTM/GRU layers require the native GDExtension library, but native layers are not available in the current editor/runtime.");
        }

        if (recurrentLayerCount > 1)
        {
            validation.Errors.Add(
                $"Group '{groupSummary.GroupId}': only one recurrent trunk layer is currently supported end-to-end. " +
                $"This network config contains {recurrentLayerCount} recurrent layers.");
        }

        if (algorithm is RLAlgorithmKind.DQN or RLAlgorithmKind.SAC)
        {
            validation.Errors.Add(
                $"Group '{groupSummary.GroupId}': {algorithm} does not support recurrent LSTM/GRU trunk layers. Use PPO or A2C for recurrent policies.");
        }
    }

    private static int CountRecurrentLayers(RLNetworkGraph graph)
    {
        var count = 0;
        foreach (var resource in graph.TrunkLayers)
        {
            if (resource is RLLstmLayerDef or RLGruLayerDef)
                count++;
        }

        return count;
    }

    private static string ResolvePairingDisplayName(RLPolicyPairingConfig pairing)
    {
        if (!string.IsNullOrWhiteSpace(pairing.PairingId))
        {
            return pairing.PairingId.Trim();
        }

        var groupA = pairing.ResolvedGroupA?.ResourceName;
        var groupB = pairing.ResolvedGroupB?.ResourceName;
        if (!string.IsNullOrWhiteSpace(groupA) && !string.IsNullOrWhiteSpace(groupB))
        {
            return $"{groupA.Trim()} vs {groupB.Trim()}";
        }

        return "<unnamed pairing>";
    }

    private static string ResolvePolicyExportName(ResolvedPolicyGroupBinding binding)
    {
        if (!string.IsNullOrWhiteSpace(binding.Config?.AgentId))
        {
            return binding.Config.AgentId.Trim();
        }

        return binding.DisplayName;
    }

    private static string ReadStringProperty(Node node, string propertyName)
        => ReadStringProperty((GodotObject)node, propertyName);

    private static string ReadStringProperty(GodotObject? obj, string propertyName)
    {
        if (obj is null) return string.Empty;
        var variant = obj.Get(propertyName);
        return variant.VariantType == Variant.Type.String ? variant.AsString() : string.Empty;
    }

    private static int ReadIntProperty(Node node, string propertyName, int defaultValue)
        => ReadIntProperty((GodotObject)node, propertyName, defaultValue);

    private static int ReadIntProperty(GodotObject? obj, string propertyName, int defaultValue)
    {
        if (obj is null) return defaultValue;
        var variant = obj.Get(propertyName);
        return variant.VariantType == Variant.Type.Int ? (int)variant : defaultValue;
    }

    private static bool ReadBoolProperty(Node node, string propertyName)
        => ReadBoolProperty((GodotObject)node, propertyName);

    private static bool ReadBoolProperty(GodotObject? obj, string propertyName)
    {
        if (obj is null) return false;
        var variant = obj.Get(propertyName);
        return variant.VariantType == Variant.Type.Bool && (bool)variant;
    }

    private static float ReadFloatProperty(Node node, string propertyName, float defaultValue)
        => ReadFloatProperty((GodotObject)node, propertyName, defaultValue);

    private static float ReadFloatProperty(GodotObject? obj, string propertyName, float defaultValue)
    {
        if (obj is null) return defaultValue;
        var variant = obj.Get(propertyName);
        return variant.VariantType switch
        {
            Variant.Type.Float => (float)(double)variant,
            Variant.Type.Int => (int)variant,
            _ => defaultValue,
        };
    }

    private static Type? ResolveManagedScriptType(Node node)
    {
        if (node.GetType() != typeof(Node) && node.GetType() != typeof(Node2D) && node.GetType() != typeof(Node3D))
        {
            return node.GetType();
        }

        var script = GetNodeScript(node);
        if (script is null)
        {
            return null;
        }

        return ResolveManagedScriptType(script);
    }

    private static Type? ResolveManagedScriptType(Script script)
    {
        if (!string.IsNullOrWhiteSpace(script.ResourcePath)
            && ScriptTypeIndex.Value.TryGetValue(script.ResourcePath, out var managedType))
        {
            return managedType;
        }

        var baseScript = script.GetBaseScript() as Script;
        return baseScript is null ? null : ResolveManagedScriptType(baseScript);
    }

    private static Script? GetNodeScript(Node node)
    {
        var scriptVariant = node.GetScript();
        return scriptVariant.VariantType == Variant.Type.Object ? scriptVariant.AsGodotObject() as Script : null;
    }

    private static bool ScriptInheritsPath(Script? script, string targetPath)
    {
        var current = script;
        while (current is not null)        {
            if (current.ResourcePath == targetPath)
            {
                return true;
            }

            current = current.GetBaseScript() as Script;
        }

        return false;
    }

    private static string BuildValidationSignature(Node? editedRoot)
    {
        if (editedRoot is null)
        {
            return string.Empty;
        }

        var builder = new System.Text.StringBuilder();
        Traverse(editedRoot, node =>
        {
            if (IsAcademyNode(node))
            {
                var typedAcademy = node as RLAcademy;
                var runConfig = ReadResourceProperty(node, "RunConfig");
                builder.Append("academy|");
                builder.Append(node.GetPath());
                builder.Append('|');
                builder.Append(ReadStringProperty(runConfig, "RunPrefix"));
                builder.Append('|');
                builder.Append(ReadIntProperty(runConfig, "CheckpointInterval", 10));
                builder.Append('|');
                builder.Append(ReadIntProperty(runConfig, "ActionRepeat", 1));
                builder.Append('|');
                builder.Append(ReadIntProperty(node, "MaxEpisodeSteps", 0));
                builder.Append('|');
                builder.Append(ReadIntProperty(runConfig, "BatchSize", 1));
                builder.Append('|');
                builder.Append(ReadFloatProperty(runConfig, "SimulationSpeed", 1.0f));
                builder.Append('|');
                builder.Append(ReadBoolProperty(node, "EnableSpyOverlay"));
                builder.Append('|');
                builder.Append(ReadResourceProperty(node, "TrainingConfig")?.ResourcePath ?? string.Empty);
                if (typedAcademy is not null)
                {
                    foreach (var pairing in typedAcademy.GetResolvedSelfPlayPairings())
                    {
                        builder.Append('|');
                        builder.Append("pairing:");
                        builder.Append(pairing.PairingId);
                        builder.Append(':');
                        builder.Append(pairing.ResolvedGroupA?.ResourcePath ?? pairing.ResolvedGroupA?.ResourceName ?? string.Empty);
                        builder.Append(':');
                        builder.Append(pairing.ResolvedGroupB?.ResourcePath ?? pairing.ResolvedGroupB?.ResourceName ?? string.Empty);
                        builder.Append(':');
                        builder.Append(pairing.TrainGroupA ? 1 : 0);
                        builder.Append(':');
                        builder.Append(pairing.TrainGroupB ? 1 : 0);
                        builder.Append(':');
                        builder.Append(pairing.HistoricalOpponentRate);
                        builder.Append(':');
                        builder.Append(pairing.FrozenCheckpointInterval);
                    }
                }
                builder.AppendLine();
            }

            if (IsAgentNode(node))
            {
                var policyGroupConfig = ReadResourceProperty(node, "PolicyGroupConfig") as RLPolicyGroupConfig;
                builder.Append("agent|");
                builder.Append(node.GetPath());
                builder.Append('|');
                builder.Append((int)ReadAgentControlMode(node));
                builder.Append('|');
                builder.Append(policyGroupConfig?.ResourcePath ?? string.Empty);
                builder.Append('|');
                builder.Append(policyGroupConfig?.ResourceName ?? string.Empty);
                builder.Append('|');
                builder.Append(policyGroupConfig?.AgentId ?? string.Empty);
                builder.Append('|');
                builder.Append(policyGroupConfig?.MaxEpisodeSteps ?? 0);
                builder.Append('|');
                builder.Append(policyGroupConfig?.InferenceModelPath ?? string.Empty);
                builder.Append('|');
                builder.Append(ReadAgentActionCount(node));
                builder.Append('|');
                builder.Append(ReadAgentContinuousDims(node));
                builder.AppendLine();
            }
        });

        return builder.ToString();
    }

    private static Dictionary<string, Type> BuildScriptTypeIndex()
    {
        var index = new Dictionary<string, Type>(StringComparer.Ordinal);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            foreach (var type in SafeGetTypes(assembly))
            {
                foreach (var scriptPath in GetScriptPaths(type))
                {
                    if (!string.IsNullOrWhiteSpace(scriptPath))
                    {
                        index[scriptPath] = type;
                    }
                }
            }
        }

        return index;
    }

    private static IEnumerable<string> GetScriptPaths(MemberInfo member)
    {
        return member
            .GetCustomAttributes<ScriptPathAttribute>(false)
            .Select(attribute => attribute.Path)
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.Ordinal);
    }

    private static IEnumerable<Type> SafeGetTypes(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException exception)
        {
            return exception.Types.Where(type => type is not null)!;
        }
    }

    private static TrainingSceneValidation BuildValidationCrashResult(string scenePath, string operation, Exception exception)
    {
        var message = $"{operation} failed while validating {scenePath}: {exception.Message}";
        GD.PushError(message);
        GD.PrintErr(message);
        GD.PrintErr(exception.ToString());

        var validation = new TrainingSceneValidation
        {
            ScenePath = scenePath,
            IsValid = false,
        };
        validation.Errors.Add(message);
        return validation;
    }

    private static void LogValidationMessages(string _, TrainingSceneValidation __)
    {
        // Validation errors are surfaced in the RL Setup UI — no console spam needed.
    }

    private static string ReadCompletedQuickTestStatus(string statusPath)
    {
        if (string.IsNullOrWhiteSpace(statusPath))
        {
            return "Quick test finished.";
        }

        var absolutePath = ProjectSettings.GlobalizePath(statusPath);
        if (!System.IO.File.Exists(absolutePath))
        {
            return "Quick test finished, but no status file was written.";
        }

        try
        {
            var content = System.IO.File.ReadAllText(absolutePath);
            var variant = Json.ParseString(content);
            if (variant.VariantType != Variant.Type.Dictionary)
            {
                return "Quick test finished, but the status summary could not be read.";
            }

            var status = variant.AsGodotDictionary();
            var message = status.ContainsKey("message") ? status["message"].ToString() : string.Empty;
            return string.IsNullOrWhiteSpace(message)
                ? "Quick test finished."
                : message;
        }
        catch (Exception exception)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Failed to read quick-test status '{statusPath}': {exception.Message}");
            return "Quick test finished, but the summary could not be read.";
        }
    }

    private static string SanitizeRunPrefix(string prefix)
    {
        if (string.IsNullOrWhiteSpace(prefix))
        {
            return string.Empty;
        }

        var builder = new System.Text.StringBuilder(prefix.Length);
        foreach (var character in prefix)
        {
            if (char.IsLetterOrDigit(character))
            {
                builder.Append(char.ToLowerInvariant(character));
            }
            else if (character is '-' or '_')
            {
                builder.Append(character);
            }
        }

        return builder.ToString().Trim('_', '-');
    }

    private static bool HasGlobalScriptClass(string className)
    {
        var globalClasses = ProjectSettings.GetGlobalClassList();
        foreach (Godot.Collections.Dictionary entry in globalClasses)
        {
            if (!entry.ContainsKey("class"))
            {
                continue;
            }

            if (string.Equals(entry["class"].AsString(), className, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }
}
