using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using RlAgentPlugin.Editor;
using RlAgentPlugin.Runtime;
using RlAgentPlugin.Runtime.Imitation;

namespace RlAgentPlugin;

[Tool]
public partial class RLAgentPluginEditor : EditorPlugin
{
    private enum AutoDaggerPhase
    {
        InitialBC,
        DAgger,
        RetrainBC,
    }

    private sealed class AutoDaggerLoopState
    {
        public string ScenePath { get; init; } = string.Empty;
        public string AgentGroupId { get; init; } = string.Empty;
        public string NetworkGraphPath { get; init; } = string.Empty;
        public string OutputNameBase { get; init; } = "dagger";
        public int AdditionalFrames { get; init; }
        public int TotalRounds { get; init; }
        public int Epochs { get; init; }
        public float LearningRate { get; init; }
        public float MixingBeta { get; init; } = 0.5f;
        public int CurrentRound { get; set; }
        public string CurrentDatasetPath { get; set; } = string.Empty;
        public string CurrentCheckpointPath { get; set; } = string.Empty;
        public AutoDaggerPhase Phase { get; set; }
        public bool CancelRequested { get; set; }
    }

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
    private EditorDock? _imitationEditorDock;
    private RLSetupDock? _setupDock;
    private RLDashboard? _dashboard;
    private RLImitationDock? _imitationDock;
    private Button? _startTrainingButton;
    private Button? _stopTrainingButton;
    private Button? _runInferenceButton;
    private RLHPOParameterInspectorPlugin? _hpoInspectorPlugin;
    private RLNetworkGraphInspectorPlugin? _networkGraphInspectorPlugin;
    private TrainingSceneValidation? _lastValidation;
    private bool _launchedTrainingRun;
    private bool _launchedQuickTestRun;
    private bool _launchedRecordingRun;
    private bool _launchedDaggerRun;
    private bool _recordingGameStarted; // true once isPlaying was observed for this recording session
    private bool _daggerGameStarted;
    private string _activeStatusPath = string.Empty;
    private string _activeRecordingOutputPath = string.Empty;
    private string _activeDaggerOutputPath = string.Empty;
    private float _activeRecordingTimeScale = 1.0f;
    private bool _recordingPaused;
    private bool _recordingCapRead;
    private bool _stepModeEnabled;
    private int _stepCount;
    private int _pendingStepAction;
    private string _lastAutoScenePath = string.Empty;
    private string _lastValidationSignature = string.Empty;
    private CancellationTokenSource? _bcTrainingCts;
    private Task? _bcTrainingTask;
    // Thread-safe handoff: background thread writes, main thread reads in SavePendingBCCheckpoint.
    private RLCheckpoint? _pendingBCCheckpoint;
    private string _pendingBCOutPath = string.Empty;
    // Final training stats — updated each progress tick, read when saving the checkpoint.
    private int _bcFinalEpoch;
    private float _bcFinalLoss;
    private int _bcTotalEpochs;
    private string _lastSavedBCCheckpointPath = string.Empty;
    private AutoDaggerLoopState? _autoDaggerLoop;
    // Reset to false on every C# assembly reload, ensuring signals are reconnected exactly once
    // per session without stale-callable disconnect attempts (which Godot can't resolve).
    private bool _imitationDockSignalsConnected;

    public override void _EnterTree()
    {
        _rlModelFormatLoader = new RLModelFormatLoader();
        ResourceLoader.AddResourceFormatLoader(_rlModelFormatLoader, true);

        _setupDock = new RLSetupDock();
        _setupDock.Connect(RLSetupDock.SignalName.StartTrainingRequested, Callable.From(OnStartTrainingRequested));
        _setupDock.Connect(RLSetupDock.SignalName.StopTrainingRequested, Callable.From(OnStopTrainingRequested));
        _setupDock.Connect(RLSetupDock.SignalName.QuickTestRequested, Callable.From(OnQuickTestRequested));
        _setupDock.Connect(RLSetupDock.SignalName.ValidateSceneRequested, Callable.From(OnValidateSceneRequested));
        _setupDock.Connect(RLSetupDock.SignalName.AutofixRequested, Callable.From<int, string>(OnAutofixRequested));
        _setupDock.Connect(RLSetupDock.SignalName.AutofixAllRequested, Callable.From(OnAutofixAllRequested));
        _setupDock.Connect(RLSetupDock.SignalName.ReviewTargetRequested, Callable.From<bool, string>(OnReviewTargetRequested));

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

        _imitationDock = new RLImitationDock();
        ConnectImitationDockSignals(_imitationDock);
        _imitationEditorDock = new EditorDock
        {
            Title = "RL Imitation",
            DefaultSlot = EditorDock.DockSlot.Bottom,
        };
        _imitationEditorDock.AddChild(_imitationDock);
        AddDock(_imitationEditorDock);
        CallDeferred(nameof(RefreshImitationDatasets));

        // Reset sentinel so first _Process frame always triggers a scene refresh.
        _lastAutoScenePath = "\x00";

        RegisterCustomTypes();
        CallDeferred(nameof(EnsureProjectScriptClassesAreFresh));
        SetProcess(true);
        // _lastAutoScenePath is set to a sentinel ("\x00") above, so _Process will
        // detect the mismatch on its first frame and call RefreshValidationFromActiveScene().
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
            var startCallable = Callable.From(OnStartTrainingRequested);
            if (_setupDock.IsConnected(RLSetupDock.SignalName.StartTrainingRequested, startCallable))
            {
                _setupDock.Disconnect(RLSetupDock.SignalName.StartTrainingRequested, startCallable);
            }

            var stopCallable = Callable.From(OnStopTrainingRequested);
            if (_setupDock.IsConnected(RLSetupDock.SignalName.StopTrainingRequested, stopCallable))
            {
                _setupDock.Disconnect(RLSetupDock.SignalName.StopTrainingRequested, stopCallable);
            }

            var quickTestCallable = Callable.From(OnQuickTestRequested);
            if (_setupDock.IsConnected(RLSetupDock.SignalName.QuickTestRequested, quickTestCallable))
            {
                _setupDock.Disconnect(RLSetupDock.SignalName.QuickTestRequested, quickTestCallable);
            }

            var validateCallable = Callable.From(OnValidateSceneRequested);
            if (_setupDock.IsConnected(RLSetupDock.SignalName.ValidateSceneRequested, validateCallable))
            {
                _setupDock.Disconnect(RLSetupDock.SignalName.ValidateSceneRequested, validateCallable);
            }

            var autofixCallable = Callable.From<int, string>(OnAutofixRequested);
            if (_setupDock.IsConnected(RLSetupDock.SignalName.AutofixRequested, autofixCallable))
            {
                _setupDock.Disconnect(RLSetupDock.SignalName.AutofixRequested, autofixCallable);
            }

            var autofixAllCallable = Callable.From(OnAutofixAllRequested);
            if (_setupDock.IsConnected(RLSetupDock.SignalName.AutofixAllRequested, autofixAllCallable))
            {
                _setupDock.Disconnect(RLSetupDock.SignalName.AutofixAllRequested, autofixAllCallable);
            }

            var reviewTargetCallable = Callable.From<bool, string>(OnReviewTargetRequested);
            if (_setupDock.IsConnected(RLSetupDock.SignalName.ReviewTargetRequested, reviewTargetCallable))
            {
                _setupDock.Disconnect(RLSetupDock.SignalName.ReviewTargetRequested, reviewTargetCallable);
            }

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

        if (_imitationEditorDock is not null)
        {
            RemoveDock(_imitationEditorDock);
            _imitationEditorDock.QueueFree();
            _imitationEditorDock = null;
            _imitationDock = null;
        }

        _bcTrainingCts?.Cancel();
        _bcTrainingCts = null;
        _bcTrainingTask = null;
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

        if (isPlaying && _launchedRecordingRun)
        {
            // Mark that the game has actually started for this recording session.
            _recordingGameStarted = true;

            // Poll capabilities file once (written by RecordingBootstrap at startup).
            if (!_recordingCapRead && !string.IsNullOrWhiteSpace(_activeRecordingOutputPath))
            {
                var capPath = _activeRecordingOutputPath + ".cap";
                if (System.IO.File.Exists(capPath))
                {
                    _recordingCapRead = true;
                    ReadRecordingCapabilities(capPath);
                }
            }

            // Poll status file for live recording stats.
            if (!string.IsNullOrWhiteSpace(_activeRecordingOutputPath))
            {
                var statusPath = _activeRecordingOutputPath + ".status";
                if (System.IO.File.Exists(statusPath))
                {
                    try
                    {
                        var text = System.IO.File.ReadAllText(statusPath);
                        if (!string.IsNullOrWhiteSpace(text))
                        {
                            var parsed = Json.ParseString(text);
                            if (parsed.VariantType == Variant.Type.Dictionary)
                            {
                                var d = parsed.AsGodotDictionary();
                                var frames = d.ContainsKey("Frames") ? d["Frames"].AsInt64() : 0L;
                                var episodes = d.ContainsKey("Episodes") ? d["Episodes"].AsInt32() : 0;
                                var episodeSteps = d.ContainsKey("EpisodeSteps") ? d["EpisodeSteps"].AsInt32() : 0;
                                var episodeReward = d.ContainsKey("EpisodeReward") ? (float)d["EpisodeReward"].AsDouble() : 0f;
                                var paused = d.ContainsKey("Paused") && d["Paused"].AsBool();
                                _imitationDock?.SetRecordingStats(frames, episodes, episodeSteps, episodeReward, paused);
                            }
                        }
                    }
                    catch { /* file may be partially written */ }
                }
            }
        }

        if (isPlaying && _launchedDaggerRun)
        {
            _daggerGameStarted = true;

            if (!string.IsNullOrWhiteSpace(_activeDaggerOutputPath))
            {
                var statusPath = _activeDaggerOutputPath + ".status";
                if (System.IO.File.Exists(statusPath))
                {
                    try
                    {
                        var text = System.IO.File.ReadAllText(statusPath);
                        if (!string.IsNullOrWhiteSpace(text))
                        {
                            var parsed = Json.ParseString(text);
                            if (parsed.VariantType == Variant.Type.Dictionary)
                            {
                                var d = parsed.AsGodotDictionary();
                                var addedFrames = d.ContainsKey("AddedFrames") ? d["AddedFrames"].AsInt32() : 0;
                                var targetFrames = d.ContainsKey("TargetFrames") ? d["TargetFrames"].AsInt32() : 0;
                                var episodes = d.ContainsKey("Episodes") ? d["Episodes"].AsInt32() : 0;
                                var progress = targetFrames > 0 ? Mathf.Clamp((float)addedFrames / targetFrames, 0f, 1f) : 0f;
                                _imitationDock?.SetDaggerProgress(progress, addedFrames, targetFrames, episodes);
                            }
                        }
                    }
                    catch { }
                }
            }
        }

        if (!isPlaying)
        {
            if (_launchedQuickTestRun)
            {
                _setupDock.SetLaunchStatus(ReadCompletedQuickTestStatus(_activeStatusPath));
                _launchedQuickTestRun = false;
                _activeStatusPath = string.Empty;
            }

            _launchedTrainingRun = false;

            // Only conclude a recording session after the game has actually started and then stopped.
            // Without this guard, _launchedRecordingRun would be reset in the frames before
            // IsPlayingScene() returns true (PlayCustomScene is asynchronous), which would prevent
            // the .cap file from ever being polled.
            if (_launchedRecordingRun && _recordingGameStarted)
            {
                _launchedRecordingRun = false;
                _recordingGameStarted = false;
                var doneMarker = _activeRecordingOutputPath + ".done";
                if (System.IO.File.Exists(doneMarker))
                {
                    _imitationDock?.SetRecordingStatus("Recording saved.", false);
                    RefreshImitationDatasets();
                }
                else
                {
                    _imitationDock?.SetRecordingStatus("Recording stopped (no output produced).", false);
                }

                _activeRecordingOutputPath = string.Empty;
                _activeRecordingTimeScale = 1.0f;
                _recordingPaused = false;
                _recordingCapRead = false;
                _stepModeEnabled = false;
                _stepCount = 0;
            }

            if (_launchedDaggerRun && _daggerGameStarted)
            {
                _launchedDaggerRun = false;
                _daggerGameStarted = false;
                var doneMarker = _activeDaggerOutputPath + ".done";
                if (System.IO.File.Exists(doneMarker))
                {
                    try
                    {
                        var statusPath = _activeDaggerOutputPath + ".status";
                        int seedFrames = 0;
                        int addedFrames = 0;
                        if (System.IO.File.Exists(statusPath))
                        {
                            var text = System.IO.File.ReadAllText(statusPath);
                            if (!string.IsNullOrWhiteSpace(text))
                            {
                                var parsed = Json.ParseString(text);
                                if (parsed.VariantType == Variant.Type.Dictionary)
                                {
                                    var d = parsed.AsGodotDictionary();
                                    seedFrames = d.ContainsKey("SeedFrames") ? d["SeedFrames"].AsInt32() : 0;
                                    addedFrames = d.ContainsKey("AddedFrames") ? d["AddedFrames"].AsInt32() : 0;
                                }
                            }
                        }

                        var resPath = "res://RL-Agent-Demos/" + System.IO.Path.GetFileName(_activeDaggerOutputPath);
                        _imitationDock?.SetDaggerStatus("DAgger round saved.", false);
                        _imitationDock?.SetDaggerSummary(resPath, seedFrames, addedFrames);
                        RefreshImitationDatasets();
                        if (_autoDaggerLoop is not null)
                            OnAutoDaggerRoundCompleted(resPath);
                    }
                    catch (Exception ex)
                    {
                        _imitationDock?.SetDaggerStatus($"DAgger finished, but status could not be read ({ex.Message}).", false);
                        if (_autoDaggerLoop is not null)
                            FinalizeAutoDaggerLoop($"Auto DAgger failed while reading round status ({ex.Message}).", false);
                    }
                }
                else
                {
                    _imitationDock?.SetDaggerStatus("DAgger stopped (no output produced).", false);
                    if (_autoDaggerLoop is not null)
                        FinalizeAutoDaggerLoop("Auto DAgger stopped before producing a dataset.", false);
                }

                _activeDaggerOutputPath = string.Empty;
            }
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

        // After a C# assembly reload, Godot re-runs the RLImitationDock constructor (clearing
        // its UI) but does NOT call _ExitTree/_EnterTree on the plugin, so _lastAutoScenePath
        // still matches and the normal path-change check never fires.
        // The dock sets native metadata to signal it was rebuilt. The plugin's _imitationDock
        // reference is a stale C# wrapper (null UI fields), so we must re-fetch it via the
        // native node tree before refreshing.
        if (_imitationDock?.HasMeta("_rl_needs_scene_refresh") == true &&
            EditorInterface.Singleton.GetEditedSceneRoot() is not null)
        {
            // Re-fetch the live C# wrapper from the native tree.
            if (_imitationEditorDock?.GetChildCount() > 0 &&
                _imitationEditorDock.GetChild(0) is RLImitationDock freshDock)
            {
                _imitationDock = freshDock;
                // Stale Callables from the old assembly are now null — reconnect with fresh delegates.
                ConnectImitationDockSignals(freshDock);
            }
            RefreshImitationSceneInfo();
            RefreshImitationDatasets();
        }

        // Auto-refresh validation when the edited scene changes.
        // After a C# assembly reload, GetEditedSceneRoot() returns null for several frames
        // even though a scene is open. Don't commit the path change until the root is
        // actually available — otherwise we'd show "no scene open" and never re-trigger.
        if (currentScenePath != _lastAutoScenePath)
        {
            var editedRootNow = EditorInterface.Singleton.GetEditedSceneRoot();
            if (editedRootNow is null)
            {
                // Scene root not settled yet — keep the sentinel, retry next frame.
                return;
            }
            ResetWizardReviewState(currentScenePath);
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

    /// <summary>
    /// Called deferred after _EnterTree. Retries every frame until GetEditedSceneRoot()
    /// returns a valid root, then fires a full scene refresh. Handles the case where
    /// the scene root is unavailable for several frames after a C# assembly reload.
    /// </summary>
    private void TryRefreshSceneInfoDeferred()
    {
        if (!IsInsideTree()) return;
        // _Process handles the first-frame refresh via the sentinel in _lastAutoScenePath.
        // Just call refresh once — do not loop; looping fills the message queue.
        RefreshValidationFromActiveScene();
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
            ResetWizardReviewState();
            _setupDock.SetScenePath(string.Empty);
            _setupDock.SetValidationSummary("No active scene. Open a scene or set a main scene in Project Settings.");
            _setupDock.SetConfigSummary(string.Empty, string.Empty, string.Empty);
            UpdateWizardUi(null);
            _imitationDock?.SetSceneInfo(string.Empty,
                Array.Empty<string>(), Array.Empty<string>(),
                Array.Empty<string>(), Array.Empty<string>(),
                Array.Empty<bool>(), Array.Empty<bool>());
            return;
        }

        _setupDock.SetScenePath(scenePath);
        var validation = ValidateSceneSafely(scenePath, "auto validation");
        _lastValidationSignature = BuildValidationSignature(EditorInterface.Singleton.GetEditedSceneRoot());
        UpdateValidationUi(validation);

        RefreshImitationSceneInfo();
    }

    private void RefreshImitationSceneInfo()
    {
        if (_imitationDock is null) { return; }

        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (editedRoot is null)
        {
            _imitationDock.SetSceneInfo(string.Empty,
                Array.Empty<string>(), Array.Empty<string>(),
                Array.Empty<string>(), Array.Empty<string>(),
                Array.Empty<bool>(), Array.Empty<bool>());
            return;
        }

        // Collect one entry per unique AgentId found in the scene.
        var seen = new System.Collections.Generic.HashSet<string>();
        var displayNames = new System.Collections.Generic.List<string>();
        var groupIds = new System.Collections.Generic.List<string>();
        var networkPaths = new System.Collections.Generic.List<string>();
        var inferencePaths = new System.Collections.Generic.List<string>();
        var humanSupports = new System.Collections.Generic.List<bool>();
        var scriptSupports = new System.Collections.Generic.List<bool>();

        CollectAgentEntries(editedRoot, seen, displayNames, groupIds, networkPaths, inferencePaths, humanSupports, scriptSupports);

        _imitationDock.SetSceneInfo(editedRoot.SceneFilePath,
            displayNames.ToArray(), groupIds.ToArray(),
            networkPaths.ToArray(), inferencePaths.ToArray(),
            humanSupports.ToArray(), scriptSupports.ToArray());
    }

    private static void CollectAgentEntries(
        Node node,
        System.Collections.Generic.HashSet<string> seen,
        System.Collections.Generic.List<string> displayNames,
        System.Collections.Generic.List<string> groupIds,
        System.Collections.Generic.List<string> networkPaths,
        System.Collections.Generic.List<string> inferencePaths,
        System.Collections.Generic.List<bool> humanSupports,
        System.Collections.Generic.List<bool> scriptSupports)
    {
        if (IsAgentNode(node))
        {
            // Read PolicyGroupConfig — direct cast first, Godot property API fallback
            // (user game scripts are not [Tool] so the C# cast may fail in editor context).
            RLPolicyGroupConfig? config = null;
            if (node is Runtime.IRLAgent directAgent)
            {
                config = directAgent.PolicyGroupConfig;
            }
            else
            {
                var variant = node.Get("PolicyGroupConfig");
                if (variant.VariantType == Variant.Type.Object)
                    config = variant.AsGodotObject() as RLPolicyGroupConfig;
            }

            var groupId = config?.AgentId ?? string.Empty;
            var key = string.IsNullOrWhiteSpace(groupId) ? node.Name.ToString() : groupId;
            var displayName = string.IsNullOrWhiteSpace(groupId) ? node.Name.ToString() : groupId;
            var supportsHuman = SupportsHumanRecording(node);
            var supportsScript = SupportsScriptRecording(node);

            if (seen.Add(key))
            {
                displayNames.Add(displayName);
                groupIds.Add(groupId);
                networkPaths.Add(config?.NetworkGraph?.ResourcePath ?? string.Empty);
                inferencePaths.Add(config?.InferenceModelPath ?? string.Empty);
                humanSupports.Add(supportsHuman);
                scriptSupports.Add(supportsScript);
            }
            else
            {
                var existingIndex = displayNames.FindIndex(name => string.Equals(name, displayName, StringComparison.Ordinal));
                if (existingIndex >= 0)
                {
                    humanSupports[existingIndex] &= supportsHuman;
                    scriptSupports[existingIndex] &= supportsScript;
                }
            }
        }

        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
                CollectAgentEntries(childNode, seen, displayNames, groupIds, networkPaths, inferencePaths, humanSupports, scriptSupports);
        }
    }

    private static bool SupportsHumanRecording(Node node)
    {
        return HasImplementedControlMethod(node, "OnHumanInput")
            || HasExternalHumanRecordingController(node);
    }

    private static bool SupportsScriptRecording(Node node)
        => HasImplementedControlMethod(node, "OnScriptedInput");

    private static bool HasImplementedControlMethod(Node node, string methodName)
    {
        var managedType = ResolveManagedScriptType(node);
        if (managedType is null)
        {
            return false;
        }

        var method = managedType.GetMethod(methodName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (method is null)
        {
            return false;
        }

        var declaringType = method.DeclaringType;
        if (declaringType is null || declaringType == typeof(RLAgent2D) || declaringType == typeof(RLAgent3D))
        {
            return false;
        }

        return MethodHasImplementationBody(method);
    }

    private static bool MethodHasImplementationBody(MethodInfo method)
    {
        var scriptPath = GetScriptPaths(method.DeclaringType!)
            .FirstOrDefault(path => !string.IsNullOrWhiteSpace(path) && path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase));
        if (string.IsNullOrWhiteSpace(scriptPath))
        {
            return true;
        }

        var absolutePath = ProjectSettings.GlobalizePath(scriptPath);
        if (!System.IO.File.Exists(absolutePath))
        {
            return true;
        }

        string source;
        try
        {
            source = System.IO.File.ReadAllText(absolutePath);
        }
        catch
        {
            return true;
        }

        var methodName = method.Name;
        var nameIndex = source.IndexOf(methodName, StringComparison.Ordinal);
        if (nameIndex < 0)
        {
            return true;
        }

        var arrowIndex = source.IndexOf("=>", nameIndex, StringComparison.Ordinal);
        var braceIndex = source.IndexOf('{', nameIndex);
        if (arrowIndex >= 0 && (braceIndex < 0 || arrowIndex < braceIndex))
        {
            var expressionEnd = source.IndexOf(';', arrowIndex);
            if (expressionEnd < 0)
            {
                return true;
            }

            var expressionBody = source.Substring(arrowIndex + 2, expressionEnd - arrowIndex - 2);
            return HasNonWhitespaceCode(expressionBody);
        }

        if (braceIndex < 0)
        {
            return true;
        }

        var depth = 0;
        for (var i = braceIndex; i < source.Length; i++)
        {
            if (source[i] == '{')
            {
                depth++;
            }
            else if (source[i] == '}')
            {
                depth--;
                if (depth == 0)
                {
                    var body = source.Substring(braceIndex + 1, i - braceIndex - 1);
                    return HasNonWhitespaceCode(body);
                }
            }
        }

        return true;
    }

    private static bool HasExternalHumanRecordingController(Node node)
    {
        for (var current = node.GetParent(); current is not null; current = current.GetParent())
        {
            var script = GetNodeScript(current);
            if (script is null || string.IsNullOrWhiteSpace(script.ResourcePath) ||
                !script.ResourcePath.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var absolutePath = ProjectSettings.GlobalizePath(script.ResourcePath);
            if (!System.IO.File.Exists(absolutePath))
            {
                continue;
            }

            try
            {
                var text = System.IO.File.ReadAllText(absolutePath);
                if (text.Contains("RLAgentControlMode.Human", StringComparison.Ordinal)
                    && text.Contains("ApplyAction(", StringComparison.Ordinal)
                    && (text.Contains("PendingStepAction", StringComparison.Ordinal)
                        || text.Contains("Input.IsActionPressed", StringComparison.Ordinal)
                        || text.Contains("Input.GetAxis", StringComparison.Ordinal)))
                {
                    return true;
                }
            }
            catch
            {
                // Ignore read errors and continue with the next ancestor script.
            }
        }

        return false;
    }

    private static bool HasNonWhitespaceCode(string source)
    {
        if (string.IsNullOrWhiteSpace(source))
        {
            return false;
        }

        source = System.Text.RegularExpressions.Regex.Replace(source, @"/\*.*?\*/", string.Empty, System.Text.RegularExpressions.RegexOptions.Singleline);
        source = System.Text.RegularExpressions.Regex.Replace(source, @"//.*?$", string.Empty, System.Text.RegularExpressions.RegexOptions.Multiline);
        return source.Any(character => !char.IsWhiteSpace(character));
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

        if (!TrySaveEditedSceneForLaunch(scenePath, "starting training", out var saveError))
        {
            _setupDock.SetLaunchStatus(saveError);
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

        var imitationWarmStartEnabled = _imitationDock?.IsWarmStartEnabled() ?? false;
        var imitationWarmStartPath = _imitationDock?.GetWarmStartCheckpointPath() ?? string.Empty;
        if (imitationWarmStartEnabled)
        {
            if (string.IsNullOrWhiteSpace(imitationWarmStartPath))
            {
                _setupDock.SetLaunchStatus("Warm-start is enabled in RL Imitation, but no checkpoint path is selected.");
                return;
            }

            manifest.OverrideResumeFromCheckpoint = true;
            manifest.ResumeFromCheckpoint = true;
            manifest.ResumeCheckpointPath = imitationWarmStartPath;
            GD.Print($"[RL Training] Resume override from RL Imitation warm-start: {imitationWarmStartPath}");
        }

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

        if (!TrySaveEditedSceneForLaunch(scenePath, "starting quick test", out var saveError))
        {
            _setupDock.SetLaunchStatus(saveError);
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

        if (!TrySaveEditedSceneForLaunch(scenePath, "starting inference", out var saveError))
        {
            _setupDock.SetLaunchStatus(saveError);
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
                { "warm_start_used", false },
                { "warm_start_source", "" },
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

    // ── Imitation Learning handlers ───────────────────────────────────────────

    private void OnStartRecordingRequested(string scenePath, string agentGroupId, string outputName, bool scriptMode)
    {
        if (_imitationDock is null) return;

        if (!Godot.FileAccess.FileExists(scenePath))
        {
            _imitationDock.SetRecordingStatus($"Scene not found: {scenePath}", false);
            return;
        }

        if (EditorInterface.Singleton.IsPlayingScene())
        {
            _imitationDock.SetRecordingStatus("Stop the current play session before recording.", false);
            return;
        }

        if (!TrySaveEditedSceneForLaunch(scenePath, "starting recording", out var saveError))
        {
            _imitationDock.SetRecordingStatus(saveError, false);
            return;
        }

        // Ensure output directory exists.
        var demosDir = ProjectSettings.GlobalizePath("res://RL-Agent-Demos");
        System.IO.Directory.CreateDirectory(demosDir);

        var safeName = string.Join("_", outputName.Split(System.IO.Path.GetInvalidFileNameChars()));
        if (string.IsNullOrWhiteSpace(safeName)) safeName = "demo";

        var outputFileName = $"{safeName}_{BuildRunTimestampSuffix()}.rldem";
        var outputAbsPath = System.IO.Path.Combine(demosDir, outputFileName);

        var manifest = new RecordingLaunchManifest
        {
            ScenePath = scenePath,
            AcademyNodePath = string.Empty, // RecordingBootstrap uses DFS fallback
            OutputFilePath = outputAbsPath,
            AgentGroupId = agentGroupId,
            ScriptMode = scriptMode,
        };

        var writeError = manifest.SaveToUserStorage();
        if (writeError != Error.Ok)
        {
            _imitationDock.SetRecordingStatus($"Failed to write manifest: {writeError}", false);
            return;
        }

        _activeRecordingOutputPath = outputAbsPath;
        _launchedRecordingRun = true;
        _recordingGameStarted = false;

        // Write an initial ctrl file so the bootstrap picks up pre-selected step mode
        // (user may have toggled step mode before clicking Start Recording).
        WriteRecordingControlFile();

        _imitationDock.SetRecordingStatus($"Recording to {outputFileName}…", true);

        EditorInterface.Singleton.PlayCustomScene(
            "res://addons/rl-agent-plugin/Scenes/Bootstrap/RecordingBootstrap.tscn");
    }

    private void OnStopRecordingRequested()
    {
        if (!_launchedRecordingRun || string.IsNullOrEmpty(_activeRecordingOutputPath)) return;

        // Write stop signal — RecordingBootstrap polls for this file.
        var stopSignalPath = _activeRecordingOutputPath + ".stop";
        try { System.IO.File.WriteAllText(stopSignalPath, "stop"); }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Could not write stop signal: {ex.Message}");
        }

        _imitationDock?.SetRecordingStatus("Stopping…", true);
    }

    private void OnSetRecordingSpeed(float timeScale)
    {
        _activeRecordingTimeScale = timeScale;
        WriteRecordingControlFile();
    }

    private void OnPauseRecording(bool paused)
    {
        _recordingPaused = paused;
        WriteRecordingControlFile();
    }

    private void OnSetStepMode(bool enabled)
    {
        _stepModeEnabled = enabled;
        if (!enabled) _stepCount = 0;
        WriteRecordingControlFile();
    }

    private void OnStepActionRequested(int actionIndex)
    {
        _pendingStepAction = actionIndex;
        _stepCount++;
        WriteRecordingControlFile();
    }

    private void ReadRecordingCapabilities(string capPath)
    {
        try
        {
            var text = System.IO.File.ReadAllText(capPath);
            var parsed = Json.ParseString(text);
            if (parsed.VariantType != Variant.Type.Dictionary) return;

            var d = parsed.AsGodotDictionary();
            var onlyDiscrete = d.ContainsKey("OnlyDiscrete") && d["OnlyDiscrete"].AsBool();
            var scriptMode = d.ContainsKey("ScriptMode") && d["ScriptMode"].AsBool();

            var labels = Array.Empty<string>();
            if (d.ContainsKey("Labels") && d["Labels"].VariantType == Variant.Type.Array)
            {
                var arr = d["Labels"].AsGodotArray();
                labels = new string[arr.Count];
                for (var i = 0; i < arr.Count; i++)
                    labels[i] = arr[i].AsString();
            }

            _imitationDock?.SetRecordingCapabilities(onlyDiscrete, labels, scriptMode);
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Could not read capabilities file: {ex.Message}");
        }
    }

    private void WriteRecordingControlFile()
    {
        if (string.IsNullOrWhiteSpace(_activeRecordingOutputPath)) return;
        var ctrlPath = _activeRecordingOutputPath + ".ctrl";
        try
        {
            System.IO.File.WriteAllText(ctrlPath, Json.Stringify(new Godot.Collections.Dictionary
            {
                { "TimeScale",      _activeRecordingTimeScale },
                { "PauseRecording", _recordingPaused },
                { "StepMode",       _stepModeEnabled },
                { "StepCount",      _stepCount },
                { "StepAction",     _pendingStepAction },
            }));
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Could not write control file: {ex.Message}");
        }
    }

    private void OnStartBCTrainingRequested(
        string datasetPath,
        string checkpointPath,
        string networkGraphPath,
        int epochs,
        float learningRate)
        => StartBCTrainingInternal(datasetPath, networkGraphPath, epochs, learningRate);

    private bool StartBCTrainingInternal(
        string datasetPath,
        string networkGraphPath,
        int epochs,
        float learningRate)
    {
        if (_imitationDock is null) return false;

        if (string.IsNullOrWhiteSpace(datasetPath))
        {
            _imitationDock.SetTrainingStatus("No dataset selected.", false);
            return false;
        }

        var dataset = DemonstrationDataset.Open(datasetPath);
        if (dataset is null)
        {
            _imitationDock.SetTrainingStatus($"Failed to load dataset: {datasetPath}", false);
            return false;
        }

        if (dataset.Frames.Count == 0)
        {
            _imitationDock.SetTrainingStatus("Dataset is empty — record some demonstrations first.", false);
            return false;
        }

        // Build the network from the configured graph (required in both warm-start and fresh modes).
        if (string.IsNullOrWhiteSpace(networkGraphPath))
        {
            _imitationDock.SetTrainingStatus("Select a network graph resource.", false);
            return false;
        }

        var networkGraph = GD.Load<RLNetworkGraph>(networkGraphPath);
        if (networkGraph is null)
        {
            _imitationDock.SetTrainingStatus($"Could not load network graph: {networkGraphPath}", false);
            return false;
        }

        var network = new PolicyValueNetwork(
            dataset.ObsSize,
            dataset.DiscreteActionCount,
            dataset.ContinuousActionDims,
            networkGraph);

        // Warm-start: load existing weights into the network before training.
        _imitationDock.SetWarmStartSource(
            _imitationDock.IsWarmStartEnabled() ? _imitationDock.GetWarmStartCheckpointPath() : string.Empty);
        GD.Print("[RL Imitation] BC Train starts from scratch. Use RL Setup -> Start Training to fine-tune from the warm-start checkpoint.");

        var config = new RLImitationConfig
        {
            Epochs = Math.Max(1, epochs),
            LearningRate = Math.Max(1e-6f, learningRate),
            BatchSize = 64,
        };

        // Prepare output path on the main thread (uses Godot ProjectSettings).
        var trainedDir = ProjectSettings.GlobalizePath("res://RL-Agent-Demos/trained");
        System.IO.Directory.CreateDirectory(trainedDir);
        var outFileName = $"bc_{BuildRunTimestampSuffix()}.rlcheckpoint";
        var outAbsPath = System.IO.Path.Combine(trainedDir, outFileName);

        _bcTrainingCts?.Cancel();
        _bcTrainingCts = new CancellationTokenSource();
        var cts = _bcTrainingCts;

        var trainer = new BCTrainer(network, dataset, config);
        _imitationDock.SetTrainingStatus($"Training… (0/{epochs} epochs)", true);
        _imitationDock.SetTrainingProgress(0f, 0f, 0, epochs);

        _bcTrainingTask = Task.Run(() =>
        {
            trainer.Train(report =>
            {
                // CallDeferred on this Node is thread-safe in Godot 4.
                CallDeferred(nameof(OnBCProgressDeferred),
                    report.Progress, report.BatchLoss, report.Epoch, report.TotalEpochs);
            }, cts.Token);

            if (cts.Token.IsCancellationRequested)
            {
                CallDeferred(nameof(OnBCTrainingCancelledDeferred));
                return;
            }

            // BuildCheckpoint is pure C# (no Godot API) — safe on background thread.
            // SaveToZip uses Json.Stringify (Godot API) so we save on the main thread via deferred.
            _pendingBCCheckpoint = trainer.BuildCheckpoint($"bc_{outFileName}");
            _pendingBCOutPath = outAbsPath;
            CallDeferred(nameof(SavePendingBCCheckpoint), outFileName);
        });

        return true;
    }

    private void OnBCProgressDeferred(float progress, float loss, int epoch, int totalEpochs)
    {
        _bcFinalEpoch = epoch;
        _bcFinalLoss = loss;
        _bcTotalEpochs = totalEpochs;
        _imitationDock?.SetTrainingProgress(progress, loss, epoch, totalEpochs);
    }

    // Runs on main thread — safe to call Godot APIs.
    private void SavePendingBCCheckpoint(string outFileName)
    {
        var checkpoint = _pendingBCCheckpoint;
        var outAbsPath = _pendingBCOutPath;
        _pendingBCCheckpoint = null;
        _pendingBCOutPath = string.Empty;
        _bcTrainingTask = null;
        _bcTrainingCts = null;

        if (checkpoint is null || _imitationDock is null) return;

        var saveError = RLCheckpoint.SaveToZip(checkpoint, outAbsPath);

        _imitationDock.SetTrainingProgress(1f, 0f, 0, 0);

        if (saveError == Error.Ok)
        {
            _lastSavedBCCheckpointPath = outAbsPath;
            _imitationDock.SetTrainingStatus("Training complete.", false);
            _imitationDock.SetTrainingSummary(_bcFinalEpoch, _bcFinalLoss, outAbsPath);
            if (_autoDaggerLoop is not null)
                OnAutoDaggerBcCompleted(outAbsPath);
        }
        else
        {
            _imitationDock.SetTrainingStatus($"Training done but save failed ({saveError}).", false);
            if (_autoDaggerLoop is not null)
                FinalizeAutoDaggerLoop($"Auto DAgger failed while saving a BC checkpoint ({saveError}).", false);
        }
    }

    private void OnBCTrainingCancelledDeferred()
    {
        _bcTrainingTask = null;
        _bcTrainingCts = null;
        _imitationDock?.SetTrainingStatus("Training cancelled.", false);
        _imitationDock?.SetTrainingProgress(0f, 0f, 0, 0);

        if (_autoDaggerLoop is not null)
            FinalizeAutoDaggerLoop("Auto DAgger cancelled.", false);
    }

    private void OnCancelBCTrainingRequested()
    {
        if (_autoDaggerLoop is not null)
            _autoDaggerLoop.CancelRequested = true;

        _bcTrainingCts?.Cancel();

        if (_launchedDaggerRun && !string.IsNullOrWhiteSpace(_activeDaggerOutputPath))
        {
            try
            {
                System.IO.File.WriteAllText(_activeDaggerOutputPath + ".stop", "stop");
                _imitationDock?.SetDaggerStatus("Stopping DAgger…", true);
            }
            catch (Exception ex)
            {
                GD.PushWarning($"[RLAgentPluginEditor] Could not write DAgger stop signal: {ex.Message}");
            }
        }
    }

    // ── Imitation dock signal wiring ─────────────────────────────────────────
    // Called at startup and after a C# assembly reload (Godot recreates dock
    // C# wrappers but does NOT call _EnterTree/_ExitTree on the plugin).
    //
    // We intentionally do NOT try to disconnect stale connections first: after
    // assembly reload the old Callables have null delegate handles and Godot
    // cannot match them for disconnection, producing a flood of console errors.
    // Instead, _imitationDockSignalsConnected (a plain C# field, so reset to
    // false on every reload) ensures we connect exactly once per session.
    // Any leftover stale connections are harmless — Godot prints "Continuing"
    // and skips them; the fresh connection added here handles the signal.

    private void ConnectImitationDockSignals(RLImitationDock dock)
    {
        if (_imitationDockSignalsConnected) return;
        _imitationDockSignalsConnected = true;

        dock.Connect(RLImitationDock.SignalName.StartRecordingRequested,
            Callable.From<string, string, string, bool>(OnStartRecordingRequested));
        dock.Connect(RLImitationDock.SignalName.StopRecordingRequested,
            Callable.From(OnStopRecordingRequested));
        dock.Connect(RLImitationDock.SignalName.StartBCTrainingRequested,
            Callable.From<string, string, string, int, float>(OnStartBCTrainingRequested));
        dock.Connect(RLImitationDock.SignalName.StartDAggerRequested,
            Callable.From<string, string, string, string, string, string, int, float>(OnStartDAggerRequested));
        dock.Connect(RLImitationDock.SignalName.StartAutoDAggerRequested,
            Callable.From<string, string, string, string, string, string, string>(OnStartAutoDAggerRequested));
        dock.Connect(RLImitationDock.SignalName.CancelBCTrainingRequested,
            Callable.From(OnCancelBCTrainingRequested));
        dock.Connect(RLImitationDock.SignalName.ExportModelRequested,
            Callable.From<string, string>(OnExportModelRequested));
        dock.Connect(RLImitationDock.SignalName.RefreshDatasetsRequested,
            Callable.From(RefreshImitationDatasets));
        dock.Connect(RLImitationDock.SignalName.SetRecordingSpeedRequested,
            Callable.From<float>(OnSetRecordingSpeed));
        dock.Connect(RLImitationDock.SignalName.PauseRecordingRequested,
            Callable.From<bool>(OnPauseRecording));
        dock.Connect(RLImitationDock.SignalName.SetStepModeRequested,
            Callable.From<bool>(OnSetStepMode));
        dock.Connect(RLImitationDock.SignalName.StepActionRequested,
            Callable.From<int>(OnStepActionRequested));
    }

    private void OnExportModelRequested(string checkpointPath, string destPath)
    {
        if (_imitationDock is null) return;

        if (string.IsNullOrWhiteSpace(checkpointPath) || !System.IO.File.Exists(checkpointPath))
        {
            _imitationDock.SetTrainingStatus($"Checkpoint not found: {checkpointPath}", false);
            return;
        }

        var result = RLModelExporter.Export(checkpointPath, destPath);

        if (result == Error.Ok)
        {
            var resPath = ProjectSettings.LocalizePath(destPath);
            _imitationDock.SetTrainingStatus($"Exported: {resPath}", false);
        }
        else
        {
            _imitationDock.SetTrainingStatus("Export failed — see Godot output for details.", false);
        }
    }

    private void OnStartAutoDAggerRequested(
        string scenePath,
        string agentGroupId,
        string datasetPath,
        string checkpointPath,
        string networkGraphPath,
        string outputName,
        string configJson)
    {
        if (_imitationDock is null) return;

        if (_autoDaggerLoop is not null || _bcTrainingTask is not null || _launchedDaggerRun)
        {
            _imitationDock.SetDaggerStatus("Another imitation task is already running.", false);
            return;
        }

        if (!Godot.FileAccess.FileExists(scenePath))
        {
            _imitationDock.SetDaggerStatus($"Scene not found: {scenePath}", false);
            return;
        }

        if (string.IsNullOrWhiteSpace(agentGroupId))
        {
            _imitationDock.SetDaggerStatus("Select a specific agent group for Auto DAgger.", false);
            return;
        }

        if (string.IsNullOrWhiteSpace(networkGraphPath))
        {
            _imitationDock.SetDaggerStatus("Select a network graph first.", false);
            return;
        }

        var parsed = Json.ParseString(configJson);
        if (parsed.VariantType != Variant.Type.Dictionary)
        {
            _imitationDock.SetDaggerStatus("Auto DAgger config was invalid.", false);
            return;
        }

        var config = parsed.AsGodotDictionary();
        var additionalFrames = config.ContainsKey("AdditionalFrames") ? config["AdditionalFrames"].AsInt32() : 2048;
        var rounds = config.ContainsKey("Rounds") ? config["Rounds"].AsInt32() : 1;
        var epochs = config.ContainsKey("Epochs") ? config["Epochs"].AsInt32() : 20;
        var learningRate = config.ContainsKey("LearningRate") ? (float)config["LearningRate"].AsDouble() : 3e-4f;
        var mixingBeta = config.ContainsKey("MixingBeta") ? (float)config["MixingBeta"].AsDouble() : 0.5f;

        if (string.IsNullOrWhiteSpace(datasetPath) && string.IsNullOrWhiteSpace(checkpointPath))
        {
            _imitationDock.SetDaggerStatus("Select a seed dataset or a learner checkpoint first.", false);
            return;
        }

        _autoDaggerLoop = new AutoDaggerLoopState
        {
            ScenePath = scenePath,
            AgentGroupId = agentGroupId,
            CurrentDatasetPath = datasetPath,
            CurrentCheckpointPath = checkpointPath?.Trim() ?? string.Empty,
            NetworkGraphPath = networkGraphPath,
            OutputNameBase = string.IsNullOrWhiteSpace(outputName) ? "dagger" : outputName.Trim(),
            AdditionalFrames = Math.Max(1, additionalFrames),
            TotalRounds = Math.Max(1, rounds),
            Epochs = Math.Max(1, epochs),
            LearningRate = Math.Max(1e-6f, learningRate),
            MixingBeta = Math.Clamp(mixingBeta, 0f, 1f),
            CurrentRound = 1,
            Phase = string.IsNullOrWhiteSpace(checkpointPath) ? AutoDaggerPhase.InitialBC : AutoDaggerPhase.DAgger,
        };

        StartNextAutoDaggerPhase();
    }

    private void StartNextAutoDaggerPhase()
    {
        if (_autoDaggerLoop is null || _imitationDock is null)
            return;

        if (_autoDaggerLoop.CancelRequested)
        {
            FinalizeAutoDaggerLoop("Auto DAgger cancelled.", false);
            return;
        }

        switch (_autoDaggerLoop.Phase)
        {
            case AutoDaggerPhase.InitialBC:
                _imitationDock.SetTrainingStatus(
                    $"Auto DAgger seed: training BC on {Path.GetFileName(_autoDaggerLoop.CurrentDatasetPath)}…",
                    true);
                if (!StartBCTrainingInternal(
                        _autoDaggerLoop.CurrentDatasetPath,
                        _autoDaggerLoop.NetworkGraphPath,
                        _autoDaggerLoop.Epochs,
                        _autoDaggerLoop.LearningRate))
                {
                    FinalizeAutoDaggerLoop("Auto DAgger could not start seed BC training.", false);
                }
                break;

            case AutoDaggerPhase.DAgger:
                _imitationDock.SetDaggerStatus(
                    $"Auto DAgger round {_autoDaggerLoop.CurrentRound}/{_autoDaggerLoop.TotalRounds}: collecting expert labels…",
                    true);
                if (!StartDAggerInternal(
                        _autoDaggerLoop.ScenePath,
                        _autoDaggerLoop.AgentGroupId,
                        _autoDaggerLoop.CurrentDatasetPath,
                        _autoDaggerLoop.CurrentCheckpointPath,
                        _autoDaggerLoop.NetworkGraphPath,
                        $"{_autoDaggerLoop.OutputNameBase}_r{_autoDaggerLoop.CurrentRound}",
                        _autoDaggerLoop.AdditionalFrames,
                        _autoDaggerLoop.MixingBeta,
                        _autoDaggerLoop.CurrentRound))
                {
                    FinalizeAutoDaggerLoop("Auto DAgger could not start the aggregation round.", false);
                }
                break;

            case AutoDaggerPhase.RetrainBC:
                _imitationDock.SetTrainingStatus(
                    $"Auto DAgger round {_autoDaggerLoop.CurrentRound}/{_autoDaggerLoop.TotalRounds}: retraining BC…",
                    true);
                if (!StartBCTrainingInternal(
                        _autoDaggerLoop.CurrentDatasetPath,
                        _autoDaggerLoop.NetworkGraphPath,
                        _autoDaggerLoop.Epochs,
                        _autoDaggerLoop.LearningRate))
                {
                    FinalizeAutoDaggerLoop("Auto DAgger could not start BC retraining.", false);
                }
                break;
        }
    }

    private void OnAutoDaggerBcCompleted(string checkpointPath)
    {
        if (_autoDaggerLoop is null)
            return;

        _autoDaggerLoop.CurrentCheckpointPath = checkpointPath;

        if (_autoDaggerLoop.CancelRequested)
        {
            FinalizeAutoDaggerLoop("Auto DAgger cancelled.", false);
            return;
        }

        if (_autoDaggerLoop.Phase == AutoDaggerPhase.InitialBC)
        {
            _autoDaggerLoop.Phase = AutoDaggerPhase.DAgger;
            StartNextAutoDaggerPhase();
            return;
        }

        if (_autoDaggerLoop.Phase != AutoDaggerPhase.RetrainBC)
            return;

        if (_autoDaggerLoop.CurrentRound >= _autoDaggerLoop.TotalRounds)
        {
            var finalDatasetPath = _autoDaggerLoop.CurrentDatasetPath;
            var completedRounds = _autoDaggerLoop.CurrentRound;
            FinalizeAutoDaggerLoop("Auto DAgger complete.", true);
            _imitationDock?.SetAutoDaggerSummary(completedRounds, finalDatasetPath, checkpointPath);
            return;
        }

        _autoDaggerLoop.CurrentRound++;
        _autoDaggerLoop.Phase = AutoDaggerPhase.DAgger;
        StartNextAutoDaggerPhase();
    }

    private void OnAutoDaggerRoundCompleted(string resDatasetPath)
    {
        if (_autoDaggerLoop is null)
            return;

        _autoDaggerLoop.CurrentDatasetPath = resDatasetPath;

        if (_autoDaggerLoop.CancelRequested)
        {
            FinalizeAutoDaggerLoop("Auto DAgger cancelled.", false);
            return;
        }

        _autoDaggerLoop.Phase = AutoDaggerPhase.RetrainBC;
        StartNextAutoDaggerPhase();
    }

    private void FinalizeAutoDaggerLoop(string message, bool success)
    {
        if (_imitationDock is not null)
        {
            _imitationDock.SetTrainingStatus(message, false);
            if (!success)
                _imitationDock.SetDaggerStatus(message, false);
        }

        _autoDaggerLoop = null;
    }

    private void OnStartDAggerRequested(
        string scenePath,
        string agentGroupId,
        string datasetPath,
        string checkpointPath,
        string networkGraphPath,
        string outputName,
        int additionalFrames,
        float mixingBeta)
        => StartDAggerInternal(scenePath, agentGroupId, datasetPath, checkpointPath, networkGraphPath, outputName, additionalFrames, mixingBeta, roundIndex: 1);

    private bool StartDAggerInternal(
        string scenePath,
        string agentGroupId,
        string datasetPath,
        string checkpointPath,
        string networkGraphPath,
        string outputName,
        int additionalFrames,
        float mixingBeta = 0.5f,
        int roundIndex = 1)
    {
        if (_imitationDock is null) return false;

        if (!Godot.FileAccess.FileExists(scenePath))
        {
            _imitationDock.SetDaggerStatus($"Scene not found: {scenePath}", false);
            return false;
        }

        if (EditorInterface.Singleton.IsPlayingScene())
        {
            _imitationDock.SetDaggerStatus("Stop the current play session before running DAgger.", false);
            return false;
        }

        if (!TrySaveEditedSceneForLaunch(scenePath, "starting DAgger", out var saveError))
        {
            _imitationDock.SetDaggerStatus(saveError, false);
            return false;
        }

        if (string.IsNullOrWhiteSpace(agentGroupId))
        {
            _imitationDock.SetDaggerStatus("Select a specific agent group for DAgger.", false);
            return false;
        }

        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            _imitationDock.SetDaggerStatus("Select a learner checkpoint first.", false);
            return false;
        }

        if (string.IsNullOrWhiteSpace(networkGraphPath))
        {
            _imitationDock.SetDaggerStatus("Select a network graph first.", false);
            return false;
        }

        var demosDir = ProjectSettings.GlobalizePath("res://RL-Agent-Demos");
        System.IO.Directory.CreateDirectory(demosDir);

        var safeName = string.Join("_", outputName.Split(System.IO.Path.GetInvalidFileNameChars()));
        if (string.IsNullOrWhiteSpace(safeName)) safeName = "dagger";
        var outputFileName = $"{safeName}_{BuildRunTimestampSuffix()}.rldem";
        var outputAbsPath = System.IO.Path.Combine(demosDir, outputFileName);

        var manifest = new DAggerLaunchManifest
        {
            ScenePath = scenePath,
            AcademyNodePath = string.Empty,
            OutputFilePath = outputAbsPath,
            AgentGroupId = agentGroupId,
            SeedDatasetPath = datasetPath,
            LearnerCheckpointPath = checkpointPath,
            NetworkGraphPath = networkGraphPath,
            AdditionalFrames = Math.Max(1, additionalFrames),
            MixingBeta = Math.Clamp(mixingBeta, 0f, 1f),
            RoundIndex = Math.Max(1, roundIndex),
        };

        var writeError = manifest.SaveToUserStorage();
        if (writeError != Error.Ok)
        {
            _imitationDock.SetDaggerStatus($"Failed to write DAgger manifest: {writeError}", false);
            return false;
        }

        _activeDaggerOutputPath = outputAbsPath;
        _launchedDaggerRun = true;
        _daggerGameStarted = false;
        _imitationDock.SetDaggerStatus($"Running DAgger to {outputFileName}…", true);
        _imitationDock.SetDaggerProgress(0f, 0, Math.Max(1, additionalFrames), 0);

        EditorInterface.Singleton.PlayCustomScene(
            "res://addons/rl-agent-plugin/Scenes/Bootstrap/DAggerBootstrap.tscn");

        return true;
    }

    private void RefreshImitationDatasets()
    {
        if (_imitationDock is null) return;

        var demosDir = ProjectSettings.GlobalizePath("res://RL-Agent-Demos");
        if (!System.IO.Directory.Exists(demosDir))
        {
            _imitationDock.RefreshDatasetList(Array.Empty<string>());
            return;
        }

        var files = System.IO.Directory.GetFiles(demosDir, "*.rldem", System.IO.SearchOption.TopDirectoryOnly);
        // Sort newest first by last-write time.
        Array.Sort(files, (a, b) => System.IO.File.GetLastWriteTimeUtc(b).CompareTo(System.IO.File.GetLastWriteTimeUtc(a)));
        var resPaths = new string[files.Length];
        for (var i = 0; i < files.Length; i++)
            resPaths[i] = "res://RL-Agent-Demos/" + System.IO.Path.GetFileName(files[i]);

        _imitationDock.RefreshDatasetList(resPaths);
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
        UpdateWizardUi(validation);
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
            var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
            var validation = editedRoot is not null && string.Equals(editedRoot.SceneFilePath, scenePath, StringComparison.Ordinal)
                ? ValidateSceneRoot(editedRoot, scenePath, ownsRoot: false)
                : ValidateScene(scenePath);
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
        var packedScene = GD.Load<PackedScene>(scenePath);
        if (packedScene is null)
        {
            var validation = new TrainingSceneValidation
            {
                ScenePath = scenePath,
                IsValid = false,
            };
            validation.AddBlockingError($"Could not load scene: {scenePath}");
            return validation;
        }

        var root = packedScene.Instantiate();
        return ValidateSceneRoot(root, scenePath, ownsRoot: true);
    }

    private static TrainingSceneValidation ValidateSceneRoot(Node root, string scenePath, bool ownsRoot)
    {
        var validation = new TrainingSceneValidation
        {
            ScenePath = scenePath,
        };

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
                        validation.AddBlockingError("More than one RLAcademy was found. Only one academy is supported.");
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
                            validation.AddIssue(
                                TrainingSceneIssueCode.MissingPolicyGroupConfig,
                                $"Agent '{root.GetPathTo(node)}' is in {modeLabel} mode but has no PolicyGroupConfig assigned.",
                                root.GetPathTo(node).ToString(),
                                TrainingSceneIssueSeverity.Blocking,
                                isAutofixable: true,
                                fixKind: TrainingSceneFixKind.CreatePolicyGroupConfig,
                                fixLabel: "Create Policy Config");
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
                validation.AddIssue(
                    TrainingSceneIssueCode.MissingAcademy,
                    "No RLAcademy node was found in the selected scene.",
                    severity: TrainingSceneIssueSeverity.Blocking,
                    isAutofixable: true,
                    fixKind: TrainingSceneFixKind.CreateAcademy,
                    fixLabel: "Create RLAcademy");
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
                    validation.AddBlockingError(
                        "RLAcademy has Curriculum assigned, but no scene node implements IRLCurriculumConsumer.");
                }

                if (trainingConfigRes is null)
                {
                    validation.AddIssue(
                        TrainingSceneIssueCode.MissingTrainingConfig,
                        "RLAcademy is missing an RLTrainingConfig resource.",
                        validation.AcademyPath,
                        TrainingSceneIssueSeverity.Blocking,
                        isAutofixable: true,
                        fixKind: TrainingSceneFixKind.CreateTrainingConfig,
                        fixLabel: "Create Training Config");
                }

                if (runConfig is null)
                {
                    validation.AddIssue(
                        TrainingSceneIssueCode.MissingRunConfig,
                        "RLAcademy has no RLRunConfig assigned. Training can still use defaults, but a run config is recommended.",
                        validation.AcademyPath,
                        TrainingSceneIssueSeverity.Warning,
                        isAutofixable: true,
                        fixKind: TrainingSceneFixKind.CreateRunConfig,
                        fixLabel: "Create Run Config");
                }

                if (trainingConfigRes is not null && trainingConfigRes.Algorithm is null)
                {
                    validation.AddIssue(
                        TrainingSceneIssueCode.MissingTrainingAlgorithm,
                        "RLAcademy.TrainingConfig has no Algorithm assigned.",
                        validation.AcademyPath,
                        TrainingSceneIssueSeverity.Blocking,
                        isAutofixable: true,
                        fixKind: TrainingSceneFixKind.CreateTrainingAlgorithm,
                        fixLabel: "Add PPO Algorithm");
                }

                if (validation.TrainAgentCount == 0)
                {
                    validation.AddBlockingError("No Train or Auto mode agents were found in the selected scene.");
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
                    validation.AddBlockingError(error);
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

                    if (binding.Config?.ResolvedNetworkGraph is null)
                    {
                        validation.AddIssue(
                            TrainingSceneIssueCode.MissingNetworkGraph,
                            $"Group '{groupSummary.GroupId}' has no network graph assigned.",
                            root.GetPathTo(firstNode).ToString(),
                            TrainingSceneIssueSeverity.Blocking,
                            isAutofixable: true,
                            fixKind: TrainingSceneFixKind.CreateNetworkGraph,
                            fixLabel: "Add Default Network");
                    }

                    if (algorithm == RLAlgorithmKind.PPO && firstActionCount == 0 && firstIsDiscrete)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': define at least one discrete action.");
                    }

                    // Skip when firstObservationSize == -1 (cast unavailable; not a real empty obs vector).
                    if (firstObservationSize == 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': could not infer a non-zero observation size.");
                    }

                    if (algorithm == RLAlgorithmKind.DQN && firstContinuousDims > 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': DQN supports discrete actions only; this group has continuous action dimensions.");
                    }

                    if (algorithm == RLAlgorithmKind.DQN && firstActionCount <= 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': DQN requires at least one discrete action.");
                    }

                    if (algorithm == RLAlgorithmKind.SAC
                        && firstActionCount >= 0
                        && firstContinuousDims >= 0
                        && firstActionCount > 0
                        && firstContinuousDims > 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': SAC does not support mixing discrete and continuous actions.");
                    }

                    if (algorithm == RLAlgorithmKind.SAC
                        && firstActionCount >= 0
                        && firstContinuousDims >= 0
                        && firstActionCount > 0
                        && firstContinuousDims <= 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': SAC currently supports continuous-only actions; this group is discrete.");
                    }

                    if (algorithm == RLAlgorithmKind.SAC
                        && firstActionCount >= 0
                        && firstContinuousDims >= 0
                        && firstActionCount <= 0
                        && firstContinuousDims <= 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': SAC requires at least one continuous action dimension.");
                    }

                    if (algorithm == RLAlgorithmKind.MCTS && firstContinuousDims > 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': MCTS supports discrete actions only; this group has continuous action dimensions.");
                    }

                    if (algorithm == RLAlgorithmKind.MCTS && firstActionCount <= 0)
                    {
                        validation.AddBlockingError($"Group '{groupSummary.GroupId}': MCTS requires at least one discrete action.");
                    }

                    // Custom: id must be provided (action-space validation is the trainer's responsibility)
                    if (algorithm == RLAlgorithmKind.Custom)
                    {
                        var customId = resolvedTrainerConfig?.CustomTrainerId ?? string.Empty;
                        if (string.IsNullOrWhiteSpace(customId))
                            validation.AddBlockingError($"Group '{groupSummary.GroupId}': Algorithm is Custom but CustomTrainerId is not set.");
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
                            validation.AddBlockingError($"Group '{groupSummary.GroupId}': {nodePath}: all agents in a group must use the same action type (discrete vs continuous).");
                        }

                        if (firstObservationSize >= 0 && observationSize >= 0 && observationSize != firstObservationSize)
                        {
                            validation.AddBlockingError($"Group '{groupSummary.GroupId}': {nodePath}: all agents must emit the same observation vector length.");
                        }

                        if (algorithm == RLAlgorithmKind.PPO && isDiscrete && firstActionCount >= 0 && actionCount >= 0 && actionCount != firstActionCount)
                        {
                            validation.AddBlockingError($"Group '{groupSummary.GroupId}': {nodePath}: all agents must share the same discrete action count.");
                        }
                    }
                }

                var configuredPairings = typedAcademy?.GetResolvedSelfPlayPairings() ?? new List<RLPolicyPairingConfig>();
                validation.HasSelfPlayPairings = configuredPairings.Count > 0;
                var requiredBatchCopies = ValidateSelfPlayPairings(configuredPairings, groupBindings, groupSummaryByBindingKey, validation);
                if (validation.BatchSize < requiredBatchCopies)
                {
                    validation.AddBlockingError(
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
            if (ownsRoot)
            {
                root.QueueFree();
            }
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
                validation.AddBlockingError($"Agent '{root.GetPathTo(node)}': observation inference failed: {error}");
                return 0;
            }

            return observationSize;
        }
        catch (Exception exception)
        {
            validation.AddBlockingError($"Agent '{root.GetPathTo(node)}': observation inference failed: {exception.Message}");
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
                validation.AddBlockingError($"Agent '{nodePath}' is in Inference mode but has no .rlmodel path.");
            }

            return false;
        }

        if (!modelPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase))
        {
            if (requireConfiguredModel)
                validation.AddBlockingError($"Agent '{nodePath}': inference model path must point to a .rlmodel file.");
            return false;
        }

        var checkpoint = RLModelLoader.LoadFromFile(modelPath);

        if (checkpoint is null)
        {
            if (requireConfiguredModel)
                validation.AddBlockingError($"Agent '{nodePath}': failed to load inference model '{modelPath}'.");
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
            validation.AddBlockingError(
                $"Agent '{nodePath}': model observation size {checkpoint.ObservationSize} " +
                $"does not match agent observation size {observationSize}.");
        }

        if (discreteCount >= 0 && checkpoint.DiscreteActionCount > 0 && checkpoint.DiscreteActionCount != discreteCount)
        {
            validation.AddBlockingError(
                $"Agent '{nodePath}': model discrete action count {checkpoint.DiscreteActionCount} " +
                $"does not match agent count {discreteCount}.");
        }

        if (continuousDims >= 0 && checkpoint.ContinuousActionDimensions > 0 && checkpoint.ContinuousActionDimensions != continuousDims)
        {
            validation.AddBlockingError(
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
                validation.AddBlockingError($"Pairing '{pairingName}' must reference two train-mode policy groups used in this scene.");
                continue;
            }

            if (string.Equals(groupAId, groupBId, StringComparison.Ordinal))
            {
                validation.AddBlockingError($"Pairing '{pairingName}' cannot pair a group against itself.");
                continue;
            }

            if (!pairing.TrainGroupA && !pairing.TrainGroupB)
            {
                validation.AddBlockingError($"Pairing '{pairingName}' must train at least one side.");
                continue;
            }

            if (!pairedGroups.Add(groupAId) || !pairedGroups.Add(groupBId))
            {
                validation.AddBlockingError(
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
            validation.AddBlockingError(
                $"Group '{groupSummary.GroupId}': LSTM/GRU layers require the native GDExtension library, but native layers are not available in the current editor/runtime.");
        }

        if (recurrentLayerCount > 1)
        {
            validation.AddBlockingError(
                $"Group '{groupSummary.GroupId}': only one recurrent trunk layer is currently supported end-to-end. " +
                $"This network config contains {recurrentLayerCount} recurrent layers.");
        }

        if (algorithm is RLAlgorithmKind.DQN or RLAlgorithmKind.SAC)
        {
            validation.AddBlockingError(
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
                builder.Append(runConfig?.ResourcePath ?? string.Empty);
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
                var trainingConfig = ReadResourceProperty(node, "TrainingConfig") as RLTrainingConfig;
                builder.Append(trainingConfig?.ResourcePath ?? string.Empty);
                builder.Append('|');
                builder.Append(trainingConfig?.Algorithm?.GetType().Name ?? string.Empty);
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
                builder.Append(policyGroupConfig?.ResolvedNetworkGraph is null ? "missing_graph" : "has_graph");
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
        validation.AddBlockingError(message);
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
