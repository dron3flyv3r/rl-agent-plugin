using System;
using System.Collections.Generic;
using System.IO;
using Godot;
using RlAgentPlugin.Runtime.Imitation;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Bottom-panel dock for the imitation learning workflow.
/// Registered via <c>AddDock(_imitationEditorDock)</c>.
///
/// The dock emits signals; the plugin editor handles them and calls
/// setter methods on this control to update status labels.
/// </summary>
[Tool]
public partial class RLImitationDock : VBoxContainer
{
    private enum TrainAlgorithmMode
    {
        BehaviorCloning = 0,
        DAgger = 1,
        AutoDAgger = 2,
    }

    // ── Signals ───────────────────────────────────────────────────────────────

    [Signal]
    public delegate void StartRecordingRequestedEventHandler(string scenePath, string agentGroupId, string outputName, bool scriptMode);

    [Signal]
    public delegate void StopRecordingRequestedEventHandler();

    [Signal]
    public delegate void SetRecordingSpeedRequestedEventHandler(float timeScale);

    [Signal]
    public delegate void PauseRecordingRequestedEventHandler(bool paused);

    [Signal]
    public delegate void SetStepModeRequestedEventHandler(bool enabled);

    [Signal]
    public delegate void StepActionRequestedEventHandler(int actionIndex);

    [Signal]
    public delegate void RefreshDatasetsRequestedEventHandler();

    [Signal]
    public delegate void StartBCTrainingRequestedEventHandler(
        string datasetPath,
        string checkpointPath,
        string networkGraphPath,
        int epochs,
        float learningRate);

    [Signal]
    public delegate void StartDAggerRequestedEventHandler(
        string scenePath,
        string agentGroupId,
        string datasetPath,
        string checkpointPath,
        string networkGraphPath,
        string outputName,
        int additionalFrames,
        float mixingBeta);

    [Signal]
    public delegate void StartAutoDAggerRequestedEventHandler(
        string scenePath,
        string agentGroupId,
        string datasetPath,
        string checkpointPath,
        string networkGraphPath,
        string outputName,
        string configJson);

    [Signal]
    public delegate void CancelBCTrainingRequestedEventHandler();

    [Signal]
    public delegate void ExportModelRequestedEventHandler(string checkpointPath, string destPath);

    // ── Record tab fields ─────────────────────────────────────────────────────
    private TabContainer? _tabs;
    private Label? _sceneNameLabel;
    private OptionButton? _agentDropdown;
    private OptionButton? _modeDropdown;
    private LineEdit? _outputNameEdit;
    private Button? _startRecordBtn;
    private Button? _stopRecordBtn;
    private Button? _pauseRecordBtn;
    private readonly List<Button> _speedButtons = new();
    private Label? _recordStatusLabel;
    private ScrollContainer? _recordStatsScroll;
    private VBoxContainer? _recordStatsRow;
    private Label? _statFramesVal;
    private Label? _statEpisodesVal;
    private Label? _statStepVal;
    private Label? _statRewardVal;
    private Label? _statPausedLabel;
    // Keep the old name as a null stub so EnsureUiReferences doesn't break.
    private Label? _recordStatsLabel;

    // Step mode — visible only for Human mode with discrete-only agents.
    private HBoxContainer? _stepModeRow;
    private CheckBox? _stepModeCheck;
    private HBoxContainer? _stepActionsRow;
    private Label? _stepModeUnavailableLabel;

    private bool _isScriptMode;
    private bool _isDiscreteOnly;
    private bool _capabilityKnown; // true once .cap file data has arrived for the current session

    // State set by SetSceneInfo.
    private string _activeScenePath = string.Empty;
    // Parallel arrays for dropdown items 1..N (item 0 = "All agents").
    private readonly List<string> _agentGroupIds = new();
    private readonly List<string> _agentNetworkGraphPaths = new();
    private readonly List<string> _agentInferencePaths = new();
    private readonly List<bool> _agentSupportsHuman = new();
    private readonly List<bool> _agentSupportsScript = new();
    // Currently resolved network graph path (from selection or override).
    private string _resolvedNetworkGraphPath = string.Empty;

    // ── Train tab fields ──────────────────────────────────────────────────────
    private OptionButton? _trainAlgorithmDropdown;
    private OptionButton? _datasetDropdown;
    private CheckBox? _warmStartCheck;
    private HBoxContainer? _warmStartRow;
    private VBoxContainer? _daggerSetupSection;
    private VBoxContainer? _bcRunSection;
    private VBoxContainer? _daggerRunSection;
    private VBoxContainer? _autoDaggerRunSection;
    private HBoxContainer? _bcHyperparamsRow;
    private HBoxContainer? _autoDaggerRoundsRow;
    private HSeparator? _bcRunDivider;
    private LineEdit? _checkpointPathEdit;
    // The abs path of the checkpoint most recently produced by BC training this session.
    private string _lastBCCheckpointPath = string.Empty;
    // Network graph display row.
    private Label? _networkGraphLabel;
    private HBoxContainer? _networkOverrideRow;
    private LineEdit? _networkOverrideEdit;
    private SpinBox? _epochsSpin;
    private LineEdit? _lrEdit;
    private Button? _trainBtn;
    private LineEdit? _daggerOutputNameEdit;
    private SpinBox? _daggerFramesSpin;
    private SpinBox? _daggerBetaSpin;
    private Button? _daggerBtn;
    private SpinBox? _autoDaggerRoundsSpin;
    private Button? _autoDaggerBtn;
    private Button? _cancelTrainBtn;
    private Button? _exportModelBtn;
    private ProgressBar? _progressBar;
    private Label? _trainStatusLabel;
    private Label? _warmStartSourceLabel;
    private Label? _trainSummaryLabel;
    private EditorFileDialog? _checkpointPickerDialog;
    private EditorFileDialog? _networkPickerDialog;
    private FileDialog? _exportDialog;
    private string _pendingExportCheckpointPath = string.Empty;
    private bool _bcTrainingRunning;
    private bool _daggerRunning;

    private string[] _datasetPaths = Array.Empty<string>();

    // ── Dataset Info tab fields ───────────────────────────────────────────────
    private Label? _infoFileLabel;
    private Label? _infoValidLabel;
    private Label? _infoFramesLabel;
    private Label? _infoEpisodesLabel;
    private Label? _infoAvgLenLabel;
    private Label? _infoAvgRewardLabel;
    private Label? _infoObsSizeLabel;
    private Label? _infoActionLabel;
    // Data preview
    private SpinBox? _previewCountSpin;
    private CheckBox? _previewFromEndCheck;
    private VBoxContainer? _previewGrid;
    private VBoxContainer? _previewContent;
    private ScrollContainer? _previewScroll;
    private DemonstrationDataset? _previewDataset;

    // Label column width shared across rows (keeps fields aligned).
    private const float LabelWidth = 110f;
    private const string NoDatasetLabel = "None";

    // Key stored as native Godot metadata so it survives C# wrapper recreation on assembly reload.
    private const string NeedsRefreshMeta = "_rl_needs_scene_refresh";
    private const string PersistedStatePath = "user://rl-agent-plugin/imitation_dock_state.json";
    private string _pendingSelectedAgentGroupId = string.Empty;
    private string _pendingSelectedDatasetPath = string.Empty;
    private bool _isRestoringState;
    private bool _statePersistenceReady;

    public RLImitationDock()
    {
        Name = "RLImitationDock";
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;
        AddThemeConstantOverride("separation", 0);

        // On C# assembly reload Godot re-runs the constructor on the existing native node,
        // which still has children from the previous run. Free them before rebuilding.
        for (var i = GetChildCount() - 1; i >= 0; i--)
            GetChild(i).Free();

        // Signal to the plugin that it must re-populate scene info.
        // Stored as native metadata so it survives across C# wrapper recreation.
        SetMeta(NeedsRefreshMeta, true);

        var tabs = new TabContainer
        {
            Name = "ImitationTabs",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        tabs.TabChanged += _ => PersistState();
        _tabs = tabs;
        AddChild(tabs);

        tabs.AddChild(BuildRecordTab());
        tabs.AddChild(BuildTrainTab());
        tabs.AddChild(BuildInfoTab());
        EnsureUiReferences();
        RestorePersistedState();
        _statePersistenceReady = true;
    }

    private void EnsureUiReferences()
    {
        _tabs ??= FindUiNode<TabContainer>("ImitationTabs");
        _sceneNameLabel ??= FindUiNode<Label>("SceneNameLabel");
        _agentDropdown ??= FindUiNode<OptionButton>("AgentDropdown");
        _modeDropdown ??= FindUiNode<OptionButton>("ModeDropdown");
        _outputNameEdit ??= FindUiNode<LineEdit>("OutputNameEdit");
        _startRecordBtn ??= FindUiNode<Button>("StartRecordButton");
        _stopRecordBtn ??= FindUiNode<Button>("StopRecordButton");
        _pauseRecordBtn ??= FindUiNode<Button>("PauseRecordButton");
        _stepModeRow ??= FindUiNode<HBoxContainer>("StepModeRow");
        _stepModeCheck ??= FindUiNode<CheckBox>("StepModeCheck");
        _stepActionsRow ??= FindUiNode<HBoxContainer>("StepActionsRow");
        _stepModeUnavailableLabel ??= FindUiNode<Label>("StepModeUnavailableLabel");
        _recordStatusLabel ??= FindUiNode<Label>("RecordStatusLabel");
        _recordStatsLabel ??= FindUiNode<Label>("RecordStatsLabel");
        _datasetDropdown ??= FindUiNode<OptionButton>("DatasetDropdown");
        _trainAlgorithmDropdown ??= FindUiNode<OptionButton>("TrainAlgorithmDropdown");
        _warmStartCheck ??= FindUiNode<CheckBox>("WarmStartCheck");
        _warmStartRow ??= FindUiNode<HBoxContainer>("WarmStartRow");
        _daggerSetupSection ??= FindUiNode<VBoxContainer>("DAggerSetupSection");
        _bcRunSection ??= FindUiNode<VBoxContainer>("BCRunSection");
        _daggerRunSection ??= FindUiNode<VBoxContainer>("DAggerRunSection");
        _autoDaggerRunSection ??= FindUiNode<VBoxContainer>("AutoDAggerRunSection");
        _bcHyperparamsRow ??= FindUiNode<HBoxContainer>("BCHyperparamsRow");
        _autoDaggerRoundsRow ??= FindUiNode<HBoxContainer>("AutoDAggerRoundsRow");
        _bcRunDivider ??= FindUiNode<HSeparator>("BCRunDivider");
        _checkpointPathEdit ??= FindUiNode<LineEdit>("CheckpointPathEdit");
        _networkGraphLabel ??= FindUiNode<Label>("NetworkGraphLabel");
        _networkOverrideRow ??= FindUiNode<HBoxContainer>("NetworkOverrideRow");
        _networkOverrideEdit ??= FindUiNode<LineEdit>("NetworkOverrideEdit");
        _epochsSpin ??= FindUiNode<SpinBox>("EpochsSpin");
        _lrEdit ??= FindUiNode<LineEdit>("LearningRateEdit");
        _trainBtn ??= FindUiNode<Button>("TrainButton");
        _daggerOutputNameEdit ??= FindUiNode<LineEdit>("DAggerOutputNameEdit");
        _daggerFramesSpin ??= FindUiNode<SpinBox>("DAggerFramesSpin");
        _daggerBetaSpin ??= FindUiNode<SpinBox>("DAggerBetaSpin");
        _daggerBtn ??= FindUiNode<Button>("DAggerButton");
        _autoDaggerRoundsSpin ??= FindUiNode<SpinBox>("AutoDAggerRoundsSpin");
        _autoDaggerBtn ??= FindUiNode<Button>("AutoDAggerButton");
        _cancelTrainBtn ??= FindUiNode<Button>("CancelTrainButton");
        _exportModelBtn ??= FindUiNode<Button>("ExportModelButton");
        _progressBar ??= FindUiNode<ProgressBar>("TrainingProgressBar");
        _trainStatusLabel ??= FindUiNode<Label>("TrainStatusLabel");
        _warmStartSourceLabel ??= FindUiNode<Label>("WarmStartSourceLabel");
        _trainSummaryLabel ??= FindUiNode<Label>("TrainSummaryLabel");

        if (_sceneNameLabel is not null && !IsInstanceValid(_sceneNameLabel))
            _sceneNameLabel = FindUiNode<Label>("SceneNameLabel");
        if (_tabs is not null && !IsInstanceValid(_tabs))
            _tabs = FindUiNode<TabContainer>("ImitationTabs");
        if (_agentDropdown is not null && !IsInstanceValid(_agentDropdown))
            _agentDropdown = FindUiNode<OptionButton>("AgentDropdown");
        if (_modeDropdown is not null && !IsInstanceValid(_modeDropdown))
            _modeDropdown = FindUiNode<OptionButton>("ModeDropdown");
        if (_outputNameEdit is not null && !IsInstanceValid(_outputNameEdit))
            _outputNameEdit = FindUiNode<LineEdit>("OutputNameEdit");
        if (_startRecordBtn is not null && !IsInstanceValid(_startRecordBtn))
            _startRecordBtn = FindUiNode<Button>("StartRecordButton");
        if (_stopRecordBtn is not null && !IsInstanceValid(_stopRecordBtn))
            _stopRecordBtn = FindUiNode<Button>("StopRecordButton");
        if (_pauseRecordBtn is not null && !IsInstanceValid(_pauseRecordBtn))
            _pauseRecordBtn = FindUiNode<Button>("PauseRecordButton");
        if (_stepModeRow is not null && !IsInstanceValid(_stepModeRow))
            _stepModeRow = FindUiNode<HBoxContainer>("StepModeRow");
        if (_stepModeCheck is not null && !IsInstanceValid(_stepModeCheck))
            _stepModeCheck = FindUiNode<CheckBox>("StepModeCheck");
        if (_stepActionsRow is not null && !IsInstanceValid(_stepActionsRow))
            _stepActionsRow = FindUiNode<HBoxContainer>("StepActionsRow");
        if (_stepModeUnavailableLabel is not null && !IsInstanceValid(_stepModeUnavailableLabel))
            _stepModeUnavailableLabel = FindUiNode<Label>("StepModeUnavailableLabel");
        if (_recordStatusLabel is not null && !IsInstanceValid(_recordStatusLabel))
            _recordStatusLabel = FindUiNode<Label>("RecordStatusLabel");
        _recordStatsScroll ??= FindUiNode<ScrollContainer>("StatsScroll");
        if (_recordStatsScroll is not null && !IsInstanceValid(_recordStatsScroll))
            _recordStatsScroll = FindUiNode<ScrollContainer>("StatsScroll");
        _recordStatsRow    ??= FindUiNode<VBoxContainer>("RecordStatsRow");
        if (_recordStatsRow is not null && !IsInstanceValid(_recordStatsRow))
            _recordStatsRow = FindUiNode<VBoxContainer>("RecordStatsRow");
        _statFramesVal     ??= FindUiNode<Label>("StatFramesVal");
        _statEpisodesVal   ??= FindUiNode<Label>("StatEpisodesVal");
        _statStepVal       ??= FindUiNode<Label>("StatStepVal");
        _statRewardVal     ??= FindUiNode<Label>("StatRewardVal");
        _statPausedLabel   ??= FindUiNode<Label>("StatPausedLabel");
        if (_datasetDropdown is not null && !IsInstanceValid(_datasetDropdown))
            _datasetDropdown = FindUiNode<OptionButton>("DatasetDropdown");
        if (_trainAlgorithmDropdown is not null && !IsInstanceValid(_trainAlgorithmDropdown))
            _trainAlgorithmDropdown = FindUiNode<OptionButton>("TrainAlgorithmDropdown");
        if (_warmStartCheck is not null && !IsInstanceValid(_warmStartCheck))
            _warmStartCheck = FindUiNode<CheckBox>("WarmStartCheck");
        if (_warmStartRow is not null && !IsInstanceValid(_warmStartRow))
            _warmStartRow = FindUiNode<HBoxContainer>("WarmStartRow");
        if (_daggerSetupSection is not null && !IsInstanceValid(_daggerSetupSection))
            _daggerSetupSection = FindUiNode<VBoxContainer>("DAggerSetupSection");
        if (_bcRunSection is not null && !IsInstanceValid(_bcRunSection))
            _bcRunSection = FindUiNode<VBoxContainer>("BCRunSection");
        if (_daggerRunSection is not null && !IsInstanceValid(_daggerRunSection))
            _daggerRunSection = FindUiNode<VBoxContainer>("DAggerRunSection");
        if (_autoDaggerRunSection is not null && !IsInstanceValid(_autoDaggerRunSection))
            _autoDaggerRunSection = FindUiNode<VBoxContainer>("AutoDAggerRunSection");
        if (_bcHyperparamsRow is not null && !IsInstanceValid(_bcHyperparamsRow))
            _bcHyperparamsRow = FindUiNode<HBoxContainer>("BCHyperparamsRow");
        if (_autoDaggerRoundsRow is not null && !IsInstanceValid(_autoDaggerRoundsRow))
            _autoDaggerRoundsRow = FindUiNode<HBoxContainer>("AutoDAggerRoundsRow");
        if (_bcRunDivider is not null && !IsInstanceValid(_bcRunDivider))
            _bcRunDivider = FindUiNode<HSeparator>("BCRunDivider");
        if (_checkpointPathEdit is not null && !IsInstanceValid(_checkpointPathEdit))
            _checkpointPathEdit = FindUiNode<LineEdit>("CheckpointPathEdit");
        if (_networkGraphLabel is not null && !IsInstanceValid(_networkGraphLabel))
            _networkGraphLabel = FindUiNode<Label>("NetworkGraphLabel");
        if (_networkOverrideRow is not null && !IsInstanceValid(_networkOverrideRow))
            _networkOverrideRow = FindUiNode<HBoxContainer>("NetworkOverrideRow");
        if (_networkOverrideEdit is not null && !IsInstanceValid(_networkOverrideEdit))
            _networkOverrideEdit = FindUiNode<LineEdit>("NetworkOverrideEdit");
        if (_epochsSpin is not null && !IsInstanceValid(_epochsSpin))
            _epochsSpin = FindUiNode<SpinBox>("EpochsSpin");
        if (_lrEdit is not null && !IsInstanceValid(_lrEdit))
            _lrEdit = FindUiNode<LineEdit>("LearningRateEdit");
        if (_trainBtn is not null && !IsInstanceValid(_trainBtn))
            _trainBtn = FindUiNode<Button>("TrainButton");
        if (_daggerOutputNameEdit is not null && !IsInstanceValid(_daggerOutputNameEdit))
            _daggerOutputNameEdit = FindUiNode<LineEdit>("DAggerOutputNameEdit");
        if (_daggerFramesSpin is not null && !IsInstanceValid(_daggerFramesSpin))
            _daggerFramesSpin = FindUiNode<SpinBox>("DAggerFramesSpin");
        if (_daggerBetaSpin is not null && !IsInstanceValid(_daggerBetaSpin))
            _daggerBetaSpin = FindUiNode<SpinBox>("DAggerBetaSpin");
        if (_daggerBtn is not null && !IsInstanceValid(_daggerBtn))
            _daggerBtn = FindUiNode<Button>("DAggerButton");
        if (_autoDaggerRoundsSpin is not null && !IsInstanceValid(_autoDaggerRoundsSpin))
            _autoDaggerRoundsSpin = FindUiNode<SpinBox>("AutoDAggerRoundsSpin");
        if (_autoDaggerBtn is not null && !IsInstanceValid(_autoDaggerBtn))
            _autoDaggerBtn = FindUiNode<Button>("AutoDAggerButton");
        if (_cancelTrainBtn is not null && !IsInstanceValid(_cancelTrainBtn))
            _cancelTrainBtn = FindUiNode<Button>("CancelTrainButton");
        if (_exportModelBtn is not null && !IsInstanceValid(_exportModelBtn))
            _exportModelBtn = FindUiNode<Button>("ExportModelButton");
        if (_progressBar is not null && !IsInstanceValid(_progressBar))
            _progressBar = FindUiNode<ProgressBar>("TrainingProgressBar");
        if (_trainStatusLabel is not null && !IsInstanceValid(_trainStatusLabel))
            _trainStatusLabel = FindUiNode<Label>("TrainStatusLabel");
        if (_warmStartSourceLabel is not null && !IsInstanceValid(_warmStartSourceLabel))
            _warmStartSourceLabel = FindUiNode<Label>("WarmStartSourceLabel");
        if (_trainSummaryLabel is not null && !IsInstanceValid(_trainSummaryLabel))
            _trainSummaryLabel = FindUiNode<Label>("TrainSummaryLabel");

        _infoFileLabel   ??= FindUiNode<Label>("InfoFileLabel");
        _infoValidLabel  ??= FindUiNode<Label>("InfoValidLabel");
        _infoFramesLabel ??= FindUiNode<Label>("InfoFramesLabel");
        _infoEpisodesLabel ??= FindUiNode<Label>("InfoEpisodesLabel");
        _infoAvgLenLabel ??= FindUiNode<Label>("InfoAvgLenLabel");
        _infoAvgRewardLabel ??= FindUiNode<Label>("InfoAvgRewardLabel");
        _infoObsSizeLabel ??= FindUiNode<Label>("InfoObsSizeLabel");
        _infoActionLabel ??= FindUiNode<Label>("InfoActionLabel");
        _previewCountSpin ??= FindUiNode<SpinBox>("PreviewCountSpin");
        _previewFromEndCheck ??= FindUiNode<CheckBox>("PreviewFromEndCheck");
        _previewGrid ??= FindUiNode<VBoxContainer>("PreviewGrid");
        _previewContent ??= FindUiNode<VBoxContainer>("PreviewContent");
        _previewScroll ??= FindUiNode<ScrollContainer>("PreviewScroll");

        if (_previewCountSpin is not null && !IsInstanceValid(_previewCountSpin))
            _previewCountSpin = FindUiNode<SpinBox>("PreviewCountSpin");
        if (_previewFromEndCheck is not null && !IsInstanceValid(_previewFromEndCheck))
            _previewFromEndCheck = FindUiNode<CheckBox>("PreviewFromEndCheck");
        if (_previewGrid is not null && !IsInstanceValid(_previewGrid))
            _previewGrid = FindUiNode<VBoxContainer>("PreviewGrid");
        if (_previewContent is not null && !IsInstanceValid(_previewContent))
            _previewContent = FindUiNode<VBoxContainer>("PreviewContent");
        if (_previewScroll is not null && !IsInstanceValid(_previewScroll))
            _previewScroll = FindUiNode<ScrollContainer>("PreviewScroll");
    }

    private T? FindUiNode<T>(string name) where T : Node
        => FindChild(name, true, false) as T;

    // ── Plugin-editor -> dock setters ─────────────────────────────────────────

    /// <summary>
    /// Called whenever the active editor scene changes.
    /// displayNames, groupIds, and networkGraphPaths are parallel arrays — one entry per unique agent group.
    /// </summary>
    public void SetSceneInfo(
        string scenePath,
        string[] displayNames,
        string[] groupIds,
        string[] networkGraphPaths,
        string[] inferencePaths,
        bool[] humanSupports,
        bool[] scriptSupports)
    {
        EnsureUiReferences();
        _activeScenePath = scenePath;
        if (!string.IsNullOrWhiteSpace(scenePath) && HasMeta(NeedsRefreshMeta))
            RemoveMeta(NeedsRefreshMeta);

        if (_sceneNameLabel is not null)
        {
            if (string.IsNullOrWhiteSpace(scenePath))
            {
                _sceneNameLabel.Text = "— no scene open —";
                _sceneNameLabel.TooltipText = string.Empty;
            }
            else
            {
                _sceneNameLabel.Text = Path.GetFileNameWithoutExtension(scenePath);
                _sceneNameLabel.TooltipText = scenePath;
            }
        }

        if (_agentDropdown is null) return;

        var previousGroupId = SelectedGroupId();

        _agentDropdown.Clear();
        _agentGroupIds.Clear();
        _agentNetworkGraphPaths.Clear();
        _agentInferencePaths.Clear();
        _agentSupportsHuman.Clear();
        _agentSupportsScript.Clear();

        _agentDropdown.AddItem("All agents");

        for (var i = 0; i < displayNames.Length; i++)
        {
            _agentDropdown.AddItem(displayNames[i]);
            _agentGroupIds.Add(i < groupIds.Length ? groupIds[i] : string.Empty);
            _agentNetworkGraphPaths.Add(i < networkGraphPaths.Length ? networkGraphPaths[i] : string.Empty);
            _agentInferencePaths.Add(i < inferencePaths.Length ? inferencePaths[i] : string.Empty);
            _agentSupportsHuman.Add(i < humanSupports.Length && humanSupports[i]);
            _agentSupportsScript.Add(i < scriptSupports.Length && scriptSupports[i]);
        }

        // Restore previous selection if it still exists.
        var restoredIdx = _agentGroupIds.IndexOf(previousGroupId);
        _agentDropdown.Selected = restoredIdx >= 0 ? restoredIdx + 1 : 0;
        ApplyPendingAgentSelection();

        var hasScene = !string.IsNullOrWhiteSpace(scenePath);
        if (_startRecordBtn is not null)
            _startRecordBtn.Disabled = !hasScene;

        RefreshNetworkGraphDisplay();
        RefreshWarmStartPath();
        // Capability data is stale when the scene changes.
        _capabilityKnown = false;
        _isDiscreteOnly = false;
        UpdateModeAvailability();
        UpdateStepModeAvailability();
    }

    public void SetRecordingStatus(string message, bool isRecording)
    {
        EnsureUiReferences();
        if (_recordStatusLabel is not null)
            _recordStatusLabel.Text = message;
        if (_startRecordBtn is not null)
            _startRecordBtn.Disabled = isRecording;
        if (_stopRecordBtn is not null)
            _stopRecordBtn.Disabled = !isRecording;
        if (_pauseRecordBtn is not null)
        {
            _pauseRecordBtn.Disabled = !isRecording;
            if (!isRecording)
            {
                _pauseRecordBtn.SetPressedNoSignal(false);
                _pauseRecordBtn.Text = "Pause";
            }
        }

        if (!isRecording && _recordStatsRow is not null)
            if (_recordStatsScroll is not null) _recordStatsScroll.Visible = false;

        // Reset step mode when recording stops — go back to selection-driven state.
        if (!isRecording)
        {
            _capabilityKnown = false;
            _isDiscreteOnly = false;
            if (_stepModeCheck is not null)
                _stepModeCheck.SetPressedNoSignal(false);
            // Clear action buttons so they don't persist from a previous session.
            PopulateStepActionButtons(Array.Empty<string>());
            UpdateModeAvailability();
            UpdateStepModeAvailability();
        }
    }

    public void SetRecordingStats(long frames, int episodes, int episodeSteps, float episodeReward, bool paused)
    {
        EnsureUiReferences();
        if (_recordStatsRow is null) return;
        if (_recordStatsScroll is not null) _recordStatsScroll.Visible = true;
        if (_statFramesVal   is not null) _statFramesVal.Text   = frames.ToString("N0");
        if (_statEpisodesVal is not null) _statEpisodesVal.Text = episodes.ToString("N0");
        if (_statStepVal     is not null) _statStepVal.Text     = episodeSteps.ToString("N0");
        if (_statRewardVal   is not null) _statRewardVal.Text   = episodeReward.ToString("F3");
        if (_statPausedLabel is not null) _statPausedLabel.Visible = paused;
    }

    /// <summary>
    /// Called by the editor when the recording bootstrap's capabilities file is read.
    /// Enables/disables the step mode controls based on whether the selected agent
    /// uses only discrete actions. scriptMode suppresses step mode (no keyboard input).
    /// </summary>
    public void SetRecordingCapabilities(bool onlyDiscrete, string[] actionLabels, bool scriptMode)
    {
        EnsureUiReferences();
        _isDiscreteOnly = onlyDiscrete;
        _capabilityKnown = true;
        UpdateStepModeAvailability();

        // Populate action buttons now that the action space is confirmed.
        if (onlyDiscrete && !scriptMode && actionLabels.Length > 0)
        {
            PopulateStepActionButtons(actionLabels);
            // Show the action row if step mode is already enabled.
            if (_stepActionsRow is not null && _stepModeCheck is not null)
                _stepActionsRow.Visible = _stepModeCheck.ButtonPressed;
        }
    }

    public void SetTrainingProgress(float progress01, float loss, int epoch, int totalEpochs)
    {
        EnsureUiReferences();
        if (_progressBar is not null)
        {
            _progressBar.Value = progress01 * 100.0;
            _progressBar.Visible = true;
        }
        if (_trainStatusLabel is not null)
            _trainStatusLabel.Text = $"Epoch {epoch}/{totalEpochs}  loss={loss:F4}";
    }

    public void SetTrainingStatus(string message, bool isRunning)
    {
        EnsureUiReferences();
        if (_trainStatusLabel is not null)
            _trainStatusLabel.Text = message;
        if (_trainBtn is not null)
            _trainBtn.Disabled = isRunning;
        if (_progressBar is not null)
            _progressBar.Visible = isRunning;
        if (isRunning && _trainSummaryLabel is not null)
            _trainSummaryLabel.Visible = false;
        if (isRunning && _exportModelBtn is not null)
            _exportModelBtn.Visible = false;
        _bcTrainingRunning = isRunning;
        RefreshTrainActionState();
    }

    public void SetWarmStartSource(string checkpointPath)
    {
        EnsureUiReferences();
        if (_warmStartSourceLabel is null)
            return;

        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            _warmStartSourceLabel.Text = string.Empty;
            _warmStartSourceLabel.TooltipText = string.Empty;
            _warmStartSourceLabel.Visible = false;
            return;
        }

        var displayPath = checkpointPath.StartsWith("res://", StringComparison.Ordinal)
            ? checkpointPath
            : ProjectSettings.LocalizePath(checkpointPath);

        _warmStartSourceLabel.Text = $"Warm-started from: {displayPath}";
        _warmStartSourceLabel.TooltipText = checkpointPath;
        _warmStartSourceLabel.Visible = true;
    }

    /// <summary>
    /// Called after training completes. Shows a summary with final stats and the saved path
    /// as a project-relative res:// path so the user can locate the file easily.
    /// </summary>
    public void SetTrainingSummary(int epochs, float finalLoss, string absOutputPath)
    {
        EnsureUiReferences();
        if (_trainSummaryLabel is null) return;

        var resPath = ProjectSettings.LocalizePath(absOutputPath);
        _trainSummaryLabel.Text =
            $"Epochs: {epochs}    Final loss: {finalLoss:F4}\n" +
            $"Saved to: {resPath}";
        _trainSummaryLabel.TooltipText = absOutputPath;
        _trainSummaryLabel.Visible = true;

        // Remember this checkpoint so warm-start auto-selects it next time.
        _lastBCCheckpointPath = absOutputPath;
        if (_checkpointPathEdit is not null)
            _checkpointPathEdit.Text = absOutputPath;

        if (_exportModelBtn is not null) _exportModelBtn.Visible = true;
    }

    public void SetDaggerStatus(string message, bool isRunning)
    {
        EnsureUiReferences();
        if (_trainStatusLabel is not null)
            _trainStatusLabel.Text = message;
        if (_progressBar is not null)
            _progressBar.Visible = isRunning;
        if (isRunning && _trainSummaryLabel is not null)
            _trainSummaryLabel.Visible = false;
        if (isRunning && _exportModelBtn is not null)
            _exportModelBtn.Visible = false;
        _daggerRunning = isRunning;
        RefreshTrainActionState();
    }

    public void SetDaggerProgress(float progress01, int collectedFrames, int targetFrames, int episodes)
    {
        EnsureUiReferences();
        if (_progressBar is not null)
        {
            _progressBar.Value = progress01 * 100.0;
            _progressBar.Visible = true;
        }
        if (_trainStatusLabel is not null)
            _trainStatusLabel.Text = $"DAgger {collectedFrames}/{targetFrames} frames  episodes={episodes}";
    }

    public void SetDaggerSummary(string resDatasetPath, int seedFrames, int addedFrames)
    {
        EnsureUiReferences();
        if (_trainSummaryLabel is null) return;

        _trainSummaryLabel.Text =
            $"DAgger dataset: {resDatasetPath}\n" +
            $"Seed frames: {seedFrames}    Added frames: {addedFrames}";
        _trainSummaryLabel.TooltipText = resDatasetPath;
        _trainSummaryLabel.Visible = true;
    }

    public void SetAutoDaggerSummary(int rounds, string resDatasetPath, string checkpointPath)
    {
        EnsureUiReferences();
        if (_trainSummaryLabel is null) return;

        var checkpointResPath = checkpointPath.StartsWith("res://", StringComparison.Ordinal)
            ? checkpointPath
            : ProjectSettings.LocalizePath(checkpointPath);

        _trainSummaryLabel.Text =
            $"Auto DAgger complete after {rounds} round(s).\n" +
            $"Final dataset: {resDatasetPath}\n" +
            $"Final checkpoint: {checkpointResPath}";
        _trainSummaryLabel.TooltipText = $"{resDatasetPath}\n{checkpointPath}";
        _trainSummaryLabel.Visible = true;

        if (_exportModelBtn is not null) _exportModelBtn.Visible = true;
    }

    public void RefreshDatasetList(string[] datasetPaths)
    {
        EnsureUiReferences();
        var previousSelectedPath = GetSelectedDatasetPath();
        _datasetPaths = datasetPaths;
        if (_datasetDropdown is null) return;

        _datasetDropdown.Clear();
        _datasetDropdown.AddItem(NoDatasetLabel);
        _datasetDropdown.Disabled = false;
        foreach (var path in datasetPaths)
            _datasetDropdown.AddItem(Path.GetFileName(path));

        var selectedIdx = 0;
        var preferredPath = string.Empty;
        if (!string.IsNullOrWhiteSpace(_pendingSelectedDatasetPath)
            && Array.IndexOf(datasetPaths, _pendingSelectedDatasetPath) >= 0)
        {
            preferredPath = _pendingSelectedDatasetPath;
        }
        else if (!string.IsNullOrWhiteSpace(previousSelectedPath)
                 && Array.IndexOf(datasetPaths, previousSelectedPath) >= 0)
        {
            preferredPath = previousSelectedPath;
        }

        if (!string.IsNullOrWhiteSpace(preferredPath))
        {
            var idx = Array.IndexOf(datasetPaths, preferredPath);
            if (idx >= 0) selectedIdx = idx + 1;
        }

        _datasetDropdown.Select(selectedIdx);
        _pendingSelectedDatasetPath = selectedIdx > 0 ? datasetPaths[selectedIdx - 1] : string.Empty;
        if (selectedIdx > 0)
            PopulateDatasetInfo(datasetPaths[selectedIdx - 1]);
        else
            ClearDatasetInfo();
        RefreshDaggerOutputName();
        PersistState();
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    public string GetSelectedDatasetPath()
    {
        EnsureUiReferences();
        if (_datasetDropdown is null || _datasetPaths.Length == 0) return string.Empty;
        var idx = _datasetDropdown.Selected;
        return idx > 0 && (idx - 1) < _datasetPaths.Length ? _datasetPaths[idx - 1] : string.Empty;
    }

    private string GetPersistedDatasetPath()
    {
        var selectedPath = GetSelectedDatasetPath();
        return !string.IsNullOrWhiteSpace(selectedPath)
            ? selectedPath
            : _pendingSelectedDatasetPath;
    }

    public bool IsWarmStartEnabled()
    {
        EnsureUiReferences();
        return _warmStartCheck?.ButtonPressed ?? false;
    }

    public string GetWarmStartCheckpointPath()
    {
        EnsureUiReferences();
        return _checkpointPathEdit?.Text?.Trim() ?? string.Empty;
    }

    private void RefreshTrainActionState()
    {
        EnsureUiReferences();
        if (_trainBtn is not null)
            _trainBtn.Disabled = _bcTrainingRunning || _daggerRunning;
        if (_daggerBtn is not null)
            _daggerBtn.Disabled = _bcTrainingRunning || _daggerRunning;
        if (_autoDaggerBtn is not null)
            _autoDaggerBtn.Disabled = _bcTrainingRunning || _daggerRunning;
        if (_cancelTrainBtn is not null)
            _cancelTrainBtn.Disabled = !_bcTrainingRunning && !_daggerRunning;
        UpdateTrainAlgorithmUi();
    }

    // ── Record tab ────────────────────────────────────────────────────────────

    private Control BuildRecordTab()
    {
        var root = new MarginContainer { Name = "Record" };
        SetMargins(root, 8);

        // ── Two-column layout ─────────────────────────────────────────────────
        var columns = new HBoxContainer();
        columns.AddThemeConstantOverride("separation", 12);
        columns.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        columns.SizeFlagsVertical = SizeFlags.ExpandFill;
        root.AddChild(columns);

        // ── Left column: setup ────────────────────────────────────────────────
        var left = new VBoxContainer();
        left.AddThemeConstantOverride("separation", 6);
        left.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        left.SizeFlagsVertical = SizeFlags.ExpandFill;
        columns.AddChild(left);

        // Scene name — read-only, auto-populated.
        var sceneRow = new HBoxContainer();
        sceneRow.AddThemeConstantOverride("separation", 6);
        left.AddChild(sceneRow);
        sceneRow.AddChild(MakeAlignedLabel("Scene"));
        _sceneNameLabel = new Label
        {
            Name = "SceneNameLabel",
            Text = "— no scene open —",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            ClipText = true,
            VerticalAlignment = VerticalAlignment.Center,
        };
        sceneRow.AddChild(_sceneNameLabel);

        // Agent — dropdown auto-populated from scene scan.
        var agentRow = new HBoxContainer();
        agentRow.AddThemeConstantOverride("separation", 6);
        left.AddChild(agentRow);
        agentRow.AddChild(MakeAlignedLabel("Agent"));
        _agentDropdown = new OptionButton
        {
            Name = "AgentDropdown",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _agentDropdown.AddItem("All agents");
        _agentDropdown.ItemSelected += _ =>
        {
            RefreshNetworkGraphDisplay();
            RefreshWarmStartPath();
            UpdateStepModeAvailability();
            PersistState();
        };
        agentRow.AddChild(_agentDropdown);

        // Mode — Human (keyboard) or Script (agent's OnScriptedInput heuristic).
        var modeRow = new HBoxContainer();
        modeRow.AddThemeConstantOverride("separation", 6);
        left.AddChild(modeRow);
        modeRow.AddChild(MakeAlignedLabel("Mode"));
        _modeDropdown = new OptionButton
        {
            Name = "ModeDropdown",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _modeDropdown.AddItem("Human (keyboard)");
        _modeDropdown.AddItem("Script (heuristic)");
        _modeDropdown.ItemSelected += OnModeChanged;
        modeRow.AddChild(_modeDropdown);

        // Output name.
        var outRow = MakeFieldRow("Output Name", out _outputNameEdit, "demo", out var noBtn);
        _outputNameEdit.Name = "OutputNameEdit";
        _outputNameEdit.Text = "demo";
        _outputNameEdit.TextChanged += _ => PersistState();
        noBtn.Visible = false;
        left.AddChild(outRow);

        // ── Divider ───────────────────────────────────────────────────────────
        columns.AddChild(new VSeparator { SizeFlagsVertical = SizeFlags.ExpandFill });

        // ── Right column: controls ────────────────────────────────────────────
        var right = new VBoxContainer();
        right.AddThemeConstantOverride("separation", 6);
        right.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        right.SizeFlagsVertical = SizeFlags.ExpandFill;
        columns.AddChild(right);

        // Speed presets.
        var speedRow = new HBoxContainer();
        speedRow.AddThemeConstantOverride("separation", 4);
        right.AddChild(speedRow);
        speedRow.AddChild(new Label
        {
            Text = "Speed",
            VerticalAlignment = VerticalAlignment.Center,
            CustomMinimumSize = new Vector2(42f, 0f),
        });

        var speedGroup = new ButtonGroup();
        var speeds = new[] { (0.25f, "1/4x"), (0.5f, "1/2x"), (1.0f, "1x"), (2.0f, "2x"), (4.0f, "4x") };
        foreach (var (scale, label) in speeds)
        {
            var btn = new Button
            {
                Text = label,
                ToggleMode = true,
                ButtonGroup = speedGroup,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0f, 24f),
            };
            var capturedScale = scale;
            btn.Pressed += () => EmitSignal(SignalName.SetRecordingSpeedRequested, capturedScale);
            speedRow.AddChild(btn);
            _speedButtons.Add(btn);
        }
        _speedButtons[2].SetPressedNoSignal(true); // default 1x

        // Start / Stop / Pause.
        var btnRow = new HBoxContainer();
        btnRow.AddThemeConstantOverride("separation", 4);
        right.AddChild(btnRow);

        _startRecordBtn = new Button
        {
            Name = "StartRecordButton",
            Text = "Start Recording",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0f, 28f),
            Disabled = true,
        };
        _startRecordBtn.Pressed += OnStartRecordPressed;
        btnRow.AddChild(_startRecordBtn);

        _stopRecordBtn = new Button
        {
            Name = "StopRecordButton",
            Text = "Stop",
            Disabled = true,
            CustomMinimumSize = new Vector2(60f, 28f),
        };
        _stopRecordBtn.Pressed += OnStopRecordPressed;
        btnRow.AddChild(_stopRecordBtn);

        _pauseRecordBtn = new Button
        {
            Name = "PauseRecordButton",
            Text = "Pause",
            ToggleMode = true,
            Disabled = true,
            TooltipText = "Pause / Resume recording (game keeps running; no frames written while paused)",
            CustomMinimumSize = new Vector2(60f, 28f),
        };
        _pauseRecordBtn.Toggled += paused =>
        {
            if (_pauseRecordBtn is not null)
                _pauseRecordBtn.Text = paused ? "Resume" : "Pause";
            EmitSignal(SignalName.PauseRecordingRequested, paused);
        };
        btnRow.AddChild(_pauseRecordBtn);

        // Step mode — only meaningful in Human mode with a discrete-only agent.
        _stepModeRow = new HBoxContainer { Name = "StepModeRow" };
        _stepModeRow.AddThemeConstantOverride("separation", 6);
        right.AddChild(_stepModeRow);

        _stepModeCheck = new CheckBox
        {
            Name = "StepModeCheck",
            Text = "Step mode",
            Disabled = true,
            TooltipText = "Only available for discrete-action agents. Start recording to enable.",
        };
        _stepModeCheck.Toggled += OnStepModeToggled;
        _stepModeRow.AddChild(_stepModeCheck);

        _stepModeUnavailableLabel = new Label
        {
            Name = "StepModeUnavailableLabel",
            Text = "(start recording to detect action space)",
            Modulate = new Color(1f, 1f, 1f, 0.45f),
            VerticalAlignment = VerticalAlignment.Center,
        };
        _stepModeRow.AddChild(_stepModeUnavailableLabel);

        // Step action buttons (hidden until step mode is active).
        _stepActionsRow = new HBoxContainer { Name = "StepActionsRow" };
        _stepActionsRow.AddThemeConstantOverride("separation", 4);
        _stepActionsRow.Visible = false;
        right.AddChild(_stepActionsRow);
        _stepActionsRow.AddChild(new Label
        {
            Text = "Step",
            VerticalAlignment = VerticalAlignment.Center,
            CustomMinimumSize = new Vector2(42f, 0f),
        });

        // Live stats — visible only while recording, wrapped in a scroll container.
        _recordStatsScroll = new ScrollContainer
        {
            Name = "StatsScroll",
            Visible = false,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        right.AddChild(_recordStatsScroll);

        _recordStatsRow = new VBoxContainer { Name = "RecordStatsRow" };
        _recordStatsRow.AddThemeConstantOverride("separation", 2);
        _recordStatsRow.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        _recordStatsScroll.AddChild(_recordStatsRow);

        static HBoxContainer StatLine(string key, string valName, out Label val)
        {
            var row = new HBoxContainer();
            row.AddThemeConstantOverride("separation", 6);
            row.AddChild(new Label
            {
                Text = key,
                VerticalAlignment = VerticalAlignment.Center,
                Modulate = new Color(1f, 1f, 1f, 0.5f),
                CustomMinimumSize = new Vector2(68f, 0f),
            });
            val = new Label
            {
                Name = valName,
                Text = "—",
                VerticalAlignment = VerticalAlignment.Center,
                Modulate = new Color(0.8f, 0.95f, 1f, 1f),
            };
            row.AddChild(val);
            return row;
        }

        _recordStatsRow.AddChild(StatLine("Frames",   "StatFramesVal",   out _statFramesVal));
        _recordStatsRow.AddChild(StatLine("Episodes", "StatEpisodesVal", out _statEpisodesVal));
        _recordStatsRow.AddChild(StatLine("Step",     "StatStepVal",     out _statStepVal));
        _recordStatsRow.AddChild(StatLine("Reward",   "StatRewardVal",   out _statRewardVal));

        _statPausedLabel = new Label
        {
            Name = "StatPausedLabel",
            Text = "PAUSED",
            Modulate = new Color(1f, 0.85f, 0.3f, 1f),
            Visible = false,
        };
        _recordStatsRow.AddChild(_statPausedLabel);

        // Status / error message — very bottom.
        _recordStatusLabel = new Label
        {
            Name = "RecordStatusLabel",
            Text = "Open a scene to begin.",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        right.AddChild(_recordStatusLabel);

        return root;
    }

    /// <summary>
    /// Re-evaluates step mode availability from current selection state.
    /// Called whenever mode, agent selection, or capability data changes.
    /// Rules:
    ///   - Script mode → row hidden (step mode meaningless without keyboard)
    ///   - "All agents" → checkbox disabled (can't step-inject multiple heterogeneous agents)
    ///   - Specific agent + Human mode → checkbox ENABLED immediately so the user can
    ///     pre-select step mode before starting recording. Action buttons only appear once the
    ///     .cap file confirms the agent is discrete-only.
    /// </summary>
    private void UpdateStepModeAvailability()
    {
        EnsureUiReferences();
        if (_stepModeRow is null || _stepModeCheck is null ||
            _stepActionsRow is null || _stepModeUnavailableLabel is null) return;

        // Script mode: hide the entire row.
        if (_isScriptMode)
        {
            _stepModeRow.Visible = false;
            _stepModeCheck.SetPressedNoSignal(false);
            _stepActionsRow.Visible = false;
            return;
        }

        _stepModeRow.Visible = true;

        var specificAgentSelected = _agentDropdown is not null && _agentDropdown.Selected > 0;

        if (!specificAgentSelected)
        {
            // "All agents" — disable and explain.
            _stepModeCheck.Disabled = true;
            _stepModeCheck.SetPressedNoSignal(false);
            _stepActionsRow.Visible = false;
            _stepModeUnavailableLabel.Visible = true;
            _stepModeUnavailableLabel.Text = "(select a specific agent to enable step mode)";
            return;
        }

        if (_capabilityKnown && !_isDiscreteOnly)
        {
            // Runtime confirmed: agent has continuous actions — step mode not supported.
            _stepModeCheck.Disabled = true;
            _stepModeCheck.SetPressedNoSignal(false);
            _stepActionsRow.Visible = false;
            _stepModeUnavailableLabel.Visible = true;
            _stepModeUnavailableLabel.Text = "(agent has continuous actions — step mode unavailable)";
            return;
        }

        // Specific agent selected in Human mode — enable the checkbox.
        _stepModeCheck.Disabled = false;

        if (!_capabilityKnown)
        {
            // Action space not yet known — show hint instead of buttons.
            _stepModeUnavailableLabel.Visible = true;
            _stepModeUnavailableLabel.Text = "(start recording to see action buttons)";
            // Keep whatever action buttons exist (none yet).
        }
        else
        {
            // Capability confirmed discrete-only — hide hint (buttons already populated).
            _stepModeUnavailableLabel.Visible = false;
        }
    }

    private void PopulateStepActionButtons(string[] actionLabels)
    {
        EnsureUiReferences();
        if (_stepActionsRow is null) return;

        // Remove all existing action buttons (keep the label child).
        foreach (var child in _stepActionsRow.GetChildren())
        {
            if (child is Label) continue;
            child.QueueFree();
        }

        for (var i = 0; i < actionLabels.Length; i++)
        {
            var label = actionLabels[i];
            var actionIndex = i;
            var btn = new Button
            {
                Text = label,
                CustomMinimumSize = new Vector2(60f, 26f),
            };
            btn.Pressed += () => EmitSignal(SignalName.StepActionRequested, actionIndex);
            _stepActionsRow.AddChild(btn);
        }
    }

    private void OnModeChanged(long _)
    {
        EnsureUiReferences();
        _isScriptMode = (_modeDropdown?.Selected ?? 0) == 1;
        UpdateModeAvailability();
        UpdateStepModeAvailability();
        PersistState();
    }

    private void OnStepModeToggled(bool enabled)
    {
        EnsureUiReferences();
        if (_stepActionsRow is not null)
            _stepActionsRow.Visible = enabled;
        EmitSignal(SignalName.SetStepModeRequested, enabled);
        PersistState();
    }

    private void OnStartRecordPressed()
    {
        EnsureUiReferences();
        if (string.IsNullOrWhiteSpace(_activeScenePath))
        {
            if (_recordStatusLabel is not null)
                _recordStatusLabel.Text = "No scene open.";
            return;
        }

        var group = SelectedGroupId();
        var name = (_outputNameEdit?.Text ?? "demo").Trim();
        if (string.IsNullOrWhiteSpace(name)) name = "demo";

        EmitSignal(SignalName.StartRecordingRequested, _activeScenePath, group, name, _isScriptMode);
    }

    private void OnStopRecordPressed()
        => EmitSignal(SignalName.StopRecordingRequested);

    /// <summary>Returns the selected AgentId, or "" for "All agents" (index 0).</summary>
    private string SelectedGroupId()
    {
        EnsureUiReferences();
        if (_agentDropdown is null) return string.Empty;
        var idx = _agentDropdown.Selected;
        if (idx <= 0) return string.Empty;
        var listIdx = idx - 1;
        return listIdx < _agentGroupIds.Count ? _agentGroupIds[listIdx] : string.Empty;
    }

    /// <summary>Updates _resolvedNetworkGraphPath and the Train tab label based on dropdown selection.</summary>
    private void RefreshNetworkGraphDisplay()
    {
        EnsureUiReferences();
        if (_agentDropdown is null) return;

        var idx = _agentDropdown.Selected;

        if (idx <= 0)
        {
            // "All agents" — use the graph from the first agent if all share one, otherwise flag it.
            if (_agentNetworkGraphPaths.Count == 0)
            {
                _resolvedNetworkGraphPath = string.Empty;
            }
            else
            {
                var first = _agentNetworkGraphPaths[0];
                var allSame = _agentNetworkGraphPaths.TrueForAll(p => p == first);
                _resolvedNetworkGraphPath = allSame ? first : string.Empty;
            }
        }
        else
        {
            var listIdx = idx - 1;
            _resolvedNetworkGraphPath = listIdx < _agentNetworkGraphPaths.Count
                ? _agentNetworkGraphPaths[listIdx]
                : string.Empty;
        }

        UpdateNetworkGraphLabel();
        UpdateModeAvailability();
    }

    private void UpdateModeAvailability()
    {
        EnsureUiReferences();
        if (_modeDropdown is null)
        {
            return;
        }

        var (supportsHuman, supportsScript) = GetSelectedModeSupport();
        _modeDropdown.Disabled = !supportsHuman && !supportsScript;
        _modeDropdown.SetItemDisabled(0, !supportsHuman);
        _modeDropdown.SetItemDisabled(1, !supportsScript);

        if (supportsHuman && !supportsScript)
        {
            _modeDropdown.Select(0);
        }
        else if (!supportsHuman && supportsScript)
        {
            _modeDropdown.Select(1);
        }
        else if (!supportsHuman && !supportsScript)
        {
            _modeDropdown.Select(0);
        }

        _isScriptMode = _modeDropdown.Selected == 1 && supportsScript;
        _modeDropdown.TooltipText = !supportsHuman && !supportsScript
            ? "Selected agent does not expose a supported Human or Script recording mode."
            : supportsHuman && supportsScript
                ? "Both Human and Script recording modes are available."
                : supportsHuman
                    ? "Only Human recording mode is available for the selected agent."
                    : "Only Script recording mode is available for the selected agent.";

        if (_startRecordBtn is not null && !_stopRecordBtn!.Disabled)
        {
            return;
        }

        var hasScene = !string.IsNullOrWhiteSpace(_activeScenePath);
        if (_startRecordBtn is not null)
        {
            _startRecordBtn.Disabled = !hasScene || (!supportsHuman && !supportsScript);
            _startRecordBtn.TooltipText = !hasScene
                ? "Open a scene to begin."
                : !supportsHuman && !supportsScript
                    ? "Selected agent does not implement a supported recording control mode."
                    : "Start recording the selected agent(s).";
        }
    }

    private (bool supportsHuman, bool supportsScript) GetSelectedModeSupport()
    {
        EnsureUiReferences();
        if (_agentDropdown is null || _agentSupportsHuman.Count == 0)
        {
            return (false, false);
        }

        var idx = _agentDropdown.Selected;
        if (idx <= 0)
        {
            return (_agentSupportsHuman.TrueForAll(supports => supports),
                _agentSupportsScript.TrueForAll(supports => supports));
        }

        var listIdx = idx - 1;
        var supportsHuman = listIdx < _agentSupportsHuman.Count && _agentSupportsHuman[listIdx];
        var supportsScript = listIdx < _agentSupportsScript.Count && _agentSupportsScript[listIdx];
        return (supportsHuman, supportsScript);
    }

    /// <summary>
    /// Auto-populates the warm-start checkpoint field from the selected agent's
    /// InferenceModelPath when no manual override has been typed yet.
    /// Only fills if the user hasn't already entered something.
    /// </summary>
    private void RefreshWarmStartPath()
    {
        EnsureUiReferences();
        if (_checkpointPathEdit is null || _agentDropdown is null) return;

        var current = _checkpointPathEdit.Text ?? string.Empty;

        // Determine the best candidate path, in priority order:
        //   1. Agent's InferenceModelPath (specific agent selected)
        //   2. Latest BC checkpoint produced this session
        //   3. Latest bc_*.rlcheckpoint on disk

        var idx = _agentDropdown.Selected;
        var inferencePath = string.Empty;
        if (idx > 0)
        {
            var listIdx = idx - 1;
            inferencePath = listIdx < _agentInferencePaths.Count ? _agentInferencePaths[listIdx] : string.Empty;
        }

        var latestOnDisk = FindLatestBCCheckpointOnDisk();
        var candidate = !string.IsNullOrWhiteSpace(inferencePath)
            ? inferencePath
            : !string.IsNullOrWhiteSpace(_lastBCCheckpointPath)
                ? _lastBCCheckpointPath
                : latestOnDisk;

        // Determine whether current text was auto-filled (safe to overwrite).
        var isAutoFilled = !string.IsNullOrWhiteSpace(current)
            && (_agentInferencePaths.Contains(current)
                || current == _lastBCCheckpointPath
                || current == latestOnDisk);

        if (string.IsNullOrWhiteSpace(current) || isAutoFilled)
            _checkpointPathEdit.Text = candidate;
    }

    private static string FindLatestBCCheckpointOnDisk()
    {
        var dir = ProjectSettings.GlobalizePath("res://RL-Agent-Demos/trained");
        if (!System.IO.Directory.Exists(dir)) return string.Empty;

        var latest = string.Empty;
        var latestTime = System.DateTime.MinValue;
        foreach (var file in System.IO.Directory.GetFiles(dir, "bc_*.rlcheckpoint"))
        {
            var t = System.IO.File.GetLastWriteTime(file);
            if (t > latestTime)
            {
                latestTime = t;
                latest = file;
            }
        }
        return latest;
    }

    private void UpdateNetworkGraphLabel()
    {
        EnsureUiReferences();
        if (_networkGraphLabel is null) return;

        // If the user has typed an override, keep showing that.
        var overrideText = _networkOverrideEdit?.Text ?? string.Empty;
        if (!string.IsNullOrWhiteSpace(overrideText))
        {
            _networkGraphLabel.Text = Path.GetFileName(overrideText);
            _networkGraphLabel.TooltipText = overrideText;
            return;
        }

        if (string.IsNullOrWhiteSpace(_resolvedNetworkGraphPath))
        {
            _networkGraphLabel.Text = "— select a specific agent —";
            _networkGraphLabel.TooltipText = string.Empty;
        }
        else
        {
            _networkGraphLabel.Text = Path.GetFileName(_resolvedNetworkGraphPath);
            _networkGraphLabel.TooltipText = _resolvedNetworkGraphPath;
        }
    }

    // ── Train tab ─────────────────────────────────────────────────────────────

    private Control BuildTrainTab()
    {
        var root = new MarginContainer { Name = "Train" };
        SetMargins(root, 8);

        var columns = new HBoxContainer();
        columns.AddThemeConstantOverride("separation", 8);
        columns.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        columns.SizeFlagsVertical = SizeFlags.ExpandFill;
        root.AddChild(columns);

        // ── Left column: Setup (scrollable) ──────────────────────────────────
        var leftScroll = new ScrollContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical   = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        columns.AddChild(leftScroll);

        var left = new VBoxContainer();
        left.AddThemeConstantOverride("separation", 6);
        left.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        leftScroll.AddChild(left);

        // Dataset row.
        var datasetRow = new HBoxContainer();
        datasetRow.AddThemeConstantOverride("separation", 6);
        left.AddChild(datasetRow);
        datasetRow.AddChild(MakeAlignedLabel("Dataset"));

        _datasetDropdown = new OptionButton
        {
            Name = "DatasetDropdown",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _datasetDropdown.AddItem("— no datasets found —");
        _datasetDropdown.Disabled = true;
        _datasetDropdown.ItemSelected += idx =>
        {
            if (idx <= 0)
            {
                _pendingSelectedDatasetPath = string.Empty;
                ClearDatasetInfo();
                RefreshDaggerOutputName();
                PersistState();
                return;
            }

            if ((idx - 1) >= 0 && (idx - 1) < _datasetPaths.Length)
            {
                _pendingSelectedDatasetPath = _datasetPaths[(int)idx - 1];
                PopulateDatasetInfo(_pendingSelectedDatasetPath);
                RefreshDaggerOutputName();
                PersistState();
            }
        };
        datasetRow.AddChild(_datasetDropdown);

        var refreshBtn = new Button { Text = "Refresh", TooltipText = "Rescan res://RL-Agent-Demos/" };
        refreshBtn.Pressed += OnRefreshDatasets;
        datasetRow.AddChild(refreshBtn);

        var algoRow = new HBoxContainer();
        algoRow.AddThemeConstantOverride("separation", 6);
        left.AddChild(algoRow);
        algoRow.AddChild(MakeAlignedLabel("Algorithm"));

        _trainAlgorithmDropdown = new OptionButton
        {
            Name = "TrainAlgorithmDropdown",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _trainAlgorithmDropdown.AddItem("Behavior Cloning (BC)");
        _trainAlgorithmDropdown.AddItem("Manual DAgger");
        _trainAlgorithmDropdown.AddItem("Auto DAgger");
        _trainAlgorithmDropdown.ItemSelected += _ => UpdateTrainAlgorithmUi();
        algoRow.AddChild(_trainAlgorithmDropdown);

        // Network graph row — auto-detected from agent selection; read-only label + optional override.
        var networkRow = new HBoxContainer();
        networkRow.AddThemeConstantOverride("separation", 6);
        left.AddChild(networkRow);
        networkRow.AddChild(MakeAlignedLabel("Network Graph"));

        _networkGraphLabel = new Label
        {
            Name = "NetworkGraphLabel",
            Text = "— select a specific agent —",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            ClipText = true,
            VerticalAlignment = VerticalAlignment.Center,
        };
        networkRow.AddChild(_networkGraphLabel);

        var overrideCheck = new CheckBox { Text = "Override" };
        overrideCheck.Toggled += on =>
        {
            if (_networkOverrideRow is not null)
                _networkOverrideRow.Visible = on;
            if (!on && _networkOverrideEdit is not null)
                _networkOverrideEdit.Text = string.Empty;
            UpdateNetworkGraphLabel();
            PersistState();
        };
        networkRow.AddChild(overrideCheck);

        // Override row (hidden by default).
        _networkOverrideRow = MakeFieldRow("Override Path", out _networkOverrideEdit,
            "res://.../MyNetwork.tres", out var browseNetBtn);
        _networkOverrideRow.Name = "NetworkOverrideRow";
        _networkOverrideEdit.Name = "NetworkOverrideEdit";
        browseNetBtn.Pressed += OnBrowseNetworkPressed;
        _networkOverrideRow.Visible = false;
        _networkOverrideEdit.TextChanged += _ =>
        {
            UpdateNetworkGraphLabel();
            PersistState();
        };
        left.AddChild(_networkOverrideRow);

        _daggerSetupSection = new VBoxContainer
        {
            Name = "DAggerSetupSection",
        };
        _daggerSetupSection.AddThemeConstantOverride("separation", 6);
        left.AddChild(_daggerSetupSection);

        _warmStartRow = MakeFieldRow("Checkpoint", out _checkpointPathEdit,
            "res://RL-Agent-Training/runs/.../checkpoint.rlcheckpoint", out var browseCkBtn);
        _warmStartRow.Name = "WarmStartRow";
        _checkpointPathEdit.Name = "CheckpointPathEdit";
        browseCkBtn.Pressed += OnBrowseCheckpointPressed;
        _checkpointPathEdit.TextChanged += _ => PersistState();
        _daggerSetupSection.AddChild(_warmStartRow);

        _warmStartCheck = new CheckBox
        {
            Name = "WarmStartCheck",
            Text = "Use this checkpoint for RL Setup warm-start",
            TooltipText = "If enabled, RL Setup -> Start Training will resume from the checkpoint path shown above.",
        };
        _warmStartCheck.Toggled += _ => PersistState();
        _daggerSetupSection.AddChild(_warmStartCheck);

        var daggerOutRow = MakeFieldRow("Output Name", out _daggerOutputNameEdit, "dagger", out var noDaggerBrowse);
        _daggerOutputNameEdit.Name = "DAggerOutputNameEdit";
        _daggerOutputNameEdit.Text = "dagger";
        _daggerOutputNameEdit.TextChanged += _ => PersistState();
        noDaggerBrowse.Visible = false;
        _daggerSetupSection.AddChild(daggerOutRow);

        var daggerFramesRow = new HBoxContainer();
        daggerFramesRow.AddThemeConstantOverride("separation", 6);
        _daggerSetupSection.AddChild(daggerFramesRow);
        daggerFramesRow.AddChild(MakeAlignedLabel("Add Frames"));
        _daggerFramesSpin = new SpinBox
        {
            Name = "DAggerFramesSpin",
            Value = 2048,
            MinValue = 1,
            MaxValue = 1000000,
            Step = 64,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _daggerFramesSpin.ValueChanged += _ => PersistState();
        daggerFramesRow.AddChild(_daggerFramesSpin);

        var daggerBetaRow = new HBoxContainer();
        daggerBetaRow.AddThemeConstantOverride("separation", 6);
        _daggerSetupSection.AddChild(daggerBetaRow);
        daggerBetaRow.AddChild(MakeAlignedLabel("Mixing Beta"));
        _daggerBetaSpin = new SpinBox
        {
            Name = "DAggerBetaSpin",
            Value = 0.5,
            MinValue = 0.0,
            MaxValue = 1.0,
            Step = 0.05,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            TooltipText = "Beta decay factor for DAgger mixing.\n" +
                          "In round i, the expert drives ≈ beta^i of steps; the learner drives the rest.\n" +
                          "1.0 = always expert (like BC), 0.0 = always learner.",
        };
        _daggerBetaSpin.ValueChanged += _ => PersistState();
        daggerBetaRow.AddChild(_daggerBetaSpin);

        _autoDaggerRoundsRow = new HBoxContainer
        {
            Name = "AutoDAggerRoundsRow",
        };
        _autoDaggerRoundsRow.AddThemeConstantOverride("separation", 6);
        _daggerSetupSection.AddChild(_autoDaggerRoundsRow);
        _autoDaggerRoundsRow.AddChild(MakeAlignedLabel("Rounds"));
        _autoDaggerRoundsSpin = new SpinBox
        {
            Name = "AutoDAggerRoundsSpin",
            Value = 3,
            MinValue = 1,
            MaxValue = 100,
            Step = 1,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            TooltipText = "Number of DAgger aggregation rounds to run automatically.",
        };
        _autoDaggerRoundsSpin.ValueChanged += _ => PersistState();
        _autoDaggerRoundsRow.AddChild(_autoDaggerRoundsSpin);

        // ── Divider ───────────────────────────────────────────────────────────
        columns.AddChild(new VSeparator { SizeFlagsVertical = SizeFlags.ExpandFill });

        // ── Right column: Run (scrollable) ───────────────────────────────────
        var rightScroll = new ScrollContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical   = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        columns.AddChild(rightScroll);

        var right = new VBoxContainer();
        right.AddThemeConstantOverride("separation", 6);
        right.CustomMinimumSize = new Vector2(200f, 0f);
        right.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        rightScroll.AddChild(right);

        _bcRunSection = new VBoxContainer
        {
            Name = "BCRunSection",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _bcRunSection.AddThemeConstantOverride("separation", 6);
        right.AddChild(_bcRunSection);

        // Hyperparams row.
        _bcHyperparamsRow = new HBoxContainer
        {
            Name = "BCHyperparamsRow",
        };
        _bcHyperparamsRow.AddThemeConstantOverride("separation", 6);
        _bcRunSection.AddChild(_bcHyperparamsRow);

        _bcHyperparamsRow.AddChild(MakeLabel("Epochs:"));
        _epochsSpin = new SpinBox
        {
            Name = "EpochsSpin",
            Value = 20,
            MinValue = 1,
            MaxValue = 1000,
            Step = 1,
        };
        _epochsSpin.CustomMinimumSize = new Vector2(80f, 0f);
        _epochsSpin.ValueChanged += _ => PersistState();
        _bcHyperparamsRow.AddChild(_epochsSpin);

        _bcHyperparamsRow.AddChild(MakeLabel("LR:"));
        _lrEdit = new LineEdit
        {
            Name = "LearningRateEdit",
            Text = "0.0003",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _lrEdit.TextChanged += _ => PersistState();
        _bcHyperparamsRow.AddChild(_lrEdit);

        // Train button (full width of right column).
        _trainBtn = new Button
        {
            Name = "TrainButton",
            Text = "Train",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0f, 28f),
        };
        _trainBtn.Pressed += OnTrainPressed;
        _bcRunSection.AddChild(_trainBtn);

        _bcRunDivider = new HSeparator
        {
            Name = "BCRunDivider",
        };
        _bcRunSection.AddChild(_bcRunDivider);

        _cancelTrainBtn = new Button
        {
            Name = "CancelTrainButton",
            Text = "Cancel",
            Disabled = true,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0f, 28f),
        };
        _cancelTrainBtn.Pressed += () => EmitSignal(SignalName.CancelBCTrainingRequested);
        _bcRunSection.AddChild(_cancelTrainBtn);

        _daggerRunSection = new VBoxContainer
        {
            Name = "DAggerRunSection",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _daggerRunSection.AddThemeConstantOverride("separation", 6);
        right.AddChild(_daggerRunSection);

        var daggerHelp = new Label
        {
            Text = "Manual DAgger keeps both actions available here: run one aggregation round, then retrain BC on the resulting dataset.",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            Modulate = new Color(0.85f, 0.92f, 1f, 0.9f),
        };
        _daggerRunSection.AddChild(daggerHelp);

        _daggerBtn = new Button
        {
            Name = "DAggerButton",
            Text = "Run DAgger Round",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0f, 28f),
            TooltipText = "Learner acts from the selected checkpoint; scripted expert labels the visited states.",
        };
        _daggerBtn.Pressed += OnDaggerPressed;
        _daggerRunSection.AddChild(_daggerBtn);

        _autoDaggerRunSection = new VBoxContainer
        {
            Name = "AutoDAggerRunSection",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _autoDaggerRunSection.AddThemeConstantOverride("separation", 6);
        right.AddChild(_autoDaggerRunSection);

        var autoHelp = new Label
        {
            Text = "Runs BC, then repeats DAgger aggregation and BC retraining for the requested number of rounds.",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            Modulate = new Color(0.85f, 0.92f, 1f, 0.9f),
        };
        _autoDaggerRunSection.AddChild(autoHelp);

        _autoDaggerBtn = new Button
        {
            Name = "AutoDAggerButton",
            Text = "Run Auto DAgger",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0f, 28f),
            TooltipText = "If a checkpoint is set, the loop starts from it. Otherwise it seeds itself with BC on the selected dataset.",
        };
        _autoDaggerBtn.Pressed += OnAutoDaggerPressed;
        _autoDaggerRunSection.AddChild(_autoDaggerBtn);

        // Progress bar.
        _progressBar = new ProgressBar
        {
            Name = "TrainingProgressBar",
            MinValue = 0, MaxValue = 100, Value = 0,
            Visible = false,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        right.AddChild(_progressBar);

        // Status label (shown during and after training).
        _trainStatusLabel = new Label
        {
            Name = "TrainStatusLabel",
            Text = "Ready.",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        right.AddChild(_trainStatusLabel);

        _warmStartSourceLabel = new Label
        {
            Name = "WarmStartSourceLabel",
            Visible = false,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            Modulate = new Color(0.8f, 0.9f, 1f, 0.95f),
        };
        right.AddChild(_warmStartSourceLabel);

        // Summary label — shown after training completes with final stats + saved path.
        _trainSummaryLabel = new Label
        {
            Name = "TrainSummaryLabel",
            Visible = false,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            Modulate = new Color(0.7f, 1f, 0.7f, 1f),
        };
        right.AddChild(_trainSummaryLabel);

        // Export button — shown after training so the user can convert the checkpoint to .rlmodel.
        _exportModelBtn = new Button
        {
            Name = "ExportModelButton",
            Text = "Export to .rlmodel",
            Visible = false,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0f, 28f),
            TooltipText = "Converts the trained checkpoint to a .rlmodel file that can be assigned\n" +
                          "to an agent's Inference Model Path for in-game inference.",
        };
        _exportModelBtn.Pressed += OnExportModelPressed;
        right.AddChild(_exportModelBtn);

        UpdateTrainAlgorithmUi();

        return root;
    }

    private void OnBrowseCheckpointPressed()
    {
        EnsureUiReferences();
        if (_checkpointPickerDialog is null)
        {
            _checkpointPickerDialog = new EditorFileDialog
            {
                FileMode = EditorFileDialog.FileModeEnum.OpenFile,
                Access = EditorFileDialog.AccessEnum.Resources,
                Title = "Select warm-start checkpoint",
            };
            _checkpointPickerDialog.AddFilter("*.rlcheckpoint,*.json", "RL Checkpoints");
            _checkpointPickerDialog.FileSelected += path =>
            {
                if (_checkpointPathEdit is not null)
                    _checkpointPathEdit.Text = path;
            };
            AddChild(_checkpointPickerDialog);
        }
        _checkpointPickerDialog.PopupCentered(new Vector2I(800, 500));
    }

    private void OnBrowseNetworkPressed()
    {
        EnsureUiReferences();
        if (_networkPickerDialog is null)
        {
            _networkPickerDialog = new EditorFileDialog
            {
                FileMode = EditorFileDialog.FileModeEnum.OpenFile,
                Access = EditorFileDialog.AccessEnum.Resources,
                Title = "Select network graph resource",
            };
            _networkPickerDialog.AddFilter("*.tres,*.res", "Network Graph Resources");
            _networkPickerDialog.FileSelected += path =>
            {
                if (_networkOverrideEdit is not null)
                    _networkOverrideEdit.Text = path;
            };
            AddChild(_networkPickerDialog);
        }
        _networkPickerDialog.PopupCentered(new Vector2I(800, 500));
    }

    private void OnRefreshDatasets()
        => EmitSignal(SignalName.RefreshDatasetsRequested);

    private TrainAlgorithmMode GetSelectedTrainAlgorithm()
    {
        EnsureUiReferences();
        return (_trainAlgorithmDropdown?.Selected ?? 0) switch
        {
            1 => TrainAlgorithmMode.DAgger,
            2 => TrainAlgorithmMode.AutoDAgger,
            _ => TrainAlgorithmMode.BehaviorCloning,
        };
    }

    private void UpdateTrainAlgorithmUi()
    {
        EnsureUiReferences();

        var algorithm = GetSelectedTrainAlgorithm();
        var isBc = algorithm == TrainAlgorithmMode.BehaviorCloning;
        var isDagger = algorithm == TrainAlgorithmMode.DAgger;
        var isAutoDagger = algorithm == TrainAlgorithmMode.AutoDAgger;
        var isDaggerLike = isDagger || isAutoDagger;

        if (_daggerSetupSection is not null)
            _daggerSetupSection.Visible = isDaggerLike;
        if (_bcRunSection is not null)
            _bcRunSection.Visible = isBc || isDagger || isAutoDagger;
        if (_daggerRunSection is not null)
            _daggerRunSection.Visible = isDagger;
        if (_autoDaggerRunSection is not null)
            _autoDaggerRunSection.Visible = isAutoDagger;
        if (_bcHyperparamsRow is not null)
            _bcHyperparamsRow.Visible = isBc || isDagger || isAutoDagger;
        if (_trainBtn is not null)
            _trainBtn.Visible = isBc || isDagger;
        if (_bcRunDivider is not null)
            _bcRunDivider.Visible = isBc || isDagger;
        if (_warmStartCheck is not null)
            _warmStartCheck.Visible = isDagger;
        if (_autoDaggerRoundsRow is not null)
            _autoDaggerRoundsRow.Visible = isAutoDagger;
        if (isDaggerLike)
            RefreshWarmStartPath();
        PersistState();
    }

    private void ApplyPendingAgentSelection()
    {
        EnsureUiReferences();
        if (_agentDropdown is null || string.IsNullOrWhiteSpace(_pendingSelectedAgentGroupId))
            return;

        var idx = _agentGroupIds.IndexOf(_pendingSelectedAgentGroupId);
        if (idx >= 0)
        {
            _agentDropdown.Selected = idx + 1;
            RefreshNetworkGraphDisplay();
            RefreshWarmStartPath();
            UpdateStepModeAvailability();
        }
    }

    private void RestorePersistedState()
    {
        EnsureUiReferences();
        if (!Godot.FileAccess.FileExists(PersistedStatePath))
            return;

        using var file = Godot.FileAccess.Open(PersistedStatePath, Godot.FileAccess.ModeFlags.Read);
        if (file is null)
            return;

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary)
            return;

        var d = parsed.AsGodotDictionary();
        _isRestoringState = true;
        try
        {
            if (_tabs is not null && d.ContainsKey("active_tab"))
                _tabs.CurrentTab = Mathf.Max(0, d["active_tab"].AsInt32());

            _pendingSelectedAgentGroupId = ReadString(d, "agent_group_id");
            _pendingSelectedDatasetPath = ReadString(d, "dataset_path");

            if (_modeDropdown is not null && d.ContainsKey("record_mode"))
            {
                var mode = Mathf.Clamp(d["record_mode"].AsInt32(), 0, Math.Max(0, _modeDropdown.ItemCount - 1));
                _modeDropdown.Select(mode);
                _isScriptMode = mode == 1;
            }

            if (_outputNameEdit is not null)
                _outputNameEdit.Text = ReadStringOrDefault(d, "record_output_name", _outputNameEdit.Text);

            if (_trainAlgorithmDropdown is not null && d.ContainsKey("train_algorithm"))
            {
                var algorithm = Mathf.Clamp(d["train_algorithm"].AsInt32(), 0, Math.Max(0, _trainAlgorithmDropdown.ItemCount - 1));
                _trainAlgorithmDropdown.Select(algorithm);
            }

            if (_checkpointPathEdit is not null)
                _checkpointPathEdit.Text = ReadStringOrDefault(d, "checkpoint_path", _checkpointPathEdit.Text);

            if (_warmStartCheck is not null && d.ContainsKey("warm_start_enabled"))
                _warmStartCheck.ButtonPressed = d["warm_start_enabled"].AsBool();

            if (_networkOverrideEdit is not null)
                _networkOverrideEdit.Text = ReadStringOrDefault(d, "network_override_path", string.Empty);

            if (_networkOverrideRow is not null)
                _networkOverrideRow.Visible = !string.IsNullOrWhiteSpace(_networkOverrideEdit?.Text);

            if (_epochsSpin is not null && d.ContainsKey("bc_epochs"))
                _epochsSpin.Value = d["bc_epochs"].AsDouble();

            if (_lrEdit is not null)
                _lrEdit.Text = ReadStringOrDefault(d, "bc_learning_rate", _lrEdit.Text);

            if (_daggerOutputNameEdit is not null)
                _daggerOutputNameEdit.Text = ReadStringOrDefault(d, "dagger_output_name", _daggerOutputNameEdit.Text);

            if (_daggerFramesSpin is not null && d.ContainsKey("dagger_additional_frames"))
                _daggerFramesSpin.Value = d["dagger_additional_frames"].AsDouble();

            if (_daggerBetaSpin is not null && d.ContainsKey("dagger_mixing_beta"))
                _daggerBetaSpin.Value = d["dagger_mixing_beta"].AsDouble();

            if (_autoDaggerRoundsSpin is not null && d.ContainsKey("auto_dagger_rounds"))
                _autoDaggerRoundsSpin.Value = d["auto_dagger_rounds"].AsDouble();

            if (_previewCountSpin is not null && d.ContainsKey("preview_count"))
                _previewCountSpin.Value = d["preview_count"].AsDouble();

            if (_previewFromEndCheck is not null && d.ContainsKey("preview_from_end"))
                _previewFromEndCheck.ButtonPressed = d["preview_from_end"].AsBool();
        }
        finally
        {
            _isRestoringState = false;
        }

        UpdateModeAvailability();
        UpdateStepModeAvailability();
        UpdateNetworkGraphLabel();
        UpdateTrainAlgorithmUi();
    }

    private void PersistState()
    {
        if (_isRestoringState || !_statePersistenceReady)
            return;

        EnsureUiReferences();

        var dirError = DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath("user://rl-agent-plugin"));
        if (dirError != Error.Ok)
            return;

        using var file = Godot.FileAccess.Open(PersistedStatePath, Godot.FileAccess.ModeFlags.Write);
        if (file is null)
            return;

        var payload = new Godot.Collections.Dictionary
        {
            { "active_tab", _tabs?.CurrentTab ?? 0 },
            { "agent_group_id", SelectedGroupId() },
            { "record_mode", _modeDropdown?.Selected ?? 0 },
            { "record_output_name", _outputNameEdit?.Text ?? string.Empty },
            { "train_algorithm", _trainAlgorithmDropdown?.Selected ?? 0 },
            { "dataset_path", GetPersistedDatasetPath() },
            { "checkpoint_path", _checkpointPathEdit?.Text ?? string.Empty },
            { "warm_start_enabled", _warmStartCheck?.ButtonPressed ?? false },
            { "network_override_path", _networkOverrideEdit?.Text ?? string.Empty },
            { "bc_epochs", _epochsSpin?.Value ?? 20.0 },
            { "bc_learning_rate", _lrEdit?.Text ?? "0.0003" },
            { "dagger_output_name", _daggerOutputNameEdit?.Text ?? string.Empty },
            { "dagger_additional_frames", _daggerFramesSpin?.Value ?? 2048.0 },
            { "dagger_mixing_beta", _daggerBetaSpin?.Value ?? 0.5 },
            { "auto_dagger_rounds", _autoDaggerRoundsSpin?.Value ?? 3.0 },
            { "preview_count", _previewCountSpin?.Value ?? 50.0 },
            { "preview_from_end", _previewFromEndCheck?.ButtonPressed ?? false },
        };

        file.StoreString(Json.Stringify(payload, "\t"));
    }

    private static string ReadString(Godot.Collections.Dictionary d, string key)
        => d.ContainsKey(key) ? d[key].ToString() : string.Empty;

    private static string ReadStringOrDefault(Godot.Collections.Dictionary d, string key, string fallback)
        => d.ContainsKey(key) ? d[key].ToString() : fallback;

    private void OnExportModelPressed()
    {
        EnsureUiReferences();
        var checkpointPath = GetWarmStartCheckpointPath();
        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            if (_trainStatusLabel is not null)
                _trainStatusLabel.Text = "No checkpoint selected — browse to a .rlcheckpoint first.";
            return;
        }

        _pendingExportCheckpointPath = checkpointPath;

        if (_exportDialog is null || !IsInstanceValid(_exportDialog))
        {
            _exportDialog = new FileDialog
            {
                FileMode = FileDialog.FileModeEnum.OpenDir,
                Access = FileDialog.AccessEnum.Filesystem,
                Title = "Choose export folder",
            };
            _exportDialog.DirSelected += OnExportDirSelected;
            AddChild(_exportDialog);
        }

        // Pre-navigate to the folder containing the checkpoint.
        var checkpointDir = System.IO.Path.GetDirectoryName(checkpointPath);
        if (!string.IsNullOrEmpty(checkpointDir))
            _exportDialog.CurrentDir = checkpointDir;

        _exportDialog.PopupCentered(new Vector2I(700, 500));
    }

    private void OnExportDirSelected(string dir)
    {
        if (string.IsNullOrWhiteSpace(_pendingExportCheckpointPath)) return;

        var exportName = System.IO.Path.GetFileNameWithoutExtension(_pendingExportCheckpointPath);
        var destPath = System.IO.Path.Combine(dir, $"{exportName}.rlmodel");
        EmitSignal(SignalName.ExportModelRequested, _pendingExportCheckpointPath, destPath);
        _pendingExportCheckpointPath = string.Empty;
    }

    private void OnTrainPressed()
    {
        EnsureUiReferences();
        var datasetPath = GetSelectedDatasetPath();
        if (string.IsNullOrWhiteSpace(datasetPath))
        {
            if (_trainStatusLabel is not null)
                _trainStatusLabel.Text = "Select a dataset first.";
            return;
        }

        // Network graph: use override if provided, otherwise the auto-detected path.
        // Always pass the graph even when warm-starting, so the network is rebuilt with the
        // correct architecture before the checkpoint weights are loaded into it.
        var overridePath = _networkOverrideEdit?.Text ?? string.Empty;
        var networkPath = string.IsNullOrWhiteSpace(overridePath) ? _resolvedNetworkGraphPath : overridePath;

        var epochs = (int)(_epochsSpin?.Value ?? 20);
        var lr = float.TryParse(_lrEdit?.Text, out var parsed) ? parsed : 3e-4f;

        EmitSignal(SignalName.StartBCTrainingRequested,
            datasetPath, string.Empty, networkPath, epochs, lr);
    }

    private void OnDaggerPressed()
    {
        EnsureUiReferences();

        if (string.IsNullOrWhiteSpace(_activeScenePath))
        {
            SetDaggerStatus("Open a scene to run DAgger.", false);
            return;
        }

        var datasetPath = GetSelectedDatasetPath();
        var groupId = SelectedGroupId();
        if (string.IsNullOrWhiteSpace(groupId))
        {
            SetDaggerStatus("Select a specific agent group for DAgger.", false);
            return;
        }

        var (_, supportsScript) = GetSelectedModeSupport();
        if (!supportsScript)
        {
            SetDaggerStatus("Selected agent does not implement Script mode, so it cannot act as a DAgger expert.", false);
            return;
        }

        var checkpointPath = GetWarmStartCheckpointPath();
        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            SetDaggerStatus("Select a learner checkpoint path first.", false);
            return;
        }

        var overridePath = _networkOverrideEdit?.Text ?? string.Empty;
        var networkPath = string.IsNullOrWhiteSpace(overridePath) ? _resolvedNetworkGraphPath : overridePath;
        if (string.IsNullOrWhiteSpace(networkPath))
        {
            SetDaggerStatus("Select a network graph resource first.", false);
            return;
        }

        var outputName = (_daggerOutputNameEdit?.Text ?? "dagger").Trim();
        if (string.IsNullOrWhiteSpace(outputName))
            outputName = "dagger";

        var additionalFrames = (int)(_daggerFramesSpin?.Value ?? 2048);
        var mixingBeta = (float)(_daggerBetaSpin?.Value ?? 0.5);
        SetDaggerStatus($"Launching DAgger… (target {additionalFrames} frames, β={mixingBeta:F2})", true);
        if (_progressBar is not null)
            _progressBar.Value = 0f;

        EmitSignal(SignalName.StartDAggerRequested,
            _activeScenePath,
            groupId,
            datasetPath,
            checkpointPath,
            networkPath,
            outputName,
            additionalFrames,
            mixingBeta);
    }

    private void OnAutoDaggerPressed()
    {
        EnsureUiReferences();

        if (string.IsNullOrWhiteSpace(_activeScenePath))
        {
            SetDaggerStatus("Open a scene to run Auto DAgger.", false);
            return;
        }

        var datasetPath = GetSelectedDatasetPath();
        var groupId = SelectedGroupId();
        if (string.IsNullOrWhiteSpace(groupId))
        {
            SetDaggerStatus("Select a specific agent group for Auto DAgger.", false);
            return;
        }

        var (_, supportsScript) = GetSelectedModeSupport();
        if (!supportsScript)
        {
            SetDaggerStatus("Selected agent does not implement Script mode, so it cannot act as a DAgger expert.", false);
            return;
        }

        var overridePath = _networkOverrideEdit?.Text ?? string.Empty;
        var networkPath = string.IsNullOrWhiteSpace(overridePath) ? _resolvedNetworkGraphPath : overridePath;
        if (string.IsNullOrWhiteSpace(networkPath))
        {
            SetDaggerStatus("Select a network graph resource first.", false);
            return;
        }

        var outputName = (_daggerOutputNameEdit?.Text ?? "dagger").Trim();
        if (string.IsNullOrWhiteSpace(outputName))
            outputName = "dagger";

        var checkpointPath = GetWarmStartCheckpointPath();
        if (string.IsNullOrWhiteSpace(datasetPath) && string.IsNullOrWhiteSpace(checkpointPath))
        {
            SetDaggerStatus("Select a seed dataset or a learner checkpoint first.", false);
            return;
        }

        var additionalFrames = (int)(_daggerFramesSpin?.Value ?? 2048);
        var rounds = Math.Max(1, (int)(_autoDaggerRoundsSpin?.Value ?? 3));
        var epochs = Math.Max(1, (int)(_epochsSpin?.Value ?? 20));
        var learningRate = float.TryParse(_lrEdit?.Text, out var parsed) ? parsed : 3e-4f;
        var mixingBeta = (float)(_daggerBetaSpin?.Value ?? 0.5);
        var configJson = Json.Stringify(new Godot.Collections.Dictionary
        {
            { "AdditionalFrames", additionalFrames },
            { "Rounds", rounds },
            { "Epochs", epochs },
            { "LearningRate", learningRate },
            { "MixingBeta", mixingBeta },
        });

        SetDaggerStatus($"Launching Auto DAgger… ({rounds} round(s))", true);
        if (_progressBar is not null)
            _progressBar.Value = 0f;

        EmitSignal(SignalName.StartAutoDAggerRequested,
            _activeScenePath,
            groupId,
            datasetPath,
            checkpointPath,
            networkPath,
            outputName,
            configJson);
    }

    private void RefreshDaggerOutputName()
    {
        EnsureUiReferences();
        if (_daggerOutputNameEdit is null) return;

        var current = _daggerOutputNameEdit.Text ?? string.Empty;
        var autoCurrent = string.IsNullOrWhiteSpace(current) || current == "dagger" || current.EndsWith("_dagger", StringComparison.Ordinal);
        if (!autoCurrent) return;

        var datasetPath = GetSelectedDatasetPath();
        if (string.IsNullOrWhiteSpace(datasetPath))
        {
            _daggerOutputNameEdit.Text = "dagger";
            return;
        }

        _daggerOutputNameEdit.Text = $"{Path.GetFileNameWithoutExtension(datasetPath)}_dagger";
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static HBoxContainer MakeFieldRow(
        string labelText,
        out LineEdit lineEdit,
        string placeholder,
        out Button browseBtn)
    {
        var row = new HBoxContainer();
        row.AddThemeConstantOverride("separation", 6);
        row.AddChild(MakeAlignedLabel(labelText));

        var edit = new LineEdit
        {
            PlaceholderText = placeholder,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        row.AddChild(edit);
        lineEdit = edit;

        var btn = new Button { Text = "Browse" };
        row.AddChild(btn);
        browseBtn = btn;

        return row;
    }

    private static Label MakeLabel(string text) => new() { Text = text };

    private Control BuildInfoTab()
    {
        const float InfoLabelWidth = 85f;

        var root = new MarginContainer { Name = "Dataset Info" };
        SetMargins(root, 8);

        var columns = new HBoxContainer();
        columns.AddThemeConstantOverride("separation", 8);
        columns.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        columns.SizeFlagsVertical = SizeFlags.ExpandFill;
        root.AddChild(columns);

        // ── Left column: Stats (fixed width, scrollable) ─────────────────────
        var leftScroll = new ScrollContainer
        {
            CustomMinimumSize = new Vector2(220f, 0f),
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        columns.AddChild(leftScroll);

        var left = new VBoxContainer();
        left.AddThemeConstantOverride("separation", 6);
        left.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        left.SizeFlagsVertical = SizeFlags.ShrinkBegin;
        leftScroll.AddChild(left);

        Label InfoVal(string name) => new()
        {
            Name = name,
            Text = "—",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            VerticalAlignment = VerticalAlignment.Center,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };

        HBoxContainer InfoRow(string label, Label val)
        {
            var row = new HBoxContainer();
            row.AddThemeConstantOverride("separation", 6);
            row.AddChild(new Label
            {
                Text = label,
                CustomMinimumSize = new Vector2(InfoLabelWidth, 0f),
                VerticalAlignment = VerticalAlignment.Center,
            });
            row.AddChild(val);
            return row;
        }

        _infoFileLabel      = InfoVal("InfoFileLabel");
        _infoValidLabel     = InfoVal("InfoValidLabel");
        _infoFramesLabel    = InfoVal("InfoFramesLabel");
        _infoEpisodesLabel  = InfoVal("InfoEpisodesLabel");
        _infoAvgLenLabel    = InfoVal("InfoAvgLenLabel");
        _infoAvgRewardLabel = InfoVal("InfoAvgRewardLabel");
        _infoObsSizeLabel   = InfoVal("InfoObsSizeLabel");
        _infoActionLabel    = InfoVal("InfoActionLabel");

        left.AddChild(InfoRow("File",         _infoFileLabel));
        left.AddChild(InfoRow("Valid",         _infoValidLabel));
        left.AddChild(InfoRow("Frames",        _infoFramesLabel));
        left.AddChild(InfoRow("Episodes",      _infoEpisodesLabel));
        left.AddChild(InfoRow("Avg length",    _infoAvgLenLabel));
        left.AddChild(InfoRow("Avg reward",    _infoAvgRewardLabel));
        left.AddChild(InfoRow("Obs size",      _infoObsSizeLabel));
        left.AddChild(InfoRow("Action space",  _infoActionLabel));

        // ── Divider ───────────────────────────────────────────────────────────
        columns.AddChild(new VSeparator { SizeFlagsVertical = SizeFlags.ExpandFill });

        // ── Right column: Preview (entire column scrolls as one unit) ─────────
        // The ScrollContainer's direct child (right VBoxContainer) always has
        // non-zero static content (controls + header row) from construction time,
        // so the scroll container is properly initialised before rows are added.
        _previewScroll = new ScrollContainer
        {
            Name = "PreviewScroll",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        columns.AddChild(_previewScroll);

        var right = new VBoxContainer { Name = "PreviewContent" };
        right.AddThemeConstantOverride("separation", 4);
        right.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        right.SizeFlagsVertical = SizeFlags.ShrinkBegin;
        _previewContent = right;
        _previewScroll.AddChild(right);

        // Preview controls.
        var previewHeader = new HBoxContainer();
        previewHeader.AddThemeConstantOverride("separation", 8);
        right.AddChild(previewHeader);
        previewHeader.AddChild(new Label { Text = "Preview", VerticalAlignment = VerticalAlignment.Center });

        _previewCountSpin = new SpinBox
        {
            Name = "PreviewCountSpin",
            MinValue = 1, MaxValue = 500, Step = 1, Value = 50,
            CustomMinimumSize = new Vector2(80, 0),
            TooltipText = "Number of frames to display",
        };
        _previewCountSpin.ValueChanged += _ =>
        {
            RebuildDataPreview();
            PersistState();
        };
        previewHeader.AddChild(_previewCountSpin);
        previewHeader.AddChild(new Label { Text = "frames", VerticalAlignment = VerticalAlignment.Center });

        _previewFromEndCheck = new CheckBox { Name = "PreviewFromEndCheck", Text = "from end" };
        _previewFromEndCheck.Toggled += _ =>
        {
            RebuildDataPreview();
            PersistState();
        };
        previewHeader.AddChild(_previewFromEndCheck);

        // Column headers.
        var colNames = new[] { "#", "Agent", "Observations", "Action", "Reward", "Done" };
        var colWidths = new[] { 55f, 45f, 0f, 60f, 65f, 40f };

        var headerRow = new HBoxContainer();
        headerRow.AddThemeConstantOverride("separation", 4);
        right.AddChild(headerRow);
        for (var c = 0; c < colNames.Length; c++)
        {
            var lbl = new Label
            {
                Text = colNames[c],
                VerticalAlignment = VerticalAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Left,
            };
            if (colWidths[c] > 0)
                lbl.CustomMinimumSize = new Vector2(colWidths[c], 0);
            else
            {
                lbl.SizeFlagsHorizontal = SizeFlags.ExpandFill;
                lbl.ClipText = true;
            }
            headerRow.AddChild(lbl);
        }
        right.AddChild(new HSeparator());

        _previewGrid = new VBoxContainer { Name = "PreviewGrid" };
        _previewGrid.AddThemeConstantOverride("separation", 1);
        _previewGrid.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        right.AddChild(_previewGrid);

        return root;
    }

    private void PopulateDatasetInfo(string resPath)
    {
        EnsureUiReferences();
        if (_infoFileLabel is null) return;

        var fileName = Path.GetFileName(resPath);
        _infoFileLabel.Text = fileName;

        var dataset = DemonstrationDataset.Open(resPath);
        if (dataset is null)
        {
            if (_infoValidLabel is not null) _infoValidLabel.Text = "Invalid — could not read file";
            if (_infoFramesLabel is not null)    _infoFramesLabel.Text    = "—";
            if (_infoEpisodesLabel is not null)  _infoEpisodesLabel.Text  = "—";
            if (_infoAvgLenLabel is not null)    _infoAvgLenLabel.Text    = "—";
            if (_infoAvgRewardLabel is not null) _infoAvgRewardLabel.Text = "—";
            if (_infoObsSizeLabel is not null)   _infoObsSizeLabel.Text   = "—";
            if (_infoActionLabel is not null)    _infoActionLabel.Text    = "—";
            return;
        }

        var frames = dataset.Frames;
        int episodes = 0;
        double totalReward = 0.0;
        double episodeReward = 0.0;
        int episodeLen = 0;
        long totalEpLen = 0;

        foreach (var f in frames)
        {
            episodeReward += f.Reward;
            episodeLen++;
            if (f.Done)
            {
                episodes++;
                totalReward += episodeReward;
                totalEpLen += episodeLen;
                episodeReward = 0.0;
                episodeLen = 0;
            }
        }

        // Partial episode at end (no terminal Done).
        if (episodeLen > 0) episodes++;

        var avgLen    = episodes > 0 ? (double)frames.Count / episodes : 0.0;
        var avgReward = episodes > 0 ? totalReward / episodes : 0.0;

        var actionDesc = dataset.DiscreteActionCount > 0 && dataset.ContinuousActionDims > 0
            ? $"Discrete ({dataset.DiscreteActionCount}) + Continuous ({dataset.ContinuousActionDims})"
            : dataset.DiscreteActionCount > 0
                ? $"Discrete ({dataset.DiscreteActionCount} actions)"
                : $"Continuous ({dataset.ContinuousActionDims} dims)";

        var valid = frames.Count > 0 ? "OK" : "Empty — no frames recorded";

        if (_infoValidLabel is not null)    _infoValidLabel.Text    = valid;
        if (_infoFramesLabel is not null)   _infoFramesLabel.Text   = frames.Count.ToString("N0");
        if (_infoEpisodesLabel is not null) _infoEpisodesLabel.Text = episodes.ToString("N0");
        if (_infoAvgLenLabel is not null)   _infoAvgLenLabel.Text   = $"{avgLen:F1} steps";
        if (_infoAvgRewardLabel is not null) _infoAvgRewardLabel.Text = $"{avgReward:F3}";
        if (_infoObsSizeLabel is not null)  _infoObsSizeLabel.Text  = dataset.ObsSize.ToString();
        if (_infoActionLabel is not null)   _infoActionLabel.Text   = actionDesc;

        _previewDataset = dataset;
        RebuildDataPreview();
    }

    private void RebuildDataPreview()
    {
        if (_previewGrid is null || _previewDataset is null) return;

        for (var i = _previewGrid.GetChildCount() - 1; i >= 0; i--)
            _previewGrid.GetChild(i).Free();

        var frames = _previewDataset.Frames;
        if (frames.Count == 0) return;

        var count = Math.Min((int)(_previewCountSpin?.Value ?? 50), frames.Count);
        var fromEnd = _previewFromEndCheck?.ButtonPressed ?? false;
        var startIdx = fromEnd ? frames.Count - count : 0;

        for (var i = 0; i < count; i++)
        {
            var fi = startIdx + i;
            var f = frames[fi];

            var obsStr = f.Obs.Length == 0 ? "—"
                : string.Join("  ", Array.ConvertAll(f.Obs, v => v.ToString("F3")));

            string actionStr;
            if (f.DiscreteAction >= 0 && f.ContinuousActions.Length > 0)
                actionStr = $"d:{f.DiscreteAction} c:[{string.Join(" ", Array.ConvertAll(f.ContinuousActions, v => v.ToString("F2")))}]";
            else if (f.DiscreteAction >= 0)
                actionStr = f.DiscreteAction.ToString();
            else
                actionStr = $"[{string.Join(" ", Array.ConvertAll(f.ContinuousActions, v => v.ToString("F2")))}]";

            var row = new HBoxContainer();
            row.AddThemeConstantOverride("separation", 4);
            row.SizeFlagsHorizontal = SizeFlags.ExpandFill;
            row.MouseFilter = MouseFilterEnum.Stop;

            void AddFixedCell(string text, float minW)
            {
                row.AddChild(new Label
                {
                    Text = text,
                    VerticalAlignment = VerticalAlignment.Top,
                    HorizontalAlignment = HorizontalAlignment.Left,
                    CustomMinimumSize = new Vector2(minW, 0),
                });
            }

            // Obs label — starts clipped; click the row to expand/collapse.
            var obsLabel = new Label
            {
                Text = obsStr,
                VerticalAlignment = VerticalAlignment.Top,
                HorizontalAlignment = HorizontalAlignment.Left,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                ClipText = true,
                TooltipText = obsStr,
            };

            row.GuiInput += (InputEvent ev) =>
            {
                if (ev is InputEventMouseButton { ButtonIndex: MouseButton.Left, Pressed: true })
                {
                    var expanded = obsLabel.AutowrapMode != TextServer.AutowrapMode.Off;
                    obsLabel.AutowrapMode = expanded
                        ? TextServer.AutowrapMode.Off
                        : TextServer.AutowrapMode.WordSmart;
                    obsLabel.ClipText = expanded;
                    RefreshPreviewLayout();
                }
            };

            AddFixedCell(fi.ToString(),           55f);
            AddFixedCell(f.AgentSlot.ToString(),  45f);
            row.AddChild(obsLabel);
            AddFixedCell(actionStr,               60f);
            AddFixedCell(f.Reward.ToString("F3"), 65f);
            AddFixedCell(f.Done ? "✓" : "",       40f);

            _previewGrid.AddChild(row);
        }

        // Notify the scroll container that content size has changed.
        RefreshPreviewLayout();
    }

    private void ClearDatasetInfo()
    {
        EnsureUiReferences();
        if (_infoFileLabel is not null)      _infoFileLabel.Text      = "—";
        if (_infoValidLabel is not null)     _infoValidLabel.Text     = "—";
        if (_infoFramesLabel is not null)    _infoFramesLabel.Text    = "—";
        if (_infoEpisodesLabel is not null)  _infoEpisodesLabel.Text  = "—";
        if (_infoAvgLenLabel is not null)    _infoAvgLenLabel.Text    = "—";
        if (_infoAvgRewardLabel is not null) _infoAvgRewardLabel.Text = "—";
        if (_infoObsSizeLabel is not null)   _infoObsSizeLabel.Text   = "—";
        if (_infoActionLabel is not null)    _infoActionLabel.Text    = "—";
        _previewDataset = null;
        if (_previewGrid is not null)
            for (var i = _previewGrid.GetChildCount() - 1; i >= 0; i--)
                _previewGrid.GetChild(i).Free();
        RefreshPreviewLayout();
    }

    private void RefreshPreviewLayout()
    {
        _previewGrid?.UpdateMinimumSize();
        _previewContent?.UpdateMinimumSize();
        _previewScroll?.UpdateMinimumSize();
    }

    private static Label MakeAlignedLabel(string text) => new()
    {
        Text = text,
        CustomMinimumSize = new Vector2(LabelWidth, 0f),
        VerticalAlignment = VerticalAlignment.Center,
    };

    private static void SetMargins(Control ctrl, int px)
    {
        ctrl.AddThemeConstantOverride("margin_left", px);
        ctrl.AddThemeConstantOverride("margin_right", px);
        ctrl.AddThemeConstantOverride("margin_top", px);
        ctrl.AddThemeConstantOverride("margin_bottom", px);
    }
}
