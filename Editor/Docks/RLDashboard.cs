using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Live training dashboard added as a main-screen tab (same row as 2D / 3D / Script).
/// Polls the active run's metrics.jsonl and status.json files every two seconds
/// and updates four charts and a stats bar in real time.
/// </summary>
[Tool]
public partial class RLDashboard : Control
{
    // ── Data model ───────────────────────────────────────────────────────────
    private sealed class Metric
    {
        public float EpisodeReward;
        public int EpisodeLength;
        public float PolicyLoss;
        public float ValueLoss;
        public float Entropy;
        public float? SacAlpha;
        public long TotalSteps;
        public long EpisodeCount;
        public string PolicyGroup = "";
        public string OpponentGroup = "";
        public string OpponentSource = "";
        public string OpponentCheckpointPath = "";
        public long? OpponentUpdateCount;
        public float? LearnerElo;
        public float? PoolWinRate;
        public float? CurriculumProgress;
        // Evaluation rollout fields
        public bool IsEval;
        public float EvalMeanReward;
        public int EvalEpisodes;
    }

    private sealed class RunStatus
    {
        public string Status = "idle";
        public long TotalSteps;
        public long EpisodeCount;
        public long WorkerEpisodeCount;
        public string Message = "";
        public string ResumedFrom = "";
    }

    private sealed record RunMeta(string DisplayName, string[] AgentNames, string[] AgentGroups, bool HasCurriculum);

    private enum CheckpointSortMode
    {
        NewestUpdate = 0,
        OldestUpdate = 1,
        MostSteps = 2,
        MostEpisodes = 3,
        BestReward = 4,
        Algorithm = 5,
        Type = 6,
    }

    // ── UI handles ───────────────────────────────────────────────────────────
    private OptionButton? _runDropdown;
    private LineEdit? _prefixFilter;
    private LineEdit? _renameEdit;
    private Button? _renameBtn;
    private Label? _policyFilterLabel;
    private OptionButton? _policyFilterDropdown;
    private Label? _liveBadge;
    private Label? _headerStatus;
    private ColorRect? _statusDot;
    private Label? _statusLabel;
    private Label? _statAvgReward;
    private Label? _statBestReward;
    private Label? _statEvalReward;
    private Label? _statTotalSteps;
    private Label? _statEpisodes;
    private LineChartPanel? _rewardChart;
    private LineChartPanel? _lossChart;
    private LineChartPanel? _entropyChart;
    private LineChartPanel? _sacAlphaChart;
    private LineChartPanel? _lengthChart;
    private LineChartPanel? _eloChart;
    private LineChartPanel? _curriculumChart;
    private FileDialog? _exportDialog;

    // Checkpoint history panel
    private Button? _checkpointToggleBtn;
    private Control? _checkpointHistoryPanel;
    private VBoxContainer? _checkpointRowContainer;
    private Label? _checkpointStatusLabel;
    private OptionButton? _checkpointSortDropdown;
    private List<CheckpointHistoryEntry> _checkpointEntries = new();
    private FileDialog? _checkpointExportDialog;
    private CheckpointHistoryEntry? _pendingExportCheckpointEntry;

    // HPO dashboard panel
    private HPODashPanel? _hpoPanel;

    // ── State ────────────────────────────────────────────────────────────────
    private readonly List<Metric> _metrics = new();
    private readonly List<string> _runIds = new();
    private readonly Dictionary<string, long> _metricsFileOffsets = new();
    private string _selectedRunId = "";
    private RunMeta _selectedRunMeta = new("", Array.Empty<string>(), Array.Empty<string>(), false);
    private string _selectedPolicyGroupFilter = "";
    private List<string> _knownPolicyGroups = new();
    private readonly HashSet<string> _rewardSeriesLabels = new();
    private readonly HashSet<string> _entropySeriesLabels = new();
    private readonly HashSet<string> _sacAlphaSeriesLabels = new();
    private readonly HashSet<string> _lengthSeriesLabels = new();
    private readonly HashSet<string> _lossSeriesLabels = new();
    private double _pollTimer;
    private double _livePulseAccum;
    private bool _isLive;
    private int _lastKnownRunCount = -1;
    private string _activeRunId = "";
    private CheckpointSortMode _checkpointSortMode = CheckpointSortMode.NewestUpdate;

    private const double PollInterval = 2.0;
    private const double LivePulseInterval = 0.8;

    // ── Palette ──────────────────────────────────────────────────────────────
    private static readonly Color CRunning = new(0.20f, 0.85f, 0.35f);
    private static readonly Color CStopped = new(0.70f, 0.70f, 0.70f);
    private static readonly Color CIdle = new(0.45f, 0.45f, 0.45f);
    private static readonly Color CPolicyLoss = new(0.92f, 0.42f, 0.22f);
    private static readonly Color CValueLoss = new(0.35f, 0.62f, 0.92f);
    private static readonly Color CElo = new(0.92f, 0.42f, 0.78f);
    private static readonly Color CCurriculum = new(0.35f, 0.82f, 0.88f);
    private static readonly Color CSacAlpha = new(0.90f, 0.78f, 0.28f);

    // Palette used to assign one color per policy group (reward / entropy / length charts).
    private static readonly Color[] CPolicyPalette =
    {
        new(0.22f, 0.82f, 0.42f),  // green
        new(0.35f, 0.62f, 0.92f),  // blue
        new(0.92f, 0.42f, 0.22f),  // orange
        new(0.88f, 0.68f, 0.22f),  // amber
        new(0.72f, 0.42f, 0.92f),  // purple
        new(0.92f, 0.42f, 0.78f),  // pink
        new(0.35f, 0.82f, 0.88f),  // cyan
        new(0.82f, 0.92f, 0.35f),  // lime
    };

    // ── Godot lifecycle ──────────────────────────────────────────────────────
    public override void _Ready()
    {
        SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;
        BuildUi();
        DiscoverAndSelectLatestRun();
    }

    public override void _Process(double delta)
    {
        try
        {
            if (!IsInsideTree() || !Visible) return;

            _pollTimer += delta;
            if (_pollTimer >= PollInterval)
            {
                _pollTimer = 0;
                PollUpdate();
            }

            // Live badge pulse
            if (_isLive && _liveBadge is not null)
            {
                _livePulseAccum += delta;
                if (_livePulseAccum >= LivePulseInterval)
                {
                    _livePulseAccum = 0;
                    _liveBadge.Modulate = _liveBadge.Modulate.A > 0.5f
                        ? new Color(1, 1, 1, 0.15f)
                        : new Color(1, 1, 1, 1.0f);
                }
            }
        }
        catch (Exception ex)
        {
            GD.PushError($"[RLDashboard] Unhandled dashboard update error: {ex.Message}");
        }
    }

    // ── Public API (called by editor plugin) ─────────────────────────────────

    /// <summary>
    /// Called by <see cref="RLAgentPluginEditor"/> immediately after launching a
    /// training bootstrap scene. Auto-selects the new run and shows the LIVE badge.
    /// </summary>
    public void OnTrainingStarted(string runId)
    {
        _activeRunId = runId;
        DiscoverAndSelectLatestRun();

        if (!_runIds.Contains(runId))
        {
            // Run directory may not exist yet; add the entry manually.
            _runIds.Insert(0, runId);
            RebuildDropdownItems();
        }

        SelectRun(runId);
        _isLive = true;
        ShowLiveBadge();
    }

    public void OnTrainingStopped()
    {
        _activeRunId = "";
        _isLive = false;
        HideLiveBadge();

        if (string.IsNullOrEmpty(_selectedRunId))
            return;

        var status = NormalizeStatus(ReadStatusFile($"res://RL-Agent-Training/runs/{_selectedRunId}/status.json"));
        SetStatusUi(status);
        RefreshStats(status);
    }

    // ── UI construction ──────────────────────────────────────────────────────
    private void BuildUi()
    {
        var scroll = new ScrollContainer();
        scroll.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        scroll.HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled;
        scroll.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        scroll.SizeFlagsVertical = SizeFlags.ExpandFill;
        AddChild(scroll);

        var margin = new MarginContainer();
        margin.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        margin.AddThemeConstantOverride("margin_left", 10);
        margin.AddThemeConstantOverride("margin_right", 10);
        margin.AddThemeConstantOverride("margin_top", 8);
        margin.AddThemeConstantOverride("margin_bottom", 8);
        scroll.AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        vbox.AddThemeConstantOverride("separation", 6);
        margin.AddChild(vbox);

        vbox.AddChild(BuildHeader());
        vbox.AddChild(new HSeparator());
        vbox.AddChild(BuildStatsBar());
        vbox.AddChild(BuildChartGrid());
        vbox.AddChild(new HSeparator());
        vbox.AddChild(BuildCheckpointHistorySection());
        vbox.AddChild(new HSeparator());
        vbox.AddChild(BuildHpoSection());
    }

    private Control BuildHeader()
    {
        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 4);

        // ── Row 1: title, filter, run selector, status, live badge ───────────
        var row1 = new HBoxContainer();
        row1.AddThemeConstantOverride("separation", 8);
        row1.CustomMinimumSize = new Vector2(0, 30);

        var title = new Label { Text = "RL Training Dashboard" };
        title.AddThemeFontSizeOverride("font_size", 18);
        title.VerticalAlignment = VerticalAlignment.Center;
        row1.AddChild(title);

        row1.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        // Prefix filter
        var filterLabel = new Label { Text = "Filter:", VerticalAlignment = VerticalAlignment.Center };
        row1.AddChild(filterLabel);

        _prefixFilter = new LineEdit
        {
            PlaceholderText = "by prefix…",
            CustomMinimumSize = new Vector2(120, 0),
        };
        _prefixFilter.TextChanged += _ => DiscoverAndSelectLatestRun();
        row1.AddChild(_prefixFilter);

        // Run selector
        var runLabel = new Label { Text = "Run:", VerticalAlignment = VerticalAlignment.Center };
        row1.AddChild(runLabel);

        _runDropdown = new OptionButton
        {
            CustomMinimumSize = new Vector2(250, 0),
            TooltipText = "Select a training run to inspect",
        };
        _runDropdown.GetPopup().MaxSize = new Vector2I(1600, 360);
        _runDropdown.ItemSelected += OnRunSelected;
        row1.AddChild(_runDropdown);

        var refreshBtn = new Button
        {
            Text = " ⟳ ",
            TooltipText = "Rescan runs directory for new runs",
        };
        refreshBtn.Pressed += () => DiscoverAndSelectLatestRun();
        row1.AddChild(refreshBtn);



        // LIVE badge (hidden until training starts)
        _liveBadge = new Label
        {
            Text = "● LIVE",
            Visible = false,
            VerticalAlignment = VerticalAlignment.Center,
        };
        _liveBadge.AddThemeColorOverride("font_color", CRunning);
        _liveBadge.AddThemeFontSizeOverride("font_size", 12);
        row1.AddChild(_liveBadge);

        vbox.AddChild(row1);

        // ── Row 2: rename + export ────────────────────────────────────────────
        var row2 = new HBoxContainer();


        row2.AddThemeConstantOverride("separation", 6);
        row2.CustomMinimumSize = new Vector2(0, 24);

        var nameLabel = new Label { Text = "Name:", VerticalAlignment = VerticalAlignment.Center };
        row2.AddChild(nameLabel);

        _renameEdit = new LineEdit
        {
            PlaceholderText = "Display name…",
            CustomMinimumSize = new Vector2(200, 0),
            Editable = false,
        };
        row2.AddChild(_renameEdit);

        _renameBtn = new Button { Text = "Rename", Visible = false };
        _renameBtn.Pressed += () => SaveDisplayName(_renameEdit?.Text ?? "");
        row2.AddChild(_renameBtn);

        // Status indicator
        var statusBox = new HBoxContainer();
        statusBox.AddThemeConstantOverride("separation", 5);

        _statusDot = new ColorRect
        {
            Color = CIdle,
            CustomMinimumSize = new Vector2(10, 10),
            SizeFlagsVertical = SizeFlags.ShrinkCenter
        };
        statusBox.AddChild(_statusDot);
        _statusLabel = new Label { Text = "Idle", VerticalAlignment = VerticalAlignment.Center };
        statusBox.AddChild(_statusLabel);
        row2.AddChild(statusBox);

        _policyFilterLabel = new Label
        {
            Text = "Policy:",
            VerticalAlignment = VerticalAlignment.Center,
            Visible = false,
        };
        row2.AddChild(_policyFilterLabel);

        _policyFilterDropdown = new OptionButton
        {
            CustomMinimumSize = new Vector2(160, 0),
            TooltipText = "Filter charts to a single policy group",
            Visible = false,
        };
        _policyFilterDropdown.AddItem("All Policies");
        _policyFilterDropdown.ItemSelected += OnPolicyFilterSelected;
        row2.AddChild(_policyFilterDropdown);

        row2.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        var exportBtn = new Button
        {
            Text = "Export Run",
            TooltipText = "Export trained model weights as .rlmodel file(s)",
        };
        exportBtn.Pressed += ExportRun;
        row2.AddChild(exportBtn);

        _headerStatus = new Label { VerticalAlignment = VerticalAlignment.Center };
        _headerStatus.AddThemeFontSizeOverride("font_size", 11);
        _headerStatus.Modulate = new Color(0.75f, 0.75f, 0.75f);
        row2.AddChild(_headerStatus);

        vbox.AddChild(row2);

        return vbox;
    }

    private Control BuildStatsBar()
    {
        var panel = new PanelContainer();
        panel.CustomMinimumSize = new Vector2(0, 40);
        panel.SizeFlagsVertical = SizeFlags.ShrinkBegin;

        var hbox = new HBoxContainer();
        hbox.AddThemeConstantOverride("separation", 0);
        panel.AddChild(hbox);

        _statAvgReward = AddStatCard(hbox, "Avg Reward (50 ep)", "—", first: true);
        hbox.AddChild(MakeVSep());
        _statBestReward = AddStatCard(hbox, "Best Reward", "—", first: false);
        hbox.AddChild(MakeVSep());
        _statEvalReward = AddStatCard(hbox, "Eval Reward", "—", first: false);
        hbox.AddChild(MakeVSep());
        _statTotalSteps = AddStatCard(hbox, "Total Steps", "—", first: false);
        hbox.AddChild(MakeVSep());
        _statEpisodes = AddStatCard(hbox, "Episodes", "—", first: false);

        return panel;
    }

    private static Label AddStatCard(HBoxContainer parent, string title, string dflt, bool first)
    {
        var margin = new MarginContainer();
        margin.SizeFlagsHorizontal = SizeFlags.ShrinkBegin;
        margin.AddThemeConstantOverride("margin_left", first ? 8 : 8);
        margin.AddThemeConstantOverride("margin_right", 8);
        margin.AddThemeConstantOverride("margin_top", 5);
        margin.AddThemeConstantOverride("margin_bottom", 5);
        parent.AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 0);
        margin.AddChild(vbox);

        var lbl = new Label { Text = title };
        lbl.AddThemeFontSizeOverride("font_size", 10);
        lbl.Modulate = new Color(0.60f, 0.60f, 0.60f);
        vbox.AddChild(lbl);

        var value = new Label { Text = dflt };
        value.AddThemeFontSizeOverride("font_size", 15);
        vbox.AddChild(value);

        return value;
    }

    private static VSeparator MakeVSep()
    {
        var sep = new VSeparator();
        sep.SizeFlagsVertical = SizeFlags.ShrinkCenter;
        sep.CustomMinimumSize = new Vector2(1, 26);
        return sep;
    }

    private Control BuildChartGrid()
    {
        var grid = new GridContainer { Columns = 2 };
        grid.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        grid.SizeFlagsVertical = SizeFlags.ShrinkBegin;
        grid.CustomMinimumSize = new Vector2(0, 340);
        grid.AddThemeConstantOverride("h_separation", 4);
        grid.AddThemeConstantOverride("v_separation", 4);

        _rewardChart = MakeChart("Episode Reward");
        _lossChart = MakeChart("Policy Loss  /  Value Loss");
        _entropyChart = MakeChart("Entropy");
        _sacAlphaChart = MakeChart("SAC Alpha");
        _lengthChart = MakeChart("Episode Length");
        _eloChart = MakeChart("Learner Elo");
        _curriculumChart = MakeChart("Curriculum Progress");
        _sacAlphaChart.Visible = false;  // shown only when SAC alpha metrics are present
        _eloChart.Visible = false;  // shown only during self-play
        _curriculumChart.Visible = false;  // shown only when curriculum is configured

        grid.AddChild(_rewardChart);
        grid.AddChild(_lossChart);
        grid.AddChild(_entropyChart);
        grid.AddChild(_sacAlphaChart);
        grid.AddChild(_lengthChart);
        grid.AddChild(_eloChart);
        grid.AddChild(_curriculumChart);

        return grid;
    }

    private static LineChartPanel MakeChart(string title)
    {
        var chart = new LineChartPanel { ChartTitle = title };
        chart.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        chart.SizeFlagsVertical = SizeFlags.ExpandFill;
        chart.CustomMinimumSize = new Vector2(0, 140);
        return chart;
    }

    // ── Run discovery ─────────────────────────────────────────────────────────

    private void DiscoverAndSelectLatestRun()
    {
        var absDir = ProjectSettings.GlobalizePath("res://RL-Agent-Training/runs");
        var prevRunId = _selectedRunId;

        _lastKnownRunCount = System.IO.Directory.Exists(absDir)
            ? System.IO.Directory.GetDirectories(absDir).Length
            : 0;

        _runIds.Clear();
        _runDropdown?.Clear();

        if (!System.IO.Directory.Exists(absDir)) return;

        var filterPrefix = _prefixFilter?.Text.Trim() ?? "";

        var dirs = System.IO.Directory.GetDirectories(absDir)
            .Select(System.IO.Path.GetFileName)
            .OfType<string>()
            .Where(n => !string.IsNullOrEmpty(n))
            .Where(n => string.IsNullOrEmpty(filterPrefix)
                        || ExtractRunPrefix(n).Contains(filterPrefix, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(n => GetRunSortValue(n, System.IO.Path.Combine(absDir, n)))
            .ThenByDescending(n => n, StringComparer.Ordinal)
            .ToList();

        foreach (var id in dirs)
        {
            var meta = ReadMeta(System.IO.Path.Combine(absDir, id));
            var label = BuildRunLabel(id, meta.DisplayName);
            _runIds.Add(id);
            _runDropdown?.AddItem(label);
        }

        // Preserve previous selection if it still appears in the filtered list.
        var existingIdx = _runIds.IndexOf(prevRunId);
        if (existingIdx >= 0)
        {
            _runDropdown?.Select(existingIdx);
        }
        else if (_runIds.Count > 0)
        {
            SelectRun(_runIds[0]);
        }
    }

    /// <summary>Rebuilds dropdown item text from _runIds (preserves order, keeps selection).</summary>
    private void RebuildDropdownItems()
    {
        if (_runDropdown is null) return;
        var absDir = ProjectSettings.GlobalizePath("res://RL-Agent-Training/runs");

        _runDropdown.Clear();
        foreach (var id in _runIds)
        {
            var meta = ReadMeta(System.IO.Path.Combine(absDir, id));
            var label = BuildRunLabel(id, meta.DisplayName);
            _runDropdown.AddItem(label);
        }

        var selIdx = _runIds.IndexOf(_selectedRunId);
        if (selIdx >= 0) _runDropdown.Select(selIdx);
    }

    private void OnRunSelected(long index)
    {
        if (index < 0 || index >= _runIds.Count) return;
        var runId = _runIds[(int)index];
        SelectRun(runId);
    }

    private void SelectRun(string runId)
    {
        _selectedRunId = runId;
        _selectedRunMeta = ReadMeta(ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{runId}"));
        _metricsFileOffsets.Clear();
        _metrics.Clear();
        if (_hpoPanel is not null)
        {
            _hpoPanel.RunIdFilter = runId;
            _hpoPanel.PollUpdate();
        }

        var idx = _runIds.IndexOf(runId);
        if (idx >= 0) _runDropdown?.Select(idx);

        _selectedPolicyGroupFilter = "";
        _knownPolicyGroups.Clear();
        _rewardSeriesLabels.Clear();
        _entropySeriesLabels.Clear();
        _lengthSeriesLabels.Clear();
        _lossSeriesLabels.Clear();
        if (_policyFilterDropdown is not null)
        {
            _policyFilterDropdown.Clear();
            _policyFilterDropdown.AddItem("All Policies");
            _policyFilterDropdown.Visible = false;
        }
        if (_policyFilterLabel is not null) _policyFilterLabel.Visible = false;

        _rewardChart?.ClearSeries();
        _lossChart?.ClearSeries();
        _entropyChart?.ClearSeries();
        _lengthChart?.ClearSeries();
        _eloChart?.ClearSeries();
        _curriculumChart?.ClearSeries();
        if (_eloChart is not null) _eloChart.Visible = false;
        if (_curriculumChart is not null) _curriculumChart.Visible = false;

        if (_renameEdit is not null) _renameEdit.Editable = true;
        if (_renameBtn is not null) _renameBtn.Visible = true;
        if (_renameEdit is not null) _renameEdit.Text = _selectedRunMeta.DisplayName;

        _checkpointEntries.Clear();
        RebuildCheckpointRows();
        if (_checkpointStatusLabel is not null) _checkpointStatusLabel.Text = "";

        SetStatusUi(new RunStatus { Status = "loading" });
        PollUpdate();
    }

    // ── Polling & data ────────────────────────────────────────────────────────
    private void PollUpdate()
    {
        try
        {
            // Passive new-run detection.
            var absDir = ProjectSettings.GlobalizePath("res://RL-Agent-Training/runs");
            if (System.IO.Directory.Exists(absDir))
            {
                var currentCount = System.IO.Directory.GetDirectories(absDir).Length;
                if (_lastKnownRunCount >= 0 && currentCount != _lastKnownRunCount)
                {
                    _lastKnownRunCount = currentCount;
                    DiscoverAndSelectLatestRun();
                    return; // DiscoverAndSelectLatestRun → SelectRun → PollUpdate handles the rest.
                }
            }

            if (string.IsNullOrEmpty(_selectedRunId)) return;

            var runDir = $"res://RL-Agent-Training/runs/{_selectedRunId}";
            var runDirAbs = ProjectSettings.GlobalizePath(runDir);
            var hasDirectMetrics = false;
            if (System.IO.Directory.Exists(runDirAbs))
            {
                var metricFiles = System.IO.Directory.GetFiles(runDirAbs, "metrics__*.jsonl");
                hasDirectMetrics = metricFiles.Length > 0;
                foreach (var file in metricFiles)
                    ReadNewMetrics(file, absolute: true);
            }
            var status = NormalizeStatus(ReadStatusFile($"{runDir}/status.json"));
            if (!hasDirectMetrics)
            {
                var hpoStatus = TryLoadHpoOverviewMetrics(_selectedRunId);
                if (hpoStatus is not null)
                    status = NormalizeStatus(hpoStatus);
            }
            SetStatusUi(status);
            RefreshPolicyFilterDropdown();
            RefreshCharts();
            RefreshStats(status);

            // Clear live badge once training explicitly finishes.
            if (_isLive && status.Status is "done" or "stopped")
            {
                _isLive = false;
                HideLiveBadge();
            }

            if (_checkpointHistoryPanel?.Visible ?? false)
                RefreshCheckpointHistory();

            _hpoPanel?.PollUpdate();
        }
        catch (Exception ex)
        {
            GD.PushError($"[RLDashboard] PollUpdate failed: {ex.Message}");
        }
    }

    private void ReadNewMetrics(string path, bool absolute = false)
    {
        var absPath = absolute ? path : ProjectSettings.GlobalizePath(path);
        if (!System.IO.File.Exists(absPath)) return;

        if (!_metricsFileOffsets.TryGetValue(absPath, out var offset))
            offset = 0;

        try
        {
            using var stream = new System.IO.FileStream(
                absPath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite);

            if (stream.Length <= offset) return;

            stream.Seek(offset, System.IO.SeekOrigin.Begin);
            using var reader = new System.IO.StreamReader(stream, leaveOpen: true);

            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var m = ParseMetricLine(line);
                if (m is not null) _metrics.Add(m);
            }

            _metricsFileOffsets[absPath] = stream.Position;
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLDashboard] Failed to read metrics: {ex.Message}");
        }
    }

    private static RunStatus ReadStatusFile(string resPath)
    {
        var absPath = ProjectSettings.GlobalizePath(resPath);
        if (!System.IO.File.Exists(absPath)) return new RunStatus();

        try
        {
            using var doc = JsonDocument.Parse(System.IO.File.ReadAllText(absPath));
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return new RunStatus();

            var d = doc.RootElement;
            return new RunStatus
            {
                Status = GetString(d, "status", "unknown"),
                TotalSteps = GetLong(d, "total_steps"),
                EpisodeCount = GetLong(d, "episode_count"),
                WorkerEpisodeCount = GetLong(d, "worker_episode_count"),
                Message = GetString(d, "message", ""),
                ResumedFrom = GetString(d, "resumed_from", ""),
            };
        }
        catch
        {
            return new RunStatus();
        }
    }

    private static Metric? ParseMetricLine(string line)
    {
        try
        {
            using var doc = JsonDocument.Parse(line);
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return null;

            var d = doc.RootElement;
            return new Metric
            {
                EpisodeReward = GetFloat(d, "episode_reward"),
                EpisodeLength = (int)GetLong(d, "episode_length"),
                PolicyLoss = GetFloat(d, "policy_loss"),
                ValueLoss = GetFloat(d, "value_loss"),
                Entropy = GetFloat(d, "entropy"),
                SacAlpha = d.TryGetProperty("sac_alpha", out _) ? GetFloat(d, "sac_alpha") : null,
                TotalSteps = GetLong(d, "total_steps"),
                EpisodeCount = GetLong(d, "episode_count"),
                PolicyGroup = GetString(d, "policy_group", ""),
                OpponentGroup = GetString(d, "opponent_group", ""),
                OpponentSource = GetString(d, "opponent_source", ""),
                OpponentCheckpointPath = GetString(d, "opponent_checkpoint_path", ""),
                OpponentUpdateCount = d.TryGetProperty("opponent_update_count", out _)
                    ? GetLong(d, "opponent_update_count")
                    : null,
                LearnerElo = d.TryGetProperty("learner_elo", out _) ? GetFloat(d, "learner_elo") : null,
                PoolWinRate = d.TryGetProperty("pool_avg_win_rate", out _) ? GetFloat(d, "pool_avg_win_rate") : null,
                CurriculumProgress = d.TryGetProperty("curriculum_progress", out _) ? GetFloat(d, "curriculum_progress") : null,
                IsEval = GetBool(d, "is_eval"),
                EvalMeanReward = d.TryGetProperty("eval_mean_reward", out _) ? GetFloat(d, "eval_mean_reward") : 0f,
                EvalEpisodes = d.TryGetProperty("eval_episodes", out _) ? (int)GetLong(d, "eval_episodes") : 0,
            };
        }
        catch
        {
            return null;
        }
    }

    private RunStatus? TryLoadHpoOverviewMetrics(string ownerRunId)
    {
        var hpoDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/hpo/{ownerRunId}");
        if (!System.IO.Directory.Exists(hpoDirAbs))
            return null;

        var stateFiles = System.IO.Directory
            .GetFiles(hpoDirAbs, "study_state.json", System.IO.SearchOption.AllDirectories)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();
        if (stateFiles.Length == 0)
            return null;

        _metrics.Clear();
        _metricsFileOffsets.Clear();

        int totalTrials = 0;
        int completeTrials = 0;
        int runningTrials = 0;

        foreach (var stateFile in stateFiles)
        {
            try
            {
                var parsed = StudyState.ParseSanitizedJson(System.IO.File.ReadAllText(stateFile));
                if (parsed.VariantType != Variant.Type.Dictionary)
                    continue;

                var data = parsed.AsGodotDictionary();
                var studyName = GetString(data, "study_name", System.IO.Path.GetFileName(System.IO.Path.GetDirectoryName(stateFile) ?? stateFile));
                if (!data.ContainsKey("trials") || data["trials"].VariantType != Variant.Type.Array)
                    continue;

                var trialDicts = data["trials"]
                    .AsGodotArray()
                    .Where(v => v.VariantType == Variant.Type.Dictionary)
                    .Select(v => v.AsGodotDictionary())
                    .OrderBy(d => (int)GetLong(d, "trial_id"))
                    .ToList();

                foreach (var trial in trialDicts)
                {
                    totalTrials++;
                    var state = GetString(trial, "state", "Pending");
                    if (string.Equals(state, "Complete", StringComparison.Ordinal))
                        completeTrials++;
                    else if (string.Equals(state, "Running", StringComparison.Ordinal)
                             || string.Equals(state, "Pending", StringComparison.Ordinal))
                        runningTrials++;

                    var runDir = GetString(trial, "run_dir", "");
                    if (string.IsNullOrWhiteSpace(runDir))
                        continue;

                    var trialRunAbs = ProjectSettings.GlobalizePath(runDir);
                    if (!System.IO.Directory.Exists(trialRunAbs))
                        continue;

                    var metricFiles = System.IO.Directory
                        .GetFiles(trialRunAbs, "metrics__*.jsonl", System.IO.SearchOption.TopDirectoryOnly)
                        .OrderBy(path => path, StringComparer.Ordinal);

                    foreach (var metricFile in metricFiles)
                    {
                        var lastLine = ReadLastNonEmptyLine(metricFile);
                        if (string.IsNullOrWhiteSpace(lastLine))
                            continue;

                        var metric = ParseMetricLine(lastLine);
                        if (metric is null)
                            continue;

                        metric.PolicyGroup = string.IsNullOrWhiteSpace(metric.PolicyGroup)
                            ? studyName
                            : stateFiles.Length > 1
                                ? $"{studyName}:{metric.PolicyGroup}"
                                : metric.PolicyGroup;

                        _metrics.Add(metric);
                    }
                }
            }
            catch (Exception ex)
            {
                GD.PushWarning($"[RLDashboard] Failed to read HPO study overview from '{stateFile}': {ex.Message}");
            }
        }

        if (totalTrials == 0)
            return null;

        return new RunStatus
        {
            Status = runningTrials > 0 ? "running" : "done",
            EpisodeCount = completeTrials,
            TotalSteps = _metrics.Count > 0 ? _metrics.Max(m => m.TotalSteps) : 0,
            Message = $"HPO study overview: {completeTrials}/{totalTrials} trial(s) complete.",
        };
    }

    private static string? ReadLastNonEmptyLine(string absPath)
    {
        using var stream = new System.IO.FileStream(
            absPath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite);
        if (stream.Length == 0)
            return null;

        var chars = new List<char>();
        for (long pos = stream.Length - 1; pos >= 0; pos--)
        {
            stream.Seek(pos, System.IO.SeekOrigin.Begin);
            var value = stream.ReadByte();
            if (value < 0)
                break;

            var ch = (char)value;
            if (ch == '\n')
            {
                if (chars.Count > 0)
                    break;
                continue;
            }

            if (ch == '\r')
                continue;

            chars.Add(ch);
        }

        if (chars.Count == 0)
            return null;

        chars.Reverse();
        return new string(chars.ToArray());
    }

    // ── Export ────────────────────────────────────────────────────────────────
    private void ExportRun()
    {
        if (string.IsNullOrEmpty(_selectedRunId))
        {
            SetHeaderStatus("No run selected.");
            return;
        }

        EnsureExportDialog();
        _exportDialog!.PopupCentered(new Vector2I(700, 450));
    }

    private void EnsureExportDialog()
    {
        if (_exportDialog is not null && IsInstanceValid(_exportDialog)) return;

        _exportDialog = new FileDialog
        {
            FileMode = FileDialog.FileModeEnum.OpenDir,
            Access = FileDialog.AccessEnum.Filesystem,
            Title = "Select Export Folder",
        };
        _exportDialog.DirSelected += OnExportDirSelected;
        AddChild(_exportDialog);
    }

    private void OnExportDirSelected(string dir)
    {
        if (string.IsNullOrEmpty(_selectedRunId)) return;

        var runDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{_selectedRunId}");
        var meta = ReadMeta(runDirAbs);

        // Build a list of (exportName, checkpointPath) pairs, deduplicating by checkpoint
        // so agents that share a policy group produce only one output file (named after the first agent).
        var exports = new List<(string Name, string CheckpointPath)>();
        var seenCheckpoints = new System.Collections.Generic.HashSet<string>();

        if (meta.AgentNames.Length > 0)
        {
            for (var i = 0; i < meta.AgentNames.Length; i++)
            {
                var agentName = meta.AgentNames[i];
                var safeGroup = i < meta.AgentGroups.Length ? meta.AgentGroups[i] : string.Empty;
                var cpPath = string.IsNullOrEmpty(safeGroup)
                    ? null
                    : RLCheckpoint.ResolveCheckpointExtension(
                        System.IO.Path.Combine(runDirAbs, $"checkpoint__{safeGroup}.json"));

                if (cpPath is null || !System.IO.File.Exists(cpPath))
                {
                    // Fall back to first checkpoint in the run directory.
                    cpPath = RLModelExporter.FindCheckpointInRunDir(runDirAbs);
                }

                if (cpPath is null) continue;
                if (!seenCheckpoints.Add(cpPath)) continue; // shared group — skip duplicate

                exports.Add((agentName, cpPath));
            }
        }
        else
        {
            // No agent names in meta — fall back to single export named after the run.
            var cpPath = RLModelExporter.FindCheckpointInRunDir(runDirAbs);
            if (cpPath is not null)
                exports.Add((_selectedRunId, cpPath));
        }

        if (exports.Count == 0)
        {
            SetHeaderStatus("Export failed: no checkpoint found in run directory.");
            return;
        }

        var failed = new List<string>();
        foreach (var (agentName, cpPath) in exports)
        {
            var destPath = System.IO.Path.Combine(dir, $"{SanitizeFileName(agentName)}.rlmodel");
            if (RLModelExporter.Export(cpPath, destPath) != Error.Ok)
                failed.Add(agentName);
        }

        SetHeaderStatus(failed.Count == 0
            ? $"Exported {exports.Count} file(s) → {dir}"
            : $"Export failed for: {string.Join(", ", failed)}");
    }

    // ── Rename / display name ─────────────────────────────────────────────────
    private void SaveDisplayName(string name)
    {
        if (string.IsNullOrEmpty(_selectedRunId)) return;

        var runDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{_selectedRunId}");
        var meta = ReadMeta(runDirAbs);
        WriteMeta(runDirAbs, meta with { DisplayName = name });

        // Update dropdown label in place.
        var idx = _runIds.IndexOf(_selectedRunId);
        if (idx >= 0)
            _runDropdown?.SetItemText(idx, BuildRunLabel(_selectedRunId, name));

        SetHeaderStatus(string.IsNullOrWhiteSpace(name) ? "Name cleared." : $"Renamed to \"{name}\".");
    }

    // ── Meta sidecar ─────────────────────────────────────────────────────────
    private static RunMeta ReadMeta(string runDirAbsPath)
    {
        var metaPath = System.IO.Path.Combine(runDirAbsPath, "meta.json");
        if (!System.IO.File.Exists(metaPath)) return new RunMeta("", Array.Empty<string>(), Array.Empty<string>(), false);

        try
        {
            using var doc = JsonDocument.Parse(System.IO.File.ReadAllText(metaPath));
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                return new RunMeta("", Array.Empty<string>(), Array.Empty<string>(), false);

            var d = doc.RootElement;
            var displayName = GetString(d, "display_name", "");
            var agentNames = GetStringArray(d, "agent_names");
            var agentGroups = GetStringArray(d, "agent_groups");
            var hasCurriculum = GetBool(d, "has_curriculum");

            return new RunMeta(displayName, agentNames, agentGroups, hasCurriculum);
        }
        catch
        {
            return new RunMeta("", Array.Empty<string>(), Array.Empty<string>(), false);
        }
    }

    private static void WriteMeta(string runDirAbsPath, RunMeta meta)
    {
        try
        {
            System.IO.Directory.CreateDirectory(runDirAbsPath);

            var agentArr = new Godot.Collections.Array();
            foreach (var name in meta.AgentNames) agentArr.Add(Variant.From(name));

            var groupArr = new Godot.Collections.Array();
            foreach (var g in meta.AgentGroups) groupArr.Add(Variant.From(g));

            var d = new Godot.Collections.Dictionary
            {
                { "display_name",  meta.DisplayName },
                { "agent_names",   agentArr },
                { "agent_groups",  groupArr },
                { "has_curriculum", meta.HasCurriculum },
            };

            System.IO.File.WriteAllText(
                System.IO.Path.Combine(runDirAbsPath, "meta.json"),
                Json.Stringify(d));
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RLDashboard] Failed to write meta.json: {ex.Message}");
        }
    }

    // ── UI updates ────────────────────────────────────────────────────────────
    private void SetStatusUi(RunStatus status)
    {
        if (_statusLabel is null || _statusDot is null) return;

        switch (status.Status)
        {
            case "running":
                _statusDot.Color = CRunning;
                _statusLabel.Text = string.IsNullOrEmpty(status.ResumedFrom) ? "Running" : "Running (resumed)";
                break;
            case "done":
            case "stopped":
                _statusDot.Color = CStopped;
                _statusLabel.Text = string.IsNullOrEmpty(status.ResumedFrom) ? "Stopped" : "Stopped (resumed)";
                break;
            case "loading":
                _statusDot.Color = CIdle;
                _statusLabel.Text = "Loading…";
                break;
            default:
                _statusDot.Color = CIdle;
                _statusLabel.Text = string.IsNullOrEmpty(_selectedRunId) ? "No run selected" : "Idle";
                break;
        }
    }

    private RunStatus NormalizeStatus(RunStatus status)
    {
        status.Status = (status.Status ?? string.Empty).Trim().ToLowerInvariant();
        if (status.Status == "live")
            status.Status = "running";

        var isSelectedRunActive = !string.IsNullOrEmpty(_activeRunId)
            && string.Equals(_selectedRunId, _activeRunId, StringComparison.Ordinal)
            && EditorInterface.Singleton.IsPlayingScene();

        if (status.Status == "running" && !isSelectedRunActive)
            status.Status = "stopped";

        return status;
    }

    private void RefreshPolicyFilterDropdown()
    {
        if (_policyFilterDropdown is null) return;

        var groups = _metrics
            .Select(m => m.PolicyGroup)
            .Where(g => !string.IsNullOrEmpty(g))
            .Distinct()
            .OrderBy(g => g)
            .ToList();

        // Skip rebuild if groups haven't changed (avoids resetting the selection mid-run).
        if (groups.SequenceEqual(_knownPolicyGroups)) return;
        _knownPolicyGroups = groups;

        var prevFilter = _selectedPolicyGroupFilter;
        _policyFilterDropdown.Clear();
        _policyFilterDropdown.AddItem("All Policies");
        foreach (var g in groups)
            _policyFilterDropdown.AddItem(g);

        // Restore previous selection when possible.
        if (!string.IsNullOrEmpty(prevFilter))
        {
            var idx = groups.IndexOf(prevFilter);
            if (idx >= 0)
                _policyFilterDropdown.Select(idx + 1);
            else
            {
                _policyFilterDropdown.Select(0);
                _selectedPolicyGroupFilter = "";
            }
        }

        // Only show the controls when there are multiple policies.
        var show = groups.Count > 1;
        _policyFilterDropdown.Visible = show;
        if (_policyFilterLabel is not null) _policyFilterLabel.Visible = show;
    }

    private void OnPolicyFilterSelected(long index)
    {
        _selectedPolicyGroupFilter = index == 0 ? "" : _knownPolicyGroups[(int)index - 1];

        // Clear charts so stale series from the previous selection are removed cleanly.
        _rewardChart?.ClearSeries();
        _lossChart?.ClearSeries();
        _entropyChart?.ClearSeries();
        _lengthChart?.ClearSeries();
        _rewardSeriesLabels.Clear();
        _entropySeriesLabels.Clear();
        _lengthSeriesLabels.Clear();
        _lossSeriesLabels.Clear();

        RefreshCharts();
        var status = NormalizeStatus(ReadStatusFile($"res://RL-Agent-Training/runs/{_selectedRunId}/status.json"));
        RefreshStats(status);
    }

    private void RefreshCharts()
    {
        if (_metrics.Count == 0) return;

        var allGroups = _metrics
            .Select(m => m.PolicyGroup)
            .Distinct()
            .ToList();

        var activeGroups = string.IsNullOrEmpty(_selectedPolicyGroupFilter)
            ? allGroups
            : allGroups.Where(g => g == _selectedPolicyGroupFilter).ToList();

        bool multi = activeGroups.Count > 1;

        var newRewardLabels = new HashSet<string>();
        var newEntropyLabels = new HashSet<string>();
        var newSacAlphaLabels = new HashSet<string>();
        var newLengthLabels = new HashSet<string>();
        var newLossLabels = new HashSet<string>();
        var hasSacAlphaData = false;

        for (int i = 0; i < activeGroups.Count; i++)
        {
            var group = activeGroups[i];
            var color = CPolicyPalette[i % CPolicyPalette.Length];
            var groupMetrics = _metrics.Where(m => m.PolicyGroup == group && !m.IsEval).ToList();

            var rewardLabel = multi ? group : "Reward";
            var entropyLabel = multi ? group : "Entropy";
            var lengthLabel = multi ? group : "Length";

            _rewardChart?.UpdateSeries(rewardLabel, color, groupMetrics.Select(m => m.EpisodeReward));
            _entropyChart?.UpdateSeries(entropyLabel, color, groupMetrics.Select(m => m.Entropy));
            _lengthChart?.UpdateSeries(lengthLabel, color, groupMetrics.Select(m => (float)m.EpisodeLength));

            newRewardLabels.Add(rewardLabel);
            newEntropyLabels.Add(entropyLabel);
            newLengthLabels.Add(lengthLabel);

            var alphaMetrics = groupMetrics.Where(m => m.SacAlpha.HasValue).ToList();
            if (alphaMetrics.Count > 0)
            {
                hasSacAlphaData = true;
                var alphaLabel = multi ? group : "Alpha";
                var alphaColor = multi ? color : CSacAlpha;
                _sacAlphaChart?.UpdateSeries(alphaLabel, alphaColor, alphaMetrics.Select(m => m.SacAlpha!.Value));
                newSacAlphaLabels.Add(alphaLabel);
            }

            // Eval series on the reward chart — lighter tint to visually separate from training.
            var evalMetrics = _metrics.Where(m => m.PolicyGroup == group && m.IsEval).ToList();
            if (evalMetrics.Count > 0)
            {
                var evalLabel = multi ? $"{group} (eval)" : "Reward (eval)";
                var evalColor = new Color(color.R, color.G, color.B, 0.55f);
                _rewardChart?.UpdateSeries(evalLabel, evalColor, evalMetrics.Select(m => m.EvalMeanReward));
                newRewardLabels.Add(evalLabel);
            }

            // Loss: each group gets a policy-loss and value-loss series.
            var polLabel = multi ? $"{group} Policy" : "Policy";
            var valLabel = multi ? $"{group} Value" : "Value";
            var valColor = multi
                ? new Color(color.R * 0.65f, color.G * 0.65f, color.B * 0.65f, 0.85f)
                : CValueLoss;

            _lossChart?.UpdateSeries(polLabel, color, groupMetrics.Select(m => m.PolicyLoss));
            _lossChart?.UpdateSeries(valLabel, valColor, groupMetrics.Select(m => m.ValueLoss));

            newLossLabels.Add(polLabel);
            newLossLabels.Add(valLabel);
        }

        // Remove any series that no longer belong in the chart.
        foreach (var stale in _rewardSeriesLabels.Except(newRewardLabels))
            _rewardChart?.RemoveSeries(stale);
        foreach (var stale in _entropySeriesLabels.Except(newEntropyLabels))
            _entropyChart?.RemoveSeries(stale);
        foreach (var stale in _sacAlphaSeriesLabels.Except(newSacAlphaLabels))
            _sacAlphaChart?.RemoveSeries(stale);
        foreach (var stale in _lengthSeriesLabels.Except(newLengthLabels))
            _lengthChart?.RemoveSeries(stale);
        foreach (var stale in _lossSeriesLabels.Except(newLossLabels))
            _lossChart?.RemoveSeries(stale);

        _rewardSeriesLabels.Clear(); _rewardSeriesLabels.UnionWith(newRewardLabels);
        _entropySeriesLabels.Clear(); _entropySeriesLabels.UnionWith(newEntropyLabels);
        _sacAlphaSeriesLabels.Clear(); _sacAlphaSeriesLabels.UnionWith(newSacAlphaLabels);
        _lengthSeriesLabels.Clear(); _lengthSeriesLabels.UnionWith(newLengthLabels);
        _lossSeriesLabels.Clear(); _lossSeriesLabels.UnionWith(newLossLabels);

        if (_sacAlphaChart is not null)
            _sacAlphaChart.Visible = hasSacAlphaData;

        // Elo and curriculum are not split by policy (self-play learner has a single Elo).
        bool hasFilter = !string.IsNullOrEmpty(_selectedPolicyGroupFilter);

        var eloMetrics = _metrics
            .Where(m => m.LearnerElo.HasValue && (!hasFilter || m.PolicyGroup == _selectedPolicyGroupFilter))
            .ToList();
        if (_eloChart is not null)
        {
            _eloChart.Visible = eloMetrics.Count > 0;
            if (eloMetrics.Count > 0)
                _eloChart.UpdateSeries("Elo", CElo, eloMetrics.Select(m => m.LearnerElo!.Value));
        }

        var curriculumMetrics = _metrics
            .Where(m => m.CurriculumProgress.HasValue && (!hasFilter || m.PolicyGroup == _selectedPolicyGroupFilter))
            .ToList();
        if (_curriculumChart is not null)
        {
            var showCurriculum = _selectedRunMeta.HasCurriculum || curriculumMetrics.Count > 0;
            _curriculumChart.Visible = showCurriculum;
            if (curriculumMetrics.Count > 0)
                _curriculumChart.UpdateSeries("Progress", CCurriculum,
                    curriculumMetrics.Select(m => m.CurriculumProgress!.Value));
        }
    }

    private void RefreshStats(RunStatus status)
    {
        if (_metrics.Count == 0) return;

        var filtered = string.IsNullOrEmpty(_selectedPolicyGroupFilter)
            ? _metrics
            : _metrics.Where(m => m.PolicyGroup == _selectedPolicyGroupFilter).ToList();

        if (filtered.Count == 0) return;

        var trainMetrics = filtered.Where(m => !m.IsEval).ToList();
        var evalMetrics = filtered.Where(m => m.IsEval).ToList();

        if (trainMetrics.Count == 0 && evalMetrics.Count == 0) return;

        var window = (trainMetrics.Count > 0 ? trainMetrics : filtered).TakeLast(50).ToList();
        var avg = window.Average(m => m.EpisodeReward);
        var best = (trainMetrics.Count > 0 ? trainMetrics : filtered).Max(m => m.EpisodeReward);
        var last = _metrics[^1];
        var steps = status.TotalSteps > 0 ? status.TotalSteps : last.TotalSteps;
        var eps = status.EpisodeCount > 0 ? status.EpisodeCount : last.EpisodeCount;

        if (_statAvgReward is not null) _statAvgReward.Text = avg.ToString("F3");
        if (_statBestReward is not null) _statBestReward.Text = best.ToString("F3");
        if (_statEvalReward is not null)
        {
            _statEvalReward.Text = evalMetrics.Count > 0
                ? evalMetrics[^1].EvalMeanReward.ToString("F3")
                : "—";
        }
        if (_statTotalSteps is not null) _statTotalSteps.Text = FormatSteps(steps);
        if (_statEpisodes is not null)
        {
            _statEpisodes.Text = status.WorkerEpisodeCount > 0
                ? $"{eps:N0} + {status.WorkerEpisodeCount:N0} simulated"
                : eps.ToString("N0");
        }

        if (!string.IsNullOrWhiteSpace(last.OpponentGroup))
        {
            var updateSuffix = last.OpponentUpdateCount.HasValue ? $" u{last.OpponentUpdateCount.Value}" : string.Empty;
            SetHeaderStatus(
                $"Latest matchup: {last.PolicyGroup} vs {last.OpponentGroup} ({last.OpponentSource}{updateSuffix})");
        }
        else if (!string.IsNullOrEmpty(status.ResumedFrom))
        {
            SetHeaderStatus($"Resumed from: {status.ResumedFrom}");
        }
    }

    private void SetHeaderStatus(string message)
    {
        if (_headerStatus is not null) _headerStatus.Text = message.Length <= 60 ? message : message[..60] + "…";
    }

    private void ShowLiveBadge()
    {
        if (_liveBadge is null) return;
        _liveBadge.Visible = true;
        _liveBadge.Modulate = Colors.White;
        _livePulseAccum = 0;
    }

    private void HideLiveBadge()
    {
        if (_liveBadge is null) return;
        _liveBadge.Visible = false;
        _liveBadge.Modulate = Colors.White;
        _livePulseAccum = 0;
    }

    // ── Run label helpers ─────────────────────────────────────────────────────

    /// <summary>
    /// Builds a human-readable dropdown label. If a display name is set, shows
    /// "display_name (prefix • Mar 14, 14:32)"; otherwise shows "prefix • Mar 14, 14:32".
    /// </summary>
    private static string BuildRunLabel(string runId, string displayName)
    {
        var timeLabel = ParseRunLabel(runId);
        return string.IsNullOrWhiteSpace(displayName)
            ? timeLabel
            : $"{displayName} ({timeLabel})";
    }

    /// <summary>
    /// Parses a RunId of the form "{prefix}_{unix_timestamp}" into a human-readable string.
    /// Falls back to the raw runId if parsing fails.
    /// </summary>
    private static string ParseRunLabel(string runId)
    {
        var lastUnderscore = runId.LastIndexOf('_');
        if (lastUnderscore <= 0 || lastUnderscore >= runId.Length - 1) return runId;

        var prefix = runId[..lastUnderscore];
        var timestampStr = runId[(lastUnderscore + 1)..];

        if (!double.TryParse(timestampStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var unixSeconds))
            return runId;

        var dto = DateTimeOffset.FromUnixTimeSeconds((long)unixSeconds).ToLocalTime();
        var today = DateTimeOffset.Now.Date;
        var time = dto.DateTime.Date == today
            ? $"Today, {dto:HH:mm}"
            : $"{dto:MMM d, HH:mm}";

        return $"{prefix} • {time}";
    }

    /// <summary>Extracts the prefix from a RunId (everything before the last '_').</summary>
    private static string ExtractRunPrefix(string runId)
    {
        var last = runId.LastIndexOf('_');
        return last > 0 ? runId[..last] : runId;
    }

    private static double GetRunSortValue(string runId, string runDirAbsPath)
    {
        var lastUnderscore = runId.LastIndexOf('_');
        if (lastUnderscore > 0 && lastUnderscore < runId.Length - 1)
        {
            var timestampStr = runId[(lastUnderscore + 1)..];
            if (double.TryParse(timestampStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var unixSeconds))
                return unixSeconds;
        }

        return System.IO.Directory.Exists(runDirAbsPath)
            ? System.IO.Directory.GetLastWriteTimeUtc(runDirAbsPath).Ticks
            : 0d;
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = System.IO.Path.GetInvalidFileNameChars();
        return new string(name.Select(c => Array.IndexOf(invalid, c) >= 0 ? '_' : c).ToArray());
    }

    // ── Variant helpers ───────────────────────────────────────────────────────
    private static float GetFloat(Godot.Collections.Dictionary d, string key)
    {
        if (!d.ContainsKey(key)) return 0f;
        var v = d[key];
        return v.VariantType == Variant.Type.Float ? v.AsSingle()
             : v.VariantType == Variant.Type.Int ? (float)v.AsInt64()
             : 0f;
    }

    private static long GetLong(Godot.Collections.Dictionary d, string key)
    {
        if (!d.ContainsKey(key)) return 0L;
        var v = d[key];
        return v.VariantType == Variant.Type.Int ? v.AsInt64()
             : v.VariantType == Variant.Type.Float ? (long)v.AsDouble()
             : 0L;
    }

    private static string GetString(Godot.Collections.Dictionary d, string key, string fallback)
    {
        if (!d.ContainsKey(key)) return fallback;
        var v = d[key];
        return v.VariantType == Variant.Type.String ? v.AsString() : fallback;
    }

    private static float GetFloat(JsonElement d, string key, float fallback = 0f)
    {
        if (!d.TryGetProperty(key, out var value))
            return fallback;

        return value.ValueKind switch
        {
            JsonValueKind.Number when value.TryGetDouble(out var n) => (float)n,
            JsonValueKind.String when float.TryParse(value.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var n) => n,
            JsonValueKind.True => 1f,
            JsonValueKind.False => 0f,
            _ => fallback,
        };
    }

    private static long GetLong(JsonElement d, string key, long fallback = 0)
    {
        if (!d.TryGetProperty(key, out var value))
            return fallback;

        return value.ValueKind switch
        {
            JsonValueKind.Number when value.TryGetInt64(out var n) => n,
            JsonValueKind.String when long.TryParse(value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) => n,
            JsonValueKind.True => 1L,
            JsonValueKind.False => 0L,
            _ => fallback,
        };
    }

    private static string GetString(JsonElement d, string key, string fallback)
    {
        if (!d.TryGetProperty(key, out var value))
            return fallback;

        return value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? fallback
            : fallback;
    }

    private static bool GetBool(JsonElement d, string key, bool fallback = false)
    {
        if (!d.TryGetProperty(key, out var value))
            return fallback;

        return value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Number when value.TryGetInt64(out var n) => n != 0,
            JsonValueKind.String when bool.TryParse(value.GetString(), out var parsed) => parsed,
            JsonValueKind.String when long.TryParse(value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) => n != 0,
            _ => fallback,
        };
    }

    private static string[] GetStringArray(JsonElement d, string key)
    {
        if (!d.TryGetProperty(key, out var value) || value.ValueKind != JsonValueKind.Array)
            return Array.Empty<string>();

        var items = new List<string>();
        foreach (var entry in value.EnumerateArray())
        {
            if (entry.ValueKind == JsonValueKind.String)
                items.Add(entry.GetString() ?? string.Empty);
        }

        return items.ToArray();
    }

    private static string FormatSteps(long n) =>
        n >= 1_000_000 ? $"{n / 1_000_000.0:F2}M"
        : n >= 1_000 ? $"{n / 1_000.0:F1}K"
        : n.ToString();

    // ── HPO dashboard panel ───────────────────────────────────────────────────

    private HPODashPanel BuildHpoSection()
    {
        _hpoPanel = new HPODashPanel();
        _hpoPanel.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        return _hpoPanel;
    }

    // ── Checkpoint history panel ──────────────────────────────────────────────

    private Control BuildCheckpointHistorySection()
    {
        var outer = new VBoxContainer();
        outer.AddThemeConstantOverride("separation", 4);

        var headerRow = new HBoxContainer();
        headerRow.AddThemeConstantOverride("separation", 6);

        _checkpointToggleBtn = new Button
        {
            Text = "▶  Checkpoint History",
            Flat = true,
            ToggleMode = true,
            ButtonPressed = false,
        };
        _checkpointToggleBtn.AddThemeFontSizeOverride("font_size", 13);
        _checkpointToggleBtn.Toggled += OnCheckpointPanelToggled;
        headerRow.AddChild(_checkpointToggleBtn);

        _checkpointStatusLabel = new Label { VerticalAlignment = VerticalAlignment.Center };
        _checkpointStatusLabel.AddThemeFontSizeOverride("font_size", 11);
        _checkpointStatusLabel.Modulate = new Color(0.6f, 0.6f, 0.6f);
        headerRow.AddChild(_checkpointStatusLabel);

        headerRow.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        var sortLabel = new Label { Text = "Sort:", VerticalAlignment = VerticalAlignment.Center };
        sortLabel.AddThemeFontSizeOverride("font_size", 11);
        sortLabel.Modulate = new Color(0.7f, 0.7f, 0.7f);
        headerRow.AddChild(sortLabel);

        _checkpointSortDropdown = new OptionButton
        {
            CustomMinimumSize = new Vector2(160, 0),
            TooltipText = "Sort checkpoints by a selected field",
        };
        _checkpointSortDropdown.AddItem("Newest Update");
        _checkpointSortDropdown.AddItem("Oldest Update");
        _checkpointSortDropdown.AddItem("Most Steps");
        _checkpointSortDropdown.AddItem("Most Episodes");
        _checkpointSortDropdown.AddItem("Best Reward");
        _checkpointSortDropdown.AddItem("Algorithm");
        _checkpointSortDropdown.AddItem("Type");
        _checkpointSortDropdown.ItemSelected += OnCheckpointSortSelected;
        headerRow.AddChild(_checkpointSortDropdown);

        outer.AddChild(headerRow);

        _checkpointHistoryPanel = new VBoxContainer();
        _checkpointHistoryPanel.Visible = false;
        _checkpointHistoryPanel.AddThemeConstantOverride("separation", 2);

        var scroll = new ScrollContainer();
        scroll.CustomMinimumSize = new Vector2(0, 180);
        scroll.SizeFlagsVertical = SizeFlags.ExpandFill;
        scroll.FollowFocus = false;

        _checkpointRowContainer = new VBoxContainer();
        _checkpointRowContainer.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        _checkpointRowContainer.AddThemeConstantOverride("separation", 1);
        scroll.AddChild(_checkpointRowContainer);

        _checkpointHistoryPanel.AddChild(scroll);
        outer.AddChild(_checkpointHistoryPanel);

        return outer;
    }

    private void OnCheckpointPanelToggled(bool pressed)
    {
        if (_checkpointToggleBtn is not null)
            _checkpointToggleBtn.Text = (pressed ? "▼" : "▶") + "  Checkpoint History";

        if (_checkpointHistoryPanel is not null)
            _checkpointHistoryPanel.Visible = pressed;

        if (pressed)
            RefreshCheckpointHistory();
    }

    private void RefreshCheckpointHistory()
    {
        if (_checkpointRowContainer is null || !(_checkpointHistoryPanel?.Visible ?? false))
            return;

        if (string.IsNullOrEmpty(_selectedRunId))
        {
            if (_checkpointStatusLabel is not null)
                _checkpointStatusLabel.Text = "(no run selected)";
            return;
        }

        var runDirAbs = ProjectSettings.GlobalizePath($"res://RL-Agent-Training/runs/{_selectedRunId}");
        _checkpointEntries = CheckpointRegistry.ListHistoryEntries(runDirAbs)
            .GroupBy(entry => entry.AbsolutePath, StringComparer.Ordinal)
            .Select(group => group.First())
            .ToList();

        RebuildCheckpointRows();

        if (_checkpointStatusLabel is not null)
        {
            _checkpointStatusLabel.Text = _checkpointEntries.Count == 0
                ? "(no history yet — snapshots appear after the first checkpoint interval)"
                : $"{_checkpointEntries.Count} snapshot(s)";
        }
    }

    private void RebuildCheckpointRows()
    {
        if (_checkpointRowContainer is null) return;

        foreach (var child in _checkpointRowContainer.GetChildren())
        {
            _checkpointRowContainer.RemoveChild(child);
            child.QueueFree();
        }

        if (_checkpointEntries.Count == 0) return;

        var byGroup = _checkpointEntries.GroupBy(e => e.PolicyGroupId).OrderBy(g => g.Key).ToList();
        var multiGroup = byGroup.Count > 1;

        foreach (var group in byGroup)
        {
            if (multiGroup)
            {
                var groupLabel = new Label { Text = $"  Policy group: {group.Key}" };
                groupLabel.AddThemeFontSizeOverride("font_size", 11);
                groupLabel.Modulate = new Color(0.75f, 0.85f, 1.0f);
                _checkpointRowContainer.AddChild(groupLabel);
            }

            _checkpointRowContainer.AddChild(BuildCheckpointColumnHeader());

            foreach (var entry in SortCheckpointEntries(group))
                _checkpointRowContainer.AddChild(BuildCheckpointRow(entry));

            _checkpointRowContainer.AddChild(new HSeparator());
        }
    }

    private static Control BuildCheckpointColumnHeader()
    {
        var row = new HBoxContainer();
        row.AddThemeConstantOverride("separation", 4);

        void AddCol(string text, int minW)
        {
            var lbl = new Label { Text = text, CustomMinimumSize = new Vector2(minW, 0) };
            lbl.AddThemeFontSizeOverride("font_size", 10);
            lbl.Modulate = new Color(0.5f, 0.5f, 0.5f);
            row.AddChild(lbl);
        }

        AddCol("Update", 64);
        AddCol("Steps", 72);
        AddCol("Episodes", 72);
        AddCol("Algo", 48);
        AddCol("Reward", 70);
        AddCol("Type", 70);
        row.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });
        return row;
    }

    private Control BuildCheckpointRow(CheckpointHistoryEntry entry)
    {
        var row = new HBoxContainer();
        row.AddThemeConstantOverride("separation", 4);

        void AddCol(string text, int minW)
        {
            var lbl = new Label { Text = text, CustomMinimumSize = new Vector2(minW, 0) };
            lbl.AddThemeFontSizeOverride("font_size", 11);
            lbl.VerticalAlignment = VerticalAlignment.Center;
            row.AddChild(lbl);
        }

        AddCol($"u{entry.UpdateCount:D6}", 64);
        AddCol(FormatSteps(entry.TotalSteps), 72);
        AddCol(entry.EpisodeCount.ToString("N0"), 72);
        AddCol(entry.Algorithm, 48);
        AddCol(entry.RewardSnapshot == 0f ? "—" : entry.RewardSnapshot.ToString("F3"), 70);

        var typeLabel = new Label
        {
            Text = entry.IsSelfPlayFrozen ? "self-play" : "history",
            CustomMinimumSize = new Vector2(70, 0),
        };
        typeLabel.AddThemeFontSizeOverride("font_size", 10);
        typeLabel.Modulate = entry.IsSelfPlayFrozen ? new Color(0.85f, 0.65f, 0.25f) : new Color(0.55f, 0.85f, 0.55f);
        typeLabel.VerticalAlignment = VerticalAlignment.Center;
        row.AddChild(typeLabel);

        row.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        var copyBtn = new Button
        {
            Text = "Copy Path",
            TooltipText = $"Copy path to clipboard for use in ResumeCheckpointPath:\n{entry.AbsolutePath}",
            CustomMinimumSize = new Vector2(80, 0),
        };
        copyBtn.Pressed += () =>
        {
            DisplayServer.ClipboardSet(entry.AbsolutePath);
            copyBtn.Text = "Copied!";
            var timer = GetTree().CreateTimer(1.5);
            timer.Timeout += () => { if (IsInstanceValid(copyBtn)) copyBtn.Text = "Copy Path"; };
        };
        row.AddChild(copyBtn);

        var exportBtn = new Button
        {
            Text = "Export",
            TooltipText = $"Export as .rlmodel: {System.IO.Path.GetFileName(entry.AbsolutePath)}",
            CustomMinimumSize = new Vector2(64, 0),
        };
        exportBtn.Pressed += () => OnExportCheckpointPressed(entry);
        row.AddChild(exportBtn);

        return row;
    }

    private void OnCheckpointSortSelected(long index)
    {
        if (index < 0)
            return;

        _checkpointSortMode = (CheckpointSortMode)index;
        RebuildCheckpointRows();
    }

    private IEnumerable<CheckpointHistoryEntry> SortCheckpointEntries(IEnumerable<CheckpointHistoryEntry> entries)
    {
        return _checkpointSortMode switch
        {
            CheckpointSortMode.OldestUpdate => entries
                .OrderBy(e => e.UpdateCount)
                .ThenBy(e => e.TotalSteps)
                .ThenBy(e => e.AbsolutePath, StringComparer.Ordinal),
            CheckpointSortMode.MostSteps => entries
                .OrderByDescending(e => e.TotalSteps)
                .ThenByDescending(e => e.UpdateCount)
                .ThenBy(e => e.AbsolutePath, StringComparer.Ordinal),
            CheckpointSortMode.MostEpisodes => entries
                .OrderByDescending(e => e.EpisodeCount)
                .ThenByDescending(e => e.UpdateCount)
                .ThenBy(e => e.AbsolutePath, StringComparer.Ordinal),
            CheckpointSortMode.BestReward => entries
                .OrderByDescending(e => e.RewardSnapshot)
                .ThenByDescending(e => e.UpdateCount)
                .ThenBy(e => e.AbsolutePath, StringComparer.Ordinal),
            CheckpointSortMode.Algorithm => entries
                .OrderBy(e => e.Algorithm, StringComparer.OrdinalIgnoreCase)
                .ThenByDescending(e => e.UpdateCount)
                .ThenBy(e => e.AbsolutePath, StringComparer.Ordinal),
            CheckpointSortMode.Type => entries
                .OrderByDescending(e => e.IsSelfPlayFrozen)
                .ThenByDescending(e => e.UpdateCount)
                .ThenBy(e => e.AbsolutePath, StringComparer.Ordinal),
            _ => entries
                .OrderByDescending(e => e.UpdateCount)
                .ThenByDescending(e => e.TotalSteps)
                .ThenBy(e => e.AbsolutePath, StringComparer.Ordinal),
        };
    }

    private void OnExportCheckpointPressed(CheckpointHistoryEntry entry)
    {
        if (!System.IO.File.Exists(entry.AbsolutePath))
        {
            SetHeaderStatus($"Checkpoint not found: {entry.AbsolutePath}");
            return;
        }

        _pendingExportCheckpointEntry = entry;
        EnsureCheckpointExportDialog();
        _checkpointExportDialog!.PopupCentered(new Vector2I(700, 450));
    }

    private void EnsureCheckpointExportDialog()
    {
        if (_checkpointExportDialog is not null && IsInstanceValid(_checkpointExportDialog)) return;

        _checkpointExportDialog = new FileDialog
        {
            FileMode = FileDialog.FileModeEnum.OpenDir,
            Access = FileDialog.AccessEnum.Filesystem,
            Title = "Select Export Folder",
        };
        _checkpointExportDialog.DirSelected += OnCheckpointExportDirSelected;
        AddChild(_checkpointExportDialog);
    }

    private void OnCheckpointExportDirSelected(string dir)
    {
        if (_pendingExportCheckpointEntry is null)
            return;

        var checkpointPath = _pendingExportCheckpointEntry.AbsolutePath;
        var exportName = BuildCheckpointExportName(_pendingExportCheckpointEntry);
        var destPath = System.IO.Path.Combine(dir, $"{SanitizeFileName(exportName)}.rlmodel");
        var result = RLModelExporter.Export(checkpointPath, destPath);
        SetHeaderStatus(result == Error.Ok
            ? $"Exported → {destPath}"
            : $"Export failed for {System.IO.Path.GetFileName(checkpointPath)}");
        _pendingExportCheckpointEntry = null;
    }

    private string BuildCheckpointExportName(CheckpointHistoryEntry entry)
    {
        var policyName = entry.PolicyGroupId;
        for (var i = 0; i < _selectedRunMeta.AgentGroups.Length; i++)
        {
            if (i < _selectedRunMeta.AgentNames.Length
                && string.Equals(_selectedRunMeta.AgentGroups[i], entry.PolicyGroupId, StringComparison.Ordinal)
                && !string.IsNullOrWhiteSpace(_selectedRunMeta.AgentNames[i]))
            {
                policyName = _selectedRunMeta.AgentNames[i];
                break;
            }
        }

        var suffix = entry.IsSelfPlayFrozen ? $"selfplay_u{entry.UpdateCount:D6}" : $"u{entry.UpdateCount:D6}";
        return $"{policyName}_{suffix}";
    }
}
