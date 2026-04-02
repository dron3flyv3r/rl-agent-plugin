using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Collapsible HPO Study section for RLDashboard.
/// Polls study_state.json files from res://RL-Agent-Training/hpo/ and renders
/// five visualisations: summary bar, trial history table, objective progression
/// chart, parameter importance bars, parallel coordinates, and scatter grid.
/// </summary>
[Tool]
public partial class HPODashPanel : VBoxContainer
{
    // ── Data model ───────────────────────────────────────────────────────────
    private sealed class HpoTrialData
    {
        public int    TrialId;
        public string State      = "Pending";
        public float? Objective;
        public long   TotalSteps;
        public Dictionary<string, float> Params = new();
    }

    private sealed class HpoStudyData
    {
        public string StudyName   = "";
        public string OwnerRunId  = "";
        public string Direction   = "Maximize";
        public List<HpoTrialData> Trials = new();
        public int?   BestTrialId;
        public float? BestObjective;
        public string FilePath    = "";
    }

    // ── UI refs ──────────────────────────────────────────────────────────────
    private Button?           _toggleBtn;
    private Control?          _contentPanel;
    private OptionButton?     _studyDropdown;
    private Label?            _summaryStats;
    private VBoxContainer?    _trialRowContainer;
    private LineChartPanel?   _objectiveChart;
    private HpoImportancePanel?     _importancePanel;
    private HpoParallelCoordsPanel? _parallelCoordsPanel;
    private HpoScatterGrid?         _scatterGrid;

    // ── State ────────────────────────────────────────────────────────────────
    private List<HpoStudyData> _studies      = new();
    private HpoStudyData?      _selectedStudy;
    private int                _selectedStudyIndex = -1;
    public string RunIdFilter { get; set; } = string.Empty;

    // ── Godot lifecycle ──────────────────────────────────────────────────────
    public override void _Ready()
    {
        AddThemeConstantOverride("separation", 4);
        BuildUi();
    }

    /// <summary>Called from RLDashboard.PollUpdate() every 2 seconds.</summary>
    public void PollUpdate()
    {
        LoadAllStudies();
        RebuildStudyDropdown();
        RefreshSelectedStudy();

        if (!(_contentPanel?.Visible ?? false)) return;

        RefreshTrialTable();
        RefreshObjectiveChart();

        if (_importancePanel is not null)
        {
            _importancePanel.Study = _selectedStudy;
            _importancePanel.QueueRedraw();
        }
        if (_parallelCoordsPanel is not null)
        {
            _parallelCoordsPanel.Study = _selectedStudy;
            _parallelCoordsPanel.QueueRedraw();
        }
        if (_scatterGrid is not null)
        {
            _scatterGrid.Study = _selectedStudy;
            _scatterGrid.QueueRedraw();
        }
    }

    // ── UI construction ───────────────────────────────────────────────────────
    private void BuildUi()
    {
        // Toggle header row
        var headerRow = new HBoxContainer();
        headerRow.AddThemeConstantOverride("separation", 6);

        _toggleBtn = new Button
        {
            Text       = "▶  HPO Study",
            Flat       = true,
            ToggleMode = true,
            ButtonPressed = false,
        };
        _toggleBtn.AddThemeFontSizeOverride("font_size", 13);
        _toggleBtn.Toggled += OnToggled;
        headerRow.AddChild(_toggleBtn);

        AddChild(headerRow);

        // Collapsible content
        _contentPanel = new VBoxContainer();
        _contentPanel.Visible = false;
        (_contentPanel as VBoxContainer)!.AddThemeConstantOverride("separation", 6);

        _contentPanel.AddChild(BuildStudySelectorRow());
        _contentPanel.AddChild(new HSeparator());
        _contentPanel.AddChild(BuildTrialHistorySection());

        _objectiveChart = new LineChartPanel
        {
            ChartTitle            = "Objective Progression",
            CustomMinimumSize     = new Vector2(0, 160),
            SizeFlagsHorizontal   = SizeFlags.ExpandFill,
            ShowSmoothed          = false,
        };
        _contentPanel.AddChild(_objectiveChart);

        var sideBySide = new HBoxContainer();
        sideBySide.AddThemeConstantOverride("separation", 6);
        sideBySide.CustomMinimumSize = new Vector2(0, 220);

        _importancePanel = new HpoImportancePanel
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical   = SizeFlags.ExpandFill,
        };
        sideBySide.AddChild(_importancePanel);

        _parallelCoordsPanel = new HpoParallelCoordsPanel
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical   = SizeFlags.ExpandFill,
        };
        sideBySide.AddChild(_parallelCoordsPanel);

        _contentPanel.AddChild(sideBySide);

        _scatterGrid = new HpoScatterGrid
        {
            CustomMinimumSize   = new Vector2(0, 260),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _contentPanel.AddChild(_scatterGrid);

        AddChild(_contentPanel);
    }

    private Control BuildStudySelectorRow()
    {
        var hbox = new HBoxContainer();
        hbox.AddThemeConstantOverride("separation", 10);

        var lbl = new Label { Text = "Study:", VerticalAlignment = VerticalAlignment.Center };
        lbl.AddThemeFontSizeOverride("font_size", 12);
        hbox.AddChild(lbl);

        _studyDropdown = new OptionButton { CustomMinimumSize = new Vector2(160, 0) };
        _studyDropdown.ItemSelected += i =>
        {
            _selectedStudyIndex = (int)i;
            RefreshSelectedStudy();
            RefreshTrialTable();
            RefreshObjectiveChart();
            if (_importancePanel     is not null) { _importancePanel.Study     = _selectedStudy; _importancePanel.QueueRedraw();     }
            if (_parallelCoordsPanel is not null) { _parallelCoordsPanel.Study = _selectedStudy; _parallelCoordsPanel.QueueRedraw(); }
            if (_scatterGrid         is not null) { _scatterGrid.Study         = _selectedStudy; _scatterGrid.QueueRedraw();         }
        };
        hbox.AddChild(_studyDropdown);

        _summaryStats = new Label { VerticalAlignment = VerticalAlignment.Center };
        _summaryStats.AddThemeFontSizeOverride("font_size", 11);
        _summaryStats.Modulate = new Color(0.55f, 0.55f, 0.55f);
        hbox.AddChild(_summaryStats);

        return hbox;
    }

    private Control BuildTrialHistorySection()
    {
        var scroll = new ScrollContainer();
        scroll.CustomMinimumSize    = new Vector2(0, 200);
        scroll.SizeFlagsHorizontal  = SizeFlags.ExpandFill;
        scroll.FollowFocus          = false;

        _trialRowContainer = new VBoxContainer();
        _trialRowContainer.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        _trialRowContainer.AddThemeConstantOverride("separation", 1);
        scroll.AddChild(_trialRowContainer);

        AddTrialTableHeader();

        return scroll;
    }

    // ── Toggle ────────────────────────────────────────────────────────────────
    private void OnToggled(bool pressed)
    {
        if (_toggleBtn is not null)
            _toggleBtn.Text = (pressed ? "▼" : "▶") + "  HPO Study";
        if (_contentPanel is not null)
            _contentPanel.Visible = pressed;
        if (pressed)
            PollUpdate();
    }

    // ── Data loading ──────────────────────────────────────────────────────────
    private void LoadAllStudies()
    {
        var hpoDir = ProjectSettings.GlobalizePath("res://RL-Agent-Training/hpo");
        if (!Directory.Exists(hpoDir)) { _studies.Clear(); return; }

        var newStudies = new List<HpoStudyData>();
        foreach (var stateFile in Directory.GetFiles(hpoDir, "study_state.json", SearchOption.AllDirectories))
        {
            var study = ParseStudyState(stateFile);
            if (study is not null
                && (string.IsNullOrWhiteSpace(RunIdFilter)
                    || string.Equals(study.OwnerRunId, RunIdFilter, StringComparison.Ordinal)))
                newStudies.Add(study);
        }
        _studies = newStudies;
    }

    private static HpoStudyData? ParseStudyState(string absPath)
    {
        try
        {
            var content = File.ReadAllText(absPath);
            var variant = StudyState.ParseSanitizedJson(content);
            if (variant.VariantType != Variant.Type.Dictionary) return null;

            var d = variant.AsGodotDictionary();
            var study = new HpoStudyData
            {
                StudyName = d.ContainsKey("study_name") ? d["study_name"].ToString() : "",
                OwnerRunId = d.ContainsKey("owner_run_id") ? d["owner_run_id"].ToString() : InferRunIdFromPath(absPath),
                Direction  = d.ContainsKey("direction")  ? d["direction"].ToString()  : "Maximize",
                FilePath   = absPath,
            };

            if (d.ContainsKey("best_trial_id") && d["best_trial_id"].VariantType == Variant.Type.Int)
                study.BestTrialId = (int)d["best_trial_id"];
            if (d.ContainsKey("best_objective") &&
                d["best_objective"].VariantType is Variant.Type.Float or Variant.Type.Int)
                study.BestObjective = (float)(double)d["best_objective"];

            if (d.ContainsKey("trials") && d["trials"].VariantType == Variant.Type.Array)
            {
                foreach (var item in d["trials"].AsGodotArray())
                {
                    if (item.VariantType != Variant.Type.Dictionary) continue;
                    var td = item.AsGodotDictionary();

                    var trial = new HpoTrialData
                    {
                        TrialId    = td.ContainsKey("trial_id")    ? (int)td["trial_id"]      : 0,
                        State      = td.ContainsKey("state")       ? td["state"].ToString()    : "Pending",
                        TotalSteps = td.ContainsKey("total_steps") ? DictGetLong(td, "total_steps") : 0L,
                    };

                    if (td.ContainsKey("objective") &&
                        td["objective"].VariantType is Variant.Type.Float or Variant.Type.Int)
                        trial.Objective = (float)(double)td["objective"];

                    if (td.ContainsKey("params") && td["params"].VariantType == Variant.Type.Dictionary)
                    {
                        var paramsDict = td["params"].AsGodotDictionary();
                        foreach (var pk in paramsDict.Keys)
                        {
                            var pv = paramsDict[pk];
                            if (pv.VariantType is Variant.Type.Float or Variant.Type.Int)
                                trial.Params[pk.ToString()] = (float)(double)pv;
                        }
                    }

                    study.Trials.Add(trial);
                }
            }

            return study;
        }
        catch (Exception ex)
        {
            GD.Print($"[HPO] Failed to parse study state '{absPath}': {ex}");
            return null;
        }
    }

    private static string InferRunIdFromPath(string absPath)
    {
        var normalized = absPath.Replace('\\', '/');
        var marker = "/RL-Agent-Training/hpo/";
        var idx = normalized.IndexOf(marker, StringComparison.Ordinal);
        if (idx < 0)
            return string.Empty;

        var after = normalized[(idx + marker.Length)..];
        var slash = after.IndexOf('/');
        return slash > 0 ? after[..slash] : string.Empty;
    }

    private static long DictGetLong(Godot.Collections.Dictionary d, string key)
    {
        if (!d.ContainsKey(key)) return 0L;
        var v = d[key];
        return v.VariantType == Variant.Type.Int   ? v.AsInt64()
             : v.VariantType == Variant.Type.Float ? (long)v.AsDouble()
             : 0L;
    }

    // ── Study selector ────────────────────────────────────────────────────────
    private void RebuildStudyDropdown()
    {
        if (_studyDropdown is null) return;

        string? prevName = _selectedStudy?.StudyName;
        _studyDropdown.Clear();

        foreach (var s in _studies)
            _studyDropdown.AddItem(s.StudyName);

        if (_studies.Count == 0) { _selectedStudyIndex = -1; _selectedStudy = null; return; }

        int newIdx = prevName is not null ? _studies.FindIndex(s => s.StudyName == prevName) : -1;
        if (newIdx < 0) newIdx = 0;

        _selectedStudyIndex = newIdx;
        _studyDropdown.Selected = newIdx;
    }

    private void RefreshSelectedStudy()
    {
        _selectedStudy = _selectedStudyIndex >= 0 && _selectedStudyIndex < _studies.Count
            ? _studies[_selectedStudyIndex]
            : null;

        if (_summaryStats is null) return;
        if (_selectedStudy is null) { _summaryStats.Text = "No HPO studies found"; return; }

        var s = _selectedStudy;
        int complete = s.Trials.Count(t => t.State == "Complete");
        int pruned   = s.Trials.Count(t => t.State == "Pruned");
        int running  = s.Trials.Count(t => t.State == "Running");
        int total    = s.Trials.Count;

        string best = s.BestObjective.HasValue
            ? $"{s.BestObjective.Value:F3} (trial {s.BestTrialId})"
            : "—";

        _summaryStats.Text =
            $"Trials: {complete} complete  {pruned} pruned  {running} running  {total} total   " +
            $"Best: {best}   Direction: {s.Direction}";
    }

    // ── Trial history table ───────────────────────────────────────────────────
    private void AddTrialTableHeader()
    {
        if (_trialRowContainer is null) return;

        var hbox = new HBoxContainer();
        hbox.AddThemeConstantOverride("separation", 4);

        void AddHeaderCell(string text, int minW)
        {
            var lbl = new Label { Text = text };
            lbl.AddThemeFontSizeOverride("font_size", 10);
            lbl.Modulate = new Color(0.55f, 0.55f, 0.55f);
            lbl.CustomMinimumSize = new Vector2(minW, 0);
            hbox.AddChild(lbl);
        }

        AddHeaderCell("#",          30);
        AddHeaderCell("State",      90);
        AddHeaderCell("Objective",  80);
        AddHeaderCell("Steps",      60);
        AddHeaderCell("Parameters",  0);

        var fill = new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        hbox.AddChild(fill);

        _trialRowContainer.AddChild(hbox);
    }

    private void RefreshTrialTable()
    {
        if (_trialRowContainer is null) return;

        // Keep header (child 0), remove data rows
        for (int i = _trialRowContainer.GetChildCount() - 1; i >= 1; i--)
        {
            var child = _trialRowContainer.GetChild(i);
            _trialRowContainer.RemoveChild(child);
            child.QueueFree();
        }

        if (_selectedStudy is null) return;

        foreach (var trial in _selectedStudy.Trials)
        {
            bool isBest = _selectedStudy.BestTrialId.HasValue
                       && trial.TrialId == _selectedStudy.BestTrialId.Value;

            var row = new HBoxContainer();
            row.AddThemeConstantOverride("separation", 4);
            if (isBest) row.Modulate = new Color(1.0f, 0.95f, 0.70f);

            void AddCell(string text, int minW, Color? col = null)
            {
                var lbl = new Label { Text = text };
                lbl.AddThemeFontSizeOverride("font_size", 11);
                lbl.CustomMinimumSize = new Vector2(minW, 0);
                if (col.HasValue) lbl.Modulate = col.Value;
                row.AddChild(lbl);
            }

            AddCell(trial.TrialId.ToString(), 30);

            Color stateColor = trial.State switch
            {
                "Complete" => new Color(0.22f, 0.82f, 0.42f),
                "Pruned"   => new Color(0.92f, 0.62f, 0.22f),
                "Failed"   => new Color(0.92f, 0.28f, 0.28f),
                "Running"  => new Color(0.35f, 0.62f, 0.92f),
                _          => new Color(0.55f, 0.55f, 0.55f),
            };
            string stateText = trial.State switch
            {
                "Complete" => "✓ Complete",
                "Pruned"   => "✗ Pruned",
                "Failed"   => "✗ Failed",
                "Running"  => "⟳ Running",
                _          => "— Pending",
            };
            AddCell(stateText, 90, stateColor);
            AddCell(trial.Objective.HasValue ? trial.Objective.Value.ToString("F3") : "—", 80);
            AddCell(FormatSteps(trial.TotalSteps), 60);

            var paramStr = string.Join("  ",
                trial.Params.Select(kv => $"{kv.Key}={FormatParam(kv.Value)}"));
            AddCell(paramStr, 0);

            _trialRowContainer.AddChild(row);
        }
    }

    // ── Objective chart ───────────────────────────────────────────────────────
    private void RefreshObjectiveChart()
    {
        if (_objectiveChart is null) return;
        if (_selectedStudy is null) { _objectiveChart.ClearSeries(); return; }

        var completed = _selectedStudy.Trials
            .Where(t => t.State == "Complete" && t.Objective.HasValue)
            .OrderBy(t => t.TrialId)
            .ToList();

        if (completed.Count == 0) { _objectiveChart.ClearSeries(); return; }

        var objectives = completed.Select(t => t.Objective!.Value).ToList();

        bool maximize = _selectedStudy.Direction != "Minimize";
        var bestSoFar = new List<float>(objectives.Count);
        float best = maximize ? float.MinValue : float.MaxValue;
        foreach (var v in objectives)
        {
            best = maximize ? Math.Max(best, v) : Math.Min(best, v);
            bestSoFar.Add(best);
        }

        _objectiveChart.UpdateSeries("Objective",   new Color(0.35f, 0.62f, 0.92f), objectives);
        _objectiveChart.UpdateSeries("Best so far", new Color(0.22f, 0.82f, 0.42f), bestSoFar);
    }

    // ── Shared helpers ────────────────────────────────────────────────────────
    private static string FormatSteps(long n) =>
        n >= 1_000_000 ? $"{n / 1_000_000.0:F2}M"
        : n >= 1_000   ? $"{n / 1_000.0:F1}K"
        : n.ToString();

    internal static string FormatParam(float v)
    {
        float abs = MathF.Abs(v);
        if (abs == 0f) return "0";
        if (abs < 0.001f || abs >= 10_000f) return v.ToString("G3");
        if (abs < 0.01f)  return v.ToString("F4");
        if (abs < 1f)     return v.ToString("F3");
        return v.ToString("F2");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INNER PANEL — Parameter Importance (Spearman ρ)
    // ─────────────────────────────────────────────────────────────────────────
    private sealed partial class HpoImportancePanel : Control
    {
        public HpoStudyData? Study;

        private static readonly Color CBg    = new(0.12f, 0.12f, 0.12f);
        private static readonly Color CBord  = new(0.28f, 0.28f, 0.28f);
        private static readonly Color CTitle = new(0.88f, 0.88f, 0.88f);
        private static readonly Color CDim   = new(0.48f, 0.48f, 0.48f);
        private static readonly Color CBar   = new(0.22f, 0.82f, 0.42f);

        public override void _Ready() => MouseFilter = MouseFilterEnum.Ignore;

        public override void _Draw()
        {
            var size = Size;
            if (size.X < 10 || size.Y < 10) return;

            DrawRect(new Rect2(Vector2.Zero, size), CBg,   filled: true);
            DrawRect(new Rect2(Vector2.Zero, size), CBord, filled: false, width: 1f);

            var font = GetThemeDefaultFont();
            int  fs  = Mathf.Clamp((int)GetThemeDefaultFontSize(), 8, 14);

            DrawString(font, new Vector2(8f, fs + 6f),
                "Parameter Importance (Spearman ρ)",
                HorizontalAlignment.Left, -1, fs, CTitle);

            if (Study is null)
            {
                DrawString(font, new Vector2(8f, size.Y * 0.5f), "No study selected",
                    HorizontalAlignment.Left, -1, fs - 1, CDim);
                return;
            }

            var completed = Study.Trials
                .Where(t => t.State == "Complete" && t.Objective.HasValue)
                .ToList();

            if (completed.Count < 5)
            {
                DrawString(font, new Vector2(8f, size.Y * 0.5f),
                    $"Not enough data ({completed.Count}/5 complete trials needed)",
                    HorizontalAlignment.Left, -1, fs - 1, CDim);
                return;
            }

            var paramNames = completed
                .SelectMany(t => t.Params.Keys).Distinct().OrderBy(k => k).ToList();
            if (paramNames.Count == 0) return;

            float[] objVals  = completed.Select(t => t.Objective!.Value).ToArray();
            float[] objRanks = Rank(objVals);

            var importances = new List<(string Name, float Rho)>();
            foreach (var name in paramNames)
            {
                float[] vals = completed
                    .Select(t => t.Params.TryGetValue(name, out var v) ? v : 0f).ToArray();
                float[] pRanks = Rank(vals);

                int    n     = vals.Length;
                double sumD2 = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = pRanks[i] - objRanks[i];
                    sumD2 += d * d;
                }
                float rho = n <= 1 ? 0f
                    : (float)(1.0 - 6.0 * sumD2 / ((double)n * ((long)n * n - 1)));
                importances.Add((name, rho));
            }
            importances.Sort((a, b) => MathF.Abs(b.Rho).CompareTo(MathF.Abs(a.Rho)));

            float titleH  = fs + 14f;
            float availH  = size.Y - titleH - 8f;
            float rowH    = Math.Min(availH / importances.Count, 26f);
            float labelW  = 120f;
            float barMaxW = size.X - labelW - 52f - 16f;

            for (int i = 0; i < importances.Count; i++)
            {
                var   (name, rho) = importances[i];
                float absRho = MathF.Abs(rho);
                float cy     = titleH + i * rowH + rowH * 0.5f;

                DrawString(font,
                    new Vector2(8f, cy + fs * 0.38f),
                    TruncateName(name, font, fs - 1, labelW - 4f),
                    HorizontalAlignment.Left, labelW, fs - 1, CTitle);

                float barW = absRho * barMaxW;
                if (barW > 1f)
                {
                    var barColor = new Color(CBar.R, CBar.G, CBar.B, 0.25f + absRho * 0.75f);
                    DrawRect(new Rect2(8f + labelW, cy - rowH * 0.28f, barW, rowH * 0.55f),
                        barColor, filled: true);
                }

                DrawString(font,
                    new Vector2(8f + labelW + barMaxW + 4f, cy + fs * 0.38f),
                    rho.ToString("F2"),
                    HorizontalAlignment.Left, 48f, fs - 1,
                    absRho > 0.5f ? CBar : CDim);
            }
        }

        private static float[] Rank(float[] vals)
        {
            int n   = vals.Length;
            var idx = vals.Select((v, i) => (v, i)).OrderBy(x => x.v).ToArray();
            var r   = new float[n];
            int j   = 0;
            while (j < n)
            {
                int k = j;
                while (k < n - 1 && idx[k].v == idx[k + 1].v) k++;
                float avg = (j + k) * 0.5f;
                for (int m = j; m <= k; m++) r[idx[m].i] = avg;
                j = k + 1;
            }
            return r;
        }

        private static string TruncateName(string name, Font font, int fs, float maxW)
        {
            while (name.Length > 3)
            {
                if (font.GetStringSize(name, HorizontalAlignment.Left, -1, fs).X <= maxW)
                    return name;
                name = name[..^1];
            }
            return name.Length > 0 ? name + "…" : name;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INNER PANEL — Parallel Coordinates
    // ─────────────────────────────────────────────────────────────────────────
    private sealed partial class HpoParallelCoordsPanel : Control
    {
        public HpoStudyData? Study;

        private static readonly Color CBg    = new(0.12f, 0.12f, 0.12f);
        private static readonly Color CBord  = new(0.28f, 0.28f, 0.28f);
        private static readonly Color CTitle = new(0.88f, 0.88f, 0.88f);
        private static readonly Color CDim   = new(0.48f, 0.48f, 0.48f);
        private static readonly Color CGold  = new(1.00f, 0.85f, 0.20f);
        private static readonly Color CAxis  = new(0.38f, 0.38f, 0.38f);

        public override void _Ready()
        {
            MouseFilter = MouseFilterEnum.Stop;
            MouseEntered += QueueRedraw;
            MouseExited  += QueueRedraw;
        }

        public override void _GuiInput(InputEvent @event)
        {
            if (@event is InputEventMouseMotion) QueueRedraw();
        }

        public override void _Draw()
        {
            var size = Size;
            if (size.X < 10 || size.Y < 10) return;

            DrawRect(new Rect2(Vector2.Zero, size), CBg,   filled: true);
            DrawRect(new Rect2(Vector2.Zero, size), CBord, filled: false, width: 1f);

            var font = GetThemeDefaultFont();
            int  fs  = Mathf.Clamp((int)GetThemeDefaultFontSize(), 8, 14);

            DrawString(font, new Vector2(8f, fs + 6f), "Parallel Coordinates",
                HorizontalAlignment.Left, -1, fs, CTitle);

            if (Study is null)
            {
                DrawNoData(font, fs, size, "No study selected"); return;
            }

            var completed = Study.Trials
                .Where(t => t.State == "Complete" && t.Objective.HasValue)
                .OrderBy(t => t.TrialId).ToList();

            if (completed.Count == 0)
            {
                DrawNoData(font, fs, size, "No completed trials yet"); return;
            }

            var paramNames = completed
                .SelectMany(t => t.Params.Keys).Distinct().OrderBy(k => k).ToList();
            if (paramNames.Count == 0) return;

            const float PadL   = 24f;
            const float PadR   = 24f;
            float       titleH = fs + 14f;
            const float PadH   = 40f;
            float plotTop = titleH + PadH;
            float plotBot = size.Y - PadH;
            float plotW   = size.X - PadL - PadR;
            if (plotBot <= plotTop || plotW < 4f) return;

            int  nAxes      = paramNames.Count;
            var  ranges     = ComputeRanges(completed, paramNames);
            bool maximize   = Study.Direction != "Minimize";

            // Rank: worst (index 0) → best (index Count-1)
            var ranked = completed
                .OrderBy(t => maximize ? -t.Objective!.Value : t.Objective!.Value)
                .ToList();

            var mp         = GetLocalMousePosition();
            int hoveredIdx = -1;
            float minDist  = 18f;

            foreach (var trial in ranked)
            {
                var pts = BuildPolyline(trial, paramNames, ranges, PadL, plotW, plotTop, plotBot, nAxes);
                int origIdx = completed.IndexOf(trial);
                float d = PolylineDist(mp, pts);
                if (d < minDist)
                {
                    minDist = d;
                    hoveredIdx = origIdx;
                }
            }

            // Draw non-best trials worst→best
            for (int ri = 0; ri < ranked.Count; ri++)
            {
                var trial = ranked[ri];
                bool isBest = Study.BestTrialId.HasValue && trial.TrialId == Study.BestTrialId.Value;
                if (isBest) continue;

                float tRank = ranked.Count > 1 ? (float)ri / (ranked.Count - 1) : 1f;
                var   pts   = BuildPolyline(trial, paramNames, ranges, PadL, plotW, plotTop, plotBot, nAxes);
                int origIdx = completed.IndexOf(trial);
                bool hovered = origIdx == hoveredIdx;
                float alpha  = hovered ? 1.0f : 0.5f;
                float width  = hovered ? 2.5f : 1.2f;
                var   color  = new Color(1f - tRank * 0.8f, 0.2f + tRank * 0.7f, 0.2f, alpha);
                DrawPolyline(pts, color, width, antialiased: true);
            }

            // Running trials — dashed grey
            foreach (var trial in Study.Trials.Where(t => t.State == "Running"))
            {
                var pts = BuildPolyline(trial, paramNames, ranges, PadL, plotW, plotTop, plotBot, nAxes);
                DrawDashedPolyline(pts, new Color(0.5f, 0.5f, 0.5f, 0.55f), 1f);
            }

            // Best trial — gold, drawn last
            if (Study.BestTrialId.HasValue)
            {
                var best = completed.FirstOrDefault(t => t.TrialId == Study.BestTrialId.Value);
                if (best is not null)
                {
                    var pts = BuildPolyline(best, paramNames, ranges, PadL, plotW, plotTop, plotBot, nAxes);
                    var bestHovered = completed.IndexOf(best) == hoveredIdx;
                    DrawPolyline(pts, CGold, bestHovered ? 3.2f : 2.5f, antialiased: true);
                }
            }

            // Axes
            const float nameW  = 96f;
            const float valueW = 72f;
            bool isEnd(int ai) => ai == 0 || ai == nAxes - 1;
            for (int ai = 0; ai < nAxes; ai++)
            {
                float ax      = AxisX(ai, nAxes, PadL, plotW);
                var   name    = paramNames[ai];
                var   (mn, mx) = ranges[name];

                DrawLine(new Vector2(ax, plotTop), new Vector2(ax, plotBot), CAxis, 2f);

                if (isEnd(ai))
                {
                    // End labels are vertical and centered along the axis span.
                    float axisMidY = (plotTop + plotBot) * 0.5f;
                    float sideGap  = 6f;
                    float localY   = ai == 0
                        ? -((fs - 2) * 0.5f)
                        :  ((fs - 2) * 0.5f + sideGap);
                    DrawSetTransform(new Vector2(ax, axisMidY), -Mathf.Pi * 0.5f, Vector2.One);
                    DrawString(font, new Vector2(-nameW * 0.5f, localY), name,
                        HorizontalAlignment.Center, nameW, fs - 2, CDim);
                    DrawSetTransform(Vector2.Zero, 0f, Vector2.One);
                }
                else
                {
                    DrawString(font, new Vector2(ax - nameW * 0.5f, plotTop - fs - 10f), name,
                        HorizontalAlignment.Center, nameW, fs - 2, CDim);
                }

                // Max value sits just above plotTop; min value below plotBot
                float valueLabelX = Math.Clamp(ax - valueW * 0.5f, 0f, size.X - valueW);
                DrawString(font, new Vector2(valueLabelX, plotTop - 4f),     FormatParam(mx),
                    HorizontalAlignment.Center, valueW, fs - 3, CDim);
                DrawString(font, new Vector2(valueLabelX, plotBot + fs),     FormatParam(mn),
                    HorizontalAlignment.Center, valueW, fs - 3, CDim);
            }

            // Tooltip
            if (hoveredIdx >= 0 && hoveredIdx < completed.Count)
                DrawTrialTooltip(font, fs, completed[hoveredIdx], mp, size);
        }

        private void DrawNoData(Font font, int fs, Vector2 size, string msg) =>
            DrawString(font, new Vector2(8f, size.Y * 0.5f), msg,
                HorizontalAlignment.Left, -1, fs - 1, CDim);

        private static float AxisX(int ai, int n, float padL, float plotW) =>
            n <= 1 ? padL + plotW * 0.5f : padL + (float)ai / (n - 1) * plotW;

        private static Vector2[] BuildPolyline(
            HpoTrialData trial, List<string> names,
            Dictionary<string, (float Min, float Max)> ranges,
            float padL, float plotW, float top, float bot, int n)
        {
            var pts = new Vector2[names.Count];
            for (int ai = 0; ai < names.Count; ai++)
            {
                float ax  = AxisX(ai, n, padL, plotW);
                var   nm  = names[ai];
                float val = trial.Params.TryGetValue(nm, out var v) ? v : ranges[nm].Min;
                var   (mn, mx) = ranges[nm];
                float t   = (val - mn) / (mx - mn);
                pts[ai]   = new Vector2(ax, bot - t * (bot - top));
            }
            return pts;
        }

        private static Dictionary<string, (float Min, float Max)> ComputeRanges(
            List<HpoTrialData> trials, List<string> names)
        {
            var r = new Dictionary<string, (float, float)>();
            foreach (var name in names)
            {
                var vals = trials.Where(t => t.Params.ContainsKey(name))
                                 .Select(t => t.Params[name]).ToList();
                if (vals.Count == 0) { r[name] = (0f, 1f); continue; }
                float mn = vals.Min(), mx = vals.Max();
                if (MathF.Abs(mx - mn) < 1e-8f) { mn -= 0.5f; mx += 0.5f; }
                r[name] = (mn, mx);
            }
            return r;
        }

        private static float PolylineDist(Vector2 p, Vector2[] pts)
        {
            float minD = float.MaxValue;
            for (int i = 0; i < pts.Length - 1; i++)
            {
                var ab    = pts[i + 1] - pts[i];
                float len = ab.LengthSquared();
                float t   = len < 1e-6f ? 0f : Math.Clamp(ab.Dot(p - pts[i]) / len, 0f, 1f);
                minD = Math.Min(minD, p.DistanceTo(pts[i] + t * ab));
            }
            return minD;
        }

        private void DrawDashedPolyline(Vector2[] pts, Color color, float width)
        {
            for (int i = 0; i < pts.Length - 1; i++)
            {
                float segLen = pts[i].DistanceTo(pts[i + 1]);
                int   dashes = Math.Max(1, (int)(segLen / 6f));
                for (int d = 0; d < dashes; d += 2)
                {
                    float t0 = (float)d       / dashes;
                    float t1 = Math.Min(1f, (float)(d + 1) / dashes);
                    DrawLine(pts[i].Lerp(pts[i + 1], t0),
                             pts[i].Lerp(pts[i + 1], t1), color, width);
                }
            }
        }

        private void DrawTrialTooltip(Font font, int fs, HpoTrialData trial, Vector2 mp, Vector2 size)
        {
            var lines = new List<(string Text, Color Col)>
            {
                ($"Trial {trial.TrialId}", new Color(0.88f, 0.88f, 0.88f)),
                ($"Obj: {(trial.Objective.HasValue ? trial.Objective.Value.ToString("F4") : "—")}",
                 new Color(0.35f, 0.82f, 0.88f)),
            };
            foreach (var (k, v) in trial.Params)
                lines.Add(($"{k}: {FormatParam(v)}", new Color(0.70f, 0.70f, 0.70f)));

            DrawTooltipBox(font, fs, lines, mp, size);
        }

        private void DrawTooltipBox(Font font, int fs,
            List<(string Text, Color Col)> lines, Vector2 mp, Vector2 size)
        {
            float lineH = fs + 3f;
            float maxW  = lines.Max(l => font.GetStringSize(l.Text, HorizontalAlignment.Left, -1, fs - 1).X);
            const float Pad = 5f;
            float bw = maxW + Pad * 2f;
            float bh = lines.Count * lineH + Pad * 2f;
            float bx = Math.Min(mp.X + 12f, size.X - bw - 4f);
            float by = Math.Clamp(mp.Y - bh * 0.5f, 4f, size.Y - bh - 4f);

            DrawRect(new Rect2(bx, by, bw, bh), new Color(0.10f, 0.10f, 0.10f, 0.92f), filled: true);
            DrawRect(new Rect2(bx, by, bw, bh), new Color(0.40f, 0.40f, 0.40f, 0.75f),
                filled: false, width: 1f);
            for (int li = 0; li < lines.Count; li++)
                DrawString(font, new Vector2(bx + Pad, by + Pad + (li + 0.82f) * lineH),
                    lines[li].Text, HorizontalAlignment.Left, -1, fs - 1, lines[li].Col);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INNER PANEL — Scatter Grid
    // ─────────────────────────────────────────────────────────────────────────
    private sealed partial class HpoScatterGrid : Control
    {
        public HpoStudyData? Study;

        private static readonly Color CBg    = new(0.12f, 0.12f, 0.12f);
        private static readonly Color CBord  = new(0.28f, 0.28f, 0.28f);
        private static readonly Color CPlot  = new(0.085f, 0.085f, 0.085f);
        private static readonly Color CTitle = new(0.88f, 0.88f, 0.88f);
        private static readonly Color CDim   = new(0.48f, 0.48f, 0.48f);
        private static readonly Color CGold  = new(1.00f, 0.85f, 0.20f);

        public override void _Ready()
        {
            MouseFilter = MouseFilterEnum.Stop;
            MouseExited  += QueueRedraw;
        }

        public override void _GuiInput(InputEvent @event)
        {
            if (@event is InputEventMouseMotion) QueueRedraw();
        }

        public override void _Draw()
        {
            var size = Size;
            if (size.X < 10 || size.Y < 10) return;

            DrawRect(new Rect2(Vector2.Zero, size), CBg,   filled: true);
            DrawRect(new Rect2(Vector2.Zero, size), CBord, filled: false, width: 1f);

            var font = GetThemeDefaultFont();
            int  fs  = Mathf.Clamp((int)GetThemeDefaultFontSize(), 8, 14);

            string dirSuffix = Study?.Direction == "Minimize" ? "  (↓ Minimize)" :
                               Study?.Direction != null       ? "  (↑ Maximize)" : "";
            DrawString(font, new Vector2(8f, fs + 6f), $"Parameter vs Objective{dirSuffix}",
                HorizontalAlignment.Left, -1, fs, CTitle);

            if (Study is null)
            {
                DrawString(font, new Vector2(8f, size.Y * 0.5f), "No study selected",
                    HorizontalAlignment.Left, -1, fs - 1, CDim);
                return;
            }

            var completed = Study.Trials
                .Where(t => t.State == "Complete" && t.Objective.HasValue)
                .OrderBy(t => t.TrialId).ToList();

            if (completed.Count == 0)
            {
                DrawString(font, new Vector2(8f, size.Y * 0.5f), "No completed trials yet",
                    HorizontalAlignment.Left, -1, fs - 1, CDim);
                return;
            }

            var paramNames = completed
                .SelectMany(t => t.Params.Keys).Distinct().OrderBy(k => k).ToList();
            if (paramNames.Count == 0) return;

            int   cols    = Math.Min(paramNames.Count, 3);
            int   rows    = (int)Math.Ceiling((double)paramNames.Count / cols);
            float titleH  = fs + 12f;
            float cellW   = (size.X - 8f) / cols;
            float cellH   = (size.Y - titleH - 8f) / rows;

            float yMin = completed.Min(t => t.Objective!.Value);
            float yMax = completed.Max(t => t.Objective!.Value);
            if (MathF.Abs(yMax - yMin) < 1e-8f) { yMin -= 0.5f; yMax += 0.5f; }

            bool maximize = Study.Direction != "Minimize";
            var  ranked   = completed
                .OrderBy(t => maximize ? -t.Objective!.Value : t.Objective!.Value)
                .ToList();

            var  mp          = GetLocalMousePosition();
            int? tooltipTrial = null;
            float minHoverD   = 12f;

            for (int pi = 0; pi < paramNames.Count; pi++)
            {
                int   col      = pi % cols;
                int   row      = pi / cols;
                var   cellRect = new Rect2(8f + col * cellW, titleH + row * cellH,
                                           cellW - 4f, cellH - 4f);

                float padL = col == 0 ? 30f : 8f;
                const float padR = 6f, padT = 16f, padB = 18f;

                var plotRect = new Rect2(
                    cellRect.Position.X + padL,
                    cellRect.Position.Y + padT,
                    cellRect.Size.X - padL - padR,
                    cellRect.Size.Y - padT - padB);

                DrawRect(cellRect, CPlot,  filled: true);
                DrawRect(cellRect, CBord,  filled: false, width: 1f);
                if (plotRect.Size.X < 4f || plotRect.Size.Y < 4f) continue;

                // Horizontal grid lines at 25 / 50 / 75 %
                var gridCol = new Color(0.28f, 0.28f, 0.28f, 1f);
                for (int gi = 1; gi <= 3; gi++)
                {
                    float gy = plotRect.Position.Y + plotRect.Size.Y * (1f - gi * 0.25f);
                    DrawLine(new Vector2(plotRect.Position.X, gy),
                             new Vector2(plotRect.Position.X + plotRect.Size.X, gy),
                             gridCol, 1f);
                }

                var name = paramNames[pi];
                DrawString(font,
                    new Vector2(cellRect.Position.X,
                                cellRect.Position.Y + padT - 3f),
                    name, HorizontalAlignment.Center, cellRect.Size.X, fs - 2, CTitle);

                var xVals = completed.Where(t => t.Params.ContainsKey(name))
                                     .Select(t => t.Params[name]).ToList();
                if (xVals.Count == 0) continue;

                float xMin = xVals.Min(), xMax = xVals.Max();
                if (MathF.Abs(xMax - xMin) < 1e-8f) { xMin -= 0.5f; xMax += 0.5f; }

                // X-axis labels
                const float axisLabelW = 48f;
                DrawString(font,
                    new Vector2(plotRect.Position.X, plotRect.Position.Y + plotRect.Size.Y + 2f),
                    FormatParam(xMin), HorizontalAlignment.Left, axisLabelW, fs - 3, CDim);
                DrawString(font,
                    new Vector2(plotRect.Position.X + plotRect.Size.X - axisLabelW,
                                plotRect.Position.Y + plotRect.Size.Y + 2f),
                    FormatParam(xMax), HorizontalAlignment.Right, axisLabelW, fs - 3, CDim);

                // Y-axis labels — leftmost column only
                if (col == 0)
                {
                    DrawString(font,
                        new Vector2(cellRect.Position.X + 2f, plotRect.Position.Y + fs),
                        FormatParam(yMax), HorizontalAlignment.Left, padL - 4f, fs - 3, CDim);
                    DrawString(font,
                        new Vector2(cellRect.Position.X + 2f,
                                    plotRect.Position.Y + plotRect.Size.Y),
                        FormatParam(yMin), HorizontalAlignment.Left, padL - 4f, fs - 3, CDim);

                    // Rotated "Obj" axis title
                    float midY = plotRect.Position.Y + plotRect.Size.Y * 0.5f;
                    DrawSetTransform(new Vector2(cellRect.Position.X + 7f, midY),
                        -Mathf.Pi * 0.5f, Vector2.One);
                    DrawString(font, new Vector2(-20f, fs * 0.4f), "Obj",
                        HorizontalAlignment.Center, 40f, fs - 3, CDim);
                    DrawSetTransform(Vector2.Zero, 0f, Vector2.One);
                }

                // Dots
                foreach (var trial in completed)
                {
                    if (!trial.Params.TryGetValue(name, out float xVal)) continue;

                    float tx     = (xVal - xMin) / (xMax - xMin);
                    float ty     = (trial.Objective!.Value - yMin) / (yMax - yMin);
                    var   dotPos = new Vector2(
                        plotRect.Position.X + tx * plotRect.Size.X,
                        plotRect.Position.Y + plotRect.Size.Y * (1f - ty));

                    bool  isBest = Study.BestTrialId.HasValue && trial.TrialId == Study.BestTrialId.Value;
                    int   rank   = ranked.IndexOf(trial);
                    float tRank  = ranked.Count > 1 ? (float)rank / (ranked.Count - 1) : 1f;

                    Color dotColor = isBest
                        ? CGold
                        : new Color(1f - tRank * 0.8f, 0.2f + tRank * 0.7f, 0.2f, 0.85f);
                    float radius = isBest ? 6f : 4f;

                    DrawCircle(dotPos, radius, dotColor);

                    float d = mp.DistanceTo(dotPos);
                    if (d < radius + 4f && d < minHoverD)
                    {
                        minHoverD    = d;
                        tooltipTrial = completed.IndexOf(trial);
                        DrawCircle(dotPos, radius + 2.5f, new Color(1f, 1f, 1f, 0.28f));
                    }
                }

                // Linear trend line
                var tPts = completed
                    .Where(t => t.Params.ContainsKey(name))
                    .Select(t => (
                        x: (t.Params[name] - xMin) / (xMax - xMin),
                        y: (t.Objective!.Value - yMin) / (yMax - yMin)))
                    .ToList();
                if (tPts.Count >= 3)
                {
                    float tn = tPts.Count, sx = 0f, sy = 0f, sxx = 0f, sxy = 0f;
                    foreach (var p in tPts) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
                    float den    = tn * sxx - sx * sx;
                    float tSlope = MathF.Abs(den) > 1e-8f ? (tn * sxy - sx * sy) / den : 0f;
                    float tInt   = (sy - tSlope * sx) / tn;
                    var   tp0    = new Vector2(
                        plotRect.Position.X,
                        plotRect.Position.Y + plotRect.Size.Y * (1f - Math.Clamp(tInt, -0.05f, 1.05f)));
                    var   tp1    = new Vector2(
                        plotRect.Position.X + plotRect.Size.X,
                        plotRect.Position.Y + plotRect.Size.Y * (1f - Math.Clamp(tSlope + tInt, -0.05f, 1.05f)));
                    DrawLine(tp0, tp1, new Color(0.55f, 0.55f, 1.0f, 0.50f), 1.5f);
                }
            }

            // Tooltip
            if (tooltipTrial.HasValue)
            {
                var trial = completed[tooltipTrial.Value];
                DrawTrialTooltip(font, fs, trial, mp, size);
            }
        }

        private void DrawTrialTooltip(Font font, int fs, HpoTrialData trial, Vector2 mp, Vector2 size)
        {
            var lines = new List<(string Text, Color Col)>
            {
                ($"Trial {trial.TrialId}", new Color(0.88f, 0.88f, 0.88f)),
                ($"Obj: {(trial.Objective.HasValue ? trial.Objective.Value.ToString("F4") : "—")}",
                 new Color(0.35f, 0.82f, 0.88f)),
            };
            foreach (var (k, v) in trial.Params)
                lines.Add(($"{k}: {FormatParam(v)}", new Color(0.70f, 0.70f, 0.70f)));

            float lineH = fs + 3f;
            float maxW  = lines.Max(l => font.GetStringSize(l.Text, HorizontalAlignment.Left, -1, fs - 1).X);
            const float Pad = 5f;
            float bw = maxW + Pad * 2f;
            float bh = lines.Count * lineH + Pad * 2f;
            float bx = Math.Min(mp.X + 12f, size.X - bw - 4f);
            float by = Math.Clamp(mp.Y - bh * 0.5f, 4f, size.Y - bh - 4f);

            DrawRect(new Rect2(bx, by, bw, bh), new Color(0.10f, 0.10f, 0.10f, 0.92f), filled: true);
            DrawRect(new Rect2(bx, by, bw, bh), new Color(0.40f, 0.40f, 0.40f, 0.75f),
                filled: false, width: 1f);
            for (int li = 0; li < lines.Count; li++)
                DrawString(font, new Vector2(bx + Pad, by + Pad + (li + 0.82f) * lineH),
                    lines[li].Text, HorizontalAlignment.Left, -1, fs - 1, lines[li].Col);
        }
    }
}
