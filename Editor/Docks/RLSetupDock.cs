using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Editor;

[Tool]
public partial class RLSetupDock : VBoxContainer
{
    private readonly Label _scenePathLabel;
    private readonly Label _wizardStageLabel;
    private readonly Label _wizardHintLabel;
    private readonly Label _wizardIssueCountLabel;
    private readonly Button _fixAllButton;
    private readonly VBoxContainer _issueList;
    private readonly VBoxContainer _reviewList;
    private readonly Label _reviewEmptyLabel;
    private readonly Label _validationStatusLabel;
    private readonly Label _validationDetailLabel;
    private readonly Label _configLabel;
    private readonly Label _launchStatusLabel;
    private readonly Button _startButton;
    private readonly Button _stopButton;
    private readonly Button _quickTestButton;
    private readonly Button _validateSceneButton;

    [Signal]
    public delegate void StartTrainingRequestedEventHandler();

    [Signal]
    public delegate void StopTrainingRequestedEventHandler();

    [Signal]
    public delegate void QuickTestRequestedEventHandler();

    [Signal]
    public delegate void ValidateSceneRequestedEventHandler();

    [Signal]
    public delegate void AutofixRequestedEventHandler(int fixKind, string targetPath);

    [Signal]
    public delegate void AutofixAllRequestedEventHandler();

    [Signal]
    public delegate void ReviewTargetRequestedEventHandler(bool isResource, string targetPath);

    private static int Ui(int value) => EditorUiScale.Px(value);

    public RLSetupDock()
    {
        Name = "RL Setup";
        CustomMinimumSize = new Vector2(240f, 0f);
        SizeFlagsHorizontal = SizeFlags.ExpandFill;

        ClearExistingChildren();

        var scroll = new ScrollContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        AddChild(scroll);

        var outer = new MarginContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        SetMargins(outer, 8);
        scroll.AddChild(outer);

        var vbox = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        vbox.AddThemeConstantOverride("separation", 6);
        outer.AddChild(vbox);

        vbox.AddChild(MakeSectionHeader("Scene"));
        _scenePathLabel = new Label
        {
            Text = "—",
            ClipText = true,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        vbox.AddChild(_scenePathLabel);

        vbox.AddChild(new HSeparator());

        vbox.AddChild(MakeSectionHeader("Wizard"));
        _wizardStageLabel = new Label { Text = "Analyze" };
        _wizardStageLabel.AddThemeFontSizeOverride("font_size", Ui(14));
        vbox.AddChild(_wizardStageLabel);

        _wizardHintLabel = new Label
        {
            Text = "Open a training scene to analyze setup.",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_wizardHintLabel);

        _wizardIssueCountLabel = new Label
        {
            Text = "Issues: 0",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_wizardIssueCountLabel);

        _fixAllButton = new Button
        {
            Text = "Fix All Safe",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0f, 30f),
            Visible = false,
        };
        _fixAllButton.Pressed += OnFixAllButtonPressed;
        vbox.AddChild(_fixAllButton);

        _issueList = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _issueList.AddThemeConstantOverride("separation", 4);
        vbox.AddChild(_issueList);

        vbox.AddChild(new HSeparator());

        vbox.AddChild(MakeSectionHeader("Review"));
        _reviewList = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _reviewList.AddThemeConstantOverride("separation", 4);
        vbox.AddChild(_reviewList);

        _reviewEmptyLabel = new Label
        {
            Text = "No autofixes have been applied in this scene yet.",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_reviewEmptyLabel);

        vbox.AddChild(new HSeparator());

        vbox.AddChild(MakeSectionHeader("Validation"));
        _validationStatusLabel = new Label { Text = "—" };
        vbox.AddChild(_validationStatusLabel);

        _validationDetailLabel = new Label
        {
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_validationDetailLabel);

        vbox.AddChild(new HSeparator());

        vbox.AddChild(MakeSectionHeader("Resources"));
        _configLabel = new Label
        {
            Text = "Configs: not resolved",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_configLabel);

        vbox.AddChild(new HSeparator());

        var buttonRow = new HBoxContainer();
        buttonRow.AddThemeConstantOverride("separation", 4);
        vbox.AddChild(buttonRow);

        _startButton = new Button
        {
            Text = "▶  Start Training",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            TooltipText = "Launch a training run using the active or main scene",
            CustomMinimumSize = new Vector2(0f, 32f),
        };
        _startButton.Pressed += OnStartButtonPressed;
        buttonRow.AddChild(_startButton);

        _stopButton = new Button
        {
            Text = "■  Stop",
            TooltipText = "Stop the active training run",
            CustomMinimumSize = new Vector2(0f, 32f),
        };
        _stopButton.Pressed += OnStopButtonPressed;
        buttonRow.AddChild(_stopButton);

        var utilityRow = new HBoxContainer();
        utilityRow.AddThemeConstantOverride("separation", 4);
        vbox.AddChild(utilityRow);

        _quickTestButton = new Button
        {
            Text = "Quick Test",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            TooltipText = "Run a short smoke test with BatchSize forced to 1.",
            CustomMinimumSize = new Vector2(0f, 30f),
        };
        _quickTestButton.Pressed += OnQuickTestButtonPressed;
        utilityRow.AddChild(_quickTestButton);

        _validateSceneButton = new Button
        {
            Text = "Validate Scene",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            TooltipText = "Run scene validation now.",
            CustomMinimumSize = new Vector2(0f, 30f),
        };
        _validateSceneButton.Pressed += OnValidateSceneButtonPressed;
        utilityRow.AddChild(_validateSceneButton);

        _launchStatusLabel = new Label
        {
            Text = "Status: idle",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_launchStatusLabel);
    }

    public void SetScenePath(string scenePath)
    {
        var fileName = System.IO.Path.GetFileNameWithoutExtension(scenePath);
        _scenePathLabel.Text = string.IsNullOrWhiteSpace(fileName) ? "—" : fileName;
        _scenePathLabel.TooltipText = scenePath;
    }

    public void SetValidationSummary(string summary, bool? isValid = null)
    {
        if (isValid == true)
        {
            _validationStatusLabel.Text = "Ready to train";
            _validationStatusLabel.AddThemeColorOverride("font_color", new Color(0.45f, 0.82f, 0.45f));
        }
        else if (isValid == false)
        {
            _validationStatusLabel.Text = "Scene has blocking issues";
            _validationStatusLabel.AddThemeColorOverride("font_color", new Color(0.90f, 0.40f, 0.40f));
        }
        else
        {
            _validationStatusLabel.Text = "—";
            _validationStatusLabel.RemoveThemeColorOverride("font_color");
        }

        _validationDetailLabel.Text = summary;
    }

    public void SetConfigSummary(string trainingPath, string networkPath, string inferenceSummary)
    {
        var trainingName = FileName(trainingPath, "(none)");
        var networkName = FileName(networkPath, "(none)");
        var inferenceName = string.IsNullOrWhiteSpace(inferenceSummary) ? "(none)" : inferenceSummary;
        _configLabel.Text = $"Training:   {trainingName}\nNetwork:     {networkName}\nInference: {inferenceName}";
        _configLabel.TooltipText = $"Training: {trainingPath}\nNetwork: {networkPath}\nInference: {inferenceName}";
    }

    public void SetLaunchStatus(string text)
    {
        _launchStatusLabel.Text = text;
        var lower = text.ToLowerInvariant();
        if (lower.Contains("launch") || lower.Contains("starting"))
        {
            _launchStatusLabel.AddThemeColorOverride("font_color", new Color(0.40f, 0.72f, 0.90f));
        }
        else if (lower.Contains("stop") || lower.Contains("block") || lower.Contains("fail"))
        {
            _launchStatusLabel.AddThemeColorOverride("font_color", new Color(0.90f, 0.50f, 0.40f));
        }
        else
        {
            _launchStatusLabel.RemoveThemeColorOverride("font_color");
        }
    }

    public void SetActionStates(
        bool canStartTraining,
        string startTrainingTooltip,
        bool canStop,
        string stopTooltip,
        bool canQuickTest,
        string quickTestTooltip,
        bool canValidateScene,
        string validateSceneTooltip)
    {
        _startButton.Disabled = !canStartTraining;
        _startButton.TooltipText = startTrainingTooltip;

        _stopButton.Disabled = !canStop;
        _stopButton.TooltipText = stopTooltip;

        _quickTestButton.Disabled = !canQuickTest;
        _quickTestButton.TooltipText = quickTestTooltip;

        _validateSceneButton.Disabled = !canValidateScene;
        _validateSceneButton.TooltipText = validateSceneTooltip;
    }

    public void SetWizardState(TrainingSceneValidation? validation, IReadOnlyList<TrainingSceneReviewEntry>? reviewEntries)
    {
        ClearContainer(_issueList);
        ClearContainer(_reviewList);

        reviewEntries ??= new List<TrainingSceneReviewEntry>();
        var orderedIssues = validation?.Issues
            .OrderByDescending(issue => issue.Severity)
            .ThenByDescending(issue => issue.IsAutofixable)
            .ThenBy(issue => issue.Code)
            .ToList() ?? new List<TrainingSceneIssue>();
        var autofixableIssues = orderedIssues.Where(issue => issue.IsAutofixable).ToList();
        var blockingCount = orderedIssues.Count(issue => issue.Severity == TrainingSceneIssueSeverity.Blocking);

        if (validation is null)
        {
            SetWizardStage("Analyze", "Open a scene to inspect RL setup and available autofixes.", Colors.WhiteSmoke);
            _wizardIssueCountLabel.Text = "Issues: —";
            _fixAllButton.Visible = false;
        }
        else if (reviewEntries.Count > 0)
        {
            var reviewColor = validation.IsValid
                ? new Color(0.45f, 0.82f, 0.45f)
                : new Color(0.90f, 0.72f, 0.35f);
            var reviewMessage = validation.IsValid
                ? "Autofix applied. Review the created setup assets, then start training."
                : "Autofix applied. Review the changes below, then address any remaining manual issues.";
            SetWizardStage("Review", reviewMessage, reviewColor);
            _wizardIssueCountLabel.Text = $"Issues: {orderedIssues.Count} total, {blockingCount} blocking";
            _fixAllButton.Visible = autofixableIssues.Count > 0;
            _fixAllButton.Disabled = autofixableIssues.Count == 0;
        }
        else if (autofixableIssues.Count > 0)
        {
            SetWizardStage(
                "Fix Setup",
                $"Found {autofixableIssues.Count} safe autofix action(s). Apply individual fixes or use Fix All Safe.",
                new Color(0.90f, 0.72f, 0.35f));
            _wizardIssueCountLabel.Text = $"Issues: {orderedIssues.Count} total, {blockingCount} blocking";
            _fixAllButton.Visible = true;
            _fixAllButton.Disabled = false;
        }
        else if (validation.IsValid)
        {
            SetWizardStage("Ready", "Required setup is present. Training and quick test are ready.", new Color(0.45f, 0.82f, 0.45f));
            _wizardIssueCountLabel.Text = "Issues: 0 blocking";
            _fixAllButton.Visible = false;
        }
        else
        {
            SetWizardStage("Analyze", "Validation found issues that still require manual work.", new Color(0.90f, 0.50f, 0.40f));
            _wizardIssueCountLabel.Text = $"Issues: {orderedIssues.Count} total, {blockingCount} blocking";
            _fixAllButton.Visible = false;
        }

        foreach (var issue in orderedIssues)
        {
            _issueList.AddChild(BuildIssueCard(issue));
        }

        _reviewEmptyLabel.Visible = reviewEntries.Count == 0;
        foreach (var entry in reviewEntries)
        {
            _reviewList.AddChild(BuildReviewRow(entry));
        }
    }

    private Control BuildIssueCard(TrainingSceneIssue issue)
    {
        var panel = new PanelContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };

        var margin = new MarginContainer();
        SetMargins(margin, 6);
        panel.AddChild(margin);

        var box = new VBoxContainer();
        box.AddThemeConstantOverride("separation", 4);
        margin.AddChild(box);

        var severity = issue.Severity == TrainingSceneIssueSeverity.Blocking ? "Blocking" : "Warning";
        var severityColor = issue.Severity == TrainingSceneIssueSeverity.Blocking
            ? new Color(0.95f, 0.45f, 0.40f)
            : new Color(0.92f, 0.74f, 0.30f);
        var severityLabel = new Label
        {
            Text = severity,
        };
        severityLabel.AddThemeColorOverride("font_color", severityColor);
        box.AddChild(severityLabel);

        var messageLabel = new Label
        {
            Text = issue.Message,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        box.AddChild(messageLabel);

        if (!string.IsNullOrWhiteSpace(issue.TargetPath))
        {
            var pathLabel = new Label
            {
                Text = issue.TargetPath,
                AutowrapMode = TextServer.AutowrapMode.WordSmart,
            };
            pathLabel.AddThemeColorOverride("font_color", new Color(0.72f, 0.72f, 0.72f));
            box.AddChild(pathLabel);
        }

        if (issue.IsAutofixable)
        {
            var button = new Button
            {
                Text = string.IsNullOrWhiteSpace(issue.FixLabel) ? "Fix" : issue.FixLabel,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0f, 28f),
            };
            button.Pressed += () => EmitSignal(SignalName.AutofixRequested, (int)issue.FixKind, issue.TargetPath);
            box.AddChild(button);
        }

        return panel;
    }

    private Control BuildReviewRow(TrainingSceneReviewEntry entry)
    {
        var row = new HBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        row.AddThemeConstantOverride("separation", 6);

        var label = new Label
        {
            Text = entry.Title,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        label.TooltipText = entry.TargetPath;
        row.AddChild(label);

        var button = new Button
        {
            Text = string.IsNullOrWhiteSpace(entry.ActionLabel) ? "Open" : entry.ActionLabel,
        };
        var isResource = entry.TargetKind == TrainingSceneReviewTargetKind.Resource;
        button.Pressed += () => EmitSignal(SignalName.ReviewTargetRequested, isResource, entry.TargetPath);
        row.AddChild(button);

        return row;
    }

    private void SetWizardStage(string stage, string hint, Color color)
    {
        _wizardStageLabel.Text = stage;
        _wizardStageLabel.AddThemeColorOverride("font_color", color);
        _wizardHintLabel.Text = hint;
    }

    private static Label MakeSectionHeader(string text)
    {
        var label = new Label { Text = text };
        label.AddThemeFontSizeOverride("font_size", Ui(13));
        return label;
    }

    private static void SetMargins(MarginContainer container, int margin)
    {
        container.AddThemeConstantOverride("margin_left", margin);
        container.AddThemeConstantOverride("margin_right", margin);
        container.AddThemeConstantOverride("margin_top", margin);
        container.AddThemeConstantOverride("margin_bottom", margin);
    }

    private void ClearExistingChildren()
    {
        for (var childIndex = GetChildCount() - 1; childIndex >= 0; childIndex--)
        {
            var child = GetChild(childIndex);
            RemoveChild(child);
            child.Free();
        }
    }

    private static void ClearContainer(Container container)
    {
        for (var childIndex = container.GetChildCount() - 1; childIndex >= 0; childIndex--)
        {
            var child = container.GetChild(childIndex);
            container.RemoveChild(child);
            child.QueueFree();
        }
    }

    private static string FileName(string path, string fallback)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return fallback;
        }

        var name = System.IO.Path.GetFileName(path);
        return string.IsNullOrWhiteSpace(name) ? fallback : name;
    }

    private void OnStartButtonPressed()
    {
        EmitSignal(SignalName.StartTrainingRequested);
    }

    private void OnStopButtonPressed()
    {
        EmitSignal(SignalName.StopTrainingRequested);
    }

    private void OnQuickTestButtonPressed()
    {
        EmitSignal(SignalName.QuickTestRequested);
    }

    private void OnValidateSceneButtonPressed()
    {
        EmitSignal(SignalName.ValidateSceneRequested);
    }

    private void OnFixAllButtonPressed()
    {
        EmitSignal(SignalName.AutofixAllRequested);
    }
}
