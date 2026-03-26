using Godot;

namespace RlAgentPlugin.Editor;

[Tool]
public partial class RLSetupDock : VBoxContainer
{
    private readonly Label _scenePathLabel;
    private readonly Label _validationStatusLabel;
    private readonly Label _validationDetailLabel;
    private readonly Label _configLabel;
    private readonly Label _launchStatusLabel;
    private readonly Button _startButton;
    private readonly Button _stopButton;
    private readonly Button _quickTestButton;
    private readonly Button _validateSceneButton;

    public RLSetupDock()
    {
        Name = "RL Setup";
        CustomMinimumSize = new Vector2(220f, 0f);
        SizeFlagsHorizontal = SizeFlags.ExpandFill;

        // Tool scripts can be reconstructed after an editor domain reload while the
        // native dock node still has its previous children. Clear any stale UI first
        // so we do not accumulate duplicate scroll containers.
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
        outer.AddChild(vbox);

        // Scene
        vbox.AddChild(MakeSectionHeader("Scene"));
        _scenePathLabel = new Label
        {
            Text = "—",
            ClipText = true,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        vbox.AddChild(_scenePathLabel);

        vbox.AddChild(MakeSpacer(4));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));

        // Validation
        vbox.AddChild(MakeSectionHeader("Validation"));
        _validationStatusLabel = new Label { Text = "—" };
        vbox.AddChild(_validationStatusLabel);

        _validationDetailLabel = new Label
        {
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_validationDetailLabel);

        vbox.AddChild(MakeSpacer(4));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));

        // Resources
        vbox.AddChild(MakeSectionHeader("Resources"));
        _configLabel = new Label
        {
            Text = "Configs: not resolved",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_configLabel);

        vbox.AddChild(MakeSpacer(6));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));

        // Start / Stop buttons
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
        _startButton.Pressed += () => StartTrainingRequested?.Invoke();
        buttonRow.AddChild(_startButton);

        _stopButton = new Button
        {
            Text = "■  Stop",
            TooltipText = "Stop the active training run",
            CustomMinimumSize = new Vector2(0f, 32f),
        };
        _stopButton.Pressed += () => StopTrainingRequested?.Invoke();
        buttonRow.AddChild(_stopButton);

        vbox.AddChild(MakeSpacer(4));

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
        _quickTestButton.Pressed += () => QuickTestRequested?.Invoke();
        utilityRow.AddChild(_quickTestButton);

        _validateSceneButton = new Button
        {
            Text = "Validate Scene",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            TooltipText = "Run scene validation now.",
            CustomMinimumSize = new Vector2(0f, 30f),
        };
        _validateSceneButton.Pressed += () => ValidateSceneRequested?.Invoke();
        utilityRow.AddChild(_validateSceneButton);

        vbox.AddChild(MakeSpacer(4));

        _launchStatusLabel = new Label
        {
            Text = "Status: idle",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        vbox.AddChild(_launchStatusLabel);

        vbox.AddChild(MakeSpacer(4));
        vbox.AddChild(new HSeparator());
        vbox.AddChild(MakeSpacer(4));
    }

    public event System.Action? StartTrainingRequested;
    public event System.Action? StopTrainingRequested;
    public event System.Action? QuickTestRequested;
    public event System.Action? ValidateSceneRequested;

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
            _validationStatusLabel.Text = "✓  Ready to train";
            _validationStatusLabel.AddThemeColorOverride("font_color", new Color(0.45f, 0.82f, 0.45f));
        }
        else if (isValid == false)
        {
            _validationStatusLabel.Text = "✗  Scene has errors";
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

    private static Label MakeSectionHeader(string text)
    {
        var label = new Label { Text = text };
        label.AddThemeFontSizeOverride("font_size", 13);
        return label;
    }

    private static Control MakeSpacer(int height)
    {
        return new Control { CustomMinimumSize = new Vector2(0f, height) };
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

    private static string FileName(string path, string fallback)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return fallback;
        }

        var name = System.IO.Path.GetFileName(path);
        return string.IsNullOrWhiteSpace(name) ? fallback : name;
    }
}
