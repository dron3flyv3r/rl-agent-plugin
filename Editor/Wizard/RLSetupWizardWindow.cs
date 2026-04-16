using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Godot;
using MouseFilterEnum = Godot.Control.MouseFilterEnum;
using RlAgentPlugin.Runtime;
using SizeFlags = Godot.Control.SizeFlags;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Step-by-step wizard window for setting up RL training in a Godot scene.
/// Walks the user through mode, dimensions, training type, node selection,
/// algorithm, network architecture, and key run settings, then invokes ApplyRequested.
/// </summary>
[Tool]
public partial class RLSetupWizardWindow : Window
{
    private const int StepCount = 8;

    // Dependencies injected at construction time
    private readonly Node _sceneRoot;

    // State collected across all steps
    private readonly RLSetupWizardState _state = new();

    // Navigation
    private int _currentStep;
    private readonly Control[] _pages = new Control[StepCount];
    private readonly Label _stepTitleLabel;
    private readonly Label _stepSubtitleLabel;
    private readonly Button _backButton;
    private readonly Button _nextButton;
    private readonly Label _errorLabel;

    // Step 3 — node selection (rebuilt when training mode changes)
    private VBoxContainer? _groupAList;
    private VBoxContainer? _groupBList;

    // Step 5 — network
    private Label? _networkSummaryLabel;

    // Step 7 — summary
    private Label? _summaryLabel;

    /// <summary>Fired when the user clicks Apply on the final step. Receives the completed wizard state.</summary>
    public event Action<RLSetupWizardState>? ApplyRequested;

    private static int Ui(int value) => EditorUiScale.Px(value);

    public RLSetupWizardWindow(Node sceneRoot)
    {
        _sceneRoot = sceneRoot;

        Title = "RL Agent Setup Wizard";
        Exclusive = true;
        Unresizable = false;
        InitialPosition = WindowInitialPosition.CenterMainWindowScreen;

        Connect(SignalName.CloseRequested, Callable.From(QueueFree));

        // Root layout
        var root = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        root.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        root.AddThemeConstantOverride("separation", 0);
        AddChild(root);

        // ── Header ───────────────────────────────────────────────────────────
        var headerMargin = new MarginContainer();
        SetMargins(headerMargin, 16, 16, 12, 8);
        root.AddChild(headerMargin);

        var headerVBox = new VBoxContainer();
        headerVBox.AddThemeConstantOverride("separation", 2);
        headerMargin.AddChild(headerVBox);

        _stepTitleLabel = new Label { Text = "Welcome" };
        _stepTitleLabel.AddThemeFontSizeOverride("font_size", Ui(16));
        headerVBox.AddChild(_stepTitleLabel);

        _stepSubtitleLabel = new Label
        {
            Text = "Set up RL training in your current scene.",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        _stepSubtitleLabel.AddThemeColorOverride("font_color", new Color(0.75f, 0.75f, 0.75f));
        headerVBox.AddChild(_stepSubtitleLabel);

        root.AddChild(new HSeparator());

        // ── Scrollable content ────────────────────────────────────────────────
        var scroll = new ScrollContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        root.AddChild(scroll);

        var contentMargin = new MarginContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        SetMargins(contentMargin, 16);
        scroll.AddChild(contentMargin);

        var pageHost = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        contentMargin.AddChild(pageHost);

        // Build pages
        _pages[0] = BuildModeStep();
        _pages[1] = BuildDimensionStep();
        _pages[2] = BuildTrainingModeStep();
        _pages[3] = BuildNodeSelectionStep();
        _pages[4] = BuildAlgorithmStep();
        _pages[5] = BuildNetworkStep();
        _pages[6] = BuildSettingsStep();
        _pages[7] = BuildSummaryStep();

        foreach (var page in _pages)
        {
            page.SizeFlagsHorizontal = SizeFlags.ExpandFill;
            pageHost.AddChild(page);
        }

        root.AddChild(new HSeparator());

        // ── Error label ───────────────────────────────────────────────────────
        var errorMargin = new MarginContainer();
        SetMargins(errorMargin, 16, 16, 4, 4);
        root.AddChild(errorMargin);

        _errorLabel = new Label
        {
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            Visible = false,
        };
        _errorLabel.AddThemeColorOverride("font_color", new Color(0.95f, 0.55f, 0.45f));
        errorMargin.AddChild(_errorLabel);

        // ── Navigation row ────────────────────────────────────────────────────
        var navMargin = new MarginContainer();
        SetMargins(navMargin, 16, 16, 8, 12);
        root.AddChild(navMargin);

        var navRow = new HBoxContainer();
        navRow.AddThemeConstantOverride("separation", 8);
        navMargin.AddChild(navRow);

        _backButton = new Button
        {
            Text = "← Back",
            CustomMinimumSize = new Vector2(Ui(90), Ui(30)),
        };
        _backButton.Pressed += OnBackPressed;
        navRow.AddChild(_backButton);

        navRow.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        _nextButton = new Button
        {
            Text = "Next →",
            CustomMinimumSize = new Vector2(Ui(120), Ui(30)),
        };
        _nextButton.Pressed += OnNextPressed;
        navRow.AddChild(_nextButton);

        ShowPage(0);
    }

    // ── Step builders ─────────────────────────────────────────────────────────

    private Control BuildModeStep()
    {
        var vbox = NewStepBox();
        vbox.AddChild(MakeDescription("Choose how to add RL training to your scene."));

        var group = new ButtonGroup();
        vbox.AddChild(MakeOptionCard(group, "Fresh Setup",
            "Add a new agent node and all required resources. Pick a character node to attach the agent to.",
            defaultSelected: true,
            onSelected: () => _state.Mode = WizardMode.Fresh));

        vbox.AddChild(MakeOptionCard(group, "Existing Scene",
            "Configure already-present RLAgent2D or RLAgent3D nodes. Assigns missing resources without touching node structure.",
            defaultSelected: false,
            onSelected: () => _state.Mode = WizardMode.Existing));

        return vbox;
    }

    private Control BuildDimensionStep()
    {
        var vbox = NewStepBox();
        vbox.AddChild(MakeDescription("What type of scene is this? This determines which agent node type is added."));

        var group = new ButtonGroup();
        vbox.AddChild(MakeOptionCard(group, "2D",
            "Node2D-based scene (CharacterBody2D, RigidBody2D, etc.). Adds RLAgent2D.",
            defaultSelected: true,
            onSelected: () => _state.Dimension = WizardDimension.TwoD));

        vbox.AddChild(MakeOptionCard(group, "3D",
            "Node3D-based scene (CharacterBody3D, RigidBody3D, etc.). Adds RLAgent3D.",
            defaultSelected: false,
            onSelected: () => _state.Dimension = WizardDimension.ThreeD));

        return vbox;
    }

    private Control BuildTrainingModeStep()
    {
        var vbox = NewStepBox();
        vbox.AddChild(MakeDescription("How many agents, and how do they interact?"));

        var group = new ButtonGroup();
        vbox.AddChild(MakeOptionCard(group, "Single Agent",
            "One agent, one policy group, one network. The simplest setup.",
            defaultSelected: true,
            onSelected: () => _state.TrainingMode = WizardTrainingMode.SingleAgent));

        vbox.AddChild(MakeOptionCard(group, "Multi-Agent — Shared Policy",
            "Multiple agents that all share one network. Used for parallel environment training (same team, same task).",
            defaultSelected: false,
            onSelected: () => _state.TrainingMode = WizardTrainingMode.SharedPolicy));

        vbox.AddChild(MakeOptionCard(group, "Multi-Agent — Individual Policies",
            "Multiple agents that each get their own independent network. Used when each agent has a distinct role or task.",
            defaultSelected: false,
            onSelected: () => _state.TrainingMode = WizardTrainingMode.IndividualPolicies));

        vbox.AddChild(MakeOptionCard(group, "Self-Play — Competitive",
            "Two groups that train against each other, each with its own network. For adversarial tasks like tag or combat.",
            defaultSelected: false,
            onSelected: () => _state.TrainingMode = WizardTrainingMode.SelfPlay));

        return vbox;
    }

    private Control BuildNodeSelectionStep()
    {
        var vbox = NewStepBox();
        _groupAList = null;
        _groupBList = null;
        RebuildNodeSelectionContent(vbox);
        return vbox;
    }

    private void RebuildNodeSelectionStep()
    {
        if (_pages[3] is not VBoxContainer vbox) return;
        ClearContainer(vbox);
        _state.GroupANodes.Clear();
        _state.GroupBNodes.Clear();
        _groupAList = null;
        _groupBList = null;
        RebuildNodeSelectionContent(vbox);
    }

    private void RebuildNodeSelectionContent(VBoxContainer vbox)
    {
        var isSelfPlay = _state.TrainingMode == WizardTrainingMode.SelfPlay;
        var isSingleAgent = _state.TrainingMode == WizardTrainingMode.SingleAgent;
        var isExisting = _state.Mode == WizardMode.Existing;
        var noun = isExisting
            ? (isSingleAgent ? "agent node" : "agent node(s)")
            : (isSingleAgent ? "character node" : "character node(s)");

        var description = isSelfPlay
            ? isExisting
                ? $"Select {noun} for Group A and Group B. Use the checkboxes to assign nodes to each group."
                : $"Select {noun} for Group A and Group B, or use the scene root as the parent for spawned agent nodes."
            : isSingleAgent
                ? isExisting
                    ? "Select exactly one agent node."
                    : "Select exactly one character node to attach an agent to."
                : isExisting
                    ? $"Select the {noun} to set up. All selected nodes will share one policy group."
                    : $"Select the {noun} to attach agents to, or use the scene root as the parent.";
        vbox.AddChild(MakeDescription(description));

        if (isSelfPlay)
        {
            var splitRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            splitRow.AddThemeConstantOverride("separation", 8);
            vbox.AddChild(splitRow);

            // Group A column
            var colA = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            colA.AddThemeConstantOverride("separation", 4);
            splitRow.AddChild(colA);

            var headerA = new Label { Text = "Group A" };
            headerA.AddThemeFontSizeOverride("font_size", Ui(13));
            colA.AddChild(headerA);

            _groupAList = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            _groupAList.AddThemeConstantOverride("separation", 2);
            var scrollA = new ScrollContainer
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0, Ui(180)),
            };
            scrollA.AddChild(_groupAList);
            colA.AddChild(scrollA);

            // Group B column
            var colB = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            colB.AddThemeConstantOverride("separation", 4);
            splitRow.AddChild(colB);

            var headerB = new Label { Text = "Group B" };
            headerB.AddThemeFontSizeOverride("font_size", Ui(13));
            colB.AddChild(headerB);

            _groupBList = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            _groupBList.AddThemeConstantOverride("separation", 2);
            var scrollB = new ScrollContainer
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0, Ui(180)),
            };
            scrollB.AddChild(_groupBList);
            colB.AddChild(scrollB);
        }
        else
        {
            var header = new Label { Text = isSingleAgent ? "Select node:" : "Select node(s):" };
            header.AddThemeFontSizeOverride("font_size", Ui(13));
            vbox.AddChild(header);

            _groupAList = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            _groupAList.AddThemeConstantOverride("separation", 2);
            var scrollA = new ScrollContainer
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0, Ui(200)),
            };
            scrollA.AddChild(_groupAList);
            vbox.AddChild(scrollA);
        }

        PopulateNodeList(_groupAList!, _state.GroupANodes, isExisting, singleSelect: isSingleAgent);
        if (_groupBList != null)
        {
            PopulateNodeList(_groupBList, _state.GroupBNodes, isExisting);
        }
    }

    private void PopulateNodeList(
        VBoxContainer container,
        List<Node> target,
        bool existingOnly,
        bool singleSelect = false)
    {
        var candidates = new List<(Node node, string display)>();
        if (!existingOnly)
        {
            candidates.Add((_sceneRoot, "<scene root>"));
        }

        CollectNodeCandidates(_sceneRoot, _sceneRoot, existingOnly, depth: 0, candidates);

        if (candidates.Count == 0)
        {
            var empty = new Label
            {
                Text = existingOnly
                    ? "No RLAgent2D or RLAgent3D nodes found in the scene."
                    : "No Node2D or Node3D nodes found in the scene.",
                AutowrapMode = TextServer.AutowrapMode.WordSmart,
            };
            empty.AddThemeColorOverride("font_color", new Color(0.6f, 0.6f, 0.6f));
            container.AddChild(empty);
            return;
        }

        // In single-select mode attach all checkboxes to one ButtonGroup so only
        // one can be active at a time (they behave like radio buttons).
        ButtonGroup? radioGroup = singleSelect ? new ButtonGroup() : null;

        foreach (var (node, display) in candidates)
        {
            var shouldAutoSelect = !existingOnly && candidates.Count == 1;
            var check = new CheckBox
            {
                Text = display,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                ButtonPressed = shouldAutoSelect,
                ButtonGroup = radioGroup,
            };
            var capturedNode = node;
            if (shouldAutoSelect && !target.Contains(capturedNode))
            {
                target.Add(capturedNode);
            }

            check.Toggled += on =>
            {
                if (on)
                {
                    if (singleSelect) target.Clear(); // radio: replace previous selection
                    if (!target.Contains(capturedNode))
                        target.Add(capturedNode);
                }
                else target.Remove(capturedNode);
                UpdateNavButtons();
            };

            container.AddChild(check);
        }
    }

    private static void CollectNodeCandidates(
        Node root,
        Node current,
        bool existingOnly,
        int depth,
        List<(Node, string)> results)
    {
        if (depth > 6) return;

        if (current != root)
        {
            var include = existingOnly
                ? IsAgentNode(current)
                : current is Node2D or Node3D;

            if (include)
            {
                var relativePath = root.GetPathTo(current).ToString();
                var display = existingOnly
                    ? $"{relativePath}  [{AgentTypeName(current)}]"
                    : relativePath;
                results.Add((current, display));
            }
        }

        foreach (var child in current.GetChildren())
        {
            CollectNodeCandidates(root, child, existingOnly, depth + 1, results);
        }
    }

    private static bool IsAgentNode(Node node)
    {
        // Prefer C# type check — works when the assembly is fully loaded.
        if (node is RLAgent2D or RLAgent3D) return true;

        // Fallback: property-list duck-typing. Reliable in the editor even when
        // the C# script type hasn't been resolved yet (e.g. right after a rename
        // or before Godot finishes hot-reloading the assembly).
        foreach (var prop in node.GetPropertyList())
        {
            if (prop["name"].AsString() == "PolicyGroupConfig") return true;
        }

        return false;
    }

    private static string AgentTypeName(Node node)
    {
        if (node is RLAgent2D) return "RLAgent2D";
        if (node is RLAgent3D) return "RLAgent3D";

        // C# type not resolved — infer from the script path if available.
        var scriptPath = node.GetScript().As<Script>()?.ResourcePath ?? string.Empty;
        if (scriptPath.Contains("RLAgent2D", StringComparison.OrdinalIgnoreCase)) return "RLAgent2D";
        if (scriptPath.Contains("RLAgent3D", StringComparison.OrdinalIgnoreCase)) return "RLAgent3D";

        return "agent";
    }

    private Control BuildAlgorithmStep()
    {
        var vbox = NewStepBox();
        vbox.AddChild(MakeDescription("Choose a training algorithm. PPO is recommended for most tasks."));

        var group = new ButtonGroup();
        vbox.AddChild(MakeAlgorithmCard(group, "PPO", "Proximal Policy Optimization",
            "Robust, general-purpose. Best starting point. Works with discrete and continuous actions.",
            "Discrete + Continuous · On-policy",
            defaultSelected: true, onSelected: () => _state.Algorithm = WizardAlgorithm.PPO));

        vbox.AddChild(MakeAlgorithmCard(group, "SAC", "Soft Actor-Critic",
            "Sample-efficient, ideal for physics-based continuous control. Continuous actions only.",
            "Continuous only · Off-policy",
            defaultSelected: false, onSelected: () => _state.Algorithm = WizardAlgorithm.SAC));

        vbox.AddChild(MakeAlgorithmCard(group, "DQN", "Deep Q-Network",
            "Classic discrete action algorithm. Good for simple action spaces (move/attack/jump).",
            "Discrete only · Off-policy",
            defaultSelected: false, onSelected: () => _state.Algorithm = WizardAlgorithm.DQN));

        vbox.AddChild(MakeAlgorithmCard(group, "A2C", "Advantage Actor-Critic",
            "Simpler and faster than PPO, less stable. Good for quick experiments.",
            "Discrete + Continuous · On-policy",
            defaultSelected: false, onSelected: () => _state.Algorithm = WizardAlgorithm.A2C));

        return vbox;
    }

    private Control BuildNetworkStep()
    {
        var vbox = NewStepBox();
        vbox.AddChild(MakeDescription("Choose a network size. Small works well for most tasks."));

        // Preset buttons
        var presetRow = new HBoxContainer();
        presetRow.AddThemeConstantOverride("separation", 6);
        vbox.AddChild(presetRow);

        var presetGroup = new ButtonGroup();
        Button MakePreset(string label, WizardNetworkPreset preset, bool isDefault)
        {
            var btn = new Button
            {
                Text = label,
                ToggleMode = true,
                ButtonGroup = presetGroup,
                ButtonPressed = isDefault,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0, Ui(48)),
            };
            if (isDefault) _state.NetworkPreset = preset;
            btn.Pressed += () => { _state.NetworkPreset = preset; UpdateNetworkSummary(); };
            return btn;
        }

        presetRow.AddChild(MakePreset("Tiny\n1×32", WizardNetworkPreset.Tiny, isDefault: false));
        presetRow.AddChild(MakePreset("Small\n2×64", WizardNetworkPreset.Small, isDefault: true));
        presetRow.AddChild(MakePreset("Medium\n2×128", WizardNetworkPreset.Medium, isDefault: false));
        presetRow.AddChild(MakePreset("Large\n3×256", WizardNetworkPreset.Large, isDefault: false));

        _networkSummaryLabel = new Label { AutowrapMode = TextServer.AutowrapMode.WordSmart };
        _networkSummaryLabel.AddThemeColorOverride("font_color", new Color(0.72f, 0.72f, 0.72f));
        vbox.AddChild(_networkSummaryLabel);
        UpdateNetworkSummary();

        vbox.AddChild(new HSeparator());

        // Activation + Optimizer
        var grid = new GridContainer { Columns = 2, SizeFlagsHorizontal = SizeFlags.ExpandFill };
        grid.AddThemeConstantOverride("h_separation", 12);
        grid.AddThemeConstantOverride("v_separation", 6);
        vbox.AddChild(grid);

        grid.AddChild(new Label { Text = "Activation:" });
        var activationDropdown = new OptionButton { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        activationDropdown.AddItem("Tanh (recommended)");   // index 0 → RLActivationKind.Tanh
        activationDropdown.AddItem("ReLU");                 // index 1 → RLActivationKind.Relu
        activationDropdown.Selected = 0;
        activationDropdown.ItemSelected += idx => _state.Activation = (RLActivationKind)idx;
        grid.AddChild(activationDropdown);

        grid.AddChild(new Label { Text = "Optimizer:" });
        var optimizerDropdown = new OptionButton { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        optimizerDropdown.AddItem("Adam (recommended)");  // index 0 → RLOptimizerKind.Adam
        optimizerDropdown.AddItem("AdamW");               // index 1 → RLOptimizerKind.AdamW
        optimizerDropdown.AddItem("SGD");                 // index 2 → RLOptimizerKind.Sgd
        optimizerDropdown.Selected = 0;
        optimizerDropdown.ItemSelected += idx => _state.Optimizer = idx switch
        {
            1 => RLOptimizerKind.AdamW,
            2 => RLOptimizerKind.Sgd,
            _ => RLOptimizerKind.Adam,
        };
        grid.AddChild(optimizerDropdown);

        return vbox;
    }

    private Control BuildSettingsStep()
    {
        var vbox = NewStepBox();
        vbox.AddChild(MakeDescription("Configure key training parameters. All settings can be changed later in the Inspector."));

        var grid = new GridContainer { Columns = 3, SizeFlagsHorizontal = SizeFlags.ExpandFill };
        grid.AddThemeConstantOverride("h_separation", 10);
        grid.AddThemeConstantOverride("v_separation", 6);
        vbox.AddChild(grid);

        // Row helper
        void AddRow(string label, Control input, string hint)
        {
            grid.AddChild(new Label { Text = label });
            grid.AddChild(input);
            var hintLabel = new Label
            {
                Text = hint,
                AutowrapMode = TextServer.AutowrapMode.WordSmart,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
            };
            hintLabel.AddThemeColorOverride("font_color", new Color(0.65f, 0.65f, 0.65f));
            grid.AddChild(hintLabel);
        }

        var episodeStepsField = new SpinBox
        {
            MinValue = 0, MaxValue = 100000, Step = 1, Value = 0, AllowGreater = true,
            CustomMinimumSize = new Vector2(Ui(110), 0),
        };
        episodeStepsField.ValueChanged += v => _state.MaxEpisodeSteps = (int)v;
        AddRow("Episode Steps:", episodeStepsField, "0 = unlimited. Recommend 1000–5000 for first run.");

        var actionRepeatField = new SpinBox
        {
            MinValue = 1, MaxValue = 20, Step = 1, Value = 1,
            CustomMinimumSize = new Vector2(Ui(110), 0),
        };
        actionRepeatField.ValueChanged += v => _state.ActionRepeat = (int)v;
        AddRow("Action Repeat:", actionRepeatField, "Physics steps per decision. 1 = most responsive.");

        var simSpeedField = new SpinBox
        {
            MinValue = 0.01, MaxValue = 20.0, Step = 0.01, Value = 1.0, AllowGreater = true,
            CustomMinimumSize = new Vector2(Ui(110), 0),
        };
        simSpeedField.ValueChanged += v => _state.SimulationSpeed = (float)v;
        AddRow("Sim Speed:", simSpeedField, "Simulation speed multiplier. Higher = faster training.");

        var checkpointField = new SpinBox
        {
            MinValue = 1, MaxValue = 1000, Step = 1, Value = 10, AllowGreater = true,
            CustomMinimumSize = new Vector2(Ui(110), 0),
        };
        checkpointField.ValueChanged += v => _state.CheckpointInterval = (int)v;
        AddRow("Checkpoint Every:", checkpointField, "Save every N trainer updates.");

        var runPrefixField = new LineEdit
        {
            PlaceholderText = "(auto-generated)",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(Ui(110), 0),
        };
        runPrefixField.TextChanged += v => _state.RunPrefix = v;
        AddRow("Run Prefix:", runPrefixField, "Labels the output folder name.");

        return vbox;
    }

    private Control BuildSummaryStep()
    {
        var vbox = NewStepBox();
        vbox.AddChild(MakeDescription("Review your configuration, then click Apply to set up the scene."));

        _summaryLabel = new Label
        {
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        vbox.AddChild(_summaryLabel);

        return vbox;
    }

    // ── Navigation ────────────────────────────────────────────────────────────

    private void ShowPage(int step)
    {
        _currentStep = step;
        for (var i = 0; i < _pages.Length; i++)
        {
            _pages[i].Visible = i == step;
        }

        UpdateHeader();
        UpdateNavButtons();

        if (step == 3) RebuildNodeSelectionStep();
        if (step == 7) RebuildSummary();
    }

    private void UpdateHeader()
    {
        var (title, subtitle) = _currentStep switch
        {
            0 => ("Welcome", "Set up RL training in your current scene."),
            1 => ("Dimensions", "Is this a 2D or 3D scene?"),
            2 => ("Training Mode", "How many agents, and how do they interact?"),
            3 => ("Select Nodes",
                  _state.Mode == WizardMode.Existing
                      ? "Select the existing agent node(s) to configure."
                      : "Select the character node(s) to attach an agent to."),
            4 => ("Algorithm", "Choose a training algorithm."),
            5 => ("Network Architecture", "Configure the neural network size and optimizer."),
            6 => ("Settings", "Configure key training parameters."),
            7 => ("Summary", "Review and apply your setup."),
            _ => ("Setup Wizard", string.Empty),
        };
        _stepTitleLabel.Text = title;
        _stepSubtitleLabel.Text = subtitle;
    }

    private void UpdateNavButtons()
    {
        if (_backButton is null || _nextButton is null || _errorLabel is null)
            return;

        _backButton.Visible = _currentStep > 0;
        _nextButton.Text = _currentStep == StepCount - 1 ? "Apply" : "Next →";

        var (isValid, _) = ValidateCurrentStep();
        _nextButton.Disabled = !isValid;

        if (isValid) _errorLabel.Visible = false;
    }

    private void OnNextPressed()
    {
        var (isValid, error) = ValidateCurrentStep();
        if (!isValid)
        {
            _errorLabel.Text = error;
            _errorLabel.Visible = true;
            return;
        }

        _errorLabel.Visible = false;

        if (_currentStep == StepCount - 1)
        {
            ApplyRequested?.Invoke(_state);
            QueueFree();
            return;
        }

        ShowPage(NextStep(_currentStep));
    }

    private void OnBackPressed()
    {
        _errorLabel.Visible = false;
        ShowPage(PrevStep(_currentStep));
    }

    private int NextStep(int current) =>
        // Skip dimensionality step (1) when configuring an existing scene
        current == 0 && _state.Mode == WizardMode.Existing ? 2 : current + 1;

    private int PrevStep(int current) =>
        current == 2 && _state.Mode == WizardMode.Existing ? 0 : current - 1;

    private (bool valid, string error) ValidateCurrentStep() =>
        _currentStep == 3 ? ValidateNodeSelection() : (true, string.Empty);

    private (bool valid, string error) ValidateNodeSelection()
    {
        if (_state.TrainingMode == WizardTrainingMode.SelfPlay)
        {
            if (_state.GroupANodes.Count == 0) return (false, "Select at least one node for Group A.");
            if (_state.GroupBNodes.Count == 0) return (false, "Select at least one node for Group B.");
        }
        else
        {
            if (_state.GroupANodes.Count == 0) return (false, "Select at least one node to set up.");
        }

        return (true, string.Empty);
    }

    // ── Summary ───────────────────────────────────────────────────────────────

    private void RebuildSummary()
    {
        if (_summaryLabel is null) return;

        var modeStr = _state.Mode == WizardMode.Fresh
            ? $"Fresh Setup — {(_state.Dimension == WizardDimension.TwoD ? "2D" : "3D")}"
            : "Existing Scene";

        var trainingModeStr = _state.TrainingMode switch
        {
            WizardTrainingMode.SingleAgent => "Single Agent",
            WizardTrainingMode.SharedPolicy => "Multi-Agent Shared Policy",
            WizardTrainingMode.IndividualPolicies => "Multi-Agent Individual Policies",
            WizardTrainingMode.SelfPlay => "Self-Play (Competitive)",
            _ => "Unknown",
        };

        static string NodeNames(IEnumerable<Node> nodes)
        {
            var names = nodes.Select(n => n.Name.ToString()).ToList();
            return names.Count > 0 ? string.Join(", ", names) : "(none)";
        }

        var (layerCount, layerSize) = _state.NetworkPreset switch
        {
            WizardNetworkPreset.Tiny => (1, 32),
            WizardNetworkPreset.Small => (2, 64),
            WizardNetworkPreset.Medium => (2, 128),
            WizardNetworkPreset.Large => (3, 256),
            _ => (2, 64),
        };
        var networkStr = $"{layerCount}×{layerSize} Dense, {_state.Activation}, {_state.Optimizer}";

        var sb = new StringBuilder();
        sb.AppendLine($"Mode:             {modeStr}");
        sb.AppendLine($"Training:         {trainingModeStr}");
        sb.AppendLine();

        if (_state.TrainingMode == WizardTrainingMode.SelfPlay)
        {
            sb.AppendLine($"Group A nodes:    {NodeNames(_state.GroupANodes)}");
            sb.AppendLine($"Group B nodes:    {NodeNames(_state.GroupBNodes)}");
        }
        else
        {
            sb.AppendLine($"Nodes:            {NodeNames(_state.GroupANodes)}");
        }

        sb.AppendLine();
        sb.AppendLine($"Algorithm:        {_state.Algorithm}");
        sb.AppendLine($"Network:          {networkStr}");
        sb.AppendLine();
        sb.AppendLine($"Episode Steps:    {(_state.MaxEpisodeSteps == 0 ? "unlimited" : _state.MaxEpisodeSteps.ToString())}");
        sb.AppendLine($"Action Repeat:    {_state.ActionRepeat}");
        sb.AppendLine($"Sim Speed:        {_state.SimulationSpeed:0.0}×");
        sb.AppendLine($"Checkpoint:       every {_state.CheckpointInterval} updates");
        if (!string.IsNullOrWhiteSpace(_state.RunPrefix))
            sb.AppendLine($"Run Prefix:       {_state.RunPrefix}");
        sb.AppendLine();
        sb.AppendLine("Resources will be saved to:  res://<scene-folder>/RL/");

        _summaryLabel.Text = sb.ToString().TrimEnd();
    }

    // ── Network summary ───────────────────────────────────────────────────────

    private void UpdateNetworkSummary()
    {
        if (_networkSummaryLabel is null) return;

        _networkSummaryLabel.Text = _state.NetworkPreset switch
        {
            WizardNetworkPreset.Tiny   => "1 layer × 32 units — very small; for simple tasks or tiny observation spaces.",
            WizardNetworkPreset.Small  => "2 layers × 64 units — good default; works for most tasks.",
            WizardNetworkPreset.Medium => "2 layers × 128 units — more capacity; for complex observations or behaviors.",
            WizardNetworkPreset.Large  => "3 layers × 256 units — large; for very complex tasks. Slower to train.",
            _ => string.Empty,
        };
    }

    // ── UI helpers ────────────────────────────────────────────────────────────

    private static VBoxContainer NewStepBox()
    {
        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 10);
        return vbox;
    }

    private static Label MakeDescription(string text)
    {
        var label = new Label
        {
            Text = text,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        label.AddThemeColorOverride("font_color", new Color(0.82f, 0.82f, 0.82f));
        return label;
    }

    /// <summary>
    /// Builds a full-width toggle button card with a title and description.
    /// Uses Godot's built-in toggle button appearance rather than custom styling.
    /// </summary>
    private static Button MakeOptionCard(
        ButtonGroup group,
        string title,
        string description,
        bool defaultSelected,
        Action onSelected)
    {
        var btn = new Button
        {
            ToggleMode = true,
            ButtonGroup = group,
            ButtonPressed = defaultSelected,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0, Ui(58)),
            Alignment = HorizontalAlignment.Left,
        };

        if (defaultSelected) onSelected();
        btn.Toggled += on => { if (on) onSelected(); };

        var margin = new MarginContainer();
        margin.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        SetMargins(margin, 10, 10, 6, 6);
        btn.AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 2);
        margin.AddChild(vbox);

        var titleLabel = new Label { Text = title, MouseFilter = MouseFilterEnum.Ignore };
        titleLabel.AddThemeFontSizeOverride("font_size", Ui(13));
        vbox.AddChild(titleLabel);

        var descLabel = new Label
        {
            Text = description,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            MouseFilter = MouseFilterEnum.Ignore,
        };
        descLabel.AddThemeColorOverride("font_color", new Color(0.72f, 0.72f, 0.72f));
        vbox.AddChild(descLabel);

        return btn;
    }

    private static Button MakeAlgorithmCard(
        ButtonGroup group,
        string shortName,
        string fullName,
        string description,
        string badge,
        bool defaultSelected,
        Action onSelected)
    {
        var btn = new Button
        {
            ToggleMode = true,
            ButtonGroup = group,
            ButtonPressed = defaultSelected,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0, Ui(58)),
            Alignment = HorizontalAlignment.Left,
        };

        if (defaultSelected) onSelected();
        btn.Toggled += on => { if (on) onSelected(); };

        var outerMargin = new MarginContainer();
        outerMargin.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        SetMargins(outerMargin, 10, 10, 6, 6);
        btn.AddChild(outerMargin);

        var hbox = new HBoxContainer();
        hbox.AddThemeConstantOverride("separation", 10);
        outerMargin.AddChild(hbox);

        var textCol = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            MouseFilter = MouseFilterEnum.Ignore,
        };
        textCol.AddThemeConstantOverride("separation", 2);
        hbox.AddChild(textCol);

        var nameRow = new HBoxContainer { MouseFilter = MouseFilterEnum.Ignore };
        nameRow.AddThemeConstantOverride("separation", 6);
        textCol.AddChild(nameRow);

        var shortLabel = new Label { Text = shortName, MouseFilter = MouseFilterEnum.Ignore };
        shortLabel.AddThemeFontSizeOverride("font_size", Ui(13));
        nameRow.AddChild(shortLabel);

        var fullLabel = new Label
        {
            Text = fullName,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = MouseFilterEnum.Ignore,
        };
        fullLabel.AddThemeColorOverride("font_color", new Color(0.72f, 0.72f, 0.72f));
        nameRow.AddChild(fullLabel);

        var descLabel = new Label
        {
            Text = description,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            MouseFilter = MouseFilterEnum.Ignore,
        };
        descLabel.AddThemeColorOverride("font_color", new Color(0.65f, 0.65f, 0.65f));
        textCol.AddChild(descLabel);

        var badgeLabel = new Label
        {
            Text = badge,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = MouseFilterEnum.Ignore,
        };
        badgeLabel.AddThemeColorOverride("font_color", new Color(0.55f, 0.82f, 0.55f));
        hbox.AddChild(badgeLabel);

        return btn;
    }

    private static void SetMargins(MarginContainer c, int left, int right, int top, int bottom)
    {
        c.AddThemeConstantOverride("margin_left", left);
        c.AddThemeConstantOverride("margin_right", right);
        c.AddThemeConstantOverride("margin_top", top);
        c.AddThemeConstantOverride("margin_bottom", bottom);
    }

    private static void SetMargins(MarginContainer c, int all) =>
        SetMargins(c, all, all, all, all);

    private static void ClearContainer(Node container)
    {
        for (var i = container.GetChildCount() - 1; i >= 0; i--)
        {
            var child = container.GetChild(i);
            container.RemoveChild(child);
            child.QueueFree();
        }
    }
}
