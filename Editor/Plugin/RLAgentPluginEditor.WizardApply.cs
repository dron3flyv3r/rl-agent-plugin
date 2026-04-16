using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Godot;
using Godot.Collections;
using RlAgentPlugin.Editor;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin;

// Partial class — handles opening and applying the setup wizard.
public partial class RLAgentPluginEditor
{
    private RLSetupWizardWindow? _wizardWindow;

    // ── Entry point: open the wizard ─────────────────────────────────────────

    private void OnWizardRequested()
    {
        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (editedRoot is null)
        {
            _setupDock?.SetLaunchStatus("Open and save a scene before using the setup wizard.");
            return;
        }

        var scenePath = ResolveTrainingScenePath();
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock?.SetLaunchStatus("Save the current scene before using the setup wizard.");
            return;
        }

        // Close any previously open wizard window
        if (_wizardWindow is not null && IsInstanceValid(_wizardWindow))
        {
            _wizardWindow.QueueFree();
            _wizardWindow = null;
        }

        _wizardWindow = new RLSetupWizardWindow(editedRoot);
        _wizardWindow.ApplyRequested += state => ApplyWizardState(state, scenePath, editedRoot);
        _wizardWindow.TreeExiting += () => _wizardWindow = null;

        Node? popupHost = _setupDock ?? EditorInterface.Singleton.GetBaseControl();
        if (popupHost is null)
        {
            _setupDock?.SetLaunchStatus("Wizard: could not resolve the editor host window.");
            _wizardWindow.QueueFree();
            _wizardWindow = null;
            return;
        }

        _wizardWindow.PopupExclusiveCentered(
            popupHost,
            new Vector2I(EditorUiScale.Px(660), EditorUiScale.Px(560)));
    }

    // ── Apply wizard state ────────────────────────────────────────────────────

    private void ApplyWizardState(RLSetupWizardState state, string scenePath, Node editedRoot)
    {
        var reviewEntries = new List<TrainingSceneReviewEntry>();

        // Step 1 — ensure Academy exists and set MaxEpisodeSteps
        EnsureWizardAcademy(state, editedRoot, reviewEntries);

        var academy = FindAcademyNode(editedRoot);
        if (academy is null)
        {
            _setupDock?.SetLaunchStatus("Wizard: failed to create or find Academy node.");
            RefreshValidationFromActiveScene();
            return;
        }

        // Step 2 — pre-generate shared scripts for modes where all nodes in a group share one
        // observation/action contract. Individual-policy mode generates a script per node inside
        // AttachFreshAgentNodes (sharedScript = null). Existing mode skips script generation.
        string? sharedScriptA = null;
        string? sharedScriptB = null;
        if (state.Mode == WizardMode.Fresh)
        {
            var sceneBaseName = Path.GetFileNameWithoutExtension(scenePath);
            switch (state.TrainingMode)
            {
                case WizardTrainingMode.SingleAgent:
                case WizardTrainingMode.SharedPolicy:
                    sharedScriptA = GenerateWizardGroupScript(scenePath, state.Dimension, $"{sceneBaseName}Agent", reviewEntries);
                    break;
                case WizardTrainingMode.SelfPlay:
                    sharedScriptA = GenerateWizardGroupScript(scenePath, state.Dimension, $"{sceneBaseName}AgentA", reviewEntries);
                    sharedScriptB = GenerateWizardGroupScript(scenePath, state.Dimension, $"{sceneBaseName}AgentB", reviewEntries);
                    break;
                // IndividualPolicies: each node generates its own script; sharedScript stays null
            }
        }

        // Insert RLAgent nodes (Fresh) or use existing nodes directly.
        var groupABaseName = state.TrainingMode == WizardTrainingMode.SelfPlay ? "AgentA" : "Agent";
        var agentsA = state.Mode == WizardMode.Fresh
            ? AttachFreshAgentNodes(state, state.GroupANodes, scenePath, editedRoot, reviewEntries, sharedScriptA, groupABaseName)
            : state.GroupANodes;

        var agentsB = state.TrainingMode == WizardTrainingMode.SelfPlay
            ? (state.Mode == WizardMode.Fresh
                ? AttachFreshAgentNodes(state, state.GroupBNodes, scenePath, editedRoot, reviewEntries, sharedScriptB, "AgentB")
                : state.GroupBNodes)
            : new List<Node>();

        // Step 3 — create RunConfig with wizard settings
        EnsureWizardRunConfig(state, scenePath, editedRoot, academy, reviewEntries);

        // Step 4 — create TrainingConfig + Algorithm
        EnsureWizardTrainingConfig(state, scenePath, editedRoot, academy, reviewEntries);

        // Step 5 — create PolicyGroupConfig(s) and NetworkGraphs
        if (state.TrainingMode == WizardTrainingMode.IndividualPolicies)
        {
            // Each agent gets its own independent PolicyGroupConfig.
            foreach (var agent in agentsA)
            {
                EnsureWizardPolicyGroupConfig(state, scenePath, editedRoot, new List<Node> { agent }, agent.Name.ToString(), reviewEntries);
            }
        }
        else
        {
            var configA = EnsureWizardPolicyGroupConfig(state, scenePath, editedRoot, agentsA, "a", reviewEntries);
            var configB = state.TrainingMode == WizardTrainingMode.SelfPlay
                ? EnsureWizardPolicyGroupConfig(state, scenePath, editedRoot, agentsB, "b", reviewEntries)
                : null;

            // Step 6 — wire self-play if needed
            if (state.TrainingMode == WizardTrainingMode.SelfPlay && configA is not null && configB is not null)
            {
                WireWizardSelfPlay(editedRoot, academy, configA, configB, reviewEntries);
            }
        }

        SetWizardReviewState(scenePath, reviewEntries);
        _setupDock?.SetLaunchStatus("Wizard setup applied. Review the created resources in the Review section.");
        RefreshValidationFromActiveScene();
    }

    // ── Step implementations ──────────────────────────────────────────────────

    private void EnsureWizardAcademy(RLSetupWizardState state, Node editedRoot, List<TrainingSceneReviewEntry> reviewEntries)
    {
        if (FindAcademyNode(editedRoot) is not null) return;

        // Reuse existing academy creation (already handles undo/redo)
        TryApplyCreateAcademy(editedRoot, reviewEntries, out _);

        // Set MaxEpisodeSteps if a limit was specified
        if (state.MaxEpisodeSteps > 0)
        {
            var academy = FindAcademyNode(editedRoot);
            if (academy is not null)
            {
                var undo = GetUndoRedo();
                undo.CreateAction("RL Wizard: Set MaxEpisodeSteps", UndoRedo.MergeMode.Disable, editedRoot);
                undo.AddDoProperty(academy, "MaxEpisodeSteps", state.MaxEpisodeSteps);
                undo.AddUndoProperty(academy, "MaxEpisodeSteps", 0);
                undo.CommitAction();
            }
        }
    }

    private List<Node> AttachFreshAgentNodes(
        RLSetupWizardState state,
        List<Node> parentNodes,
        string scenePath,
        Node editedRoot,
        List<TrainingSceneReviewEntry> reviewEntries,
        string? sharedScriptPath = null,
        string agentBaseName = "Agent")
    {
        var agents = new List<Node>();
        var insertionParents = parentNodes.Count > 0
            ? parentNodes
            : new List<Node> { editedRoot };

        for (var i = 0; i < insertionParents.Count; i++)
        {
            var parent = insertionParents[i];

            // IndividualPolicies: sequential Agent1, Agent2 … per parent node.
            // All other modes: use the provided base name (e.g. "Agent", "AgentA", "AgentB").
            var nodeBaseName = sharedScriptPath is null ? $"Agent{i + 1}" : agentBaseName;
            var agentName = FindAvailableChildName(parent, nodeBaseName);

            // Use the shared script (SingleAgent / SharedPolicy / SelfPlay), or generate
            // a unique script per node (IndividualPolicies).
            string? scriptPath = sharedScriptPath;
            if (scriptPath is null)
            {
                var sceneBaseName = Path.GetFileNameWithoutExtension(scenePath);
                scriptPath = GenerateWizardGroupScript(
                    scenePath, state.Dimension,
                    $"{sceneBaseName}Agent{i + 1}",
                    reviewEntries);
            }

            var undo = GetUndoRedo();
            undo.CreateAction($"RL Wizard: Add agent to {parent.Name}", UndoRedo.MergeMode.Disable, editedRoot);
            undo.AddDoMethod(
                this,
                nameof(DoCreateWizardAgentNode),
                parent,
                agentName,
                state.Dimension == WizardDimension.TwoD,
                scriptPath ?? string.Empty,
                parent.GetChildCount());
            undo.AddUndoMethod(this, nameof(DoRemoveWizardChildByName), parent, agentName);
            undo.CommitAction();

            var agentNode = parent.GetNodeOrNull<Node>(agentName);
            var agentPath = parent == editedRoot
                ? $"/{agentName}"
                : $"{editedRoot.GetPathTo(parent)}/{agentName}";

            reviewEntries.Add(new TrainingSceneReviewEntry
            {
                Title = $"Added agent node: {agentName} → {parent.Name}",
                TargetPath = agentPath,
                TargetKind = TrainingSceneReviewTargetKind.Node,
                ActionLabel = "Select",
            });

            if (agentNode is not null)
            {
                agents.Add(agentNode);

                if (!string.IsNullOrWhiteSpace(scriptPath))
                {
                    var nodePath = editedRoot.GetPathTo(agentNode).ToString();
                    RegisterPendingScriptAssignment(scenePath, nodePath, scriptPath);
                }
            }
        }

        return agents;
    }

    private void EnsureWizardRunConfig(
        RLSetupWizardState state,
        string scenePath,
        Node editedRoot,
        Node academy,
        List<TrainingSceneReviewEntry> reviewEntries)
    {
        var existing = ReadResourceProperty(academy, "RunConfig") as RLRunConfig;
        if (existing is not null)
        {
            // RunConfig exists — just patch the wizard-relevant fields
            ApplyRunConfigSettings(existing, state, editedRoot);
            return;
        }

        var resourcePath = BuildUniqueSceneResourcePath(scenePath, ".run.tres");
        var runConfig = new RLRunConfig
        {
            ResourceName = Path.GetFileNameWithoutExtension(resourcePath),
            ActionRepeat = state.ActionRepeat,
            SimulationSpeed = state.SimulationSpeed,
            CheckpointInterval = state.CheckpointInterval,
            RunPrefix = state.RunPrefix,
        };

        if (!TrySaveResource(runConfig, resourcePath, out var saveError))
        {
            GD.PushWarning($"[RLWizard] Failed to save RunConfig: {saveError}");
            return;
        }

        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Assign RunConfig", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(academy, "RunConfig", runConfig);
        undo.AddUndoProperty(academy, "RunConfig", academy.Get("RunConfig"));
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Created run config: {Path.GetFileName(resourcePath)}",
            TargetPath = resourcePath,
            TargetKind = TrainingSceneReviewTargetKind.Resource,
            ActionLabel = "Edit",
        });
    }

    private void ApplyRunConfigSettings(RLRunConfig config, RLSetupWizardState state, Node editedRoot)
    {
        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Configure RunConfig", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(config, "ActionRepeat", state.ActionRepeat);
        undo.AddUndoProperty(config, "ActionRepeat", config.ActionRepeat);
        undo.AddDoProperty(config, "SimulationSpeed", state.SimulationSpeed);
        undo.AddUndoProperty(config, "SimulationSpeed", config.SimulationSpeed);
        undo.AddDoProperty(config, "CheckpointInterval", state.CheckpointInterval);
        undo.AddUndoProperty(config, "CheckpointInterval", config.CheckpointInterval);
        if (!string.IsNullOrWhiteSpace(state.RunPrefix))
        {
            undo.AddDoProperty(config, "RunPrefix", state.RunPrefix);
            undo.AddUndoProperty(config, "RunPrefix", config.RunPrefix);
        }

        if (!string.IsNullOrWhiteSpace(config.ResourcePath))
        {
            undo.AddDoMethod(this, nameof(SaveExternalResource), config);
            undo.AddUndoMethod(this, nameof(SaveExternalResource), config);
        }

        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();
    }

    private void EnsureWizardTrainingConfig(
        RLSetupWizardState state,
        string scenePath,
        Node editedRoot,
        Node academy,
        List<TrainingSceneReviewEntry> reviewEntries)
    {
        var existing = ReadResourceProperty(academy, "TrainingConfig") as RLTrainingConfig;
        if (existing is not null)
        {
            // TrainingConfig exists — replace or set the algorithm
            ReplaceAlgorithmOnTrainingConfig(state, existing, editedRoot, reviewEntries, academy);
            return;
        }

        var resourcePath = BuildUniqueSceneResourcePath(scenePath, ".training.tres");
        var trainingConfig = new RLTrainingConfig
        {
            ResourceName = Path.GetFileNameWithoutExtension(resourcePath),
            Algorithm = BuildAlgorithmConfig(state),
        };

        if (!TrySaveResource(trainingConfig, resourcePath, out var saveError))
        {
            GD.PushWarning($"[RLWizard] Failed to save TrainingConfig: {saveError}");
            return;
        }

        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Assign TrainingConfig", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(academy, "TrainingConfig", trainingConfig);
        undo.AddUndoProperty(academy, "TrainingConfig", academy.Get("TrainingConfig"));
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Created training config ({state.Algorithm}): {Path.GetFileName(resourcePath)}",
            TargetPath = resourcePath,
            TargetKind = TrainingSceneReviewTargetKind.Resource,
            ActionLabel = "Edit",
        });
    }

    private void ReplaceAlgorithmOnTrainingConfig(
        RLSetupWizardState state,
        RLTrainingConfig trainingConfig,
        Node editedRoot,
        List<TrainingSceneReviewEntry> reviewEntries,
        Node academy)
    {
        var newAlgorithm = BuildAlgorithmConfig(state);
        var previousAlgorithm = trainingConfig.Get("Algorithm");

        var undo = GetUndoRedo();
        undo.CreateAction($"RL Wizard: Set {state.Algorithm} algorithm", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(trainingConfig, "Algorithm", newAlgorithm);
        undo.AddUndoProperty(trainingConfig, "Algorithm", previousAlgorithm);
        if (!string.IsNullOrWhiteSpace(trainingConfig.ResourcePath))
        {
            undo.AddDoMethod(this, nameof(SaveExternalResource), trainingConfig);
            undo.AddUndoMethod(this, nameof(SaveExternalResource), trainingConfig);
        }

        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        var targetPath = string.IsNullOrWhiteSpace(trainingConfig.ResourcePath)
            ? academy.GetPath().ToString()
            : trainingConfig.ResourcePath;
        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Set algorithm to {state.Algorithm}",
            TargetPath = targetPath,
            TargetKind = string.IsNullOrWhiteSpace(trainingConfig.ResourcePath)
                ? TrainingSceneReviewTargetKind.Node
                : TrainingSceneReviewTargetKind.Resource,
            ActionLabel = "Edit",
        });
    }

    private static RLAlgorithmConfig BuildAlgorithmConfig(RLSetupWizardState state) =>
        state.Algorithm switch
        {
            WizardAlgorithm.SAC => new RLSACConfig(),
            WizardAlgorithm.DQN => new RLDQNConfig(),
            WizardAlgorithm.A2C => new RLA2CConfig(),
            _ => new RLPPOConfig(),
        };

    private RLPolicyGroupConfig? EnsureWizardPolicyGroupConfig(
        RLSetupWizardState state,
        string scenePath,
        Node editedRoot,
        List<Node> agentNodes,
        string groupSuffix,
        List<TrainingSceneReviewEntry> reviewEntries)
    {
        if (agentNodes.Count == 0) return null;

        // Derive a stable agent ID from the first node's name
        var existingIds = CollectExistingAgentIds(editedRoot);
        var baseId = agentNodes.First().Name.ToString();
        var agentId = BuildUniqueAgentId(baseId, existingIds);

        var networkGraph = BuildNetworkGraphFromState(state);
        var resourcePath = BuildUniqueScenePolicyResourcePath(scenePath, agentId);

        var policyConfig = new RLPolicyGroupConfig
        {
            ResourceName = Path.GetFileNameWithoutExtension(resourcePath),
            AgentId = agentId,
            MaxEpisodeSteps = state.MaxEpisodeSteps,
            NetworkGraph = networkGraph,
        };

        if (!TrySaveResource(policyConfig, resourcePath, out var saveError))
        {
            GD.PushWarning($"[RLWizard] Failed to save PolicyGroupConfig: {saveError}");
            return null;
        }

        // Assign the same config resource to ALL agents in the group
        foreach (var agentNode in agentNodes)
        {
            var undo = GetUndoRedo();
            undo.CreateAction($"RL Wizard: Assign PolicyGroupConfig to {agentNode.Name}", UndoRedo.MergeMode.Disable, editedRoot);
            undo.AddDoProperty(agentNode, "PolicyGroupConfig", policyConfig);
            undo.AddUndoProperty(agentNode, "PolicyGroupConfig", agentNode.Get("PolicyGroupConfig"));
            undo.CommitAction();
        }

        EditorInterface.Singleton.MarkSceneAsUnsaved();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Created policy config '{agentId}' ({agentNodes.Count} agent(s)): {Path.GetFileName(resourcePath)}",
            TargetPath = resourcePath,
            TargetKind = TrainingSceneReviewTargetKind.Resource,
            ActionLabel = "Edit",
        });

        return policyConfig;
    }

    private void WireWizardSelfPlay(
        Node editedRoot,
        Node academy,
        RLPolicyGroupConfig configA,
        RLPolicyGroupConfig configB,
        List<TrainingSceneReviewEntry> reviewEntries)
    {
        var pairing = new RLPolicyPairingConfig
        {
            PairingId = $"{configA.AgentId}_vs_{configB.AgentId}",
            GroupA = configA,
            GroupB = configB,
            TrainGroupA = true,
            TrainGroupB = true,
        };

        var selfPlayConfig = new RLSelfPlayConfig();
        selfPlayConfig.Pairings.Add(pairing);

        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Configure SelfPlay", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(academy, "SelfPlay", selfPlayConfig);
        undo.AddUndoProperty(academy, "SelfPlay", academy.Get("SelfPlay"));
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Configured self-play: {configA.AgentId} vs {configB.AgentId}",
            TargetPath = academy.GetPath().ToString(),
            TargetKind = TrainingSceneReviewTargetKind.Node,
            ActionLabel = "Select",
        });
    }

    // ── Network graph builder ─────────────────────────────────────────────────

    private string? GenerateWizardGroupScript(
        string scenePath,
        WizardDimension dimension,
        string scriptStem,
        List<TrainingSceneReviewEntry> reviewEntries)
    {
        var baseClassName = dimension == WizardDimension.TwoD ? nameof(RLAgent2D) : nameof(RLAgent3D);
        var scriptPath = BuildUniqueSceneScriptPath(scenePath, scriptStem);
        var className = BuildWizardAgentClassName(scriptPath);
        var scriptContent = BuildWizardAgentScriptTemplate(className, baseClassName);

        if (!TryWriteTextFile(scriptPath, scriptContent, out var errorMessage))
        {
            GD.PushWarning($"[RLWizard] {errorMessage}");
            return null;
        }

        EditorInterface.Singleton.GetResourceFilesystem()?.Scan();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Generated agent script: {Path.GetFileName(scriptPath)}",
            TargetPath = scriptPath,
            TargetKind = TrainingSceneReviewTargetKind.Resource,
            ActionLabel = "Edit",
        });

        return scriptPath;
    }

    private static string BuildWizardAgentClassName(string scriptPath)
    {
        var fileStem = Path.GetFileNameWithoutExtension(scriptPath);
        var builder = new StringBuilder(fileStem.Length + 5);
        var capitalizeNext = true;

        foreach (var character in fileStem)
        {
            if (!char.IsLetterOrDigit(character))
            {
                capitalizeNext = true;
                continue;
            }

            builder.Append(capitalizeNext
                ? char.ToUpperInvariant(character)
                : character);
            capitalizeNext = false;
        }

        if (builder.Length == 0 || !char.IsLetter(builder[0]))
        {
            builder.Insert(0, "Generated");
        }

        return builder.ToString();
    }

    private static string BuildWizardAgentScriptTemplate(string className, string baseClassName)
    {
        return
$@"using Godot;
using RlAgentPlugin.Runtime;

public partial class {className} : {baseClassName}
{{
    public override void DefineActions(ActionSpaceBuilder builder)
    {{
        // Define your action space here.
        // Example:
        // builder.AddDiscreteBranch(3);
    }}

    public override void CollectObservations(ObservationBuffer buffer)
    {{
        // Add observations here.
        // Example:
        // buffer.Add(GlobalPosition.X);
    }}

    public override void OnEpisodeBegin()
    {{
        // Reset your environment state here.
    }}

    public override void OnStep()
    {{
        // Compute rewards and episode termination here.
        // Example:
        // AddReward(0.01f);
        // EndEpisode();
    }}

    protected override void OnHumanInput()
    {{
        // Optional: read keyboard or controller input here.
    }}

    protected override void OnScriptedInput()
    {{
        // Optional: implement a heuristic policy here.
    }}
}}
";
    }

    private static RLNetworkGraph BuildNetworkGraphFromState(RLSetupWizardState state)
    {
        var (layerCount, layerSize) = state.NetworkPreset switch
        {
            WizardNetworkPreset.Tiny   => (1, 32),
            WizardNetworkPreset.Small  => (2, 64),
            WizardNetworkPreset.Medium => (2, 128),
            WizardNetworkPreset.Large  => (3, 256),
            _                          => (2, 64),
        };

        var layers = new Array<Resource>();
        for (var i = 0; i < layerCount; i++)
        {
            layers.Add(new RLDenseLayerDef { Size = layerSize, Activation = state.Activation });
        }

        return new RLNetworkGraph { TrunkLayers = layers, Optimizer = state.Optimizer };
    }
}
