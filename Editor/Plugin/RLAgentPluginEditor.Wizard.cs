using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Godot;
using RlAgentPlugin.Editor;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin;

public partial class RLAgentPluginEditor
{
    private readonly List<TrainingSceneReviewEntry> _wizardReviewEntries = new();
    private string _wizardScenePath = string.Empty;

    private void UpdateWizardUi(TrainingSceneValidation? validation)
    {
        _setupDock?.SetWizardState(validation, _wizardReviewEntries);
    }

    private void ResetWizardReviewState(string scenePath = "")
    {
        if (!string.Equals(_wizardScenePath, scenePath, StringComparison.Ordinal))
        {
            _wizardReviewEntries.Clear();
        }

        _wizardScenePath = scenePath;
    }

    private void SetWizardReviewState(string scenePath, IEnumerable<TrainingSceneReviewEntry> entries)
    {
        _wizardReviewEntries.Clear();
        _wizardReviewEntries.AddRange(entries);
        _wizardScenePath = scenePath;
    }

    private void OnAutofixRequested(int fixKindValue, string targetPath)
    {
        var scenePath = ResolveTrainingScenePath();
        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (_setupDock is null || editedRoot is null || string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock?.SetLaunchStatus("Wizard autofix requires an open saved scene.");
            return;
        }

        var validation = _lastValidation ?? ValidateSceneSafely(scenePath, "wizard autofix");
        var requestedKind = (TrainingSceneFixKind)fixKindValue;
        var plan = BuildFixPlans(validation, editedRoot)
            .FirstOrDefault(candidate =>
                candidate.Kind == requestedKind &&
                string.Equals(candidate.TargetPath, targetPath ?? string.Empty, StringComparison.Ordinal));

        if (plan is null)
        {
            _setupDock.SetLaunchStatus("The selected autofix is no longer applicable. Re-run validation.");
            RefreshValidationFromActiveScene();
            return;
        }

        if (!TryApplyFixPlans(scenePath, editedRoot, new[] { plan }, out var reviewEntries, out var message))
        {
            _setupDock.SetLaunchStatus(message);
            RefreshValidationFromActiveScene();
            return;
        }

        SetWizardReviewState(scenePath, reviewEntries);
        _setupDock.SetLaunchStatus(message);
        RefreshValidationFromActiveScene();
    }

    private void OnAutofixAllRequested()
    {
        var scenePath = ResolveTrainingScenePath();
        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (_setupDock is null || editedRoot is null || string.IsNullOrWhiteSpace(scenePath))
        {
            _setupDock?.SetLaunchStatus("Wizard autofix requires an open saved scene.");
            return;
        }

        var validation = _lastValidation ?? ValidateSceneSafely(scenePath, "wizard autofix");
        var plans = BuildFixPlans(validation, editedRoot);
        if (plans.Count == 0)
        {
            _setupDock.SetLaunchStatus("No safe autofixes are currently available.");
            RefreshValidationFromActiveScene();
            return;
        }

        if (!TryApplyFixPlans(scenePath, editedRoot, plans, out var reviewEntries, out var message))
        {
            _setupDock.SetLaunchStatus(message);
            RefreshValidationFromActiveScene();
            return;
        }

        SetWizardReviewState(scenePath, reviewEntries);
        _setupDock.SetLaunchStatus(message);
        RefreshValidationFromActiveScene();
    }

    private void OnReviewTargetRequested(bool isResource, string targetPath)
    {
        if (string.IsNullOrWhiteSpace(targetPath))
        {
            _setupDock?.SetLaunchStatus("No review target was provided.");
            return;
        }

        if (isResource)
        {
            var resource = ResourceLoader.Load<Resource>(targetPath);
            if (resource is null)
            {
                _setupDock?.SetLaunchStatus($"Could not load review resource: {targetPath}");
                return;
            }

            EditorInterface.Singleton.EditResource(resource);
            _setupDock?.SetLaunchStatus($"Opened resource: {targetPath}");
            return;
        }

        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (editedRoot is null)
        {
            _setupDock?.SetLaunchStatus("No edited scene is currently open.");
            return;
        }

        var node = editedRoot.GetNodeOrNull(targetPath);
        if (node is null)
        {
            _setupDock?.SetLaunchStatus($"Could not find review node: {targetPath}");
            return;
        }

        var selection = EditorInterface.Singleton.GetSelection();
        selection.Clear();
        selection.AddNode(node);
        EditorInterface.Singleton.EditNode(node);
        _setupDock?.SetLaunchStatus($"Selected node: {targetPath}");
    }

    private bool TrySaveEditedSceneForLaunch(string scenePath, string launchKind, out string errorMessage)
    {
        errorMessage = string.Empty;
        var editedRoot = EditorInterface.Singleton.GetEditedSceneRoot();
        if (editedRoot is null || !string.Equals(editedRoot.SceneFilePath, scenePath, StringComparison.Ordinal))
        {
            return true;
        }

        var saveError = EditorInterface.Singleton.SaveScene();
        if (saveError == Error.Ok)
        {
            return true;
        }

        errorMessage = $"Could not save the current scene before {launchKind}: {saveError}";
        return false;
    }

    private bool TryApplyFixPlans(
        string scenePath,
        Node editedRoot,
        IReadOnlyList<TrainingSceneFixPlan> plans,
        out List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        reviewEntries = new List<TrainingSceneReviewEntry>();
        message = "No autofixes were applied.";

        var appliedCount = 0;
        foreach (var plan in plans.OrderBy(GetFixPriority))
        {
            if (!TryApplySingleFixPlan(scenePath, editedRoot, plan, reviewEntries, out var stepMessage))
            {
                if (appliedCount == 0)
                {
                    message = stepMessage;
                    return false;
                }

                message = $"Applied {appliedCount} autofix(es). Remaining fix failed: {stepMessage}";
                return true;
            }

            appliedCount += 1;
            message = stepMessage;
        }

        if (plans.Count > 1)
        {
            message = $"Applied {appliedCount} safe autofix(es).";
        }

        return appliedCount > 0;
    }

    private bool TryApplySingleFixPlan(
        string scenePath,
        Node editedRoot,
        TrainingSceneFixPlan plan,
        List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        message = string.Empty;
        return plan.Kind switch
        {
            TrainingSceneFixKind.CreateAcademy => TryApplyCreateAcademy(editedRoot, reviewEntries, out message),
            TrainingSceneFixKind.CreateTrainingConfig => TryApplyCreateTrainingConfig(scenePath, editedRoot, plan.TargetPath, reviewEntries, out message),
            TrainingSceneFixKind.CreateRunConfig => TryApplyCreateRunConfig(scenePath, editedRoot, plan.TargetPath, reviewEntries, out message),
            TrainingSceneFixKind.CreatePolicyGroupConfig => TryApplyCreatePolicyGroupConfig(scenePath, editedRoot, plan, reviewEntries, out message),
            TrainingSceneFixKind.CreateNetworkGraph => TryApplyCreateNetworkGraph(editedRoot, plan.TargetPath, reviewEntries, out message),
            TrainingSceneFixKind.CreateTrainingAlgorithm => TryApplyCreateTrainingAlgorithm(editedRoot, plan.TargetPath, reviewEntries, out message),
            _ => Fail("Unsupported autofix.", out message),
        };
    }

    private bool TryApplyCreateAcademy(
        Node editedRoot,
        List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        if (FindAcademyNode(editedRoot) is not null)
        {
            return Fail("The scene already has an RLAcademy.", out message);
        }

        var academy = new RLAcademy
        {
            Name = FindAvailableChildName(editedRoot, "Academy"),
        };

        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Create RLAcademy", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoMethod(this, nameof(DoAttachWizardNode), editedRoot, academy, editedRoot.GetChildCount());
        undo.AddUndoMethod(this, nameof(DoDetachWizardNode), academy);
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Created academy node: {academy.Name}",
            TargetPath = academy.Name,
            TargetKind = TrainingSceneReviewTargetKind.Node,
            ActionLabel = "Select",
        });

        message = $"Created RLAcademy node '{academy.Name}'.";
        return true;
    }

    private bool TryApplyCreateTrainingConfig(
        string scenePath,
        Node editedRoot,
        string academyPath,
        List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        var academy = ResolveNodeAtPath(editedRoot, academyPath);
        if (academy is null || !IsAcademyNode(academy))
        {
            return Fail("The target academy could not be found.", out message);
        }

        if (ReadResourceProperty(academy, "TrainingConfig") is not null)
        {
            return Fail("TrainingConfig is already assigned.", out message);
        }

        var resourcePath = BuildUniqueSceneResourcePath(scenePath, ".training.tres");
        var trainingConfig = new RLTrainingConfig
        {
            ResourceName = Path.GetFileNameWithoutExtension(resourcePath),
            Algorithm = new RLPPOConfig(),
        };

        if (!TrySaveResource(trainingConfig, resourcePath, out message))
        {
            return false;
        }

        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Assign TrainingConfig", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(academy, "TrainingConfig", trainingConfig);
        undo.AddUndoProperty(academy, "TrainingConfig", academy.Get("TrainingConfig"));
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Created training config: {Path.GetFileName(resourcePath)}",
            TargetPath = resourcePath,
            TargetKind = TrainingSceneReviewTargetKind.Resource,
            ActionLabel = "Edit",
        });

        message = $"Created training config '{resourcePath}'.";
        return true;
    }

    private bool TryApplyCreateRunConfig(
        string scenePath,
        Node editedRoot,
        string academyPath,
        List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        var academy = ResolveNodeAtPath(editedRoot, academyPath);
        if (academy is null || !IsAcademyNode(academy))
        {
            return Fail("The target academy could not be found.", out message);
        }

        if (ReadResourceProperty(academy, "RunConfig") is not null)
        {
            return Fail("RunConfig is already assigned.", out message);
        }

        var resourcePath = BuildUniqueSceneResourcePath(scenePath, ".run.tres");
        var runConfig = new RLRunConfig
        {
            ResourceName = Path.GetFileNameWithoutExtension(resourcePath),
        };

        if (!TrySaveResource(runConfig, resourcePath, out message))
        {
            return false;
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

        message = $"Created run config '{resourcePath}'.";
        return true;
    }

    private bool TryApplyCreatePolicyGroupConfig(
        string scenePath,
        Node editedRoot,
        TrainingSceneFixPlan plan,
        List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        var agent = ResolveNodeAtPath(editedRoot, plan.TargetPath);
        if (agent is null || !IsAgentNode(agent))
        {
            return Fail("The target agent could not be found.", out message);
        }

        if (ReadResourceProperty(agent, "PolicyGroupConfig") is not null)
        {
            return Fail("PolicyGroupConfig is already assigned.", out message);
        }

        var suggestedAgentId = string.IsNullOrWhiteSpace(plan.SuggestedAgentId)
            ? BuildUniqueAgentId(agent.Name, CollectExistingAgentIds(editedRoot))
            : plan.SuggestedAgentId;
        var resourcePath = BuildUniqueScenePolicyResourcePath(scenePath, suggestedAgentId);
        var config = new RLPolicyGroupConfig
        {
            ResourceName = Path.GetFileNameWithoutExtension(resourcePath),
            AgentId = suggestedAgentId,
            NetworkGraph = RLNetworkGraph.CreateDefault(),
        };

        if (!TrySaveResource(config, resourcePath, out message))
        {
            return false;
        }

        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Assign PolicyGroupConfig", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(agent, "PolicyGroupConfig", config);
        undo.AddUndoProperty(agent, "PolicyGroupConfig", agent.Get("PolicyGroupConfig"));
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = $"Created policy config for '{agent.Name}': {Path.GetFileName(resourcePath)}",
            TargetPath = resourcePath,
            TargetKind = TrainingSceneReviewTargetKind.Resource,
            ActionLabel = "Edit",
        });

        message = $"Created policy config '{resourcePath}' for agent '{agent.Name}'.";
        return true;
    }

    private bool TryApplyCreateNetworkGraph(
        Node editedRoot,
        string targetPath,
        List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        var agent = ResolveNodeAtPath(editedRoot, targetPath);
        if (agent is null || !IsAgentNode(agent))
        {
            return Fail("The target agent could not be found.", out message);
        }

        var config = ReadResourceProperty(agent, "PolicyGroupConfig") as RLPolicyGroupConfig;
        if (config is null)
        {
            return Fail("PolicyGroupConfig must be assigned before creating a network graph.", out message);
        }

        if (config.ResolvedNetworkGraph is not null)
        {
            return Fail("The policy group already has a network graph.", out message);
        }

        var networkGraph = RLNetworkGraph.CreateDefault();
        var previousGraph = config.Get("NetworkGraph");
        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Create NetworkGraph", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(config, "NetworkGraph", networkGraph);
        undo.AddUndoProperty(config, "NetworkGraph", previousGraph);
        if (!string.IsNullOrWhiteSpace(config.ResourcePath))
        {
            undo.AddDoMethod(this, nameof(SaveExternalResource), config);
            undo.AddUndoMethod(this, nameof(SaveExternalResource), config);
        }
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        var targetResourcePath = string.IsNullOrWhiteSpace(config.ResourcePath) ? targetPath : config.ResourcePath;
        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = string.IsNullOrWhiteSpace(config.ResourcePath)
                ? $"Added default network graph to inline policy config on '{agent.Name}'"
                : $"Added default network graph to {Path.GetFileName(config.ResourcePath)}",
            TargetPath = targetResourcePath,
            TargetKind = string.IsNullOrWhiteSpace(config.ResourcePath)
                ? TrainingSceneReviewTargetKind.Node
                : TrainingSceneReviewTargetKind.Resource,
            ActionLabel = string.IsNullOrWhiteSpace(config.ResourcePath) ? "Select" : "Edit",
        });

        message = $"Added a default network graph for agent '{agent.Name}'.";
        return true;
    }

    private bool TryApplyCreateTrainingAlgorithm(
        Node editedRoot,
        string academyPath,
        List<TrainingSceneReviewEntry> reviewEntries,
        out string message)
    {
        var academy = ResolveNodeAtPath(editedRoot, academyPath);
        if (academy is null || !IsAcademyNode(academy))
        {
            return Fail("The target academy could not be found.", out message);
        }

        var trainingConfig = ReadResourceProperty(academy, "TrainingConfig") as RLTrainingConfig;
        if (trainingConfig is null)
        {
            return Fail("TrainingConfig must be assigned before creating an algorithm.", out message);
        }

        if (trainingConfig.Algorithm is not null)
        {
            return Fail("TrainingConfig already has an Algorithm assigned.", out message);
        }

        var algorithm = new RLPPOConfig();
        var previousAlgorithm = trainingConfig.Get("Algorithm");
        var undo = GetUndoRedo();
        undo.CreateAction("RL Wizard: Create PPO Algorithm", UndoRedo.MergeMode.Disable, editedRoot);
        undo.AddDoProperty(trainingConfig, "Algorithm", algorithm);
        undo.AddUndoProperty(trainingConfig, "Algorithm", previousAlgorithm);
        if (!string.IsNullOrWhiteSpace(trainingConfig.ResourcePath))
        {
            undo.AddDoMethod(this, nameof(SaveExternalResource), trainingConfig);
            undo.AddUndoMethod(this, nameof(SaveExternalResource), trainingConfig);
        }
        undo.CommitAction();
        EditorInterface.Singleton.MarkSceneAsUnsaved();

        var targetResourcePath = string.IsNullOrWhiteSpace(trainingConfig.ResourcePath) ? academyPath : trainingConfig.ResourcePath;
        reviewEntries.Add(new TrainingSceneReviewEntry
        {
            Title = string.IsNullOrWhiteSpace(trainingConfig.ResourcePath)
                ? "Added default PPO algorithm to inline training config"
                : $"Added default PPO algorithm to {Path.GetFileName(trainingConfig.ResourcePath)}",
            TargetPath = targetResourcePath,
            TargetKind = string.IsNullOrWhiteSpace(trainingConfig.ResourcePath)
                ? TrainingSceneReviewTargetKind.Node
                : TrainingSceneReviewTargetKind.Resource,
            ActionLabel = string.IsNullOrWhiteSpace(trainingConfig.ResourcePath) ? "Select" : "Edit",
        });

        message = "Added a default PPO algorithm to the training config.";
        return true;
    }

    private static List<TrainingSceneFixPlan> BuildFixPlans(TrainingSceneValidation validation, Node editedRoot)
    {
        var plans = new List<TrainingSceneFixPlan>();
        var reservedAgentIds = CollectExistingAgentIds(editedRoot);

        foreach (var issue in validation.Issues.Where(issue => issue.IsAutofixable))
        {
            var plan = new TrainingSceneFixPlan
            {
                Kind = issue.FixKind,
                TargetPath = issue.TargetPath,
                Label = string.IsNullOrWhiteSpace(issue.FixLabel) ? "Fix" : issue.FixLabel,
                Description = issue.Message,
            };

            if (issue.FixKind == TrainingSceneFixKind.CreatePolicyGroupConfig)
            {
                var agent = ResolveNodeAtPath(editedRoot, issue.TargetPath);
                if (agent is null)
                {
                    continue;
                }

                plan.SuggestedAgentId = BuildUniqueAgentId(agent.Name, reservedAgentIds);
                reservedAgentIds.Add(plan.SuggestedAgentId);
            }

            plans.Add(plan);
        }

        return plans
            .OrderBy(GetFixPriority)
            .ThenBy(plan => plan.TargetPath, StringComparer.Ordinal)
            .ToList();
    }

    private static int GetFixPriority(TrainingSceneFixPlan plan)
    {
        return plan.Kind switch
        {
            TrainingSceneFixKind.CreateAcademy => 0,
            TrainingSceneFixKind.CreateTrainingConfig => 1,
            TrainingSceneFixKind.CreateRunConfig => 2,
            TrainingSceneFixKind.CreatePolicyGroupConfig => 3,
            TrainingSceneFixKind.CreateNetworkGraph => 4,
            TrainingSceneFixKind.CreateTrainingAlgorithm => 5,
            _ => 100,
        };
    }

    private static Node? ResolveNodeAtPath(Node root, string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        return root.GetNodeOrNull(path);
    }

    private static Node? FindAcademyNode(Node root)
    {
        Node? academy = null;
        Traverse(root, node =>
        {
            if (academy is null && IsAcademyNode(node))
            {
                academy = node;
            }
        });

        return academy;
    }

    private static HashSet<string> CollectExistingAgentIds(Node root)
    {
        var ids = new HashSet<string>(StringComparer.Ordinal);
        Traverse(root, node =>
        {
            if (!IsAgentNode(node))
            {
                return;
            }

            var policyGroup = ReadResourceProperty(node, "PolicyGroupConfig") as RLPolicyGroupConfig;
            if (!string.IsNullOrWhiteSpace(policyGroup?.AgentId))
            {
                ids.Add(policyGroup.AgentId.Trim());
            }
        });
        return ids;
    }

    private static string BuildUniqueAgentId(string rawName, ISet<string> reservedIds)
    {
        return RLSetupWizardDefaults.MakeUniqueIdentifier(rawName, reservedIds, "agent");
    }

    private static string FindAvailableChildName(Node parent, string baseName)
    {
        var candidate = baseName;
        var suffix = 2;
        while (parent.HasNode(candidate))
        {
            candidate = $"{baseName}{suffix++}";
        }

        return candidate;
    }

    private static string BuildUniqueSceneResourcePath(string scenePath, string suffix)
    {
        var folderPath = EnsureSceneWizardResourceDirectory(scenePath);
        var sceneBaseName = Path.GetFileNameWithoutExtension(scenePath);
        var baseName = $"{sceneBaseName}{suffix}";
        return EnsureUniqueResourcePath(folderPath, baseName);
    }

    private static string BuildUniqueScenePolicyResourcePath(string scenePath, string groupId)
    {
        var folderPath = EnsureSceneWizardResourceDirectory(scenePath);
        var sceneBaseName = Path.GetFileNameWithoutExtension(scenePath);
        var sanitizedGroupId = RLSetupWizardDefaults.SanitizeIdentifier(groupId);
        if (string.IsNullOrWhiteSpace(sanitizedGroupId))
        {
            sanitizedGroupId = "agent";
        }

        var baseName = $"{sceneBaseName}.{sanitizedGroupId}.policy.tres";
        return EnsureUniqueResourcePath(folderPath, baseName);
    }

    private static string EnsureSceneWizardResourceDirectory(string scenePath)
    {
        var sceneDirectory = Path.GetDirectoryName(scenePath)?.Replace('\\', '/') ?? "res://";
        var resourceDirectory = $"{sceneDirectory.TrimEnd('/')}/RL";
        Directory.CreateDirectory(ProjectSettings.GlobalizePath(resourceDirectory));
        return resourceDirectory;
    }

    private static string EnsureUniqueResourcePath(string resourceDirectory, string fileName)
    {
        var absoluteDirectory = ProjectSettings.GlobalizePath(resourceDirectory);
        var existingFileNames = Directory.Exists(absoluteDirectory)
            ? Directory.GetFiles(absoluteDirectory).Select(Path.GetFileName).Where(name => !string.IsNullOrWhiteSpace(name))!
            : Array.Empty<string>();
        var uniqueName = RLSetupWizardDefaults.MakeUniqueFileName(fileName, existingFileNames!);
        return $"{resourceDirectory}/{uniqueName}";
    }

    private static bool TrySaveResource(Resource resource, string resourcePath, out string message)
    {
        var saveError = ResourceSaver.Save(resource, resourcePath);
        if (saveError == Error.Ok)
        {
            message = string.Empty;
            return true;
        }

        message = $"Failed to save resource '{resourcePath}': {saveError}";
        return false;
    }

    private static bool Fail(string error, out string message)
    {
        message = error;
        return false;
    }

    public void DoAttachWizardNode(Node parent, Node child, int index)
    {
        if (child.GetParent() != parent)
        {
            child.GetParent()?.RemoveChild(child);
            parent.AddChild(child);
        }

        if (index >= 0 && index < parent.GetChildCount())
        {
            parent.MoveChild(child, index);
        }

        child.Owner = parent == EditorInterface.Singleton.GetEditedSceneRoot()
            ? parent
            : parent.Owner ?? parent;
    }

    public void DoDetachWizardNode(Node child)
    {
        child.GetParent()?.RemoveChild(child);
        child.Owner = null;
    }

    public void SaveExternalResource(Resource resource)
    {
        if (string.IsNullOrWhiteSpace(resource.ResourcePath))
        {
            return;
        }

        var saveError = ResourceSaver.Save(resource, resource.ResourcePath);
        if (saveError != Error.Ok)
        {
            GD.PushWarning($"[RLAgentPluginEditor] Failed to save resource '{resource.ResourcePath}': {saveError}");
        }
    }
}
