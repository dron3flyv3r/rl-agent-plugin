using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace RlAgentPlugin.Editor;

public enum TrainingSceneIssueSeverity
{
    Warning = 0,
    Blocking = 1,
}

public enum TrainingSceneIssueCode
{
    GenericValidationError = 0,
    MissingAcademy = 1,
    MissingTrainingConfig = 2,
    MissingRunConfig = 3,
    MissingPolicyGroupConfig = 4,
    MissingNetworkGraph = 5,
    MissingTrainingAlgorithm = 6,
}

public enum TrainingSceneFixKind
{
    None = 0,
    CreateAcademy = 1,
    CreateTrainingConfig = 2,
    CreateRunConfig = 3,
    CreatePolicyGroupConfig = 4,
    CreateNetworkGraph = 5,
    CreateTrainingAlgorithm = 6,
}

public enum TrainingSceneReviewTargetKind
{
    Node = 0,
    Resource = 1,
}

public sealed class TrainingSceneIssue
{
    public TrainingSceneIssueCode Code { get; set; } = TrainingSceneIssueCode.GenericValidationError;
    public TrainingSceneIssueSeverity Severity { get; set; } = TrainingSceneIssueSeverity.Blocking;
    public string Message { get; set; } = string.Empty;
    public string TargetPath { get; set; } = string.Empty;
    public bool IsAutofixable { get; set; }
    public TrainingSceneFixKind FixKind { get; set; } = TrainingSceneFixKind.None;
    public string FixLabel { get; set; } = string.Empty;
}

public sealed class TrainingSceneFixPlan
{
    public TrainingSceneFixKind Kind { get; set; } = TrainingSceneFixKind.None;
    public string TargetPath { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string SuggestedAgentId { get; set; } = string.Empty;
}

public sealed class TrainingSceneReviewEntry
{
    public string Title { get; set; } = string.Empty;
    public string TargetPath { get; set; } = string.Empty;
    public TrainingSceneReviewTargetKind TargetKind { get; set; }
    public string ActionLabel { get; set; } = string.Empty;
}

public sealed class PolicyGroupSummary
{
    public string GroupId { get; set; } = string.Empty;
    public int AgentCount { get; set; }
    public int ObservationSize { get; set; }
    public int ActionCount { get; set; }
    public bool IsContinuous { get; set; }
    public int ContinuousActionDimensions { get; set; }
    public bool UsesExplicitConfig { get; set; }
    public string PolicyConfigPath { get; set; } = string.Empty;
    public bool SelfPlay { get; set; }
    public string OpponentGroupId { get; set; } = string.Empty;
    public float HistoricalOpponentRate { get; set; }
    public int FrozenCheckpointInterval { get; set; } = 10;
    public List<string> AgentPaths { get; } = new();
}

public sealed class TrainingSceneValidation
{
    public string ScenePath { get; set; } = string.Empty;
    public string AcademyPath { get; set; } = string.Empty;
    public string TrainingConfigPath { get; set; } = string.Empty;
    public string NetworkConfigPath { get; set; } = string.Empty;
    public string RunPrefix { get; set; } = string.Empty;
    public int CheckpointInterval { get; set; } = 10;
    public float SimulationSpeed { get; set; } = 1.0f;
    public int ActionRepeat { get; set; } = 1;
    public int BatchSize { get; set; } = 1;
    public bool EnableSpyOverlay { get; set; }
    public bool HasCurriculum { get; set; }
    public bool HasSelfPlayPairings { get; set; }
    public int ExpectedActionCount { get; set; }
    public int TrainAgentCount { get; set; }
    public int InferenceAgentCount { get; set; }
    public bool IsValid { get; set; }
    public List<string> AgentNames { get; } = new();
    public List<string> AgentGroups { get; } = new();
    public List<string> ExportNames { get; } = new();
    public List<string> ExportGroups { get; } = new();
    public List<string> Errors { get; } = new();
    public List<TrainingSceneIssue> Issues { get; } = new();
    public List<PolicyGroupSummary> PolicyGroups { get; } = new();

    public int AutofixableIssueCount => Issues.Count(issue => issue.IsAutofixable);

    public bool HasBlockingIssues => Issues.Any(issue => issue.Severity == TrainingSceneIssueSeverity.Blocking);

    public void AddIssue(
        TrainingSceneIssueCode code,
        string message,
        string targetPath = "",
        TrainingSceneIssueSeverity severity = TrainingSceneIssueSeverity.Blocking,
        bool isAutofixable = false,
        TrainingSceneFixKind fixKind = TrainingSceneFixKind.None,
        string fixLabel = "")
    {
        var issue = new TrainingSceneIssue
        {
            Code = code,
            Message = message,
            TargetPath = targetPath ?? string.Empty,
            Severity = severity,
            IsAutofixable = isAutofixable,
            FixKind = fixKind,
            FixLabel = fixLabel ?? string.Empty,
        };

        Issues.Add(issue);
        if (severity == TrainingSceneIssueSeverity.Blocking)
        {
            Errors.Add(message);
        }
    }

    public void AddBlockingError(string message)
    {
        AddIssue(TrainingSceneIssueCode.GenericValidationError, message);
    }

    public string BuildSummary()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Scene: {ScenePath}");
        builder.AppendLine(IsValid ? "Validation: ready to train" : "Validation: blocked");
        if (!string.IsNullOrWhiteSpace(AcademyPath))
        {
            builder.AppendLine($"Academy: {AcademyPath}");
        }

        builder.AppendLine($"Agents: {AgentNames.Count}");
        builder.AppendLine($"Train Agents: {TrainAgentCount}");
        builder.AppendLine($"Inference Agents: {InferenceAgentCount}");

        if (PolicyGroups.Count > 1)
        {
            builder.AppendLine($"Policy Groups: {PolicyGroups.Count}");
            foreach (var group in PolicyGroups)
            {
                var actionInfo = group.IsContinuous
                    ? $"{group.ContinuousActionDimensions}D continuous"
                    : $"{group.ActionCount} discrete";
                var sourceInfo = group.UsesExplicitConfig
                    ? $"explicit config ({group.PolicyConfigPath})"
                    : "inline policy-group config";
                var selfPlayInfo = group.SelfPlay
                    ? $", self-play vs '{group.OpponentGroupId}' (historical={group.HistoricalOpponentRate:0.00}, freeze={group.FrozenCheckpointInterval})"
                    : string.Empty;
                builder.AppendLine($"  '{group.GroupId}': {group.AgentCount} agent(s), obs={group.ObservationSize}, {actionInfo}, {sourceInfo}{selfPlayInfo}");
                foreach (var agentPath in group.AgentPaths)
                {
                    builder.AppendLine($"    - {agentPath}");
                }
            }
        }

        foreach (var issue in Issues.OrderByDescending(issue => issue.Severity).ThenBy(issue => issue.Code))
        {
            var prefix = issue.Severity == TrainingSceneIssueSeverity.Blocking ? "error" : "warning";
            builder.AppendLine($"- [{prefix}] {issue.Message}");
        }

        return builder.ToString().TrimEnd();
    }
}
