using System.Collections.Generic;
using System.Text;

namespace RlAgentPlugin.Editor;

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
    public List<string> AgentNames  { get; } = new();
    /// <summary>Safe group ID for each entry in AgentNames (same index). Used to locate per-agent checkpoint files.</summary>
    public List<string> AgentGroups { get; } = new();
    public List<string> ExportNames { get; } = new();
    public List<string> ExportGroups { get; } = new();
    public List<string> Errors { get; } = new();
    public List<PolicyGroupSummary> PolicyGroups { get; } = new();

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

        foreach (var error in Errors)
        {
            builder.AppendLine($"- {error}");
        }

        return builder.ToString().TrimEnd();
    }
}
