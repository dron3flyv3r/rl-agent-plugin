using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

public sealed class ObservationSizeInferenceResult
{
    public Dictionary<IRLAgent, int> AgentSizes { get; } = new();
    public Dictionary<IRLAgent, ResolvedPolicyGroupBinding> AgentBindings { get; } = new();
    public Dictionary<string, int> GroupSizes { get; } = new(System.StringComparer.Ordinal);
    public List<string> Errors { get; } = new();

    public bool IsValid => Errors.Count == 0;
}
