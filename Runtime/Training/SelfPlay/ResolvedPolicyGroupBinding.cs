namespace RlAgentPlugin.Runtime;

public sealed class ResolvedPolicyGroupBinding
{
    public string BindingKey { get; init; } = string.Empty;
    public string DisplayName { get; init; } = string.Empty;
    public string SafeGroupId { get; init; } = string.Empty;
    public string AgentRelativePath { get; init; } = string.Empty;
    public RLPolicyGroupConfig? Config { get; init; }
    public string ConfigPath { get; init; } = string.Empty;

    public bool UsesExplicitConfig => Config is not null;
}
