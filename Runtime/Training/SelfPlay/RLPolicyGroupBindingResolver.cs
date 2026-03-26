using Godot;

namespace RlAgentPlugin.Runtime;

public static class RLPolicyGroupBindingResolver
{
    public static ResolvedPolicyGroupBinding? Resolve(Node sceneRoot, Node agentNode)
    {
        var agentRelativePath = sceneRoot.GetPathTo(agentNode).ToString();
        var groupConfig = ResolvePolicyGroupConfig(agentNode);

        if (groupConfig is null)
        {
            return null;
        }

        var key = ResolveExplicitGroupKey(groupConfig, agentRelativePath);
        var displayName = ResolveExplicitDisplayName(groupConfig, key);
        return new ResolvedPolicyGroupBinding
        {
            BindingKey = key,
            DisplayName = displayName,
            SafeGroupId = MakeSafeGroupId(key),
            AgentRelativePath = agentRelativePath,
            Config = groupConfig,
            ConfigPath = groupConfig.ResourcePath,
        };
    }

    public static string MakeSafeGroupId(string groupId)
    {
        var safe = new System.Text.StringBuilder(groupId.Length);
        foreach (var c in groupId)
        {
            safe.Append(char.IsLetterOrDigit(c) || c == '-' ? c : '_');
        }

        var result = safe.ToString().Trim('_');
        if (string.IsNullOrEmpty(result))
        {
            result = "default";
        }

        if (result.Length > 64)
        {
            result = result[..64];
        }

        return result;
    }

    private static RLPolicyGroupConfig? ResolvePolicyGroupConfig(Node agentNode)
    {
        if (agentNode is IRLAgent agent)
        {
            return agent.PolicyGroupConfig;
        }

        var variant = agentNode.Get("PolicyGroupConfig");
        return variant.VariantType == Variant.Type.Object
            ? variant.AsGodotObject() as RLPolicyGroupConfig
            : null;
    }

    private static string ResolveExplicitGroupKey(RLPolicyGroupConfig groupConfig, string agentRelativePath)
    {
        if (!string.IsNullOrWhiteSpace(groupConfig.ResourcePath))
        {
            return groupConfig.ResourcePath;
        }

        if (!string.IsNullOrWhiteSpace(groupConfig.ResourceName))
        {
            return groupConfig.ResourceName.Trim();
        }

        return $"__policycfg__{agentRelativePath}";
    }

    private static string ResolveExplicitDisplayName(RLPolicyGroupConfig groupConfig, string fallbackKey)
    {
        if (!string.IsNullOrWhiteSpace(groupConfig.ResourceName))
        {
            return groupConfig.ResourceName.Trim();
        }

        if (!string.IsNullOrWhiteSpace(groupConfig.ResourcePath))
        {
            return groupConfig.ResourcePath;
        }

        return fallbackKey;
    }
}
