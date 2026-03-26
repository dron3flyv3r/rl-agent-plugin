using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public static class ObservationSizeInference
{
    public static ObservationSizeInferenceResult Infer(Node sceneRoot, IEnumerable<IRLAgent> agents, bool resetEpisodes = true)
    {
        var result = new ObservationSizeInferenceResult();
        var firstSizeByGroup = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var agent in agents)
        {
            var binding = RLPolicyGroupBindingResolver.Resolve(sceneRoot, agent.AsNode());
            if (binding == null)
            {
                result.Errors.Add($"Agent '{sceneRoot.GetPathTo(agent.AsNode())}': failed to resolve policy group binding.");
                continue;
            }

            result.AgentBindings[agent] = binding;

            if (!TryInferAgentObservationSize(agent, out var observationSize, out var error, resetEpisodes))
            {
                result.Errors.Add(BuildAgentError(sceneRoot, agent, error));
                continue;
            }

            result.AgentSizes[agent] = observationSize;
            if (observationSize <= 0)
            {
                result.Errors.Add(
                    $"Group '{binding.DisplayName}': agent '{sceneRoot.GetPathTo(agent.AsNode())}' did not emit a non-zero observation vector.");
                continue;
            }

            if (firstSizeByGroup.TryGetValue(binding.BindingKey, out var firstSize))
            {
                if (firstSize != observationSize)
                {
                    result.Errors.Add(
                        $"Group '{binding.DisplayName}': agent '{sceneRoot.GetPathTo(agent.AsNode())}' emitted {observationSize} observations, " +
                        $"expected {firstSize}.");
                }

                continue;
            }

            firstSizeByGroup[binding.BindingKey] = observationSize;
            result.GroupSizes[binding.BindingKey] = observationSize;
        }

        return result;
    }

    public static bool TryInferAgentObservationSize(
        IRLAgent agent,
        out int observationSize,
        out string error,
        bool resetEpisode = true)
    {
        try
        {
            if (resetEpisode)
            {
                agent.ResetEpisode();
            }

            observationSize = agent.CollectObservationArray().Length;
            error = string.Empty;
            return true;
        }
        catch (Exception exception)
        {
            observationSize = 0;
            error = exception.Message;
            return false;
        }
    }

    private static string BuildAgentError(Node sceneRoot, IRLAgent agent, string error)
    {
        var agentPath = sceneRoot.GetPathTo(agent.AsNode());
        return $"Agent '{agentPath}': observation inference failed: {error}";
    }
}
