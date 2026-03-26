using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

public static class TrainerFactory
{
    private static readonly Dictionary<string, Func<PolicyGroupConfig, ITrainer>> _customFactories =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Register a custom trainer factory. Call this before training starts (e.g. in your scene's _Ready).
    /// The <paramref name="id"/> must match <c>RLTrainerConfig.CustomTrainerId</c> on the academy.
    /// </summary>
    public static void Register(string id, Func<PolicyGroupConfig, ITrainer> factory)
    {
        if (string.IsNullOrWhiteSpace(id))
            throw new ArgumentException("Custom trainer id cannot be blank.", nameof(id));
        _customFactories[id.Trim()] = factory ?? throw new ArgumentNullException(nameof(factory));
    }

    /// <summary>Remove a previously registered custom trainer factory.</summary>
    public static void Unregister(string id)
    {
        if (!string.IsNullOrWhiteSpace(id))
            _customFactories.Remove(id.Trim());
    }

    public static ITrainer Create(PolicyGroupConfig config)
    {
        if (config.Algorithm == RLAlgorithmKind.Custom)
        {
            var id = config.CustomTrainerId?.Trim() ?? string.Empty;
            if (_customFactories.TryGetValue(id, out var factory))
                return factory(config);
            throw new InvalidOperationException(
                $"[TrainerFactory] No custom trainer registered with id '{id}'. " +
                $"Call TrainerFactory.Register(\"{id}\", ...) before training starts.");
        }

        return config.Algorithm switch
        {
            RLAlgorithmKind.PPO => new PpoTrainer(config),
            RLAlgorithmKind.SAC => new SacTrainer(config),
            _ => throw new NotSupportedException($"Unknown algorithm: {config.Algorithm}"),
        };
    }
}
