using System;
using Godot.Collections;
using Godot;

namespace RlAgentPlugin.Runtime;

public static class InferencePolicyFactory
{
    private static readonly System.Collections.Generic.Dictionary<string, Func<RLCheckpoint, RLNetworkGraph?, IInferencePolicy>> _customFactories =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Register a custom inference policy factory keyed by algorithm name.
    /// The <paramref name="algorithmName"/> should match the string stored in <c>RLCheckpoint.Algorithm</c>
    /// (e.g. the value your custom <see cref="ITrainer"/> writes to checkpoints via
    /// <see cref="CheckpointMetadataBuilder"/> or directly).
    /// Custom factories take priority over built-in PPO/SAC handling.
    /// </summary>
    public static void Register(string algorithmName, Func<RLCheckpoint, RLNetworkGraph?, IInferencePolicy> factory)
    {
        if (string.IsNullOrWhiteSpace(algorithmName))
            throw new ArgumentException("Algorithm name cannot be blank.", nameof(algorithmName));
        _customFactories[algorithmName.Trim()] = factory ?? throw new ArgumentNullException(nameof(factory));
    }

    /// <summary>Remove a previously registered custom inference policy factory.</summary>
    public static void Unregister(string algorithmName)
    {
        if (!string.IsNullOrWhiteSpace(algorithmName))
            _customFactories.Remove(algorithmName.Trim());
    }

    public static IInferencePolicy Create(RLCheckpoint checkpoint, RLNetworkGraph? fallbackGraph = null, bool stochasticInference = false)
    {
        var graph = ReconstructGraph(checkpoint, fallbackGraph);

        // Custom factories take priority over built-in handlers.
        if (_customFactories.TryGetValue(checkpoint.Algorithm, out var customFactory))
            return customFactory(checkpoint, graph);

        if (string.Equals(checkpoint.Algorithm, RLCheckpoint.SacAlgorithm, StringComparison.OrdinalIgnoreCase))
        {
            return new SacInferencePolicy(
                checkpoint.ObsSpec?.TotalSize ?? checkpoint.ObservationSize,
                checkpoint.ContinuousActionDimensions > 0
                    ? checkpoint.ContinuousActionDimensions
                    : checkpoint.DiscreteActionCount,
                checkpoint.ContinuousActionDimensions > 0,
                graph,
                stochasticInference);
        }

        // PPO: use spec-aware constructor when the checkpoint has a multi-stream or image spec.
        if (checkpoint.ObsSpec is { } spec &&
            (spec.Streams.Count > 1 || spec.Streams[0].Kind == ObservationStreamKind.Image))
        {
            return new PpoInferencePolicyMultiStream(
                spec,
                checkpoint.DiscreteActionCount,
                checkpoint.ContinuousActionDimensions,
                graph,
                stochasticInference);
        }

        return new PpoInferencePolicy(
            checkpoint.ObservationSize,
            checkpoint.DiscreteActionCount,
            checkpoint.ContinuousActionDimensions,
            graph,
            stochasticInference);
    }

    /// <summary>
    /// Rebuilds an <see cref="RLNetworkGraph"/> from checkpoint metadata.
    /// Prefers explicitly stored graph fields; falls back to the provided graph when none are present.
    /// </summary>
    private static RLNetworkGraph ReconstructGraph(RLCheckpoint checkpoint, RLNetworkGraph? fallbackGraph)
    {
        if (checkpoint.NetworkLayers.Count > 0)
        {
            var layers = new Array<Resource>();
            foreach (var layer in checkpoint.NetworkLayers)
            {
                switch (layer.Type)
                {
                    case "dense":
                        layers.Add(new RLDenseLayerDef
                        {
                            Size       = layer.Size,
                            Activation = ActivationFromString(layer.Activation),
                        });
                        break;
                    case "dropout":
                        layers.Add(new RLDropoutLayerDef { Rate = layer.Rate });
                        break;
                    case "layer_norm":
                        layers.Add(new RLLayerNormDef());
                        break;
                    case "flatten":
                        layers.Add(new RLFlattenLayerDef());
                        break;
                }
            }

            return new RLNetworkGraph
            {
                TrunkLayers = layers,
                Optimizer   = OptimizerFromString(checkpoint.NetworkOptimizer),
            };
        }

        if (fallbackGraph is not null) return fallbackGraph;

        return new RLNetworkGraph();
    }

    private static RLActivationKind ActivationFromString(string activation) => activation switch
    {
        "relu" => RLActivationKind.Relu,
        _      => RLActivationKind.Tanh,
    };

    private static RLOptimizerKind OptimizerFromString(string optimizer) => optimizer switch
    {
        "sgd"  => RLOptimizerKind.Sgd,
        "none" => RLOptimizerKind.None,
        _      => RLOptimizerKind.Adam,
    };
}
