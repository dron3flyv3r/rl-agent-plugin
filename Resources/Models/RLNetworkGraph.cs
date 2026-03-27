using System.Collections.Generic;
using Godot;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLNetworkGraph : Resource
{
    private const string LayerResourceTypes =
        $"{nameof(RLDenseLayerDef)},{nameof(RLDropoutLayerDef)},{nameof(RLFlattenLayerDef)},{nameof(RLLayerNormDef)}";

    [Export(PropertyHint.ResourceType, LayerResourceTypes)]
    public Array<Resource> TrunkLayers { get; set; } = new();

    [Export] public RLOptimizerKind Optimizer { get; set; } = RLOptimizerKind.Adam;

    /// <summary>Output size of the trunk; chains GetOutputSize through each layer def.</summary>
    public int OutputSize(int inputSize)
    {
        var prev = inputSize;
        foreach (var def in EnumerateTrunkLayerDefs())
        {
            prev = def.GetOutputSize(prev);
        }

        return prev;
    }

    /// <summary>
    /// Constructs runtime layer instances wired in order from <paramref name="inputSize"/>.
    /// Pass <paramref name="overrideOptimizer"/> to force a specific optimizer (e.g.
    /// <see cref="RLOptimizerKind.None"/> for frozen SAC target networks).
    /// </summary>
    internal NetworkLayer[] BuildTrunkLayers(int inputSize, RLOptimizerKind? overrideOptimizer = null)
    {
        if (TrunkLayers.Count == 0) return System.Array.Empty<NetworkLayer>();

        var optimizer = overrideOptimizer ?? Optimizer;
        var layers = new List<NetworkLayer>(TrunkLayers.Count);
        var prev = inputSize;
        foreach (var def in EnumerateTrunkLayerDefs())
        {
            layers.Add(def.CreateLayer(prev, optimizer));
            prev = def.GetOutputSize(prev);
        }

        return layers.ToArray();
    }

    /// <summary>Returns the Dense layer sizes (used for checkpoint metadata display).</summary>
    public int[] GetLayerSizes()
    {
        var sizes = new int[TrunkLayers.Count];
        for (var i = 0; i < TrunkLayers.Count; i++)
            sizes[i] = TrunkLayers[i] is RLDenseLayerDef d ? d.Size : 0;
        return sizes;
    }

    /// <summary>Returns the Dense layer activations (used for checkpoint metadata display).</summary>
    public int[] GetLayerActivations()
    {
        var activations = new int[TrunkLayers.Count];
        for (var i = 0; i < TrunkLayers.Count; i++)
            activations[i] = TrunkLayers[i] is RLDenseLayerDef d ? (int)d.Activation : -1;
        return activations;
    }

    /// <summary>Returns a sensible default network: two 64-unit Tanh layers with Adam optimizer.</summary>
    public static RLNetworkGraph CreateDefault()
    {
        return new RLNetworkGraph
        {
            TrunkLayers = new Array<Resource>
            {
                new RLDenseLayerDef { Size = 64, Activation = RLActivationKind.Tanh },
                new RLDenseLayerDef { Size = 64, Activation = RLActivationKind.Tanh },
            },
            Optimizer = RLOptimizerKind.Adam,
        };
    }

    private IEnumerable<RLLayerDef> EnumerateTrunkLayerDefs()
    {
        foreach (var resource in TrunkLayers)
        {
            if (resource is RLLayerDef layerDef)
            {
                yield return layerDef;
            }
        }
    }
}
