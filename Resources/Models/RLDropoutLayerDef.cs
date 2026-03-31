using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for inverted dropout.
/// During training, each neuron is zeroed with probability <see cref="Rate"/> and
/// surviving activations are scaled by 1/(1−Rate) to preserve expected magnitude.
/// During inference the layer is a pure passthrough.
/// </summary>
[GlobalClass]
public partial class RLDropoutLayerDef : RLLayerDef
{
    [Export] public float Rate { get; set; } = 0.1f;

    internal override NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer,
                                               bool useNativeLayers = false)
        => new DropoutLayer(inputSize, Rate);

    public override int GetOutputSize(int inputSize) => inputSize;
}
