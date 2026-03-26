using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for a fully-connected (dense / linear) layer.
/// </summary>
[GlobalClass]
public partial class RLDenseLayerDef : RLLayerDef
{
    [Export] public int Size { get; set; } = 64;
    [Export] public RLActivationKind Activation { get; set; } = RLActivationKind.Tanh;

    internal override NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer)
        => new DenseLayer(inputSize, Size, Activation, optimizer);

    public override int GetOutputSize(int inputSize) => Size;
}
