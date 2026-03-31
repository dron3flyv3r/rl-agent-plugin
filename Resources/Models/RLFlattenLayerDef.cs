using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for a flatten operation.
/// Stage 1: pure passthrough (vector observations are already flat).
/// Stage 2: will reshape [C, H, W] image tensors to a 1-D feature vector.
/// </summary>
[GlobalClass]
public partial class RLFlattenLayerDef : RLLayerDef
{
    internal override NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer,
                                               bool useNativeLayers = false)
        => new FlattenLayer(inputSize);

    public override int GetOutputSize(int inputSize) => inputSize;
}
