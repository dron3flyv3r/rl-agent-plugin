using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for layer normalisation.
/// Normalises activations across the feature dimension, then applies a learned
/// per-feature affine transform (gamma/beta). Reduces internal covariate shift.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLLayerNormDef : RLLayerDef
{
    internal override NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer,
                                               bool useNativeLayers = false)
    {
        if (useNativeLayers && NativeLayerSupport.IsAvailable)
            return new NativeLayerNormLayer(inputSize);
        return new LayerNormLayer(inputSize);
    }

    public override int GetOutputSize(int inputSize) => inputSize;
}
