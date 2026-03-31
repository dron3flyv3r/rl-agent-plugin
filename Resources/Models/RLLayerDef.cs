using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Abstract Godot Resource base for all neural-network layer definitions.
///
/// To create a custom layer type:
///   1. Subclass <c>RLLayerDef</c> and mark it <c>[GlobalClass]</c> so it appears in the
///      Inspector asset picker.
///   2. Add <c>[Export]</c> properties for any user-configurable hyperparameters (e.g. Size,
///      Rate).
///   3. Implement <c>CreateLayer</c> to return your runtime layer instance.
///   4. Implement <c>GetOutputSize</c> — return <c>inputSize</c> for passthrough layers,
///      or a fixed value for layers that change the feature dimension.
/// </summary>
[GlobalClass]
[Tool]
public abstract partial class RLLayerDef : Resource
{
    /// <summary>
    /// Constructs the runtime layer.
    /// <paramref name="optimizer"/> is threaded from <see cref="RLNetworkGraph"/>; add new
    /// <see cref="RLOptimizerKind"/> variants without touching this signature.
    /// Non-parameterised layers (Dropout, Flatten) may ignore it.
    /// When <paramref name="useNativeLayers"/> is true and the native library is available,
    /// implementations should return a native-backed layer for better SIMD performance.
    /// </summary>
    internal abstract NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer,
                                               bool useNativeLayers = false);

    /// <summary>Returns the output feature size this layer produces for a given input size.</summary>
    public abstract int GetOutputSize(int inputSize);
}
