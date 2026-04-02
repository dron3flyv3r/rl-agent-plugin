using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for a fully-connected (dense / linear) layer.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLDenseLayerDef : RLLayerDef
{
    private int _size = 64;
    private RLActivationKind _activation = RLActivationKind.Tanh;

    [Export]
    public int Size
    {
        get => _size;
        set
        {
            var clamped = Mathf.Max(1, value);
            if (_size == clamped) return;
            _size = clamped;
            EmitChanged();
        }
    }

    [Export]
    public RLActivationKind Activation
    {
        get => _activation;
        set
        {
            if (_activation == value) return;
            _activation = value;
            EmitChanged();
        }
    }

    internal override NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer,
                                               bool useNativeLayers = false)
    {
        if (useNativeLayers && NativeLayerSupport.IsAvailable)
            return new NativeDenseLayer(inputSize, Size, Activation, optimizer);
        return new DenseLayer(inputSize, Size, Activation, optimizer);
    }

    public override int GetOutputSize(int inputSize) => Size;
}
