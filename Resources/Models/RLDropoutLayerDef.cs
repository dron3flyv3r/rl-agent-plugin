using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for inverted dropout.
/// During training, each neuron is zeroed with probability <see cref="Rate"/> and
/// surviving activations are scaled by 1/(1−Rate) to preserve expected magnitude.
/// During inference the layer is a pure passthrough.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLDropoutLayerDef : RLLayerDef
{
    private float _rate = 0.1f;

    [Export]
    public float Rate
    {
        get => _rate;
        set
        {
            var clamped = Mathf.Clamp(value, 0f, 0.99f);
            if (Mathf.IsEqualApprox(_rate, clamped)) return;
            _rate = clamped;
            EmitChanged();
        }
    }

    internal override NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer,
                                               bool useNativeLayers = false)
        => new DropoutLayer(inputSize, Rate);

    public override int GetOutputSize(int inputSize) => inputSize;
}
