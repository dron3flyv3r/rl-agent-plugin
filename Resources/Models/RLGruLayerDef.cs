using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for a GRU (Gated Recurrent Unit) layer.
///
/// A lighter-weight alternative to LSTM — fewer parameters, often trains faster,
/// with competitive performance on shorter temporal dependencies.
///
/// <see cref="GradClipNorm"/> clips the L2 norm of recurrent weight gradients
/// per BPTT update. Set to 0 to disable.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLGruLayerDef : RLLayerDef
{
    private int _hiddenSize = 64;
    private float _gradClipNorm = 1.0f;

    [Export]
    public int HiddenSize
    {
        get => _hiddenSize;
        set
        {
            var clamped = Mathf.Max(1, value);
            if (_hiddenSize == clamped) return;
            _hiddenSize = clamped;
            EmitChanged();
        }
    }

    [Export]
    public float GradClipNorm
    {
        get => _gradClipNorm;
        set
        {
            var clamped = Mathf.Clamp(value, 0f, 100f);
            if (Mathf.IsEqualApprox(_gradClipNorm, clamped)) return;
            _gradClipNorm = clamped;
            EmitChanged();
        }
    }

    internal override NetworkLayer CreateLayer(int inputSize, RLOptimizerKind optimizer,
                                               bool useNativeLayers = false)
    {
        if (useNativeLayers && NativeLayerSupport.IsAvailable)
            return new NativeGruLayer(inputSize, HiddenSize, optimizer, GradClipNorm);

        if (useNativeLayers)
        {
            Godot.GD.PushError("[RLGruLayerDef] GRU requires the native C++ library. " +
                               "Build the GDExtension or disable UseNativeLayers.");
        }

        return new DenseLayer(inputSize, HiddenSize, RLActivationKind.Tanh, optimizer);
    }

    public override int GetOutputSize(int inputSize) => HiddenSize;
}
