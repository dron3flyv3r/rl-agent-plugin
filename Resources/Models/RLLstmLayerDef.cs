using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer definition for an LSTM (Long Short-Term Memory) recurrent layer.
///
/// Adds temporal context to the agent's policy — useful for partially-observable
/// environments where a single observation frame is not sufficient.
///
/// Hidden size determines both the output dimension and the size of the internal
/// cell state. A single LSTM layer is typically sufficient; stack by placing another
/// LSTM layer directly after the first.
///
/// <see cref="GradClipNorm"/> clips the L2 norm of the recurrent weight gradients
/// per BPTT update to prevent exploding gradients. Set to 0 to disable.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLLstmLayerDef : RLLayerDef
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
            return new NativeLstmLayer(inputSize, HiddenSize, optimizer, GradClipNorm);

        if (useNativeLayers)
        {
            Godot.GD.PushError("[RLLstmLayerDef] LSTM requires the native C++ library. " +
                               "Build the GDExtension or disable UseNativeLayers.");
        }

        return new DenseLayer(inputSize, HiddenSize, RLActivationKind.Tanh, optimizer);
    }

    public override int GetOutputSize(int inputSize) => HiddenSize;
}
