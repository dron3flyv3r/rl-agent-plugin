namespace RlAgentPlugin.Runtime;

/// <summary>
/// GDExtension method names exposed by the native RlGruLayer class.
/// </summary>
internal static class RlGruLayerNativeFunctions
{
    public const string Initialize                    = "initialize";
    public const string Forward                       = "forward";
    public const string ForwardSequence               = "forward_sequence";
    public const string AccumulateSequenceGradients   = "accumulate_sequence_gradients";
    public const string ApplyGradients                = "apply_gradients";
    public const string CreateGradientBuffer          = "create_gradient_buffer";
    public const string GradNormSquared               = "grad_norm_squared";
    public const string CopyWeightsFrom               = "copy_weights_from";
    public const string SoftUpdateFrom                = "soft_update_from";
    public const string GetWeights                    = "get_weights";
    public const string GetShapes                     = "get_shapes";
    public const string SetWeights                    = "set_weights";
    public const string GetInputSize                  = "get_input_size";
    public const string GetHiddenSize                 = "get_hidden_size";
}
