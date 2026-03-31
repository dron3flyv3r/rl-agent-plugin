namespace RlAgentPlugin.Runtime;

/// <summary>
/// GDExtension method names exposed by the native RlDenseLayer class.
/// </summary>
internal static class RlDenseLayerNativeFunctions
{
    public const string Initialize            = "initialize";
    public const string Forward               = "forward";
    public const string ForwardBatch          = "forward_batch";
    public const string Backward              = "backward";
    public const string CreateGradientBuffer  = "create_gradient_buffer";
    public const string AccumulateGradients   = "accumulate_gradients";
    public const string AccumulateGradientsWithBuffer = "accumulate_gradients_with_buffer";
    public const string ApplyGradients        = "apply_gradients";
    public const string ComputeInputGrad      = "compute_input_grad";
    public const string GradNormSquared       = "grad_norm_squared";
    public const string CopyWeightsFrom       = "copy_weights_from";
    public const string SoftUpdateFrom        = "soft_update_from";
    public const string GetWeights            = "get_weights";
    public const string GetShapes             = "get_shapes";
    public const string SetWeights            = "set_weights";
}
