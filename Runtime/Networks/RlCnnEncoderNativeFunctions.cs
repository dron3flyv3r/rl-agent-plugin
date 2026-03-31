namespace RlAgentPlugin.Runtime;

/// <summary>
/// GDExtension method names exposed by the native RlCnnEncoder class.
/// </summary>
internal static class RlCnnEncoderNativeFunctions
{
    public const string Initialize = "initialize";
    public const string Forward = "forward";
    public const string CreateGradientBuffer = "create_gradient_buffer";
    public const string AccumulateGradients = "accumulate_gradients";
    public const string AccumulateGradientsWithBuffer = "accumulate_gradients_with_buffer";
    public const string ApplyGradients = "apply_gradients";
    public const string GradNormSquared = "grad_norm_squared";
    public const string GetWeights = "get_weights";
    public const string GetShapes = "get_shapes";
    public const string SetWeights = "set_weights";
}
