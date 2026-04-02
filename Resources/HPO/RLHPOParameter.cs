using Godot;
using Godot.Collections;

namespace RlAgentPlugin;

/// <summary>
/// Defines a single hyperparameter axis in an <see cref="RLHPOStudy"/> search space.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLHPOParameter : Resource
{
    /// <summary>
    /// Name of the property on <see cref="Runtime.RLTrainerConfig"/> to override.
    /// Examples: "LearningRate", "Gamma", "ClipEpsilon", "SacTau", "DqnEpsilonDecaySteps".
    /// </summary>
    [Export] public string ParameterName { get; set; } = "";

    /// <summary>Distribution used to sample this parameter.</summary>
    [Export] public RLHPOParameterKind Kind { get; set; } = RLHPOParameterKind.FloatLog;

    /// <summary>Lower bound (inclusive) for numeric parameter kinds.</summary>
    [Export] public float Low { get; set; } = 0.00001f;

    /// <summary>Upper bound (inclusive) for numeric parameter kinds.</summary>
    [Export] public float High { get; set; } = 0.01f;

    /// <summary>
    /// Options for <see cref="RLHPOParameterKind.Categorical"/>.
    /// Each entry is parsed as a float so it can be applied to numeric config fields.
    /// Example: ["64", "128", "256"] for hidden layer width.
    /// </summary>
    [Export] public Array<string> Choices { get; set; } = new();
}
