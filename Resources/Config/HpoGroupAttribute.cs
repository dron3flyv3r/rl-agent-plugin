using System;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Marks a property on <see cref="RLTrainerConfig"/> as a tunable hyperparameter
/// and assigns it to a display group in the HPO parameter picker.
/// </summary>
[AttributeUsage(AttributeTargets.Property)]
public sealed class HpoGroupAttribute : Attribute
{
    public string Group { get; }
    public HpoGroupAttribute(string group) => Group = group;
}
