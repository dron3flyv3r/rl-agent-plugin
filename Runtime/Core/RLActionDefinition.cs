using System;

namespace RlAgentPlugin.Runtime;

public enum RLActionVariableType
{
    Discrete = 0,
    Continuous = 1,
}

public readonly struct RLActionDefinition
{
    public RLActionDefinition(
        string name,
        RLActionVariableType variableType = RLActionVariableType.Discrete,
        string[]? labels = null,
        int dimensions = 0,
        float minValue = -1f,
        float maxValue = 1f)
    {
        Name = string.IsNullOrWhiteSpace(name) ? "Action" : name;
        VariableType = variableType;
        Labels = labels ?? Array.Empty<string>();
        Dimensions = dimensions;
        MinValue = minValue;
        MaxValue = maxValue;
    }

    public string Name { get; }
    public RLActionVariableType VariableType { get; }
    public string[] Labels { get; }
    public int Dimensions { get; }
    public float MinValue { get; }
    public float MaxValue { get; }
}
