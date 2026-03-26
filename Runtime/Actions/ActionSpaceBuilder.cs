using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

public sealed class ActionSpaceBuilder
{
    private readonly List<RLActionDefinition> _actions = new();

    public bool HasActions => _actions.Count > 0;

    public void AddDiscrete(string name, params string[] labels)
    {
        if (labels is null || labels.Length == 0)
        {
            throw new ArgumentException("Discrete actions require at least one label.", nameof(labels));
        }

        _actions.Add(new RLActionDefinition(
            name,
            RLActionVariableType.Discrete,
            labels: (string[])labels.Clone(),
            dimensions: 1));
    }

    public void AddDiscrete(string name, int labelCount)
    {
        if (labelCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(labelCount), "Discrete actions require at least one label.");
        }

        var labels = new string[labelCount];
        for (var i = 0; i < labelCount; i++)
        {
            labels[i] = i.ToString();
        }

        _actions.Add(new RLActionDefinition(
            name,
            RLActionVariableType.Discrete,
            labels: labels,
            dimensions: 1));
    }

    public void AddDiscrete<TEnum>() where TEnum : Enum
    {
        var type = typeof(TEnum);
        var name = type.Name;
        var labels = Enum.GetNames(type);
        AddDiscrete(name, labels);
    }

    public void AddContinuous(string name, int dimensions, float min = -1f, float max = 1f)
    {
        if (dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Continuous actions require at least one dimension.");
        }

        _actions.Add(new RLActionDefinition(
            name,
            RLActionVariableType.Continuous,
            dimensions: dimensions,
            minValue: min,
            maxValue: max));
    }

    public RLActionDefinition[] Build()
    {
        return _actions.ToArray();
    }

    public bool SupportsOnlyDiscreteActions()
    {
        foreach (var action in _actions)
        {
            if (action.VariableType != RLActionVariableType.Discrete)
            {
                return false;
            }
        }

        return _actions.Count > 0;
    }

    public int GetDiscreteActionCount()
    {
        var total = 1;
        var foundDiscreteAction = false;

        foreach (var action in _actions)
        {
            if (action.VariableType != RLActionVariableType.Discrete)
            {
                continue;
            }

            foundDiscreteAction = true;
            total *= Math.Max(1, action.Labels.Length);
        }

        return foundDiscreteAction ? total : 0;
    }

    public int GetContinuousActionDimensions()
    {
        var total = 0;
        foreach (var action in _actions)
        {
            if (action.VariableType == RLActionVariableType.Continuous)
            {
                total += Math.Max(0, action.Dimensions);
            }
        }

        return total;
    }

    public string[] BuildDiscreteActionLabels()
    {
        var discreteActions = new List<RLActionDefinition>();
        foreach (var action in _actions)
        {
            if (action.VariableType == RLActionVariableType.Discrete)
            {
                discreteActions.Add(action);
            }
        }

        if (discreteActions.Count == 0)
        {
            return Array.Empty<string>();
        }

        var total = 1;
        foreach (var action in discreteActions)
        {
            total *= Math.Max(1, action.Labels.Length);
        }

        var labels = new string[total];
        for (var flatIndex = 0; flatIndex < total; flatIndex++)
        {
            var remaining = flatIndex;
            var parts = new string[discreteActions.Count];
            for (var actionIndex = 0; actionIndex < discreteActions.Count; actionIndex++)
            {
                var action = discreteActions[actionIndex];
                var actionCount = Math.Max(1, action.Labels.Length);
                var valueIndex = remaining % actionCount;
                remaining /= actionCount;
                parts[actionIndex] = $"{action.Name}={action.Labels[valueIndex]}";
            }

            labels[flatIndex] = string.Join(", ", parts);
        }

        return labels;
    }

    public ActionBuffer CreateDiscreteActionBuffer(int flatIndex)
    {
        var buffer = new ActionBuffer();
        var remaining = flatIndex;
        foreach (var action in _actions)
        {
            if (action.VariableType != RLActionVariableType.Discrete)
            {
                continue;
            }

            var actionCount = Math.Max(1, action.Labels.Length);
            var valueIndex = remaining % actionCount;
            remaining /= actionCount;
            Tuple<int, string> value = new(valueIndex, action.Labels.Length > 0 ? action.Labels[Math.Min(valueIndex, action.Labels.Length - 1)] : valueIndex.ToString());
            buffer.SetDiscrete(action.Name, value);
        }

        return buffer;
    }

    public ActionBuffer CreateContinuousActionBuffer(float[] actions)
    {
        var buffer = new ActionBuffer();
        var offset = 0;

        foreach (var action in _actions)
        {
            if (action.VariableType != RLActionVariableType.Continuous)
            {
                continue;
            }

            var dimensions = Math.Max(0, action.Dimensions);
            var slice = new float[dimensions];
            Array.Copy(actions, offset, slice, 0, Math.Min(dimensions, Math.Max(0, actions.Length - offset)));
            buffer.SetContinuous(action.Name, slice);
            offset += dimensions;
        }

        return buffer;
    }
}
