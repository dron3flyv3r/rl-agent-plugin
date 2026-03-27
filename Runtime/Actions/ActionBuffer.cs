using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

public sealed class ActionBuffer
{
    private readonly Dictionary<string, Tuple<int, string>> _discreteActions = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float[]> _continuousActions = new(StringComparer.Ordinal);

    public bool HasDiscreteActions => _discreteActions.Count > 0;
    public bool HasContinuousActions => _continuousActions.Count > 0;

    public int GetDiscrete(string name)
    {
        if (!_discreteActions.TryGetValue(name, out var value))
        {
            throw new KeyNotFoundException($"Discrete action '{name}' is not defined in the current action buffer.");
        }

        return value.Item1;
    }

    public TEnum GetDiscreteAsEnum<TEnum>(string name) where TEnum : Enum
    {
        if (!_discreteActions.TryGetValue(name, out var value))
        {
            throw new KeyNotFoundException($"Discrete action '{name}' is not defined in the current action buffer.");
        }

        var enumType = typeof(TEnum);
        if (!Enum.IsDefined(enumType, value.Item1))
        {
            throw new ArgumentException($"Value '{value.Item1}' is not a valid member of enum '{enumType.Name}'.");
        }

        return (TEnum)Enum.ToObject(enumType, value.Item1);
    }

    public TEnum GetDiscreteAsEnum<TEnum>() where TEnum : Enum
    {
        var name = typeof(TEnum).Name;
        return GetDiscreteAsEnum<TEnum>(name);
    }

    public string GetDiscreteLabel(string name)
    {
        if (!_discreteActions.TryGetValue(name, out var value))
        {
            throw new KeyNotFoundException($"Discrete action '{name}' is not defined in the current action buffer.");
        }

        return value.Item2;
    }

    public float[] GetContinuous(string name)
    {
        if (!_continuousActions.TryGetValue(name, out var value))
        {
            throw new KeyNotFoundException($"Continuous action '{name}' is not defined in the current action buffer.");
        }

        var copy = new float[value.Length];
        Array.Copy(value, copy, value.Length);
        return copy;
    }

    public float GetContinuous(string name, int index)
    {
        if (!_continuousActions.TryGetValue(name, out var value))
        {
            throw new KeyNotFoundException($"Continuous action '{name}' is not defined in the current action buffer.");
        }

        if (index < 0 || index >= value.Length)
        {
            throw new IndexOutOfRangeException($"Index {index} is out of range for continuous action '{name}' with length {value.Length}.");
        }

        return value[index];
    }

    public bool TryGetDiscrete(string name, out Tuple<int, string>? value)
    {
        if (_discreteActions.TryGetValue(name, out var stored))
        {
            value = new Tuple<int, string>(stored.Item1, stored.Item2);
            return true;
        }
        value = null;
        return false;
    }

    public bool TryGetContinuous(string name, out float[] values)
    {
        if (_continuousActions.TryGetValue(name, out var stored))
        {
            values = new float[stored.Length];
            Array.Copy(stored, values, stored.Length);
            return true;
        }

        values = Array.Empty<float>();
        return false;
    }

    internal void SetDiscrete(string name, Tuple<int, string> value)
    {
        _discreteActions[name] = value;
    }

    internal void SetContinuous(string name, float[] values)
    {
        var copy = new float[values.Length];
        Array.Copy(values, copy, values.Length);
        _continuousActions[name] = copy;
    }
}
