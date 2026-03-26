using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class RelativePositionSensor2D : IObservationSensor, IObservationDebugLabels
{
    private readonly Func<Vector2> _sourceProvider;
    private readonly Func<Vector2> _targetProvider;
    private readonly Vector2 _min;
    private readonly Vector2 _max;
    private readonly bool _normalize;

    public RelativePositionSensor2D(
        Func<Vector2> sourceProvider,
        Func<Vector2> targetProvider,
        Vector2 min,
        Vector2 max,
        bool normalize = true)
    {
        _sourceProvider = sourceProvider ?? throw new ArgumentNullException(nameof(sourceProvider));
        _targetProvider = targetProvider ?? throw new ArgumentNullException(nameof(targetProvider));
        _min = min;
        _max = max;
        _normalize = normalize;
    }

    public int Size => 2;

    public IReadOnlyList<string> DebugLabels { get; } = new[] { "delta_x", "delta_y" };

    public void Write(ObservationBuffer buffer)
    {
        var delta = _targetProvider() - _sourceProvider();
        if (_normalize)
        {
            buffer.AddNormalized(delta, _min, _max);
            return;
        }

        buffer.Add(delta);
    }
}
