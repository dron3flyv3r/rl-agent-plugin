using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class RelativePositionSensor3D : IObservationSensor, IObservationDebugLabels
{
    private readonly Func<Vector3> _sourceProvider;
    private readonly Func<Vector3> _targetProvider;
    private readonly Vector3 _min;
    private readonly Vector3 _max;
    private readonly bool _normalize;

    public RelativePositionSensor3D(
        Func<Vector3> sourceProvider,
        Func<Vector3> targetProvider,
        Vector3 min,
        Vector3 max,
        bool normalize = true)
    {
        _sourceProvider = sourceProvider ?? throw new ArgumentNullException(nameof(sourceProvider));
        _targetProvider = targetProvider ?? throw new ArgumentNullException(nameof(targetProvider));
        _min = min;
        _max = max;
        _normalize = normalize;
    }

    public int Size => 3;

    public IReadOnlyList<string> DebugLabels { get; } = new[] { "delta_x", "delta_y", "delta_z" };

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
