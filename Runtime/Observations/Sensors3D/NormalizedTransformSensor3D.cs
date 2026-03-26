using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class NormalizedTransformSensor3D : IObservationSensor, IObservationDebugLabels
{
    private readonly Node3D _node;
    private readonly Vector3 _minPosition;
    private readonly Vector3 _maxPosition;
    private readonly bool _includeRotation;
    private readonly bool _includeScale;
    private readonly Vector3 _minScale;
    private readonly Vector3 _maxScale;
    private readonly string[] _debugLabels;

    public NormalizedTransformSensor3D(
        Node3D node,
        Vector3 minPosition,
        Vector3 maxPosition,
        bool includeRotation = true,
        bool includeScale = false,
        Vector3? minScale = null,
        Vector3? maxScale = null)
    {
        _node = node ?? throw new ArgumentNullException(nameof(node));
        _minPosition = minPosition;
        _maxPosition = maxPosition;
        _includeRotation = includeRotation;
        _includeScale = includeScale;
        _minScale = minScale ?? Vector3.Zero;
        _maxScale = maxScale ?? Vector3.One;

        var labels = new List<string> { "position_x", "position_y", "position_z" };
        if (_includeRotation)
        {
            labels.Add("rotation_x");
            labels.Add("rotation_y");
            labels.Add("rotation_z");
        }

        if (_includeScale)
        {
            labels.Add("scale_x");
            labels.Add("scale_y");
            labels.Add("scale_z");
        }

        _debugLabels = labels.ToArray();
    }

    public int Size => 3 + (_includeRotation ? 3 : 0) + (_includeScale ? 3 : 0);

    public IReadOnlyList<string> DebugLabels => _debugLabels;

    public void Write(ObservationBuffer buffer)
    {
        buffer.AddNormalized(_node.GlobalPosition, _minPosition, _maxPosition);

        if (_includeRotation)
        {
            var euler = _node.GlobalRotation;
            buffer.AddNormalized(euler.X, -Mathf.Pi, Mathf.Pi);
            buffer.AddNormalized(euler.Y, -Mathf.Pi, Mathf.Pi);
            buffer.AddNormalized(euler.Z, -Mathf.Pi, Mathf.Pi);
        }

        if (_includeScale)
        {
            buffer.AddNormalized(_node.Scale, _minScale, _maxScale);
        }
    }
}
