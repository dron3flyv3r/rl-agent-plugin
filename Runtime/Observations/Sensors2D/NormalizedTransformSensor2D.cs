using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class NormalizedTransformSensor2D : IObservationSensor, IObservationDebugLabels
{
    private readonly Node2D _node;
    private readonly Vector2 _minPosition;
    private readonly Vector2 _maxPosition;
    private readonly bool _includeRotation;
    private readonly bool _includeScale;
    private readonly Vector2 _minScale;
    private readonly Vector2 _maxScale;
    private readonly string[] _debugLabels;

    public NormalizedTransformSensor2D(
        Node2D node,
        Vector2 minPosition,
        Vector2 maxPosition,
        bool includeRotation = true,
        bool includeScale = false,
        Vector2? minScale = null,
        Vector2? maxScale = null)
    {
        _node = node ?? throw new ArgumentNullException(nameof(node));
        _minPosition = minPosition;
        _maxPosition = maxPosition;
        _includeRotation = includeRotation;
        _includeScale = includeScale;
        _minScale = minScale ?? Vector2.Zero;
        _maxScale = maxScale ?? Vector2.One;

        var labels = new List<string> { "position_x", "position_y" };
        if (_includeRotation)
        {
            labels.Add("rotation");
        }

        if (_includeScale)
        {
            labels.Add("scale_x");
            labels.Add("scale_y");
        }

        _debugLabels = labels.ToArray();
    }

    public int Size => 2 + (_includeRotation ? 1 : 0) + (_includeScale ? 2 : 0);

    public IReadOnlyList<string> DebugLabels => _debugLabels;

    public void Write(ObservationBuffer buffer)
    {
        buffer.AddNormalized(_node.GlobalPosition, _minPosition, _maxPosition);

        if (_includeRotation)
        {
            buffer.AddNormalized(_node.GlobalRotation, -Mathf.Pi, Mathf.Pi);
        }

        if (_includeScale)
        {
            buffer.AddNormalized(_node.Scale, _minScale, _maxScale);
        }
    }
}
