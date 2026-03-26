using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class NormalizedVelocitySensor2D : IObservationSensor, IObservationDebugLabels
{
    private readonly Func<Vector2> _velocityProvider;
    private readonly Vector2 _minVelocity;
    private readonly Vector2 _maxVelocity;
    private readonly bool _normalize;

    public NormalizedVelocitySensor2D(
        Func<Vector2> velocityProvider,
        Vector2 minVelocity,
        Vector2 maxVelocity,
        bool normalize = true)
    {
        _velocityProvider = velocityProvider ?? throw new ArgumentNullException(nameof(velocityProvider));
        _minVelocity = minVelocity;
        _maxVelocity = maxVelocity;
        _normalize = normalize;
    }

    public int Size => 2;

    public IReadOnlyList<string> DebugLabels { get; } = new[] { "velocity_x", "velocity_y" };

    public void Write(ObservationBuffer buffer)
    {
        var velocity = _velocityProvider();
        if (_normalize)
        {
            buffer.AddNormalized(velocity, _minVelocity, _maxVelocity);
            return;
        }

        buffer.Add(velocity);
    }
}
