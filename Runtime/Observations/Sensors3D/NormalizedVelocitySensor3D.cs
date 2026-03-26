using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class NormalizedVelocitySensor3D : IObservationSensor, IObservationDebugLabels
{
    private readonly Func<Vector3> _velocityProvider;
    private readonly Vector3 _minVelocity;
    private readonly Vector3 _maxVelocity;
    private readonly bool _normalize;

    public NormalizedVelocitySensor3D(
        Func<Vector3> velocityProvider,
        Vector3 minVelocity,
        Vector3 maxVelocity,
        bool normalize = true)
    {
        _velocityProvider = velocityProvider ?? throw new ArgumentNullException(nameof(velocityProvider));
        _minVelocity = minVelocity;
        _maxVelocity = maxVelocity;
        _normalize = normalize;
    }

    public int Size => 3;

    public IReadOnlyList<string> DebugLabels { get; } = new[] { "velocity_x", "velocity_y", "velocity_z" };

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
