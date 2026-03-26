using System;
using System.Collections.Generic;
using Godot;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

public sealed class RaycastSensor2D : IObservationSensor, IObservationDebugLabels
{
    private readonly Node2D _origin;
    private readonly float[] _angleOffsets;
    private readonly float _rayLength;
    private readonly uint _collisionMask;
    private readonly bool _collideWithAreas;
    private readonly bool _collideWithBodies;
    private readonly bool _excludeParent;
    private readonly string[] _debugLabels;

    public RaycastSensor2D(
        Node2D origin,
        IReadOnlyList<float> angleOffsetsRadians,
        float rayLength,
        uint collisionMask = uint.MaxValue,
        bool collideWithAreas = false,
        bool collideWithBodies = true,
        bool excludeParent = true)
    {
        _origin = origin ?? throw new ArgumentNullException(nameof(origin));
        _rayLength = Math.Max(0f, rayLength);
        _collisionMask = collisionMask;
        _collideWithAreas = collideWithAreas;
        _collideWithBodies = collideWithBodies;
        _excludeParent = excludeParent;
        _angleOffsets = new float[angleOffsetsRadians.Count];
        _debugLabels = new string[angleOffsetsRadians.Count];

        for (var i = 0; i < angleOffsetsRadians.Count; i++)
        {
            _angleOffsets[i] = angleOffsetsRadians[i];
            _debugLabels[i] = $"ray_{i}";
        }
    }

    public int Size => _angleOffsets.Length;

    public IReadOnlyList<string> DebugLabels => _debugLabels;

    public void Write(ObservationBuffer buffer)
    {
        if (!_origin.IsInsideTree())
        {
            for (var i = 0; i < _angleOffsets.Length; i++)
            {
                buffer.Add(0f);
            }

            return;
        }

        var spaceState = _origin.GetWorld2D().DirectSpaceState;
        var origin = _origin.GlobalPosition;
        var baseRotation = _origin.GlobalRotation;

        foreach (var angleOffset in _angleOffsets)
        {
            var direction = Vector2.Right.Rotated(baseRotation + angleOffset);
            var target = origin + direction * _rayLength;
            var query = PhysicsRayQueryParameters2D.Create(origin, target);
            query.CollisionMask = _collisionMask;
            query.CollideWithAreas = _collideWithAreas;
            query.CollideWithBodies = _collideWithBodies;

            if (_excludeParent && _origin.GetParent() is CollisionObject2D parentCollider)
            {
                query.Exclude = new Array<Rid> { parentCollider.GetRid() };
            }

            var hit = spaceState.IntersectRay(query);
            if (hit.Count == 0)
            {
                buffer.Add(0f);
                continue;
            }

            var hitPosition = hit["position"].AsVector2();
            var distance = origin.DistanceTo(hitPosition);
            var normalizedDistance = _rayLength > 0f ? 1f - Mathf.Clamp(distance / _rayLength, 0f, 1f) : 0f;
            buffer.Add(normalizedDistance);
        }
    }
}
