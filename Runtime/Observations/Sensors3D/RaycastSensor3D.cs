using System;
using System.Collections.Generic;
using Godot;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

public sealed class RaycastSensor3D : IObservationSensor, IObservationDebugLabels
{
    private readonly Node3D _origin;
    private readonly Vector3[] _directions;
    private readonly float _rayLength;
    private readonly uint _collisionMask;
    private readonly bool _collideWithAreas;
    private readonly bool _collideWithBodies;
    private readonly bool _excludeParent;
    private readonly string[] _debugLabels;

    /// <summary>
    /// Creates a 3D raycast sensor.
    /// </summary>
    /// <param name="origin">The node from which rays are cast.</param>
    /// <param name="directions">
    ///   Local-space direction vectors for each ray.
    ///   They are transformed by the node's global basis before casting.
    /// </param>
    /// <param name="rayLength">Maximum ray length in world units.</param>
    /// <param name="collisionMask">Physics layer mask.</param>
    /// <param name="collideWithAreas">Whether rays collide with Area3D nodes.</param>
    /// <param name="collideWithBodies">Whether rays collide with PhysicsBody3D nodes.</param>
    /// <param name="excludeParent">Automatically exclude the parent CollisionObject3D from hits.</param>
    public RaycastSensor3D(
        Node3D origin,
        IReadOnlyList<Vector3> directions,
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
        _directions = new Vector3[directions.Count];
        _debugLabels = new string[directions.Count];

        for (var i = 0; i < directions.Count; i++)
        {
            _directions[i] = directions[i].Normalized();
            _debugLabels[i] = $"ray_{i}";
        }
    }

    public int Size => _directions.Length;

    public IReadOnlyList<string> DebugLabels => _debugLabels;

    public void Write(ObservationBuffer buffer)
    {
        if (!_origin.IsInsideTree())
        {
            for (var i = 0; i < _directions.Length; i++)
            {
                buffer.Add(0f);
            }

            return;
        }

        var spaceState = _origin.GetWorld3D().DirectSpaceState;
        var originPos = _origin.GlobalPosition;
        var globalBasis = _origin.GlobalBasis;

        foreach (var localDir in _directions)
        {
            var worldDir = globalBasis * localDir;
            var target = originPos + worldDir * _rayLength;
            var query = PhysicsRayQueryParameters3D.Create(originPos, target);
            query.CollisionMask = _collisionMask;
            query.CollideWithAreas = _collideWithAreas;
            query.CollideWithBodies = _collideWithBodies;

            if (_excludeParent && _origin.GetParent() is CollisionObject3D parentCollider)
            {
                query.Exclude = new Array<Rid> { parentCollider.GetRid() };
            }

            var hit = spaceState.IntersectRay(query);
            if (hit.Count == 0)
            {
                buffer.Add(0f);
                continue;
            }

            var hitPosition = hit["position"].AsVector3();
            var distance = originPos.DistanceTo(hitPosition);
            var normalizedDistance = _rayLength > 0f ? 1f - Mathf.Clamp(distance / _rayLength, 0f, 1f) : 0f;
            buffer.Add(normalizedDistance);
        }
    }
}
