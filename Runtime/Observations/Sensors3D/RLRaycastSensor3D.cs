using System.Collections.Generic;
using Godot;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Node-based 3D raycast sensor with a live editor preview.
///
/// Spawns <see cref="HorizontalRayCount"/> × <see cref="VerticalRayCount"/>
/// <c>RayCast3D</c> children that are visible in the Godot editor viewport.
///
/// Output per ray:
///   [normalized_dist]                          — when IncludeHitClass is false
///   [normalized_dist, hit_flag]                — when IncludeHitClass is true
///
/// hit_flag = 1 if the hit object's collision layer overlaps DetectionMask, else 0.
///
/// Use in CollectObservations: <c>obs.AddSensor("rays", sensor)</c>
/// </summary>
[Tool]
[GlobalClass]
public partial class RLRaycastSensor3D : Node3D, IObservationSensor, IObservationDebugLabels
{
    // ── Backing fields (setters call RebuildRays immediately) ─────────────────
    private int   _horizontalRayCount  = 7;
    private int   _verticalRayCount    = 1;
    private float _horizontalSpreadDeg = 120f;
    private float _verticalSpreadDeg   = 60f;
    private float _maxDistance         = 10f;
    private uint  _collisionMask       = uint.MaxValue;
    private bool  _collideWithAreas    = false;
    private bool  _collideWithBodies   = true;
    private bool  _excludeAncestor     = true;
    private bool  _showDebug           = true;

    [Export(PropertyHint.Range, "1,32,1")]
    public int HorizontalRayCount
    {
        get => _horizontalRayCount;
        set { _horizontalRayCount = Mathf.Max(1, value); RebuildRays(); }
    }

    [Export(PropertyHint.Range, "1,32,1")]
    public int VerticalRayCount
    {
        get => _verticalRayCount;
        set { _verticalRayCount = Mathf.Max(1, value); RebuildRays(); }
    }

    /// <summary>Total horizontal fan angle. 0 = all rays forward, 360 = full circle.</summary>
    [Export(PropertyHint.Range, "0,360,1")]
    public float HorizontalSpreadDeg
    {
        get => _horizontalSpreadDeg;
        set { _horizontalSpreadDeg = value; RebuildRays(); }
    }

    /// <summary>Total vertical fan angle. Only used when VerticalRayCount > 1.</summary>
    [Export(PropertyHint.Range, "0,180,1")]
    public float VerticalSpreadDeg
    {
        get => _verticalSpreadDeg;
        set { _verticalSpreadDeg = value; RebuildRays(); }
    }

    [Export(PropertyHint.Range, "0.1,200,0.1")]
    public float MaxDistance
    {
        get => _maxDistance;
        set { _maxDistance = Mathf.Max(0.1f, value); RebuildRays(); }
    }

    /// <summary>What the rays can see.</summary>
    [Export(PropertyHint.Layers3DPhysics)]
    public uint CollisionMask
    {
        get => _collisionMask;
        set { _collisionMask = value; RebuildRays(); }
    }

    [Export] public bool CollideWithAreas
    {
        get => _collideWithAreas;
        set { _collideWithAreas = value; RebuildRays(); }
    }

    [Export] public bool CollideWithBodies
    {
        get => _collideWithBodies;
        set { _collideWithBodies = value; RebuildRays(); }
    }

    /// <summary>Automatically exclude the nearest CollisionObject3D ancestor from hits.</summary>
    [Export] public bool ExcludeAncestor
    {
        get => _excludeAncestor;
        set { _excludeAncestor = value; CacheExcludedAncestor(); RebuildRays(); }
    }

    [Export] public bool ShowDebug
    {
        get => _showDebug;
        set { _showDebug = value; foreach (var r in _rays) r.Visible = value; }
    }

    /// <summary>
    /// Layers that count as a detection hit. Only used when
    /// <see cref="IncludeHitClass"/> is enabled.
    /// </summary>
    [Export(PropertyHint.Layers3DPhysics)]
    public uint DetectionMask { get; set; } = 0;

    /// <summary>
    /// When enabled, appends a hit_flag float (0 or 1) after each ray's distance.
    /// hit_flag = 1 if the hit object's layer overlaps <see cref="DetectionMask"/>.
    /// </summary>
    [Export] public bool IncludeHitClass { get; set; } = false;

    /// <summary>Debug color used when a ray hits an object matching the DetectionMask.</summary>
    [Export] public Color DetectionHitColor { get; set; } = new Color(0f, 0.8f, 0.8f);

    // ── IObservationSensor / IObservationDebugLabels ──────────────────────────
    public int Size => _horizontalRayCount * _verticalRayCount * (IncludeHitClass ? 2 : 1);

    public IReadOnlyList<string> DebugLabels => _debugLabels;

    // ── Internal ──────────────────────────────────────────────────────────────
    private readonly List<RayCast3D> _rays          = new();
    private readonly List<Vector3>   _rayDirections = new();
    private string[]                 _debugLabels   = System.Array.Empty<string>();
    private CollisionObject3D?       _excludedAncestor;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    public override void _Ready()
    {
        CacheExcludedAncestor();
        RebuildRays();
    }

    public override void _Process(double delta)
    {
        if (_showDebug)
            UpdateDebugColors();
    }

    // ── IObservationSensor ────────────────────────────────────────────────────

    public void Write(ObservationBuffer buffer)
    {
        if (!IsInsideTree())
        {
            WriteZeros(buffer);
            return;
        }

        // Parent nodes can query observations during their own _Ready(), before this sensor's
        // _Ready() has rebuilt the internal RayCast3D children. Recover lazily so the segment
        // size stays stable during inference / validation.
        if (_rays.Count == 0)
        {
            CacheExcludedAncestor();
            RebuildRays();
        }

        if (_rays.Count * (IncludeHitClass ? 2 : 1) != Size)
        {
            WriteZeros(buffer);
            return;
        }

        // Enable rays only while reading data
        foreach (var ray in _rays)
            ray.Enabled = true;

        foreach (var ray in _rays)
        {
            if (ray.IsColliding())
            {
                var dist = GlobalPosition.DistanceTo(ray.GetCollisionPoint());
                buffer.Add(1f - Mathf.Clamp(dist / _maxDistance, 0f, 1f));

                if (IncludeHitClass)
                {
                    var isDetected = ray.GetCollider() is CollisionObject3D col
                                     && (col.CollisionLayer & DetectionMask) != 0;
                    buffer.Add(isDetected ? 1f : 0f);
                }
            }
            else
            {
                buffer.Add(0f);
                if (IncludeHitClass) buffer.Add(0f);
            }
        }

        // Disable rays after reading data
        foreach (var ray in _rays)
            ray.Enabled = false;
    }

    private void WriteZeros(ObservationBuffer buffer)
    {
        for (var i = 0; i < Size; i++)
            buffer.Add(0f);
    }

    // ── Ray construction ──────────────────────────────────────────────────────

    private void RebuildRays()
    {
        if (!IsInsideTree()) return;

        var old = GetNodeOrNull("__rays");
        if (old != null) { RemoveChild(old); old.Free(); }
        _rays.Clear();
        _rayDirections.Clear();

        var container = new Node3D { Name = "__rays" };
        AddChild(container);

        var labelList = new List<string>(Size);
        var hSpread   = _horizontalSpreadDeg * Mathf.Pi / 180f;
        var vSpread   = _verticalSpreadDeg   * Mathf.Pi / 180f;

        for (var v = 0; v < _verticalRayCount; v++)
        {
            var pitch = _verticalRayCount == 1
                ? 0f
                : Mathf.Lerp(-vSpread / 2f, vSpread / 2f, v / (float)(_verticalRayCount - 1));

            for (var h = 0; h < _horizontalRayCount; h++)
            {
                var yaw = _horizontalRayCount == 1
                    ? 0f
                    : Mathf.Lerp(-hSpread / 2f, hSpread / 2f, h / (float)(_horizontalRayCount - 1));

                var dir = Basis.FromEuler(new Vector3(-pitch, yaw, 0f)) * new Vector3(0f, 0f, -1f);
                _rayDirections.Add(dir);

                var ray = new RayCast3D
                {
                    TargetPosition        = dir * _maxDistance,
                    CollisionMask         = _collisionMask,
                    CollideWithAreas      = _collideWithAreas,
                    CollideWithBodies     = _collideWithBodies,
                    Enabled               = false,
                    Visible               = _showDebug,
                    DebugShapeThickness   = 2,
                    DebugShapeCustomColor = new Color(0.2f, 1f, 0.2f),
                };

                if (_excludeAncestor && _excludedAncestor != null)
                    ray.AddException(_excludedAncestor);

                container.AddChild(ray);
                _rays.Add(ray);

                labelList.Add($"h{h}_v{v}_dist");
                if (IncludeHitClass) labelList.Add($"h{h}_v{v}_hit");
            }
        }

        _debugLabels = labelList.ToArray();

        if (Engine.IsEditorHint())
        {
            var sceneRoot = GetTree().EditedSceneRoot;
            if (sceneRoot != null)
            {
                container.Owner = sceneRoot;
                foreach (var ray in _rays)
                    ray.Owner = sceneRoot;
            }
        }
    }

    // ── Debug colors (updated every _Process) ─────────────────────────────────

    private void UpdateDebugColors()
    {
        if (!IsInsideTree()) return;

        var spaceState  = GetWorld3D().DirectSpaceState;
        var from        = GlobalPosition;
        var excludeList = BuildExcludeList();

        for (var i = 0; i < _rays.Count; i++)
        {
            var ray = _rays[i];
            var to  = from + GlobalBasis * _rayDirections[i] * _maxDistance;

            // Cast main ray to find nearest hit and distance.
            var mainQuery = PhysicsRayQueryParameters3D.Create(from, to, _collisionMask);
            mainQuery.CollideWithAreas  = _collideWithAreas;
            mainQuery.CollideWithBodies = _collideWithBodies;
            if (excludeList != null) mainQuery.Exclude = excludeList;
            var mainResult = spaceState.IntersectRay(mainQuery);

            if (mainResult.Count == 0)
            {
                ray.DebugShapeCustomColor = new Color(0.2f, 1f, 0.2f);
                continue;
            }

            var mainDist = from.DistanceTo(mainResult["position"].AsVector3());
            var frac     = 1f - Mathf.Clamp(mainDist / _maxDistance, 0f, 1f);

            // Check if that nearest hit is on the detection mask.
            var isDetected = false;
            if (DetectionMask != 0)
            {
                var detQuery = PhysicsRayQueryParameters3D.Create(from, to, DetectionMask);
                detQuery.CollideWithAreas  = _collideWithAreas;
                detQuery.CollideWithBodies = _collideWithBodies;
                if (excludeList != null) detQuery.Exclude = excludeList;
                var detResult = spaceState.IntersectRay(detQuery);

                if (detResult.Count > 0)
                {
                    var detDist = from.DistanceTo(detResult["position"].AsVector3());
                    // Same nearest object if the detection hit is within 2 cm of the main hit.
                    isDetected = Mathf.Abs(detDist - mainDist) < 0.02f;
                }
            }

            ray.DebugShapeCustomColor = isDetected
                ? DetectionHitColor
                : new Color(1f, 1f - frac, 0f);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private Array<Rid>? BuildExcludeList()
    {
        if (_excludedAncestor == null) return null;
        return new Array<Rid> { _excludedAncestor.GetRid() };
    }

    // ── Ancestor exclusion ────────────────────────────────────────────────────

    private void CacheExcludedAncestor()
    {
        _excludedAncestor = null;
        if (!_excludeAncestor) return;
        var node = GetParent();
        while (node != null)
        {
            if (node is CollisionObject3D col) { _excludedAncestor = col; return; }
            node = node.GetParent();
        }
    }
}
