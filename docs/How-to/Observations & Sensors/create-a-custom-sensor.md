# How to Create a Custom Sensor

<!-- markdownlint-disable MD029 MD032 -->

This guide shows how to build reusable custom sensors using `IObservationSensor`.

---

## Why Build A Custom Sensor

Create a custom sensor when:

- the same observation logic is reused across multiple agents
- you want cleaner `CollectObservations` methods
- you want optional debug labels per sensor output channel

The plugin sensor contract is small and explicit.

---

## Step 1: Implement IObservationSensor

Create a class that implements:

- `int Size { get; }`
- `void Write(ObservationBuffer buffer)`

Optional: also implement `IObservationDebugLabels` to name each emitted value.

Example:

```csharp
using System;
using System.Collections.Generic;
using Godot;
using RlAgentPlugin.Runtime;

public sealed class DistanceToGoalSensor : IObservationSensor, IObservationDebugLabels
{
    private readonly Func<Vector3> _source;
    private readonly Func<Vector3> _goal;
    private readonly float _maxDistance;

    public DistanceToGoalSensor(Func<Vector3> source, Func<Vector3> goal, float maxDistance)
    {
        _source = source;
        _goal = goal;
        _maxDistance = Mathf.Max(1f, maxDistance);
    }

    public int Size => 1;

    public IReadOnlyList<string> DebugLabels { get; } = new[] { "distance_to_goal" };

    public void Write(ObservationBuffer buffer)
    {
        var distance = _source().DistanceTo(_goal());
        buffer.AddNormalized(distance, 0f, _maxDistance);
    }
}
```

---

## Step 2: Instantiate Sensor In Agent Setup

Initialize the sensor once (for example in `_Ready()` or lazy-init before use):

```csharp
private DistanceToGoalSensor? _distanceSensor;

public override void _Ready()
{
    _distanceSensor = new DistanceToGoalSensor(
        () => _player.GlobalPosition,
        () => _goal.GlobalPosition,
        50f);
}
```

---

## Step 3: Add It In CollectObservations

Use a named segment for clarity:

```csharp
public override void CollectObservations(ObservationBuffer obs)
{
    if (_distanceSensor is null) return;
    obs.AddSensor("goal_distance", _distanceSensor);
}
```

This keeps observation code modular while preserving deterministic output size.

---

## Step 4: Validate Size And Data Quality

`ObservationBuffer` validates sensor output count against `Size`.

If your sensor writes a different number of values than declared, the plugin logs an error. Keep `Write()` deterministic.

---

## Advanced Pattern: Scene Node Sensor

If you want editor-exposed settings and scene visualization, create a `Node2D`/`Node3D` class that implements `IObservationSensor`, similar to built-in raycast node sensors.

That pattern is useful when designers should tune sensor parameters in Inspector.

---

## Common Mistakes

1. Writing variable-length output while `Size` is fixed.
2. Reading invalid node state before the sensor is ready.
3. Emitting unbounded values without normalization.
4. Recreating sensor objects every frame.

---

## Minimal Checklist

- `IObservationSensor` implemented
- `Size` matches exactly what `Write()` emits
- values normalized or intentionally bounded
- sensor added via `obs.AddSensor("name", sensor)`
- sensor instantiated once, reused every step
