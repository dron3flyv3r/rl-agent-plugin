# How to Use Raycast Sensors for Navigation

<!-- markdownlint-disable MD029 MD032 -->

This guide shows how to use raycast sensors for obstacle awareness and navigation decisions.

---

## Why Raycast Observations Work Well

Ray-based features provide compact spatial awareness:

- free-space direction cues
- obstacle proximity cues
- optional hit-class channel for target types

They are often cheaper than image observations and easier to debug.

---

## Step 1: Choose Sensor Type

Use one of these based on your scene:

- `RaycastSensor2D` for 2D tasks
- `RaycastSensor3D` for code-driven 3D rays
- `RLRaycastSensor3D` for node-based 3D setup with inspector controls

`RLRaycastSensor3D` is convenient when designers tune ray count/spread directly in editor.

---

## Step 2: Configure Ray Layout

Pick a layout that matches movement space:

- forward-focused fan for racing/chase tasks
- wide spread for maze or cluttered arenas
- extra vertical rays for 3D climbing/jumping

Start small (for example 5-9 rays), then increase only if policy needs finer detail.

---

## Step 3: Add Sensor To Observations

Example with named sensor segment:

```csharp
private RLRaycastSensor3D? _sensor;

public override void CollectObservations(ObservationBuffer obs)
{
    _sensor ??= GetNodeOrNull<RLRaycastSensor3D>("RLRaycastSensor3D");
    if (_sensor is null) return;

    obs.AddSensor("rays", _sensor);
}
```

This appends ray outputs in a fixed order each step.

---

## Step 4: Interpret Ray Outputs Correctly

Typical output behavior in built-in ray sensors:

- hit distance converted to normalized proximity signal
- no hit often emits `0`
- optional hit-class flags emit additional binary channels

If using hit classes, keep collision layers/masks consistent across scenes.

---

## Step 5: Combine Rays With Core Kinematics

Ray features are strongest when combined with:

- velocity
- relative target direction
- grounded state / orientation cues

This gives the policy both spatial context and dynamic state.

---

## Debugging Tips

1. Verify ray origins and directions in scene debug views.
2. Confirm collision masks include intended obstacles.
3. Check for self-collision exclusion when rays start near colliders.
4. Log or inspect observation segments if behavior looks random.

---

## Common Mistakes

1. Overcrowding with too many rays too early.
2. Wrong collision masks (rays miss important geometry).
3. Depending only on rays without motion/goal context.
4. Changing ray count during an ongoing experiment.

---

## Minimal Checklist

- correct 2D/3D ray sensor selected
- ray layout tuned for task geometry
- sensor added via `obs.AddSensor("rays", sensor)`
- collision masks validated
- ray features combined with core state observations
