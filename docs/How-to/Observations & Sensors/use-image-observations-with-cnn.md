# How to Use Image Observations with CNN

<!-- markdownlint-disable MD029 MD032 -->

This guide covers `RLCameraSensor2D`, image streams, and CNN training flow in RL Agent Plugin.

---

## What Happens When You Add An Image Stream

When your agent calls `obs.AddImage(...)`:

- camera pixels are captured from `RLCameraSensor2D`
- image bytes are normalized to `[0, 1]` in `ObservationBuffer`
- the network creates an image encoder path (CNN) for that stream

For PPO, the plugin can switch to GPU-backed CNN training when Vulkan compute is available.

---

## Step 1: Add RLCameraSensor2D Under Agent Node

Scene placement pattern:

```text
Player (CharacterBody2D)
└── Agent (Node2D + RLAgent2D)
    └── Camera (RLCameraSensor2D)
```

The sensor should be a child of the agent node, not the player root.

---

## Step 2: Configure Sensor Output Settings

Tune these exports first:

- `RenderSize`
- `CropSize`
- `CropOffset`
- `Grayscale`
- `Zoom`

Start simple:

- low resolution (`64` or `128`)
- grayscale enabled unless color is required
- crop centered before trying offsets

---

## Step 3: Add Image Stream In CollectObservations

Use a stable stream name:

```csharp
[Export] private RLCameraSensor2D? _camera;

public override void CollectObservations(ObservationBuffer obs)
{
    if (_camera is null) return;

    // Named stream. Keep name stable for checkpoint compatibility.
    obs.AddImage("camera", _camera);

    // Optional vector context can be mixed with image input.
    obs.AddNormalized(_speed, 0f, _maxSpeed);
}
```

You can also use `obs.AddImage(_camera)` to default to sensor name.

---

## Step 4: Verify Camera Coverage And Live Feed

Use two debug paths:

1. Sensor overlay in scene view to verify framing/crop.
2. `RLAcademy` camera debug overlay during runtime/training.

If training fails, confirm the camera actually sees meaningful motion/state changes.

---

## Step 5: Enable Correct Distributed Setup (If Used)

For distributed workers with camera sensors:

- enable renderer-required workers
- keep worker batch size at `1` for camera pipelines
- keep agent logic in `_PhysicsProcess`

This avoids stale frame readback and keeps image observations aligned with physics ticks.

---

## Performance Tips

1. Keep image dimensions as small as task allows.
2. Prefer grayscale when possible.
3. Profile steps/sec with and without image stream.
4. Increase worker count rather than per-worker batch for camera-heavy setups.

---

## Common Mistakes

1. Putting `RLCameraSensor2D` outside the agent hierarchy.
2. Using huge image sizes too early.
3. Changing stream names mid-project.
4. Running renderer-dependent camera sensors in pure headless setups.

---

## Minimal Checklist

- camera sensor placed under agent
- stream added with `obs.AddImage(...)`
- camera framing and crop validated visually
- image size/grayscale tuned for throughput
- distributed settings aligned for renderer-based sensors
