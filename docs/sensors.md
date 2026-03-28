# Sensors

Sensors are helper nodes that collect observations on behalf of an agent. They always live as **children of the `RLAgent2D` or `RLAgent3D` node**, not on the player node.

```text
Player (CharacterBody2D)
└── Agent (Node2D + RLAgent2D script)
    ├── RaycastSensor2D       ← sensor child
    └── Camera (RLCameraSensor2D)  ← sensor child
```

The agent calls sensors manually inside `CollectObservations`. There is no auto-discovery — you decide the order and the stream name.

---

## Built-in vector sensors

| Node | What it provides |
|------|-----------------|
| `RaycastSensor2D` | Cast rays outward, return hit distance and optional hit-tag encoding |
| `RaycastSensor3D` | Same for 3D |
| `RelativePositionSensor2D` | Normalised XY offset to a target node |
| `NormalizedTransformSensor2D` | Normalised position + rotation of a node |
| `NormalizedVelocitySensor2D` | Normalised velocity of a `CharacterBody2D` |

Call them with `obs.AddSensor("name", sensor)` or `obs.Add(sensor.Read())`.

---

## RLCameraSensor2D

`RLCameraSensor2D` captures a 2D scene region as pixels and feeds them into the observation as an image stream. The network builds a CNN encoder for the stream automatically.

When training with PPO, that CNN can also switch to the GPU-backed training path automatically when Vulkan compute is available. See [gpu-cnn.md](gpu-cnn.md) for the activation rules and execution model.

### Node placement

Add `RLCameraSensor2D` as a child of the `Agent` node. Position and rotate it in the scene like any `Node2D` — it behaves like a camera placed at that point.

```text
Agent (Node2D + RLAgent2D)
└── Camera (RLCameraSensor2D)
    # position this node to frame the area you want the agent to see
```

### How it works internally

```
RLCameraSensor2D (Node2D — positioned in world)
└── SubViewport  (renders at RenderSize)
    └── Camera2D (mirrors the sensor's world transform every frame)
```

Every physics frame the internal `Camera2D` is synced to the sensor's `GlobalPosition`, `GlobalRotation`, and `Zoom`. When `Capture()` is called, the rendered texture is read back, optionally cropped, and optionally converted to grayscale.

### Exports

| Property | Default | Description |
|----------|---------|-------------|
| `RenderSize` | `128 × 128` | Resolution of the internal SubViewport. Higher = more detail, more GPU cost. |
| `CropSize` | `64 × 64` | Size of the pixel region passed to the network. Must be ≤ `RenderSize`. |
| `CropOffset` | `0, 0` | Top-left pixel offset of the crop window within the rendered image. `(RenderSize - CropSize) / 2` centres it. |
| `Grayscale` | `true` | Convert to single-channel luminance (BT.601). Halves input size vs RGB. |
| `Zoom` | `1.0` | Camera zoom. Decrease to see a wider area (`0.18` fits an 800×600 arena into a 128×128 viewport). |

### Choosing RenderSize vs CropSize

Render at a larger size than you feed to the network when you want:
- Flexible cropping without re-rendering (shift `CropOffset` at runtime).
- Anti-aliasing effect from downscaling a high-res render.
- To experiment with different crop sizes without changing GPU load.

For most training tasks a matching `RenderSize = CropSize` is fine.

### Choosing Zoom

`Zoom = viewport_pixels / world_units_to_cover`.

Examples for a 128-pixel viewport:
| World area to cover | Zoom |
|---------------------|------|
| 128 world units | 1.0 |
| 400 world units | 0.32 |
| 720 world units | 0.18 |

### Using the sensor in an agent

Export a reference to the sensor node, then call `obs.AddImage` from `CollectObservations`:

```csharp
public partial class MyAgent : RLAgent2D
{
    [Export] private RLCameraSensor2D? _camera;

    public override void CollectObservations(ObservationBuffer obs)
    {
        // The stream name "camera" is used as the key in the ObservationSpec.
        // Use a consistent name — it is saved in checkpoints.
        obs.AddImage("camera", _camera);

        // You can mix camera and vector observations freely.
        obs.AddNormalized(someValue, 0f, 100f);
    }
}
```

`AddImage(string name, RLCameraSensor2D sensor)` calls `sensor.Capture()` internally and registers the result as an image stream in the `ObservationSpec`. The network will build a CNN encoder for it automatically.

There is no separate "use GPU CNN" flag on the sensor. The GPU backend is selected later by the PPO trainer if the policy contains image streams and Vulkan is available.

You can add **multiple** camera sensors with different names:

```csharp
obs.AddImage("front", _frontCamera);
obs.AddImage("top",   _topCamera);
```

Each gets its own CNN encoder. The embeddings are concatenated before the shared policy/value trunk.

### Editor overlay

When the `RLCameraSensor2D` node is visible in the scene (no toggle needed):

- **White rectangle** — the full render area in world space (`RenderSize / Zoom`).
- **Yellow rectangle** — the crop region actually fed to the network.
- **Corner ticks + crosshair** — orientation helpers.

Use the overlay to verify that the render area covers the right part of the scene and that the crop is positioned where you expect.

### Debugging during training

Enable **Debug → Enable Camera Debug** on the `RLAcademy` node. This spawns `RLCameraDebugOverlay`, which shows a live panel in the top-right corner of the screen for every `RLCameraSensor2D` found on any agent — including during training runs.

Each panel shows:
- The sensor name
- Crop size and channel mode
- The raw SubViewport texture (what the CNN actually receives, before normalisation)

This works in normal play mode and in training mode launched from the editor.

### Distributed training

`RLCameraSensor2D` requires a GPU renderer. Enable `WorkersRequireRenderer` on `RLDistributedConfig` to launch workers with a real renderer instead of `--headless`.

When `WorkersRequireRenderer` is true, workers automatically run in **1:1 physics/render lock-step** (`Engine.MaxPhysicsStepsPerFrame = 1`, `Engine.MaxFps = 0`). This means every physics tick has exactly one freshly rendered `SubViewport` frame — zero stale observations regardless of GPU speed.

**BatchSize must be 1** for camera-sensor workers. All environments within a worker share one render thread; `BatchSize > 1` causes most observations to come from stale frames. Scale throughput by increasing `WorkerCount` instead.

All agent logic must be in `_PhysicsProcess` (not `_Process`). `_PhysicsProcess(delta)` always receives a fixed `1 / PhysicsTicksPerSecond` delta — physics is deterministic even at low real-time FPS.

See [configuration.md — Camera sensors in distributed training](configuration.md#camera-sensors-in-distributed-training) for the full setup.

### Performance notes

- GPU readback (`GetImage()`) is lazy — it only happens once per rendered frame, on the first `Capture()` call. Subsequent calls within the same frame return a cached result.
- Keep `RenderSize` small (64–128) during training to reduce GPU readback cost.
- Use `Grayscale = true` unless colour is essential. It cuts CNN input size by 3×.
- For profiling, compare training steps/sec with and without the camera sensor active to measure the actual overhead for your scene.
