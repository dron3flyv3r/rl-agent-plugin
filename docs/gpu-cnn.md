# GPU CNN Training

This page describes the current GPU-backed CNN path for image observations:

- what it does
- when it activates automatically
- what still stays on CPU
- how to confirm which path is active

The short version is:

- adding an image stream creates a CNN encoder automatically
- PPO can train that CNN encoder on the GPU through Vulkan compute shaders
- rollout inference still stays on the CPU
- there is no separate "enable GPU CNN" checkbox

---

## What This Feature Does

When a policy uses one or more image observation streams, the plugin builds a per-stream CNN encoder before the shared policy/value trunk.

Current backend choices:

- `CnnEncoder` = CPU path via the native `RlCnnEncoder` GDExtension
- `GpuCnnEncoder` = GPU path via Godot `RenderingDevice` compute shaders

The GPU path only accelerates the image encoder part of PPO training. The shared trunk, policy head, and value head still run on the CPU.

That means the actual split today is:

- rollout collection and action sampling: CPU
- image encoder training update: GPU when available
- shared MLP trunk and PPO heads: CPU
- checkpoint save/load: shared serialization format between CPU and GPU encoders

---

## How It Activates

There is no explicit GPU toggle in the Inspector.

Activation is automatic when all of the following are true:

1. Your observation spec contains at least one image stream.
2. The trainer is PPO.
3. Vulkan compute is available on the current machine.

In practice this usually means:

1. Add `RLCameraSensor2D` to the agent.
2. Call `obs.AddImage(...)` in `CollectObservations()`.
3. Train with PPO.
4. Run on a machine where `RenderingServer.CreateLocalRenderingDevice()` succeeds.

Once that happens, `PpoTrainer` detects that the policy has image streams and creates a dedicated GPU training thread. That thread builds a second copy of the policy network with `preferGpuImageEncoders: true`, which causes `PolicyValueNetwork` to instantiate `GpuCnnEncoder` for image streams instead of `CnnEncoder`.

Important distinction:

- the normal rollout network is still created with `preferGpuImageEncoders: false`
- the GPU encoder is used by the dedicated PPO training network only

So "CNN is present" and "GPU CNN is active" are separate decisions:

- image stream present -> a CNN encoder exists
- PPO + Vulkan available -> the training copy of that CNN encoder uses the GPU backend

---

## What Creates A CNN

An image stream is created by `ObservationBuffer.AddImage(...)`.

For camera sensors:

- `obs.AddImage("camera", _camera)` captures pixels from `RLCameraSensor2D`
- the stream is tagged as `ObservationStreamKind.Image`
- the sensor's `EncoderConfig` is attached to that stream automatically

When `PolicyValueNetwork` is built:

- image streams get a CNN encoder
- vector streams stay as raw vectors or optional per-stream MLPs
- embeddings from all streams are concatenated before the shared trunk

If no explicit `RLCnnEncoderDef` is supplied for an image stream, the network uses a default CNN definition.

---

## PPO GPU Execution Model

The GPU path is designed around a dedicated training thread because Godot local rendering devices are thread-bound.

The flow is:

1. The main PPO network collects rollouts and performs action sampling on CPU.
2. If image streams are present and Vulkan is available, `PpoTrainer` starts a dedicated GPU training thread.
3. That thread creates a local `RenderingDevice` through `RenderingServer.CreateLocalRenderingDevice()`.
4. A training-only copy of the policy network is created with GPU image encoders.
5. Each PPO update loads the latest checkpoint into that GPU network, runs training, then returns an updated checkpoint back to the main trainer.

This keeps the Vulkan device and all GPU buffers owned by one long-lived thread, which matches Godot's threading requirements and avoids recompiling shaders every update.

---

## What Runs On GPU

`GpuCnnEncoder` currently handles:

- convolution forward pass
- ReLU activation caching
- projection layer forward pass
- batched backward pass for conv and projection layers
- Adam updates for CNN/projection parameters
- gradient norm accumulation for global clipping

The compute shaders are embedded in `GpuShaderSources.cs` and compiled at runtime into Vulkan compute pipelines.

Current shader set:

- `ConvForward`
- `ConvBackwardFilter`
- `ConvBackwardInput`
- `ReluBackward`
- `LinearForward`
- `LinearBackward`
- `AdamUpdate`
- `AdamUpdateNative`
- `NormSquaredAccumulate`

All image encoder tensors stay on the GPU during a batch update except where the CPU trunk/head path needs the projected embedding batch.

---

## PPO Batch Update Flow

Inside `PolicyValueNetwork.ApplyGradients(...)`, the mixed CPU/GPU path works like this:

1. For every image stream whose encoder reports `SupportsBatchedTraining = true`, the full minibatch is packed into one contiguous input array.
2. `ForwardBatch(...)` runs once for that stream on the GPU and produces one embedding batch.
3. The shared trunk, policy head, value head, PPO loss, and most gradient bookkeeping still run sample-by-sample on the CPU.
4. The gradient with respect to each CNN embedding is accumulated into a batched gradient buffer.
5. `AccumulateGradientsBatch(...)` runs once per batched CNN stream on the GPU.
6. Global gradient clipping includes the CNN gradient norm.
7. `ApplyGradients(...)` applies Adam updates to the GPU CNN weights.

This is why the feature helps most when the CNN is the bottleneck and the MLP heads are relatively small.

---

## Fallback Behavior

If Vulkan is unavailable, PPO falls back to the CPU CNN path automatically.

If GPU encoder initialization fails after PPO requested it, the code currently treats that as a hard failure for that GPU training network creation path instead of silently continuing.

For the encoder selection itself:

- `preferGpuImageEncoders = false` -> always use `CnnEncoder`
- `preferGpuImageEncoders = true` and Vulkan available -> try `GpuCnnEncoder`
- `preferGpuImageEncoders = true` and Vulkan unavailable -> warn and use `CnnEncoder`

---

## How To Verify It Is Active

Look for these log messages in the Godot output:

- `[PPO] Group '...': using GPU image encoder for training updates; rollout inference remains on CPU.`
- `[PPO] GPU training network ready on dedicated thread for group '...'.`
- `[PolicyValueNetwork] Using GpuCnnEncoder for image stream WxHxC.`

If Vulkan is not available, you should instead see:

- `[PPO] Group '...': GPU image encoder unavailable; training updates will run on CPU.`
- `[PolicyValueNetwork] Vulkan unavailable — falling back to CnnEncoder for image stream WxHxC.`

Practical signs:

- PPO updates should get noticeably faster when camera observations are the bottleneck
- rollout inference cost will not change much, because action sampling still uses the CPU network

---

## Current Scope And Limits

The current implementation is intentionally narrow:

- PPO image-stream training supports the GPU CNN path
- rollout inference still uses the CPU encoder
- distributed headless workers do not use camera rendering
- SAC does not yet use the per-stream CNN/GPU encoder path

Two implications matter in practice:

- if you use camera observations with headless distributed workers, the camera sensor warns that capture is disabled in headless mode
- if you train with SAC, image observations are not routed through this GPU CNN training path today

---

## Typical Setup

Example workflow for automatic activation:

1. Add `RLCameraSensor2D` under the agent.
2. Assign an `RLStreamEncoderConfig` if you want to customize the CNN.
3. In `CollectObservations()`, call:

```csharp
obs.AddImage("camera", _camera);
```

4. Train with PPO.
5. Watch the console for the GPU activation messages above.

If you do not assign an encoder config, the image stream still gets a default CNN encoder.

---

## Related Docs

- [sensors.md](sensors.md) for camera/image stream setup
- [architecture.md](architecture.md) for the training/runtime structure
- [configuration.md](configuration.md) for PPO and network configuration
