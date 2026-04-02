# Feature Catalog

<!-- markdownlint-disable MD024 -->

This page lists the RL Agent Plugin features in one place, with a short explanation and a concrete way to use each one.

If you are new to the plugin, read [get-started.md](get-started.md) first, then use this page as your "what can it do?" reference.

---

## At a glance

| Area | Included features |
| --- | --- |
| Core training | PPO and SAC fully documented here; A2C, DQN, and MCTS are referenced elsewhere in the docs |
| Runtime modes | Training mode + inference mode with `.rlmodel` |
| Editor workflow | Start/Stop/Run toolbar, RL Setup dock, RLDash charts |
| Observations | Vector observations, built-in sensors, image observations |
| Advanced training | Curriculum, self-play + PFSP, distributed workers, recurrent PPO/A2C |
| Performance | Async gradient updates, PPO GPU CNN path for image streams |
| Debugging | Spy/training overlay, camera debug overlay, live metrics |

---

## 1) PPO trainer (discrete + continuous)

### What it is

- On-policy trainer with clipped updates.
- Supports both discrete and continuous actions.

### Why use it

- Great default for first projects and most discrete-action tasks.

### Show me

```text
RLTrainingConfig
- Algorithm: RLPPOConfig
```

Key knobs you will tune first:

- `RolloutLength`
- `LearningRate`
- `EntropyCoefficient`
- `ClipEpsilon`

See also: [algorithms.md](algorithms.md), [configuration.md](configuration.md#rlppoconfig)

---

## 2) SAC trainer (continuous control)

### What it is

- Off-policy Soft Actor-Critic with replay buffer, double critics, and auto entropy tuning.
- Continuous actions only.

### Why use it

- Higher sample efficiency on difficult continuous-control problems.

### Show me

```text
RLTrainingConfig
- Algorithm: RLSACConfig
```

Key knobs you will tune first:

- `ReplayBufferCapacity`
- `WarmupSteps`
- `BatchSize`
- `UpdatesPerStep`

See also: [algorithms.md](algorithms.md), [configuration.md](configuration.md#rlsacconfig)

---

## 3) Shared policy groups across agents

### What it is

- Multiple agents can share one policy by using the same `AgentId` in `RLPolicyGroupConfig`.
- Different `AgentId` values train separate policies.

### Why use it

- Multi-agent scenes where all agents should learn a common behavior.

### Show me

```text
Agent A -> PolicyGroupConfig.AgentId = "bot"
Agent B -> PolicyGroupConfig.AgentId = "bot"   (shared weights)
Agent C -> PolicyGroupConfig.AgentId = "boss"  (separate weights)
```

See also: [configuration.md](configuration.md#rlpolicygroupconfig)

---

## 4) Recurrent policy layers (LSTM / GRU)

### What it is

- `RLNetworkGraph` can include `RLLstmLayerDef` or `RLGruLayerDef` in the shared trunk.
- End-to-end recurrent rollout, checkpointing, inference, and BPTT are implemented for PPO and A2C.

### Why use it

- Good fit for partially observable environments where one frame is not enough.
- Works with shared-policy multi-agent setups: each agent keeps its own hidden state even when weights are shared.

### Show me

```text
RLNetworkGraph
- Dense(64, Tanh)
- LSTM(HiddenSize=64)
- Dense(64, Tanh)
```

Current limits:

- Requires the native C++ GDExtension.
- Only one recurrent trunk layer is supported per network.
- Supported end-to-end for PPO and A2C only.
- DQN and SAC do not support recurrent trunks.

See also: [configuration.md](configuration.md#rlnetworkgraph), [algorithms.md](algorithms.md#recurrent-policies-lstm--gru), [architecture.md](architecture.md#neural-network-architecture)

---

## 5) In-editor training workflow

### What it is

- Start/stop training from the editor.
- Dedicated RL Setup dock and top toolbar controls.

### Why use it

- Fast iteration without leaving Godot.

### Show me

After enabling the plugin and building once:

- Top toolbar: **Start Training**, **Stop Training**, **Run Inference**
- Right dock: **RL Setup**

See also: [get-started.md](get-started.md)

---

## 6) Live dashboard metrics (RLDash)

### What it is

- Live charting UI that polls metrics files during training.
- Common charts: reward, episode length, policy/value loss, entropy.

### Why use it

- Understand whether learning is healthy while training runs.

### Show me

- Open **RLDash** tab in the editor.
- Metrics are read from `RL-Agent-Training/<RunId>/metrics.jsonl` (or per-group metrics files).

See also: [tuning.md](tuning.md), [architecture.md](architecture.md#metrics--dashboard)

---

## 7) Model export/import (`.rlmodel`) for deployment

### What it is

- Export trained checkpoints to a compact `.rlmodel` format.
- Load the model back for inference.

### Why use it

- Ship trained behavior in your game build.

### Show me

1. In RLDash, export a run/checkpoint to `.rlmodel`.
2. Set `PolicyGroupConfig.InferenceModelPath`.
3. Run with **Run Inference**.

See also: [README.md](../README.md), [architecture.md](architecture.md#inference)

---

## 8) Training and inference modes

### What it is

- Training mode collects transitions and updates weights.
- Inference mode loads `.rlmodel` and runs deterministic forward passes.

### Why use it

- Same scene can be used both for learning and for production play.

### Show me

- During training, agents use PPO/SAC training policies.
- During inference, the academy creates `IInferencePolicy` instances for each policy group.

See also: [architecture.md](architecture.md#inference)

---

## 9) Rich observation system

### What it is

- Supports scalar/vector observations, normalized inputs, reusable sensors, and image streams.
- You define observation order explicitly in `CollectObservations`.

### Why use it

- Full control over what the policy sees.

### Show me

```csharp
public override void CollectObservations(ObservationBuffer obs)
{
    obs.AddNormalized(GlobalPosition.X, -500f, 500f);
    obs.AddSensor("ray", _raycastSensor);
    obs.AddImage("camera", _cameraSensor);
}
```

See also: [configuration.md](configuration.md#observationbuffer-methods), [sensors.md](sensors.md)

---

## 10) Built-in sensors

### What it is

Built-in sensor nodes/interfaces include:

- Raycast sensors (2D and 3D)
- Relative position sensors
- Normalized transform/velocity sensors
- `RLCameraSensor2D` image sensor

### Why use it

- Faster setup and less custom sensor code for common tasks.

### Show me

```text
Player
└── Agent (RLAgent2D/RLAgent3D)
    ├── RaycastSensor2D or RLRaycastSensor3D
    └── RLCameraSensor2D (2D only)
```

Then add them in `CollectObservations` via `obs.AddSensor(...)` / `obs.AddImage(...)`.

See also: [sensors.md](sensors.md), [configuration.md](configuration.md#built-in-sensors)

---

## 11) Camera observations with optional GPU CNN training path (PPO)

### What it is

- Image observations automatically get CNN encoders.
- For PPO, image-encoder training can move to Vulkan compute automatically.

### Why use it

- Better performance when camera streams are the training bottleneck.

### Show me

Activation conditions:

1. At least one image observation stream exists.
2. Trainer is PPO.
3. Vulkan compute is available.

```csharp
obs.AddImage("camera", _camera);
```

See also: [gpu-cnn.md](gpu-cnn.md), [sensors.md](sensors.md#rlcamerasensor2d)

---

## 12) Curriculum learning

### What it is

- Built-in curriculum config with two modes:
- `StepBased`: progress by total steps
- `SuccessRate`: adapt from recent success

### Why use it

- Start easy, then increase task difficulty as the agent improves.

### Show me

```csharp
public override void OnTrainingProgress(float progress)
{
    _difficulty = progress; // 0.0 easy -> 1.0 hard
}
```

Attach `RLCurriculumConfig` to `RLAcademy.Curriculum`.

See also: [configuration.md](configuration.md#rlcurriculumconfig)

---

## 13) Self-play with historical opponents and PFSP

### What it is

- Configure competitive pairings between policy groups.
- Supports historical opponent pools, Elo tracking, and PFSP sampling.

### Why use it

- Robust training for adversarial and competitive environments.

### Show me

```text
RLSelfPlayConfig
- Pairings[] -> RLPolicyPairingConfig
  - HistoricalOpponentRate
  - FrozenCheckpointInterval
  - PfspEnabled / PfspAlpha
```

See also: [configuration.md](configuration.md#rlselfplayconfig)

---

## 14) Distributed rollout workers

### What it is

- Launch multiple worker processes that collect experience and stream it to a master trainer over TCP.
- Master owns weights and performs updates.

### Why use it

- Major throughput gains for expensive environments.

### Show me

```text
RLDistributedConfig
- WorkerCount: 4
- AutoLaunchWorkers: true
- WorkerSimulationSpeed: 4.0
```

For camera-sensor workers, use `WorkersRequireRenderer = true` and keep `BatchSize = 1`.

See also: [architecture.md](architecture.md#distributed-training), [configuration.md](configuration.md#rldistributedconfig)

---

## 14) Schedules for dynamic hyperparameters

### What it is

- Time-varying schedules for training parameters such as learning rate and entropy coefficient.
- Built-ins: constant, exponential decay, cosine schedule.

### Why use it

- Better stability and convergence across long runs.

### Show me

```text
RLTrainingConfig.Schedules
- LearningRate: RLCosineSchedule(...)
- EntropyCoefficient: RLExponentialSchedule(...)
```

See also: [configuration.md](configuration.md#rlscheduleconfig)

---

## 15) Flexible action and reward API

### What it is

- Mixed action spaces via `ActionSpaceBuilder`.
- Named reward components for debugging and reward-balance inspection.

### Why use it

- Cleaner agent code and easier reward diagnostics.

### Show me

```csharp
public override void DefineActions(ActionSpaceBuilder builder)
{
    builder.AddDiscrete("Move", "Left", "Right", "Idle");
    builder.AddContinuous("Aim", dimensions: 2, min: -1f, max: 1f);
}

public override void OnStep()
{
    AddReward(-0.001f, "step_penalty");
    AddReward(_distanceProgress, "distance");
}
```

See also: [configuration.md](configuration.md#agent-api-reference)

---

## 16) Training overlays and debugging tools

### What it is

- Training overlay / spy overlay for runtime introspection.
- Camera debug overlay that previews `RLCameraSensor2D` streams during training.

### Why use it

- Quickly validate that observations, actions, and rewards are sane.

### Show me

- Enable camera debug on `RLAcademy` (Debug section).
- Enable training overlay in distributed settings with `ShowTrainingOverlay`.

See also: [sensors.md](sensors.md#debugging-during-training), [tuning.md](tuning.md)

---

## 17) 2D and 3D agent support

### What it is

- Supports both `RLAgent2D` and `RLAgent3D` workflows.
- Shared config model across dimensions.

### Why use it

- Build lightweight 2D prototypes and scale to 3D tasks using the same plugin architecture.

### Show me

| Topic | 2D | 3D |
| --- | --- | --- |
| Agent type | `RLAgent2D` | `RLAgent3D` |
| Typical body | `CharacterBody2D` | `CharacterBody3D` |
| Camera sensor | `RLCameraSensor2D` available | no equivalent 3D camera sensor today |

See also: [get-started.md](get-started.md#2-2d-vs-3d-start-with-the-right-base), [configuration.md](configuration.md#2d-vs-3d-what-actually-changes)

---

## Feature map by goal

| Goal | Start here |
| --- | --- |
| First successful training run | [get-started.md](get-started.md) |
| Choose PPO vs SAC | [algorithms.md](algorithms.md) |
| Tune unstable training | [tuning.md](tuning.md) |
| Add sensors/camera | [sensors.md](sensors.md) |
| Enable distributed workers | [configuration.md](configuration.md#rldistributedconfig) |
| Understand internal architecture | [architecture.md](architecture.md) |
| Use GPU CNN for image observations | [gpu-cnn.md](gpu-cnn.md) |
