# Configuration Reference

All configuration in the RL Agent Plugin is done through **Godot Resources** attached to the `RLAcademy` node in the Inspector. Every parameter described here can be set without writing code.

If you are new to the plugin, start with the **Get Started Guide** (`get-started.md`) first, then use this page as a parameter-by-parameter reference.

---

## Overview: Config Hierarchy

```
RLAcademy
├── TrainingConfig (RLTrainingConfig)
│   ├── Algorithm          → RLPPOConfig  or  RLSACConfig
│   └── Schedules          → RLScheduleConfig  (optional)
│
├── RunConfig (RLRunConfig)
│
├── DistributedConfig (RLDistributedConfig)   (optional)
│
├── Curriculum (RLCurriculumConfig)            (optional)
│
└── SelfPlay (RLSelfPlayConfig)               (optional)
    └── Pairings[]
        └── RLPolicyPairingConfig
            ├── GroupA  → RLPolicyGroupConfig
            └── GroupB  → RLPolicyGroupConfig
```

Each agent also has its own `PolicyGroupConfig` property that links it to a shared policy.

---

## RLPPOConfig

Configuration for the **Proximal Policy Optimization** algorithm.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RolloutLength` | int | 256 | Number of transitions to collect before running a gradient update. Larger = more stable, slower feedback. |
| `EpochsPerUpdate` | int | 4 | How many passes over the rollout data per update. More epochs extract more value from data but risk instability. |
| `MiniBatchSize` | int | 64 | Samples per gradient step within each epoch. Should evenly divide `RolloutLength`. |
| `LearningRate` | float | 0.0005 | Step size for the Adam optimizer. Controls how fast the network weights change. |
| `Gamma` | float | 0.99 | Discount factor for future rewards. `0.99` = care a lot about the future; `0.9` = more myopic. |
| `GaeLambda` | float | 0.95 | GAE lambda. Higher values reduce variance (smoother advantage estimates) at the cost of more bias. |
| `ClipEpsilon` | float | 0.2 | PPO clipping range. Prevents the new policy from deviating more than this from the old policy. |
| `MaxGradientNorm` | float | 0.5 | Gradient clipping threshold. Prevents exploding gradients. |
| `ValueLossCoefficient` | float | 0.5 | Weight of the value loss relative to policy loss. |
| `UseValueClipping` | bool | false | Whether to apply PPO-style clipping to the value loss as well. Can help stability. |
| `ValueClipEpsilon` | float | 0.2 | Clipping range for value loss (only used if `UseValueClipping = true`). |
| `EntropyCoefficient` | float | 0.01 | Weight of entropy bonus. Higher values encourage more exploration (more random actions). |

### Example: PPO for a simple discrete task

```
RLPPOConfig:
  RolloutLength: 512
  EpochsPerUpdate: 4
  MiniBatchSize: 128
  LearningRate: 0.0003
  Gamma: 0.99
  GaeLambda: 0.95
  ClipEpsilon: 0.2
  EntropyCoefficient: 0.01
```

### Example: PPO for a continuous locomotion task

```
RLPPOConfig:
  RolloutLength: 2048
  EpochsPerUpdate: 10
  MiniBatchSize: 256
  LearningRate: 0.0003
  Gamma: 0.995
  GaeLambda: 0.95
  ClipEpsilon: 0.2
  EntropyCoefficient: 0.005
  ValueLossCoefficient: 1.0
```

---

## RLSACConfig

Configuration for the **Soft Actor-Critic** algorithm (continuous actions only).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `LearningRate` | float | 0.0003 | Step size for all networks (actor, critics, alpha). |
| `Gamma` | float | 0.99 | Discount factor for future rewards. |
| `MaxGradientNorm` | float | 0.5 | Gradient clipping threshold. |
| `ReplayBufferCapacity` | int | 100000 | Maximum transitions stored in the replay buffer. When full, oldest entries are overwritten. |
| `BatchSize` | int | 256 | Mini-batch size sampled from the replay buffer per gradient update. |
| `WarmupSteps` | int | 1000 | Training only begins after this many transitions are collected. Ensures a diverse initial buffer. |
| `Tau` | float | 0.005 | Polyak averaging rate for target network updates. Lower = more stable but slower target tracking. |
| `InitAlpha` | float | 0.2 | Initial entropy temperature. Controls how strongly entropy is weighted at the start. |
| `AutoTuneAlpha` | bool | true | Automatically adjust alpha to maintain the target entropy. Strongly recommended. |
| `UpdateEverySteps` | int | 1 | Run gradient updates every N environment steps. Set to 2 or more if updates are too frequent for your hardware. |
| `UpdatesPerStep` | int | 0 | Number of gradient updates per new transition. `0` = auto-scale based on worker count. |
| `TargetEntropyFraction` | float | 0.5 | Fraction of max entropy used as the entropy target for automatic alpha tuning. |

### Example: SAC for a locomotion task

```
RLSACConfig:
  LearningRate: 0.0003
  Gamma: 0.99
  ReplayBufferCapacity: 200000
  BatchSize: 256
  WarmupSteps: 2000
  Tau: 0.005
  InitAlpha: 0.2
  AutoTuneAlpha: true
  UpdatesPerStep: 1
```

---

## RLRunConfig

Controls how the training process executes — parallelism, checkpointing, and async behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RunPrefix` | string | `""` | Prefix for the run ID used in checkpoint folder names. Useful for organizing runs. |
| `SimulationSpeed` | float | 1.0 | Speed multiplier for the Godot physics engine during training. Set to 4.0 for faster training (no visual output needed). |
| `BatchSize` | int | 4 | Number of parallel environment instances to run in the master process. More instances = more diverse data per step. |
| `ActionRepeat` | int | 1 | Repeat the sampled action for N physics frames before sampling again. Reduces action frequency; useful for agents that don't need per-frame decisions. |
| `CheckpointInterval` | int | 50 | Save a checkpoint every N gradient updates. |
| `AsyncGradientUpdates` | bool | false | Run backpropagation on a background thread. The simulation continues while the gradient update happens. Can increase throughput on multi-core machines. |
| `ParallelPolicyGroups` | bool | false | Parallelize forward passes across multiple policy groups. Only useful with many distinct agent types. |
| `AsyncRolloutPolicy` | enum | `Pause` | What to do with rollouts that arrive while a background gradient update is running. `Pause` = discard and wait. `Cap` = buffer up to a limit. |

### AsyncGradientUpdates

With `AsyncGradientUpdates = true`:

```
Physics thread:   [collect] [collect] [collect] [collect] ...
Background thread:         [backprop]            [backprop] ...
```

The simulation doesn't pause during backpropagation. This improves GPU/CPU utilization at the cost of slightly stale gradients (the data used for training may be one update behind).

**Recommendation:** Enable this when using `SimulationSpeed > 1` or when training with distributed workers, where backprop is the bottleneck.

---

## RLPolicyGroupConfig

Defines one population of agents that share a neural network. All agents with the same `AgentId` string share weights.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `AgentId` | string | `"agent_0"` | Identifier for this policy group. Agents with the same `AgentId` share weights. |
| `MaxEpisodeSteps` | int | 0 | Maximum steps per episode. `0` = unlimited (use `EndEpisode()` manually). |
| `InferenceModelPath` | string | `""` | Path to a `.rlmodel` file for inference mode. Leave empty during training. |
| `NetworkGraph` | RLNetworkGraph | default | Neural network architecture for this group. |

---

## RLNetworkGraph

Defines the neural network architecture. Create a new `RLNetworkGraph` resource and attach it to `PolicyGroupConfig.NetworkGraph`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TrunkLayers` | Array<RLLayerDef> | 2×Dense(64, Tanh) | The shared layers before the policy and value heads. |
| `Optimizer` | RLOptimizerKind | Adam | Optimizer used for gradient updates. |

### RLLayerDef — Layer types

Each element in `TrunkLayers` is an `RLLayerDef` with:

| Kind | Parameters | Description |
|------|-----------|-------------|
| `Dense` | `Size`, `Activation` | Fully connected layer. The workhorse of the network. |
| `Dropout` | `Rate` | Randomly zeroes activations during training. Reduces overfitting. |
| `LayerNorm` | — | Normalizes activations across features. Stabilizes training for deep networks. |
| `Flatten` | — | Flattens input to 1D. Rarely needed for flat observation vectors. |

### Activation functions

| Name | Use case |
|------|----------|
| `Tanh` | Default. Maps to `[-1, 1]`, works well for most RL tasks. |
| `Relu` | Faster, but can cause dead neurons. Good for very deep networks. |

### Network architecture examples

**Small (fast training, simple tasks):**
```
TrunkLayers:
  - Dense(64, Tanh)
  - Dense(64, Tanh)
```

**Medium (most tasks):**
```
TrunkLayers:
  - Dense(128, Tanh)
  - Dense(128, Tanh)
```

**Large (complex observations, locomotion):**
```
TrunkLayers:
  - Dense(256, Tanh)
  - Dense(256, Tanh)
  - Dense(128, Tanh)
```

**With regularization (overfitting on small datasets):**
```
TrunkLayers:
  - Dense(128, Tanh)
  - LayerNorm
  - Dense(128, Tanh)
  - Dropout(0.1)
```

**General rule:** Deeper is not always better in RL. Start with 2×64 and scale up only if needed.

---

## RLScheduleConfig

Optionally override hyperparameters with time-varying schedules. Attach to `RLTrainingConfig.Schedules`.

Each field accepts any `RLHyperparamSchedule` resource (or `null` to use the static value from the algorithm config):

| Field | Applied to |
|-------|-----------|
| `LearningRate` | PPO + SAC |
| `EntropyCoefficient` | PPO |
| `ClipEpsilon` | PPO |
| `SacAlpha` | SAC (only when `AutoTuneAlpha = false`) |

### Built-in schedule types

**RLConstantSchedule** — fixed value throughout training.

```
RLConstantSchedule:
  Value: 0.0003
```

**RLExponentialSchedule** — decays exponentially.

```
RLExponentialSchedule:
  InitialValue: 0.001
  FinalValue:   0.0001
  DecaySteps:   500000
```

Result: `value = initial × (final/initial)^(step/decaySteps)`

**RLCosineSchedule** — cosine annealing (smooth, widely used).

```
RLCosineSchedule:
  InitialValue: 0.001
  FinalValue:   0.0001
  CycleSteps:   1000000
```

### Custom schedule

```csharp
[GlobalClass]
public partial class WarmupLinearDecaySchedule : RLHyperparamSchedule
{
    [Export] public float WarmupValue  = 0.00001f;
    [Export] public float PeakValue    = 0.0005f;
    [Export] public int   WarmupUpdates = 50;
    [Export] public float FinalValue   = 0.00005f;
    [Export] public int   TotalUpdates = 2000;

    public override float Evaluate(ScheduleContext ctx)
    {
        if (ctx.UpdateCount < WarmupUpdates)
            return Mathf.Lerp(WarmupValue, PeakValue, ctx.UpdateCount / (float)WarmupUpdates);
        float t = (ctx.UpdateCount - WarmupUpdates) / (float)(TotalUpdates - WarmupUpdates);
        return Mathf.Lerp(PeakValue, FinalValue, Mathf.Clamp(t, 0f, 1f));
    }
}
```

---

## RLCurriculumConfig

Progressively increase task difficulty as the agent improves. Attach to `RLAcademy.Curriculum`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Mode` | enum | `StepBased` | `StepBased` = progress linearly over `MaxCurriculumSteps`. `SuccessRate` = adapt based on recent episode outcomes. |
| `MaxCurriculumSteps` | long | 1000000 | (StepBased only) Total steps to go from difficulty 0 → 1. |
| `SuccessWindowEpisodes` | int | 50 | (SuccessRate only) Rolling window for computing success rate. |
| `SuccessRewardThreshold` | float | 0.5 | Episode reward above this threshold counts as a "success". |
| `PromoteThreshold` | float | 0.8 | Success rate above this → increase difficulty. |
| `DemoteThreshold` | float | 0.2 | Success rate below this → decrease difficulty. |
| `ProgressStepUp` | float | 0.1 | How much to increase curriculum progress on promotion. |
| `ProgressStepDown` | float | 0.1 | How much to decrease curriculum progress on demotion. |
| `RequireFullWindow` | bool | true | Wait until the full `SuccessWindowEpisodes` have completed before adapting. |
| `DebugProgress` | float | -1 | Force a fixed curriculum progress for testing. Set to `0.5` to test mid-difficulty. `-1` = disabled. |

### Implementing curriculum in your agent

```csharp
public partial class MyAgent : RLAgent2D
{
    private float _curriculumProgress = 0f;

    public override void OnTrainingProgress(float progress)
    {
        _curriculumProgress = progress;
    }

    public override void OnEpisodeBegin()
    {
        // Scale difficulty: start easy (small arena), grow to full size
        float arenaSize = Mathf.Lerp(100f, 500f, _curriculumProgress);
        SetupArena(arenaSize);
    }
}
```

---

## RLDistributedConfig

Launch headless worker processes to collect experience faster. Attach to `RLAcademy.DistributedConfig`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `WorkerCount` | int | 0 | Number of headless worker processes to launch. `0` = single-process training. |
| `MasterPort` | int | 7890 | TCP port the master listens on. Workers connect to this. |
| `AutoLaunchWorkers` | bool | true | Automatically start worker processes when training begins. |
| `EngineExecutablePath` | string | `""` | Custom Godot executable. Leave empty to use the current Godot binary. |
| `WorkerSimulationSpeed` | float | 4.0 | Simulation speed multiplier for workers. Higher = faster data collection. |
| `MonitorIntervalUpdates` | int | 10 | Print distributed stats to console every N updates. |
| `ShowTrainingOverlay` | bool | false | Show the spy overlay on workers (useful for debugging, but slow). |
| `VerboseLog` | bool | false | Enable verbose worker connection logs. |

### Example: 4 workers

```
RLDistributedConfig:
  WorkerCount: 4
  WorkerSimulationSpeed: 4.0
  AutoLaunchWorkers: true
  MonitorIntervalUpdates: 20
```

With 4 workers at 4× speed, you collect experience ~16× faster than single-process at 1× speed.

**Notes:**
- Workers are headless (no window). They use less RAM and CPU for rendering.
- Each worker runs the full training scene but only collects data; the master owns the neural network.
- Workers and master must use the same Godot project.

---

## RLSelfPlayConfig

Train competitive multi-agent scenarios where agents play against historical versions of themselves.

| Parameter | Type | Description |
|-----------|------|-------------|
| `Pairings` | Array<RLPolicyPairingConfig> | List of agent matchups. |

### RLPolicyPairingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `GroupA` | RLPolicyGroupConfig | — | First agent group. |
| `GroupB` | RLPolicyGroupConfig | — | Second agent group (opponent). |
| `TrainGroupA` | bool | true | Whether Group A receives gradient updates. |
| `TrainGroupB` | bool | false | Whether Group B receives gradient updates (usually frozen). |
| `HistoricalOpponentRate` | float | 0.5 | Probability of sampling a historical (frozen) opponent vs. the latest policy. |
| `FrozenCheckpointInterval` | int | 10 | Archive the current learner as a frozen opponent every N checkpoints. |
| `MaxPoolSize` | int | 20 | Maximum number of historical opponents in the pool. |
| `PfspEnabled` | bool | true | Enable Prioritized Fictitious Self-Play (harder opponents sampled more). |
| `PfspAlpha` | float | 4.0 | PFSP weighting exponent. Higher = stronger bias toward difficult opponents. |
| `WinThreshold` | float | 0.7 | Win rate above this → the opponent is "solved" and deprioritized by PFSP. |

### How self-play works

1. At each episode start, Group A (learner) is paired with an opponent.
2. With probability `HistoricalOpponentRate`, the opponent is sampled from the historical pool (weighted by PFSP).
3. Otherwise, the opponent uses the latest learner weights.
4. After the episode, Elo ratings are updated for both participants.
5. Every `FrozenCheckpointInterval` checkpoints, the current learner policy is archived into the pool.

### PFSP (Prioritized Fictitious Self-Play)

Standard self-play can degrade: the agent learns to beat recent opponents but forgets how to beat older ones. PFSP prevents this by weighting sampling toward opponents the agent *currently loses to*, ensuring it continuously improves against its hardest challenges.

```
weight(opponent) = (1 - win_rate)^PfspAlpha
```

An opponent with a 20% win rate gets much higher sampling weight than one with 80%.

---

## Agent API Reference

### Key methods to override

```csharp
public partial class MyAgent : RLAgent2D  // or RLAgent3D
{
    // Declare the action space (called once at startup)
    protected override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddDiscrete("Move", "Left", "Right", "Idle");
        builder.AddContinuous("Steer", dimensions: 2, min: -1f, max: 1f);
    }

    // Fill observations each step (called every physics frame)
    public override void CollectObservations(ObservationBuffer obs)
    {
        obs.AddNormalized(GlobalPosition.X, -500f, 500f);
        obs.Add(IsGrounded ? 1f : 0f);
        obs.AddNormalized(Target.GlobalPosition.X, -500f, 500f);
    }

    // Apply the selected action to the simulation
    protected override void OnActionsReceived(ActionBuffer actions)
    {
        int move = actions.GetDiscrete("Move");
        float[] steer = actions.GetContinuous("Steer");
    }

    // Compute rewards and end the episode when done
    protected override void OnStep()
    {
        AddReward(-0.001f);                  // step penalty
        AddReward(progress, "distance");     // named component (visible in debug overlay)
        if (reached) { AddReward(1f); EndEpisode(); }
        if (EpisodeSteps > 500) EndEpisode();
    }

    // Reset the environment for a new episode
    public override void OnEpisodeBegin()
    {
        Position = SpawnPoint();
    }

    // Receive curriculum difficulty (0 = easiest, 1 = hardest)
    public override void OnTrainingProgress(float progress)
    {
        _difficulty = progress;
    }
}
```

### ObservationBuffer methods

| Method | Description |
|--------|-------------|
| `obs.Add(float)` | Add a raw float value. |
| `obs.Add(Vector2)` | Add both X and Y components. |
| `obs.Add(Vector3)` | Add X, Y, and Z components. |
| `obs.Add(bool)` | Add `1f` for true, `0f` for false. |
| `obs.AddNormalized(value, min, max)` | Map `[min, max]` → `[-1, 1]`. |
| `obs.AddSensor(name, sensor)` | Add a reusable `IObservationSensor`. |

**Always normalize!** Neural networks learn much better when inputs are in `[-1, 1]` rather than raw pixel coordinates or world units.

### ActionBuffer methods

| Method | Description |
|--------|-------------|
| `actions.GetDiscrete(name)` | Returns `int` index of chosen discrete action. |
| `actions.GetDiscreteAsEnum<T>(name)` | Returns discrete action cast to an enum type. |
| `actions.GetContinuous(name)` | Returns `float[]` in `[-1, 1]` (tanh-squashed). |

### Reward API

| Method | Description |
|--------|-------------|
| `AddReward(float)` | Accumulate a reward this step. |
| `AddReward(float, string tag)` | Accumulate with a named tag (visible in debug overlay). |
| `SetReward(float)` | Replace the current accumulated reward. |
| `EndEpisode()` | Mark this episode as complete. The agent resets next frame. |

---

## Built-in Sensors

Reusable sensor components you can attach and add with `obs.AddSensor()`.

| Sensor | Observation size | Description |
|--------|-----------------|-------------|
| `NormalizedTransformSensor2D` | 2 | Normalized position (X, Y). |
| `NormalizedTransformSensor3D` | 3 | Normalized position (X, Y, Z). |
| `NormalizedVelocitySensor2D` | 2 | Normalized velocity. |
| `NormalizedVelocitySensor3D` | 3 | Normalized velocity. |
| `RelativePositionSensor2D` | 2 | Normalized vector to target. |
| `RelativePositionSensor3D` | 3 | Normalized vector to target. |
| `RaycastSensor3D` | N rays × K | Ray distances + hit detection. |

### Custom sensor

```csharp
public class MyHealthSensor : IObservationSensor, IObservationDebugLabels
{
    public int Size => 1;
    private float _health;

    public void Write(ObservationBuffer buffer)
    {
        buffer.AddNormalized(_health, 0f, 100f);
    }

    public IReadOnlyList<string> DebugLabels => new[] { "health" };
}
```
