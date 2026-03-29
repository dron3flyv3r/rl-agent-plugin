# Get Started Guide

This guide is the shortest concrete path from "plugin installed" to "my first agent is training".

It covers:

1. project setup
2. the core 2D/3D differences
3. building a minimal first training scene
4. creating the required config resources
5. starting training from the editor
6. exporting a trained `.rlmodel`

The first walkthrough uses a simple 2D task because it is the fastest way to confirm your setup works. After that, there is a dedicated section on how the 3D version differs.

---

## 1) Before You Start

Prerequisites:

- Godot 4.6+ with C# support
- .NET 8 SDK installed
- the plugin copied to `addons/rl-agent-plugin` from release tag `v0.1.0-beta`

Recommended install source (release-pinned):

```bash
git submodule add https://github.com/dron3flyv3r/rl-agent-plugin.git addons/rl-agent-plugin
git submodule update --init --recursive
git -C addons/rl-agent-plugin fetch --tags
git -C addons/rl-agent-plugin checkout v0.1.0-beta
```

Install steps:

1. Open the project in Godot.
2. Go to **Project Settings -> Plugins**.
3. Enable **RL Agent Plugin**.
4. Build once with `Alt+B`.

You should now see:

- **RLDash** in the main screen tabs
- **RL Setup** as a right-side dock
- **Start Training**, **Stop Training**, and **Run Inference** in the top toolbar

If those do not appear, build once and restart the editor.

---

## 2) 2D vs 3D: Start With The Right Base

The plugin supports both 2D and 3D, but the major differences are practical rather than architectural.

| Topic | 2D | 3D |
|------|----|----|
| Agent base class | `RLAgent2D` | `RLAgent3D` |
| Agent node type | `Node2D` | `Node3D` |
| Typical player/body node | `CharacterBody2D` | `CharacterBody3D` |
| Common movement plane | `X/Y` | usually `X/Z` with `Y` as height |
| Best first environment | top-down navigation, lane tasks, simple avoidance | target reaching, locomotion, physics-heavy tasks |
| Camera sensor | `RLCameraSensor2D` is available | no equivalent 3D camera sensor in this plugin today |
| Common sensor style | vector observations, 2D raycasts, optional 2D camera | vector observations, 3D raycasts, velocity/transform sensors |
| Debug difficulty | lower | higher because physics and spatial setup are usually more complex |

Recommended starting point:

- new to the plugin: start with 2D
- already building a 3D game and comfortable with Godot physics: start with 3D

The training loop, resources, dashboard, checkpoints, and export flow are the same in both versions.

---

## 3) What You Are Building

The first example is a minimal 2D "move to target" task:

- one player object moves in a bounded arena
- one target marker is placed randomly
- the agent observes its own position and the target position
- the agent gets a small reward for moving closer
- the episode ends when the target is reached

This is deliberately simple. If this setup does not learn, the issue is usually in scene wiring, observations, or rewards, not in hyperparameter tuning.

---

## 4) Create The Scene

Create a new scene with this structure:

```text
MoveToPoint2D (Node2D)
├── RLAcademy
├── Player (CharacterBody2D)
│   ├── CollisionShape2D
│   ├── Sprite2D
│   └── Agent (RLAgent2D + RLAgent2D script)
└── Target (Marker2D)
```

Responsibilities:

- `RLAcademy`: owns the training/run resources
- `Player`: owns movement and physics
- `Agent`: owns observations, rewards, and actions
- `Target`: a simple marker the agent should reach

Keep the gameplay logic on `Player` and the RL logic on the child `Agent`. That separation is one of the most important conventions in this plugin.

---

## 5) Create The Player Script

Attach this to `Player`:

```csharp
using Godot;

public partial class SimplePlayer2D : CharacterBody2D
{
    [Export] public float MoveSpeed { get; set; } = 180f;

    private Vector2 _moveInput = Vector2.Zero;

    public void SetMoveInput(Vector2 move)
    {
        _moveInput = move.LimitLength(1f);
    }

    public override void _PhysicsProcess(double delta)
    {
        Velocity = _moveInput * MoveSpeed;
        MoveAndSlide();
    }
}
```

This script does only one thing: apply movement each physics frame based on the latest intent from the agent.

---

## 6) Create The Agent Script

Select `Player/Agent`, create a script that inherits `RLAgent2D`, and attach this:

```csharp
using Godot;
using RlAgentPlugin.Runtime;

public partial class MoveToPointAgent2D : RLAgent2D
{
    [Export] public NodePath PlayerPath { get; set; } = "..";
    [Export] public NodePath TargetPath { get; set; } = "../../Target";
    [Export] public Vector2 ArenaHalfExtents { get; set; } = new(300f, 180f);
    [Export] public float ReachDistance { get; set; } = 24f;

    private readonly RandomNumberGenerator _rng = new();

    private SimplePlayer2D? _player;
    private Node2D? _target;
    private float _previousDistance;

    public override void _Ready()
    {
        base._Ready();
        _rng.Randomize();
        _player = GetNodeOrNull<SimplePlayer2D>(PlayerPath);
        _target = GetNodeOrNull<Node2D>(TargetPath);
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddContinuous("move", dimensions: 2);
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is null || _target is null)
        {
            obs.Add(0f); obs.Add(0f);
            obs.Add(0f); obs.Add(0f);
            obs.Add(0f); obs.Add(0f);
            return;
        }

        var playerPos = _player.GlobalPosition;
        var targetPos = _target.GlobalPosition;
        var delta = targetPos - playerPos;

        obs.AddNormalized(playerPos.X, -ArenaHalfExtents.X, ArenaHalfExtents.X);
        obs.AddNormalized(playerPos.Y, -ArenaHalfExtents.Y, ArenaHalfExtents.Y);
        obs.AddNormalized(targetPos.X, -ArenaHalfExtents.X, ArenaHalfExtents.X);
        obs.AddNormalized(targetPos.Y, -ArenaHalfExtents.Y, ArenaHalfExtents.Y);
        obs.AddNormalized(delta.X, -ArenaHalfExtents.X * 2f, ArenaHalfExtents.X * 2f);
        obs.AddNormalized(delta.Y, -ArenaHalfExtents.Y * 2f, ArenaHalfExtents.Y * 2f);
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_player is null) return;

        var move = actions.GetContinuous("move");
        _player.SetMoveInput(new Vector2(move[0], move[1]));
    }

    public override void OnStep()
    {
        if (_player is null || _target is null) return;

        AddReward(-0.001f, "step_penalty");

        var currentDistance = _player.GlobalPosition.DistanceTo(_target.GlobalPosition);
        var progress = _previousDistance - currentDistance;
        AddReward(progress * 0.01f, "distance_progress");

        if (currentDistance <= ReachDistance)
        {
            AddReward(1.0f, "goal_reached");
            EndEpisode();
        }

        _previousDistance = currentDistance;
    }

    public override void OnEpisodeBegin()
    {
        if (_player is null || _target is null) return;

        var playerPos = new Vector2(
            _rng.RandfRange(-ArenaHalfExtents.X, ArenaHalfExtents.X),
            _rng.RandfRange(-ArenaHalfExtents.Y, ArenaHalfExtents.Y));

        var targetPos = new Vector2(
            _rng.RandfRange(-ArenaHalfExtents.X, ArenaHalfExtents.X),
            _rng.RandfRange(-ArenaHalfExtents.Y, ArenaHalfExtents.Y));

        while (playerPos.DistanceTo(targetPos) < ReachDistance * 3f)
        {
            targetPos = new Vector2(
                _rng.RandfRange(-ArenaHalfExtents.X, ArenaHalfExtents.X),
                _rng.RandfRange(-ArenaHalfExtents.Y, ArenaHalfExtents.Y));
        }

        _player.GlobalPosition = playerPos;
        _player.SetMoveInput(Vector2.Zero);
        _target.GlobalPosition = targetPos;
        _previousDistance = playerPos.DistanceTo(targetPos);
    }
}
```

What this script does:

- declares a 2D continuous action space
- collects normalized observations
- converts actions into player movement
- gives a shaped reward for progress toward the target
- resets both player and target every episode

---

## 7) Create The Required Resources

You need four resources for a first run:

1. `RLTrainingConfig`
2. `RLPPOConfig`
3. `RLRunConfig`
4. `RLPolicyGroupConfig`

You will also usually want:

5. `RLNetworkGraph`

### Training resource chain

Create an `RLTrainingConfig` resource.

Inside it:

- set `Algorithm` to a new `RLPPOConfig`

Recommended first values:

```text
RLPPOConfig
- RolloutLength: 256
- EpochsPerUpdate: 4
- MiniBatchSize: 64
- LearningRate: 0.0005
- EntropyCoefficient: 0.01
```

The defaults are already close to a good first run, so you do not need to over-tune this.

### Run config

Create an `RLRunConfig` resource and start with:

```text
RLRunConfig
- BatchSize: 4
- SimulationSpeed: 1.0
- ActionRepeat: 1
- CheckpointInterval: 10
- AsyncGradientUpdates: false
```

Why these values:

- `BatchSize = 4` gives enough parallel data to learn quickly
- `ActionRepeat = 1` keeps the first environment simple and predictable
- `SimulationSpeed = 1.0` makes debugging easier

### Policy group config

Create an `RLPolicyGroupConfig` resource and set:

```text
RLPolicyGroupConfig
- AgentId: "player"
- MaxEpisodeSteps: 300
- InferenceModelPath: leave empty for training
- NetworkGraph: new RLNetworkGraph
```

Important:

- every agent that should share weights must use the same `AgentId`
- for this first scene, you only have one policy group

### Network graph

Create an `RLNetworkGraph` resource and keep the default trunk if you want the quickest first success:

```text
RLNetworkGraph
- TrunkLayers: Dense(64, Tanh) -> Dense(64, Tanh)
- Optimizer: Adam
```

---

## 8) Assign The Resources In The Inspector

Assign resources like this:

### On `RLAcademy`

- `TrainingConfig` -> your `RLTrainingConfig`
- `RunConfig` -> your `RLRunConfig`
- `MaxEpisodeSteps` -> leave `0` if the policy group sets it, or set a global cap here

### On `Player/Agent`

- `PolicyGroupConfig` -> your `RLPolicyGroupConfig`
- `ControlMode` -> leave at `Auto` for normal training/inference workflow

For this first run, do not assign:

- `DistributedConfig`
- `Curriculum`
- `SelfPlay`
- `InferenceModelPath`

Those features are useful later, but they add failure modes you do not want in your first validation run.

---

## 9) Validate The Scene

Before training:

1. Open the **RL Setup** dock.
2. Make sure it points at your current scene.
3. Read the validation output.
4. Fix any missing config errors before pressing Start.

The most common first-run mistakes are:

- `RLAcademy` missing `TrainingConfig`
- agent missing `PolicyGroupConfig`
- scene builds but the agent never calls `AddReward()`
- observations are not normalized

---

## 10) Start The First Training Run

Start training from either:

- the top toolbar: **Start Training**
- the right dock: **RL Setup -> Start Training**

Then open **RLDash** and watch:

- episode reward
- episode length
- policy loss
- value loss
- entropy

What a healthy first run usually looks like:

- reward starts noisy and low
- entropy starts high
- reward slowly rises over time
- entropy gradually declines as the policy becomes more confident

If reward stays completely flat for a long time, check the scene logic before touching hyperparameters.

---

## 11) What To Check If It Is Not Learning

Use this checklist before tuning:

- `CollectObservations()` is actually writing data every step
- observations are roughly in `[-1, 1]`
- `OnActionsReceived()` changes the player behavior
- `OnStep()` calls `AddReward()`
- `EndEpisode()` can happen
- `MaxEpisodeSteps` is not so low that the agent never reaches the target
- the target is not spawning on top of the player or impossibly far away

If any of those are wrong, training quality will be poor no matter which algorithm you choose.

---

## 12) Export A Model

After the run starts producing useful behavior:

1. Open **RLDash**.
2. Use **Export Run** or export from a checkpoint row.
3. Save the generated `.rlmodel`.

That file contains the trained policy for inference.

---

## 13) Run Inference

To use the exported model:

### Option A: per-agent inference setup

On the agent:

- set `ControlMode = Inference`
- set `PolicyGroupConfig.InferenceModelPath` to the exported `.rlmodel`
- press **Run Inference** from the toolbar

### Option B: Auto mode

- keep `ControlMode = Auto`
- set `PolicyGroupConfig.InferenceModelPath`
- press **Run Inference** from the toolbar

Inference uses the trained model without gradient updates.

---

## 14) How The 3D Version Differs

Once the 2D version works, moving to 3D is mostly a scene and movement conversion exercise.

### The major differences

| Area | 2D version | 3D version |
|------|------------|------------|
| Agent type | `RLAgent2D` | `RLAgent3D` |
| Agent node | `Node2D` | `Node3D` |
| Player body | `CharacterBody2D` | `CharacterBody3D` |
| Position observations | usually `X`, `Y` | usually `X`, `Z` for planar movement |
| Gravity | usually none unless platformer | usually part of movement logic |
| Movement output | `Vector2` | `Vector3` |
| Camera/image path | 2D camera sensor is available | no matching 3D camera sensor here |
| Common first task | top-down target reach | target reach on a flat plane |

### Typical 3D scene structure

```text
MoveToPoint3D (Node3D)
├── RLAcademy
├── Player (CharacterBody3D)
│   ├── MeshInstance3D
│   ├── CollisionShape3D
│   └── Agent (Node3D + RLAgent3D script)
└── Target (Marker3D)
```

### Code changes you usually make

- change `RLAgent2D` to `RLAgent3D`
- change `Node2D` to `Node3D`
- change `Vector2` to `Vector3`
- use `X/Z` for planar movement and keep `Y` for height or gravity
- update the player script to move in 3D

### When 3D gets harder

Compared with 2D, 3D setups more often fail because of:

- unstable physics
- unnormalized vertical velocity or height signals
- rewards that do not reflect partial progress
- too-large arenas
- action spaces that are too big for a first run

If you are new, keep the first 3D task as simple as the 2D one:

- flat floor
- one target
- no jumping
- no rigidbody interactions
- one clear success condition

For concrete 3D examples, look at the demo scenes in [demos.md](demos.md).

---

## 15) Recommended First-Run Checklist

- scene contains exactly one `RLAcademy`
- agent is a child of the player, not the other way around
- player owns movement and physics
- agent owns observations, rewards, and actions
- `PolicyGroupConfig` is assigned on the agent
- `TrainingConfig` and `RunConfig` are assigned on `RLAcademy`
- reward is non-zero during training
- episode resets are happening
- `RLDash` shows metrics updating

---

## Tips & Tricks

- start with PPO unless you already know you need SAC
- start with one agent and one objective
- keep observation values normalized
- keep rewards simple and interpretable
- use a small step penalty to encourage faster solutions
- do not start with curriculum, self-play, or distributed workers
- if learning stalls, simplify the environment before increasing model size
- if you use camera observations, read [gpu-cnn.md](gpu-cnn.md) to understand how GPU training activation actually works

---

## Next Docs

- configuration details: [configuration.md](configuration.md)
- algorithm tradeoffs: [algorithms.md](algorithms.md)
- charts and debugging learning: [tuning.md](tuning.md)
- plugin architecture: [architecture.md](architecture.md)
- sensors and camera streams: [sensors.md](sensors.md)
- demo walkthroughs: [demos.md](demos.md)
