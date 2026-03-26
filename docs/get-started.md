# Get Started Guide

This guide walks a new user through:
1. Plugin setup
2. Creating a simple training scene
3. Building a first agent (Demo 04 style: cube to target point)
4. Starting training from the editor UI
5. Exporting and using a `.rlmodel`

---

## 1) Setup

### Prerequisites

- Godot 4.6+ with C# enabled
- .NET 8 SDK installed

### Install plugin

1. Copy the plugin to `addons/rl-agent-plugin` in your project.
2. Enable it in **Project Settings → Plugins**.
3. Build the project once (`Alt+B`).

After setup, verify the editor UI:
- **RLDash** tab exists (main screen row)
- **RL Setup** dock exists (right side)
- Top toolbar has **Start Training**, **Stop Training**, **Run Inference**

---

## 2) Build a simple scene (cube agent to target point)

This is a simple plan similar to Demo 04:
- A cube (player) starts at random position.
- A target point is spawned somewhere in the arena.
- Agent gets reward for moving closer to the target.
- Episode ends when target is reached or max steps are exceeded.

### Recommended node structure

```text
MoveToPoint3D (Node3D)
├── RLAcademy
├── Player (CharacterBody3D)               # can also be Node3D/Node2D style depending on game
│   ├── MeshInstance3D (CubeMesh)
│   └── Agent (RLAgent3D script)           # extend script here for RL logic
└── Target (Marker3D)
```

> 2D equivalent pattern:
> - `Player` can be `CharacterBody2D` or `Node2D` with your movement script.
> - Add `Agent` as a child using `RLAgent2D`.

### Why this structure matters

Use your **player node** for movement/physics and your **agent child node** for RL decisions.

Flow each step:
1. Agent collects observations.
2. Agent receives an action.
3. Agent forwards that decision to the parent player node.
4. Player applies movement.

This keeps gameplay code and RL code cleanly separated.

---

## 3) Create the player script (movement owner)

Attach this to `Player` (`CharacterBody3D`):

```csharp
using Godot;

public partial class SimplePlayer3D : CharacterBody3D
{
    [Export] public float MoveSpeed = 6.0f;

    private Vector2 _moveInput = Vector2.Zero;

    public void SetAgentMoveInput(Vector2 move)
    {
        _moveInput = move.ClampLength(1.0f);
    }

    public override void _PhysicsProcess(double delta)
    {
        var input = new Vector3(_moveInput.X, 0f, _moveInput.Y);
        Velocity = input * MoveSpeed;
        MoveAndSlide();
    }
}
```

---

## 4) Create the agent script (RL logic on child node)

In the scene tree, select `Player/Agent` and click **Extend Script**. Make it inherit `RLAgent3D`.

Attach this script to that child `Agent` node:

```csharp
using Godot;
using RlAgentPlugin.Runtime;

public partial class MoveToPointAgent3D : RLAgent3D
{
    [Export] public NodePath PlayerPath = "..";
    [Export] public NodePath TargetPath = "../../Target";
    [Export] public float ArenaHalfSize = 8f;
    [Export] public float ReachDistance = 0.75f;

    private SimplePlayer3D? _player;
    private Node3D? _target;

    public override void _Ready()
    {
        base._Ready();
        _player = GetNodeOrNull<SimplePlayer3D>(PlayerPath);
        _target = GetNodeOrNull<Node3D>(TargetPath);
    }

    public override void DefineActions(ActionSpaceBuilder builder)
    {
        // 2 continuous actions: move X and move Z in [-1, 1]
        builder.AddContinuous("MoveX", 1, -1f, 1f);
        builder.AddContinuous("MoveZ", 1, -1f, 1f);
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        if (_player is null || _target is null)
        {
            obs.Add(0f); obs.Add(0f); obs.Add(0f); obs.Add(0f);
            return;
        }

        var p = _player.GlobalPosition;
        var t = _target.GlobalPosition;

        // Basic normalized observations
        obs.AddNormalized(p.X, -ArenaHalfSize, ArenaHalfSize);
        obs.AddNormalized(p.Z, -ArenaHalfSize, ArenaHalfSize);
        obs.AddNormalized(t.X, -ArenaHalfSize, ArenaHalfSize);
        obs.AddNormalized(t.Z, -ArenaHalfSize, ArenaHalfSize);
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        if (_player is null) return;

        // Forward RL decision to parent player node
        var move = new Vector2(
            actions.GetContinuous("MoveX")[0],
            actions.GetContinuous("MoveZ")[0]);

        _player.SetAgentMoveInput(move);
    }

    public override void OnStep()
    {
        if (_player is null || _target is null) return;

        var dist = _player.GlobalPosition.DistanceTo(_target.GlobalPosition);

        // Dense shaping toward target
        AddReward(-dist * 0.001f);

        // Success condition
        if (dist <= ReachDistance)
        {
            AddReward(1.0f);
            EndEpisode();
        }

        if (EpisodeSteps > 600)
            EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        if (_player is null || _target is null) return;

        _player.GlobalPosition = new Vector3(
            (float)GD.RandRange(-ArenaHalfSize, ArenaHalfSize),
            0f,
            (float)GD.RandRange(-ArenaHalfSize, ArenaHalfSize));

        _target.GlobalPosition = new Vector3(
            (float)GD.RandRange(-ArenaHalfSize, ArenaHalfSize),
            0f,
            (float)GD.RandRange(-ArenaHalfSize, ArenaHalfSize));

        _player.SetAgentMoveInput(Vector2.Zero);
    }
}
```

---

## 5) Configure `RLAcademy`

On the `RLAcademy` node:

1. Create and assign `RLTrainingConfig`
2. Set `Algorithm` to `RLPPOConfig` (good starting point)
3. Assign `RLRunConfig` and start with:
   - `BatchSize = 4`
   - `SimulationSpeed = 1.0`
4. Set `MaxEpisodeSteps = 600`

For first experiments, defaults are usually enough.

---

## 6) Use RL Setup dock (right side)

Use **RL Setup** as your pre-launch checklist:

- **Scene**: confirms which scene will launch
- **Validation**: checks missing configs/invalid setup
- **Resources**: shows resolved config assets
- **Start/Stop/Quick Test**: launch controls

If validation shows errors, fix those before training.

---

## 7) Start training

You can start training in two places:

- Top toolbar: **Start Training**
- Right dock: **RL Setup → Start Training**

You do **not** need to switch to RLDash to launch training.

Open **RLDash** to monitor progress (reward, losses, entropy, episode length, and optional curriculum/Elo charts).

---

## 8) Export model as `.rlmodel`

After or during a run, export from **RLDash**:

- Use **Export Run** to export run outputs
- Or expand **Checkpoint History** and use row-level **Export**

This creates one or more `.rlmodel` files.

---

## 9) Use exported model for inference

Set your agent to inference and point to the exported `.rlmodel`.

### Option A (per-agent)
- In the agent Inspector:
  - `ControlMode = Inference`
  - `PolicyGroupConfig.InferenceModelPath = res://path/to/model.rlmodel`

### Option B (Auto mode)
- Keep `ControlMode = Auto`
- Set `PolicyGroupConfig.InferenceModelPath` to `.rlmodel`
- Use **Run Inference** in toolbar

---

## 10) Recommended first run checklist

- Observation values are normalized (roughly `[-1, 1]`)
- Reward is not always zero
- `EndEpisode()` is reachable
- `MaxEpisodeSteps` is reasonable
- Training starts from **Start Training** button
- Reward trend improves over time in RLDash
- `.rlmodel` export succeeds and loads in inference

---


## Tips & Tricks

- **Start with one clear objective**: one agent, one target, one success condition. Avoid adding multiple goals in your first environment.
- **Use normalized observations**: keep values around `[-1, 1]` whenever possible.
- **Keep rewards simple**: a small shaping reward + clear terminal success reward is usually enough for first runs.
- **Add a small step penalty**: this encourages faster solutions and prevents stalling behavior.
- **Use RL Setup validation before every run**: if validation reports issues, fix those first instead of tuning hyperparameters.
- **Use Quick Test first**: run short smoke tests to confirm the loop works before long training sessions.
- **Watch entropy with reward**: reward rising while entropy gradually falls is often healthier than reward-only tracking.
- **Export often**: save useful `.rlmodel` snapshots during good runs so you can compare behavior quickly.
- **Separate gameplay from RL code**: keep movement/physics in player scripts and policy logic in the child `RLAgent2D/3D` node.
- **If learning stalls**: simplify the task (smaller arena, shorter distance, easier reset) before increasing model size.

---

## Next docs

- Configuration details: `configuration.md`
- Algorithm tradeoffs: `algorithms.md`
- Reading charts and debugging learning: `tuning.md`
- Demo-specific walkthroughs: `demos.md`
