# Demo Environments

Example demo environments are maintained in the companion demo repository:

- https://github.com/dron3flyv3r/rl-agent-godot

That repository contains sample scenes demonstrating single-agent navigation, competitive self-play, curriculum learning, 3D continuous control, and locomotion.

---

## Demo 01 — Single Agent (ReachTargetDemo)

**Scene:** `demo/01 SingleAgent/ReachTargetDemo.tscn` in the companion demo repository

**Algorithm:** PPO (discrete actions)

![ReachTargetDemo screenshot](images/demo_reach_target.png)
> **[Image placeholder]** Screenshot of the ReachTargetDemo: a small 2D agent on a horizontal track, with a colored target sphere it must navigate toward.

### What It Is

The simplest possible RL task: a 2D agent moving along a one-dimensional track must reach a randomly placed target. Each episode, both the agent and target are placed at random positions on the track.

This demo is the "hello world" of the plugin. Use it to verify your setup is working, understand the agent lifecycle, and experiment with reward shaping.

### How It Works

**Agent:** `ReachTargetAgent` (extends `RLAgent2D`)

**Actions (discrete):**
- `Left` — move left along the track
- `Right` — move right along the track
- `Idle` — stand still

**Observations (3 values):**
- Normalized agent position on the track `[-1, 1]`
- Normalized target position on the track `[-1, 1]`
- Normalized distance to target `[-1, 1]`

**Rewards:**
- Small per-step penalty (`-0.001`) to encourage speed
- Positive reward proportional to distance progress each step
- Large bonus (`+1.0`) when the agent reaches the target

**Episode end:** Target reached, or max steps exceeded.

### What Makes It Special

- Dead simple observation and reward design — ideal for learning the agent API
- No physics complications: pure 2D position arithmetic
- Trains in minutes with default PPO settings

### Running It

1. Open `ReachTargetDemo.tscn` from the companion demo repository
2. Click **Start Training** (top toolbar or RL Setup dock)
3. The reward should start rising within a few hundred episodes

### Training Tips

- The task is easy. If the agent isn't learning in 50k steps, the reward or observation is likely broken.
- Try removing the distance observation and see how it affects learning speed — this shows the importance of informative observations.
- Try removing the shaping reward (distance progress) and only rewarding goal-reaching — training will work but take much longer.

---

## Demo 02 — Multi-Agent Self-Play (TagDemo)

**Scene:** `demo/02 MultiAgentSelfPlay/TagDemo.tscn` in the companion demo repository

**Algorithm:** PPO (discrete actions) + Self-Play

![TagDemo screenshot](images/demo_tag.png)
> **[Image placeholder]** Screenshot of the TagDemo: a grid arena with 8 agents, half in one color and half in another, competing in a tag-like game.

### What It Is

Eight 2D agents compete in a tag-style game. One team tags; the other team evades. Agents are trained using **self-play**: they play against historical frozen snapshots of themselves, preventing overfitting to a fixed opponent.

This demo shows how to set up competitive multi-agent scenarios with the `RLSelfPlayConfig` and `RLPolicyPairingConfig` resources.

### How It Works

**Agent:** `TagAgent` (extends `RLAgent2D`)

**Actions (discrete):**
- `Idle`
- `Up`, `Down`, `Left`, `Right`

**Observations:**
- Own position (normalized)
- Positions of nearby opponents (normalized)
- Own role (tagger / runner)
- Distance to nearest opponent

**Rewards:**
- Tagger: positive reward for closing distance to runners; large bonus for tagging
- Runner: positive reward for maintaining distance from taggers; large penalty for being tagged

**Arena Controller (`TagArenaController`):**
- Tracks who tagged whom
- Resets positions each episode
- Assigns roles

### What Makes It Special

- **Self-play loop**: the agent plays against its own past checkpoints. As it improves, its opponents improve too, ensuring it always faces an appropriate challenge.
- **Elo tracking**: the dashboard shows Elo ratings rising as the learner improves.
- **PFSP**: hard opponents (those with low win rates vs. the learner) are sampled more frequently, preventing skill stagnation.

### Self-Play Configuration

```
RLSelfPlayConfig:
  Pairings:
    - GroupA: "tagger"  (TrainGroupA: true)
      GroupB: "runner"  (TrainGroupB: true)
      HistoricalOpponentRate: 0.5
      FrozenCheckpointInterval: 10
      MaxPoolSize: 20
      PfspEnabled: true
      PfspAlpha: 4.0
```

### Running It

1. Open `TagDemo.tscn` from the companion demo repository
2. Click **Start Training** (top toolbar or RL Setup dock)
3. Watch the Elo chart — it should rise as the learner improves

### Training Tips

- Self-play training is slower to show progress early: the agent starts against a random baseline.
- Monitor Elo rather than episode reward — Elo is the true signal of improvement.
- If Elo plateaus: try reducing `HistoricalOpponentRate` (face current policy more) or increasing `PfspAlpha` (harder focus).
- Training both groups (`TrainGroupA` and `TrainGroupB`) makes both groups stronger but doubles compute.

---

## Demo 03 — Curriculum Learning (WallClimbDemo)

**Scene:** `demo/03 WallClimbCurriculum/WallClimbDemo.tscn` in the companion demo repository

**Algorithm:** PPO (continuous actions) + Curriculum

![WallClimbDemo screenshot](images/demo_wall_climb.png)
> **[Image placeholder]** Screenshot of the WallClimbDemo: a 3D arena with a box and a wall of varying height. The agent must push the box over the wall to reach the goal on the other side.

### What It Is

A 3D agent must push a box over a wall to reach a goal on the other side. The wall height is controlled by the curriculum: at difficulty 0, the wall is low (easy); at difficulty 1, the wall is tall (hard).

This demo shows how to implement curriculum learning to gradually increase task difficulty as the agent improves, preventing it from facing an unsolvable task from scratch.

### How It Works

**Agent:** `WallClimbAgent` (extends `RLAgent3D`)

**Actions (continuous, 3 dimensions):**
- Move X (`[-1, 1]`)
- Move Z (`[-1, 1]`)
- Jump (`[-1, 1]`, threshold at 0.5)

**Observations:**
- Agent position (X, Y, Z), normalized
- Agent velocity
- Is grounded (bool)
- Box position and velocity
- Relative vector to box
- Relative vector to goal
- Current wall height (curriculum indicator)
- Optional raycasts around the agent

**Rewards:**
- Positive reward for moving the box toward the goal
- Large bonus for box reaching the goal zone
- Small step penalty

**Curriculum integration:**

```csharp
public override void OnTrainingProgress(float progress)
{
    _curriculumProgress = progress;
}

public override void OnEpisodeBegin()
{
    float wallHeight = Mathf.Lerp(MinWallHeight, MaxWallHeight, _curriculumProgress);
    Arena.SetWallHeight(wallHeight);
}
```

**Curriculum config:**
```
RLCurriculumConfig:
  Mode: SuccessRate
  SuccessWindowEpisodes: 50
  SuccessRewardThreshold: 0.7
  PromoteThreshold: 0.8
  DemoteThreshold: 0.2
  ProgressStepUp: 0.1
  ProgressStepDown: 0.05
```

### What Makes It Special

- The task is **impossible** to solve from scratch at full difficulty. Without curriculum, the agent never sees a success and never learns.
- The curriculum automatically adapts: if the agent is succeeding too easily, difficulty increases; if it's struggling, difficulty decreases.
- Watch the **Curriculum Progress** chart in the dashboard — it should show a staircase pattern as the agent is promoted through difficulty levels.

### Running It

1. Open `WallClimbDemo.tscn` from the companion demo repository
2. Click **Start Training** (top toolbar or RL Setup dock)
3. Watch both the reward chart and the curriculum progress chart
4. Expect training to take longer than simpler tasks (the agent needs to master each difficulty level)

### Training Tips

- Set `DebugProgress = 0.0` in `RLCurriculumConfig` to test the easiest version. Verify the agent can reliably solve it before enabling curriculum.
- If the curriculum stalls at low progress: the agent is stuck; try adding more informative shaping rewards.
- Including the wall height as an observation lets the agent adapt its strategy to the current difficulty. Without this, the agent can't distinguish easy from hard episodes.
- Continuous actions give the agent more precise control for this task than discrete actions would.

---

## Demo 04 — Move to Target 3D

**Scene:** `demo/04 MoveToTarget3D/MoveToTarget3D.tscn` in the companion demo repository

**Algorithm:** PPO or SAC (continuous actions)

![MoveToTarget3D screenshot](images/demo_move_to_target_3d.png)
> **[Image placeholder]** Screenshot of the MoveToTarget3D demo: a 3D environment with a capsule agent and a floating target sphere, set in a simple flat arena.

### What It Is

The 3D equivalent of Demo 01. A capsule agent must navigate to a randomly placed target in a 3D arena. This is the simplest 3D continuous control task.

Use this demo to:
- Learn the `RLAgent3D` API
- Compare PPO vs. SAC on a simple continuous task
- Test 3D observation normalization

### How It Works

**Agent:** `MoveToTarget3DAgent` (extends `RLAgent3D`)

**Actions (continuous, 3 dimensions):**
- Move X, Move Y, Move Z (`[-1, 1]` each)

**Observations:**
- Agent position (normalized)
- Agent velocity
- Target position (normalized) or relative vector to target

**Rewards:**
- Per-step shaping: negative distance to target
- Goal reached: `+1.0` bonus

### What Makes It Special

- Direct comparison point for 2D vs. 3D agent setup
- Good benchmark for testing network sizes on a continuous task
- Shows how to handle 3D physics bodies as RL agents

### Running It

Try both PPO and SAC configs and compare convergence speed:

**PPO (faster start):**
```
RLPPOConfig:
  RolloutLength: 2048
  EpochsPerUpdate: 10
  LearningRate: 0.0003
```

**SAC (better final performance):**
```
RLSACConfig:
  LearningRate: 0.0003
  WarmupSteps: 2000
  AutoTuneAlpha: true
```

---

## Demo 05 — Crawler (Locomotion)

**Scene:** `demo/05 Crawler/CrawlerDemo.tscn` in the companion demo repository

**Algorithm:** SAC or PPO (continuous actions)

![Crawler screenshot](images/demo_crawler.png)
> **[Image placeholder]** Screenshot of the Crawler demo: a multi-limbed creature built from rigid bodies and joints, navigating forward across a flat surface.

### What It Is

A multi-limbed creature — built from Godot `RigidBody3D` and `Joint` nodes — must learn to walk forward. This is the most complex demo and the closest to real-world locomotion research tasks.

Locomotion is a hard problem: the agent must coordinate multiple joints simultaneously to produce coherent motion, while a naive random policy immediately falls over.

### How It Works

**Agent:** `CrawlerAgent` (extends `RLAgent3D`)

**Body:** `CrawlerBody` — manages joint targets and computes proprioceptive observations.

**Actions (continuous):**
- Target angles for each joint (`[-1, 1]` scaled to joint range)
- Each limb has 2–3 degrees of freedom

**Observations (proprioception):**
- Base position and orientation
- Base linear and angular velocity
- Per-joint: current angle, angular velocity
- Forward direction vector (from base orientation)

**Rewards:**
- Forward velocity bonus (primary)
- Small penalty for action magnitude (encourages smooth motion)
- Penalty if the body falls below a height threshold

### What Makes It Special

- **Coordination challenge**: the agent must learn to synchronize many joints.
- **Emergent gait**: the walking gait is not programmed; it emerges from reward optimization.
- **Proprioception**: unlike the simpler demos, the agent observes its own joint states, not just position.
- Great for experimenting with network size — larger networks (3×256) significantly outperform small ones here.

### Running It

This task benefits from SAC:

```
RLSACConfig:
  LearningRate: 0.0003
  ReplayBufferCapacity: 500000
  BatchSize: 256
  WarmupSteps: 5000
  UpdatesPerStep: 1
  AutoTuneAlpha: true
```

And a larger network:

```
RLNetworkGraph:
  TrunkLayers:
    - Dense(256, Tanh)
    - Dense(256, Tanh)
    - Dense(128, Tanh)
```

**Distributed training recommended:** Use 4–8 workers to speed up data collection.

```
RLDistributedConfig:
  WorkerCount: 4
  WorkerSimulationSpeed: 4.0
```

### Training Tips

- Locomotion takes much longer than navigation tasks. Budget 1–5 million steps.
- Monitor the **forward velocity** reward component specifically (use named rewards).
- If the agent immediately falls and stays down: add a larger orientation penalty and check that joint limits are configured correctly.
- Action smoothing (penalizing large action deltas) prevents jittery motion: `penalty = -0.001 × mean(|a_t - a_{t-1}|²)`.
- For fast prototyping, reduce the number of joints by simplifying the body.

---

## Comparison Table

| Demo | Dimensions | Actions | Algorithm | Difficulty | Key Feature |
|------|-----------|---------|-----------|------------|------------|
| 01 Reach Target | 2D | Discrete | PPO | Easy | Agent API basics |
| 02 Tag Self-Play | 2D | Discrete | PPO | Medium | Multi-agent, self-play, Elo |
| 03 Wall Climb | 3D | Continuous | PPO | Medium-Hard | Curriculum learning |
| 04 Move to Target 3D | 3D | Continuous | PPO/SAC | Easy | 3D continuous control |
| 05 Crawler | 3D | Continuous | SAC | Hard | Locomotion, proprioception |

---

## Building Your Own Environment

Use the demos as templates. The minimum required structure:

```
TrainingScene (Node)
└── RLAcademy           (configure all resources here)
    └── Environment
        └── MyAgent     (your RLAgent2D or RLAgent3D subclass)
```

Start from the simplest demo that resembles your task:

- **Navigation or simple control** → start from Demo 01 (2D) or Demo 04 (3D)
- **Competitive multiplayer** → start from Demo 02
- **Gradually increasing difficulty** → start from Demo 03
- **Locomotion or complex joints** → start from Demo 05

The key things to implement are `DefineActions()`, `CollectObservations()`, `OnActionsReceived()`, `OnStep()`, and `OnEpisodeBegin()`. Everything else is handled by the plugin.
