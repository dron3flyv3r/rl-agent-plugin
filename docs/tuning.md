# Tuning Guide: Understanding Your Training Data

This guide explains how to read your training charts, what healthy vs. unhealthy curves look like, and how to systematically improve your agent's performance.

---

## The Dashboard

Open **RLDash** from the editor main-screen tabs. It polls `RL-Agent-Training/runs/<RunId>/metrics__*.jsonl` (one file per policy group) every 2 seconds and displays live charts.

![Dashboard charts screenshot](images/dashboard_charts.png)
> **[Image placeholder]** Screenshot of the dashboard showing episode reward, policy loss, entropy, and episode length charts for a healthy training run.

### Available Charts

| Chart | What it shows |
|-------|--------------|
| **Episode Reward** | Mean reward per episode. The most important metric. Should trend upward. |
| **Episode Length** | Mean steps per episode. May increase (agent survives longer) or decrease (agent solves faster). |
| **Policy Loss** | PPO: how much the policy changed relative to the clipped target. SAC: actor loss. |
| **Value Loss** | PPO: how well the value network predicts returns. Lower is better. |
| **Entropy** | How random the policy is. Too low = not exploring. Too high = ignoring rewards. |
| **Curriculum Progress** | Current difficulty [0, 1]. Only shown when curriculum is active. |
| **Elo** | Learner strength in self-play. Should trend upward. |

---

## Reading the Reward Curve

The episode reward chart is your primary signal.

### Healthy reward curve

```
 Reward
   │
1.0┤                              ████████████
   │                         █████
0.5┤                    █████
   │              ██████
0.0┤  ████████████
   └─────────────────────────────────── Steps
    0      100k    200k    300k    400k
```

- Starts low or near zero (random policy).
- Gradually increases as the agent learns.
- Plateaus when converged.
- Some noise is expected — use a smoothed trend line.

### Flat reward curve (not learning)

```
 Reward
0.05┤  ─────────────────────────────────────
0.00┤
    └─────────────────────────────────────── Steps
```

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| Reward too sparse | Add intermediate shaping rewards (distance progress, intermediate milestones) |
| Observation missing key information | Add more observations — the agent can't learn what it can't see |
| Learning rate too low | Try 2–5× higher learning rate |
| Entropy too low (collapsed early) | Increase `EntropyCoefficient` |
| MaxEpisodeSteps too short | Give the agent more time to reach the goal early on |
| Bad observation normalization | Use `obs.AddNormalized()` — raw coordinates often cause issues |

### Oscillating or diverging reward

```
Reward
1.0┤         ██
   │       ██   ██
0.5┤     ██        ██    ██
   │   ██             ██    ██
0.0┤ ██                        ▼▼▼
   └─────────────────────────────────── Steps
```

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| Learning rate too high | Reduce by 2–5× |
| `ClipEpsilon` too large (PPO) | Reduce to 0.1 |
| Reward function encourages instability | Review reward shaping |
| Observation NaN/Inf | Check for divide-by-zero in `CollectObservations` |

---

## Reading the Entropy Chart

Entropy measures how random the policy is. A policy with maximum entropy takes uniformly random actions; one with zero entropy always takes the same action.

### Healthy entropy curve

```
Entropy
1.5┤  ████
   │      ████
1.0┤          ████████
   │                  ████████████
0.5┤                              ████████
   └──────────────────────────────────── Steps
```

- Starts high (random policy = high entropy).
- Gradually decreases as the policy becomes more confident.
- Settles at a low but non-zero value.

### Entropy collapsed too early

```
Entropy
1.5┤  ██
0.0┤    ████████████████████████████
   └──────────────────────────────── Steps
```

The policy became deterministic before learning anything useful. The agent got stuck in a local optimum.

**Fix:** Increase `EntropyCoefficient`. Values of `0.01–0.05` are typical. If using SAC, check that `AutoTuneAlpha` is enabled and that `TargetEntropyFraction` isn't too low.

### Entropy never decreases

```
Entropy
1.5┤  ████████████████████████████
   └──────────────────────────────── Steps
```

The agent isn't converging at all. Reward signals are not reaching the policy.

**Fix:** Check reward values. Print them during training with `ShowTrainingOverlay = true`. Ensure `AddReward()` is actually being called with non-zero values.

---

## Reading the Loss Charts (PPO)

### Policy Loss

Policy loss measures how much the policy changed relative to the last update.

- Should oscillate at a moderate level and trend downward over time.
- **Spiking upward**: learning rate is too high, or `ClipEpsilon` is too large.
- **Continuously near zero**: no useful gradient signal — usually means very low advantage variance.

### Value Loss

Value loss measures how accurately the value network predicts returns.

- Should trend downward over time.
- **High value loss early** is normal — the value network is still learning the landscape.
- **Value loss not decreasing**: the network is too small, or `ValueLossCoefficient` is too low.
- **Value loss much higher than policy loss**: increase `ValueLossCoefficient` to rebalance.

### Clip Fraction

The clip fraction (not always charted, but logged) is the fraction of transitions where the PPO clip was active.

- Healthy range: `0.05 – 0.25`.
- > `0.5`: update is too large — reduce learning rate or reduce `ClipEpsilon`.
- < `0.01`: updates are too conservative — try increasing learning rate.

---

## Reading the Episode Length Chart

### Shorter episodes over time (goal-reaching task)

```
Length
500┤  ████████
   │          █████
250┤               ████████████████
   └──────────────────────────────── Steps
```

The agent is solving faster. This is healthy.

### Longer episodes over time (survival task)

Same shape but the reward went up — the agent is surviving longer. Also healthy.

### Sudden drop in episode length

The agent learned to end episodes quickly (possibly by triggering a terminal condition early). Check whether `EndEpisode()` can be triggered by undesirable behaviors.

---

## Common Tuning Workflows

### Workflow 1: Agent is not learning at all

1. Enable `ShowTrainingOverlay = true` (in `RLDistributedConfig`) and run the scene. Watch the debug overlay to verify observations and rewards are non-zero and sensible.
2. Check observation normalization: all values should be in roughly `[-1, 1]`. Raw world coordinates (e.g., `GlobalPosition.X = 1500.0`) will confuse the network.
3. Check reward scale: total episode reward should be in the range `[0, 1]` to `[-1, 10]`. Rewards of `0.001` per step with an episode length of 10,000 accumulate to `10` — fine. Rewards of `0.000001` are basically zero signal.
4. Reduce `MaxEpisodeSteps` to force more frequent episodes and faster feedback.
5. Add a dense shaping reward if goal is only achieved rarely. For example, reward negative distance to goal every step.

### Workflow 2: Agent is learning but plateauing too early

1. Check whether the plateau is near the theoretical max reward. If so, training is done.
2. If not: the agent may be stuck in a local optimum.
   - Increase `EntropyCoefficient` to escape the local optimum.
   - Add curriculum learning to guide the agent through difficulty levels.
   - Switch to SAC for better exploration in continuous spaces.
3. Check the network size. A `2×64` network may be too small for complex tasks — try `2×128` or `3×128`.

### Workflow 3: Policy is collapsing (reward drops sharply mid-training)

1. Lower the learning rate by 2–5×.
2. Reduce `EpochsPerUpdate` (PPO) — fewer passes over each rollout is more stable.
3. Reduce `UpdatesPerStep` (SAC) — train less aggressively per transition.
4. Check for reward explosions: a huge positive or negative reward for an edge case can destabilize the value network.
5. Enable `UseValueClipping` (PPO) for extra stability.

### Workflow 4: Training is too slow

1. Enable `AsyncGradientUpdates = true` in `RLRunConfig`.
2. Increase `SimulationSpeed` (e.g., `4.0`).
3. Add distributed workers: `WorkerCount = 4` with `WorkerSimulationSpeed = 4.0` multiplies data collection speed ~16×.
4. Reduce `RolloutLength` (PPO) for faster update cycles.
5. Increase `BatchSize` in `RLRunConfig` for more parallel environments.

---

## Reward Shaping Guidelines

Reward shaping is often the most impactful factor in whether an agent learns successfully.

### Principles

**1. Reward the goal, shape the path.**

Give a large reward for reaching the goal (e.g., `+1.0`). Add smaller intermediate rewards to guide exploration toward the goal. Never let intermediate rewards outweigh the goal reward.

```csharp
// Good: goal reward dominates
float distProgress = prevDist - currentDist;
AddReward(distProgress * 0.001f, "distance");   // small shaping
if (reached) { AddReward(1.0f, "goal"); EndEpisode(); }
```

**2. Always include a step penalty.**

A small step penalty (`-0.001` per step) encourages the agent to solve the task quickly. Without it, the agent may "drag out" episodes to avoid negative terminal rewards.

```csharp
AddReward(-0.001f);  // encourages speed
```

**3. Keep total episode reward in a reasonable range.**

Aim for typical episode rewards in `[-1, 5]`. Very large rewards (thousands) don't hurt correctness but can cause value network instability.

**4. Avoid reward cliffs.**

Reward cliffs are large negative rewards for actions that are difficult to predict (e.g., `-100` for touching a wall). The agent may learn to stand still to avoid them. Use smaller penalties (`-1.0` max for terminal failures) and positively reward success instead.

**5. Be consistent.**

Reward the same behavior the same way every episode. Don't change the reward function mid-training (except via curriculum).

### Debugging rewards

Use named rewards to track individual components:

```csharp
AddReward(distanceProgress, "distance_progress");
AddReward(alignmentBonus, "alignment");
AddReward(-0.001f, "step_penalty");
```

In the dashboard or spy overlay, you can see how much each component contributes to total reward. If one component dominates too much, the agent optimizes only for that.

---

## Observation Design Guidelines

**Rule 1: Include everything the agent needs to solve the task.**

If the agent needs to dodge bullets, it needs to see where the bullets are. If it needs to navigate, it needs to see its position and the goal position.

**Rule 2: Don't include irrelevant information.**

Irrelevant observations add noise and make learning harder. Don't include things like wall textures, unrelated NPC positions, or metadata the agent can't act on.

**Rule 3: Normalize everything.**

```csharp
// Bad: raw world coordinates
obs.Add(GlobalPosition.X);              // could be -5000 to 5000

// Good: normalized to [-1, 1]
obs.AddNormalized(GlobalPosition.X, -500f, 500f);
```

**Rule 4: Use relative coordinates.**

The goal's absolute position is less useful than "the goal is 200 units to my left." Relative inputs generalize better across episode resets.

```csharp
Vector2 toGoal = Goal.GlobalPosition - GlobalPosition;
obs.AddNormalized(toGoal.X, -maxDist, maxDist);
obs.AddNormalized(toGoal.Y, -maxDist, maxDist);
```

**Rule 5: Include velocity.**

Position alone tells the agent where it is; velocity tells it where it's going. For physics-based agents, velocity is often as important as position.

---

## Hyperparameter Tuning Cheatsheet

### PPO

| Symptom | Try |
|---------|-----|
| Not learning | ↑ LearningRate, ↑ EntropyCoefficient |
| Oscillating reward | ↓ LearningRate, ↓ ClipEpsilon |
| Very slow convergence | ↑ BatchSize, ↑ RolloutLength |
| Policy collapses mid-training | ↓ EpochsPerUpdate, ↓ LearningRate |
| Entropy drops too fast | ↑ EntropyCoefficient |
| Value loss not decreasing | ↑ ValueLossCoefficient, ↑ network size |

### SAC

| Symptom | Try |
|---------|-----|
| Not learning at all | Check WarmupSteps, ↑ ReplayBufferCapacity |
| Q-values diverging | ↓ LearningRate, ↓ UpdatesPerStep |
| Very random behavior persists | ↓ InitAlpha, check AutoTuneAlpha |
| Very deterministic too early | ↑ InitAlpha, ↑ TargetEntropyFraction |
| Training unstable | ↓ Tau (slower target updates), ↓ BatchSize |

---

## Curriculum Tuning

When using curriculum learning, the key question is: *how fast should the agent progress?*

**Too fast:** The agent faces difficulty it hasn't mastered, training destabilizes.

**Too slow:** The agent wastes time on trivial tasks.

### Tips

- Start with `DebugProgress = 0.0` to test the easiest version of the task. Verify the agent solves it reliably before enabling curriculum.
- Set `DebugProgress = 0.5` to test mid-difficulty. Does the agent sometimes succeed?
- Set `DebugProgress = 1.0` to test the hardest version. It's fine if the agent can't solve it from scratch — that's why we use curriculum.
- In `SuccessRate` mode: set `PromoteThreshold = 0.7` and `DemoteThreshold = 0.3` as a starting point. Narrow the gap (`0.6 / 0.4`) for smoother transitions.
- In `StepBased` mode: set `MaxCurriculumSteps` to roughly 30–50% of your total training budget. This ensures the agent has time to adapt at each difficulty level.

---

## Self-Play Tuning

### Monitoring Elo

A healthy self-play Elo curve rises steadily:

```
   Elo
1500┤                    ████████
    │              ████████
1200┤  ████████████
    └──────────────────────────────── Steps
```

If Elo is flat, the learner isn't improving against its current opponent pool. Try increasing `HistoricalOpponentRate` so it faces more diverse opponents.

### PFSP Alpha

Higher `PfspAlpha` (e.g., `6.0`) makes the sampling more aggressive — the hardest opponents are selected almost exclusively. Start with `4.0` and only increase if the agent is too easily beating all historical opponents.

### Pool Size

`MaxPoolSize = 20` is usually enough. Very large pools are less important than ensuring checkpoints are taken frequently (`FrozenCheckpointInterval = 5`).
