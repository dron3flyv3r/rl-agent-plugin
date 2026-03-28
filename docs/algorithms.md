# Algorithms: PPO and SAC

The plugin ships two algorithms: **PPO** (Proximal Policy Optimization) and **SAC** (Soft Actor-Critic). Choosing between them is the first decision you make when setting up a new training run.


---
## Available algorithms:

Algorithm | Action Space | On/Off Policy | Sample Efficiency | Exploration
---|---|---|---|---
PPO | Discrete + Continuous | On-policy | Moderate | Manual entropy bonus
SAC | Continuous only | Off-policy | High | Auto-tuned alpha

## PPO vs SAC: Which to choose?
Algorithm | When to use | When not to use
---|---|---
PPO | Discrete actions, simple continuous control, dense rewards, fast iteration | Expensive simulations, complex continuous control
SAC | Complex continuous control, sample efficiency, built-in exploration | Discrete actions, short episodes, memory constraints

---

## PPO — Proximal Policy Optimization

### What it is

PPO is an **on-policy** algorithm. The agent interacts with the environment, collects a batch of experience (a "rollout"), updates the network, and then throws the experience away. This cycle repeats.

PPO is a policy gradient method: it directly optimizes the policy to maximize expected reward, with a clipping mechanism that prevents the update from changing the policy too drastically in one step.

### When to use PPO

- **Discrete action spaces** (moving left/right, choosing from a menu, picking a direction).
- **Continuous action spaces** that are relatively low-dimensional (locomotion with a few joints).
- **Tasks with dense rewards** where the agent gets frequent feedback.
- Tasks where you want **predictable, stable training** and fast iteration.
- **On-policy requirements**: tasks where the reward function changes during training (curriculum), or where the agent's behavior strongly affects what data it sees.

### When not to use PPO

- Tasks where each simulation step is very expensive (sample efficiency matters). SAC is more data-efficient.
- Complex continuous control with many joints (PPO can work but SAC usually learns faster).

---

### How PPO Works (intuition)

PPO collects `RolloutLength` transitions, then runs `EpochsPerUpdate` passes over mini-batches, optimizing three things:

1. **Policy loss** — push the policy toward actions that led to high advantage, while clipping the policy ratio so it doesn't change too fast.
2. **Value loss** — train the value network to accurately predict how much reward the agent will collect from a given state.
3. **Entropy bonus** — add a small reward for keeping the policy distribution spread out (exploration).

**Advantage** measures "was this action better or worse than average?" It's computed with GAE (Generalized Advantage Estimation), which blends multi-step returns for lower variance.

```
Advantage(t) = Σ (γλ)^k × δ(t+k)
  where δ(t) = r(t) + γ·V(s_{t+1}) - V(s_t)    (TD error)
```

**Clipped policy objective:**

```
L_CLIP = E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]
  where r = π_new(a|s) / π_old(a|s)
```

The clip prevents the new policy from moving too far from the old one, which stabilizes training.

### Continuous Actions in PPO

For continuous action spaces, the policy outputs a **Gaussian distribution**: a mean and a log-standard-deviation per action dimension. The agent samples from this distribution during training (for exploration) and takes the mean during inference (for deterministic behavior).

Actions are squashed through `tanh` so they're always in `[-1, 1]`, then rescaled to your declared range.

---

### PPO Configuration Quick Reference

See [configuration.md](configuration.md#rlppoconfig) for full parameter docs.

Key parameters:

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `RolloutLength` | 128–2048 | More data per update = more stable, but slower |
| `EpochsPerUpdate` | 3–10 | More epochs = use data more, risk instability |
| `LearningRate` | 1e-4 – 3e-3 | Step size for gradient updates |
| `ClipEpsilon` | 0.1–0.3 | How much policy is allowed to change per update |
| `EntropyCoefficient` | 0.001–0.05 | Exploration bonus strength |
| `Gamma` | 0.95–0.999 | How much future rewards are discounted |
| `GaeLambda` | 0.9–0.99 | GAE advantage blend (higher = more bias, less variance) |

---

## SAC — Soft Actor-Critic

### What it is

SAC is an **off-policy** algorithm. It stores all past experience in a **replay buffer** and samples random mini-batches for training. This means SAC is much more **sample-efficient** — it reuses data many times.

SAC optimizes a modified objective that includes an **entropy bonus**: it maximizes reward *while also* maximizing the entropy (randomness) of the policy. This built-in exploration mechanism is one of SAC's biggest strengths.

### When to use SAC

- **Continuous action spaces only** (SAC does not support discrete actions in this plugin).
- Tasks requiring **high sample efficiency**: expensive simulations, slow physics, or limited data.
- **Complex locomotion or manipulation**: many joints, nuanced control.
- When you want **built-in exploration** without tuning an explicit entropy coefficient (SAC auto-tunes alpha).

### When not to use SAC

- Discrete action spaces — use PPO instead.
- Very short episodes or sparse rewards — SAC needs enough transitions to fill the replay buffer before training begins (`WarmupSteps`).
- When memory is limited — the replay buffer stores `ReplayBufferCapacity` full transitions in RAM.

---

### How SAC Works (intuition)

SAC maintains:
- An **actor** (policy network): produces a Gaussian distribution over actions.
- Two **critic networks** (Q-functions): estimate the expected return of taking action `a` in state `s`.
- **Target critic networks**: slow-moving copies of the critics, updated by polyak averaging (τ = 0.005). This stabilizes training.
- An **entropy temperature** (alpha, α): automatically tuned to keep policy entropy near a target value.

**Training loop** (every `UpdateEverySteps` environment steps, run `UpdatesPerStep` gradient updates):

1. Sample a mini-batch from the replay buffer.
2. **Critic update**: minimize TD error against target critics.
3. **Actor update**: maximize `Q(s, a) - α·log π(a|s)` (reward + entropy).
4. **Alpha update**: adjust α so that `log π` stays near `TargetEntropy`.
5. **Target update**: polyak-average the target critics toward the live critics.

**Double critics** (two Q networks, take the minimum) prevent overestimation of Q values, which is a major source of instability in Q-learning methods.

### Updates-to-Data Ratio (UTD)

The UTD ratio controls how many gradient updates run per new transition collected. With distributed workers, the plugin auto-scales UTD:

```
UTD = min(DataSources × BatchSize, 8)
```

A higher UTD makes SAC more sample-efficient but increases compute cost. Setting `UpdatesPerStep = 1` is a conservative starting point.

---

### SAC Configuration Quick Reference

See [configuration.md](configuration.md#rlsacconfig) for full parameter docs.

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `LearningRate` | 3e-4 – 1e-3 | Step size for all networks |
| `ReplayBufferCapacity` | 50k–1M | Memory vs. data diversity trade-off |
| `WarmupSteps` | 500–5000 | Steps before training begins (fill buffer first) |
| `Tau` | 0.001–0.01 | Target network update rate (lower = more stable) |
| `BatchSize` | 128–512 | Mini-batch size per gradient update |
| `InitAlpha` | 0.1–0.5 | Starting entropy temperature |
| `AutoTuneAlpha` | true | Recommended; lets SAC adapt exploration automatically |
| `UpdatesPerStep` | 1–4 | Gradient updates per new transition |

---

## PPO vs SAC at a Glance

| | PPO | SAC |
|--|-----|-----|
| **On/off policy** | On-policy | Off-policy |
| **Action spaces** | Discrete + Continuous | Continuous only |
| **Sample efficiency** | Moderate | High |
| **Memory usage** | Low (rollout buffer) | High (replay buffer) |
| **Exploration** | Entropy bonus (manual) | Auto-tuned alpha |
| **Convergence** | Often faster to first good policy | Often better final performance |
| **Hyperparameter sensitivity** | Lower | Moderate |
| **Self-play support** | Yes | Yes |
| **Curriculum support** | Yes | Yes |
| **Distributed training** | Yes | Yes |

**Rule of thumb:** Start with PPO if you have discrete actions or want fast setup. Switch to SAC if you have a continuous control task and care about sample efficiency or final performance.

---

## Training Stability Tips

### PPO

- If policy loss diverges early: lower `LearningRate` or lower `ClipEpsilon`.
- If the agent stops exploring: increase `EntropyCoefficient`.
- If updates are slow: reduce `RolloutLength` or increase `BatchSize`.
- If the value loss is very high relative to policy loss: increase `ValueLossCoefficient`.

### SAC

- If Q values diverge: lower `LearningRate` or lower `UpdatesPerStep`.
- If the agent acts randomly for too long: reduce `WarmupSteps`.
- If alpha (entropy temperature) collapses to near-zero: check `TargetEntropyFraction`; lowering it reduces the entropy target.
- NaN in training: usually caused by unstable alpha updates or invalid observations/rewards. Confirm `AutoTuneAlpha` settings and check metrics for divergence before increasing update intensity.

### Both

- Always normalize observations to approximately `[-1, 1]` using `obs.AddNormalized(value, min, max)`.
- Use `MaxGradientNorm` (gradient clipping) — the default of 0.5 is usually fine.
- Watch the **entropy chart** in the dashboard. A steady entropy decline during training is healthy. A sudden drop to near zero means the policy collapsed.
