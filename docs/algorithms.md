# Algorithms

The plugin ships five built-in algorithms. Choosing the right one is the first decision you make when setting up a new training run.

---

## Capability matrix

Algorithm | Config resource | Action space | On/Off policy | Multi-agent | Recurrent LSTM/GRU | Sample efficiency | Exploration
---|---|---|---|---|---|---|---
**PPO** | `RLPPOConfig` | Discrete **or** Continuous | On-policy | ✓ (policy sharing) | ✓ | Moderate | Entropy bonus
**SAC** | `RLSACConfig` | Continuous only | Off-policy | ✓ (policy sharing) | No | High | Auto-tuned entropy (α)
**DQN / DDQN** | `RLDQNConfig` | **Discrete only** | Off-policy | ✓ (policy sharing) | No | Moderate-High | ε-greedy (decaying)
**A2C** | `RLA2CConfig` | Discrete **or** Continuous | On-policy | ✓ (policy sharing) | ✓ | Low-Moderate | Entropy bonus
**MCTS** | `RLMCTSConfig` | **Discrete only** | Planning (no replay) | ✓ (independent) | N/A | N/A (no learning) | UCT exploration term

> **Multi-agent note:** All algorithms support the standard policy-sharing model (many agents, one network per group). None implement *centralized* multi-agent coordination (CTDE). If you need cross-agent communication (cooperative teams with joint rewards), consider adding a custom trainer via `TrainerFactory.Register()` — QMIX and MADDPG are natural choices.

> **Recurrent note:** Recurrent state is per agent, not per group. In a shared-policy setup with 10 agents, all 10 agents share weights but each agent keeps its own hidden state.

---

## Quick selection guide

| Situation | Recommended |
|-----------|-------------|
| Discrete actions (menus, movement directions, jump/attack) | **DQN** (simpler) or **PPO** |
| Complex continuous control (locomotion, robotic arms) | **SAC** |
| Simple continuous control or mixed discrete+continuous | **PPO** |
| Fastest-to-converge baseline for a new environment | **A2C** |
| Limited memory / no replay buffer | **PPO** or **A2C** |
| Maximum sample efficiency for continuous control | **SAC** |
| Planning / model-based updates on top of DQN | **DQN** with `DynaModelUpdatesPerStep > 0` |
| Pure tree-search planning (no neural net, simulable environment) | **MCTS** |

---

## PPO vs A2C

Both are on-policy actor-critic algorithms using the same network architecture. The difference is in the update rule:

- **PPO** clips the policy ratio to prevent large policy updates — more stable but slightly more overhead
- **A2C** applies vanilla policy gradient without clipping — simpler, faster per-step, but potentially less stable

Start with A2C if you want a quick baseline. Switch to PPO if training is unstable or you want to tune the rollout/epoch count.

Both PPO and A2C are also the plugin's recurrent-capable built-in trainers.

## DQN vs SAC

Both are off-policy with replay buffers. The key difference is action space:

- **DQN** supports **discrete actions only**, uses a Q-value table per action, and ε-greedy exploration
- **SAC** supports **continuous actions only**, uses a stochastic actor and auto-tuned entropy

If your environment has discrete actions and you don't need continuous control, prefer DQN — it is simpler to tune and faster per update. Enable `UseDoubleDqn` (default: true) to reduce overestimation.

### Dyna-Q (planning extension for DQN)

Set `DynaModelUpdatesPerStep > 0` in `RLDQNConfig` to enable Dyna-Q. This trains a small MLP world model alongside the Q-network and uses it to generate *imagined* transitions for additional Q-learning updates — improving sample efficiency without more environment steps. Good values are 5–50 for small/medium environments.

---

## Recurrent Policies (LSTM / GRU)

The plugin supports recurrent policy trunks through `RLLstmLayerDef` and `RLGruLayerDef`.

What is supported:

- PPO with one recurrent trunk layer
- A2C with one recurrent trunk layer
- shared-policy multi-agent rollouts, where each agent has its own hidden state
- recurrent inference from checkpoints and frozen/self-play policies

What is not supported:

- more than one recurrent layer in the same trunk
- recurrent DQN
- recurrent SAC

Why the DQN/SAC limit exists:

- PPO and A2C train directly from fresh on-policy sequences, so truncated BPTT is straightforward.
- DQN and SAC are off-policy and would need sequence replay, stored hidden states, burn-in, and target-network handling for recurrent updates. That is a different training design, not just a missing forward call.

How to think about shared-policy multi-agent recurrence:

- one policy group = one shared set of weights
- each agent in that group = its own `h` / `c` state
- episode reset clears only that agent's recurrent state

Practical recommendation:

- start with one recurrent layer and a hidden size of `32` or `64`
- use PPO or A2C
- keep `BpttLength` modest at first, such as `16` or `32`

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
| `BpttLength` | 8–64 | Truncated sequence length for recurrent PPO/A2C |
| `EpochsPerUpdate` | 3–10 | More epochs = use data more, risk instability |
| `LearningRate` | 1e-4 – 3e-3 | Step size for gradient updates |
| `ClipEpsilon` | 0.1–0.3 | How much policy is allowed to change per update |
| `EntropyCoefficient` | 0.001–0.05 | Exploration bonus strength |
| `Gamma` | 0.95–0.999 | How much future rewards are discounted |
| `GaeLambda` | 0.9–0.99 | GAE advantage blend (higher = more bias, less variance) |

BpttLength applies to recurrent PPO and recurrent A2C, with the same 8–64 practical range.

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

---

## MCTS — Monte Carlo Tree Search

### What it is

MCTS is a **pure planning** algorithm — it does not learn from experience and has no neural network. At each decision point it runs `NumSimulations` simulated rollouts through your environment model to build a search tree, then selects the action with the most visits.

MCTS is best when:
- The environment is **simulable** (you can implement `IEnvironmentModel` fast enough for hundreds of calls per step).
- Actions are **discrete**.
- You want **strong decisions without training time** — MCTS improves with more compute per step, not more experience.

### Setup

1. Implement `IEnvironmentModel` on your scene node (or a helper):

```csharp
public (float[] nextObservation, float reward, bool done) SimulateStep(float[] obs, int action)
{
    // Decode state from obs — no scene changes!
    var x    = obs[0] * MaxRange;
    var goal = obs[1] * MaxRange;

    float dx  = action == 0 ? -Speed : Speed;
    var newX  = Mathf.Clamp(x + dx, -MaxRange, MaxRange);

    var nextObs = new float[] { newX / MaxRange, goal / MaxRange };
    var reward  = Mathf.Abs(newX - goal) < 0.1f ? 1f : -0.01f;
    return (nextObs, reward, reward > 0f);
}
```

2. Register the model **before training starts** (e.g., `_Ready`):

```csharp
MctsTrainer.SetEnvironmentModel(this);
```

3. Assign an `RLMCTSConfig` resource to your policy group's Algorithm field.

### How MCTS works (UCT)

Each simulation follows four phases:

1. **Selection** — walk the existing tree using UCT: `score = Q(s,a) + c × sqrt(ln N(s) / (N(s,a)+1))`
2. **Expansion** — add one new child node for an unvisited action
3. **Evaluation** — estimate the leaf value via a random rollout to depth `RolloutDepth`
4. **Backpropagation** — update visit counts and Q values along the path

After `NumSimulations` simulations, return the action with the **highest visit count** (most robust, not most optimistic).

### MCTS Configuration Quick Reference

See [configuration.md](configuration.md#rlmctsconfig) for full parameter docs.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `NumSimulations` | 50 | Simulations per action decision — higher = stronger but slower |
| `MaxSearchDepth` | 20 | Max tree depth during selection phase |
| `RolloutDepth` | 10 | Random rollout steps for leaf evaluation |
| `ExplorationConstant` | 1.414 | UCT exploration weight (√2 is the theoretical default) |
| `Gamma` | 0.99 | Discount factor applied during rollout accumulation |

### Performance tips

- Start with `NumSimulations = 50` and increase if you have CPU budget.
- Keep `SimulateStep` **allocation-free** — it is called `NumSimulations` times per action.
- Avoid Godot API calls inside `SimulateStep`; work entirely with floats decoded from the observation.
- If the environment is stochastic, MCTS still works but needs more simulations to average out noise.

### When not to use MCTS

- Continuous action spaces (not supported).
- Environments where simulating a step is expensive (physics, RPC calls, etc.) — each action decision runs `NumSimulations` simulations.
- When you want the agent to **improve over time** — MCTS has no learning. Use DQN with Dyna-Q for a planning+learning hybrid.
