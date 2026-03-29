# How to Choose Between PPO and SAC

<!-- markdownlint-disable MD029 MD032 -->

This guide gives a practical decision flow for selecting `PPO` or `SAC` in RL Agent Plugin.

---

## Quick Decision Rule

Use `PPO` when:

- your action space is discrete
- you want a robust default starting point
- you prefer simpler training behavior

Use `SAC` when:

- your action space is continuous
- sample efficiency matters a lot
- you are training harder control tasks

If unsure, start with `PPO` unless your environment clearly needs continuous control quality and sample efficiency.

---

## Step 1: Check Action Space Type

1. If your policy outputs **discrete actions**, choose `PPO`.
2. If your policy outputs **continuous actions**, choose `SAC` (or `PPO` continuous if you want on-policy behavior).

In this plugin, `SAC` is designed for continuous control workflows.

---

## Step 2: Match The Training Objective

Pick `PPO` when you want:

- stable first runs
- easier baseline comparison
- straightforward rollout/update loop

Pick `SAC` when you want:

- better data reuse through replay buffer
- faster improvement per environment sample
- stronger performance on complex locomotion/control

---

## Step 3: Start With Minimal Baselines

Create two baseline configs and compare quickly:

- `PPO` baseline with conservative entropy and rollout length
- `SAC` baseline with safe warmup and replay capacity

Keep all other environment settings identical so algorithm differences are measurable.

---

## Step 4: Evaluate With The Same Metrics

Compare both runs using the same windows in `RLDash`:

- mean episode reward
- episode length trend
- stability of policy/value losses
- wall-clock time to reach target score

Choose the algorithm that reaches your target performance faster and more reliably.

---

## Recommended First Defaults

### PPO first defaults

- moderate `RolloutLength`
- moderate `LearningRate`
- small `EntropyCoefficient`
- standard `ClipEpsilon`

### SAC first defaults

- replay buffer large enough for diverse transitions
- non-zero `WarmupSteps`
- batch size that your hardware can sustain
- conservative `UpdatesPerStep` initially

Use these only as starting points; tune after you confirm learning signal quality.

---

## Common Mistakes

1. Choosing `SAC` for purely discrete tasks.
2. Comparing algorithms with different reward scales or termination rules.
3. Declaring a winner too early before curves stabilize.
4. Tuning many knobs at once instead of one variable per experiment.

---

## Minimal Checklist

- action space identified
- baseline configs for both algorithms created
- same environment and reward logic used for comparison
- metrics compared over consistent training windows
- selected algorithm documented for the scene
