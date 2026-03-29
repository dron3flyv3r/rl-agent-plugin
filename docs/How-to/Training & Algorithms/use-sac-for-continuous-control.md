# How to Use SAC for Continuous Control

<!-- markdownlint-disable MD029 MD032 -->

This guide focuses on practical `SAC` setup in RL Agent Plugin for continuous action tasks.

---

## When SAC Is A Good Fit

Use `SAC` when your task has:

- continuous actuators (steering, torque, motor targets)
- complex physics interactions
- expensive environment steps where sample efficiency matters

Typical examples: crawling, balancing, smooth navigation, articulated control.

---

## Step 1: Select SAC In Training Config

In `RLTrainingConfig`:

1. Set `Algorithm` to `RLSACConfig`.
2. Confirm your agent outputs continuous actions.
3. Keep reward scaling reasonable (avoid extreme magnitudes).

---

## Step 2: Configure Core SAC Parameters

Start with conservative settings:

- `ReplayBufferCapacity`: large enough for transition diversity
- `WarmupSteps`: collect random/early experience before heavy updates
- `BatchSize`: fit your GPU/CPU memory budget
- `UpdatesPerStep`: start low, increase carefully

Conservative first runs are easier to debug than aggressive throughput settings.

---

## Step 3: Verify Environment And Action Scaling

Before long training runs:

1. Confirm action ranges map correctly to your movement/actuators.
2. Ensure no actuator instantly saturates or explodes physics.
3. Check that observations are normalized or bounded.

Bad action scaling is one of the most common SAC failure causes.

---

## Step 4: Monitor SAC-Specific Health Signals

In `RLDash`, watch for:

- reward trend improving over time
- critic loss not diverging continuously
- policy behavior becoming smoother and less random
- entropy behavior consistent with exploration phase

If critic/policy metrics are highly unstable, reduce learning pressure first.

---

## Stabilization Playbook

If SAC is unstable:

1. Lower learning rate.
2. Reduce `UpdatesPerStep`.
3. Increase `WarmupSteps`.
4. Re-check reward scale and clipping.
5. Re-check action range mapping in your controller.

Change one variable per run so you can attribute outcomes.

---

## Common Mistakes

1. Using SAC with badly scaled actions.
2. Running with tiny replay buffers on diverse tasks.
3. Setting update intensity too high too early.
4. Mixing major reward redesign and hyperparameter changes in one run.

---

## Minimal Checklist

- `RLSACConfig` selected in `RLTrainingConfig`
- continuous actions verified end-to-end
- replay/warmup/batch/update parameters set conservatively
- reward and action scales validated
- metrics monitored for stability before long runs
