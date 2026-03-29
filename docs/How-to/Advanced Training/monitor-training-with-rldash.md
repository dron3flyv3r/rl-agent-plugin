# How to Monitor Training with RLDash

<!-- markdownlint-disable MD029 MD032 -->

This guide explains how to use `RLDash` as your main live diagnosis tool during training.

---

## What RLDash Shows

`RLDash` polls run metrics files every ~2 seconds and charts:

- episode reward
- episode length
- policy loss
- value loss
- entropy
- curriculum progress (when active)
- Elo (self-play)

Your first diagnosis signal is always the reward trend.

---

## Step 1: Start Training And Open RLDash

1. Launch training from toolbar or RL Setup.
2. Open the `RLDash` editor tab.
3. Select the active run (and policy group, if multiple).

Metrics come from `RL-Agent-Training/runs/<RunId>/metrics__*.jsonl` (per-group) and related run files.

---

## Step 2: Read Curves In The Right Priority

Use this order:

1. episode reward (learning direction)
2. entropy (exploration health)
3. policy/value losses (optimization stability)
4. episode length (task-dependent interpretation)

Do not overreact to short-window noise; look for trend over enough updates.

---

## Step 3: Use Curve Patterns To Decide Next Action

Typical patterns:

- reward flat: check reward shaping + observations first
- entropy collapses early: increase exploration pressure
- losses spike/diverge: reduce learning aggressiveness
- reward drops after initial gains: check policy collapse or reward cliffs

RLDash is most useful when paired with one-change-per-run tuning.

---

## Step 4: Correlate Dashboard With Runtime Overlays

When debugging signal quality:

- enable `ShowTrainingOverlay` in distributed config
- optionally use spy/camera overlays where relevant

Use overlays to verify observations and reward components match what charts suggest.

---

## Step 5: Compare Runs Systematically

For trustworthy conclusions:

1. keep one baseline run unchanged
2. change one variable (hyperparameter, reward term, sensor)
3. compare same training window lengths
4. keep notes of run id, change, and outcome

This avoids tuning drift and false positives.

---

## Common Mistakes

1. Optimizing from a very short metric window.
2. Tuning multiple variables between runs.
3. Ignoring entropy and loss while only watching reward.
4. Comparing runs with different reward scales as if equivalent.

---

## Minimal Checklist

- run selected in RLDash
- reward, entropy, and losses reviewed together
- conclusions based on trend, not single spikes
- runtime overlays used when chart signals are ambiguous
- changes tracked one variable per experiment
