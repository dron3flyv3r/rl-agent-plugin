# How to Tune Hyperparameters Effectively

<!-- markdownlint-disable MD029 MD032 -->

This guide gives a systematic method to tune key training parameters without guesswork.

---

## Tuning Principles

Use a disciplined process:

- keep one stable baseline
- change one variable at a time
- evaluate over consistent training windows
- use reward + entropy + loss together

Random multi-parameter edits make results hard to trust.

---

## Step 1: Establish A Baseline Run

Start from known working values.

For PPO, track these first:

- `LearningRate`
- `RolloutLength`
- `EntropyCoefficient`

Record run id, config values, and baseline metrics before tuning.

---

## Step 2: Tune LearningRate First

Why first:

- it strongly controls stability and convergence speed

Typical actions:

- flat/no learning: try increasing moderately
- oscillation/divergence: decrease by 2-5x

Keep all other settings fixed while testing learning-rate variants.

---

## Step 3: Tune RolloutLength Next

`RolloutLength` tradeoff:

- larger: more stable updates, slower feedback cycles
- smaller: faster updates, potentially noisier gradients

After learning rate is acceptable, adjust rollout length to improve stability/iteration speed balance.

---

## Step 4: Tune EntropyCoefficient For Exploration

Use entropy chart as guide:

- entropy collapses too early: increase `EntropyCoefficient`
- entropy remains very high with no reward gain: reduce it carefully

Aim for a gradual entropy decline, not an immediate collapse.

---

## Step 5: Use A Standard Evaluation Window

For each variant compare the same window, for example:

- first N updates
- same wall-clock duration
- same scene seed policy where practical

Comparing different horizons often leads to wrong conclusions.

---

## Suggested PPO Tuning Order

1. `LearningRate`
2. `RolloutLength`
3. `EntropyCoefficient`
4. then secondary knobs (`ClipEpsilon`, `EpochsPerUpdate`, `MiniBatchSize`)

Do not jump to secondary knobs before core signal quality is good.

---

## Common Mistakes

1. Tuning 3-5 parameters at once.
2. Using short noisy windows to declare winners.
3. Ignoring entropy/loss while watching reward only.
4. Changing rewards and hyperparameters in the same run.

---

## Minimal Checklist

- baseline run documented
- one variable changed per run
- learning rate tuned first
- rollout length and entropy tuned with chart evidence
- all variants compared on same evaluation window
