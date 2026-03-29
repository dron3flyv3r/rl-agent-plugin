# How to Run Multiple Experiments in Parallel

<!-- markdownlint-disable MD029 MD032 -->

This guide shows a practical workflow for parallel experiment runs with different hyperparameters.

---

## Parallel Experiment Strategy

Goal: increase learning throughput while preserving experimental clarity.

Use parallel runs to test alternatives such as:

- PPO vs SAC
- reward coefficient variants
- network size variants
- exploration coefficient variants

Keep each run isolated and clearly labeled.

---

## Step 1: Define A Small Experiment Matrix

Start with 3-6 variants, not dozens.

Example matrix:

- Run A: PPO baseline
- Run B: PPO lower learning rate
- Run C: PPO higher entropy coefficient
- Run D: SAC baseline (continuous tasks)

Large parameter sweeps are harder to diagnose than focused comparisons.

---

## Step 2: Tag Runs With Clear Prefixes

Set `RLRunConfig.RunPrefix` for each variant so run folders are self-describing.

Example:

```text
baseline_ppo
ppo_lr_1e-4
ppo_entropy_0p02
sac_baseline
```

This makes RLDash selection and result tracking much faster.

---

## Step 3: Keep Non-Test Variables Fixed

For valid comparisons, keep these identical unless they are your test variable:

- scene and task version
- reward function
- observation design
- max episode steps
- evaluation window length

Only change what you are trying to measure.

---

## Step 4: Allocate Compute Per Run Intentionally

Parallel runs compete for resources.

Control per-run load with:

- `WorkerCount`
- `WorkerSimulationSpeed`
- `BatchSize`
- network size and image resolution

Prefer fewer high-quality runs over many starved runs.

---

## Step 5: Evaluate With Shared Success Criteria

Define success criteria before launching:

- target reward threshold
- max wall-clock time
- stability constraints (no major collapse)

Then compare all runs against the same criteria.

---

## Step 6: Promote Winners, Retire Losers

After initial windows:

1. stop clearly underperforming variants
2. continue top 1-2 runs longer
3. branch new variants from the current best baseline

This keeps iteration loops tight and compute-efficient.

---

## Common Mistakes

1. Launching many runs without naming conventions.
2. Changing reward and hyperparameters simultaneously.
3. Comparing runs over different training durations.
4. Keeping weak runs active too long.

---

## Minimal Checklist

- small experiment matrix defined
- run prefixes configured for each variant
- only one primary variable changed per run
- compute budget per run planned
- shared comparison criteria defined before launch
