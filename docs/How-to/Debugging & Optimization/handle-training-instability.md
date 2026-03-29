# How to Handle Training Instability

<!-- markdownlint-disable MD029 MD032 -->

This guide provides a practical recovery playbook for unstable training runs.

---

## What Instability Looks Like

Common symptoms:

- reward spikes then crashes
- policy/value losses diverge or oscillate heavily
- entropy collapses too early
- behavior regresses after initial progress

Treat instability as a system issue (signal + optimization), not a single-parameter issue.

---

## Step 1: Confirm Signal Quality First

Before changing hyperparameters, verify:

- observations are normalized and meaningful
- reward magnitudes are reasonable
- `EndEpisode` and reset logic are correct

Bad signal quality can mimic optimizer instability.

---

## Step 2: Apply Conservative Stabilization Changes

For PPO instability, try in this order:

1. lower `LearningRate`
2. lower `ClipEpsilon`
3. reduce `EpochsPerUpdate`
4. increase `RolloutLength` (if updates are too noisy)

Change one variable per run.

---

## Step 3: Manage Exploration Health

Watch entropy curve:

- collapse too early -> increase exploration pressure (`EntropyCoefficient`)
- persistently high entropy with no gains -> decrease exploration pressure gradually

Balance exploration with reward exploitation, do not force either extreme.

---

## Step 4: Reduce Throughput Pressure If Needed

Very aggressive throughput can destabilize updates.

Try:

- lower simulation speed
- lower batch pressure
- reduce update intensity (especially SAC `UpdatesPerStep`)

Stability usually beats raw steps/sec when runs keep collapsing.

---

## Step 5: Use Recovery Protocol

When a run becomes unstable:

1. stop and preserve the last known-good checkpoint/model
2. branch a recovery run with conservative settings
3. compare recovery run against prior stable window
4. only then reintroduce speed/aggressive settings gradually

---

## Step 6: Track Root Cause Notes

Log each instability event with:

- run id
- config delta
- symptom pattern
- recovery result

This creates a reusable internal playbook for future environments.

---

## Common Mistakes

1. Treating reward crash as purely hyperparameter issue without signal checks.
2. Applying multiple stabilization changes at once.
3. Prioritizing throughput over stability.
4. Continuing unstable runs too long without checkpointing/rollback strategy.

---

## Minimal Checklist

- signal quality validated first
- conservative stabilization sequence applied
- entropy behavior monitored and adjusted
- throughput pressure reduced when needed
- recovery runs tracked with clear run notes
