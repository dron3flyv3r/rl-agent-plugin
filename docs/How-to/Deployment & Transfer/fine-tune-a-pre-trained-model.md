# How to Fine-Tune a Pre-Trained Model

<!-- markdownlint-disable MD029 MD032 -->

This guide covers practical transfer-learning workflows for RL Agent Plugin and clarifies current limitations.

---

## Current Support Status

The editor training flow is designed around fresh training runs and checkpoint generation.

Direct one-click warm-start training from a `.rlmodel` in `Start Training` is not exposed as a standard Inspector option in the current workflow.

What is fully supported today:

- export trained models as `.rlmodel`
- run those models in inference mode
- continue improving behavior by launching new training runs with transfer-friendly environment design

---

## Transfer Strategy That Works Well

Use this staged approach:

1. start from a working source-task model
2. run it in inference on the target task
3. shape target task with curriculum/rewards for gradual adaptation
4. train a new target policy with compatible observation/action schema

This gives practical transfer even without direct weight warm-start in the default launch path.

---

## Step 1: Keep Action/Observation Compatibility

For transfer-friendly adaptation, keep these stable between source and target tasks:

- action definitions and dimensions
- core observation semantics/order
- stream names for multimodal observations

Large schema changes reduce transfer value and usually require retraining from scratch.

---

## Step 2: Use Inference Baseline As Reference

Deploy the source `.rlmodel` in target scene first:

- set `InferenceModelPath`
- run in inference mode
- record baseline reward/success metrics

This gives a concrete baseline before adaptation runs.

---

## Step 3: Build A Gentle Adaptation Curriculum

Reduce target-task difficulty initially:

- easier spawn geometry
- slower opponents/obstacles
- denser shaping rewards

Then ramp difficulty with `RLCurriculumConfig` as performance improves.

---

## Step 4: Train New Target Runs Systematically

Run targeted experiments (one variable at a time):

- reward coefficients
- curriculum pacing
- exploration settings

Track all runs in `RLDash` and compare against inference baseline.

---

## Step 5: Promote And Export Best Adapted Policy

When adapted training beats baseline reliably:

1. export the new run/checkpoint to `.rlmodel`
2. update deployment path/version
3. regression-test behavior in production scenes

---

## Advanced (Code-Level) Warm-Start Option

If you need true weight initialization from existing checkpoints before updates:

- implement a custom training launch/trainer initialization path that loads checkpoint weights before the first update

This is an advanced customization path and requires code changes outside the default editor one-click flow.

---

## Common Mistakes

1. Expecting drop-in transfer after major observation/action redesign.
2. Skipping baseline inference measurement on the new task.
3. Changing many adaptation variables in one run.
4. Treating transfer as successful without production-scene validation.

---

## Minimal Checklist

- source model baseline measured on target scene
- action/observation compatibility preserved where possible
- adaptation curriculum and rewards staged
- controlled adaptation runs executed and compared
- best adapted policy exported and validated
