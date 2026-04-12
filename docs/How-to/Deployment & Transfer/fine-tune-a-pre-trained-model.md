# How to Fine-Tune a Pre-Trained Model

<!-- markdownlint-disable MD029 MD032 -->

This guide covers practical fine-tuning and transfer-learning workflows for RL Agent Plugin.

---

## Current Support Status

The current editor training flow can warm-start from an existing checkpoint before RL updates begin.

What is supported today:

- warm-starting a training run from an `.rlcheckpoint` or checkpoint `.json`
- setting that path directly in `RLRunConfig`, or indirectly through the RL Imitation dock warm-start bridge
- continuing PPO / SAC / A2C / DQN / MCTS training from loaded network weights
- exporting trained models as `.rlmodel` for deployment
- running `.rlmodel` files in inference mode

What is not supported as a full resume:

- optimizer state restore
- replay-buffer restore
- rollout-buffer restore

What `.rlmodel` is for:

- deployment and inference

What `.rlmodel` is not for:

- direct training resume in the default warm-start path

For checkpoint-based imitation workflows, see [RL Imitation](../../imitation.md).

---

## Recommended Fine-Tune Strategy

Use this staged approach:

1. start from a working source checkpoint
2. keep action/observation schema compatible on the target task
3. warm-start a new RL run from that checkpoint
4. shape the target task with curriculum and reward design
5. compare against the source-policy inference baseline

---

## Step 1: Keep Action/Observation Compatibility

For transfer-friendly adaptation, keep these stable between source and target tasks:

- action definitions and dimensions
- core observation semantics/order
- stream names for multimodal observations

Large schema changes reduce transfer value and usually require retraining from scratch.

---

## Step 2: Choose The Right Source Artifact

Use:

- `.rlcheckpoint`
- checkpoint `.json`

Do not use `.rlmodel` as your primary training-resume artifact. `.rlmodel` is for inference/deployment.

---

## Step 3: Configure Warm-Start

You have two practical options.

### Option A: Run-level checkpoint resume

Set on `RLRunConfig`:

- `ResumeFromCheckpoint = true`
- `ResumeCheckpointPath = "<path to checkpoint>"`

### Option B: RL Imitation warm-start bridge

If you already produced a BC checkpoint through the RL Imitation dock:

1. enable **Warm-start from existing checkpoint** in the dock
2. confirm the checkpoint path
3. start a normal RL run from **RL Setup**

This path writes the warm-start override into the training manifest before launch.

---

## Step 4: Use Inference Baseline As Reference

Before adaptation, deploy the source `.rlmodel` or checkpoint-exported model in the target scene:

- set `InferenceModelPath`
- run in inference mode
- record baseline reward/success metrics

This gives a concrete baseline before adaptation runs.

---

## Step 5: Build A Gentle Adaptation Curriculum

Reduce target-task difficulty initially:

- easier spawn geometry
- slower opponents/obstacles
- denser shaping rewards

Then ramp difficulty with `RLCurriculumConfig` as performance improves.

---

## Step 6: Train New Target Runs Systematically

Run targeted experiments (one variable at a time):

- reward coefficients
- curriculum pacing
- exploration settings

Track all runs in `RLDash` and compare against inference baseline.

---

## Step 7: Promote And Export Best Adapted Policy

When adapted training beats baseline reliably:

1. export the new run/checkpoint to `.rlmodel`
2. update deployment path/version
3. regression-test behavior in production scenes

---

## Common Mistakes

1. Expecting drop-in transfer after major observation/action redesign.
2. Using `.rlmodel` as if it were the primary checkpoint-resume format.
3. Assuming warm-start restores optimizer/replay state.
4. Skipping baseline inference measurement on the new task.
5. Changing many adaptation variables in one run.
6. Treating transfer as successful without production-scene validation.

---

## Minimal Checklist

- source checkpoint chosen
- warm-start path configured
- source model baseline measured on target scene
- action/observation compatibility preserved where possible
- adaptation curriculum and rewards staged
- controlled adaptation runs executed and compared
- best adapted policy exported and validated
