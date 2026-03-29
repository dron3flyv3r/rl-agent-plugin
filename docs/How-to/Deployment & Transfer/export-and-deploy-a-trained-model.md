# How to Export and Deploy a Trained Model

<!-- markdownlint-disable MD029 MD032 -->

This guide covers exporting a trained policy to `.rlmodel` and deploying it in gameplay scenes.

---

## What Gets Exported

`Export Run` / checkpoint export writes a `.rlmodel` file that contains:

- policy weights
- action-space metadata
- observation metadata used by inference

The file is meant for inference deployment, not raw training metrics.

---

## Step 1: Train Until Behavior Is Stable

Before export, verify in `RLDash`:

- reward trend has stabilized
- behavior in scene is acceptable
- no obvious policy collapse near latest checkpoints

Exporting too early usually creates noisy gameplay AI.

---

## Step 2: Export From RLDash

1. Open `RLDash`.
2. Select the run.
3. Use `Export Run` or export from a checkpoint row.
4. Save the generated `.rlmodel`.

For release builds, prefer exporting from a checkpoint that is both stable and recent.

---

## Step 3: Place Model In Project Assets

Store the `.rlmodel` in a predictable project location, for example:

```text
res://models/agents/my_agent_v1.rlmodel
```

Use versioned names so gameplay scenes can be pinned to known-good policies.

---

## Step 4: Wire Model To Policy Group

On each deployed agent’s `PolicyGroupConfig`:

- set `InferenceModelPath` to your `.rlmodel`
- use `ControlMode = Inference` (or `Auto` with inference launch flow)

Then launch with `Run Inference`.

---

## Step 5: Validate In Production Scene

In your target gameplay scene, verify:

- action behavior matches expected policy
- observation/action shape is compatible with the model
- no runtime warnings about missing/invalid model paths

If a model and scene diverge, retrain or export a model from a matching setup.

---

## Deployment Tips

1. Keep observation stream names stable (especially image stream names).
2. Do not change action definitions for deployed agents without retraining.
3. Version model files (`v1`, `v2`, `hotfix1`) for safe rollback.
4. Keep one known-good fallback model for release stability.

---

## Common Mistakes

1. Exporting before training converges.
2. Pointing `InferenceModelPath` to a non-`.rlmodel` file.
3. Changing observation/action schema after export.
4. Replacing model files without version tracking.

---

## Minimal Checklist

- stable training run selected
- `.rlmodel` exported from RLDash
- `InferenceModelPath` set on target policy group
- scene launched in inference flow
- production behavior sanity-checked
