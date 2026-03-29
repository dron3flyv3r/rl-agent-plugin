# How to Use Inference Mode

<!-- markdownlint-disable MD029 MD032 -->

This guide explains how to run trained models in normal gameplay using inference mode.

---

## What Inference Mode Does

In inference mode:

- the agent loads a `.rlmodel`
- actions are produced by an `IInferencePolicy`
- no gradient updates or rollout training buffers are used

It is the correct runtime mode for shipped gameplay behavior.

---

## Step 1: Prepare The Model Path

For each policy group used in gameplay:

- assign `PolicyGroupConfig.InferenceModelPath`
- ensure the file exists and ends with `.rlmodel`

Example path:

```text
res://models/agents/enemy_v3.rlmodel
```

---

## Step 2: Choose ControlMode Strategy

Use one of these:

- explicit inference: `ControlMode = Inference`
- auto workflow: `ControlMode = Auto` + valid `InferenceModelPath`

`Auto` maps to inference in normal play / `Run Inference` flow when a model is assigned.

---

## Step 3: Launch Inference Runtime

From editor tooling:

1. open scene with `RLAcademy`
2. click `Run Inference`
3. confirm agents are model-driven in play

You can also mix `Human` and `Inference` agents in the same scene for testing.

---

## Step 4: Verify Compatibility

Inference requires model and agent compatibility:

- same action-space definition
- compatible observation structure/size
- compatible modality setup (vector/image streams)

If incompatible, runtime logs warnings/errors and behavior will be invalid.

---

## Step 5: Optional Stochastic Inference

If your policy group supports it and you want non-deterministic behavior:

- enable `PolicyGroupConfig.StochasticInference`

For production competitive gameplay, deterministic behavior is usually easier to test and balance.

---

## Common Mistakes

1. Forgetting to set `InferenceModelPath`.
2. Expecting training updates while in inference mode.
3. Using a model trained with different observation/action schema.
4. Assuming `Auto` implies human control.

---

## Minimal Checklist

- valid `.rlmodel` path configured
- `Inference` or `Auto` mode chosen intentionally
- launched via `Run Inference` / normal play inference path
- model-agent compatibility verified
- runtime behavior confirmed in scene
