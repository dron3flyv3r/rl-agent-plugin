# How to Switch Between Training Modes

<!-- markdownlint-disable MD029 MD032 -->

This guide explains how to switch agents between `Train`, `Inference`, `Human`, and `Auto` control modes.

---

## Mode Behavior Summary

`ControlMode` determines how each agent is stepped:

- `Train`: agent participates in training bootstrap loops
- `Inference`: agent runs with loaded model inference
- `Human`: agent uses `OnHumanInput()` in normal play loop
- `Auto`: context-based behavior
  - maps to `Train` during **Start Training**
  - maps to `Inference` during normal play / **Run Inference**

---

## Step 1: Set Per-Agent ControlMode

For each `RLAgent2D` or `RLAgent3D`:

1. Select the agent node.
2. In Inspector, set `ControlMode`.
3. Repeat for all agents in multi-agent scenes.

Use explicit modes for deterministic behavior during debugging.

---

## Step 2: Choose The Runtime Entry Point

Use the editor action that matches your intent:

- **Start Training**: bootstraps training systems
- **Run Inference**: loads inference paths and runs policies
- normal scene play: executes non-training runtime path

`Auto` relies on this entry point, so launching path matters.

---

## Step 3: Mix Modes Deliberately

You can run mixed-mode scenes, for example:

- one `Human` player agent
- one `Inference` opponent
- one `Train` bot in dedicated training scene (separate run)

Keep mixed-mode goals clear to avoid confusion in metrics and behavior interpretation.

---

## Step 4: Validate The Active Mode

At runtime, verify mode by behavior and logs:

- Human: `OnHumanInput()` is active
- Inference: model-driven decisions are active
- Train: episode statistics and trainer updates are active

If behavior is unexpected, check `ControlMode` plus launch path first.

---

## Common Mode Pitfalls

1. Expecting `Auto` to behave like `Human`.
2. Forgetting to assign `InferenceModelPath` for inference workflows.
3. Running gameplay scene in training mode unintentionally.
4. Interpreting mixed-mode results as pure training metrics.

---

## Recommended Workflow

1. Development/debug: use explicit `Human` or explicit `Inference`.
2. Training scenes: use explicit `Train` (or `Auto` with controlled launch flow).
3. Production gameplay: use `Inference` or `Auto` with known model paths.

---

## Minimal Checklist

- every agent has intentional `ControlMode`
- launch path matches expected runtime mode
- inference model paths assigned where needed
- mixed-mode scenes documented to avoid metric confusion
