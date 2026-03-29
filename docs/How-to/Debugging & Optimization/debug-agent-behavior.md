# How to Debug Agent Behavior

<!-- markdownlint-disable MD029 MD032 -->

This guide shows a practical debug flow using spy overlay, camera debug, training overlay, and live charts.

---

## Debug Mindset

When behavior is wrong, isolate one question at a time:

1. does the agent observe the right state?
2. does action output affect gameplay correctly?
3. do rewards reinforce intended behavior?
4. do metrics reflect the same story as runtime visuals?

---

## Step 1: Enable Spy Overlay

On `RLAcademy`:

- enable `EnableSpyOverlay`

The spy overlay helps inspect observation/reward/action information while the scene runs.

Use it first when you suspect bad observations or reward tagging.

---

## Step 2: Enable Camera Debug (Image Tasks)

If using `RLCameraSensor2D`:

- enable `Debug -> Enable Camera Debug` on `RLAcademy`

This displays live camera sensor outputs so you can confirm:

- the camera sees meaningful scene content
- crop/zoom settings are correct
- observations are not static or empty

---

## Step 3: Enable Training Overlay For Throughput + Loss Context

In `RLDistributedConfig`:

- set `ShowTrainingOverlay = true`

Use this when diagnosing:

- unexpected step/update cadence
- rollout vs update bottlenecks
- suspicious loss behavior during fast training

---

## Step 4: Correlate Runtime With RLDash

Open `RLDash` and compare overlays with charts:

- reward trend vs visible behavior changes
- entropy trend vs exploration pattern
- policy/value loss vs stability
- episode length vs task objective

If runtime visuals and chart trends disagree, inspect observation and reward wiring first.

---

## Step 5: Validate Agent Lifecycle Hooks

Check agent script behavior in order:

1. `CollectObservations` writes expected values every step
2. `OnActionsReceived` actually moves/acts in world
3. `OnStep` emits non-zero useful rewards
4. `EndEpisode` is reachable and meaningful
5. `OnEpisodeBegin` resets all relevant state

Most “not learning” bugs come from one broken hook.

---

## Common Debug Scenarios

1. Agent does nothing:
- action mapping in `OnActionsReceived` is broken or clamped wrong.

2. Agent behaves randomly forever:
- observations are missing critical task context.

3. Agent exploits weird behavior:
- reward terms unintentionally favor shortcuts.

4. Camera policy fails despite correct code:
- camera framing/crop or renderer setup is wrong.

---

## Minimal Checklist

- spy overlay enabled and inspected
- camera debug validated for image tasks
- training overlay enabled for throughput/loss context
- RLDash trends correlated with runtime behavior
- agent lifecycle hooks checked end-to-end
