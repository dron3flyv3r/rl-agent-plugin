# How to Set Up Distributed Workers

<!-- markdownlint-disable MD029 MD032 -->

This guide shows how to configure multi-process distributed rollout collection for faster training.

---

## What Distributed Workers Do

With `RLDistributedConfig` attached to `RLAcademy`:

- the master process runs trainers and gradient updates
- worker processes run environment rollouts and send transitions
- updated weights are broadcast back to workers

This increases data collection throughput without changing your agent logic.

---

## Step 1: Attach RLDistributedConfig To RLAcademy

On your `RLAcademy` node:

1. Create and assign `RLDistributedConfig`.
2. Set `WorkerCount` (start with 2-4).
3. Keep `AutoLaunchWorkers = true` for first setup.

Minimal starting example:

```text
RLDistributedConfig:
  WorkerCount: 4
  WorkerSimulationSpeed: 4.0
  AutoLaunchWorkers: true
  MonitorIntervalUpdates: 20
```

---

## Step 2: Configure Runtime Throughput Safely

Primary speed knobs:

- `WorkerCount`
- `WorkerSimulationSpeed` (headless workers)
- `RLRunConfig.BatchSize`
- `RLRunConfig.AsyncGradientUpdates`

Recommended first distributed baseline:

```text
RLRunConfig:
  BatchSize: 4
  SimulationSpeed: 1.0
  AsyncGradientUpdates: true
```

Then increase one knob at a time and compare steps/sec + reward trend.

---

## Step 3: Handle Camera Sensor Scenarios Correctly

If your scene uses `RLCameraSensor2D`:

- set `WorkersRequireRenderer = true`
- on Linux servers, configure `XvfbWrapperArgs` if needed
- set `BatchSize = 1` for renderer workers

Important behavior:

- renderer workers run in physics/render lock-step
- `WorkerSimulationSpeed` is effectively ignored in renderer-worker mode
- scale throughput with `WorkerCount` instead

---

## Step 4: Verify Worker Connectivity And Data Flow

Use:

- distributed console logs (`MonitorIntervalUpdates`)
- training overlay (`ShowTrainingOverlay`)
- RLDash reward/loss curves

Healthy setup signs:

- workers connect successfully
- transitions keep arriving
- updates continue at regular cadence

---

## Step 5: Troubleshoot Common Distributed Issues

If workers do not launch/connect:

1. Check `MasterPort` conflicts.
2. Verify `EngineExecutablePath` when using custom Godot binaries.
3. Enable `VerboseLog` for connection details.

If learning slows or destabilizes:

1. Reduce simulation pressure (`WorkerSimulationSpeed`, `BatchSize`).
2. Enable/keep `AsyncGradientUpdates`.
3. Re-check reward scale and observation normalization.

---

## Common Mistakes

1. Using renderer-dependent camera sensors in headless worker mode.
2. Pushing too many throughput knobs at once.
3. Ignoring worker/master executable mismatch.
4. Treating throughput increase as quality increase without reward validation.

---

## Minimal Checklist

- `RLDistributedConfig` assigned to `RLAcademy`
- worker count and speed configured intentionally
- camera-sensor renderer requirements handled
- worker connectivity verified in logs
- reward/loss health monitored in RLDash while scaling
