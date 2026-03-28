# How to Use Curriculum Learning

<!-- markdownlint-disable MD029 MD032 -->

This guide is based on the actual runtime implementation (`TrainingBootstrap`, `RLAcademy`, and `RLCurriculumConfig`) and shows the intended setup flow.

---

## What Curriculum Does In This Plugin

Curriculum is a scalar progress value in `[0, 1]`.

- `0` means easiest difficulty.
- `1` means hardest difficulty.

The bootstrap updates progress during training, then pushes it to all academies via `RLAcademy.SetCurriculumProgress(progress)`, which immediately calls `OnTrainingProgress(progress)` on every discovered `RLAgent2D` / `RLAgent3D`.

Important timing detail:
- Progress is updated before `ResetEpisode()`.
- `ResetEpisode()` calls `OnEpisodeBegin()`.
- So your `OnEpisodeBegin()` sees the newest difficulty.

---

## Step 1: Attach Curriculum Config

In your training scene:

1. Select `RLAcademy`.
2. Assign a new `RLCurriculumConfig` to `Curriculum`.
3. Choose mode:
   - `StepBased`
   - `SuccessRate`

---

## Step 2: Consume Progress In Your Agent

Override `OnTrainingProgress(float progress)` and store the value. Use it in `OnEpisodeBegin()` to scale environment difficulty.

```csharp
public partial class MyAgent : RLAgent2D
{
    private float _curriculum = 0f;

    public override void OnTrainingProgress(float progress)
    {
        _curriculum = progress;
    }

    public override void OnEpisodeBegin()
    {
        // Example: grow arena size as training progresses.
        float arenaRadius = Mathf.Lerp(100f, 500f, _curriculum);
        ConfigureArena(arenaRadius);
    }
}
```

Use this same pattern for `RLAgent3D`.

---

## Step 3A: Configure Step-Based Curriculum

`StepBased` advances using total training steps.

Recommended starter values:

```text
RLCurriculumConfig
- Mode: StepBased
- MaxSteps: 1_000_000
```

Runtime behavior:
- Progress is computed as `combined_steps / MaxSteps`, clamped to `[0, 1]`.
- In distributed training, combined steps include master + workers, so progression tracks real total throughput.

---

## Step 3B: Configure Success-Rate Curriculum

`SuccessRate` adapts difficulty from episode outcomes.

Recommended starter values:

```text
RLCurriculumConfig
- Mode: SuccessRate
- SuccessWindowEpisodes: 50
- SuccessRewardThreshold: 0.5
- PromoteThreshold: 0.7
- DemoteThreshold: 0.3
- ProgressStepUp: 0.1
- ProgressStepDown: 0.1
- RequireFullWindow: true
```

Runtime behavior:
- Episode is counted as success when `episode_reward >= SuccessRewardThreshold`.
- Success rate is computed over the rolling window.
- If success rate >= promote threshold, progress increases.
- If success rate <= demote threshold, progress decreases.
- After a promotion/demotion, the adaptive window is reset.

Implementation note:
- The current bootstrap uses one shared adaptive outcome queue for the run (not per policy group).

---

## Step 4: Verify It Is Working

You can verify curriculum in several ways:

1. Agent-side logging:

```csharp
public override void OnTrainingProgress(float progress)
{
    _curriculum = progress;
    GD.Print($"Curriculum progress: {progress:F2}");
}
```

2. Dashboard:
- `curriculum_progress` is written to metrics when curriculum is active.
- RLDash shows a Curriculum Progress chart.

3. Environment behavior:
- Confirm `OnEpisodeBegin()` actually changes difficulty using `_curriculum`.

---

## Distributed Training Behavior

In distributed mode:

- Master computes curriculum.
- Master broadcasts `CurriculumSync` messages to workers.
- Workers do not compute local curriculum; they apply the latest synced value before episode processing.

This keeps all workers aligned to one authoritative curriculum state.

---

## Debug And Testing Tips

`DebugProgress` in `RLCurriculumConfig` can force a fixed progress for local testing.

Notes from implementation:
- In quick-test mode, bootstrap seeds curriculum to `DebugCurriculumProgress` when it is greater than `0`.
- Outside bootstrap (`RLAcademy._Ready`), debug progress is also applied only when it is greater than `0`.

Practical tip:
- Use values like `0.25`, `0.5`, `1.0` to test easy/mid/hard setups quickly.

---

## Common Mistakes

1. Progress is received, but difficulty never changes:
- You stored progress but do not apply it in `OnEpisodeBegin()`.

2. Success-rate never moves:
- Reward threshold is too high or reward scale is inconsistent.
- Window is too large for your episode frequency.

3. Large oscillations in difficulty:
- Promote/demote thresholds are too close.
- Step up/down values are too large.

4. Expecting per-group adaptive windows:
- Current implementation uses one shared adaptive outcome queue.

---

## Minimal Checklist

- `RLAcademy.Curriculum` assigned.
- Agent overrides `OnTrainingProgress`.
- Agent applies difficulty in `OnEpisodeBegin`.
- Reward definition makes success meaningful (for `SuccessRate`).
- RLDash shows curriculum progress during training.
