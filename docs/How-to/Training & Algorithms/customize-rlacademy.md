# How to Customize RLAcademy

<!-- markdownlint-disable MD029 MD032 -->

This guide explains the custom `RLAcademy` extension model used by this plugin, and how the two demo academies work:

- `demo/11 CustomAcademyGameDev/Scripts/GameDevAcademy.cs`
- `demo/12 CustomAcademyResearch/Scripts/ResearchAcademy.cs`

It is based on the actual runtime flow in `RLAcademy`, `IAcademyContext`, and `TrainingBootstrap`.

---

## What RLAcademy Is Responsible For

`RLAcademy` is the scene-level coordinator for RL runs.

It does two different jobs depending on how far you customize it:

- configuration holder for training, run, curriculum, self-play, and distributed resources
- extension point for custom training behavior

The important design point is that you do not need to replace the whole training loop just to add custom behavior.

There are three levels of customization:

1. Use plain `RLAcademy` with no overrides.
2. Override lifecycle hooks only.
3. Take ownership of the per-frame training loop.

The two demo academies map directly to levels 2 and 3.

---

## The Two Customization Tiers

### Tier 1: Lifecycle Hooks Only

This is the `GameDevAcademy` approach.

You keep the bootstrap's standard training loop and only override hooks such as:

- `OnTrainingInitialized`
- `OnBeforeStep`
- `OnAfterStep`
- `OnEpisodeEnd`
- `OnBeforeCheckpoint`
- `ShouldStop`

Use this tier when you want to:

- print run summaries
- track extra counters
- drive curriculum from reward outcomes
- stop training on your own condition
- write extra logs around checkpoints

You are changing training behavior around the loop, not replacing the loop itself.

### Tier 2 and Tier 3: Own The Training Step

This is the `ResearchAcademy` approach.

You set `OwnsTrainingStep => true`, and then `TrainingBootstrap` calls your `TrainingStep(IAcademyContext ctx)` each physics frame instead of running its own default group loop.

Use this tier when you want to:

- change group scheduling order
- run the four phases manually
- call trainers directly
- control which metrics are logged
- experiment with future parallel rollout strategies

This is full loop ownership, not just a callback layer.

---

## Runtime Call Order

When training starts, `TrainingBootstrap` creates one `IAcademyContext` and passes it to all academy instances.

The runtime order is:

1. `OnTrainingInitialized(ctx)` runs once after trainers and agents are ready.
2. Every physics frame, `OnBeforeStep(ctx)` runs before agent ticking and decision work.
3. The bootstrap gathers pending decisions from train agents.
4. One of two paths runs:
   - default path: bootstrap runs the standard group pipeline itself
   - custom path: primary academy's `TrainingStep(ctx)` runs instead
5. Trainers perform update attempts.
6. Stop conditions are evaluated.
7. `OnAfterStep(ctx)` runs at the end of the frame.

Two important mid-frame hooks happen inside that flow:

- `OnEpisodeEnd(args)` fires during phase B when a finished episode is recorded and the agent is reset.
- `OnBeforeCheckpoint(ctx)` fires immediately before a checkpoint write.

Important details from the implementation:

- `OnEpisodeEnd` may run multiple times in the same frame if several agents finish at once.
- `ShouldStop` is checked every frame after stepping.
- In batched training, lifecycle hooks are broadcast to every academy instance.
- If you own `TrainingStep`, only the primary academy instance drives that custom loop.

---

## What IAcademyContext Gives You

`IAcademyContext` is the runtime API exposed to academy subclasses.

It gives you access to:

- counters such as `TotalSteps` and `EpisodeCountByGroup`
- group discovery via `GroupIds`
- trainer access with `GetTrainer(groupId)`
- all four decision phases
- checkpoint triggering
- custom metric logging
- curriculum control
- graceful stop requests

The phase API is intentionally split:

- phase A: `EstimateNextValues`
- phase B: `RecordTransitionsAndReset`
- phase C: `SampleActions`
- phase D: `ApplyDecisions`

Threading contract:

- phases A and C are pure math and are designed to be thread-safe
- phases B and D touch the Godot scene tree and must stay on the main thread

That split is what makes the research demo useful: it shows how to reorder or eventually parallelize safe parts of the loop without losing the engine-thread guarantees.

---

## How GameDevAcademy Works

`GameDevAcademy` is the simple, production-friendly example.

It does not override `TrainingStep`, so the default bootstrap loop stays in charge.

### What it customizes

- `OnTrainingInitialized(ctx)` prints a one-time run summary
- `OnEpisodeEnd(args)` tracks episode reward totals and advances curriculum when reward crosses a threshold
- `OnBeforeCheckpoint(ctx)` prints the average reward since the last checkpoint
- `ShouldStop(ctx)` stops training when the step budget is exhausted

### Curriculum behavior

The academy keeps a reward gate:

- if `EpisodeReward < RewardThreshold`, nothing changes
- if `EpisodeReward >= RewardThreshold`, curriculum increases by `CurriculumStep`

The new value is pushed through `SetCurriculumProgress(next)`, which clamps the value into `[0, 1]` and immediately notifies curriculum consumers.

### Why this is useful

This pattern is appropriate when you want game-specific control over:

- difficulty progression
- run budgeting
- reporting

without becoming responsible for action sampling, transition recording, or trainer update logic.

---

## How ResearchAcademy Works

`ResearchAcademy` is the experimental example.

It sets `OwnsTrainingStep => true`, which means the academy now controls the per-frame group execution order.

### What it changes

Inside `TrainingStep(ctx)` it:

1. reads `ctx.GroupIds`
2. processes groups in reverse registration order
3. manually runs phases A, B, C, and D
4. fetches each group's trainer with `ctx.GetTrainer(gid)`
5. calls `trainer.TryUpdate(...)` directly
6. logs only selected metrics through `ctx.LogMetric(...)`
7. caches per-group entropy for convergence checks

### Why the demo uses manual phases

The point is not just "different code style". The manual phase split demonstrates control over:

- scheduling order
- where trainer updates happen
- what metrics are emitted
- which parts of the loop are candidates for future parallel execution

### Convergence stop condition

`ResearchAcademy.ShouldStop(ctx)` does not stop on step count.

Instead it waits until:

- every group has reported entropy data
- every group's last known entropy is at or below `ConvergenceEntropyThreshold`
- that condition holds for `ConvergenceGracePeriod` consecutive frames

This is a research-style stopping rule based on policy behavior, not raw throughput.

### One subtle but important detail

Because `ResearchAcademy` calls `trainer.TryUpdate(...)` itself, it drains the rollout buffer there.

That means the bootstrap's later update pass in the same frame will usually find nothing left to update. This is intentional. The academy becomes the owner of both:

- update timing
- per-update metric logging

Episode metrics still come from the normal episode-end handling inside phase B.

---

## When To Use Each Pattern

Choose a `GameDevAcademy`-style subclass when:

- you want custom curriculum logic
- you want custom stop rules
- you want better logging or checkpoint summaries
- the default training loop is already correct for your project

Choose a `ResearchAcademy`-style subclass when:

- you need to control group ordering
- you need direct trainer access
- you want custom update scheduling
- you are testing alternate loop structures or metrics

If you only need extra behavior around training, Tier 1 is the better default. Tier 2 and 3 are more powerful, but they also move loop correctness into your code.

---

## Minimal Starting Templates

### Hook-based academy

```csharp
public partial class MyAcademy : RLAcademy
{
    public override void OnTrainingInitialized(IAcademyContext ctx)
    {
        GD.Print("Training started.");
    }

    public override void OnEpisodeEnd(AcademyEpisodeEndArgs args)
    {
        // Inspect reward / curriculum / group counters here.
    }

    public override bool ShouldStop(IAcademyContext ctx)
        => false;
}
```

### Loop-owning academy

```csharp
public partial class MyResearchAcademy : RLAcademy
{
    public override bool OwnsTrainingStep => true;

    public override void TrainingStep(IAcademyContext ctx)
    {
        foreach (var gid in ctx.GroupIds)
            ctx.RunGroupDecisionPipeline(gid);
    }
}
```

The second template is the safest way to start loop ownership: first reproduce the default group behavior, then change ordering or phase usage one step at a time.

---

## Common Mistakes

1. Owning `TrainingStep` when hooks would have been enough.
2. Calling phase B or D from worker threads.
3. Forgetting that `OnEpisodeEnd` can fire multiple times in one frame.
4. Assuming every batched academy instance independently owns the custom loop.
5. Draining trainer updates manually and then expecting the bootstrap to log the same update stats again.

---

## Recommended Reading Order

1. `Runtime/Core/RLAcademy.cs`
2. `Runtime/Core/IAcademyContext.cs`
3. `Scenes/Bootstrap/TrainingBootstrap.cs`
4. `demo/11 CustomAcademyGameDev/Scripts/GameDevAcademy.cs`
5. `demo/12 CustomAcademyResearch/Scripts/ResearchAcademy.cs`

Read them in that order if you want to understand where each override is entered from and how much responsibility each tier takes on.
