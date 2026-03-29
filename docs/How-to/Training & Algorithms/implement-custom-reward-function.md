# How to Implement a Custom Reward Function

<!-- markdownlint-disable MD029 MD032 -->

This guide shows a reliable pattern for designing and implementing custom rewards in `RLAgent2D` and `RLAgent3D`.

---

## Reward Design Principles

Use rewards that are:

- **aligned** with the exact behavior you want
- **dense enough** to guide early learning
- **bounded** to avoid unstable scales
- **simple** before adding complexity

A small, clear reward function usually outperforms a complex one.

---

## Step 1: Define Success And Failure Events

Write down terminal conditions first:

- success condition (`EndEpisode()`)
- failure condition (`EndEpisode()`)
- timeout/max-step behavior

Then define reward terms for progress and penalties.

Example structure:

- step penalty: keep episodes efficient
- progress reward: moving toward goal
- terminal bonus: reaching goal
- terminal penalty: failing condition

---

## Step 2: Add Reward Logic In OnStep

Use `OnStep()` as the main reward update location.

```csharp
public override void OnStep()
{
    AddReward(-0.001f, "step_penalty");

    float distance = GlobalPosition.DistanceTo(_target.GlobalPosition);
    float progress = _prevDistance - distance;
    AddReward(progress * 0.1f, "progress");
    _prevDistance = distance;

    if (distance < 0.5f)
    {
        AddReward(1.0f, "goal_reached");
        EndEpisode();
    }

    if (_fellOutOfBounds)
    {
        AddReward(-1.0f, "out_of_bounds");
        EndEpisode();
    }
}
```

Keep term names stable so metrics and debugging are easier.

---

## Step 3: Reset State In OnEpisodeBegin

Reinitialize any reward-related cached state.

```csharp
public override void OnEpisodeBegin()
{
    _prevDistance = GlobalPosition.DistanceTo(_target.GlobalPosition);
}
```

If this step is skipped, progress rewards can become incorrect after resets.

---

## Step 4: Validate Reward Scale

Check in `RLDash` and logs:

- total reward per episode has useful variance
- no single term dominates by orders of magnitude
- rewards correlate with visible behavior improvements

If training is unstable, reduce large terminal or progress coefficients first.

---

## Common Reward Anti-Patterns

1. Rewarding conflicting behaviors at the same time.
2. Using very large sparse rewards with no shaping.
3. Forgetting to terminate episodes on success/failure.
4. Adding too many terms before validating a minimal version.

---

## Iteration Strategy

1. Start with 2-3 reward terms maximum.
2. Train briefly and inspect behavior.
3. Adjust one coefficient at a time.
4. Keep notes of each reward change and run outcome.

---

## Minimal Checklist

- success/failure conditions defined
- reward terms implemented in `OnStep()`
- cached state reset in `OnEpisodeBegin()`
- reward scales validated in metrics
- one-variable-at-a-time tuning process in place
