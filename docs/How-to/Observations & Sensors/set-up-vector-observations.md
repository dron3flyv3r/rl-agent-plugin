# How to Set Up Vector Observations

<!-- markdownlint-disable MD029 MD032 -->

This guide shows a practical workflow for designing vector observations in `CollectObservations(ObservationBuffer obs)`.

---

## What Good Vector Observations Look Like

Good observation vectors are:

- relevant to the decision the agent must make now
- normalized to stable ranges
- minimal (avoid redundant features)
- consistent (same order and meaning every step)

In this plugin, the `ObservationBuffer` supports raw adds and normalized adds.

---

## Step 1: Define The Decision Inputs First

Before writing code, list what the policy needs:

- self state (position, velocity, grounded state)
- task state (goal position, remaining time)
- environment state (obstacles, moving targets)

Only include signals that can improve action selection.

---

## Step 2: Normalize Every Continuous Signal

Prefer `obs.AddNormalized(...)` for world values.

Example 2D pattern:

```csharp
public override void CollectObservations(ObservationBuffer obs)
{
    if (_player is null) return;

    var laneMin = Mathf.Min(_player.LaneMinX, _player.LaneMaxX);
    var laneMax = Mathf.Max(_player.LaneMinX, _player.LaneMaxX);
    var laneWidth = Mathf.Max(1.0f, laneMax - laneMin);

    obs.AddNormalized(_player.Position.X, laneMin, laneMax);
    obs.AddNormalized(_player.GoalX, laneMin, laneMax);
    obs.AddNormalized(_player.GoalX - _player.Position.X, -laneWidth, laneWidth);
}
```

`ObservationBuffer` maps normalized values into `[-1, 1]` and clamps out-of-range values.

---

## Step 3: Use Stable Ordering And Segment Names

Keep a fixed feature order over the full project lifetime.

If you use sensors, add names for easier debugging:

```csharp
obs.AddSensor("rays", _raycastSensor);
```

For direct values, keep comments with index ranges when vectors are long.

---

## Step 4: Add Agent + Scene Context

Useful context often includes:

- relative vector to target
- own velocity
- relative vectors to nearest agents/objects
- normalized remaining steps/time

Example multi-agent style:

```csharp
obs.AddNormalized(selfPosition, ArenaMin, ArenaMax);
obs.AddNormalized(remainingSteps, 0, Math.Max(1, EpisodeStepLimit));
```

Relative values usually generalize better than absolute world-space values.

---

## Step 5: Keep Dimension Stable Across Episodes

Do not add/remove features at runtime. Observation size and semantic meaning must stay constant.

If optional data can be missing (for example no target found), write default values (often zeros) instead of changing shape.

---

## Common Mistakes

1. Feeding raw world coordinates with very large magnitudes.
2. Mixing normalized and unnormalized values without intent.
3. Changing observation order during refactors.
4. Adding many weak features instead of a few high-signal ones.

---

## Minimal Checklist

- decision-critical features identified
- continuous values normalized
- feature ordering fixed and documented
- observation size constant every step
- short comments kept for index mapping in long vectors
