# How to Use Human Control Mode

<!-- markdownlint-disable MD029 MD032 -->

This guide is based on runtime behavior in `RLAcademy`, `RLAgent2D`/`RLAgent3D`, and demo agent implementations.

---

## What Human Mode Does

Set an agent `ControlMode` to `Human` when you want keyboard/player input to drive it.

In normal play (outside training bootstrap):

- `RLAcademy` collects all agents with `ControlMode = Human`.
- Every physics tick it calls:
  1. `HandleHumanInput()` -> your overridden `OnHumanInput()`
  2. `TickStep()` -> your `OnStep()` logic
  3. reward consumption/accumulation
  4. episode reset when done or max-step reached

So Human mode still uses the RL episode lifecycle (`OnStep`, rewards, `EndEpisode`, `OnEpisodeBegin`) while actions come from player input.

---

## Step 1: Set Control Mode To Human

On your `RLAgent2D` or `RLAgent3D` node:

1. Open Inspector.
2. Set `ControlMode` to `Human`.

Important:
- `Auto` does not become Human automatically.
- `Auto` maps to Train during **Start Training**, and to Inference during normal play/inference launch.

---

## Step 2: Override OnHumanInput

Implement your input-to-action logic in `OnHumanInput()`.

Example 3D pattern (used in demos):

```csharp
protected override void OnHumanInput()
{
    if (_player is null) return;

    var input = Vector3.Zero;
    if (Input.IsKeyPressed(Key.W) || Input.IsKeyPressed(Key.Up))    input.X += 1f;
    if (Input.IsKeyPressed(Key.S) || Input.IsKeyPressed(Key.Down))  input.X -= 1f;
    if (Input.IsKeyPressed(Key.A) || Input.IsKeyPressed(Key.Left))  input.Z -= 1f;
    if (Input.IsKeyPressed(Key.D) || Input.IsKeyPressed(Key.Right)) input.Z += 1f;

    _player.SetMoveIntent(input.Normalized());
}
```

Another pattern from crawler demo uses actions (`ui_up/down/left/right`) and directly sets motor targets.

---

## Step 3: Keep OnStep And Episode Logic

Human mode does not bypass `OnStep`.

You should still:

- compute rewards in `OnStep`
- call `EndEpisode()` on terminal conditions
- reset scene state in `OnEpisodeBegin()`

Example flow:

```csharp
public override void OnStep()
{
    AddReward(-0.001f, "step_penalty");
    AddReward(_progressReward, "progress");

    if (_reachedGoal || _fellOut)
        EndEpisode();
}
```

---

## Step 4: Run The Scene Normally

Human mode is initialized outside training bootstrap.

Use:
- normal play/run of the scene, or
- Run Inference for other agents while keeping some agents Human.

At startup, `RLAcademy` resets Human agents once and logs:

```text
[RLAcademy] Human mode active for '<agent-name>'.
```

---

## Mixing Human With Inference

You can mix modes in the same scene:

- Human agents are stepped by the human loop.
- Inference/Auto(with model) agents are stepped by inference loop.

This is useful for:

- player vs AI testing
- imitation/gameplay QA while still collecting episode signals

---

## Debugging Tips

1. Enable `RLAcademy.EnableSpyOverlay`.
2. RL spy includes Human and Inference agents by default.
3. For Human agents, overlay triggers observation collection so you can inspect observation values while playing.

---

## Practical Demo References

Current workspace demos using `OnHumanInput()`:

- `demo/03 WallClimbCurriculum/Scripts/WallClimbAgent.cs`
- `demo/04 MoveToTarget3D/Scripts/MoveToTarget3DAgent.cs`
- `demo/05 Crawler/Scripts/CrawlerAgent.cs`

These are good templates for keyboard movement and direct actuator control.

---

## Common Mistakes

1. Setting agent to `Auto` and expecting manual control.
- `Auto` is not Human mode.

2. Overriding `OnHumanInput` but never applying motion/actuation.
- Input must drive your body/controller explicitly.

3. Forgetting episode end conditions.
- If `EndEpisode()` never happens, resets may never occur except max-step limits.

4. Testing inside training bootstrap and expecting human input.
- Human initialization is for normal play (non-training bootstrap path).

---

## Minimal Checklist

- Agent `ControlMode = Human`.
- `OnHumanInput()` implemented.
- `OnStep()` reward/termination logic implemented.
- `OnEpisodeBegin()` reset logic implemented.
- Scene run in normal play mode for manual control.
