# Hyperparameter Optimization (HPO)

The HPO system automates repeated training runs with different hyperparameter values and keeps track of the best trial found so far.

At runtime it works like this:

1. You add an `RLHPOOrchestrator` node as a direct child of `RLAcademy`.
2. You assign an `RLHPOStudy` resource to that orchestrator.
3. When training starts, the orchestrator launches headless trial subprocesses of the same scene.
4. Each trial applies sampled overrides to the trainer config, trains until completion or pruning, and reports its objective value.
5. Study state is written to `res://RL-Agent-Training/hpo/<run-id>/<study-name>/study_state.json`.

Use this when the scene is already learning and you want to search for better values systematically instead of changing one setting by hand.

---

## Quick start

### 1. Add the orchestrator

- Select your `RLAcademy` node.
- Add a child node: `RLHPOOrchestrator`.
- Create a new `RLHPOStudy` resource and assign it to `Study`.

The orchestrator must be a **direct child** of `RLAcademy`. That is how `TrainingBootstrap` detects HPO mode.

### 2. Define the objective

Create an `RLHPOObjectiveConfig` and add one or more `RLHPOObjectiveSource` resources.

For a normal single-policy scene, add one source:

- `PolicyGroup`: your agent group id, for example `reach_target`
- `Metric`: usually `MeanEpisodeReward`

For self-play or multi-policy scenes, add multiple sources and choose how to combine them.

### 3. Add the search space

Create one `RLHPOParameter` per parameter you want to tune.

Common first choices:

- `LearningRate`
- `Gamma`
- `EntropyCoefficient`
- `PpoMiniBatchSize`
- `ClipEpsilon`
- `SacTau`

In the inspector, `ParameterName` uses a grouped dropdown built from `RLTrainerConfig`. You can also choose `Custom` and type a property name manually.

### 4. Set a budget

Start small:

- `TrialBudget`: `8` to `20`
- `MaxTrialSteps`: enough to show early learning signal
- `PruneAfterSteps`: about 25% to 50% of `MaxTrialSteps`
- `PollIntervalSeconds`: `2` to `5`

### 5. Start training

Start training the same way you would for a normal run.

If HPO is active, the academy will not execute one long training session. Instead, it will hand control to the orchestrator, which runs the study loop and spawns trial subprocesses.

---

## Main resources

### `RLHPOOrchestrator`

Node placed under `RLAcademy`.

| Property | Type | Description |
|----------|------|-------------|
| `Study` | `RLHPOStudy` | The study definition to execute. |

Behavior notes:

- Must be a direct child of `RLAcademy`.
- Runs studies sequentially today.
- Each trial is launched as a headless subprocess of the same project and scene.

### `RLHPOStudy`

Top-level HPO configuration resource.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `StudyName` | string | `hpo_study` | Human-readable name used in output paths and run ids. |
| `Direction` | enum | `Maximize` | Whether larger or smaller objective values are better. |
| `ObjectiveConfig` | `RLHPOObjectiveConfig` | `null` | Defines how trial metrics are converted into one scalar objective. Required. |
| `TrialBudget` | int | `20` | Total number of trials to run, including pruned and failed trials. |
| `MaxTrialSteps` | long | `10000` | Hard environment-step cap per trial. `0` disables the cap. |
| `MaxTrialSeconds` | double | `0` | Hard wall-clock cap per trial. `0` disables the cap. |
| `MaxConcurrentTrials` | int | `1` | Reserved for future parallel execution. Only sequential execution is supported today. |
| `SamplerKind` | enum | `TPE` | Sampling strategy for choosing the next parameter set. |
| `PrunerKind` | enum | `Median` | Early stopping strategy for weak trials. |
| `PruneAfterSteps` | long | `10000` | Minimum progress before pruning can happen. |
| `PollIntervalSeconds` | float | `5` | How often the orchestrator checks metrics and pruning status. |
| `SearchSpace` | `Array<RLHPOParameter>` | empty | Parameters to sample each trial. Required. |
| `BaseTrainingConfig` | `RLTrainingConfig` | `null` | Present on the resource, but not currently applied by the trial bootstrap path. Trials still derive training settings from the academy scene's `TrainingConfig` plus sampled overrides. |
| `BaseRunConfig` | `RLRunConfig` | `null` | Optional run settings shared by all HPO trials. When null, the academy scene's run config is used. |

Validation rules enforced by the orchestrator:

- `SearchSpace` must contain at least one parameter.
- `ObjectiveConfig` must exist.
- `ObjectiveConfig.Sources` must contain at least one non-empty source.
- If a source uses `Custom`, `CustomMetricKey` must be set.

### `RLHPOParameter`

One tunable hyperparameter axis.

| Property | Type | Description |
|----------|------|-------------|
| `ParameterName` | string | Name of the `RLTrainerConfig` property to override. Case-sensitive. |
| `Kind` | enum | Sampling distribution. |
| `Low` | float | Inclusive lower bound for numeric kinds. |
| `High` | float | Inclusive upper bound for numeric kinds. |
| `Choices` | `Array<string>` | Numeric string choices for categorical sampling. |

Supported `Kind` values:

| Kind | Use case |
|------|----------|
| `FloatUniform` | Linear range such as `EntropyCoefficient: 0.0 -> 0.05` |
| `FloatLog` | Log-scale range such as `LearningRate: 1e-5 -> 1e-3` |
| `IntUniform` | Integer range such as `PpoMiniBatchSize: 32 -> 256` |
| `IntLog` | Integer log-scale range such as replay sizes or decay steps |
| `Categorical` | Numeric options encoded as strings, for example `["64", "128", "256"]` |

Notes:

- `FloatLog` and `IntLog` require positive bounds.
- `Categorical` choices are parsed as floats before being applied to the trainer config.
- Integer properties are rounded before assignment.
- Bool properties can technically be set through numeric overrides, but the intended workflow is tuning numeric hyperparameters.

### `RLHPOObjectiveConfig`

Defines how trial metrics are scored.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EvaluationWindow` | int | `20` | Number of trailing JSONL metric entries to average per source. |
| `Aggregation` | enum | `Mean` | How multiple sources are combined. |
| `Sources` | `Array<RLHPOObjectiveSource>` | empty | Policy-group metric sources that feed the objective. |

### `RLHPOObjectiveSource`

One metric source inside the objective.

| Property | Type | Description |
|----------|------|-------------|
| `PolicyGroup` | string | Policy group id whose metrics file will be read. |
| `Metric` | enum | Metric to read from that group's JSONL metrics file. |
| `CustomMetricKey` | string | JSONL key used when `Metric = Custom`. |
| `Weight` | float | Relative weight used only by `WeightedMean`. |

Supported `Metric` values:

- `MeanEpisodeReward`
- `MeanEpisodeLength`
- `PolicyLoss`
- `ValueLoss`
- `Custom`

Supported `Aggregation` values:

- `Mean`
- `WeightedMean`
- `Min`
- `Max`

Practical examples:

- Single-policy training: one source, `MeanEpisodeReward`, `Maximize`
- Cooperative self-play: two sources, `WeightedMean`, `Maximize`
- Competitive self-play: two sources, `Min`, `Maximize` if you want settings that avoid one side collapsing

---

## Samplers and pruners

### Samplers

#### `Random`

Samples every parameter independently from its configured distribution.

Use it when:

- you want a simple baseline
- your trial budget is very small
- you want broad exploration with minimal assumptions

#### `TPE`

Tree-structured Parzen Estimator. Early trials are random, then later trials bias toward regions that performed well in completed runs.

Use it when:

- you have at least a modest trial budget
- the search space is mostly numeric
- you want better sample efficiency than pure random search

Implementation note:

- TPE falls back to random suggestions until at least 5 completed trials exist.

### Pruners

#### `None`

Never stops a trial early.

#### `Median`

After `PruneAfterSteps`, prune the current trial if it is worse than the median of completed trials.

Good default when:

- objective values are noisy but still comparable
- you want conservative early stopping

Implementation note:

- Median pruning only activates after at least 3 completed trials.

#### `SuccessiveHalving`

Compares trials at evenly spaced rungs up to `PruneAfterSteps` and keeps only the stronger fraction.

Good when:

- you have a larger trial budget
- weak runs separate clearly from strong runs early

Tradeoff:

- more aggressive than median pruning
- can kill slow-starting but ultimately good settings

---

## Recommended first studies

### PPO

Good first search space:

| Parameter | Kind | Suggested range |
|----------|------|-----------------|
| `LearningRate` | `FloatLog` | `0.0001` to `0.003` |
| `Gamma` | `FloatUniform` | `0.97` to `0.999` |
| `EntropyCoefficient` | `FloatLog` | `0.0005` to `0.03` |
| `PpoMiniBatchSize` | `IntUniform` | `32` to `256` |

Start with:

- `TrialBudget = 12`
- `SamplerKind = TPE`
- `PrunerKind = Median`
- `MaxTrialSteps` set to the smallest budget that still shows reward movement

### SAC

Good first search space:

| Parameter | Kind | Suggested range |
|----------|------|-----------------|
| `LearningRate` | `FloatLog` | `0.0001` to `0.001` |
| `Gamma` | `FloatUniform` | `0.97` to `0.999` |
| `SacTau` | `FloatLog` | `0.001` to `0.02` |
| `SacBatchSize` | `IntUniform` | `128` to `512` |
| `SacInitAlpha` | `FloatLog` | `0.05` to `0.5` |

---

## Monitoring and outputs

Study files are written here:

```text
res://RL-Agent-Training/hpo/<run-id>/<study-name>/
```

Important outputs:

- `study_state.json`: study summary, trial list, best trial id, best objective
- trial run directories under `res://RL-Agent-Training/runs/`
- normal per-trial metrics and status files inside each run directory

The RLDashboard includes an HPO section that reads `study_state.json` files and shows:

- study overview
- trial history
- objective progression
- parameter importance bars
- parallel coordinates
- scatter grid

At the end of a study, the orchestrator prints:

- best trial id
- best objective value
- best sampled parameters

---

## Single-policy example

This is the same pattern used in `demo/07 HPOBasic/ReachTargetHpoDemo.tscn`.

```text
StudyName: reach_target_hpo
Direction: Maximize
Objective:
  EvaluationWindow: 12
  Sources:
    - PolicyGroup: reach_target
      Metric: MeanEpisodeReward
TrialBudget: 12
MaxTrialSteps: 8000
PruneAfterSteps: 2500
PollIntervalSeconds: 2
SearchSpace:
  - LearningRate: FloatLog(0.001, 0.003)
  - Gamma: FloatUniform(0.97, 0.999)
  - EntropyCoefficient: FloatLog(0.0005, 0.03)
  - PpoMiniBatchSize: IntUniform(16, 128)
```

## Self-play example

This is the same pattern used in `demo/08 HPOSelfPlay/TagHpoDemo.tscn`.

```text
StudyName: tag_self_play_hpo
Direction: Maximize
Objective:
  EvaluationWindow: 16
  Aggregation: Min
  Sources:
    - PolicyGroup: chasers
      Metric: MeanEpisodeReward
    - PolicyGroup: runners
      Metric: MeanEpisodeReward
TrialBudget: 8
MaxTrialSteps: 6000
PruneAfterSteps: 4000
PollIntervalSeconds: 3
SamplerKind: TPE
PrunerKind: Median
```

Using `Min` here means a trial only looks good when both sides are doing reasonably well, which is often a better objective than letting one policy dominate while the other collapses.

---

## Limitations and caveats

- Only sequential execution is implemented today. `MaxConcurrentTrials > 1` is not yet active.
- The orchestrator must be a direct child of `RLAcademy`.
- `ParameterName` must match an `RLTrainerConfig` property exactly when you enter it manually.
- `BaseTrainingConfig` is currently exposed on `RLHPOStudy` but not applied by the bootstrap path.
- `Categorical` choices are numeric-only today.
- Objective sources read metrics from policy-group JSONL files, so `PolicyGroup` must match the resolved policy-group id used by training.
- Pruners only compare against completed trials, so pruning is intentionally conservative early in the study.

---

## Troubleshooting

### HPO does not start

Check:

- `RLHPOOrchestrator` is a direct child of `RLAcademy`
- `Study` is assigned
- `SearchSpace` is not empty
- `ObjectiveConfig` exists
- `ObjectiveConfig.Sources` contains at least one valid source

### Trials fail with no objective

Usually one of these:

- `PolicyGroup` does not match the actual metrics file suffix
- the chosen metric key does not exist
- `MaxTrialSteps` or `MaxTrialSeconds` is too small to produce usable metrics

### A parameter does not seem to change anything

Check:

- the name matches the runtime trainer property exactly
- the parameter applies to the current algorithm
- the sampled range is wide enough to matter

For example, `SacTau` does nothing in PPO, and `PpoMiniBatchSize` does nothing in SAC.
