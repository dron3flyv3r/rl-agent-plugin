# How to Use Self-Play

<!-- markdownlint-disable MD029 MD032 -->

This guide is based on the runtime self-play implementation in `TrainingBootstrap`, `PolicyPool`, and self-play config resources.

---

## What Self-Play Means In This Plugin

Self-play is configured as pairings of policy groups.

- One side can train while the other side is frozen.
- Or both sides can be trainable.
- Opponents are sampled from either:
  - latest checkpoint, or
  - historical frozen snapshots.

During training, matchups are refreshed and frozen opponent policies are loaded for the current episode.

---

## Step 1: Prepare Two Policy Groups In Your Scene

You need at least two train-mode groups in the same training scene.

1. Add agents (`RLAgent2D` / `RLAgent3D`) for both sides.
2. Assign each side a `RLPolicyGroupConfig`.
3. Ensure each group has a distinct `AgentId`.

Example:

```text
Team A agents -> AgentId = "team_a"
Team B agents -> AgentId = "team_b"
```

---

## Step 2: Add Self-Play Config To RLAcademy

On `RLAcademy`:

1. Create `RLSelfPlayConfig`.
2. Add one `RLPolicyPairingConfig` in `Pairings`.
3. Set:
   - `GroupA` and `GroupB`
   - `TrainGroupA` / `TrainGroupB`
   - `HistoricalOpponentRate`
   - `FrozenCheckpointInterval`
   - PFSP settings (`PfspEnabled`, `PfspAlpha`, `MaxPoolSize`, `WinThreshold`)

Starter pairing (both sides train):

```text
RLPolicyPairingConfig
- GroupA: team_a config
- GroupB: team_b config
- TrainGroupA: true
- TrainGroupB: true
- HistoricalOpponentRate: 0.5
- FrozenCheckpointInterval: 10
- MaxPoolSize: 20
- PfspEnabled: true
- PfspAlpha: 4.0
- WinThreshold: 0.7
```

---

## Step 3: Choose One-Sided vs Two-Sided Learning

### One-sided (common first setup)

- `TrainGroupA = true`
- `TrainGroupB = false`

Behavior:
- Group A is learner.
- Group B is frozen opponent role.

### Two-sided (both train)

- `TrainGroupA = true`
- `TrainGroupB = true`

Behavior from runtime:
- Environments alternate learner side by environment index.
- Even index -> Group A learns vs Group B frozen snapshot.
- Odd index -> Group B learns vs Group A frozen snapshot.

---

## Step 4: Set Batch Size Correctly

Self-play enforces a minimum batch size from pairing learner count.

- If one side trains: required batch copies = 1.
- If both sides train: required batch copies = 2.

So for two-sided learning, set at least:

```text
RLRunConfig
- BatchSize: 2 or higher
```

---

## Step 5: Start Training (Not Quick Test)

Use **Start Training**.

Important limitation from editor/bootstrap:
- Quick test mode skips self-play setup and forces `BatchSize = 1`.
- Self-play scenes should be run with normal training launch.

---

## Step 6: Understand Opponent Sampling

Per episode, learner opponent selection does this:

1. If historical pool has entries and random < `HistoricalOpponentRate`, sample historical.
2. Otherwise use latest checkpoint.

If historical snapshot loading fails, runtime falls back to latest checkpoint.

---

## Step 7: Frozen Snapshot Bank

For self-play participant groups, the trainer writes:

- latest checkpoint (normal checkpoint path)
- periodic frozen opponent snapshots every `FrozenCheckpointInterval` updates

Frozen snapshot file pattern:

```text
<RunDirectory>/selfplay/<safe-group-id>/opponent__u000010.json
```

Where `<safe-group-id>` is derived from the group binding id.

Workers in distributed mode periodically rescan this bank so they pick up new frozen snapshots created by the master.

---

## PFSP And Elo Behavior (As Implemented)

### Elo

Each self-play group has an `EloTracker`.

- Initial rating: `1200`
- K-factor: `32`
- Updated from learner win/loss against selected opponent snapshot.

### PFSP weighting

Historical sampling weights come from `OpponentRecord.PfspWeight(alpha)`:

$$
w = \exp\left(-\alpha\,(\text{winRate} - 0.5)^2\right)
$$

Implication:
- Opponents near 50% win-rate get highest weight.
- Very easy and very hard opponents are down-weighted.

Also note:
- Pool eviction removes the easiest snapshot (highest learner win-rate) when full.

---

## Metrics You Can Monitor

Self-play writes extra metric fields per episode, including:

- `opponent_group`
- `opponent_source` (`latest` or `historical`)
- `opponent_checkpoint_path`
- `opponent_update_count`
- `learner_elo`
- `pool_avg_win_rate`

In RLDash, learner Elo appears on the Elo chart when data is present.

---

## Constraints And Validation Rules

Runtime/editor enforce these constraints:

1. Pairing must reference two valid policy groups used by train-mode agents.
2. Pairing cannot use the same group on both sides.
3. At least one side must be trainable.
4. Groups must be disjoint across pairings (v1 does not allow one group in multiple pairings).
5. Batch size must satisfy learner-count requirement.

---

## Minimal Setup Checklist

- `RLAcademy.SelfPlay` assigned.
- At least one valid `RLPolicyPairingConfig`.
- Distinct `AgentId` values for Group A / Group B.
- `BatchSize >= required learners per pairing`.
- Launch via **Start Training**, not quick test.
- Check RLDash for opponent fields + Elo trend.
