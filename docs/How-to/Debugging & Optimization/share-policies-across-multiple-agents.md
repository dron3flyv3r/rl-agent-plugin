# How to Share Policies Across Multiple Agents

<!-- markdownlint-disable MD029 MD032 -->

This guide explains policy sharing with `PolicyGroupConfig.AgentId`.

---

## How Policy Sharing Works

In this plugin, agents that use the same `AgentId` share one policy/trainer.

- same `AgentId` -> shared weights
- different `AgentId` -> separate policies

This is useful for symmetric multi-agent teams.

---

## Step 1: Create Policy Group Config Resources

Create one `RLPolicyGroupConfig` per intended policy population.

For shared team behavior:

- reuse the same `RLPolicyGroupConfig` resource or
- use different resources with the same `AgentId` (if intentionally identical)

Keep this explicit to avoid accidental policy coupling.

---

## Step 2: Assign AgentId Deliberately

Example:

```text
Team A Agent 1 -> AgentId: "team_a"
Team A Agent 2 -> AgentId: "team_a"   (shared policy)
Team B Agent 1 -> AgentId: "team_b"   (separate policy)
```

Choose stable, descriptive IDs because they define training groups and checkpoints.

---

## Step 3: Ensure Shared Agents Are Compatible

Agents sharing a policy must have compatible:

- action definitions
- observation structure/size
- task role assumptions

If behavior roles are very different, separate `AgentId` values usually work better.

---

## Step 4: Verify Grouping At Training Start

At bootstrap, confirm logs/group summaries reflect expected grouping:

- group count matches your intended populations
- each group has expected number of agents

If grouping is wrong, fix `PolicyGroupConfig` assignments and `AgentId` values first.

---

## Step 5: Monitor Shared-Policy Performance

In `RLDash`, inspect metrics per group:

- one shared group should show a single training stream
- separate groups should have separate metrics files/curves

For shared policies, evaluate whether one role dominates learning signal.

---

## When Not To Share

Use separate policies when agents have:

- fundamentally different objectives
- different sensors/action spaces
- asymmetric game mechanics

Forcing policy sharing here often harms both agents.

---

## Common Mistakes

1. Reusing `AgentId` accidentally across unrelated agents.
2. Sharing policy across incompatible observation/action schemas.
3. Expecting separate metrics for agents that intentionally share one policy.
4. Renaming `AgentId` mid-experiment and mixing comparisons.

---

## Minimal Checklist

- intended policy populations defined
- `AgentId` values assigned intentionally
- shared agents validated for compatibility
- training groups verified at launch
- per-group metrics reviewed in RLDash
