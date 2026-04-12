# RL Imitation

This guide documents the plugin's current imitation-learning workflow end to end.

Today, "RL Imitation" means:

- recording demonstrations from a running scene into `.rldem` datasets
- training a policy with behavior cloning (BC) from those demonstrations
- running a script-expert DAgger aggregation round to collect learner-visited states
- optionally using the resulting checkpoint as a warm-start for a later RL run

What it does not currently include:

- GAIL / AIRL
- offline RL
- inverse reward learning
- BC training that resumes optimizer state from an earlier run

The current implementation is intentionally narrow: recording, BC, one-round script-expert DAgger dataset aggregation, and a bridge into PPO/SAC/A2C/DQN warm-start through the normal RL training flow.

---

## Workflow At A Glance

1. Open a scene that contains an `RLAcademy` and one or more `RLAgent2D` / `RLAgent3D` agents.
2. Open the **RL Imitation** dock.
3. Record demonstrations in either:
   - **Human** mode: keyboard-driven
   - **Script** mode: the agent's `OnScriptedInput()` heuristic
4. Inspect the generated `.rldem` dataset.
5. Train a BC policy from that dataset.
6. Optionally run a DAgger round to produce an aggregated dataset.
7. Use the saved `.rlcheckpoint`:
   - as a saved training artifact, and
   - as a warm-start checkpoint for a normal RL run from **RL Setup -> Start Training**

The companion demos that exercise this flow are:

- `demo/09 ImitationDemo`
- `demo/10 ImitationMazeDemo`

---

## System Pieces

The RL Imitation system is made of four main parts.

### 1. RL Imitation dock

The editor dock provides three tabs:

- **Record**
- **Train**
- **Dataset Info**

It is implemented in `Editor/Docks/RLImitationDock.cs`.

### 2. Recording bootstrap

When you click **Start Recording**, the editor launches `RecordingBootstrap.tscn`, which:

- opens the selected gameplay scene
- finds the target `RLAcademy`
- filters the agents by selected policy group
- drives them in Human or Script mode
- writes step data to a `.rldem` file

This logic lives in `Scenes/Bootstrap/RecordingBootstrap.cs`.

### 3. Demonstration dataset format

Recorded data is stored as a binary `.rldem` file. Each frame contains:

- `agent_slot`
- observation vector
- discrete action index
- continuous action vector
- reward
- done flag

The loader/writer lives in `Runtime/Imitation/DemonstrationDataset.cs`.

### 4. Behavior cloning trainer

BC training rebuilds a `PolicyValueNetwork`, then runs supervised updates over recorded frames.

Current BC trainer characteristics:

- mini-batch SGD with Adam-backed network updates
- fixed batch size of `64`
- configurable `Epochs`
- configurable `LearningRate`
- gradient clipping via `MaxGradientNorm`
- dataset shuffling each epoch

This logic lives in:

- `Runtime/Imitation/BCTrainer.cs`
- `Runtime/Imitation/RLImitationConfig.cs`

---

## Record Tab

The **Record** tab is the front door to the system.

### Scene

The dock always works against the currently open editor scene.

If no scene is open, recording is unavailable.

### Agent

You can record:

- **All agents**
- one specific policy group / agent group

Use **All agents** only when the recorded agents truly share the same observation and action schema. The recording format assumes one common schema for the dataset.

### Mode

The dock exposes two recording modes.

#### Human mode

Human mode records the action actually applied while the scene is driven by your control path.

Typical pattern:

- your gameplay node reads keyboard input when the agent is in `RLAgentControlMode.Human`
- the agent applies that action
- the bootstrap records the resulting `(obs, action, reward, done)` frame

See also: [How to Use Human Control Mode](How-to/Training%20&%20Algorithms/human-control-mode.md)

#### Script mode

Script mode calls the agent's scripted heuristic through `OnScriptedInput()`.

Use this when:

- you already have a good hand-written policy
- you want fast dataset generation
- keyboard demonstration would be too slow or inconsistent

The maze demo is a good example of this pattern.

### Speed

Recording speed presets write a control file that the recording bootstrap polls at runtime. This lets you speed up or slow down data collection without editing the scene.

### Pause

Pause stops frame writing, but the game keeps running. This is useful when you need to reposition yourself mentally or inspect the current scene without polluting the dataset.

### Step mode

Step mode is a recording aid for discrete-action human-controlled agents.

What it does:

- lets you inject one discrete action at a time
- records one frame per requested step
- exposes labeled action buttons after the recording bootstrap confirms the action space

Current constraints:

- only available in **Human** mode
- only available for **specific agent selection**, not **All agents**
- only available for **discrete-only** action spaces
- not available in Script mode

This is useful for:

- high-precision demonstrations
- debugging which action label maps to which behavior
- creating tiny deterministic datasets

---

## What Gets Written During Recording

The main dataset file is written to:

- `res://RL-Agent-Demos/<name>_<timestamp>.rldem`

During recording, a few sidecar files are also used:

- `.stop`: written by the editor when you click **Stop**
- `.done`: written by the bootstrap after the dataset is finalized
- `.ctrl`: current speed / pause / step-mode control state
- `.status`: live recording stats for the dock
- `.cap`: action-space capability data for step-mode enablement

These sidecars are part of the editor/bootstrap handshake. The important long-lived artifact is the `.rldem` file.

### Recorded frame contents

Each frame stores:

- the observation array collected for that step
- the current discrete or continuous action
- the last-step reward
- whether the agent reached a terminal state
- the recording slot index of the agent that produced the frame

The first frame after launch is intentionally skipped to avoid capturing stale pre-step state.

---

## Train Tab

The **Train** tab runs behavior cloning against a selected dataset.

### Dataset

The dataset dropdown scans:

- `res://RL-Agent-Demos/*.rldem`

The list is sorted newest-first.

### Network graph

BC training always rebuilds a network before applying dataset updates.

The graph is resolved from:

1. the selected agent's configured network graph, or
2. a manual override entered in the dock

If you selected **All agents**, the dock can auto-resolve the graph only when all listed agents share the same graph. Otherwise, select a specific agent or provide an override.

### Hyperparameters

The current UI exposes:

- `Epochs`
- `LearningRate`

Internally, BC also uses:

- `BatchSize = 64`
- `MaxGradientNorm = 0.5`
- `ShuffleEachEpoch = true`

These are currently code-level defaults, not full train-tab controls.

### Output

BC training writes checkpoints to:

- `res://RL-Agent-Demos/trained/bc_<timestamp>.rlcheckpoint`

The dock keeps the newest BC checkpoint around as the default warm-start candidate for later RL runs.

---

## DAgger

The train tab exposes both:

- **Manual DAgger**
- **Run Auto DAgger**

Current DAgger support is:

- one aggregation round at a time
- scripted expert only
- seed dataset + newly collected expert-labeled learner states written into a new `.rldem`
- an optional automated outer loop that chains BC and DAgger rounds together

Current DAgger support is not:

- human-in-the-loop expert relabeling
- a separate discriminator / reward-learning method

### How it works

During a DAgger round:

1. the selected learner checkpoint acts in the scene
2. the agent's `OnScriptedInput()` heuristic is queried as the expert
3. the visited observation is written with the expert action label
4. the resulting output dataset contains:
   - the original seed dataset frames
   - the newly collected learner-visited states

You then run BC again on that aggregated dataset.

### Requirements

DAgger currently requires:

- a specific selected agent group
- a scripted expert implementation through `OnScriptedInput()`
- a learner checkpoint
- a compatible network graph
- a seed dataset

### Output

DAgger datasets are written to:

- `res://RL-Agent-Demos/<name>_<timestamp>.rldem`

The dataset is immediately available in the normal dataset dropdown after the run completes.

### Manual DAgger

1. record or script-generate an initial dataset
2. train BC
3. run one DAgger round from that checkpoint
4. train BC again on the aggregated dataset
5. repeat as needed

### Auto DAgger

`Auto DAgger` automates the outer loop in the editor.

If you provide a checkpoint, the loop starts from that learner.
If you leave the checkpoint field empty, the loop first trains a seed BC checkpoint from the selected dataset.

Each round then does:

1. run one DAgger aggregation round
2. write a new aggregated dataset
3. retrain BC on that aggregated dataset

After the last round, the final BC checkpoint remains selected in the dock and the final aggregated dataset remains available in the dataset list.

---

## Dataset Info Tab

The **Dataset Info** tab is for validation before you train.

It shows:

- file name
- whether the dataset could be read
- total frames
- inferred episode count
- average episode length
- average episode reward
- observation size
- action-space summary

It also includes a frame preview with:

- frame index
- agent slot
- observation vector
- action
- reward
- done marker

Use this tab before BC training. It is the fastest way to catch:

- empty recordings
- obviously wrong action values
- observation vectors that are all zero or clearly malformed
- datasets that end without meaningful terminal states

---

## Warm-Start Behavior

This is the most important distinction in the current implementation.

### BC train warm-start checkbox

The **Train** tab contains a checkbox labeled **Warm-start from existing checkpoint** and a checkpoint path field.

Current behavior:

- the UI stores and displays the warm-start path
- the path is forwarded into the normal RL training launch flow
- **BC training itself still starts from scratch**

In other words, the BC trainer currently does not load an existing checkpoint before supervised updates.

### RL training warm-start

The warm-start path does matter for the normal RL training flow.

If you enable the RL Imitation warm-start option and then start an RL run from **RL Setup -> Start Training**, the training bootstrap:

- resolves the selected checkpoint path
- loads the checkpoint into the live trainer network
- starts PPO/SAC/A2C/DQN/MCTS training from those weights

This is the intended `BC -> RL fine-tune` bridge.

### DAgger learner checkpoint

The same checkpoint path is also used by the DAgger round as the learner policy source.

That means the current train tab has two distinct checkpoint uses:

- BC train: displayed only, not loaded into BC yet
- DAgger round: loaded and used to act in the environment

### What "resume" means here

Checkpoint loading currently restores:

- network weights
- training counters such as steps / episodes / updates
- checkpoint reward snapshot
- curriculum progress

It does not restore:

- optimizer state
- PPO rollout buffers
- SAC / DQN replay buffers

So this is best thought of as **weight warm-start / partial resume**, not a perfect continuation of every internal trainer state.

---

## Recommended Workflows

### Workflow A: Human demonstrations -> BC

Use this when:

- the behavior is easy for a human to demonstrate
- the action space is small and intuitive
- you want to quickly prototype a controllable imitation task

Recommended examples:

- top-down navigation
- simple 2D steering
- short scripted interactions

### Workflow B: Scripted heuristic -> BC

Use this when:

- you already know a reasonable heuristic
- you want large, consistent datasets
- human data would be noisy

Recommended examples:

- navigation with a known pathing rule
- simple avoidance heuristics
- deterministic puzzle or routing tasks

### Workflow C: BC -> PPO fine-tune

Use this when:

- BC gets you close but not fully robust
- you want recovery from off-demo states
- you want the learned policy to exceed the demonstrator

Typical flow:

1. record or script-generate demonstrations
2. train BC to produce `bc_*.rlcheckpoint`
3. enable warm-start from that checkpoint
4. launch PPO training from RL Setup

This is the strongest current imitation-related workflow in the plugin.

### Workflow D: BC -> DAgger -> BC

Use this when:

- BC works on demonstration states but drifts on its own rollouts
- you already have a decent scripted oracle
- you want to fix covariate shift before switching to RL fine-tuning

Typical flow:

1. train BC
2. run a DAgger round from the BC checkpoint
3. retrain BC on the aggregated dataset
4. repeat or switch to PPO fine-tuning

---

## Agent Requirements

For RL Imitation to work cleanly, your scene should follow the same separation used elsewhere in the plugin:

- gameplay/physics node owns movement
- child `RLAgent2D` / `RLAgent3D` owns observations, rewards, and actions

### To support Human recording

Your runtime control path must respond when the agent is in human mode.

Common pattern:

- if `ControlMode == Human`, read keyboard input
- convert the input into the same action representation used in training
- call `ApplyAction(...)` or route the action through the player/body

### To support Script recording

Override `OnScriptedInput()` in your agent and apply a legal action each step.

### To support step mode

Your recorded agent must:

- use discrete actions only
- expose a stable action definition
- allow `PendingStepAction` injection during human-mode stepping

The built-in demos show the intended pattern.

---

## Best Practices

- Keep observation semantics stable between recording and later RL fine-tuning.
- Use a specific agent selection unless you truly want a shared multi-agent dataset.
- Check the **Dataset Info** tab before training every new dataset.
- Prefer script-generated demos when consistency matters more than human style.
- Start with short, correct demonstrations before collecting large noisy datasets.
- If BC performance is decent but brittle, switch to `BC -> PPO` instead of just recording more low-quality data.
- If BC performance is decent but brittle and you have a scripted oracle, try `BC -> DAgger -> BC`.
- Version your datasets and checkpoints with descriptive output names.

---

## Troubleshooting

### "No datasets found"

The train tab only scans:

- `res://RL-Agent-Demos/*.rldem`

Make sure the recording finished and the file ended up there.

### Recording starts but nothing useful is learned

Check:

- observation vectors in **Dataset Info**
- whether the recorded action labels match what you think they mean
- whether your demonstrations actually terminate episodes and collect reward
- whether the dataset covers the important states the policy will see later

### Step mode never enables

Step mode requires:

- Human mode
- a specific agent, not All agents
- a discrete-only action space
- a running recording session long enough for the `.cap` file to be read

### Warm-start path shows up, but BC still starts fresh

That is the current implementation. The warm-start field is for later RL training, not BC training itself.

### DAgger button is unavailable or fails immediately

Check:

- a specific agent group is selected
- the selected agent supports Script mode
- a learner checkpoint path is set
- the selected network graph matches the learner checkpoint architecture

### Fine-tuned RL run behaves strangely after BC

Check:

- network graph compatibility
- observation/action schema compatibility
- whether the BC demonstrations and RL reward function are pulling the policy in different directions

---

## Current Limitations

- Imitation learning is BC-only today.
- DAgger is currently script-expert only and runs one aggregation round at a time.
- BC training does not yet load an earlier checkpoint before supervised updates.
- The train tab exposes only epochs and learning rate; batch size and other BC knobs are fixed in code.
- Step mode supports discrete-only agents.
- The dataset format is vector-observation oriented; it stores flattened observation arrays, not high-level semantic labels.
- Multi-agent recording is practical only when all recorded agents share one compatible schema.

---

## Related Docs

- [Get Started Guide](get-started.md)
- [How to Use Human Control Mode](How-to/Training%20&%20Algorithms/human-control-mode.md)
- [Algorithms](algorithms.md)
- [Configuration Reference](configuration.md)
- [How to Fine-Tune a Pre-Trained Model](How-to/Deployment%20&%20Transfer/fine-tune-a-pre-trained-model.md)
