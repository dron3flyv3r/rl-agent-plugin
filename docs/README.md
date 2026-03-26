# Documentation Index

## New users
1. [Get Started Guide](get-started.md)
2. [Configuration Reference](configuration.md)
3. [Get Started Tips & Tricks](get-started.md#tips--tricks)

## Deep dives
- [Algorithms (PPO & SAC)](algorithms.md)
- [Tuning Guide](tuning.md)
- [Architecture Overview](architecture.md)

## Demo project
- Demo scenes are maintained in the companion workspace repository:
	- https://github.com/dron3flyv3r/rl-agent-godot
- In this workspace clone, demos are under `../../demo/` from this docs folder.

## Common tasks
- **First agent structure**: keep movement on a player node and put `RLAgent2D/RLAgent3D` as a child agent node
- **Start training**: top toolbar **Start Training** or right-side **RL Setup** dock
- **Watch metrics**: open **RLDash**
- **Export model**: in RLDash, use **Export Run** (or checkpoint-row **Export**) to create `.rlmodel`
- **Run inference**: set `PolicyGroupConfig.InferenceModelPath` and click **Run Inference**
