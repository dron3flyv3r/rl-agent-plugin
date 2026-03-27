# Documentation Index

## New users
1. [Get Started Guide](get-started.md)
2. [Configuration Reference](configuration.md)
3. [Get Started Tips & Tricks](get-started.md#tips--tricks)

## Deep dives
- [Algorithms (PPO & SAC)](algorithms.md)
- [Tuning Guide](tuning.md)
- [Architecture Overview](architecture.md)
- [Sensors (raycast, camera)](sensors.md)

## Demo project
- Demo scenes are maintained in the companion workspace repository:
	- https://github.com/dron3flyv3r/rl-agent-godot
- In this workspace clone, demos are under `../../demo/` from this docs folder.

## Common tasks
- **First agent structure**: player node owns physics; `RLAgent2D/3D` is a child; sensors are children of the agent — see [get-started.md](get-started.md#recommended-node-structure)
- **Add a camera sensor**: add `RLCameraSensor2D` as a child of the agent node, then call `obs.AddImage("name", _camera)` — see [sensors.md](sensors.md)
- **Debug camera during training**: tick **Enable Camera Debug** on `RLAcademy` → live preview panel appears top-right
- **Start training**: top toolbar **Start Training** or right-side **RL Setup** dock
- **Watch metrics**: open **RLDash**
- **Export model**: in RLDash, use **Export Run** (or checkpoint-row **Export**) to create `.rlmodel`
- **Run inference**: set `PolicyGroupConfig.InferenceModelPath` and click **Run Inference**
