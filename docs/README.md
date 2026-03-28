# Documentation Index

## New users
1. [Get Started Guide](get-started.md)
2. [Feature Catalog](features.md)
3. [Configuration Reference](configuration.md)
4. [2D vs 3D setup notes](get-started.md)
5. [Get Started Tips & Tricks](get-started.md#tips--tricks)

## Deep dives
- [Algorithms (PPO & SAC)](algorithms.md)
- [Tuning Guide](tuning.md)
- [Architecture Overview](architecture.md)
- [GPU CNN Training](gpu-cnn.md)
- [Sensors (raycast, camera)](sensors.md)

## How-to guides
- [How to Use Curriculum Learning](How-to/curriculum-learning.md)
- [How to Use Self-Play](How-to/self-play.md)
- [How to Use Human Control Mode](How-to/human-control-mode.md)

## Demo scenes
- Example scenes are maintained in the companion demo repository:
	- https://github.com/dron3flyv3r/rl-agent-godot
- See [demos.md](demos.md) for an overview of those scenes and what each one demonstrates.

## Common tasks
- **Choose between 2D and 3D**: start with the differences table and first-run guide in [get-started.md](get-started.md)
- **Do a first training run**: follow the concrete walkthrough in [get-started.md](get-started.md)
- **First agent structure**: player node owns physics; `RLAgent2D/3D` is a child; sensors are children of the agent — see [get-started.md](get-started.md)
- **Add a camera sensor**: add `RLCameraSensor2D` as a child of the agent node, then call `obs.AddImage("name", _camera)` — see [sensors.md](sensors.md)
- **Understand GPU CNN activation**: see [gpu-cnn.md](gpu-cnn.md)
- **Debug camera during training**: tick **Enable Camera Debug** on `RLAcademy` → live preview panel appears top-right
- **Start training**: top toolbar **Start Training** or right-side **RL Setup** dock
- **Watch metrics**: open **RLDash**
- **Export model**: in RLDash, use **Export Run** (or checkpoint-row **Export**) to create `.rlmodel`
- **Run inference**: set `PolicyGroupConfig.InferenceModelPath` and click **Run Inference**
