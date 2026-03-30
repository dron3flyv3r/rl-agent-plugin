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
- [Building the Native C++ GDExtension](build-native.md)

## Release notes
- [0.1.0-beta Full Technical Release Notes](releases/0.1.0-beta-full-details.md)

## How-to guides

### Training & Algorithms
- [How to Choose Between PPO and SAC](How-to/Training%20&%20Algorithms/choose-between-ppo-and-sac.md)
- [How to Implement a Custom Reward Function](How-to/Training%20&%20Algorithms/implement-custom-reward-function.md)
- [How to Use SAC for Continuous Control](How-to/Training%20&%20Algorithms/use-sac-for-continuous-control.md)
- [How to Switch Between Training Modes](How-to/Training%20&%20Algorithms/switch-between-training-modes.md)
- [How to Use Human Control Mode](How-to/Training%20&%20Algorithms/human-control-mode.md)

### Training Methods
- [How to Use Curriculum Learning](How-to/Training%20Methods/curriculum-learning.md)
- [How to Use Self-Play](How-to/Training%20Methods/self-play.md)

### Observations & Sensors
- [How to Set Up Vector Observations](How-to/Observations%20&%20Sensors/set-up-vector-observations.md)
- [How to Create a Custom Sensor](How-to/Observations%20&%20Sensors/create-a-custom-sensor.md)
- [How to Use Image Observations with CNN](How-to/Observations%20&%20Sensors/use-image-observations-with-cnn.md)
- [How to Use Raycast Sensors for Navigation](How-to/Observations%20&%20Sensors/use-raycast-sensors-for-navigation.md)

### Advanced Training
- [How to Set Up Distributed Workers](How-to/Advanced%20Training/set-up-distributed-workers.md)
- [How to Monitor Training with RLDash](How-to/Advanced%20Training/monitor-training-with-rldash.md)
- [How to Run Multiple Experiments in Parallel](How-to/Advanced%20Training/run-multiple-experiments-in-parallel.md)

### Deployment & Transfer
- [How to Export and Deploy a Trained Model](How-to/Deployment%20&%20Transfer/export-and-deploy-a-trained-model.md)
- [How to Use Inference Mode](How-to/Deployment%20&%20Transfer/use-inference-mode.md)
- [How to Fine-Tune a Pre-Trained Model](How-to/Deployment%20&%20Transfer/fine-tune-a-pre-trained-model.md)

### Debugging & Optimization
- [How to Debug Agent Behavior](How-to/Debugging%20&%20Optimization/debug-agent-behavior.md)
- [How to Tune Hyperparameters Effectively](How-to/Debugging%20&%20Optimization/tune-hyperparameters-effectively.md)
- [How to Share Policies Across Multiple Agents](How-to/Debugging%20&%20Optimization/share-policies-across-multiple-agents.md)
- [How to Handle Training Instability](How-to/Debugging%20&%20Optimization/handle-training-instability.md)

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
