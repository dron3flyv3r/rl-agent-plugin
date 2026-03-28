# RL Agent Plugin for Godot 4

Reinforcement learning plugin for Godot 4 with in-editor training, live metrics, curriculum learning, self-play, distributed rollout workers, and `.rlmodel` export for inference.

## Features

- PPO and SAC trainers
- Shared policy groups across multiple agents
- RL editor tooling: `RLDash` and `RL Setup` dock
- Curriculum and self-play configuration resources
- Distributed rollout collection with headless workers
- Binary `.rlmodel` export/import for deployment

## Requirements

- Godot 4.6+ with C# support
- .NET SDK 8.0+

## Install in a Godot project

Option A: copy plugin folder

1. Copy this repository into your game project as `addons/rl-agent-plugin`.
2. Open Godot -> Project Settings -> Plugins.
4. Build once (`Alt+B`).
3. Enable `RL Agent Plugin`.
4. if RLDash aren't showing in the editor, restart the editor.

Option B: add as submodule

```bash
git submodule add https://github.com/dron3flyv3r/rl-agent-plugin.git addons/rl-agent-plugin
git submodule update --init --recursive
```

## Quick usage

1. Add an `RLAcademy` node in your training scene.
2. Add one or more `RLAgent2D` or `RLAgent3D` nodes.
3. Assign `RLTrainingConfig`, `RLRunConfig`, and per-agent `PolicyGroupConfig` resources.
4. Start training from toolbar (`Start Training`) or `RL Setup` dock.
5. Monitor metrics in `RLDash`.
6. Export checkpoints to `.rlmodel`.

## Documentation

- Docs index: `docs/README.md`
- Feature catalog: `docs/features.md`
- Get started: `docs/get-started.md`
- Configuration reference: `docs/configuration.md`
- Architecture: `docs/architecture.md`
- Algorithms: `docs/algorithms.md`
- Tuning: `docs/tuning.md`

## Companion demo project

If you want ready-to-run demo scenes, use:

- https://github.com/dron3flyv3r/rl-agent-godot

That repo contains training environments and example game scenes that consume this plugin.

## License

MIT
