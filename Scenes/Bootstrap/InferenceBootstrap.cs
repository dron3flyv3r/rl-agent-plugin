using System;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin;

/// <summary>
/// Launched by the editor "Run Inference" button via
/// <c>EditorInterface.Singleton.PlayCustomScene("…/InferenceBootstrap.tscn")</c>.
///
/// Reads <see cref="InferenceLaunchManifest"/>, instantiates the training scene, forces any
/// Train or Auto mode agents into Inference mode, then adds the scene so that
/// <see cref="RLAcademy.TryInitializeInference"/> can load their checkpoints and run the policy.
/// </summary>
public partial class InferenceBootstrap : Node
{
    private double _previousTimeScale = 1.0;
    private int _previousPhysicsTicksPerSecond = 60;
    private int _previousMaxPhysicsStepsPerFrame = 8;

    public override void _Ready()
    {
        var manifest = InferenceLaunchManifest.LoadFromUserStorage();
        if (manifest is null)
        {
            GD.PushError(
                "[InferenceBootstrap] Inference manifest not found. " +
                "Use 'Run Inference' in the RL Agent Plugin toolbar to launch inference mode.");
            GetTree().Quit(1);
            return;
        }

        if (string.IsNullOrWhiteSpace(manifest.ScenePath))
        {
            GD.PushError("[InferenceBootstrap] Manifest has no scene path.");
            GetTree().Quit(1);
            return;
        }

        var packedScene = GD.Load<PackedScene>(manifest.ScenePath);
        if (packedScene is null)
        {
            GD.PushError($"[InferenceBootstrap] Could not load scene: {manifest.ScenePath}");
            GetTree().Quit(1);
            return;
        }

        var instance = packedScene.Instantiate();

        // Locate the academy before adding to the scene tree so we can override agent modes.
        var academy = FindAcademy(instance, manifest.AcademyNodePath);
        if (academy is null)
        {
            GD.PushError(
                $"[InferenceBootstrap] RLAcademy not found at '{manifest.AcademyNodePath}' " +
                $"in scene '{manifest.ScenePath}'.");
            instance.QueueFree();
            return;
        }

        // Force every Train / Auto agent to Inference so TryInitializeInference() picks them up.
        var overridden = 0;
        foreach (var agent in academy.GetAgents())
        {
            if (agent.ControlMode == RLAgentControlMode.Train
                || agent.ControlMode == RLAgentControlMode.Auto)
            {
                agent.ControlMode = RLAgentControlMode.Inference;
                overridden++;
            }
        }

        if (overridden == 0)
        {
            GD.PushWarning(
                "[InferenceBootstrap] No Train or Auto mode agents found. " +
                "Switch at least one agent to Train or Auto mode to run quick inference.");
        }
        else
        {
            GD.Print($"[InferenceBootstrap] Switched {overridden} agent(s) to Inference mode.");
        }

        _previousTimeScale = Engine.TimeScale;
        _previousPhysicsTicksPerSecond = Engine.PhysicsTicksPerSecond;
        _previousMaxPhysicsStepsPerFrame = Engine.MaxPhysicsStepsPerFrame;

        // Add directly as a child — ResolveSceneRoot() stops at InferenceBootstrap.
        AddChild(instance);

        // Inference should run at the project's normal speed, independent of training settings.
        Engine.TimeScale = 1.0;
        Engine.PhysicsTicksPerSecond = _previousPhysicsTicksPerSecond;
        Engine.MaxPhysicsStepsPerFrame = _previousMaxPhysicsStepsPerFrame;
    }

    public override void _ExitTree()
    {
        Engine.TimeScale = _previousTimeScale;
        Engine.PhysicsTicksPerSecond = _previousPhysicsTicksPerSecond;
        Engine.MaxPhysicsStepsPerFrame = _previousMaxPhysicsStepsPerFrame;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static RLAcademy? FindAcademy(Node root, string nodePath)
    {
        // Fast path: try NodePath resolution first.
        if (!string.IsNullOrWhiteSpace(nodePath))
        {
            try
            {
                var found = root.GetNode(nodePath);
                if (found is RLAcademy directHit) return directHit;
            }
            catch (Exception) { /* NodePath invalid — fall through to traversal */ }
        }

        // Fallback: depth-first traversal.
        return FindFirstAcademy(root);
    }

    private static RLAcademy? FindFirstAcademy(Node node)
    {
        if (node is RLAcademy academy) return academy;
        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
            {
                var result = FindFirstAcademy(childNode);
                if (result is not null) return result;
            }
        }
        return null;
    }
}
