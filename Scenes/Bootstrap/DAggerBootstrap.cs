using System;
using System.Collections.Generic;
using Godot;
using RlAgentPlugin.Runtime;
using RlAgentPlugin.Runtime.Imitation;

namespace RlAgentPlugin;

/// <summary>
/// Launches one DAgger aggregation round:
/// the learner policy acts in the scene, while the agent's scripted heuristic
/// provides the expert action label for each visited state.
///
/// The resulting dataset is written as:
///   seed dataset frames + newly collected expert-labeled learner states
/// </summary>
public partial class DAggerBootstrap : Node
{
    private DemonstrationWriter? _writer;
    private DemonstrationDataset? _seedDataset;
    private readonly List<IRLAgent> _agents = new();
    private readonly Dictionary<IRLAgent, RecurrentState?> _recurrentStates = new();
    private IInferencePolicy? _policy;
    private string _doneMarkerPath = string.Empty;
    private string _statusFilePath = string.Empty;
    private string _stopSignalPath = string.Empty;
    private bool _skipFirstFrame = true;
    private bool _stopping;
    private int _episodeCount;
    private int _seedFrameCount;
    private int _collectedFrameCount;
    private int _targetAdditionalFrames;
    private int _obsSize;
    private int _discreteCount;
    private int _contDims;
    private int _statusFrameCounter;
    private const int StatusWriteInterval = 30;

    // Beta mixing: probability that the expert (not the learner) drives a given step.
    // Computed as MixingBeta^RoundIndex from the manifest so it decays each round.
    private float _effectiveBeta;
    private Random? _rng;
    private int _expertDrivenSteps;

    // Action repeat: only collect a frame and make a new decision every N physics steps.
    private int _actionRepeat = 1;
    private int _physicsStepsSinceDecision;

    public override void _Ready()
    {
        var manifest = DAggerLaunchManifest.LoadFromUserStorage();
        if (manifest is null)
        {
            GD.PushError("[DAggerBootstrap] DAgger manifest not found.");
            GetTree().Quit(1);
            return;
        }

        if (string.IsNullOrWhiteSpace(manifest.ScenePath))
        {
            GD.PushError("[DAggerBootstrap] Manifest has no scene path.");
            GetTree().Quit(1);
            return;
        }

        if (string.IsNullOrWhiteSpace(manifest.LearnerCheckpointPath))
        {
            GD.PushError("[DAggerBootstrap] Manifest has no learner checkpoint path.");
            GetTree().Quit(1);
            return;
        }

        var packedScene = GD.Load<PackedScene>(manifest.ScenePath);
        if (packedScene is null)
        {
            GD.PushError($"[DAggerBootstrap] Could not load scene: {manifest.ScenePath}");
            GetTree().Quit(1);
            return;
        }

        // Accept both .rlcheckpoint/.json training checkpoints and .rlmodel inference files.
        var checkpoint = manifest.LearnerCheckpointPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase)
            ? RLModelLoader.LoadFromFile(manifest.LearnerCheckpointPath)
            : RLCheckpoint.LoadFromFile(manifest.LearnerCheckpointPath);
        if (checkpoint is null)
        {
            GD.PushError($"[DAggerBootstrap] Could not load learner checkpoint: {manifest.LearnerCheckpointPath}");
            GetTree().Quit(1);
            return;
        }

        // Network graph: required when the checkpoint does not carry its own layer definitions
        // (e.g. BC checkpoints exported as .rlmodel before NetworkLayers metadata was written).
        RLNetworkGraph? networkGraph = null;
        if (!string.IsNullOrWhiteSpace(manifest.NetworkGraphPath))
        {
            networkGraph = GD.Load<RLNetworkGraph>(manifest.NetworkGraphPath);
            if (networkGraph is null)
                GD.PushWarning($"[DAggerBootstrap] Could not load network graph '{manifest.NetworkGraphPath}' — will rely on checkpoint metadata.");
        }
        else if (checkpoint.NetworkLayers.Count == 0)
        {
            GD.PushError("[DAggerBootstrap] No network graph path set and the checkpoint carries no layer definitions. " +
                         "Provide a NetworkGraphPath in the DAgger settings.");
            GetTree().Quit(1);
            return;
        }

        var instance = packedScene.Instantiate();
        var academy = FindAcademy(instance, manifest.AcademyNodePath);
        if (academy is null)
        {
            GD.PushError($"[DAggerBootstrap] RLAcademy not found in scene '{manifest.ScenePath}'.");
            instance.QueueFree();
            GetTree().Quit(1);
            return;
        }

        foreach (var agent in academy.GetAgents())
        {
            var inGroup = string.IsNullOrEmpty(manifest.AgentGroupId)
                || agent.PolicyGroupConfig?.AgentId == manifest.AgentGroupId;
            if (!inGroup) continue;
            _agents.Add(agent);
        }

        if (_agents.Count == 0)
        {
            GD.PushError("[DAggerBootstrap] No agents matched the selected policy group.");
            instance.QueueFree();
            GetTree().Quit(1);
            return;
        }

        AddChild(instance);

        foreach (var agent in _agents)
            agent.ControlMode = RLAgentControlMode.Auto;

        var spec = _agents[0].CollectObservationSpec();
        _obsSize = spec.TotalSize;
        _discreteCount = _agents[0].GetDiscreteActionCount();
        _contDims = _agents[0].GetContinuousActionDimensions();
        _actionRepeat = Math.Max(1, academy.ActionRepeat);
        _targetAdditionalFrames = Math.Max(1, manifest.AdditionalFrames);

        var beta = Math.Clamp(manifest.MixingBeta, 0f, 1f);
        var round = Math.Max(1, manifest.RoundIndex);
        _effectiveBeta = (float)Math.Pow(beta, round);
        _rng = new Random();
        GD.Print($"[DAggerBootstrap] Beta mixing: base={beta:F3} round={round} effective={_effectiveBeta:F3} " +
                 $"(~{_effectiveBeta * 100:F0}% expert-driven steps), action_repeat={_actionRepeat}");

        foreach (var agent in _agents)
        {
            var agentObs = agent.CollectObservationSpec().TotalSize;
            if (agentObs != _obsSize
                || agent.GetDiscreteActionCount() != _discreteCount
                || agent.GetContinuousActionDimensions() != _contDims)
            {
                GD.PushError("[DAggerBootstrap] All selected agents must share the same observation and action schema.");
                GetTree().Quit(1);
                return;
            }
        }

        if (!string.IsNullOrWhiteSpace(manifest.SeedDatasetPath))
        {
            _seedDataset = DemonstrationDataset.Open(manifest.SeedDatasetPath);
            if (_seedDataset is null)
            {
                GD.PushError($"[DAggerBootstrap] Could not load seed dataset: {manifest.SeedDatasetPath}");
                GetTree().Quit(1);
                return;
            }

            if (_seedDataset.ObsSize != _obsSize
                || _seedDataset.DiscreteActionCount != _discreteCount
                || _seedDataset.ContinuousActionDims != _contDims)
            {
                GD.PushError("[DAggerBootstrap] Seed dataset schema does not match the active agent schema.");
                GetTree().Quit(1);
                return;
            }
        }

        if (checkpoint.ObservationSize > 0 && checkpoint.ObservationSize != _obsSize)
        {
            GD.PushError("[DAggerBootstrap] Learner checkpoint observation size does not match the active agent.");
            GetTree().Quit(1);
            return;
        }

        if (checkpoint.DiscreteActionCount > 0 && checkpoint.DiscreteActionCount != _discreteCount)
        {
            GD.PushError("[DAggerBootstrap] Learner checkpoint discrete action count does not match the active agent.");
            GetTree().Quit(1);
            return;
        }

        if (checkpoint.ContinuousActionDimensions > 0 && checkpoint.ContinuousActionDimensions != _contDims)
        {
            GD.PushError("[DAggerBootstrap] Learner checkpoint continuous action dimensions do not match the active agent.");
            GetTree().Quit(1);
            return;
        }

        if (checkpoint.ObservationSize <= 0)
            checkpoint.ObservationSize = _obsSize;
        if (checkpoint.DiscreteActionCount <= 0)
            checkpoint.DiscreteActionCount = _discreteCount;
        if (checkpoint.ContinuousActionDimensions <= 0)
            checkpoint.ContinuousActionDimensions = _contDims;

        _policy = InferencePolicyFactory.Create(checkpoint, networkGraph);
        _policy.LoadCheckpoint(checkpoint);
        foreach (var agent in _agents)
            _recurrentStates[agent] = _policy.CreateZeroRecurrentState();

        _doneMarkerPath = manifest.OutputFilePath + ".done";
        _statusFilePath = manifest.OutputFilePath + ".status";
        _stopSignalPath = manifest.OutputFilePath + ".stop";
        _writer = DemonstrationWriter.Create(manifest.OutputFilePath, _obsSize, _discreteCount, _contDims);
        if (_writer is null)
        {
            GD.PushError($"[DAggerBootstrap] Failed to create output file: {manifest.OutputFilePath}");
            GetTree().Quit(1);
            return;
        }

        if (_seedDataset is not null)
        {
            foreach (var frame in _seedDataset.Frames)
                _writer.WriteFrame(frame);
            _seedFrameCount = _seedDataset.Frames.Count;
        }

        GD.Print(
            $"[DAggerBootstrap] Collecting {_targetAdditionalFrames} additional frame(s) " +
            $"for group '{manifest.AgentGroupId}' from learner '{manifest.LearnerCheckpointPath}'.");
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_stopping) return;

        if (FileAccess.FileExists(_stopSignalPath))
        {
            StopCollection();
            return;
        }

        if (_skipFirstFrame)
        {
            _skipFirstFrame = false;
            return;
        }

        if (_collectedFrameCount >= _targetAdditionalFrames)
        {
            StopCollection();
            return;
        }

        // Advance the step counter and decide whether this is a decision point.
        _physicsStepsSinceDecision++;
        var isDecisionStep = _physicsStepsSinceDecision >= _actionRepeat;
        if (isDecisionStep)
            _physicsStepsSinceDecision = 0;

        for (var slot = 0; slot < _agents.Count; slot++)
        {
            var agent = _agents[slot];

            if (isDecisionStep && _collectedFrameCount < _targetAdditionalFrames)
            {
                var obs = agent.CollectObservationArray();
                var learnerDecision = PredictDecision(agent, obs);

                // Ask the expert what it would do in this state.
                // HandleScriptedInput sets CurrentActionIndex / CurrentContinuousActions via ApplyAction.
                agent.HandleScriptedInput();
                var expertDiscrete = _discreteCount > 0 ? agent.CurrentActionIndex : -1;
                var expertContinuous = (float[])agent.CurrentContinuousActions.Clone();

                // Beta mixing: with probability effectiveBeta the expert drives the step;
                // otherwise the learner drives it. The expert labels are always recorded
                // in the dataset regardless of who is at the wheel.
                var useExpert = _effectiveBeta >= 1f || (_rng!.NextDouble() < _effectiveBeta);
                if (!useExpert)
                    ApplyDecision(agent, learnerDecision);
                // else: expert action is already applied via HandleScriptedInput above.

                if (useExpert) _expertDrivenSteps++;

                _writer!.WriteFrame(new DemonstrationFrame
                {
                    AgentSlot = slot,
                    Obs = obs,
                    DiscreteAction = expertDiscrete,
                    ContinuousActions = expertContinuous,
                    Reward = agent.LastStepReward,
                    Done = agent.IsDone,
                });
                _collectedFrameCount++;
            }

            // Tick the agent every physics step regardless of action repeat.
            agent.TickStep();
            var stepReward = agent.ConsumePendingReward();
            var stepBreakdown = agent.ConsumePendingRewardBreakdown();
            agent.AccumulateReward(stepReward, stepBreakdown.Count > 0 ? stepBreakdown : null);
            if (agent.ConsumeDonePending() || agent.HasReachedEpisodeLimit())
            {
                agent.ResetEpisode();
                _episodeCount++;
                _recurrentStates[agent] = _policy?.CreateZeroRecurrentState();
                // Reset the repeat counter so the new episode starts with a fresh decision.
                _physicsStepsSinceDecision = 0;
            }
        }

        _statusFrameCounter++;
        if (_statusFrameCounter >= StatusWriteInterval || _collectedFrameCount >= _targetAdditionalFrames)
        {
            _statusFrameCounter = 0;
            WriteStatusFile();
        }

        if (_collectedFrameCount >= _targetAdditionalFrames)
            StopCollection();
    }

    public override void _ExitTree()
    {
        _writer?.Close();
        _writer = null;
    }

    private PolicyDecision PredictDecision(IRLAgent agent, float[] observation)
    {
        if (_policy is null)
            return new PolicyDecision();

        if (_recurrentStates.TryGetValue(agent, out var recurrentState) && recurrentState is not null)
            return _policy.PredictRecurrent(observation, recurrentState);

        return _policy.Predict(observation);
    }

    private static void ApplyDecision(IRLAgent agent, PolicyDecision decision)
    {
        if (decision.DiscreteAction >= 0)
        {
            agent.ApplyAction(decision.DiscreteAction);
        }
        else if (decision.ContinuousActions.Length > 0)
        {
            agent.ApplyAction(decision.ContinuousActions);
        }
    }

    private void WriteStatusFile()
    {
        if (string.IsNullOrEmpty(_statusFilePath)) return;

        using var file = FileAccess.Open(_statusFilePath, FileAccess.ModeFlags.Write);
        if (file is null) return;

        var expertRate = _collectedFrameCount > 0
            ? (float)_expertDrivenSteps / _collectedFrameCount
            : _effectiveBeta;

        file.StoreString(Json.Stringify(new Godot.Collections.Dictionary
        {
            { "SeedFrames", _seedFrameCount },
            { "AddedFrames", _collectedFrameCount },
            { "TargetFrames", _targetAdditionalFrames },
            { "TotalFrames", (_writer?.FrameCount ?? 0) },
            { "Episodes", _episodeCount },
            { "EffectiveBeta", _effectiveBeta },
            { "ExpertDrivenRate", expertRate },
        }));
    }

    private void StopCollection()
    {
        _stopping = true;
        WriteStatusFile();
        _writer?.Close();
        _writer = null;

        using var marker = FileAccess.Open(_doneMarkerPath, FileAccess.ModeFlags.Write);
        marker?.StoreString("done");

        GD.Print($"[DAggerBootstrap] Aggregation complete — added {_collectedFrameCount} frame(s).");
        GetTree().Quit();
    }

    private static RLAcademy? FindAcademy(Node root, string nodePath)
    {
        if (!string.IsNullOrWhiteSpace(nodePath))
        {
            try
            {
                if (root.GetNode(nodePath) is RLAcademy hit) return hit;
            }
            catch (Exception) { }
        }

        return FindFirstAcademy(root);
    }

    private static RLAcademy? FindFirstAcademy(Node node)
    {
        if (node is RLAcademy academy) return academy;
        foreach (var child in node.GetChildren())
        {
            if (child is not Node childNode) continue;
            var result = FindFirstAcademy(childNode);
            if (result is not null) return result;
        }

        return null;
    }
}
