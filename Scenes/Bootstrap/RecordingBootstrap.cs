using System;
using System.Collections.Generic;
using Godot;
using RlAgentPlugin.Runtime;
using RlAgentPlugin.Runtime.Imitation;

namespace RlAgentPlugin;

/// <summary>
/// Launched by the editor via PlayCustomScene when the user starts a recording session.
/// Loads the developer's game scene, forces all target agents to Human mode,
/// and records (obs, action, reward, done) each physics step to a .rldem file.
///
/// Stop signal: the editor writes OutputFilePath + ".stop" when the user clicks Stop.
/// On receiving the stop signal this bootstrap flushes the writer and calls GetTree().Quit().
/// A ".done" marker is written after the file is finalized so the editor can confirm completion.
///
/// Capabilities file: OutputFilePath + ".cap" is written immediately after agents are discovered
/// so the editor dock can enable step mode for discrete-only agents.
///
/// Control file: OutputFilePath + ".ctrl" is written by the editor dock to change speed/pause/step.
/// Polled every ControlPollInterval physics frames (every frame in step mode).
/// </summary>
public partial class RecordingBootstrap : Node
{
    private DemonstrationWriter? _writer;
    private List<IRLAgent> _agents = new();
    private string _stopSignalPath = string.Empty;
    private string _doneMarkerPath = string.Empty;
    private string _controlFilePath = string.Empty;
    private string _statusFilePath = string.Empty;
    private bool _skipFirstFrame = true;
    private bool _stopping;
    private bool _pauseRecording;
    private bool _stepMode;
    private bool _scriptMode;
    private int _stepCount;
    private int _lastProcessedStep;
    private int _pendingStepAction;
    private bool _pendingStepRecord;
    private int _pollFrameCounter;
    private int _statusFrameCounter;
    private const int ControlPollInterval = 15; // ~0.25 s at 60 fps
    private const int StatusWriteInterval = 30; // ~0.5 s at 60 fps

    // Recording stats.
    private int _episodeCount;

    // Action repeat: only record a frame and make a new decision every N physics steps.
    private int _actionRepeat = 1;
    private int _physicsStepsSinceDecision;

    // Dimensions detected from first agent.
    private int _obsSize;
    private int _discreteCount;
    private int _contDims;

    public override void _Ready()
    {
        var manifest = RecordingLaunchManifest.LoadFromUserStorage();
        if (manifest is null)
        {
            GD.PushError("[RecordingBootstrap] Recording manifest not found. Use 'RL Imitation -> Record' to start a session.");
            GetTree().Quit(1);
            return;
        }

        if (string.IsNullOrWhiteSpace(manifest.ScenePath))
        {
            GD.PushError("[RecordingBootstrap] Manifest has no scene path.");
            GetTree().Quit(1);
            return;
        }

        var packedScene = GD.Load<PackedScene>(manifest.ScenePath);
        if (packedScene is null)
        {
            GD.PushError($"[RecordingBootstrap] Could not load scene: {manifest.ScenePath}");
            GetTree().Quit(1);
            return;
        }

        var instance = packedScene.Instantiate();
        var academy = FindAcademy(instance, manifest.AcademyNodePath);
        if (academy is null)
        {
            GD.PushError($"[RecordingBootstrap] RLAcademy not found in scene '{manifest.ScenePath}'.");
            instance.QueueFree();
            GetTree().Quit(1);
            return;
        }

        _actionRepeat = Math.Max(1, academy.ActionRepeat);

        // Collect agents before adding to tree so we can identify them.
        var allAgents = academy.GetAgents();
        _scriptMode = manifest.ScriptMode;

        foreach (var agent in allAgents)
        {
            var inGroup = string.IsNullOrEmpty(manifest.AgentGroupId)
                || agent.PolicyGroupConfig?.AgentId == manifest.AgentGroupId;

            if (!inGroup) continue;
            _agents.Add(agent);
        }

        if (_agents.Count == 0)
        {
            GD.PushError("[RecordingBootstrap] No agents matched the recording filter. Check agent group IDs.");
            instance.QueueFree();
            GetTree().Quit(1);
            return;
        }

        // Add the game scene to the tree FIRST so every node's _Ready() runs and all
        // inter-node references (e.g. player → agent) are wired up before we probe them.
        // Do NOT set ControlMode before AddChild: Academy.TryInitializeHumanMode() runs
        // during _Ready() and would activate its Human pipeline if agents are already Human,
        // causing double-driving. Agents remain Auto (scene default) when Academy caches them.
        AddChild(instance);
        Engine.TimeScale = manifest.TimeScale;

        // Now set control mode. Academy already finished _Ready() with agents in Auto mode,
        // so its Human pipeline is NOT activated. RecordingBootstrap is the sole lifecycle driver.
        foreach (var agent in _agents)
            agent.ControlMode = _scriptMode ? RLAgentControlMode.Auto : RLAgentControlMode.Human;

        // Collect obs/action dimensions AFTER _Ready() so agent sub-references are initialised.
        var spec = _agents[0].CollectObservationSpec();
        _obsSize = spec.TotalSize;
        _discreteCount = _agents[0].GetDiscreteActionCount();
        _contDims = _agents[0].GetContinuousActionDimensions();

        // Open the writer.
        _stopSignalPath = manifest.OutputFilePath + ".stop";
        _doneMarkerPath = manifest.OutputFilePath + ".done";
        _controlFilePath = manifest.OutputFilePath + ".ctrl";
        _statusFilePath = manifest.OutputFilePath + ".status";
        _writer = DemonstrationWriter.Create(manifest.OutputFilePath, _obsSize, _discreteCount, _contDims);

        if (_writer is null)
        {
            GD.PushError($"[RecordingBootstrap] Failed to create output file: {manifest.OutputFilePath}");
            GetTree().Quit(1);
            return;
        }

        // Write capabilities file so the editor dock can enable step mode.
        WriteCapabilitiesFile(manifest.OutputFilePath + ".cap");
        GD.Print($"[RecordingBootstrap] Recording {_agents.Count} agent(s) — obs={_obsSize}, disc={_discreteCount}, cont={_contDims}, speed={manifest.TimeScale}x, action_repeat={_actionRepeat}");
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_stopping) return;

        // Check for stop signal from editor.
        if (FileAccess.FileExists(_stopSignalPath))
        {
            StopRecording();
            return;
        }

        // Skip first frame: since RecordingBootstrap (parent) processes before the game scene
        // (child), we read stale initial values on frame 0. The data from frame 0's academy
        // step is what we capture on frame 1.
        if (_skipFirstFrame)
        {
            _skipFirstFrame = false;
            ReadControlFile();
            return;
        }

        // In step mode, poll every frame for responsiveness; otherwise poll periodically.
        if (_stepMode)
        {
            ReadControlFile();
            HandleStepMode();
            return;
        }

        // Poll control file periodically for speed/pause changes from the editor.
        _pollFrameCounter++;
        if (_pollFrameCounter >= ControlPollInterval)
        {
            _pollFrameCounter = 0;
            ReadControlFile();
        }

        // Write status file periodically so the editor dock can display live stats.
        _statusFrameCounter++;
        if (_statusFrameCounter >= StatusWriteInterval)
        {
            _statusFrameCounter = 0;
            WriteStatusFile();
        }

        // Pause recording: game keeps running (agent can move) but no frames are written.
        if (_pauseRecording) return;

        RecordAndTickAllAgents();
    }

    public override void _ExitTree()
    {
        _writer?.Close();
        _writer = null;
    }

    // ── Step mode ────────────────────────────────────────────────────────────

    private void HandleStepMode()
    {
        if (_pendingStepRecord)
        {
            // Player ran last frame with the injected action — record it now.
            // Step mode bypasses action repeat: each user click is one decision regardless of repeat.
            RecordAndTickAllAgents(forceDecisionStep: true);
            _pendingStepRecord = false;
            WriteStatusFile(); // immediate update so the dock reflects each step
            return;
        }

        if (_stepCount > _lastProcessedStep)
        {
            _lastProcessedStep = _stepCount;
            // Inject action: player reads PendingStepAction before keyboard next physics step.
            foreach (var agent in _agents)
                agent.PendingStepAction = _pendingStepAction;
            _pendingStepRecord = true;
        }
        // Idle frames: no recording, player applies Stay/default via keyboard (no keys held).
    }

    // ── Recording helpers ─────────────────────────────────────────────────────

    private void RecordAndTickAllAgents(bool forceDecisionStep = false)
    {
        // Advance the step counter and decide whether this is a decision point.
        // Step mode (forceDecisionStep=true) always treats the call as a decision,
        // bypassing action repeat so each user click produces exactly one frame.
        bool isDecisionStep;
        if (forceDecisionStep)
        {
            isDecisionStep = true;
            _physicsStepsSinceDecision = 0;
        }
        else
        {
            _physicsStepsSinceDecision++;
            isDecisionStep = _physicsStepsSinceDecision >= _actionRepeat;
            if (isDecisionStep)
                _physicsStepsSinceDecision = 0;
        }

        for (var slot = 0; slot < _agents.Count; slot++)
        {
            var agent = _agents[slot];

            if (isDecisionStep)
            {
                if (_scriptMode)
                {
                    // Script mode: agent drives itself via OnScriptedInput() heuristic.
                    agent.HandleScriptedInput();
                }

                // Collect fresh observations. GetLastObservation() only returns cached data;
                // CollectObservationArray() calls CollectObservations() to get the live state.
                var obs = agent.CollectObservationArray();

                _writer!.WriteFrame(new DemonstrationFrame
                {
                    AgentSlot = slot,
                    Obs = obs,
                    DiscreteAction = _discreteCount > 0 ? agent.CurrentActionIndex : -1,
                    ContinuousActions = agent.CurrentContinuousActions,
                    Reward = agent.LastStepReward,
                    Done = agent.IsDone,
                });
            }

            // Drive the agent lifecycle every physics step regardless of action repeat.
            // The RLAcademy's Human mode pipeline is not activated during recording
            // (agents start with ControlMode.Auto; academy caches this at _Ready).
            agent.TickStep();
            var stepReward = agent.ConsumePendingReward();
            var stepBreakdown = agent.ConsumePendingRewardBreakdown();
            agent.AccumulateReward(stepReward, stepBreakdown.Count > 0 ? stepBreakdown : null);
            if (agent.ConsumeDonePending() || agent.HasReachedEpisodeLimit())
            {
                agent.ResetEpisode();
                _episodeCount++;
                // Reset the repeat counter so the new episode starts with a fresh decision.
                _physicsStepsSinceDecision = 0;
            }
        }
    }

    // ── Stop ─────────────────────────────────────────────────────────────────

    private void StopRecording()
    {
        _stopping = true;
        var frameCount = _writer?.FrameCount ?? 0;
        _writer?.Close();
        _writer = null;

        GD.Print($"[RecordingBootstrap] Recording stopped — {frameCount} frames written.");

        // Write done marker so the editor knows the file is finalised.
        using var marker = FileAccess.Open(_doneMarkerPath, FileAccess.ModeFlags.Write);
        marker?.StoreString("done");

        GetTree().Quit();
    }

    // ── Control file ─────────────────────────────────────────────────────────

    private void ReadControlFile()
    {
        if (!FileAccess.FileExists(_controlFilePath)) return;

        using var file = FileAccess.Open(_controlFilePath, FileAccess.ModeFlags.Read);
        if (file is null) return;

        var text = file.GetAsText();
        if (string.IsNullOrWhiteSpace(text)) return;

        var json = new Json();
        if (json.Parse(text) != Error.Ok) return;
        var parsed = json.Data;
        if (parsed.VariantType != Variant.Type.Dictionary) return;

        var d = parsed.AsGodotDictionary();

        if (d.ContainsKey("TimeScale"))
            Engine.TimeScale = d["TimeScale"].AsDouble();
        if (d.ContainsKey("PauseRecording"))
            _pauseRecording = d["PauseRecording"].AsBool();
        if (d.ContainsKey("StepMode"))
            _stepMode = d["StepMode"].AsBool();
        if (d.ContainsKey("StepCount"))
            _stepCount = d["StepCount"].AsInt32();
        if (d.ContainsKey("StepAction"))
            _pendingStepAction = d["StepAction"].AsInt32();
    }

    // ── Status file ───────────────────────────────────────────────────────────

    private void WriteStatusFile()
    {
        if (string.IsNullOrEmpty(_statusFilePath)) return;

        var firstAgent = _agents.Count > 0 ? _agents[0] : null;

        using var file = FileAccess.Open(_statusFilePath, FileAccess.ModeFlags.Write);
        if (file is null) return;

        file.StoreString(Json.Stringify(new Godot.Collections.Dictionary
        {
            { "Frames",        _writer?.FrameCount ?? 0 },
            { "Episodes",      _episodeCount },
            { "EpisodeSteps",  firstAgent?.EpisodeSteps ?? 0 },
            { "EpisodeReward", firstAgent?.EpisodeReward ?? 0f },
            { "Paused",        _pauseRecording },
        }));
    }

    // ── Capabilities file ─────────────────────────────────────────────────────

    private void WriteCapabilitiesFile(string capPath)
    {
        var firstAgent = _agents[0];
        var labels = firstAgent.GetDiscreteActionLabels();
        var onlyDiscrete = firstAgent.SupportsOnlyDiscreteActions();


        var labelsArr = new Godot.Collections.Array();
        foreach (var lbl in labels)
            labelsArr.Add(Variant.From(lbl));

        using var file = FileAccess.Open(capPath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            GD.PushError($"[RecordingBootstrap] Failed to write .cap file: {capPath} error={FileAccess.GetOpenError()}");
            return;
        }

        file.StoreString(Json.Stringify(new Godot.Collections.Dictionary
        {
            { "OnlyDiscrete",  onlyDiscrete },
            { "DiscreteCount", _discreteCount },
            { "Labels",        labelsArr },
            { "ScriptMode",    _scriptMode },
        }));
    }

    // ── Scene helpers ─────────────────────────────────────────────────────────

    private static RLAcademy? FindAcademy(Node root, string nodePath)
    {
        if (!string.IsNullOrWhiteSpace(nodePath))
        {
            try
            {
                if (root.GetNode(nodePath) is RLAcademy hit) return hit;
            }
            catch (Exception) { /* NodePath invalid — fall through */ }
        }

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
