using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

public partial class TrainingBootstrap : Node
{
    private TrainingLaunchManifest? _manifest;
    private readonly List<RLAcademy> _academies = new();
    private readonly List<EnvironmentRuntime> _environments = new();
    private List<IRLAgent> _allTrainAgents = new();
    private RunMetricsWriter? _statusWriter;
    private readonly RandomNumberGenerator _selfPlayRng = new();

    // Per-group state
    private readonly Dictionary<string, PolicyGroupConfig> _groupConfigsByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ResolvedPolicyGroupBinding> _groupBindingsByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ITrainer> _trainersByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, RunMetricsWriter> _metricsWritersByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _episodeCountByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _workerEpisodeCountByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _updateCountByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _lastPolicyLossByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _lastValueLossByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _lastEntropyByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float?> _lastClipFractionByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _selfPlayOpponentByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _historicalOpponentRateByLearnerGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _frozenCheckpointIntervalByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, SelfPlayPairRuntime> _selfPlayPairsByKey = new(StringComparer.Ordinal);
    private readonly Dictionary<string, OpponentBankRuntime> _opponentBanksByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, IInferencePolicy> _frozenPoliciesBySnapshotKey = new(StringComparer.Ordinal);
    private readonly HashSet<string> _selfPlayParticipantGroups = new(StringComparer.Ordinal);
    private readonly Dictionary<string, EloTracker> _eloTrackersByGroup  = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _lastEpisodeRewardByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float>      _winThresholdByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, bool>       _pfspEnabledByGroup  = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float>      _pfspAlphaByGroup    = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int>        _maxPoolSizeByGroup  = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float> _quickTestRewardTotalsByGroup = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _quickTestEpisodeLengthTotalsByGroup = new(StringComparer.Ordinal);
    private readonly Queue<bool> _curriculumEpisodeOutcomes = new();

    // Per-agent state
    private readonly Dictionary<IRLAgent, AgentRuntimeState> _agentStates = new();

    private long _totalSteps;
    private RLStoppingConfig? _stoppingConfig;
    private double _trainingElapsedSeconds;
    private readonly Dictionary<string, Queue<float>> _rewardWindowByGroup = new(StringComparer.Ordinal);
    private RLTrainingConfig? _trainingConfig;
    private RLTrainerConfig?  _trainerConfig;
    private double _previousTimeScale = 1.0;
    private int _previousPhysicsTicksPerSecond = 60;
    private int _previousMaxPhysicsStepsPerFrame = 8;
    private int _checkpointInterval = 10;
    private int _actionRepeat = 1;
    private int _batchSize = 1;
    private bool _showBatchGrid;
    private bool _asyncGradientUpdates;
    private bool _parallelPolicyGroups;
    private RLAsyncRolloutPolicy _asyncRolloutPolicy = RLAsyncRolloutPolicy.Pause;
    private bool _quickTestMode;
    private int _quickTestEpisodeLimit;
    private bool _quickTestShowSpyOverlay;
    private string _finalStatus = "stopped";
    private string _finalStatusMessage = "Training ended.";
    private readonly List<SubViewport> _viewports = new();
    private int _curriculumSuccessCount;

    // Distributed training
    private bool _isWorkerMode;
    private int  _workerId;
    private int  _masterPort = 7890;
    private DistributedMaster?    _distributedMaster;
    private DistributedWorker?    _distributedWorker;
    private CanvasLayer?          _trainingOverlay;
    private RLDistributedConfig?  _distributedConfig;
    private int                   _nextWorkerId;

    // Self-play bank rescan (workers only): rate-limit disk scans to once per N steps.
    private long _lastSelfPlayBankScanStep = long.MinValue;
    private const long SelfPlayBankScanInterval = 200;

    // Smooth steps counter for the overlay — interpolates between real values to avoid
    // jarring jumps when a worker rollout arrives (e.g. +2048 steps at once).
    private double _smoothDisplaySteps;
    private double _smoothDisplayRate;

    private void SetCurriculumProgressForAllAcademies(float progress)
    {
        foreach (var academy in _academies)
            academy.SetCurriculumProgress(progress);
    }

    private void ResetAdaptiveCurriculumWindow()
    {
        _curriculumEpisodeOutcomes.Clear();
        _curriculumSuccessCount = 0;
    }

    private void UpdateAdaptiveCurriculum(RLCurriculumConfig config, float episodeReward)
    {
        var windowSize = Math.Max(1, config.SuccessWindowEpisodes);
        var wasSuccess = episodeReward >= config.SuccessRewardThreshold;
        _curriculumEpisodeOutcomes.Enqueue(wasSuccess);
        if (wasSuccess)
            _curriculumSuccessCount += 1;

        while (_curriculumEpisodeOutcomes.Count > windowSize)
        {
            if (_curriculumEpisodeOutcomes.Dequeue())
                _curriculumSuccessCount -= 1;
        }

        if (config.RequireFullWindow && _curriculumEpisodeOutcomes.Count < windowSize)
            return;

        var sampleCount = _curriculumEpisodeOutcomes.Count;
        if (sampleCount == 0)
            return;

        var successRate = _curriculumSuccessCount / (float)sampleCount;
        var currentProgress = _academies.Count > 0 ? _academies[0].CurriculumProgress : 0f;
        var nextProgress = currentProgress;

        if (successRate >= config.PromoteThreshold)
            nextProgress = Mathf.Clamp(currentProgress + config.ProgressStepUp, 0f, 1f);
        else if (successRate <= config.DemoteThreshold)
            nextProgress = Mathf.Clamp(currentProgress - config.ProgressStepDown, 0f, 1f);

        if (!Mathf.IsEqualApprox(nextProgress, currentProgress))
        {
            SetCurriculumProgressForAllAcademies(nextProgress);
            ResetAdaptiveCurriculumWindow();
        }
    }

    public override void _Ready()
    {
        _selfPlayRng.Randomize();

        // Detect distributed worker mode via custom command-line args (passed after --).
        var userArgs = OS.GetCmdlineUserArgs();
        _isWorkerMode = System.Array.Exists(userArgs, a => a == "--rl-worker");
        _workerId     = ParseDistributedIntArg(userArgs, "--worker-id", 0);
        _masterPort   = ParseDistributedIntArg(userArgs, "--master-port", 7890);

        _manifest = TrainingLaunchManifest.LoadFromUserStorage();
        if (_manifest is null)
        {
            GD.PushError("RL Agent Plugin could not load the training manifest.");
            GetTree().Quit(1);
            return;
        }

        var packedScene = GD.Load<PackedScene>(_manifest.ScenePath);
        if (packedScene is null)
        {
            GD.PushError($"Could not load training scene {_manifest.ScenePath}.");
            GetTree().Quit(1);
            return;
        }

        var firstSceneInstance = packedScene.Instantiate();
        var firstAcademy = FindNodeByPath(firstSceneInstance, _manifest.AcademyNodePath) as RLAcademy;
        if (firstAcademy is null)
        {
            GD.PushError("[RL] Could not resolve RLAcademy in the training scene.");
            firstSceneInstance.QueueFree();
            GetTree().Quit(1);
            return;
        }

        _quickTestMode = _manifest.QuickTestMode;
        _quickTestEpisodeLimit = Math.Max(1, _manifest.QuickTestEpisodeLimit);
        _quickTestShowSpyOverlay = _quickTestMode && _manifest.QuickTestShowSpyOverlay;

        _batchSize = Math.Max(1, firstAcademy.BatchSize);
        _checkpointInterval = Math.Max(1, firstAcademy.CheckpointInterval);
        _actionRepeat = Math.Max(1, firstAcademy.ActionRepeat);
        _showBatchGrid = firstAcademy.ShowBatchGrid;
        _asyncGradientUpdates = firstAcademy.AsyncGradientUpdates;
        _parallelPolicyGroups = firstAcademy.ParallelPolicyGroups;
        _asyncRolloutPolicy   = firstAcademy.AsyncRolloutPolicy;
        _stoppingConfig = firstAcademy.RunConfig?.StoppingConditions;
        if (_quickTestMode)
        {
            _batchSize = 1;
            _showBatchGrid = false;
            _asyncGradientUpdates = false;
            _parallelPolicyGroups = false;
        }

        // In distributed master mode, run only 1 game instance for live monitoring.
        // Workers handle bulk data collection; the master stays free for training coordination.
        if (!_isWorkerMode && firstAcademy.DistributedConfig is not null)
            _batchSize = 1;

        _previousTimeScale = Engine.TimeScale;
        _previousPhysicsTicksPerSecond = Engine.PhysicsTicksPerSecond;
        _previousMaxPhysicsStepsPerFrame = Engine.MaxPhysicsStepsPerFrame;
        var simulationSpeed = _quickTestMode
            ? 1.0
            : Math.Max(0.1f, _manifest.SimulationSpeed);
        var scaledPhysicsTicksPerSecond = Math.Max(
            1,
            (int)Math.Ceiling(_previousPhysicsTicksPerSecond * simulationSpeed));
        Engine.TimeScale = simulationSpeed;
        Engine.PhysicsTicksPerSecond = scaledPhysicsTicksPerSecond;
        Engine.MaxPhysicsStepsPerFrame = Math.Max(_previousMaxPhysicsStepsPerFrame, scaledPhysicsTicksPerSecond);
        GD.Print(
            $"[RL] Training speed applied: simulation_speed={simulationSpeed:0.###}, " +
            $"physics_ticks_per_second={Engine.PhysicsTicksPerSecond}, " +
            $"max_physics_steps_per_frame={Engine.MaxPhysicsStepsPerFrame}");

        var manifestBatchSize = _quickTestMode ? 1 : _manifest.BatchSize;
        if (manifestBatchSize != _batchSize
            || _manifest.CheckpointInterval != _checkpointInterval
            || _manifest.ActionRepeat != _actionRepeat)
        {
            GD.PushWarning(
                $"[RL] Training manifest settings differed from scene academy settings. " +
                $"Using scene values: batch={_batchSize}, checkpoint_interval={_checkpointInterval}, action_repeat={_actionRepeat}.");
        }

        RLTrainingConfig? trainingConfig = null;
        RLTrainerConfig? trainerConfig = null;
        var groupedAgents = new Dictionary<string, List<IRLAgent>>(StringComparer.Ordinal);

        for (var batchIdx = 0; batchIdx < _batchSize; batchIdx++)
        {
            var sceneInstance = batchIdx == 0 ? firstSceneInstance : packedScene.Instantiate();

            var viewport = new SubViewport();
            viewport.OwnWorld3D = true;
            viewport.RenderTargetUpdateMode = SubViewport.UpdateMode.Disabled;
            viewport.HandleInputLocally = false;
            viewport.AddChild(sceneInstance);
            _viewports.Add(viewport);

            var academy = batchIdx == 0
                ? firstAcademy
                : FindNodeByPath(sceneInstance, _manifest.AcademyNodePath) as RLAcademy;
            if (academy is null)
            {
                GD.PushError($"Could not resolve RLAcademy in batch copy {batchIdx}.");
                GetTree().Quit(1);
                return;
            }

            _academies.Add(academy);

            if (batchIdx == 0)
            {
                trainingConfig = academy.TrainingConfig;
                if (trainingConfig is null && !string.IsNullOrWhiteSpace(_manifest.TrainingConfigPath))
                {
                    trainingConfig = GD.Load<RLTrainingConfig>(_manifest.TrainingConfigPath);
                }

                trainerConfig = trainingConfig?.ToTrainerConfig();

                _trainingConfig = trainingConfig;
                _trainerConfig  = trainerConfig;
            }

            var environment = new EnvironmentRuntime
            {
                Index = batchIdx,
                SceneRoot = sceneInstance,
                Academy = academy,
            };

            var batchAgents = academy.GetAgents(RLAgentControlMode.Train);
            foreach (var agent in batchAgents)
            {
                var binding = RLPolicyGroupBindingResolver.Resolve(sceneInstance, agent.AsNode());
                if (binding is null)
                {
                    GD.PushError($"[RL] Agent '{agent.AsNode().Name}' has no PolicyGroupConfig assigned and will not be trained.");
                    continue;
                }

                if (!environment.AgentsByGroup.TryGetValue(binding.BindingKey, out var environmentGroup))
                {
                    environmentGroup = new List<IRLAgent>();
                    environment.AgentsByGroup[binding.BindingKey] = environmentGroup;
                }

                environmentGroup.Add(agent);

                if (!groupedAgents.TryGetValue(binding.BindingKey, out var grouped))
                {
                    grouped = new List<IRLAgent>();
                    groupedAgents[binding.BindingKey] = grouped;
                    _groupBindingsByGroup[binding.BindingKey] = binding;
                }

                grouped.Add(agent);
            }

            _environments.Add(environment);
        }

        SetupBatchDisplay();

        if (_quickTestShowSpyOverlay)
        {
            var spyOverlay = new RLAgentSpyOverlay();
            spyOverlay.Initialize(firstAcademy, includeTrainAgents: true);
            firstAcademy.AddChild(spyOverlay);
        }

        if (_academies.Count == 0)
        {
            GD.PushError("No academy instances could be created.");
            GetTree().Quit(1);
            return;
        }

        var agents = _academies.SelectMany(a => a.GetAgents(RLAgentControlMode.Train)).ToList();
        if (trainerConfig is null)
        {
            GD.PushError(
                "[RL] Missing trainer configuration. Assign RLAcademy.TrainingConfig with a non-null Algorithm, " +
                "then relaunch training.");
            GetTree().Quit(1);
            return;
        }

        if (agents.Count == 0)
        {
            GD.PushError(
                "[RL] No trainable agents found. Ensure at least one agent is set to Train or Auto control mode " +
                "and has a PolicyGroupConfig binding.");
            GetTree().Quit(1);
            return;
        }

        var algorithm = trainerConfig.Algorithm;
        var customTrainerId = trainerConfig.CustomTrainerId;
        foreach (var (groupId, groupAgents) in groupedAgents)
        {
            var binding = _groupBindingsByGroup[groupId];
            var firstAgent = groupAgents[0];
            if (!ObservationSizeInference.TryInferAgentObservationSize(firstAgent, out var obsSize, out var observationError))
            {
                GD.PushError($"[RL] Group '{groupId}': observation inference failed: {observationError}");
                GetTree().Quit(1);
                return;
            }

            var discreteCount = firstAgent.GetDiscreteActionCount();
            var continuousDims = firstAgent.GetContinuousActionDimensions();
            var actionDefinitions = firstAgent.GetActionSpace();

            if (algorithm == RLAlgorithmKind.PPO && discreteCount <= 0 && continuousDims <= 0)
            {
                GD.PushError($"[RL] Group '{groupId}': PPO requires at least one discrete or continuous action.");
                GetTree().Quit(1);
                return;
            }

            if (algorithm == RLAlgorithmKind.PPO && discreteCount > 0 && continuousDims > 0)
            {
                GD.PushError($"[RL] Group '{groupId}': PPO does not support mixing discrete and continuous actions.");
                GetTree().Quit(1);
                return;
            }

            if (algorithm == RLAlgorithmKind.SAC && discreteCount > 0 && continuousDims > 0)
            {
                GD.PushError($"[RL] Group '{groupId}': SAC does not support mixing discrete and continuous actions.");
                GetTree().Quit(1);
                return;
            }

            // Custom trainers declare their own action-space constraints; skip built-in checks.
            if (algorithm == RLAlgorithmKind.Custom && string.IsNullOrWhiteSpace(customTrainerId))
            {
                GD.PushError($"[RL] Group '{groupId}': Algorithm is Custom but CustomTrainerId is not set.");
                GetTree().Quit(1);
                return;
            }

            foreach (var agent in groupAgents)
            {
                if (!ObservationSizeInference.TryInferAgentObservationSize(agent, out var agentObservationSize, out observationError))
                {
                    GD.PushError($"[RL] Group '{groupId}': observation inference failed for '{agent.AsNode().Name}': {observationError}");
                    GetTree().Quit(1);
                    return;
                }

                if (agentObservationSize != obsSize)
                {
                    GD.PushError($"[RL] Group '{groupId}': all agents must emit the same observation vector length.");
                    GetTree().Quit(1);
                    return;
                }

                if (algorithm == RLAlgorithmKind.PPO && discreteCount > 0 && agent.GetDiscreteActionCount() != discreteCount)
                {
                    GD.PushError($"[RL] Group '{groupId}': all PPO agents must have the same discrete action count.");
                    GetTree().Quit(1);
                    return;
                }

                if (algorithm == RLAlgorithmKind.PPO && continuousDims > 0 && agent.GetContinuousActionDimensions() != continuousDims)
                {
                    GD.PushError($"[RL] Group '{groupId}': all PPO agents must have the same continuous action dimensions.");
                    GetTree().Quit(1);
                    return;
                }
            }

            var safeGroupId = binding.SafeGroupId;
            var checkpointPath = $"{_manifest.RunDirectory}/checkpoint__{safeGroupId}.json";
            var metricsPath = $"{_manifest.RunDirectory}/metrics__{safeGroupId}.jsonl";

            var groupConfig = new PolicyGroupConfig
            {
                GroupId = groupId,
                RunId = _manifest.RunId,
                Algorithm = algorithm,
                CustomTrainerId = customTrainerId,
                SharedPolicy = binding.Config,
                TrainerConfig = trainerConfig,
                NetworkGraph = binding.Config?.ResolvedNetworkGraph ?? RLNetworkGraph.CreateDefault(),
                ActionDefinitions = actionDefinitions,
                ObservationSize = obsSize,
                DiscreteActionCount = discreteCount,
                ContinuousActionDimensions = continuousDims,
                CheckpointPath = checkpointPath,
                MetricsPath = metricsPath,
            };

            _groupConfigsByGroup[groupId] = groupConfig;
            _trainersByGroup[groupId] = TrainerFactory.Create(groupConfig);
            _metricsWritersByGroup[groupId] = new RunMetricsWriter(metricsPath, _manifest.StatusPath);
            _episodeCountByGroup[groupId] = 0;
            _workerEpisodeCountByGroup[groupId] = 0;
            _updateCountByGroup[groupId] = 0;
            _lastPolicyLossByGroup[groupId] = 0f;
            _lastValueLossByGroup[groupId] = 0f;
            _lastEntropyByGroup[groupId] = 0f;
            _lastClipFractionByGroup[groupId] = algorithm == RLAlgorithmKind.PPO ? 0f : null;

            GD.Print($"[RL] Group '{binding.DisplayName}': {groupAgents.Count} agent(s), {algorithm}, obs={obsSize}, discrete={discreteCount}, continuous={continuousDims}");
            GD.Print($"[RL]   Checkpoint: {checkpointPath}");
            GD.Print($"[RL]   Metrics:    {metricsPath}");
        }

        // Inform SAC trainers how many parallel envs are running per process so the
        // auto UTD formula can scale gradient updates to match the full data rate.
        foreach (var trainer in _trainersByGroup.Values)
            if (trainer is SacTrainer sac) sac.EnvBatchSize = _batchSize;

        if (!TryConfigureSelfPlay(out var selfPlayError))
        {
            GD.PushError($"[RL] {selfPlayError}");
            GetTree().Quit(1);
            return;
        }

        ConfigureEnvironmentRoles();
        InitializeOpponentBanks();
        if (!_isWorkerMode) SaveInitialCheckpoints();

        // ── Distributed training setup ────────────────────────────────────────
        var distributedTrainers = _trainersByGroup
            .Where(kvp => kvp.Value is IDistributedTrainer)
            .ToDictionary(kvp => kvp.Key, kvp => (IDistributedTrainer)kvp.Value, StringComparer.Ordinal);

        if (distributedTrainers.Count > 0)
        {
            if (_isWorkerMode)
            {
                _distributedWorker = new DistributedWorker("127.0.0.1", _masterPort, distributedTrainers);
                _distributedWorker.Connect();
                // Override simulation speed for workers.
                if (firstAcademy.DistributedConfig is { } wCfg)
                    ApplySimulationSpeed(wCfg.WorkerSimulationSpeed);
                GD.Print($"[RL Distributed] Running as worker {_workerId}.");
            }
            else if (firstAcademy.DistributedConfig is { } distCfg)
            {
                _distributedMaster = new DistributedMaster(
                    distCfg.MasterPort,
                    distCfg.WorkerCount,
                    distCfg.MonitorIntervalUpdates,
                    distCfg.VerboseLog,
                    distributedTrainers,
                    _asyncRolloutPolicy,
                    _trainerConfig?.RolloutLength ?? 256);
                _distributedMaster.Start();
                _distributedConfig = distCfg;
                _nextWorkerId      = distCfg.WorkerCount;
                if (distCfg.AutoLaunchWorkers)
                    LaunchDistributedWorkers(distCfg);
                if (distCfg.ShowTrainingOverlay)
                    _trainingOverlay = CreateTrainingOverlay();
                GD.Print($"[RL Distributed] Running as master on port {distCfg.MasterPort}.");
            }
        }
        // ─────────────────────────────────────────────────────────────────────

        foreach (var environment in _environments)
        {
            // In quick-test mode, seed the curriculum to the debug value so OnEpisodeBegin sees the right difficulty.
            if (_quickTestMode && environment.Academy.DebugCurriculumProgress > 0f)
                environment.Academy.SetCurriculumProgress(environment.Academy.DebugCurriculumProgress);
            else if (!_quickTestMode)
                environment.Academy.SetCurriculumProgress(0f);

            if (!TryInitializeEnvironment(environment, out var initializationError))
            {
                GD.PushError($"[RL] {initializationError}");
                GetTree().Quit(1);
                return;
            }
        }

        _allTrainAgents = _academies.SelectMany(a => a.GetAgents(RLAgentControlMode.Train)).ToList();

        _statusWriter = new RunMetricsWriter(string.Empty, _manifest.StatusPath);
        // Workers must not write to the shared status/metrics files — they use the same
        // manifest paths as the master, and would overwrite correct data with worker-local
        // (wrong) values. Only the master owns these files.
        if (!_isWorkerMode)
        {
            _statusWriter.WriteStatus(
                "running",
                _manifest.ScenePath,
                _totalSteps,
                0,
                _quickTestMode
                    ? $"Quick test running ({_quickTestEpisodeLimit} episode target)."
                    : "Training started.");
        }
        GD.Print($"[RL] Run: {_manifest.RunId}");
        if (_quickTestMode)
        {
            GD.Print($"[RL] Quick test mode: stopping after {_quickTestEpisodeLimit} completed episode(s).");
        }
        else if (_batchSize > 1)
        {
            GD.Print($"[RL] Batch size: {_batchSize} ({_allTrainAgents.Count} total agents)");
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_academies.Count == 0 || _manifest is null || _statusWriter is null)
        {
            return;
        }

        if (!_isWorkerMode)
            _trainingElapsedSeconds += delta;

        // Workers receive curriculum progress from master; apply it before episode processing
        // so OnEpisodeBegin sees the correct difficulty this frame.
        if (_distributedWorker is not null)
        {
            var syncedProgress = _distributedWorker.ConsumeCurriculumProgress();
            if (syncedProgress.HasValue)
                SetCurriculumProgressForAllAcademies(syncedProgress.Value);
        }

        // Master drains worker episode summaries and writes them to the metrics log so
        // RLDash shows episodes from all processes, not just the master's single game.
        if (_distributedMaster is not null && !_quickTestMode)
        {
            foreach (var (groupId, summaries) in _distributedMaster.DrainWorkerEpisodeSummaries())
            {
                if (!_metricsWritersByGroup.TryGetValue(groupId, out var writer)) continue;
                _workerEpisodeCountByGroup[groupId] = _workerEpisodeCountByGroup.GetValueOrDefault(groupId) + summaries.Count;
                var metricSteps     = _totalSteps + _distributedMaster.TotalWorkerSteps;
                var localEpisodes   = _episodeCountByGroup.GetValueOrDefault(groupId);
                var workerEpisodes  = _workerEpisodeCountByGroup[groupId];
                foreach (var s in summaries)
                {
                    writer.AppendMetric(
                        s.Reward,
                        s.Steps,
                        _lastPolicyLossByGroup.GetValueOrDefault(groupId),
                        _lastValueLossByGroup.GetValueOrDefault(groupId),
                        _lastEntropyByGroup.GetValueOrDefault(groupId),
                        _lastClipFractionByGroup.GetValueOrDefault(groupId),
                        metricSteps,
                        localEpisodes + workerEpisodes,
                        s.RewardBreakdown,
                        policyGroup: GetGroupDisplayName(groupId));
                }
            }
        }

        var pendingLearningDecisionsByGroup = new Dictionary<string, List<PendingDecisionContext>>(StringComparer.Ordinal);
        var pendingFrozenDecisions = new List<PendingDecisionContext>();

        foreach (var agent in _allTrainAgents)
        {
            if (!_agentStates.TryGetValue(agent, out var state))
            {
                continue;
            }

            agent.TickStep();
            var reward = agent.ConsumePendingReward();
            var rewardBreakdown = agent.ConsumePendingRewardBreakdown();
            agent.AccumulateReward(reward, rewardBreakdown);
            state.WindowReward += reward;
            state.StepsSinceDecision++;

            var done = agent.ConsumeDonePending() || agent.HasReachedEpisodeLimit();
            if (done || state.StepsSinceDecision >= _actionRepeat)
            {
                var nextObservation = agent.CollectObservationArray();
                var pending = new PendingDecisionContext
                {
                    Agent = agent,
                    State = state,
                    TransitionObservation = nextObservation,
                    Done = done,
                };

                var role = GetEnvironmentRole(state.EnvironmentIndex, state.GroupId);
                if (role.Control == EnvironmentGroupControl.FrozenOpponent)
                {
                    pendingFrozenDecisions.Add(pending);
                }
                else
                {
                    if (!pendingLearningDecisionsByGroup.TryGetValue(state.GroupId, out var pendingGroup))
                    {
                        pendingGroup = new List<PendingDecisionContext>();
                        pendingLearningDecisionsByGroup[state.GroupId] = pendingGroup;
                    }

                    pendingGroup.Add(pending);
                }
            }
            else
            {
                ReapplyAction(agent, state);
            }
        }

        if (_parallelPolicyGroups && pendingLearningDecisionsByGroup.Count > 1)
        {
            RunParallelGroupDecisions(pendingLearningDecisionsByGroup);
        }
        else
        {
            foreach (var (groupId, pendingDecisions) in pendingLearningDecisionsByGroup)
            {
                if (_trainersByGroup.TryGetValue(groupId, out var trainer))
                    ProcessLearningDecisions(groupId, trainer, pendingDecisions);
            }
        }

        if (pendingFrozenDecisions.Count > 0)
        {
            ProcessFrozenDecisions(pendingFrozenDecisions);
        }

        RLCheckpoint? lastCheckpoint = null;
        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var episodeCount = _episodeCountByGroup[groupId];

            if (_trainingConfig is not null && _trainerConfig is not null)
            {
                _trainingConfig.ApplySchedules(_trainerConfig, new ScheduleContext
                {
                    UpdateCount  = _updateCountByGroup[groupId],
                    TotalSteps   = _totalSteps,
                    EpisodeCount = episodeCount,
                });
            }

            TrainerUpdateStats? updateStats;
            if (_distributedWorker is not null && trainer is IDistributedTrainer workerDt)
            {
                // Worker: send rollout to master instead of training locally.
                updateStats = _distributedWorker.TickUpdate(workerDt, groupId, _totalSteps, episodeCount);
            }
            else if (_distributedMaster is not null && trainer is IDistributedTrainer masterDt)
            {
                // Master: wait for all workers, then train on the combined buffer.
                updateStats = _distributedMaster.TickUpdate(masterDt, groupId, _totalSteps, episodeCount);
            }
            else if (_asyncGradientUpdates && trainer is IAsyncTrainer asyncTrainer)
            {
                // Poll for any result completed since last frame, then schedule the next job.
                updateStats = asyncTrainer.TryPollResult(groupId, _totalSteps, episodeCount);
                // Cap the snapshot to one rollout so transitions that accumulated during
                // training do not inflate the next batch (Pause and Cap both cap here;
                // there are no workers to discard from in the single-process async path).
                var maxT = _trainerConfig?.RolloutLength ?? int.MaxValue;
                asyncTrainer.TryScheduleBackgroundUpdate(groupId, _totalSteps, episodeCount, maxT);
            }
            else
            {
                updateStats = trainer.TryUpdate(groupId, _totalSteps, episodeCount);
            }

            if (updateStats is null)
            {
                continue;
            }

            _updateCountByGroup[groupId] += 1;
            _lastPolicyLossByGroup[groupId] = updateStats.PolicyLoss;
            _lastValueLossByGroup[groupId] = updateStats.ValueLoss;
            _lastEntropyByGroup[groupId] = updateStats.Entropy;
            _lastClipFractionByGroup[groupId] = updateStats.ClipFraction;

            var currentCheckpoint = trainer.CreateCheckpoint(groupId, _totalSteps, episodeCount, _updateCountByGroup[groupId]);
            currentCheckpoint.RewardSnapshot = _lastEpisodeRewardByGroup.GetValueOrDefault(groupId);
            lastCheckpoint = currentCheckpoint;

            PersistCheckpoint(groupId, currentCheckpoint, _updateCountByGroup[groupId]);

            // Broadcast curriculum progress alongside each weight update so workers stay in sync.
            if (_distributedMaster is not null && _academies.Count > 0 && _academies[0].Curriculum is not null)
                _distributedMaster.BroadcastCurriculumProgress(_academies[0].CurriculumProgress);
        }

        if (lastCheckpoint is not null)
        {
            foreach (var academy in _academies)
            {
                academy.Checkpoint = lastCheckpoint;
            }
        }

        var trainerConfig = _trainerConfig;
        if (!_isWorkerMode && trainerConfig is not null && _totalSteps % Math.Max(1, trainerConfig.StatusWriteIntervalSteps) == 0)
        {
            var totalEpisodes = _episodeCountByGroup.Values.Sum();
            var statusMessage = _quickTestMode
                ? $"Quick test running: {totalEpisodes}/{_quickTestEpisodeLimit} episodes complete."
                : $"Training update {_updateCountByGroup.Values.Sum()}";
            var reportedSteps    = _totalSteps + (_distributedMaster?.TotalWorkerSteps ?? 0L);
            var workerEpisodes   = _distributedMaster?.TotalWorkerEpisodes ?? 0L;
            _statusWriter.WriteStatus("running", _manifest.ScenePath, reportedSteps, totalEpisodes, statusMessage,
                workerEpisodes);
        }

        if (_quickTestMode)
        {
            var totalEpisodes = _episodeCountByGroup.Values.Sum();
            if (totalEpisodes >= _quickTestEpisodeLimit)
            {
                CompleteQuickTest();
            }
        }

        if (!_quickTestMode && !_isWorkerMode && ShouldStopTraining())
        {
            TriggerStoppingConditionShutdown();
        }
    }

    public override void _ExitTree()
    {
        Engine.TimeScale = _previousTimeScale;
        Engine.PhysicsTicksPerSecond = _previousPhysicsTicksPerSecond;
        Engine.MaxPhysicsStepsPerFrame = _previousMaxPhysicsStepsPerFrame;

        _distributedMaster?.Shutdown();
        _distributedMaster?.Dispose();
        _distributedWorker?.Dispose();

        // Workers do not own checkpoints — only the master persists them.
        if (_isWorkerMode || _manifest is null)
            return;

        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var episodeCount = _episodeCountByGroup.GetValueOrDefault(groupId);
            var updateCount = _updateCountByGroup.GetValueOrDefault(groupId);

            // If an async gradient update is in flight, wait for it and apply the result so the
            // final checkpoint reflects the most recent trained weights.
            if (_asyncGradientUpdates && trainer is IAsyncTrainer asyncTrainer)
                asyncTrainer.FlushPendingUpdate(groupId, _totalSteps, episodeCount);

            var finalCheckpoint = trainer.CreateCheckpoint(groupId, _totalSteps, episodeCount, updateCount);
            PersistCheckpoint(groupId, finalCheckpoint, updateCount, forceLatestWrite: true, allowFrozenSnapshot: false);

            foreach (var academy in _academies)
            {
                academy.Checkpoint = finalCheckpoint;
            }
        }

        if (!_isWorkerMode)
        {
            var totalEpisodes      = _episodeCountByGroup.Values.Sum();
            var finalReportedSteps = _totalSteps + (_distributedMaster?.TotalWorkerSteps ?? 0L);
            var workerEpisodes     = _distributedMaster?.TotalWorkerEpisodes ?? 0L;
            _statusWriter?.WriteStatus(_finalStatus, _manifest.ScenePath, finalReportedSteps, totalEpisodes,
                _finalStatusMessage, workerEpisodes);
        }
    }

    private void TriggerStoppingConditionShutdown()
    {
        var totalEpisodes = _episodeCountByGroup.Values.Sum();
        _finalStatus = "done";
        _finalStatusMessage =
            $"Training stopped by stopping condition — steps: {_totalSteps}, " +
            $"episodes: {totalEpisodes}, elapsed: {_trainingElapsedSeconds:0.#}s.";
        GD.Print($"[RL] {_finalStatusMessage}");
        GetTree().Quit();
    }

    private bool ShouldStopTraining()
    {
        var cfg = _stoppingConfig;
        if (cfg is null) return false;

        var totalEpisodes = _episodeCountByGroup.Values.Sum();
        var results = new System.Collections.Generic.List<bool>();

        if (cfg.MaxSteps > 0)
            results.Add(_totalSteps >= cfg.MaxSteps);
        if (cfg.MaxSeconds > 0.0)
            results.Add(_trainingElapsedSeconds >= cfg.MaxSeconds);
        if (cfg.MaxEpisodes > 0)
            results.Add(totalEpisodes >= cfg.MaxEpisodes);
        if (cfg.RewardThresholdWindow > 0)
        {
            var avg = ComputeRollingReward(cfg);
            if (!float.IsNaN(avg))
                results.Add(avg >= cfg.RewardThreshold);
        }
        if (cfg.CustomCondition is not null)
        {
            var avg = cfg.RewardThresholdWindow > 0 ? ComputeRollingReward(cfg) : float.NaN;
            results.Add(cfg.CustomCondition.ShouldStop(_totalSteps, totalEpisodes, _trainingElapsedSeconds, avg));
        }

        if (results.Count == 0) return false;
        return cfg.CombineMode == RLStoppingCombineMode.All
            ? results.TrueForAll(r => r)
            : results.Exists(r => r);
    }

    private float ComputeRollingReward(RLStoppingConfig cfg)
    {
        IEnumerable<Queue<float>> queues;
        if (!string.IsNullOrEmpty(cfg.RewardThresholdGroup) &&
            _rewardWindowByGroup.TryGetValue(cfg.RewardThresholdGroup, out var single))
            queues = new[] { single };
        else
            queues = _rewardWindowByGroup.Values;

        var all = new System.Collections.Generic.List<float>();
        foreach (var q in queues)
            foreach (var v in q)
                all.Add(v);

        var minSize = string.IsNullOrEmpty(cfg.RewardThresholdGroup)
            ? cfg.RewardThresholdWindow * Math.Max(1, _rewardWindowByGroup.Count)
            : cfg.RewardThresholdWindow;

        if (all.Count < minSize) return float.NaN;
        var sum = 0f;
        foreach (var v in all) sum += v;
        return sum / all.Count;
    }

    private void CompleteQuickTest()
    {
        if (!_quickTestMode)
        {
            return;
        }

        _finalStatus = "done";
        _finalStatusMessage = BuildQuickTestSummary();
        GD.Print($"[RL] {_finalStatusMessage}");
        GetTree().Quit();
    }

    private string BuildQuickTestSummary()
    {
        var totalEpisodes = _episodeCountByGroup.Values.Sum();
        var summaries = new List<string>();
        foreach (var (groupId, episodeCount) in _episodeCountByGroup.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            if (episodeCount <= 0)
            {
                continue;
            }

            var totalReward = _quickTestRewardTotalsByGroup.GetValueOrDefault(groupId);
            var totalLength = _quickTestEpisodeLengthTotalsByGroup.GetValueOrDefault(groupId);
            var displayName = GetGroupDisplayName(groupId);
            summaries.Add(
                $"{displayName}: avg reward {(totalReward / episodeCount):0.###}, avg length {(totalLength / (float)episodeCount):0.#}");
        }

        var suffix = summaries.Count > 0
            ? $" {string.Join(" | ", summaries)}"
            : string.Empty;
        return $"Quick test complete: {totalEpisodes} episode(s), {_totalSteps} step(s).{suffix}";
    }

    private bool TryConfigureSelfPlay(out string error)
    {
        error = string.Empty;
        _selfPlayOpponentByGroup.Clear();
        _historicalOpponentRateByLearnerGroup.Clear();
        _frozenCheckpointIntervalByGroup.Clear();
        _selfPlayPairsByKey.Clear();
        _selfPlayParticipantGroups.Clear();
        _winThresholdByGroup.Clear();
        _pfspEnabledByGroup.Clear();
        _pfspAlphaByGroup.Clear();
        _maxPoolSizeByGroup.Clear();
        _eloTrackersByGroup.Clear();

        if (_quickTestMode)
        {
            GD.Print("[RL] Quick test mode: skipping self-play setup — all agents will train independently.");
            return true;
        }

        var configuredPairings = GetConfiguredSelfPlayPairings();
        return configuredPairings.Count == 0
            ? true
            : TryConfigureSelfPlayFromPairings(configuredPairings, out error);
    }

    private bool TryConfigureSelfPlayFromPairings(IReadOnlyList<RLPolicyPairingConfig> configuredPairings, out string error)
    {
        error = string.Empty;
        foreach (var pairing in configuredPairings)
        {
            var groupAId = pairing.ResolvedGroupA is null ? string.Empty : ResolveGroupIdForConfig(pairing.ResolvedGroupA);
            var groupBId = pairing.ResolvedGroupB is null ? string.Empty : ResolveGroupIdForConfig(pairing.ResolvedGroupB);

            if (string.IsNullOrWhiteSpace(groupAId) || string.IsNullOrWhiteSpace(groupBId))
            {
                error = $"Pairing '{ResolvePairingDisplayName(pairing)}' must reference two policy groups used by train-mode agents in the scene.";
                return false;
            }

            if (string.Equals(groupAId, groupBId, StringComparison.Ordinal))
            {
                error = $"Pairing '{ResolvePairingDisplayName(pairing)}' cannot use the same group for both sides.";
                return false;
            }

            if (!pairing.TrainGroupA && !pairing.TrainGroupB)
            {
                error = $"Pairing '{ResolvePairingDisplayName(pairing)}' must train at least one side.";
                return false;
            }

            if (_selfPlayParticipantGroups.Contains(groupAId) || _selfPlayParticipantGroups.Contains(groupBId))
            {
                error =
                    $"Groups '{GetGroupDisplayName(groupAId)}' and '{GetGroupDisplayName(groupBId)}' already belong to another self-play pairing. " +
                    "v1 supports disjoint 2-group pairings only.";
                return false;
            }

            var pairKey = MakePairKey(groupAId, groupBId);
            _selfPlayPairsByKey[pairKey] = new SelfPlayPairRuntime
            {
                PairKey = pairKey,
                GroupA = groupAId,
                GroupB = groupBId,
                TrainGroupA = pairing.TrainGroupA,
                TrainGroupB = pairing.TrainGroupB,
            };

            _selfPlayParticipantGroups.Add(groupAId);
            _selfPlayParticipantGroups.Add(groupBId);
            _frozenCheckpointIntervalByGroup[groupAId] = Math.Max(1, pairing.FrozenCheckpointInterval);
            _frozenCheckpointIntervalByGroup[groupBId] = Math.Max(1, pairing.FrozenCheckpointInterval);

            _maxPoolSizeByGroup[groupAId]  = pairing.MaxPoolSize;
            _maxPoolSizeByGroup[groupBId]  = pairing.MaxPoolSize;
            _pfspEnabledByGroup[groupAId]  = pairing.PfspEnabled;
            _pfspEnabledByGroup[groupBId]  = pairing.PfspEnabled;
            _pfspAlphaByGroup[groupAId]    = pairing.PfspAlpha;
            _pfspAlphaByGroup[groupBId]    = pairing.PfspAlpha;

            if (pairing.TrainGroupA)
            {
                _selfPlayOpponentByGroup[groupAId] = groupBId;
                _historicalOpponentRateByLearnerGroup[groupAId] = Mathf.Clamp(pairing.HistoricalOpponentRate, 0f, 1f);
                _winThresholdByGroup[groupAId] = pairing.WinThreshold;
            }

            if (pairing.TrainGroupB)
            {
                _selfPlayOpponentByGroup[groupBId] = groupAId;
                _historicalOpponentRateByLearnerGroup[groupBId] = Mathf.Clamp(pairing.HistoricalOpponentRate, 0f, 1f);
                _winThresholdByGroup[groupBId] = pairing.WinThreshold;
            }
        }

        return ValidateSelfPlayBatchSize(out error);
    }

    private bool ValidateSelfPlayBatchSize(out string error)
    {
        error = string.Empty;
        var requiredBatchCopies = _selfPlayPairsByKey.Values.Count == 0
            ? 1
            : _selfPlayPairsByKey.Values.Max(pair => pair.LearnerCount);
        if (_batchSize < requiredBatchCopies)
        {
            error = _quickTestMode
                ? $"Quick test does not support the configured self-play setup because it forces BatchSize = 1, " +
                  $"but self-play requires at least {requiredBatchCopies} batch copies."
                : $"Self-play requires at least {requiredBatchCopies} batch copies for the configured rival groups, " +
                  $"but BatchSize is {_batchSize}.";
            return false;
        }

        return true;
    }

    private void ConfigureEnvironmentRoles()
    {
        foreach (var environment in _environments)
        {
            foreach (var groupId in environment.AgentsByGroup.Keys)
            {
                environment.GroupRoles[groupId] = new EnvironmentGroupRole
                {
                    GroupId = groupId,
                    Control = EnvironmentGroupControl.LiveTrainer,
                };
            }

            foreach (var pair in _selfPlayPairsByKey.Values)
            {
                var learnerGroupId = pair.GetLearnerForEnvironment(environment.Index);
                var frozenGroupId = pair.GetOpponentForLearner(learnerGroupId);

                if (!environment.GroupRoles.TryGetValue(learnerGroupId, out var learnerRole)
                    || !environment.GroupRoles.TryGetValue(frozenGroupId, out var frozenRole))
                {
                    continue;
                }

                learnerRole.IsSelfPlayLearner = true;
                learnerRole.OpponentGroupId = frozenGroupId;
                learnerRole.HistoricalOpponentRate = _historicalOpponentRateByLearnerGroup.GetValueOrDefault(learnerGroupId);

                frozenRole.Control = EnvironmentGroupControl.FrozenOpponent;
                frozenRole.LearnerGroupId = learnerGroupId;
            }

            environment.NeedsMatchupRefresh = environment.GroupRoles.Values.Any(role => role.IsSelfPlayLearner);
        }
    }

    private void InitializeOpponentBanks()
    {
        foreach (var groupId in _selfPlayParticipantGroups)
        {
            var maxPool     = _maxPoolSizeByGroup.GetValueOrDefault(groupId, 20);
            var pfspEnabled = _pfspEnabledByGroup.GetValueOrDefault(groupId, true);
            var pfspAlpha   = _pfspAlphaByGroup.GetValueOrDefault(groupId, 4.0f);
            var pool        = new PolicyPool(maxPool, pfspEnabled, pfspAlpha, _selfPlayRng)
            {
                LatestCheckpointPath = GetGroupCheckpointPath(groupId) ?? string.Empty,
            };

            // Re-hydrate historical snapshots from disk (win/loss state starts fresh at Laplace prior).
            var directory = GetSelfPlayBankDirectory(groupId);
            var dir       = DirAccess.Open(directory);
            if (dir is not null)
            {
                var entries = new List<string>();
                dir.ListDirBegin();
                while (true)
                {
                    var entry = dir.GetNext();
                    if (string.IsNullOrEmpty(entry)) break;
                    if (dir.CurrentIsDir() || entry.StartsWith(".") || !entry.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                        continue;
                    entries.Add(entry);
                }
                dir.ListDirEnd();
                entries.Sort(StringComparer.Ordinal);

                foreach (var entry in entries)
                {
                    var filePath    = $"{directory}/{entry}";
                    var snapshotKey = ExtractSnapshotKeyFromFileName(entry, filePath);
                    pool.AddSnapshot(filePath, snapshotKey, EloTracker.InitialRating);
                }
            }

            _opponentBanksByGroup[groupId] = new OpponentBankRuntime
            {
                GroupId = groupId,
                Pool    = pool,
            };
            _eloTrackersByGroup[groupId] = new EloTracker();
        }
    }

    /// <summary>
    /// Derives a snapshot key from a historical checkpoint filename.
    /// Expected format: <c>opponent__u{updateCount:D6}.json</c>
    /// </summary>
    private static string ExtractSnapshotKeyFromFileName(string fileName, string filePath)
    {
        const string prefix = "opponent__u";
        const string suffix = ".json";
        if (fileName.StartsWith(prefix, StringComparison.Ordinal)
            && fileName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
        {
            var countStr = fileName[prefix.Length..^suffix.Length];
            if (long.TryParse(countStr, out var updateCount))
                return $"{filePath}::{updateCount}";
        }

        return $"{filePath}::0";
    }

    /// <summary>
    /// Rescans the self-play bank directory for a group and registers any newly-written snapshot
    /// files that the master has persisted since the last scan. Safe to call repeatedly —
    /// <see cref="PolicyPool.AddSnapshot"/> is a no-op for keys that already exist.
    /// Used by workers, which never call <see cref="PersistCheckpoint"/> themselves.
    /// </summary>
    private void RescanOpponentBankDirectory(string groupId)
    {
        if (!_opponentBanksByGroup.TryGetValue(groupId, out var bank)) return;
        var directory = GetSelfPlayBankDirectory(groupId);
        var dir = DirAccess.Open(directory);
        if (dir is null) return;

        dir.ListDirBegin();
        while (true)
        {
            var entry = dir.GetNext();
            if (string.IsNullOrEmpty(entry)) break;
            if (dir.CurrentIsDir() || entry.StartsWith(".") || !entry.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                continue;
            var filePath    = $"{directory}/{entry}";
            var snapshotKey = ExtractSnapshotKeyFromFileName(entry, filePath);
            bank.Pool.AddSnapshot(filePath, snapshotKey, EloTracker.InitialRating);
        }
        dir.ListDirEnd();
    }

    private void SaveInitialCheckpoints()
    {
        if (_quickTestMode) return;

        foreach (var (groupId, trainer) in _trainersByGroup)
        {
            var checkpoint = trainer.CreateCheckpoint(groupId, _totalSteps, _episodeCountByGroup[groupId], _updateCountByGroup[groupId]);
            PersistCheckpoint(groupId, checkpoint, _updateCountByGroup[groupId], forceLatestWrite: true, allowFrozenSnapshot: false);
        }
    }

    private bool TryInitializeEnvironment(EnvironmentRuntime environment, out string error)
    {
        error = string.Empty;

        foreach (var groupAgents in environment.AgentsByGroup.Values)
        {
            foreach (var agent in groupAgents)
            {
                agent.ResetEpisode();
            }
        }

        if (!TryRefreshEnvironmentMatchups(environment, out error))
        {
            return false;
        }

        var entropyByGroup = new Dictionary<string, float>(StringComparer.Ordinal);
        var decisionCountByGroup = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var (groupId, groupAgents) in environment.AgentsByGroup)
        {
            var role = GetEnvironmentRole(environment.Index, groupId);
            var observations = new List<float[]>(groupAgents.Count);
            foreach (var agent in groupAgents)
            {
                observations.Add(agent.CollectObservationArray());
            }

            if (role.Control == EnvironmentGroupControl.FrozenOpponent)
            {
                if (role.FrozenPolicy is null)
                {
                    error = $"Environment {environment.Index}: frozen opponent policy for group '{GetGroupDisplayName(groupId)}' was not prepared.";
                    return false;
                }

                for (var index = 0; index < groupAgents.Count; index++)
                {
                    var agent = groupAgents[index];
                    var observation = observations[index];
                    var decision = role.FrozenPolicy.Predict(observation);
                    _agentStates[agent] = CreateAgentState(groupId, environment.Index, observation, decision, isLearningEnabled: false);
                    ApplyDecision(agent, decision);
                }

                continue;
            }

            if (!_trainersByGroup.TryGetValue(groupId, out var trainer))
            {
                error = $"Environment {environment.Index}: no trainer exists for group '{GetGroupDisplayName(groupId)}'.";
                return false;
            }

            var decisions = trainer.SampleActions(VectorBatch.FromRows(observations));
            var entropySum = 0f;
            for (var index = 0; index < groupAgents.Count; index++)
            {
                var agent = groupAgents[index];
                var observation = observations[index];
                var decision = decisions[index];
                entropySum += decision.Entropy;
                _agentStates[agent] = CreateAgentState(groupId, environment.Index, observation, decision, isLearningEnabled: true);
                ApplyDecision(agent, decision);
            }

            entropyByGroup.TryGetValue(groupId, out var currentEntropy);
            entropyByGroup[groupId] = currentEntropy + entropySum;
            decisionCountByGroup.TryGetValue(groupId, out var currentCount);
            decisionCountByGroup[groupId] = currentCount + groupAgents.Count;
        }

        foreach (var (groupId, entropySum) in entropyByGroup)
        {
            var count = Math.Max(1, decisionCountByGroup.GetValueOrDefault(groupId));
            _lastEntropyByGroup[groupId] = entropySum / count;
        }

        return true;
    }

    // ── Learning decision pipeline ────────────────────────────────────────────
    //
    // The pipeline is split into four phases so that the pure-math phases (A and C) can be
    // parallelised across policy groups when ParallelPolicyGroups is enabled.
    //
    //  Phase A  EstimateNextValues         — pure math, parallelisable across groups
    //  Phase B  RecordTransitionsAndReset  — Godot API + list mutations, main thread only
    //  Phase C  trainer.SampleActions      — pure math, parallelisable across groups
    //  Phase D  ApplyGroupDecisions        — Godot API (ApplyDecision), main thread only

    /// <summary>Single-group orchestrator used when parallelism is off or only one group is active.</summary>
    private void ProcessLearningDecisions(string groupId, ITrainer trainer, List<PendingDecisionContext> pendingDecisions)
    {
        var nextValues = EstimateNextValues(trainer, pendingDecisions);
        var decisionObservations = RecordTransitionsAndResetEpisodes(groupId, trainer, pendingDecisions, nextValues);
        var decisions = trainer.SampleActions(VectorBatch.FromRows(decisionObservations));
        ApplyGroupDecisions(groupId, pendingDecisions, decisions, decisionObservations);
    }

    /// <summary>
    /// Multi-group orchestrator: phases A and C run in parallel across groups;
    /// phases B and D stay sequential on the main thread.
    /// </summary>
    private void RunParallelGroupDecisions(Dictionary<string, List<PendingDecisionContext>> pendingByGroup)
    {
        var groups = pendingByGroup.ToList();
        var nextValuesPerGroup      = new float[groups.Count][];
        var decisionObsPerGroup     = new List<float[]>[groups.Count];
        var decisionsPerGroup       = new PolicyDecision[groups.Count][];

        // Phase A: parallel value estimation (pure math, no Godot API, no shared writes)
        Parallel.For(0, groups.Count, i =>
        {
            var (groupId, pending) = groups[i];
            if (_trainersByGroup.TryGetValue(groupId, out var t))
                nextValuesPerGroup[i] = EstimateNextValues(t, pending);
        });

        // Phase B: main-thread transition recording + episode resets (Godot API)
        for (var i = 0; i < groups.Count; i++)
        {
            var (groupId, pending) = groups[i];
            if (_trainersByGroup.TryGetValue(groupId, out var trainer) && nextValuesPerGroup[i] is not null)
                decisionObsPerGroup[i] = RecordTransitionsAndResetEpisodes(groupId, trainer, pending, nextValuesPerGroup[i]);
        }

        // Phase C: parallel action sampling (pure math, no Godot API, no shared writes)
        Parallel.For(0, groups.Count, i =>
        {
            var (groupId, _) = groups[i];
            if (_trainersByGroup.TryGetValue(groupId, out var t) && decisionObsPerGroup[i] is not null)
                decisionsPerGroup[i] = t.SampleActions(VectorBatch.FromRows(decisionObsPerGroup[i]));
        });

        // Phase D: main-thread ApplyDecision (Godot API) + state updates
        for (var i = 0; i < groups.Count; i++)
        {
            var (groupId, pending) = groups[i];
            if (decisionsPerGroup[i] is not null && decisionObsPerGroup[i] is not null)
                ApplyGroupDecisions(groupId, pending, decisionsPerGroup[i], decisionObsPerGroup[i]);
        }
    }

    // ── Phase A ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Pure-math: calls <c>EstimateValues</c> for non-terminal agents to obtain bootstrap
    /// values for GAE. Safe to call from any thread; each group has its own trainer instance.
    /// </summary>
    private static float[] EstimateNextValues(ITrainer trainer, IReadOnlyList<PendingDecisionContext> pendingDecisions)
    {
        var nextValues = new float[pendingDecisions.Count];
        var nonTerminalObservations = new List<float[]>();
        var nonTerminalIndices = new List<int>();

        for (var index = 0; index < pendingDecisions.Count; index++)
        {
            if (!pendingDecisions[index].Done)
            {
                nonTerminalIndices.Add(index);
                nonTerminalObservations.Add(pendingDecisions[index].TransitionObservation);
            }
        }

        if (nonTerminalObservations.Count > 0)
        {
            var estimatedValues = trainer.EstimateValues(VectorBatch.FromRows(nonTerminalObservations));
            for (var index = 0; index < nonTerminalIndices.Count; index++)
                nextValues[nonTerminalIndices[index]] = estimatedValues[index];
        }

        return nextValues;
    }

    // ── Phase B ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Main-thread: record transitions into the trainer buffer, handle episode-done logic
    /// (ELO, metrics, curriculum, matchup refresh, ResetEpisode), and collect next observations.
    /// Returns the observation list used to feed Phase C's action sampling.
    /// </summary>
    private List<float[]> RecordTransitionsAndResetEpisodes(
        string groupId, ITrainer trainer, List<PendingDecisionContext> pendingDecisions, float[] nextValues)
    {
        var decisionObservations = new List<float[]>(pendingDecisions.Count);

        for (var index = 0; index < pendingDecisions.Count; index++)
        {
            var pending = pendingDecisions[index];
            var state = pending.State;
            var role = GetEnvironmentRole(state.EnvironmentIndex, state.GroupId);

            var transition = new Transition
            {
                Observation = state.LastObservation,
                DiscreteAction = state.PendingAction,
                ContinuousActions = state.PendingContinuousActions,
                Reward = state.WindowReward,
                Done = pending.Done,
                NextObservation = pending.TransitionObservation,
                OldLogProbability = state.LastLogProbability,
                Value = state.LastValue,
                NextValue = pending.Done ? 0f : nextValues[index],
            };
            trainer.RecordTransition(transition);
            _totalSteps += 1;

            state.WindowReward = 0f;
            state.StepsSinceDecision = 0;

            if (pending.Done)
            {
                _episodeCountByGroup[groupId] += 1;
                _lastEpisodeRewardByGroup[groupId] = pending.Agent.EpisodeReward;
                if (_stoppingConfig?.RewardThresholdWindow > 0)
                {
                    if (!_rewardWindowByGroup.TryGetValue(groupId, out var rewardWindow))
                    {
                        rewardWindow = new Queue<float>(_stoppingConfig.RewardThresholdWindow);
                        _rewardWindowByGroup[groupId] = rewardWindow;
                    }
                    rewardWindow.Enqueue(pending.Agent.EpisodeReward);
                    while (rewardWindow.Count > _stoppingConfig.RewardThresholdWindow)
                        rewardWindow.Dequeue();
                }
                _quickTestRewardTotalsByGroup[groupId] = _quickTestRewardTotalsByGroup.GetValueOrDefault(groupId) + pending.Agent.EpisodeReward;
                _quickTestEpisodeLengthTotalsByGroup[groupId] = _quickTestEpisodeLengthTotalsByGroup.GetValueOrDefault(groupId) + pending.Agent.EpisodeSteps;
                var episodeCount = _episodeCountByGroup[groupId];
                var curriculumEnvironment = _environments[state.EnvironmentIndex];

                if (role.IsSelfPlayLearner)
                {
                    var won = pending.Agent.EpisodeReward > _winThresholdByGroup.GetValueOrDefault(groupId, 0f);
                    if (_opponentBanksByGroup.TryGetValue(role.OpponentGroupId, out var ob))
                        ob.Pool.RecordOutcome(role.ActiveSnapshotKey, won);
                    if (_eloTrackersByGroup.TryGetValue(groupId, out var elo))
                    {
                        var rec = ob?.Pool.Records.FirstOrDefault(r => r.SnapshotKey == role.ActiveSnapshotKey);
                        elo.Update(rec?.SnapshotElo ?? EloTracker.InitialRating, won ? 1.0f : 0.0f);
                    }
                }

                if (!_quickTestMode && !_isWorkerMode)
                {
                    var metricSteps = _totalSteps + (_distributedMaster?.TotalWorkerSteps ?? 0L);
                    var totalEpisodeCount = episodeCount + _workerEpisodeCountByGroup.GetValueOrDefault(groupId);
                    _metricsWritersByGroup[groupId].AppendMetric(
                        pending.Agent.EpisodeReward,
                        pending.Agent.EpisodeSteps,
                        _lastPolicyLossByGroup[groupId],
                        _lastValueLossByGroup[groupId],
                        _lastEntropyByGroup[groupId],
                        _lastClipFractionByGroup[groupId],
                        metricSteps,
                        totalEpisodeCount,
                        pending.Agent.GetEpisodeRewardBreakdown(),
                        policyGroup: GetGroupDisplayName(groupId),
                        opponentGroup: role.IsSelfPlayLearner ? GetGroupDisplayName(role.OpponentGroupId) : string.Empty,
                        opponentSource: role.IsSelfPlayLearner ? role.ActiveOpponentSource : string.Empty,
                        opponentCheckpointPath: role.IsSelfPlayLearner ? role.ActiveOpponentCheckpointPath : string.Empty,
                        opponentUpdateCount: role.IsSelfPlayLearner ? role.ActiveOpponentUpdateCount : null,
                        learnerElo: role.IsSelfPlayLearner && _eloTrackersByGroup.TryGetValue(groupId, out var ep)
                            ? ep.Rating : null,
                        poolWinRate: role.IsSelfPlayLearner && _opponentBanksByGroup.TryGetValue(role.OpponentGroupId, out var opb)
                            ? opb.Pool.AverageWinRate : null,
                        curriculumProgress: curriculumEnvironment.Academy.Curriculum is not null
                            ? curriculumEnvironment.Academy.CurriculumProgress
                            : null);
                }

                // Worker: queue episode summary for the master to write to the metrics log.
                if (_isWorkerMode && _distributedWorker is not null)
                {
                    var breakdown = pending.Agent.GetEpisodeRewardBreakdown();
                    _distributedWorker.EnqueueEpisodeSummary(
                        groupId,
                        pending.Agent.EpisodeReward,
                        pending.Agent.EpisodeSteps,
                        breakdown.Count > 0 ? breakdown : null);
                }

                // Curriculum: update progress before reset so OnEpisodeBegin sees the new difficulty.
                // Workers receive the master's authoritative progress via CurriculumSync messages
                // (applied at the top of _PhysicsProcess), so they skip local calculation entirely.
                if (!_isWorkerMode)
                {
                    var curriculumConfig = curriculumEnvironment.Academy.Curriculum;
                    if (curriculumConfig is not null)
                    {
                        if (curriculumConfig.Mode == RLCurriculumMode.SuccessRate)
                        {
                            UpdateAdaptiveCurriculum(curriculumConfig, pending.Agent.EpisodeReward);
                        }
                        else if (curriculumEnvironment.Academy.MaxCurriculumSteps > 0)
                        {
                            // Include worker steps so the master's curriculum advances at the true
                            // combined throughput rate rather than only counting local steps.
                            var combinedSteps = _totalSteps + (_distributedMaster?.TotalWorkerSteps ?? 0L);
                            var progress = Mathf.Clamp(combinedSteps / (float)curriculumEnvironment.Academy.MaxCurriculumSteps, 0f, 1f);
                            SetCurriculumProgressForAllAcademies(progress);
                        }
                    }
                }

                PrepareEnvironmentForNextEpisode(state.EnvironmentIndex);
                if (!EnsureEnvironmentMatchupsReady(state.EnvironmentIndex))
                    GD.PushWarning($"[RL] Could not refresh self-play matchup for environment {state.EnvironmentIndex}; reusing the previous opponent policy.");

                pending.Agent.ResetEpisode();
                decisionObservations.Add(pending.Agent.CollectObservationArray());
            }
            else
            {
                decisionObservations.Add(pending.TransitionObservation);
            }
        }

        return decisionObservations;
    }

    // ── Phase D ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Main-thread: writes the sampled decisions back into each agent's runtime state and
    /// calls <c>ApplyDecision</c> on the Godot scene tree.
    /// </summary>
    private void ApplyGroupDecisions(
        string groupId,
        List<PendingDecisionContext> pendingDecisions,
        PolicyDecision[] decisions,
        IReadOnlyList<float[]> decisionObservations)
    {
        var entropySum = 0f;
        for (var index = 0; index < pendingDecisions.Count; index++)
        {
            var pending = pendingDecisions[index];
            var state = pending.State;
            var decision = decisions[index];
            state.LastObservation = decisionObservations[index];
            state.PendingAction = decision.DiscreteAction;
            state.PendingContinuousActions = decision.ContinuousActions;
            state.LastLogProbability = decision.LogProbability;
            state.LastValue = decision.Value;
            entropySum += decision.Entropy;
            ApplyDecision(pending.Agent, decision);
        }

        _lastEntropyByGroup[groupId] = pendingDecisions.Count > 0 ? entropySum / pendingDecisions.Count : 0f;
    }

    private void ProcessFrozenDecisions(List<PendingDecisionContext> pendingDecisions)
    {
        foreach (var pending in pendingDecisions)
        {
            var state = pending.State;
            state.WindowReward = 0f;
            state.StepsSinceDecision = 0;

            if (pending.Done)
            {
                PrepareEnvironmentForNextEpisode(state.EnvironmentIndex);
                if (!EnsureEnvironmentMatchupsReady(state.EnvironmentIndex))
                {
                    GD.PushWarning($"[RL] Could not refresh self-play matchup for environment {state.EnvironmentIndex}; reusing the previous opponent policy.");
                }

                pending.Agent.ResetEpisode();
            }

            var role = GetEnvironmentRole(state.EnvironmentIndex, state.GroupId);
            if (role.FrozenPolicy is null)
            {
                GD.PushWarning($"[RL] Frozen opponent policy missing for group '{GetGroupDisplayName(state.GroupId)}'.");
                continue;
            }

            var decisionObservation = pending.Done
                ? pending.Agent.CollectObservationArray()
                : pending.TransitionObservation;
            var decision = role.FrozenPolicy.Predict(decisionObservation);
            state.LastObservation = decisionObservation;
            state.PendingAction = decision.DiscreteAction;
            state.PendingContinuousActions = decision.ContinuousActions;
            state.LastLogProbability = 0f;
            state.LastValue = 0f;
            ApplyDecision(pending.Agent, decision);
        }
    }

    private void PersistCheckpoint(
        string groupId,
        RLCheckpoint checkpoint,
        long updateCount,
        bool forceLatestWrite = false,
        bool allowFrozenSnapshot = true)
    {
        if (_quickTestMode) return;
        var checkpointPath = GetGroupCheckpointPath(groupId);
        var participatesInSelfPlay = _selfPlayParticipantGroups.Contains(groupId);
        var shouldWriteLatest = forceLatestWrite
            || participatesInSelfPlay
            || updateCount == 0
            || updateCount % _checkpointInterval == 0;

        if (!string.IsNullOrEmpty(checkpointPath) && shouldWriteLatest)
        {
            RLCheckpoint.SaveToFile(checkpoint, checkpointPath);
        }

        // Write a named history snapshot alongside the latest checkpoint (skip update 0).
        if (shouldWriteLatest && updateCount > 0 && _manifest is not null)
        {
            var safeId = _groupBindingsByGroup.TryGetValue(groupId, out var histBinding)
                ? histBinding.SafeGroupId
                : RLPolicyGroupBindingResolver.MakeSafeGroupId(groupId);
            var historyResPath = $"{_manifest.RunDirectory}/history/checkpoint__{safeId}__u{updateCount:D6}.json";
            RLCheckpoint.SaveToFile(checkpoint, historyResPath);
            WriteCheckpointSidecar(checkpoint, historyResPath);
        }

        if (!_opponentBanksByGroup.TryGetValue(groupId, out var bank) || string.IsNullOrEmpty(checkpointPath))
        {
            return;
        }

        bank.Pool.LatestCheckpointPath = checkpointPath;

        if (!allowFrozenSnapshot)
        {
            return;
        }

        var frozenInterval = Math.Max(1, _frozenCheckpointIntervalByGroup.GetValueOrDefault(groupId, 10));
        if (updateCount <= 0 || updateCount % frozenInterval != 0)
        {
            return;
        }

        var frozenPath   = $"{GetSelfPlayBankDirectory(groupId)}/opponent__u{updateCount:D6}.json";
        var snapshotKey  = $"{frozenPath}::{updateCount}";
        var currentElo   = _eloTrackersByGroup.TryGetValue(groupId, out var eloTracker)
            ? eloTracker.Rating
            : EloTracker.InitialRating;
        RLCheckpoint.SaveToFile(checkpoint, frozenPath);
        bank.Pool.AddSnapshot(frozenPath, snapshotKey, currentElo);
    }

    private void PrepareEnvironmentForNextEpisode(int environmentIndex)
    {
        if (environmentIndex < 0 || environmentIndex >= _environments.Count)
        {
            return;
        }

        _environments[environmentIndex].NeedsMatchupRefresh = _environments[environmentIndex].GroupRoles.Values.Any(role => role.IsSelfPlayLearner);
    }

    private bool EnsureEnvironmentMatchupsReady(int environmentIndex)
    {
        if (environmentIndex < 0 || environmentIndex >= _environments.Count)
        {
            return true;
        }

        var environment = _environments[environmentIndex];
        if (!environment.NeedsMatchupRefresh)
        {
            return true;
        }

        return TryRefreshEnvironmentMatchups(environment, out _);
    }

    private bool TryRefreshEnvironmentMatchups(EnvironmentRuntime environment, out string error)
    {
        error = string.Empty;
        if (!environment.NeedsMatchupRefresh)
        {
            return true;
        }

        // Workers never call PersistCheckpoint (they don't train), so their in-memory opponent
        // pools never grow. Periodically rescan the shared bank directory on disk so workers
        // pick up snapshots that the master has written during training.
        if (_isWorkerMode && _totalSteps - _lastSelfPlayBankScanStep >= SelfPlayBankScanInterval)
        {
            foreach (var groupId in _selfPlayParticipantGroups)
                RescanOpponentBankDirectory(groupId);
            _lastSelfPlayBankScanStep = _totalSteps;
        }

        foreach (var learnerRole in environment.GroupRoles.Values.Where(role => role.IsSelfPlayLearner))
        {
            if (!TrySelectOpponentSnapshot(learnerRole.GroupId, learnerRole.OpponentGroupId, learnerRole.HistoricalOpponentRate, out var snapshot, out error))
            {
                return false;
            }

            if (!environment.GroupRoles.TryGetValue(learnerRole.OpponentGroupId, out var frozenRole))
            {
                error = $"Environment {environment.Index}: opponent group '{GetGroupDisplayName(learnerRole.OpponentGroupId)}' is missing.";
                return false;
            }

            frozenRole.FrozenPolicy = snapshot.Policy;
            frozenRole.ActiveOpponentCheckpointPath = snapshot.CheckpointPath;
            frozenRole.ActiveOpponentSource = snapshot.Source;
            frozenRole.ActiveOpponentUpdateCount = snapshot.UpdateCount;
            frozenRole.ActiveSnapshotKey = snapshot.SnapshotKey;

            learnerRole.ActiveOpponentCheckpointPath = snapshot.CheckpointPath;
            learnerRole.ActiveOpponentSource = snapshot.Source;
            learnerRole.ActiveOpponentUpdateCount = snapshot.UpdateCount;
            learnerRole.ActiveSnapshotKey = snapshot.SnapshotKey;
        }

        environment.NeedsMatchupRefresh = false;
        return true;
    }

    private bool TrySelectOpponentSnapshot(
        string learnerGroupId,
        string opponentGroupId,
        float historicalOpponentRate,
        out LoadedInferenceSnapshot snapshot,
        out string error)
    {
        snapshot = default;
        error = string.Empty;

        if (!_opponentBanksByGroup.TryGetValue(opponentGroupId, out var bank))
        {
            error = $"Group '{GetGroupDisplayName(opponentGroupId)}' has no opponent bank.";
            return false;
        }

        var useHistorical = bank.Pool.Records.Count > 0
            && _selfPlayRng.Randf() < Mathf.Clamp(historicalOpponentRate, 0f, 1f);

        OpponentRecord? historicalRecord = null;
        string selectedPath;
        if (useHistorical)
        {
            historicalRecord = bank.Pool.SampleHistorical();
            selectedPath     = historicalRecord?.CheckpointPath ?? bank.Pool.LatestCheckpointPath;
        }
        else
        {
            selectedPath = bank.Pool.LatestCheckpointPath;
        }

        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            error = $"Group '{GetGroupDisplayName(opponentGroupId)}' has no checkpoint available for self-play.";
            return false;
        }

        if (TryLoadInferenceSnapshot(selectedPath, opponentGroupId, useHistorical ? "historical" : "latest", out snapshot, out error))
        {
            return true;
        }

        if (!useHistorical || string.Equals(selectedPath, bank.Pool.LatestCheckpointPath, StringComparison.Ordinal))
        {
            error = $"Could not load opponent snapshot for learner '{GetGroupDisplayName(learnerGroupId)}': {error}";
            return false;
        }

        return TryLoadInferenceSnapshot(bank.Pool.LatestCheckpointPath, opponentGroupId, "latest", out snapshot, out error);
    }

    private bool TryLoadInferenceSnapshot(
        string checkpointPath,
        string groupId,
        string source,
        out LoadedInferenceSnapshot snapshot,
        out string error)
    {
        snapshot = default;
        error = string.Empty;

        RLCheckpoint? checkpoint = checkpointPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase)
            ? RLModelLoader.LoadFromFile(checkpointPath)
            : RLCheckpoint.LoadFromFile(checkpointPath);

        if (checkpoint is null)
        {
            error = $"checkpoint '{checkpointPath}' could not be loaded.";
            return false;
        }

        var cacheKey = $"{checkpointPath}::{checkpoint.UpdateCount}";
        if (!_frozenPoliciesBySnapshotKey.TryGetValue(cacheKey, out var policy))
        {
            var fallbackGraph = _groupBindingsByGroup.TryGetValue(groupId, out var binding)
                ? binding.Config?.ResolvedNetworkGraph
                : null;

            policy = InferencePolicyFactory.Create(checkpoint, fallbackGraph);
            policy.LoadCheckpoint(checkpoint);
            _frozenPoliciesBySnapshotKey[cacheKey] = policy;
        }

        snapshot = new LoadedInferenceSnapshot
        {
            CheckpointPath = checkpointPath,
            SnapshotKey    = cacheKey,
            Source         = source,
            UpdateCount    = checkpoint.UpdateCount,
            Policy         = policy,
        };
        return true;
    }

    private List<RLPolicyPairingConfig> GetConfiguredSelfPlayPairings()
    {
        return _academies.Count > 0
            ? _academies[0].GetResolvedSelfPlayPairings()
            : new List<RLPolicyPairingConfig>();
    }

    private static string ResolvePairingDisplayName(RLPolicyPairingConfig pairing)
    {
        if (!string.IsNullOrWhiteSpace(pairing.PairingId))
        {
            return pairing.PairingId.Trim();
        }

        var groupA = pairing.ResolvedGroupA?.ResourceName;
        var groupB = pairing.ResolvedGroupB?.ResourceName;
        if (!string.IsNullOrWhiteSpace(groupA) && !string.IsNullOrWhiteSpace(groupB))
        {
            return $"{groupA.Trim()} vs {groupB.Trim()}";
        }

        return "<unnamed pairing>";
    }

    private string ResolveGroupIdForConfig(RLPolicyGroupConfig config)
    {
        foreach (var (groupId, binding) in _groupBindingsByGroup)
        {
            if (ReferenceEquals(binding.Config, config))
            {
                return groupId;
            }
        }

        if (!string.IsNullOrWhiteSpace(config.ResourcePath))
        {
            foreach (var (groupId, binding) in _groupBindingsByGroup)
            {
                if (string.Equals(binding.ConfigPath, config.ResourcePath, StringComparison.Ordinal))
                {
                    return groupId;
                }
            }
        }

        return string.Empty;
    }

    private string GetGroupDisplayName(string groupId)
    {
        return _groupBindingsByGroup.TryGetValue(groupId, out var binding)
            ? binding.DisplayName
            : groupId;
    }

    private EnvironmentGroupRole GetEnvironmentRole(int environmentIndex, string groupId)
    {
        var environment = _environments[environmentIndex];
        return environment.GroupRoles[groupId];
    }

    private string GetSelfPlayBankDirectory(string groupId)
    {
        var safeId = _groupBindingsByGroup.TryGetValue(groupId, out var binding)
            ? binding.SafeGroupId
            : RLPolicyGroupBindingResolver.MakeSafeGroupId(groupId);
        return $"{_manifest?.RunDirectory}/selfplay/{safeId}";
    }

    private static void WriteCheckpointSidecar(RLCheckpoint checkpoint, string checkpointResPath)
    {
        var sidecarResPath = checkpointResPath.Replace(".json", ".meta.json", StringComparison.Ordinal);
        var training = new Godot.Collections.Dictionary
        {
            { "total_steps",   checkpoint.TotalSteps   },
            { "episode_count", checkpoint.EpisodeCount },
            { "update_count",  checkpoint.UpdateCount  },
        };
        var data = new Godot.Collections.Dictionary
        {
            { "format_version", checkpoint.FormatVersion },
            { "run_id",         checkpoint.RunId         },
            { "training",       training                 },
            { "meta",           checkpoint.CreateMetadataDictionary() },
        };
        var absPath = ProjectSettings.GlobalizePath(sidecarResPath);
        try
        {
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(absPath)!);
            System.IO.File.WriteAllText(absPath, Json.Stringify(data));
        }
        catch (Exception ex)
        {
            GD.PushWarning($"[RL] Sidecar write failed: {ex.Message}");
        }
    }

    private string? GetGroupCheckpointPath(string groupId)
    {
        var safeId = _groupBindingsByGroup.TryGetValue(groupId, out var binding)
            ? binding.SafeGroupId
            : RLPolicyGroupBindingResolver.MakeSafeGroupId(groupId);
        return $"{_manifest?.RunDirectory}/checkpoint__{safeId}.json";
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

    private static void ReapplyAction(IRLAgent agent, AgentRuntimeState state)
    {
        if (state.PendingAction >= 0)
        {
            agent.ApplyAction(state.PendingAction);
        }
        else if (state.PendingContinuousActions.Length > 0)
        {
            agent.ApplyAction(state.PendingContinuousActions);
        }
    }

    private static AgentRuntimeState CreateAgentState(string groupId, int environmentIndex, float[] observation, PolicyDecision decision, bool isLearningEnabled)
    {
        return new AgentRuntimeState
        {
            GroupId = groupId,
            EnvironmentIndex = environmentIndex,
            IsLearningEnabled = isLearningEnabled,
            LastObservation = observation,
            PendingAction = decision.DiscreteAction,
            PendingContinuousActions = decision.ContinuousActions,
            LastLogProbability = decision.LogProbability,
            LastValue = decision.Value,
        };
    }

    private static string MakePairKey(string left, string right)
    {
        return string.CompareOrdinal(left, right) <= 0
            ? $"{left}|{right}"
            : $"{right}|{left}";
    }

    private void SetupBatchDisplay()
    {
        var canvasLayer = new CanvasLayer();
        AddChild(canvasLayer);

        if (!_showBatchGrid || _viewports.Count == 1)
        {
            // Single view: only env 0 renders, fullscreen. All others are update-disabled orphans.
            var container = new SubViewportContainer();
            container.Stretch = true;
            container.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
            _viewports[0].RenderTargetUpdateMode = SubViewport.UpdateMode.Always;
            container.AddChild(_viewports[0]);
            canvasLayer.AddChild(container);

            for (var i = 1; i < _viewports.Count; i++)
            {
                AddChild(_viewports[i]);
            }
        }
        else
        {
            // Grid view: all envs rendered at full resolution, displayed scaled-to-fit with padding.
            var cols = (int)Math.Ceiling(Math.Sqrt(_viewports.Count));
            var rows = (int)Math.Ceiling((float)_viewports.Count / cols);
            var windowSize = DisplayServer.WindowGetSize();
            const int padding = 8;
            var cellW = (windowSize.X - padding * (cols + 1)) / cols;
            var cellH = (windowSize.Y - padding * (rows + 1)) / rows;

            var root = new Control();
            root.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
            canvasLayer.AddChild(root);

            for (var i = 0; i < _viewports.Count; i++)
            {
                var col = i % cols;
                var row = i / cols;

                // Render at full window resolution so the camera sees the whole scene.
                _viewports[i].Size = windowSize;
                _viewports[i].RenderTargetUpdateMode = SubViewport.UpdateMode.Always;
                AddChild(_viewports[i]);

                // Display the viewport texture scaled to fit the cell, keeping aspect ratio.
                var rect = new TextureRect();
                rect.Texture = _viewports[i].GetTexture();
                rect.ExpandMode = TextureRect.ExpandModeEnum.IgnoreSize;
                rect.StretchMode = TextureRect.StretchModeEnum.KeepAspectCentered;
                rect.Position = new Vector2(padding + col * (cellW + padding), padding + row * (cellH + padding));
                rect.Size = new Vector2(cellW, cellH);
                root.AddChild(rect);
            }
        }
    }

    private static Node? FindNodeByPath(Node root, string nodePath)
    {
        if (string.IsNullOrWhiteSpace(nodePath))
        {
            return null;
        }

        return root.GetNodeOrNull(new NodePath(nodePath));
    }

    private sealed class AgentRuntimeState
    {
        public string GroupId { get; init; } = string.Empty;
        public int EnvironmentIndex { get; init; }
        public bool IsLearningEnabled { get; init; }
        public float[] LastObservation { get; set; } = Array.Empty<float>();
        public int PendingAction { get; set; } = -1;
        public float[] PendingContinuousActions { get; set; } = Array.Empty<float>();
        public float LastLogProbability { get; set; }
        public float LastValue { get; set; }
        public float WindowReward { get; set; }
        public int StepsSinceDecision { get; set; }
    }

    private sealed class PendingDecisionContext
    {
        public IRLAgent Agent { get; init; } = null!;
        public AgentRuntimeState State { get; init; } = null!;
        public float[] TransitionObservation { get; init; } = Array.Empty<float>();
        public bool Done { get; init; }
    }

    private sealed class EnvironmentRuntime
    {
        public int Index { get; init; }
        public Node SceneRoot { get; init; } = null!;
        public RLAcademy Academy { get; init; } = null!;
        public Dictionary<string, List<IRLAgent>> AgentsByGroup { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, EnvironmentGroupRole> GroupRoles { get; } = new(StringComparer.Ordinal);
        public bool NeedsMatchupRefresh { get; set; }
    }

    private sealed class EnvironmentGroupRole
    {
        public string GroupId { get; init; } = string.Empty;
        public EnvironmentGroupControl Control { get; set; } = EnvironmentGroupControl.LiveTrainer;
        public bool IsSelfPlayLearner { get; set; }
        public string OpponentGroupId { get; set; } = string.Empty;
        public string LearnerGroupId { get; set; } = string.Empty;
        public float HistoricalOpponentRate { get; set; }
        public IInferencePolicy? FrozenPolicy { get; set; }
        public string ActiveOpponentCheckpointPath { get; set; } = string.Empty;
        public string ActiveOpponentSource { get; set; } = string.Empty;
        public long? ActiveOpponentUpdateCount { get; set; }
        public string ActiveSnapshotKey { get; set; } = string.Empty;
    }

    private enum EnvironmentGroupControl
    {
        LiveTrainer = 0,
        FrozenOpponent = 1,
    }

    private sealed class OpponentBankRuntime
    {
        public string GroupId { get; init; } = string.Empty;
        public PolicyPool Pool { get; init; } = null!;
    }

    private sealed class SelfPlayPairRuntime
    {
        public string PairKey { get; init; } = string.Empty;
        public string GroupA { get; init; } = string.Empty;
        public string GroupB { get; init; } = string.Empty;
        public bool TrainGroupA { get; set; }
        public bool TrainGroupB { get; set; }

        public int LearnerCount => (TrainGroupA ? 1 : 0) + (TrainGroupB ? 1 : 0);

        public string GetLearnerForEnvironment(int environmentIndex)
        {
            if (TrainGroupA && TrainGroupB)
            {
                return environmentIndex % 2 == 0 ? GroupA : GroupB;
            }

            return TrainGroupA ? GroupA : GroupB;
        }

        public string GetOpponentForLearner(string learnerGroupId)
        {
            return string.Equals(learnerGroupId, GroupA, StringComparison.Ordinal)
                ? GroupB
                : GroupA;
        }
    }

    private readonly struct LoadedInferenceSnapshot
    {
        public string CheckpointPath { get; init; }
        public string SnapshotKey { get; init; }
        public string Source { get; init; }
        public long UpdateCount { get; init; }
        public IInferencePolicy Policy { get; init; }
    }

    // ── Distributed training helpers ─────────────────────────────────────────

    /// <summary>
    /// Launches <see cref="RLDistributedConfig.WorkerCount"/> headless Godot worker processes.
    /// Workers connect back to the master on localhost:<see cref="RLDistributedConfig.MasterPort"/>.
    /// </summary>
    private void LaunchDistributedWorkers(RLDistributedConfig config)
    {
        GD.Print($"[RL Distributed] Executable : {(string.IsNullOrWhiteSpace(config.EngineExecutablePath) ? OS.GetExecutablePath() : config.EngineExecutablePath)}");
        GD.Print($"[RL Distributed] Project    : {ProjectSettings.GlobalizePath("res://")}");
        GD.Print($"[RL Distributed] Bootstrap  : res://addons/rl_agent_plugin/Scenes/Bootstrap/TrainingBootstrap.tscn");

        for (var i = 0; i < config.WorkerCount; i++)
            LaunchSingleWorker(config, i);
    }

    private void LaunchSingleWorker(RLDistributedConfig config, int workerId)
    {
        var executable     = !string.IsNullOrWhiteSpace(config.EngineExecutablePath)
            ? config.EngineExecutablePath
            : OS.GetExecutablePath();
        var projectPath    = ProjectSettings.GlobalizePath("res://");
        var bootstrapScene = "res://addons/rl_agent_plugin/Scenes/Bootstrap/TrainingBootstrap.tscn";

        var args = new[]
        {
            "--headless",
            "--path", projectPath,
            bootstrapScene,
            "--",
            "--rl-worker",
            $"--master-port={config.MasterPort}",
            $"--worker-id={workerId}",
        };

        GD.Print($"[RL Distributed] Launching worker {workerId}: {executable} {string.Join(" ", args)}");
        var pid = OS.CreateProcess(executable, args);
        if (pid > 0)
            GD.Print($"[RL Distributed] Worker {workerId} launched (PID {pid}).");
        else
            GD.PushError($"[RL Distributed] Failed to launch worker {workerId} — OS.CreateProcess returned -1.");
    }

    public override void _Process(double delta)
    {
        // Worker self-destruct: master is gone.
        if (_distributedWorker?.ShouldQuit == true)
        {
            GD.Print("[RL Distributed] Worker shutting down — master is gone.");
            GetTree().Quit();
            return;
        }

        // Relaunch any workers that crashed or disconnected.
        if (_distributedMaster is not null && _distributedConfig is { AutoLaunchWorkers: true })
        {
            var crashed = _distributedMaster.DrainPendingRelaunches();
            for (var i = 0; i < crashed; i++)
            {
                GD.Print($"[RL Distributed] Relaunching crashed worker as worker {_nextWorkerId}.");
                LaunchSingleWorker(_distributedConfig, _nextWorkerId++);
            }
        }

        if (_trainingOverlay is null || _distributedMaster is null) return;

        var s = _distributedMaster.GetStats(_totalSteps);

        // ── Smooth steps counter (rate-only PLL) ─────────────────────────────
        // The display position is NEVER directly snapped — only the rate is
        // adjusted.  This means the counter always ticks upward monotonically;
        // it just runs slightly faster when behind reality and slightly slower
        // when ahead.  Workers deliver steps in rollout-sized chunks, but those
        // arrivals only influence the rate, so the display stays perfectly smooth.
        var realSteps  = (double)s.TotalSteps;
        var targetRate = (double)s.StepsPerSec;

        if (_smoothDisplaySteps <= 0 && realSteps > 0)
        {
            // First data: start the counter slightly behind real so it always
            // appears to be running forward toward a known value.
            _smoothDisplaySteps = Math.Max(0.0, realSteps - targetRate * 0.5);
            _smoothDisplayRate  = targetRate;
        }
        else if (targetRate > 0)
        {
            // Phase error: positive = display is behind, negative = ahead.
            var error = realSteps - _smoothDisplaySteps;

            // Blend the target rate with a small correction proportional to
            // the error.  The gain of 0.15 means a 2 048-step gap adds only
            // ~307 steps/sec to the rate — barely perceptible, but it will
            // close the gap in a few seconds without any visible lurch.
            var correctedRate = targetRate + error * 0.15;
            correctedRate = Math.Max(0.0, correctedRate);

            // Smooth the rate change itself so sudden step-rate changes
            // (e.g. a worker disconnecting) ease in rather than snap.
            _smoothDisplayRate += (correctedRate - _smoothDisplayRate) * Math.Min(1.0, delta * 2.0);

            // Advance — position is only ever changed via the rate.
            _smoothDisplaySteps += _smoothDisplayRate * delta;
            _smoothDisplaySteps  = Math.Max(0.0, _smoothDisplaySteps);
        }

        var displaySteps = (long)_smoothDisplaySteps;

        // Overlay is always visible when enabled; only the text changes.
        if (_trainingOverlay.GetChildCount() > 0
            && _trainingOverlay.GetChild(0) is Control panel
            && panel.GetChildCount() > 0
            && panel.GetChild(0) is Label label)
        {
            label.Text = BuildOverlayText(s, displaySteps);
        }
    }

    private static string BuildOverlayText(DistributedMasterStats s, long displaySteps)
    {
        // All lines are always present so the panel never reflowsof jumps.
        // Use "---" placeholders for values not yet available.
        var sb = new System.Text.StringBuilder();

        // ── Title / status (fixed 1 line) ─────────────────────────────────────
        string status;
        if (s.ConnectedWorkers == 0)
            status = "Waiting for workers...";
        else if (s.IsTraining)
            status = "Training  — backprop in progress";
        else
            status = "Collecting rollouts";
        sb.AppendLine($"  RL Distributed  |  {status}");
        sb.AppendLine();

        // ── Connection (fixed 2 lines) ────────────────────────────────────────
        sb.AppendLine($"  Workers         {s.ConnectedWorkers,2} / {s.ExpectedWorkers,-2}");
        sb.AppendLine($"  Rollouts recv   {(s.ConnectedWorkers > 0 ? $"{s.RolloutsThisRound} / {s.ConnectedWorkers}" : "---")}");
        sb.AppendLine();

        // ── Progress (fixed 3 lines) ──────────────────────────────────────────
        sb.AppendLine($"  Update #        {s.TotalUpdates}");
        sb.AppendLine($"  Total steps     {(displaySteps > 0 ? displaySteps.ToString("N0") : "---")}");
        sb.AppendLine($"  Steps / sec     {(s.StepsPerSec > 0f ? s.StepsPerSec.ToString("N0") : "---")}");
        sb.AppendLine();

        // ── Timing (fixed 2 lines) ────────────────────────────────────────────
        sb.AppendLine($"  Backprop time   {(s.IsTraining && s.TrainingElapsedSec > 0.05f ? $"{s.TrainingElapsedSec:F2}s" : s.LastUpdateDurationSec > 0.01f ? $"{s.LastUpdateDurationSec:F2}s (last)" : "---")}");
        sb.AppendLine();

        // ── Training quality (fixed 4 lines — shown as --- until first update) ─
        var hasUpdate = s.TotalUpdates > 0;
        sb.AppendLine($"  Batch size      {(hasUpdate ? s.LastBatchSteps.ToString("N0") + " steps" : "---")}");
        sb.AppendLine($"  Policy loss     {(hasUpdate ? s.LastPolicyLoss.ToString("F4") : "---")}");
        sb.AppendLine($"  Value loss      {(hasUpdate ? s.LastValueLoss.ToString("F4")  : "---")}");
        sb.AppendLine($"  Entropy         {(hasUpdate ? s.LastEntropy.ToString("F4")    : "---")}");
        sb.Append    ($"  Clip frac       {(hasUpdate && s.LastClipFraction > 0f ? s.LastClipFraction.ToString("F4") : "---")}");

        return sb.ToString();
    }

    /// <summary>
    /// Builds a simple full-screen overlay that appears while the master runs backprop.
    /// Uses a high CanvasLayer so it renders on top of everything in the scene.
    /// </summary>
    private CanvasLayer CreateTrainingOverlay()
    {
        var canvas = new CanvasLayer { Layer = 128, Visible = true };

        // Panel anchored to the top-left corner — grows to fit content.
        var panel = new Panel
        {
            AnchorLeft   = 0f,
            AnchorTop    = 0f,
            AnchorRight  = 0f,
            AnchorBottom = 0f,
            OffsetRight  = 360f,
            OffsetBottom = 380f,
            Position     = new Vector2(16f, 16f),
        };
        var style = new StyleBoxFlat
        {
            BgColor      = new Color(0.05f, 0.05f, 0.08f, 0.88f),
            CornerRadiusTopLeft     = 6,
            CornerRadiusTopRight    = 6,
            CornerRadiusBottomLeft  = 6,
            CornerRadiusBottomRight = 6,
            ContentMarginLeft   = 12f,
            ContentMarginRight  = 12f,
            ContentMarginTop    = 10f,
            ContentMarginBottom = 10f,
        };
        panel.AddThemeStyleboxOverride("panel", style);

        var label = new Label
        {
            Text                = "Training — processing batch...",
            HorizontalAlignment = HorizontalAlignment.Left,
            VerticalAlignment   = VerticalAlignment.Top,
            AutowrapMode        = TextServer.AutowrapMode.Off,
        };
        label.SetAnchorsPreset(Control.LayoutPreset.FullRect);
        label.AddThemeColorOverride("font_color", new Color(0.9f, 0.95f, 1f, 1f));
        label.AddThemeFontSizeOverride("font_size", 14);

        panel.AddChild(label);
        canvas.AddChild(panel);
        AddChild(canvas);
        return canvas;
    }

    /// <summary>Overrides <c>Engine.TimeScale</c> and scales physics ticks accordingly.</summary>
    private void ApplySimulationSpeed(float speed)
    {
        var basePhysicsTicks = _previousPhysicsTicksPerSecond;
        Engine.TimeScale              = speed;
        Engine.PhysicsTicksPerSecond  = Math.Max(1, (int)Math.Ceiling(basePhysicsTicks * speed));
        Engine.MaxPhysicsStepsPerFrame = Math.Max(
            _previousMaxPhysicsStepsPerFrame,
            Engine.PhysicsTicksPerSecond);
    }

    /// <summary>Parses an integer from a <c>--key=value</c> command-line argument array.</summary>
    private static int ParseDistributedIntArg(string[] args, string key, int defaultValue)
    {
        foreach (var arg in args)
        {
            if (arg.StartsWith(key + "=", StringComparison.Ordinal)
                && int.TryParse(arg.AsSpan(key.Length + 1), out var v))
                return v;
        }
        return defaultValue;
    }
}
