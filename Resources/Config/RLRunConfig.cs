using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Run-level execution settings shared by all academies in a training session.
/// Controls simulation pacing, environment batching, checkpoint cadence, and
/// optional threading optimizations.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLRunConfig : Resource
{
    /// <summary>
    /// Optional prefix used when generating run IDs and output folders.
    /// </summary>
    [Export] public string RunPrefix { get; set; } = string.Empty;
    /// <summary>
    /// Global simulation speed multiplier applied during training.
    /// </summary>
    [Export(PropertyHint.Range, "0.0,10.0,0.1,or_greater")] public float SimulationSpeed { get; set; } = 1.0f;
    /// <summary>
    /// Number of parallel academy environments stepped per decision tick.
    /// </summary>
    [Export(PropertyHint.Range, "1,256,or_greater")] public int BatchSize { get; set; } = 1;
    /// <summary>
    /// Repeat each selected action for N physics steps before requesting a new decision.
    /// </summary>
    [Export(PropertyHint.Range, "1,15,or_greater")] public int ActionRepeat { get; set; } = 4;
    /// <summary>
    /// Save policy checkpoints every N trainer updates.
    /// Ignored when <see cref="CheckpointIntervalSteps"/> is greater than zero.
    /// </summary>
    [Export(PropertyHint.Range, "0,1000,or_greater")] public int CheckpointInterval { get; set; } = 10;
    /// <summary>
    /// If greater than zero, save history checkpoints every N total environment steps instead of
    /// every N trainer updates. Takes precedence over <see cref="CheckpointInterval"/>.
    /// </summary>
    [Export(PropertyHint.Range, "0,1000000,or_greater")] public long CheckpointIntervalSteps { get; set; } = 0;
    /// <summary>
    /// Number of most-recent history checkpoints to always keep intact.
    /// Older checkpoints beyond this count are thinned by <see cref="HistoryKeepEveryNth"/>.
    /// Set to 0 to disable thinning entirely.
    /// </summary>
    [Export(PropertyHint.Range, "0,100,or_greater")] public int HistoryKeepRecentCount { get; set; } = 20;
    /// <summary>
    /// For history checkpoints older than <see cref="HistoryKeepRecentCount"/>, keep every Nth
    /// and delete the rest. Set to 0 to disable thinning.
    /// </summary>
    [Export(PropertyHint.Range, "0,100,or_greater")] public int HistoryKeepEveryNth { get; set; } = 10;
    /// <summary>
    /// Compress history checkpoint files into a ZIP archive (.rlcheckpoint).
    /// The archive contains an uncompressed <c>meta.json</c> for fast metadata reads
    /// and a deflate-compressed <c>weights.json</c> for bulk weight data.
    /// Old plain-JSON checkpoints are still loaded correctly when this is disabled.
    /// </summary>
    [Export] public bool CompressCheckpoints { get; set; } = true;
    /// <summary>
    /// Show a tiled debug grid for batched environments in training scenes.
    /// </summary>
    [Export] public bool ShowBatchGrid { get; set; } = false;
    /// <summary>
    /// Run PPO gradient updates on a background thread while the main thread continues
    /// collecting transitions. Eliminates PPO backprop spikes from the main-thread frame budget.
    /// Opt-in; disabled by default for predictable single-threaded behavior.
    /// </summary>
    [Export] public bool AsyncGradientUpdates { get; set; } = false;
    /// <summary>
    /// When two or more policy groups are active, run their value-estimation (Phase A) and
    /// action-sampling (Phase C) forward passes in parallel across <c>System.Threading.Tasks</c>
    /// worker threads. Phase B (episode resets, Godot API) and Phase D (ApplyDecision, Godot API)
    /// remain on the main thread. Opt-in; has no effect with a single policy group.
    /// </summary>
    [Export] public bool ParallelPolicyGroups { get; set; } = false;
    /// <summary>
    /// Controls how rollout data is handled while an async gradient update is in flight.
    /// Only applies when <see cref="AsyncGradientUpdates"/> is enabled.
    /// <list type="bullet">
    ///   <item><b>Pause</b> (default) — incoming worker rollouts are discarded during training;
    ///   a fresh batch is gathered after each update so batch sizes stay consistent.</item>
    ///   <item><b>Cap</b> — rollouts are always accepted but the training batch is capped at
    ///   <c>RolloutLength × (WorkerCount + 1)</c>; prevents unbounded growth while retaining
    ///   recently collected data.</item>
    /// </list>
    /// </summary>
    [Export] public RLAsyncRolloutPolicy AsyncRolloutPolicy { get; set; } = RLAsyncRolloutPolicy.Pause;

    /// <summary>
    /// Optional stopping conditions for this training run.
    /// When null, training continues indefinitely until manually stopped.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLStoppingConfig))]
    public RLStoppingConfig? StoppingConditions { get; set; }

    /// <summary>
    /// Optional periodic evaluation config. When set, the training loop runs greedy
    /// evaluation episodes every <see cref="RLEvaluationConfig.EvaluationFrequencySteps"/>
    /// steps and logs <c>eval_mean_reward</c> separately from training reward.
    /// When null, no evaluation runs are performed.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLEvaluationConfig))]
    public RLEvaluationConfig? Evaluation { get; set; }
}
