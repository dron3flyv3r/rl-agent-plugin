using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Attach this to <see cref="RLAcademy.DistributedConfig"/> to enable distributed training.
///
/// In distributed mode one master process owns all trainers and N headless worker processes
/// collect rollouts in parallel.  Workers send completed rollouts to the master over a local
/// TCP socket; the master trains and broadcasts updated weights back.
///
/// To run:  start the scene normally (master) — worker processes are launched automatically
/// when <see cref="AutoLaunchWorkers"/> is true.  Workers can also be started manually with
/// the command-line flag <c>-- --rl-worker --master-port=PORT</c>.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLDistributedConfig : Resource
{
    /// <summary>Number of headless worker processes that collect rollouts in parallel.</summary>
    [Export(PropertyHint.Range, "1,32,1")]
    public int WorkerCount { get; set; } = 4;

    /// <summary>TCP port the master listens on.  All workers connect to this port on localhost.</summary>
    [Export(PropertyHint.Range, "1024,65535,1")]
    public int MasterPort { get; set; } = 7890;

    /// <summary>
    /// When true the master automatically spawns worker sub-processes on startup.
    /// </summary>
    [Export]
    public bool AutoLaunchWorkers { get; set; } = true;

    /// <summary>
    /// Absolute path to the Godot engine executable used to launch workers.
    /// Leave blank to use <c>OS.GetExecutablePath()</c> (the current running binary).
    /// Set this if your Godot binary has a non-standard name such as <c>godot-mono</c>,
    /// <c>godot4-mono</c>, or a versioned path like <c>/usr/local/bin/godot4.3-mono</c>.
    /// </summary>
    [Export(PropertyHint.GlobalFile)]
    public string EngineExecutablePath { get; set; } = string.Empty;

    /// <summary>
    /// When true, workers are launched with a renderer instead of <c>--headless</c>.
    /// Required when any agent uses <see cref="RLCameraSensor2D"/>, because headless
    /// workers have no GPU renderer and <c>SubViewport</c> produces no pixels.
    ///
    /// On a desktop machine workers will open minimised windows — this is normal.
    /// On a true headless server set <see cref="XvfbWrapperArgs"/> so each worker
    /// gets a virtual framebuffer via <c>xvfb-run</c>.
    /// </summary>
    [Export]
    public bool WorkersRequireRenderer { get; set; } = false;

    /// <summary>
    /// Arguments passed to <c>xvfb-run</c> when <see cref="WorkersRequireRenderer"/> is true
    /// and you are running on a headless server with no physical display.
    /// Leave blank on a desktop machine (workers will open minimised windows instead).
    ///
    /// Typical value: <c>-a</c>  (auto-selects a free display number).
    /// The Godot executable and its args are appended after these arguments automatically.
    /// </summary>
    [Export]
    public string XvfbWrapperArgs { get; set; } = string.Empty;

    /// <summary>
    /// <c>Engine.TimeScale</c> applied to worker processes.
    /// Workers have no display so they can run much faster than the master.
    /// </summary>
    [Export(PropertyHint.Range, "1,20,0.5,or_greater")]
    public float WorkerSimulationSpeed { get; set; } = 4.0f;

    [ExportGroup("Monitor")]
    /// <summary>
    /// Print a status summary to the Godot console every N training updates.
    /// Shows worker count, batch size, training losses, and throughput.
    /// Set to 0 to disable.
    /// </summary>
    [Export(PropertyHint.Range, "0,500,1,or_greater")]
    public int MonitorIntervalUpdates { get; set; } = 5;

    /// <summary>
    /// Show an on-screen overlay while the master is running backprop.
    /// Displays live performance stats: steps/sec, batch size, losses, worker count, etc.
    /// </summary>
    [Export]
    public bool ShowTrainingOverlay { get; set; } = true;

    /// <summary>
    /// Log individual events: worker connect/disconnect, each rollout arrival, weight broadcasts.
    /// Useful for debugging the distributed setup but noisy during normal training.
    /// </summary>
    [Export]
    public bool VerboseLog { get; set; } = false;
    
    /// <summary>
    /// If true, the master prints diagnostic info about each rollout as it arrives.
    /// Useful for debugging the distributed setup and understanding rollout quality,
    /// but can be very verbose during normal training.
    /// </summary>
    [Export]
    public bool ShowRolloutDiagnostics { get; set; } = true;
}
