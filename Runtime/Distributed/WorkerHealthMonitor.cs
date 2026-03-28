using System;
using System.Text;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Tracks physics and render throughput on a distributed worker and emits periodic health reports.
/// Reports are routed through <paramref name="log"/> — pass <c>DistributedWorker.SendLogMessage</c>
/// to forward them to the master console, or <c>GD.Print</c> for local output.
///
/// Paused time (e.g. waiting for PPO weight updates) is excluded from throughput calculations
/// so on-policy training pauses do not trigger false dropped-step warnings.
///
/// Add as a child node during worker startup; it self-manages via <c>_Process</c> / <c>_PhysicsProcess</c>.
/// </summary>
public partial class WorkerHealthMonitor : Node
{
    /// <summary>Warn when actual physics rate falls below this fraction of PhysicsTicksPerSecond.</summary>
    private const float DroppedStepWarnThreshold = 0.85f;
    private const int ConsecutiveWarnsToReport = 3;

    private readonly Action<string> _log;
    private readonly bool           _isRendererMode;
    private readonly Func<bool>?    _isPaused;
    private readonly float          _reportIntervalSec;

    private int    _physicsSteps;
    private int    _renderFrames;
    private int    _renderWarnStreak;
    private int    _dropWarnStreak;
    private double _activeMs;      // accumulated real milliseconds while NOT paused
    private ulong  _lastTickMs;

    /// <param name="log">Where to send health report strings.</param>
    /// <param name="isRendererMode">True when the worker has a GPU renderer (tracks render fps).</param>
    /// <param name="isPaused">Optional delegate; when it returns true the window timer is frozen (e.g. waiting for PPO weights).</param>
    /// <param name="reportIntervalSec">How often to emit a health report (active real-time seconds, excluding pauses).</param>
    public WorkerHealthMonitor(Action<string> log, bool isRendererMode, Func<bool>? isPaused = null, float reportIntervalSec = 10f)
    {
        _log               = log;
        _isRendererMode    = isRendererMode;
        _isPaused          = isPaused;
        _reportIntervalSec = reportIntervalSec;
    }

    public override void _Ready()
    {
        _lastTickMs = Time.GetTicksMsec();
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_isPaused?.Invoke() == true) return;
        _physicsSteps++;
    }

    public override void _Process(double delta)
    {
        var nowMs   = Time.GetTicksMsec();
        var frameMs = (double)(nowMs - _lastTickMs);
        _lastTickMs = nowMs;

        if (_isPaused?.Invoke() == true)
            return; // freeze the window — don't accumulate paused time

        if (_isRendererMode)
            _renderFrames++;

        _activeMs += frameMs;
        if (_activeMs < _reportIntervalSec * 1000.0)
            return;

        Report();

        _physicsSteps = 0;
        _renderFrames = 0;
        _activeMs     = 0;
    }

    private void Report()
    {
        var elapsedSec        = _activeMs / 1000.0;
        var actualPhysicsHz   = _physicsSteps / elapsedSec;
        var expectedPhysicsHz = (float)Engine.PhysicsTicksPerSecond;
        var dropFraction      = 1f - (actualPhysicsHz / expectedPhysicsHz);
        var dropWarning       = dropFraction > 1f - DroppedStepWarnThreshold;

        var sb = new StringBuilder("[Worker Health] ");
        sb.Append($"physics={actualPhysicsHz:F0}/{expectedPhysicsHz:F0} Hz");

        var renderWarning = false;
        if (_isRendererMode)
        {
            var renderFps = _renderFrames / elapsedSec;
            sb.Append($"  render={renderFps:F0} fps");
            if (renderFps < actualPhysicsHz * 0.95)
            {
                renderWarning = true;
                sb.Append("  !! render fps below physics rate — observations may be stale");
            }
        }

        if (dropWarning)
            sb.Append($"  !! DROPPING ~{dropFraction * 100f:F0}% of physics steps (CPU bottleneck)");

        _renderWarnStreak = renderWarning ? _renderWarnStreak + 1 : 0;
        _dropWarnStreak   = dropWarning ? _dropWarnStreak + 1 : 0;

        // Emit once when a warning first looks persistent; suppress repeated logs while it continues.
        var shouldLog = _renderWarnStreak == ConsecutiveWarnsToReport || _dropWarnStreak == ConsecutiveWarnsToReport;
        if (!shouldLog)
            return;

        _log(sb.ToString());
    }
}
