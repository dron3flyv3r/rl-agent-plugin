using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Godot;

namespace RlAgentPlugin.Runtime;

// ── Public stats snapshot ─────────────────────────────────────────────────────

/// <summary>
/// Live performance snapshot exposed by <see cref="DistributedMaster"/>.
/// Safe to read from the main thread (e.g. in <c>_Process</c> to update a UI overlay).
/// </summary>
public struct DistributedMasterStats
{
    /// <summary>Background training job is currently in flight.</summary>
    public bool IsTraining;
    /// <summary>Seconds elapsed since the current background training job was scheduled.</summary>
    public float TrainingElapsedSec;
    /// <summary>Wall-clock seconds the most recently completed training update took.</summary>
    public float LastUpdateDurationSec;

    /// <summary>Workers currently connected.</summary>
    public int ConnectedWorkers;
    /// <summary>Workers configured (target count).</summary>
    public int ExpectedWorkers;
    /// <summary>Rollouts received from workers in the current round (resets each update).</summary>
    public int RolloutsThisRound;

    /// <summary>Total training updates completed across all groups.</summary>
    public long TotalUpdates;
    /// <summary>Total simulation steps seen by the master (local + worker).</summary>
    public long TotalSteps;
    /// <summary>Steps contributed by worker processes, cumulative.</summary>
    public long TotalWorkerSteps;
    /// <summary>Steps in the most recently completed training batch.</summary>
    public int LastBatchSteps;

    /// <summary>Rolling-average simulation steps per wall-clock second (last 8 rounds).</summary>
    public float StepsPerSec;

    /// <summary>Losses from the most recent completed update (zeroed until first update).</summary>
    public float LastPolicyLoss;
    public float LastValueLoss;
    public float LastEntropy;
    public float LastClipFraction;
}

// ── Master implementation ─────────────────────────────────────────────────────

/// <summary>
/// Runs on the master process.  Opens a TCP server, accepts worker connections,
/// collects rollout data from each worker, injects it into the local trainers,
/// and broadcasts updated weights after every training update.
///
/// When the trainer implements <see cref="IAsyncTrainer"/> backprop runs on a
/// background thread so the master's game loop never freezes.
/// Read <see cref="GetStats"/> each frame to drive a UI overlay.
/// </summary>
public sealed class DistributedMaster : IDisposable
{
    // ── Worker connection handle ─────────────────────────────────────────────

    private sealed class WorkerConnection
    {
        public TcpClient    Client { get; }
        public BinaryWriter Writer { get; }
        private readonly object _writeLock = new();

        public WorkerConnection(TcpClient client)
        {
            Client = client;
            Writer = new BinaryWriter(client.GetStream());
        }

        public void Send(DistributedMessageType type, string groupId, byte[] payload)
        {
            try   { lock (_writeLock) DistributedProtocol.WriteMessage(Writer, type, groupId, payload); }
            catch (Exception ex) { GD.PushWarning($"[RL Distributed] Send failed: {ex.Message}"); }
        }

        public void Close() => Client.Close();
    }

    // ── State ────────────────────────────────────────────────────────────────

    private readonly int  _port;
    private readonly int  _workerCount;
    private readonly int  _monitorInterval;
    private readonly bool _verbose;
    private readonly Dictionary<string, IDistributedTrainer> _trainers;
    private readonly Dictionary<string, IAsyncTrainer>       _asyncTrainers = new(StringComparer.Ordinal);
    private readonly RLAsyncRolloutPolicy _asyncRolloutPolicy;
    private readonly int                  _rolloutLength;

    private TcpListener?  _listener;
    private Thread?       _acceptThread;
    private volatile bool _running;
    private volatile int  _pendingRelaunches;

    // Diagnostic: log the first N rollouts received from workers to verify data quality.
    private const int DiagnosticRolloutCount = 6;
    private int _diagnosticRolloutsLogged;

    private readonly List<WorkerConnection> _connections     = new();
    private readonly object                 _connectionsLock = new();

    private readonly Queue<(string groupId, byte[] data)>                        _pendingRollouts         = new();
    private readonly Queue<(string groupId, List<WorkerEpisodeSummary> items)>  _pendingEpisodeSummaries = new();
    // Hello requests are queued by the read thread and processed on the main thread so that
    // ExportWeights() never races with SampleAction() (both use the TorchSharp network).
    private readonly Queue<(WorkerConnection conn, string groupId)>             _pendingHellos           = new();
    private readonly object                                                      _rolloutsLock            = new();

    // Per-group counters (current round).
    private readonly Dictionary<string, int>  _rolloutsThisRound    = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _workerStepsThisRound = new(StringComparer.Ordinal);

    // Async training state.
    private readonly HashSet<string>          _trainingInProgress = new(StringComparer.Ordinal);
    private readonly Dictionary<string, DateTime> _trainStartTime = new(StringComparer.Ordinal);

    // Cumulative stats.
    private readonly Dictionary<string, long>               _totalUpdates     = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long>               _totalWorkerSteps = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long>               _totalStepsAll    = new(StringComparer.Ordinal);
    private readonly Dictionary<string, TrainerUpdateStats> _lastStats        = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float>              _lastUpdateDurSec = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int>                _lastBatchSteps   = new(StringComparer.Ordinal);
    private readonly Dictionary<string, DateTime>           _lastUpdateTime   = new(StringComparer.Ordinal);

    // Rolling steps/sec window (stores (steps, durationSec) for last N rounds).
    private const int StepsWindow = 8;
    private readonly Dictionary<string, Queue<(long steps, float dur)>> _throughputWindow = new(StringComparer.Ordinal);

    // ── Construction / lifecycle ─────────────────────────────────────────────

    public DistributedMaster(
        int  port,
        int  workerCount,
        int  monitorInterval,
        bool verbose,
        Dictionary<string, IDistributedTrainer> trainers,
        RLAsyncRolloutPolicy asyncRolloutPolicy = RLAsyncRolloutPolicy.Pause,
        int  rolloutLength = 256)
    {
        _port               = port;
        _workerCount        = workerCount;
        _monitorInterval    = monitorInterval;
        _verbose            = verbose;
        _trainers           = trainers;
        _asyncRolloutPolicy = asyncRolloutPolicy;
        _rolloutLength      = rolloutLength;

        foreach (var (g, t) in trainers)
        {
            _rolloutsThisRound[g]    = 0;
            _workerStepsThisRound[g] = 0;
            _totalUpdates[g]         = 0;
            _totalWorkerSteps[g]     = 0;
            _totalStepsAll[g]        = 0;
            _lastUpdateDurSec[g]     = 0f;
            _lastBatchSteps[g]       = 0;
            _lastUpdateTime[g]       = DateTime.UtcNow;
            _throughputWindow[g]     = new Queue<(long, float)>();

            if (t is IAsyncTrainer at) _asyncTrainers[g] = at;
        }
    }

    public int  ConnectedWorkers { get { lock (_connectionsLock) return _connections.Count; } }
    public bool IsTraining       => _trainingInProgress.Count > 0;

    /// <summary>
    /// Returns how many workers have disconnected since the last call and resets the counter.
    /// Call from the main thread to decide whether to relaunch replacement processes.
    /// </summary>
    public int DrainPendingRelaunches() => Interlocked.Exchange(ref _pendingRelaunches, 0);

    /// <summary>Cumulative episodes completed by all worker processes (counted from Done flags in rollouts).</summary>
    public long TotalWorkerEpisodes { get; private set; }

    /// <summary>Cumulative steps contributed by all worker processes across all groups.</summary>
    public long TotalWorkerSteps
    {
        get
        {
            var total = 0L;
            foreach (var g in _trainers.Keys)
                total += _totalWorkerSteps.GetValueOrDefault(g);
            return total;
        }
    }

    // ── Stats snapshot ────────────────────────────────────────────────────────

    /// <summary>
    /// Returns a combined performance snapshot across all groups.
    /// Safe to call from <c>_Process</c>.
    /// </summary>
    public DistributedMasterStats GetStats(long masterTotalSteps)
    {
        var connected = ConnectedWorkers;

        // Aggregate across groups (typical case: one group).
        var totalUpdates     = 0L;
        var totalWorkerSteps = 0L;
        var lastBatch        = 0;
        var stepsPerSec      = 0f;
        var lastDur          = 0f;
        var policyLoss       = 0f;
        var valueLoss        = 0f;
        var entropy          = 0f;
        var clipFrac         = 0f;
        var groupCount       = Math.Max(1, _trainers.Count);

        foreach (var g in _trainers.Keys)
        {
            totalUpdates     += _totalUpdates.GetValueOrDefault(g);
            totalWorkerSteps += _totalWorkerSteps.GetValueOrDefault(g);
            lastBatch        += _lastBatchSteps.GetValueOrDefault(g);
            stepsPerSec      += CalcStepsPerSec(g);
            lastDur           = Math.Max(lastDur, _lastUpdateDurSec.GetValueOrDefault(g));

            if (_lastStats.TryGetValue(g, out var s))
            {
                policyLoss += s.PolicyLoss;
                valueLoss  += s.ValueLoss;
                entropy    += s.Entropy;
                clipFrac   += s.ClipFraction;
            }
        }

        var trainingSec = 0f;
        foreach (var g in _trainingInProgress)
            if (_trainStartTime.TryGetValue(g, out var t))
                trainingSec = Math.Max(trainingSec, (float)(DateTime.UtcNow - t).TotalSeconds);

        return new DistributedMasterStats
        {
            IsTraining             = IsTraining,
            TrainingElapsedSec     = trainingSec,
            LastUpdateDurationSec  = lastDur,
            ConnectedWorkers       = connected,
            ExpectedWorkers        = _workerCount,
            RolloutsThisRound      = _rolloutsThisRound.Values.Sum(),
            TotalUpdates           = totalUpdates,
            TotalSteps             = masterTotalSteps + totalWorkerSteps,
            TotalWorkerSteps       = totalWorkerSteps,
            LastBatchSteps         = lastBatch,
            StepsPerSec            = stepsPerSec,
            LastPolicyLoss         = policyLoss / groupCount,
            LastValueLoss          = valueLoss  / groupCount,
            LastEntropy            = entropy    / groupCount,
            LastClipFraction       = clipFrac   / groupCount,
        };
    }

    private float CalcStepsPerSec(string groupId)
    {
        if (!_throughputWindow.TryGetValue(groupId, out var window) || window.Count == 0)
            return 0f;
        var totalSteps = 0L;
        var totalDur   = 0f;
        foreach (var (s, d) in window) { totalSteps += s; totalDur += d; }
        return totalDur > 0f ? totalSteps / totalDur : 0f;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    public void Start()
    {
        _running  = true;
        _listener = new TcpListener(IPAddress.Loopback, _port);
        _listener.Start();
        GD.Print($"[RL Distributed] Master listening on port {_port}, expecting {_workerCount} worker(s).");
        if (_asyncTrainers.Count > 0)
            GD.Print("[RL Distributed] Async training active — no main-thread freeze during backprop.");

        _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "DistMaster-Accept" };
        _acceptThread.Start();
    }

    // ── Background networking ─────────────────────────────────────────────────

    private void AcceptLoop()
    {
        while (_running)
        {
            try
            {
                if (!(_listener?.Pending() ?? false)) { Thread.Sleep(10); continue; }
                var client = _listener!.AcceptTcpClient();
                client.NoDelay = true;
                var conn = new WorkerConnection(client);
                lock (_connectionsLock) _connections.Add(conn);
                GD.Print($"[RL Distributed] Worker connected ({ConnectedWorkers}/{_workerCount}).");
                var reader = new BinaryReader(client.GetStream());
                new Thread(() => ClientReadLoop(conn, reader)) { IsBackground = true, Name = "DistMaster-Client" }.Start();
            }
            catch when (!_running) { break; }
            catch (Exception ex) { GD.PushWarning($"[RL Distributed] Accept error: {ex.Message}"); }
        }
    }

    private void ClientReadLoop(WorkerConnection conn, BinaryReader reader)
    {
        try
        {
            while (_running && conn.Client.Connected)
            {
                var (type, groupId, payload) = DistributedProtocol.ReadMessage(reader);
                switch (type)
                {
                    case DistributedMessageType.Hello:
                        // Queue for the main thread — ExportWeights() touches the TorchSharp
                        // network and must not race with SampleAction() on the physics thread.
                        lock (_rolloutsLock) _pendingHellos.Enqueue((conn, groupId));
                        if (_verbose) GD.Print($"[RL Distributed] Worker hello '{groupId}' queued.");
                        break;
                    case DistributedMessageType.Rollout:
                        lock (_rolloutsLock) _pendingRollouts.Enqueue((groupId, payload));
                        break;
                    case DistributedMessageType.EpisodeSummary:
                        var summaries = DistributedProtocol.DeserializeEpisodeSummaries(payload);
                        if (summaries.Count > 0)
                            lock (_rolloutsLock) _pendingEpisodeSummaries.Enqueue((groupId, summaries));
                        break;
                }
            }
        }
        catch (Exception ex) when (_running) { GD.PushWarning($"[RL Distributed] Worker disconnected: {ex.Message}"); }
        finally
        {
            lock (_connectionsLock) _connections.Remove(conn);
            conn.Close();
            GD.Print($"[RL Distributed] Worker removed ({ConnectedWorkers} remaining).");
            Interlocked.Increment(ref _pendingRelaunches);
        }
    }

    // ── Main-thread API ───────────────────────────────────────────────────────

    public void ProcessIncoming()
    {
        lock (_rolloutsLock)
        {
            // Handle Hello requests on the main thread so ExportWeights() never races
            // with SampleAction() — both access the shared TorchSharp network.
            while (_pendingHellos.Count > 0)
            {
                var (hConn, hGroup) = _pendingHellos.Dequeue();
                if (_trainers.TryGetValue(hGroup, out var ht))
                    hConn.Send(DistributedMessageType.Weights, hGroup, ht.ExportWeights());
                if (_verbose) GD.Print($"[RL Distributed] Sent initial weights to worker for '{hGroup}'.");
            }

            while (_pendingRollouts.Count > 0)
            {
                var (groupId, data) = _pendingRollouts.Dequeue();
                if (!_trainers.TryGetValue(groupId, out var trainer)) continue;

                // In Pause mode, discard rollouts that arrive while an on-policy training job is
                // in flight so the transition buffer does not grow unboundedly between updates.
                // Off-policy trainers (SAC) are never paused: they benefit from a larger replay
                // buffer and sample a fixed mini-batch regardless of how many transitions exist.
                var discard = _asyncRolloutPolicy == RLAsyncRolloutPolicy.Pause
                              && _trainingInProgress.Contains(groupId)
                              && !trainer.IsOffPolicy;
                if (discard)
                {
                    if (_verbose)
                        GD.Print($"[RL Distributed] Rollout '{groupId}' discarded (Pause policy — training in progress).");
                    continue;
                }

                trainer.InjectRollout(data);

                if (_diagnosticRolloutsLogged < DiagnosticRolloutCount)
                {
                    LogRolloutDiagnostic(groupId, data, _diagnosticRolloutsLogged + 1);
                    _diagnosticRolloutsLogged++;
                }

                var steps    = data.Length >= 4 ? BitConverter.ToInt32(data, 0) : 0;
                var episodes = DistributedProtocol.CountEpisodesInRollout(data);
                _rolloutsThisRound[groupId]    = _rolloutsThisRound.GetValueOrDefault(groupId) + 1;
                _workerStepsThisRound[groupId] = _workerStepsThisRound.GetValueOrDefault(groupId) + steps;
                TotalWorkerEpisodes           += episodes;
                if (_verbose)
                    GD.Print($"[RL Distributed] Rollout '{groupId}': {steps} steps ({_rolloutsThisRound[groupId]}/{ConnectedWorkers}).");
            }
        }
    }

    public TrainerUpdateStats? TickUpdate(
        IDistributedTrainer trainer,
        string groupId,
        long   totalSteps,
        long   episodeCount)
    {
        ProcessIncoming();

        // Inform SAC trainer how many data sources are active so auto-UTD scales correctly.
        if (trainer is SacTrainer sacTrainer)
            sacTrainer.DataSources = ConnectedWorkers + 1;

        // Poll running background job.
        if (_trainingInProgress.Contains(groupId))
        {
            if (!_asyncTrainers.TryGetValue(groupId, out var running)) return null;
            var result = running.TryPollResult(groupId, totalSteps, episodeCount);
            if (result is null) return null;
            _trainingInProgress.Remove(groupId);
            OnTrainComplete(groupId, trainer, result, totalSteps);
            return result;
        }

        // Gate on training readiness.
        // On-policy: synchronous barrier — wait for all workers before training.
        // Off-policy: no gate — master trains on its own schedule; workers asynchronously
        //             enrich the replay buffer. TryUpdate/TryScheduleBackgroundUpdate
        //             apply their own internal conditions (warmup, step cadence).
        if (!trainer.IsOffPolicy)
        {
            var connected = ConnectedWorkers;
            if (connected > 0)
            {
                if (_rolloutsThisRound.GetValueOrDefault(groupId) < connected) return null;
            }
            else
            {
                if (!trainer.IsRolloutReady) return null;
            }
        }

        // Prefer async (non-blocking backprop).
        if (_asyncTrainers.TryGetValue(groupId, out var asyncT))
        {
            // In Cap mode, limit the on-policy snapshot to one full round of rollouts so that
            // excess transitions accumulated while waiting for workers do not inflate the batch.
            // Off-policy trainers (SAC) sample a fixed mini-batch internally; the cap is a no-op for them.
            var maxTransitions = _asyncRolloutPolicy == RLAsyncRolloutPolicy.Cap && !trainer.IsOffPolicy
                ? (_workerCount + 1) * _rolloutLength
                : int.MaxValue;

            if (asyncT.TryScheduleBackgroundUpdate(groupId, totalSteps, episodeCount, maxTransitions))
            {
                _trainingInProgress.Add(groupId);
                _trainStartTime[groupId] = DateTime.UtcNow;
                if (_verbose) GD.Print($"[RL Distributed] Background training scheduled for '{groupId}'.");
                return null;
            }
        }

        // Synchronous fallback.
        var stats = trainer.TryUpdate(groupId, totalSteps, episodeCount);
        if (stats is not null) OnTrainComplete(groupId, trainer, stats, totalSteps);
        return stats;
    }

    private void OnTrainComplete(
        string groupId,
        IDistributedTrainer trainer,
        TrainerUpdateStats stats,
        long totalSteps)
    {
        var now     = DateTime.UtcNow;
        var durSec  = _trainStartTime.TryGetValue(groupId, out var start)
                      ? (float)(now - start).TotalSeconds : 0f;
        var wSteps  = _workerStepsThisRound.GetValueOrDefault(groupId);
        var bSteps  = (int)wSteps; // local steps were already in the buffer

        // Throughput window.
        if (_throughputWindow.TryGetValue(groupId, out var win))
        {
            win.Enqueue((bSteps, Math.Max(durSec, 0.001f)));
            while (win.Count > StepsWindow) win.Dequeue();
        }

        _totalUpdates[groupId]     = _totalUpdates.GetValueOrDefault(groupId) + 1;
        _totalWorkerSteps[groupId] = _totalWorkerSteps.GetValueOrDefault(groupId) + wSteps;
        _lastStats[groupId]        = stats;
        _lastUpdateDurSec[groupId] = durSec;
        _lastBatchSteps[groupId]   = bSteps;
        _lastUpdateTime[groupId]   = now;

        if (_verbose) GD.Print($"[RL Distributed] Broadcasting weights '{groupId}' to {ConnectedWorkers} worker(s).");
        BroadcastWeights(groupId, trainer);

        if (_monitorInterval > 0 && _totalUpdates[groupId] % _monitorInterval == 0)
            PrintMonitorSummary(groupId, totalSteps, ConnectedWorkers);

        _rolloutsThisRound[groupId]    = 0;
        _workerStepsThisRound[groupId] = 0;
    }

    // ── Monitor console output ────────────────────────────────────────────────

    private void PrintMonitorSummary(string groupId, long totalSteps, int connected)
    {
        var updates  = _totalUpdates.GetValueOrDefault(groupId);
        var wSteps   = _totalWorkerSteps.GetValueOrDefault(groupId);
        var batch    = _lastBatchSteps.GetValueOrDefault(groupId);
        var dur      = _lastUpdateDurSec.GetValueOrDefault(groupId);
        var sps      = CalcStepsPerSec(groupId);
        var stats    = _lastStats.GetValueOrDefault(groupId);
        var sources  = connected + 1;

        var sb = new StringBuilder();
        sb.AppendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        sb.AppendLine($"[RL Distributed]  Update #{updates}  |  {groupId}");
        sb.AppendLine($"  Workers      : {connected}/{_workerCount}  ({sources} sources incl. master)");
        sb.AppendLine($"  Total steps  : {totalSteps + wSteps:N0}  (master: {totalSteps:N0}  workers: {wSteps:N0})");
        sb.AppendLine($"  Batch size   : {batch} worker steps + local rollout");
        sb.AppendLine($"  Steps/sec    : {sps:N0}");
        if (dur > 0.01f) sb.AppendLine($"  Update time  : {dur:F2}s");
        if (stats is not null)
        {
            sb.AppendLine($"  Policy loss  : {stats.PolicyLoss:F4}");
            sb.AppendLine($"  Value loss   : {stats.ValueLoss:F4}");
            sb.AppendLine($"  Entropy      : {stats.Entropy:F4}");
            if (stats.ClipFraction > 0f) sb.AppendLine($"  Clip frac    : {stats.ClipFraction:F4}");
        }
        sb.Append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        GD.Print(sb.ToString());
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void LogRolloutDiagnostic(string groupId, byte[] data, int rolloutIndex)
    {
        List<DistributedTransition> transitions;
        try { transitions = DistributedProtocol.DeserializeRollout(data); }
        catch (Exception ex)
        {
            GD.PushError($"[RL Diagnostic] Failed to deserialize rollout #{rolloutIndex}: {ex.Message}");
            return;
        }

        if (transitions.Count == 0)
        {
            GD.PushWarning($"[RL Diagnostic] Rollout #{rolloutIndex} '{groupId}' is empty.");
            return;
        }

        var n           = transitions.Count;
        var rewardSum   = 0f;
        var rewardMin   = float.MaxValue;
        var rewardMax   = float.MinValue;
        var nonZero     = 0;
        var doneCount   = 0;
        var nanCount    = 0;
        var infCount    = 0;
        var nextObsLen  = transitions[0].NextObservation.Length;

        // Action accumulators — sized from first transition.
        var actionDims = transitions[0].ContinuousActions.Length;
        var actionAbsSum = new float[actionDims];

        foreach (var t in transitions)
        {
            rewardSum += t.Reward;
            if (t.Reward < rewardMin) rewardMin = t.Reward;
            if (t.Reward > rewardMax) rewardMax = t.Reward;
            if (Math.Abs(t.Reward) > 1e-6f) nonZero++;
            if (t.Done) doneCount++;

            for (var i = 0; i < t.ContinuousActions.Length && i < actionDims; i++)
                actionAbsSum[i] += Math.Abs(t.ContinuousActions[i]);

            foreach (var v in t.Observation)
            {
                if (float.IsNaN(v))        nanCount++;
                else if (float.IsInfinity(v)) infCount++;
            }
        }

        var sb = new StringBuilder();
        sb.AppendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        sb.AppendLine($"[RL Diagnostic] Worker rollout #{rolloutIndex}/{DiagnosticRolloutCount}  |  group '{groupId}'");
        sb.AppendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        sb.AppendLine($"  Transitions     : {n}");
        sb.AppendLine($"  Episodes done   : {doneCount}");
        sb.AppendLine($"  Reward  mean    : {rewardSum / n:F5}   min: {rewardMin:F5}   max: {rewardMax:F5}");
        sb.AppendLine($"  Non-zero rewards: {nonZero}/{n}  ({100f * nonZero / n:F1}%)");

        if (actionDims > 0)
        {
            var parts = new string[actionDims];
            for (var i = 0; i < actionDims; i++)
                parts[i] = (actionAbsSum[i] / n).ToString("F3");
            sb.AppendLine($"  |Action| means  : [{string.Join(", ", parts)}]  (each joint 0-1)");

            var maxAbsMean = 0f;
            foreach (var v in actionAbsSum) if (v / n > maxAbsMean) maxAbsMean = v / n;
            if (maxAbsMean < 0.01f)
                sb.AppendLine("  *** WARNING: all actions near zero — policy may not be applied on workers ***");
        }

        sb.Append($"  Obs NaN / Inf   : {nanCount} / {infCount}");
        if (nanCount > 0 || infCount > 0)
            sb.Append("  ← PROBLEM: physics exploded or observations are broken");
        sb.AppendLine();

        sb.Append($"  NextObs present : {(nextObsLen > 0 ? $"yes ({nextObsLen} floats)" : "NO — SAC Bellman targets will be zero!")}");
        if (nextObsLen == 0)
            sb.Append("  ← PROBLEM");
        sb.AppendLine();

        if (nonZero == 0)
            sb.AppendLine("  *** WARNING: ALL rewards are zero — check reward function or env state ***");

        sb.AppendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        GD.Print(sb.ToString());
    }

    private void BroadcastWeights(string groupId, IDistributedTrainer trainer)
    {
        var bytes = trainer.ExportWeights();
        List<WorkerConnection> snapshot;
        lock (_connectionsLock) snapshot = new List<WorkerConnection>(_connections);
        foreach (var conn in snapshot)
        {
            var c = conn; var g = groupId; var w = bytes;
            ThreadPool.QueueUserWorkItem(_ => c.Send(DistributedMessageType.Weights, g, w));
        }
    }

    /// <summary>
    /// Drains all pending worker episode summaries received since the last call.
    /// Call from the main thread each frame to write worker episodes to the metrics log.
    /// </summary>
    public List<(string groupId, List<WorkerEpisodeSummary> summaries)> DrainWorkerEpisodeSummaries()
    {
        var result = new List<(string, List<WorkerEpisodeSummary>)>();
        lock (_rolloutsLock)
        {
            while (_pendingEpisodeSummaries.Count > 0)
                result.Add(_pendingEpisodeSummaries.Dequeue());
        }
        return result;
    }

    /// <summary>
    /// Broadcasts the current curriculum progress to all connected workers so they stay
    /// in sync with the master's curriculum calculation.  Call after each training update.
    /// </summary>
    public void BroadcastCurriculumProgress(float progress)
    {
        var bytes = BitConverter.GetBytes(progress);
        List<WorkerConnection> snapshot;
        lock (_connectionsLock) snapshot = new List<WorkerConnection>(_connections);
        foreach (var conn in snapshot)
        {
            var c = conn; var b = bytes;
            ThreadPool.QueueUserWorkItem(_ => c.Send(DistributedMessageType.CurriculumSync, "", b));
        }
    }

    public void Shutdown()
    {
        List<WorkerConnection> snapshot;
        lock (_connectionsLock) snapshot = new List<WorkerConnection>(_connections);
        foreach (var conn in snapshot)
            conn.Send(DistributedMessageType.Shutdown, "", Array.Empty<byte>());
    }

    public void Dispose()
    {
        _running = false;
        _listener?.Stop();
    }
}
