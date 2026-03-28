using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Runs on each worker process.  Connects to the master, sends completed rollouts,
/// and applies weight updates received from the master.
///
/// Call <see cref="TickUpdate"/> each physics frame instead of <c>trainer.TryUpdate()</c>.
/// Check <see cref="ShouldQuit"/> each frame — when true the master is gone and the worker
/// should call <c>GetTree().Quit()</c>.
/// </summary>
public sealed class DistributedWorker : IDisposable
{
    private readonly string _masterHost;
    private readonly int    _masterPort;
    private readonly Dictionary<string, IDistributedTrainer> _trainers;

    private TcpClient?    _client;
    private BinaryWriter? _writer;
    private readonly object _writeLock = new();

    // Per-group flag: true while waiting for updated weights after sending a rollout.
    private readonly HashSet<string> _waitingForWeights = new(StringComparer.Ordinal);

    // Weight updates and curriculum syncs arrive on the read thread; applied on the main thread.
    private readonly Queue<(string groupId, byte[] data)> _pendingWeights            = new();
    private readonly Queue<float>                         _pendingCurriculumUpdates  = new();
    private readonly object                               _pendingLock               = new();

    // Episode summaries accumulated on the main thread; sent alongside the next rollout.
    private readonly Dictionary<string, List<WorkerEpisodeSummary>> _pendingSummaries = new(StringComparer.Ordinal);
    private readonly object                                          _summariesLock    = new();

    private Thread?          _readThread;
    private volatile bool    _running;

    // ── Health-check state ────────────────────────────────────────────────────

    /// <summary>
    /// Set to true after the master is unreachable for
    /// <see cref="MaxMissedChecks"/> × <see cref="CheckIntervalMs"/> milliseconds.
    /// Poll this from the main thread; when true call <c>GetTree().Quit()</c>.
    /// </summary>
    public volatile bool ShouldQuit;

    /// <summary>
    /// True while waiting for updated weights from the master (PPO on-policy pause).
    /// The worker is intentionally idle during this window — health monitors should
    /// exclude this time from throughput calculations to avoid false drop warnings.
    /// Only meaningful for on-policy (PPO) trainers; always false for off-policy (SAC).
    /// </summary>
    public bool IsWaitingForWeights => _waitingForWeights.Count > 0;

    private volatile bool _connectionLost;
    private int           _missedChecks;

    private const int CheckIntervalMs = 2_000;
    private const int MaxMissedChecks = 5; // ≈ 10 seconds before self-termination

    // ── Construction ──────────────────────────────────────────────────────────

    public DistributedWorker(
        string masterHost,
        int    masterPort,
        Dictionary<string, IDistributedTrainer> trainers)
    {
        _masterHost = masterHost;
        _masterPort = masterPort;
        _trainers   = trainers;
    }

    // ── Connect ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Connects to the master (retries up to 20 × 500 ms = 10 s so the master has time to start),
    /// then starts the background read thread and the health-check watchdog.
    /// </summary>
    public void Connect()
    {
        _client         = new TcpClient();
        _client.NoDelay = true;

        for (var attempt = 0; attempt < 20; attempt++)
        {
            try   { _client.Connect(_masterHost, _masterPort); break; }
            catch (SocketException) when (attempt < 19) { Thread.Sleep(500); }
        }

        if (!_client.Connected)
        {
            GD.PushError($"[RL Distributed] Worker could not connect to {_masterHost}:{_masterPort}.");
            ShouldQuit = true;
            return;
        }

        var stream = _client.GetStream();
        _writer  = new BinaryWriter(stream);
        _running = true;

        foreach (var groupId in _trainers.Keys)
        {
            lock (_writeLock)
                DistributedProtocol.WriteMessage(
                    _writer, DistributedMessageType.Hello, groupId, Array.Empty<byte>());
        }

        _readThread = new Thread(() => ReadLoop(new BinaryReader(stream)))
        {
            IsBackground = true,
            Name         = "DistWorker-Read",
        };
        _readThread.Start();

        new Thread(HealthCheckLoop)
        {
            IsBackground = true,
            Name         = "DistWorker-Health",
        }.Start();

        GD.Print($"[RL Distributed] Worker connected to {_masterHost}:{_masterPort}.");
    }

    // ── Background: read + health ─────────────────────────────────────────────

    private void ReadLoop(BinaryReader reader)
    {
        try
        {
            while (_running)
            {
                var (type, groupId, payload) = DistributedProtocol.ReadMessage(reader);
                switch (type)
                {
                    case DistributedMessageType.Weights:
                        lock (_pendingLock)
                            _pendingWeights.Enqueue((groupId, payload));
                        break;

                    case DistributedMessageType.CurriculumSync:
                        if (payload.Length == 4)
                        {
                            var progress = BitConverter.ToSingle(payload, 0);
                            lock (_pendingLock)
                                _pendingCurriculumUpdates.Enqueue(progress);
                        }
                        break;

                    case DistributedMessageType.Shutdown:
                        GD.Print("[RL Distributed] Worker received shutdown — exiting.");
                        _running = false;
                        ShouldQuit = true;
                        return;
                }
            }
        }
        catch (Exception ex) when (_running)
        {
            GD.PushWarning($"[RL Distributed] Master connection lost: {ex.Message}");
            _connectionLost = true;
        }
    }

    /// <summary>
    /// Runs every <see cref="CheckIntervalMs"/> ms on a background thread.
    /// Increments a missed-check counter whenever the connection is gone;
    /// sets <see cref="ShouldQuit"/> after <see cref="MaxMissedChecks"/> consecutive misses.
    /// </summary>
    private void HealthCheckLoop()
    {
        while (_running && !ShouldQuit)
        {
            Thread.Sleep(CheckIntervalMs);

            var alive = !_connectionLost && (_client?.Connected ?? false);
            if (alive)
            {
                _missedChecks = 0;
                continue;
            }

            _missedChecks++;
            GD.Print($"[RL Distributed] Master unreachable — missed check {_missedChecks}/{MaxMissedChecks}.");

            if (_missedChecks >= MaxMissedChecks)
            {
                GD.Print("[RL Distributed] Master gone — worker self-terminating.");
                ShouldQuit = true;
                return;
            }
        }
    }

    // ── Main-thread API ───────────────────────────────────────────────────────

    /// <summary>
    /// Drop-in replacement for <c>trainer.TryUpdate()</c>.
    /// Never trains locally — exports rollouts to the master and applies incoming weights.
    /// Always returns null.
    /// </summary>
    public TrainerUpdateStats? TickUpdate(
        IDistributedTrainer trainer,
        string groupId,
        long   totalSteps,
        long   episodeCount)
    {
        if (ShouldQuit || _connectionLost) return null;

        // Apply weight updates that arrived on the background thread.
        lock (_pendingLock)
        {
            while (_pendingWeights.Count > 0)
            {
                var (wGroup, weightData) = _pendingWeights.Dequeue();
                if (_trainers.TryGetValue(wGroup, out var t))
                {
                    t.ImportWeights(weightData);
                    _waitingForWeights.Remove(wGroup);
                }
            }
        }

        // On-policy (PPO): block until master returns fresh weights.
        if (!trainer.IsOffPolicy && _waitingForWeights.Contains(groupId)) return null;

        if (!trainer.IsRolloutReady) return null;

        var rolloutBytes = trainer.ExportAndClearRollout();

        if (!trainer.IsOffPolicy)
        {
            // PPO: block until master returns fresh weights (ensures on-policy data).
            _waitingForWeights.Add(groupId);
        }
        // SAC: don't block — continue collecting immediately.
        // Incoming weights are applied eagerly in the weight-drain loop above.

        if (_writer is not null)
        {
            // Drain and send episode summaries alongside this rollout.
            List<WorkerEpisodeSummary>? summaries = null;
            lock (_summariesLock)
            {
                if (_pendingSummaries.TryGetValue(groupId, out var pending) && pending.Count > 0)
                {
                    summaries = new List<WorkerEpisodeSummary>(pending);
                    pending.Clear();
                }
            }

            lock (_writeLock)
            {
                DistributedProtocol.WriteMessage(_writer, DistributedMessageType.Rollout, groupId, rolloutBytes);
                if (summaries is not null)
                {
                    var summaryBytes = DistributedProtocol.SerializeEpisodeSummaries(summaries);
                    DistributedProtocol.WriteMessage(_writer, DistributedMessageType.EpisodeSummary, groupId, summaryBytes);
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Queues an episode summary to be sent to the master with the next rollout for <paramref name="groupId"/>.
    /// Call from the main thread each time an episode completes on a worker.
    /// </summary>
    public void EnqueueEpisodeSummary(string groupId, float reward, int steps, IReadOnlyDictionary<string, float>? breakdown)
    {
        var summary = new WorkerEpisodeSummary
        {
            Reward          = reward,
            Steps           = steps,
            RewardBreakdown = breakdown is not null
                ? new Dictionary<string, float>(breakdown, StringComparer.Ordinal)
                : new Dictionary<string, float>(),
        };
        lock (_summariesLock)
        {
            if (!_pendingSummaries.TryGetValue(groupId, out var list))
                _pendingSummaries[groupId] = list = new List<WorkerEpisodeSummary>();
            list.Add(summary);
        }
    }

    /// <summary>
    /// Sends a log string to the master to be printed on its console.
    /// Safe to call from the main thread at any time.
    /// </summary>
    public void SendLogMessage(string message)
    {
        if (_writer is null) return;
        var payload = System.Text.Encoding.UTF8.GetBytes(message);
        lock (_writeLock)
            DistributedProtocol.WriteMessage(_writer, DistributedMessageType.LogMessage, "", payload);
    }

    /// <summary>
    /// Returns the latest curriculum progress broadcast received from the master since the
    /// last call, or <c>null</c> if no update has arrived.  Drains all queued values and
    /// returns only the most recent (ignoring stale intermediate frames).
    /// </summary>
    public float? ConsumeCurriculumProgress()
    {
        lock (_pendingLock)
        {
            if (_pendingCurriculumUpdates.Count == 0) return null;
            float latest = 0f;
            while (_pendingCurriculumUpdates.Count > 0)
                latest = _pendingCurriculumUpdates.Dequeue();
            return latest;
        }
    }

    public void Dispose()
    {
        _running = false;
        _client?.Close();
    }
}
