using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

internal sealed class PolicyPool
{
    private readonly List<OpponentRecord> _records = new();
    private readonly RandomNumberGenerator _rng;
    private readonly int   _maxPoolSize;
    private readonly bool  _pfspEnabled;
    private readonly float _pfspAlpha;

    public string LatestCheckpointPath { get; set; } = string.Empty;
    public IReadOnlyList<OpponentRecord> Records => _records;
    public float AverageWinRate => _records.Count == 0
        ? 0.5f
        : _records.Average(r => r.WinRate);

    public PolicyPool(int maxPoolSize, bool pfspEnabled, float pfspAlpha, RandomNumberGenerator rng)
    {
        _maxPoolSize = Math.Max(1, maxPoolSize);
        _pfspEnabled = pfspEnabled;
        _pfspAlpha   = pfspAlpha;
        _rng         = rng;
    }

    /// <summary>Add a snapshot to the pool. No-op if the key already exists. Evicts the
    /// easiest opponent (highest win rate) when the pool is full.</summary>
    public void AddSnapshot(string checkpointPath, string snapshotKey, float learnerElo)
    {
        if (_records.Any(r => r.SnapshotKey == snapshotKey))
            return;

        if (_records.Count >= _maxPoolSize)
        {
            var easiest = _records.OrderByDescending(r => r.WinRate).First();
            _records.Remove(easiest);
        }

        _records.Add(new OpponentRecord
        {
            CheckpointPath = checkpointPath,
            SnapshotKey    = snapshotKey,
            SnapshotElo    = learnerElo,
        });
    }

    /// <summary>Sample a record using PFSP weights (or uniform if PFSP is disabled / pool has
    /// one entry). Returns null only if the pool is empty.</summary>
    public OpponentRecord? SampleHistorical()
    {
        if (_records.Count == 0)
            return null;

        if (_records.Count == 1 || !_pfspEnabled)
            return _records[_rng.RandiRange(0, _records.Count - 1)];

        var weights     = _records.Select(r => r.PfspWeight(_pfspAlpha)).ToArray();
        var totalWeight = weights.Sum();
        if (totalWeight <= 0f)
            return _records[_rng.RandiRange(0, _records.Count - 1)];

        var sample     = _rng.Randf() * totalWeight;
        var cumulative = 0f;
        for (var i = 0; i < _records.Count; i++)
        {
            cumulative += weights[i];
            if (sample <= cumulative)
                return _records[i];
        }

        return _records[^1];
    }

    /// <summary>Increment episode count (and win count) for the given snapshot key.
    /// No-op if the key is not in the pool (e.g. latest-policy matchups).</summary>
    public void RecordOutcome(string snapshotKey, bool learnerWon)
    {
        var record = _records.FirstOrDefault(r => r.SnapshotKey == snapshotKey);
        if (record is null)
            return;

        record.Episodes += 1;
        if (learnerWon)
            record.Wins += 1;
    }
}
