using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace RlAgentPlugin.Runtime;

// ── Message types ────────────────────────────────────────────────────────────

public enum DistributedMessageType : byte
{
    /// <summary>Worker → Master on connect.  Payload: empty (group-id field carries the group).</summary>
    Hello   = 1,
    /// <summary>Master → Worker.  Payload: serialised weights for <c>groupId</c>.</summary>
    Weights = 2,
    /// <summary>Worker → Master.  Payload: serialised rollout for <c>groupId</c>.</summary>
    Rollout = 3,
    /// <summary>Master → Worker.  Payload: empty.  Tells the worker to exit cleanly.</summary>
    Shutdown = 5,
    /// <summary>Master → Worker.  Payload: 4-byte little-endian float — current curriculum progress in [0, 1].</summary>
    CurriculumSync = 6,
    /// <summary>Worker → Master.  Payload: batch of episode summaries (reward, steps, breakdown) for metrics aggregation.</summary>
    EpisodeSummary = 7,
    /// <summary>Worker → Master.  Payload: UTF-8 log string to be printed on the master console.</summary>
    LogMessage = 8,
}

// ── Wire-format helpers ───────────────────────────────────────────────────────

/// <summary>
/// Binary protocol for master ↔ worker communication over a TCP stream.
///
/// Every message:
/// <code>
/// [1 byte  : DistributedMessageType]
/// [4 bytes : group-id length (int32 LE)]
/// [N bytes : group-id UTF-8]
/// [4 bytes : payload length (int32 LE)]
/// [M bytes : payload]
/// </code>
/// </summary>
public static class DistributedProtocol
{
    public static void WriteMessage(
        BinaryWriter writer,
        DistributedMessageType type,
        string groupId,
        byte[] payload)
    {
        writer.Write((byte)type);
        var groupBytes = Encoding.UTF8.GetBytes(groupId);
        writer.Write(groupBytes.Length);
        writer.Write(groupBytes);
        writer.Write(payload.Length);
        if (payload.Length > 0)
            writer.Write(payload);
        writer.Flush();
    }

    public static (DistributedMessageType type, string groupId, byte[] payload)
        ReadMessage(BinaryReader reader)
    {
        var type      = (DistributedMessageType)reader.ReadByte();
        var groupLen  = reader.ReadInt32();
        var groupId   = Encoding.UTF8.GetString(reader.ReadBytes(groupLen));
        var payLen    = reader.ReadInt32();
        var payload   = payLen > 0 ? reader.ReadBytes(payLen) : Array.Empty<byte>();
        return (type, groupId, payload);
    }

    // ── Weight serialisation ─────────────────────────────────────────────────

    public static byte[] SerializeWeights(float[] weightBuffer, int[] shapeBuffer)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        bw.Write(weightBuffer.Length);
        foreach (var f in weightBuffer) bw.Write(f);
        bw.Write(shapeBuffer.Length);
        foreach (var i in shapeBuffer) bw.Write(i);
        bw.Flush();
        return ms.ToArray();
    }

    public static (float[] weights, int[] shapes) DeserializeWeights(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        var wLen    = br.ReadInt32();
        var weights = new float[wLen];
        for (var i = 0; i < wLen; i++) weights[i] = br.ReadSingle();
        var sLen   = br.ReadInt32();
        var shapes = new int[sLen];
        for (var i = 0; i < sLen; i++) shapes[i] = br.ReadInt32();
        return (weights, shapes);
    }

    // ── Rollout serialisation ────────────────────────────────────────────────

    /// <summary>
    /// Per-transition layout:
    /// obs_len, obs[], discrete_action, cont_len, cont[], reward, done, next_obs_len, next_obs[], log_prob, value, next_value
    /// </summary>
    public static byte[] SerializeRollout(IReadOnlyList<DistributedTransition> transitions)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        bw.Write(transitions.Count);
        foreach (var t in transitions)
        {
            bw.Write(t.Observation.Length);
            foreach (var f in t.Observation) bw.Write(f);
            bw.Write(t.DiscreteAction);
            bw.Write(t.ContinuousActions.Length);
            foreach (var f in t.ContinuousActions) bw.Write(f);
            bw.Write(t.Reward);
            bw.Write(t.Done);
            bw.Write(t.NextObservation.Length);
            foreach (var f in t.NextObservation) bw.Write(f);
            bw.Write(t.OldLogProbability);
            bw.Write(t.Value);
            bw.Write(t.NextValue);
        }
        bw.Flush();
        return ms.ToArray();
    }

    /// <summary>
    /// Counts completed episodes (transitions where <c>Done == true</c>) in a serialized rollout
    /// without allocating transition objects.
    /// </summary>
    public static int CountEpisodesInRollout(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        var count    = br.ReadInt32();
        var episodes = 0;
        for (var i = 0; i < count; i++)
        {
            var obsLen = br.ReadInt32();
            ms.Seek(obsLen * sizeof(float), SeekOrigin.Current); // skip obs
            ms.Seek(sizeof(int), SeekOrigin.Current);             // skip discrete_action
            var contLen = br.ReadInt32();
            ms.Seek(contLen * sizeof(float), SeekOrigin.Current); // skip cont
            ms.Seek(sizeof(float), SeekOrigin.Current);           // skip reward
            if (br.ReadBoolean()) episodes++;                      // done
            var nextObsLen = br.ReadInt32();
            ms.Seek(nextObsLen * sizeof(float), SeekOrigin.Current); // skip next_obs
            ms.Seek(sizeof(float) * 3, SeekOrigin.Current);       // skip log_prob, value, next_value
        }
        return episodes;
    }

    // ── Episode summary serialisation ────────────────────────────────────────

    public static byte[] SerializeEpisodeSummaries(IList<WorkerEpisodeSummary> summaries)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        bw.Write(summaries.Count);
        foreach (var s in summaries)
        {
            bw.Write(s.Reward);
            bw.Write(s.Steps);
            var breakdown = s.RewardBreakdown ?? new Dictionary<string, float>();
            bw.Write(breakdown.Count);
            foreach (var (tag, amount) in breakdown)
            {
                var tagBytes = Encoding.UTF8.GetBytes(tag);
                bw.Write(tagBytes.Length);
                bw.Write(tagBytes);
                bw.Write(amount);
            }
        }
        bw.Flush();
        return ms.ToArray();
    }

    public static List<WorkerEpisodeSummary> DeserializeEpisodeSummaries(byte[] data)
    {
        var result = new List<WorkerEpisodeSummary>();
        if (data.Length == 0) return result;
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        var count = br.ReadInt32();
        for (var i = 0; i < count; i++)
        {
            var reward = br.ReadSingle();
            var steps  = br.ReadInt32();
            var bdCount = br.ReadInt32();
            var breakdown = new Dictionary<string, float>(StringComparer.Ordinal);
            for (var j = 0; j < bdCount; j++)
            {
                var tagLen = br.ReadInt32();
                var tag    = Encoding.UTF8.GetString(br.ReadBytes(tagLen));
                breakdown[tag] = br.ReadSingle();
            }
            result.Add(new WorkerEpisodeSummary { Reward = reward, Steps = steps, RewardBreakdown = breakdown });
        }
        return result;
    }

    public static List<DistributedTransition> DeserializeRollout(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        var count  = br.ReadInt32();
        var result = new List<DistributedTransition>(count);
        for (var i = 0; i < count; i++)
        {
            var obsLen = br.ReadInt32();
            var obs    = new float[obsLen];
            for (var j = 0; j < obsLen; j++) obs[j] = br.ReadSingle();
            var action  = br.ReadInt32();
            var contLen = br.ReadInt32();
            var cont    = new float[contLen];
            for (var j = 0; j < contLen; j++) cont[j] = br.ReadSingle();
            var reward  = br.ReadSingle();
            var done    = br.ReadBoolean();
            var nextObsLen = br.ReadInt32();
            var nextObs = new float[nextObsLen];
            for (var j = 0; j < nextObsLen; j++) nextObs[j] = br.ReadSingle();
            result.Add(new DistributedTransition
            {
                Observation        = obs,
                DiscreteAction     = action,
                ContinuousActions  = cont,
                Reward             = reward,
                Done               = done,
                NextObservation    = nextObs,
                OldLogProbability  = br.ReadSingle(),
                Value              = br.ReadSingle(),
                NextValue          = br.ReadSingle(),
            });
        }
        return result;
    }
}

// ── Episode summary (worker → master) ────────────────────────────────────────

/// <summary>
/// Lightweight episode outcome sent by workers to the master for metrics aggregation.
/// The master writes these to the metrics log so RLDash shows episodes from all processes.
/// </summary>
public struct WorkerEpisodeSummary
{
    public float Reward;
    public int   Steps;
    /// <summary>Per-component reward breakdown. Empty dict when no tags were used.</summary>
    public Dictionary<string, float> RewardBreakdown;
}

// ── Shared data transfer object ───────────────────────────────────────────────

public struct DistributedTransition
{
    public float[] Observation;
    public int     DiscreteAction;
    public float[] ContinuousActions;
    public float   Reward;
    public bool    Done;
    public float[] NextObservation;   // SAC Bellman target; PPO sends Array.Empty
    public float   OldLogProbability;
    public float   Value;
    public float   NextValue;
}
