using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime.Imitation;

/// <summary>
/// Reads a .rldem binary file into memory.
/// Binary format (little-endian):
///   Header (28 bytes):
///     [4]  magic "RLDE"
///     [4]  version (uint32)
///     [4]  obs_size (uint32)
///     [4]  discrete_action_count (uint32, 0 = none)
///     [4]  continuous_action_dims (uint32, 0 = none)
///     [8]  frame_count (uint64)
///   Per frame:
///     [1]  agent_slot (byte)
///     [obs_size*4] observations (float32[])
///     [4]  discrete_action (int32)
///     [cont_dims*4] continuous_actions (float32[])
///     [4]  reward (float32)
///     [1]  done (byte, 0 or 1)
/// </summary>
internal sealed class DemonstrationDataset
{
    public int ObsSize { get; }
    public int DiscreteActionCount { get; }
    public int ContinuousActionDims { get; }
    public IReadOnlyList<DemonstrationFrame> Frames { get; }

    private DemonstrationDataset(int obsSize, int discreteCount, int contDims, List<DemonstrationFrame> frames)
    {
        ObsSize = obsSize;
        DiscreteActionCount = discreteCount;
        ContinuousActionDims = contDims;
        Frames = frames;
    }

    /// <summary>Loads a .rldem file from a res:// or absolute path. Returns null on failure.</summary>
    public static DemonstrationDataset? Open(string path)
    {
        if (!FileAccess.FileExists(path))
        {
            GD.PushError($"[DemonstrationDataset] File not found: {path}");
            return null;
        }

        using var file = FileAccess.Open(path, FileAccess.ModeFlags.Read);
        if (file is null)
        {
            GD.PushError($"[DemonstrationDataset] Could not open: {path} — {FileAccess.GetOpenError()}");
            return null;
        }

        // Magic
        var magic = new byte[4];
        for (var i = 0; i < 4; i++) magic[i] = file.Get8();
        if (magic[0] != 'R' || magic[1] != 'L' || magic[2] != 'D' || magic[3] != 'E')
        {
            GD.PushError($"[DemonstrationDataset] Invalid magic in {path}");
            return null;
        }

        var version = file.Get32();
        if (version != 1)
        {
            GD.PushError($"[DemonstrationDataset] Unsupported version {version} in {path}");
            return null;
        }

        var obsSize = (int)file.Get32();
        var discreteCount = (int)file.Get32();
        var contDims = (int)file.Get32();
        var frameCount = (long)file.Get64();

        var frames = new List<DemonstrationFrame>((int)Math.Min(frameCount, 1_000_000));

        for (long f = 0; f < frameCount; f++)
        {
            var slot = (int)file.Get8();

            var obs = new float[obsSize];
            for (var i = 0; i < obsSize; i++) obs[i] = file.GetFloat();

            var discrete = (int)file.Get32();

            var continuous = new float[contDims];
            for (var i = 0; i < contDims; i++) continuous[i] = file.GetFloat();

            var reward = file.GetFloat();
            var done = file.Get8() != 0;

            frames.Add(new DemonstrationFrame
            {
                AgentSlot = slot,
                Obs = obs,
                DiscreteAction = discrete,
                ContinuousActions = continuous,
                Reward = reward,
                Done = done,
            });
        }

        return new DemonstrationDataset(obsSize, discreteCount, contDims, frames);
    }

    /// <summary>Returns a metadata summary without loading all frames.</summary>
    public static DemonstrationDatasetMeta? ReadMeta(string path)
    {
        if (!FileAccess.FileExists(path)) return null;
        using var file = FileAccess.Open(path, FileAccess.ModeFlags.Read);
        if (file is null) return null;

        var magic = new byte[4];
        for (var i = 0; i < 4; i++) magic[i] = file.Get8();
        if (magic[0] != 'R' || magic[1] != 'L' || magic[2] != 'D' || magic[3] != 'E') return null;

        file.Get32(); // version
        var obsSize = (int)file.Get32();
        var discreteCount = (int)file.Get32();
        var contDims = (int)file.Get32();
        var frameCount = (long)file.Get64();

        return new DemonstrationDatasetMeta(path, obsSize, discreteCount, contDims, frameCount);
    }
}

internal sealed record DemonstrationDatasetMeta(
    string Path,
    int ObsSize,
    int DiscreteActionCount,
    int ContinuousActionDims,
    long FrameCount);

/// <summary>
/// Streams frames to a .rldem file during a recording session.
/// Call <see cref="WriteFrame"/> for each step, then <see cref="Close"/> when done.
/// Close patches the frame_count field in the header.
/// </summary>
internal sealed class DemonstrationWriter : IDisposable
{
    private const ulong FrameCountOffset = 20; // bytes: 4 magic + 4 version + 4 obs + 4 disc + 4 cont

    private FileAccess? _file;
    private readonly int _obsSize;
    private readonly int _contDims;
    private long _frameCount;

    private DemonstrationWriter(FileAccess file, int obsSize, int contDims)
    {
        _file = file;
        _obsSize = obsSize;
        _contDims = contDims;
    }

    /// <summary>Opens a new .rldem file for writing. Returns null on failure.</summary>
    public static DemonstrationWriter? Create(string absolutePath, int obsSize, int discreteCount, int contDims)
    {
        // Ensure directory exists.
        var dir = System.IO.Path.GetDirectoryName(absolutePath);
        if (!string.IsNullOrEmpty(dir))
            System.IO.Directory.CreateDirectory(dir);

        var file = FileAccess.Open(absolutePath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            GD.PushError($"[DemonstrationWriter] Could not create: {absolutePath} — {FileAccess.GetOpenError()}");
            return null;
        }

        // Write header.
        file.Store8((byte)'R');
        file.Store8((byte)'L');
        file.Store8((byte)'D');
        file.Store8((byte)'E');
        file.Store32(1); // version
        file.Store32((uint)obsSize);
        file.Store32((uint)discreteCount);
        file.Store32((uint)contDims);
        file.Store64(0); // frame_count placeholder — patched in Close()

        return new DemonstrationWriter(file, obsSize, contDims);
    }

    public void WriteFrame(DemonstrationFrame frame)
    {
        if (_file is null) return;

        _file.Store8((byte)Math.Clamp(frame.AgentSlot, 0, 255));

        var obs = frame.Obs;
        for (var i = 0; i < _obsSize; i++)
            _file.StoreFloat(i < obs.Length ? obs[i] : 0f);

        _file.Store32((uint)frame.DiscreteAction); // -1 stored as 0xFFFFFFFF, cast back to int on read

        var cont = frame.ContinuousActions;
        for (var i = 0; i < _contDims; i++)
            _file.StoreFloat(i < cont.Length ? cont[i] : 0f);

        _file.StoreFloat(frame.Reward);
        _file.Store8(frame.Done ? (byte)1 : (byte)0);

        _frameCount++;
    }

    /// <summary>Patches the frame_count in the header and closes the file.</summary>
    public void Close()
    {
        if (_file is null) return;
        _file.Seek(FrameCountOffset);
        _file.Store64((ulong)_frameCount);
        _file.Close();
        _file = null;
    }

    public void Dispose() => Close();

    public long FrameCount => _frameCount;
}
