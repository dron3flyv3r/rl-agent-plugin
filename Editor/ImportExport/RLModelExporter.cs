using System;
using System.IO;
using System.Text;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

/// <summary>
/// Exports a trained checkpoint to a compact self-describing binary .rlmodel file.
///
/// Binary layout (little-endian):
///   [8  bytes] magic "RLMODEL\0"
///   [2  bytes] uint16 version = 2
///   [4  bytes] int32  obs_size
///   [4  bytes] int32  action_dims
///   [4  bytes] int32  layer_count
///   Per layer:
///     [4  bytes] int32  in_size
///     [4  bytes] int32  out_size
///     [4  bytes] int32  activation  (0=linear, 1=Tanh, 2=Relu — matches internal encoding)
///     [n*4 bytes] float32[in_size * out_size]  weights (row-major)
///     [m*4 bytes] float32[out_size]            biases
///   [4  bytes] int32  metadata_json_length
///   [n bytes] UTF-8 metadata JSON
/// </summary>
public static class RLModelExporter
{
    private static readonly byte[] Magic =
    {
        (byte)'R', (byte)'L', (byte)'M', (byte)'O',
        (byte)'D', (byte)'E', (byte)'L', 0,
    };

    /// <summary>
    /// Reads a checkpoint JSON at <paramref name="checkpointAbsPath"/> and writes a
    /// .rlmodel binary to <paramref name="destAbsPath"/>.
    /// Returns <see cref="Error.Ok"/> on success, <see cref="Error.Failed"/> otherwise.
    /// </summary>
    public static Error Export(string checkpointAbsPath, string destAbsPath)
    {
        checkpointAbsPath = ResolveCheckpointSourcePath(checkpointAbsPath);
        var checkpoint = LoadCheckpointJson(checkpointAbsPath);
        if (checkpoint is null) return Error.Failed;

        var rawShapes = checkpoint.LayerShapeBuffer;

        // Format v6+ prepends a flat (0) or multi-stream (1) marker before the typed layer
        // shapes. The .rlmodel binary format does not include this marker, so strip it.
        // We detect it when (length - 1) is divisible by 4 but length itself is not.
        if (rawShapes.Length > 0 && rawShapes.Length % 4 != 0 && (rawShapes.Length - 1) % 4 == 0)
        {
            if (rawShapes[0] == 1)
            {
                GD.PushError($"[RLModelExporter] Multi-stream checkpoints cannot be exported to .rlmodel format: {checkpointAbsPath}");
                return Error.Failed;
            }
            rawShapes = rawShapes[1..]; // strip flat marker
        }

        if (!TryNormalizeDenseLayerShapes(rawShapes, checkpointAbsPath, out var shapes))
            return Error.Failed;

        var weights = checkpoint.WeightBuffer;

        var layerCount = shapes.Length / 3;
        var obsSize = checkpoint.ObservationSize;
        var actionDims = checkpoint.DiscreteActionCount > 0
            ? checkpoint.DiscreteActionCount
            : checkpoint.ContinuousActionDimensions;
        var metadataJson = checkpoint.CreateMetadataJson();
        var metadataBytes = Encoding.UTF8.GetBytes(metadataJson);

        try
        {
            var dir = Path.GetDirectoryName(destAbsPath);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            using var stream = File.Open(destAbsPath, FileMode.Create, System.IO.FileAccess.Write);
            using var writer = new BinaryWriter(stream);

            writer.Write(Magic);
            writer.Write((ushort)RLCheckpoint.CurrentFormatVersion);
            writer.Write(obsSize);
            writer.Write(actionDims);
            writer.Write(layerCount);

            var weightOffset = 0;
            for (var i = 0; i < layerCount; i++)
            {
                var inSize     = shapes[i * 3];
                var outSize    = shapes[i * 3 + 1];
                var activation = shapes[i * 3 + 2];

                writer.Write(inSize);
                writer.Write(outSize);
                writer.Write(activation);

                var numWeights = inSize * outSize;
                for (var j = 0; j < numWeights; j++)
                    writer.Write(weights[weightOffset++]);
                for (var j = 0; j < outSize; j++)
                    writer.Write(weights[weightOffset++]);
            }

            writer.Write(metadataBytes.Length);
            writer.Write(metadataBytes);

            GD.Print($"[RLModelExporter] Exported {layerCount} layers → {destAbsPath}");
            return Error.Ok;
        }
        catch (Exception ex)
        {
            GD.PushError($"[RLModelExporter] Export failed: {ex.Message}");
            return Error.Failed;
        }
    }

    /// <summary>
    /// Searches <paramref name="runDirAbsPath"/> for a checkpoint JSON file
    /// (any .json that is not status.json or meta.json).
    /// Returns the absolute path, or null if none found.
    /// </summary>
    public static string? FindCheckpointInRunDir(string runDirAbsPath)
    {
        if (!Directory.Exists(runDirAbsPath)) return null;

        // Prefer the ZIP format (.rlcheckpoint) over plain JSON.
        foreach (var file in Directory.GetFiles(runDirAbsPath, "*.rlcheckpoint"))
            return file;

        foreach (var file in Directory.GetFiles(runDirAbsPath, "*.json"))
        {
            var name = Path.GetFileName(file);
            if (name == "status.json" || name == "meta.json") continue;
            return file;
        }

        return null;
    }

    // ── Checkpoint loading ────────────────────────────────────────────────────

    private static RLCheckpoint? LoadCheckpointJson(string absPath)
    {
        var checkpoint = RLCheckpoint.LoadFromFile(absPath);
        if (checkpoint is null)
        {
            GD.PushError($"[RLModelExporter] Failed to parse checkpoint: {absPath}");
        }

        return checkpoint;
    }

    private static string ResolveCheckpointSourcePath(string checkpointAbsPath)
    {
        if (checkpointAbsPath.EndsWith(".meta.json", StringComparison.Ordinal))
        {
            var fullCheckpointPath = checkpointAbsPath.Replace(".meta.json", ".json", StringComparison.Ordinal);
            if (File.Exists(fullCheckpointPath))
            {
                return fullCheckpointPath;
            }
        }

        return checkpointAbsPath;
    }

    private static bool TryNormalizeDenseLayerShapes(int[] shapeBuffer, string checkpointAbsPath, out int[] denseShapes)
    {
        denseShapes = Array.Empty<int>();

        if (shapeBuffer.Length == 0)
        {
            GD.PushError($"[RLModelExporter] Invalid LayerShapeBuffer length 0 in {checkpointAbsPath}");
            return false;
        }

        if (shapeBuffer.Length % 4 == 0)
        {
            var normalized = new int[(shapeBuffer.Length / 4) * 3];
            for (int sourceIndex = 0, targetIndex = 0; sourceIndex < shapeBuffer.Length; sourceIndex += 4)
            {
                var layerKind = shapeBuffer[sourceIndex];
                if (layerKind != (int)RLLayerKind.Dense)
                {
                    GD.PushError($"[RLModelExporter] Unsupported layer kind {layerKind} in {checkpointAbsPath}");
                    return false;
                }

                normalized[targetIndex++] = shapeBuffer[sourceIndex + 1];
                normalized[targetIndex++] = shapeBuffer[sourceIndex + 2];
                normalized[targetIndex++] = shapeBuffer[sourceIndex + 3];
            }

            denseShapes = normalized;
            return true;
        }

        if (shapeBuffer.Length % 3 == 0)
        {
            denseShapes = shapeBuffer;
            return true;
        }

        GD.PushError($"[RLModelExporter] Invalid LayerShapeBuffer length {shapeBuffer.Length} in {checkpointAbsPath}");
        return false;
    }
}
