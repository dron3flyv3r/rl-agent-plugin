using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Loads a trained model from a <c>.rlmodel</c> binary file and converts it
/// into an <see cref="RLCheckpoint"/> so the existing inference path can use it.
/// </summary>
public static class RLModelLoader
{
    private static readonly byte[] ExpectedMagic =
    {
        (byte)'R', (byte)'L', (byte)'M', (byte)'O',
        (byte)'D', (byte)'E', (byte)'L', 0,
    };

    /// <summary>
    /// Loads a <c>.rlmodel</c> file from <paramref name="path"/> (res://, user://, or
    /// an absolute filesystem path) and returns a populated <see cref="RLCheckpoint"/>,
    /// or <c>null</c> on failure.
    /// </summary>
    public static RLCheckpoint? LoadFromFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            GD.PushWarning("[RLModelLoader] Empty path.");
            return null;
        }

        // Resolve Godot virtual paths to absolute paths.
        var absPath = (path.StartsWith("res://") || path.StartsWith("user://"))
            ? ProjectSettings.GlobalizePath(path)
            : path;

        if (!File.Exists(absPath))
        {
            GD.PushWarning($"[RLModelLoader] File not found: {absPath}");
            return null;
        }

        try
        {
            using var stream = File.OpenRead(absPath);
            using var reader = new BinaryReader(stream);

            // ── Header ────────────────────────────────────────────────────────
            var magic = reader.ReadBytes(8);
            if (!magic.SequenceEqual(ExpectedMagic))
            {
                GD.PushError($"[RLModelLoader] Bad magic bytes in {absPath} — not a .rlmodel file.");
                return null;
            }

            var version = reader.ReadUInt16();
            if (version < 1 || version > RLCheckpoint.CurrentFormatVersion)
            {
                GD.PushError($"[RLModelLoader] Unsupported .rlmodel version {version} in {absPath}.");
                return null;
            }

            var headerObsSize = reader.ReadInt32();
            var headerActionDims = reader.ReadInt32();
            var layerCount = reader.ReadInt32();

            // ── Layers ────────────────────────────────────────────────────────
            var usesTypedLayerShapes = version >= 3;
            var shapes  = new List<int>(layerCount * (usesTypedLayerShapes ? 4 : 3));
            var weights = new List<float>();

            for (var i = 0; i < layerCount; i++)
            {
                var inSize     = reader.ReadInt32();
                var outSize    = reader.ReadInt32();
                var activation = reader.ReadInt32();

                if (usesTypedLayerShapes)
                {
                    shapes.Add((int)RLLayerKind.Dense);
                }

                shapes.Add(inSize);
                shapes.Add(outSize);
                shapes.Add(activation);

                var numWeights = inSize * outSize;
                for (var j = 0; j < numWeights; j++) weights.Add(reader.ReadSingle());
                for (var j = 0; j < outSize;          j++) weights.Add(reader.ReadSingle());
            }

            var checkpoint = new RLCheckpoint
            {
                WeightBuffer = weights.ToArray(),
                LayerShapeBuffer = shapes.ToArray(),
            };

            if (version >= RLCheckpoint.CurrentFormatVersion && stream.Position < stream.Length)
            {
                var metadataLength = reader.ReadInt32();
                var metadataBytes = reader.ReadBytes(metadataLength);
                checkpoint.ApplyMetadataJson(Encoding.UTF8.GetString(metadataBytes));

                if (checkpoint.ObservationSize <= 0)
                {
                    checkpoint.ObservationSize = headerObsSize;
                }

                if (checkpoint.DiscreteActionCount <= 0 && checkpoint.ContinuousActionDimensions <= 0)
                {
                    checkpoint.DiscreteActionCount = headerActionDims;
                }
            }
            else
            {
                checkpoint.PopulateLegacyMetadata();
                checkpoint.ObservationSize = checkpoint.ObservationSize > 0 ? checkpoint.ObservationSize : headerObsSize;
                if (checkpoint.DiscreteActionCount <= 0)
                {
                    checkpoint.DiscreteActionCount = headerActionDims;
                }
            }

            GD.Print($"[RLModelLoader] Loaded {layerCount} layers from {absPath}");
            return checkpoint;
        }
        catch (Exception ex)
        {
            GD.PushError($"[RLModelLoader] Failed to read {absPath}: {ex.Message}");
            return null;
        }
    }
}
