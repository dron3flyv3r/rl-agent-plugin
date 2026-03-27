using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class ObservationBuffer
{
    private readonly List<float> _values = new();
    private readonly List<ObservationSegment> _segments = new();
    // Tracks named streams for multi-modal observation support.
    private readonly List<ObservationStreamSpec> _streamSpecs = new();

    public int Count => _values.Count;
    public IReadOnlyList<ObservationSegment> Segments => _segments;

    /// <summary>Add a single float value as-is.</summary>
    public void Add(float value) => _values.Add(value);

    /// <summary>Add a boolean as 0 or 1.</summary>
    public void Add(bool value) => _values.Add(value ? 1f : 0f);

    /// <summary>Add a Vector2 as two floats (X, Y).</summary>
    public void Add(Vector2 value)
    {
        _values.Add(value.X);
        _values.Add(value.Y);
    }

    /// <summary>Add a Vector3 as three floats (X, Y, Z).</summary>
    public void Add(Vector3 value)
    {
        _values.Add(value.X);
        _values.Add(value.Y);
        _values.Add(value.Z);
    }

    /// <summary>Add a value linearly mapped from [min, max] to [-1, 1].</summary>
    public void AddNormalized(float value, float min, float max)
    {
        if (float.IsNaN(value) || float.IsInfinity(value))
        {
            GD.PushWarning($"[ObservationBuffer] Attempted to add invalid value {value}. Adding 0 instead.");
            _values.Add(0f);
            return;
        }
        var range = max - min;
        var normalized = range > 0f ? (value - min) / range * 2f - 1f : 0f;
        _values.Add(Mathf.Clamp(normalized, -1f, 1f));
    }

    public void AddNormalized(Vector2 value, Vector2 min, Vector2 max)
    {
        AddNormalized(value.X, min.X, max.X);
        AddNormalized(value.Y, min.Y, max.Y);
    }

    public void AddNormalized(Vector3 value, Vector3 min, Vector3 max)
    {
        AddNormalized(value.X, min.X, max.X);
        AddNormalized(value.Y, min.Y, max.Y);
        AddNormalized(value.Z, min.Z, max.Z);
    }

    public void AddNormalized(int value, int min, int max) => AddNormalized((float)value, (float)min, (float)max);

    // ── Multi-modal stream API ────────────────────────────────────────────────

    /// <summary>
    /// Adds a named vector stream. All floats in <paramref name="values"/> are written as-is
    /// and tagged as a distinct <see cref="ObservationStreamKind.Vector"/> stream in the spec.
    /// </summary>
    public void AddVector(string name, ReadOnlySpan<float> values)
    {
        var startIndex = Count;
        foreach (var v in values)
            _values.Add(v);
        var length = Count - startIndex;
        _streamSpecs.Add(new ObservationStreamSpec(name, ObservationStreamKind.Vector, length, 0, 0, 0));
        if (length > 0)
            _segments.Add(new ObservationSegment(name, startIndex, length));
    }

    /// <summary>
    /// Captures <paramref name="sensor"/> and adds the result as a named image stream.
    /// Width, height, and channel count are read from the sensor's output properties.
    /// </summary>
    public void AddImage(string name, RLCameraSensor2D sensor)
    {
        var pixels = sensor.Capture();
        AddImage(name, pixels, sensor.OutputWidth, sensor.OutputHeight, sensor.OutputChannels);
    }

    /// <summary>
    /// Adds a named image stream. <paramref name="pixels"/> must be row-major bytes of length
    /// <c>width * height * channels</c>. Each byte is normalized to [0, 1] before storage.
    /// The stream is tagged as <see cref="ObservationStreamKind.Image"/> in the spec.
    /// </summary>
    public void AddImage(string name, byte[] pixels, int width, int height, int channels)
    {
        var expectedLength = width * height * channels;
        if (pixels.Length != expectedLength)
        {
            GD.PushError(
                $"[ObservationBuffer] AddImage '{name}': expected {expectedLength} bytes " +
                $"({width}×{height}×{channels}), got {pixels.Length}. Writing zeros.");
            for (var i = 0; i < expectedLength; i++)
                _values.Add(0f);
        }
        else
        {
            var startIndex = Count;
            foreach (var b in pixels)
                _values.Add(b / 255f);
            _segments.Add(new ObservationSegment(name, startIndex, expectedLength));
        }
        _streamSpecs.Add(new ObservationStreamSpec(name, ObservationStreamKind.Image, expectedLength, width, height, channels));
    }

    public void AddSegment(string name, System.Action<ObservationBuffer> writer, int? expectedSize = null, IReadOnlyList<string>? debugLabels = null)
    {
        var startIndex = Count;
        writer(this);
        FinalizeSegment(name, startIndex, expectedSize, debugLabels);
    }

    public void AddSensor(IObservationSensor sensor)
    {
        var expectedSize = sensor.Size;
        var debugLabels = sensor is IObservationDebugLabels labeledSensor ? labeledSensor.DebugLabels : null;
        var startIndex = Count;
        sensor.Write(this);
        FinalizeSegment(null, startIndex, expectedSize, debugLabels);
    }

    public void AddSensor(string name, IObservationSensor sensor)
    {
        AddSensor(name, sensor, sensor is IObservationDebugLabels labeledSensor ? labeledSensor.DebugLabels : null);
    }

    public void AddSensor(string name, IObservationSensor sensor, IReadOnlyList<string>? debugLabels)
    {
        var startIndex = Count;
        sensor.Write(this);
        FinalizeSegment(name, startIndex, sensor.Size, debugLabels);
    }

    internal float[] ToArray() => _values.ToArray();

    internal ObservationSegment[] GetSegmentsSnapshot() => _segments.ToArray();

    /// <summary>
    /// Builds an <see cref="ObservationSpec"/> from any <see cref="AddVector"/> /
    /// <see cref="AddImage"/> calls made since the last <see cref="Clear"/>.
    /// Falls back to a single unnamed flat-vector stream when only legacy
    /// <c>Add()</c> / <c>AddNormalized()</c> / <c>AddSensor()</c> calls were made.
    /// </summary>
    internal ObservationSpec BuildSpec()
    {
        if (_streamSpecs.Count > 0)
            return new ObservationSpec(_streamSpecs.ToArray());

        // Legacy path: treat all values as one unnamed vector stream.
        return ObservationSpec.Flat(_values.Count);
    }

    internal void Clear()
    {
        _values.Clear();
        _segments.Clear();
        _streamSpecs.Clear();
    }

    private void FinalizeSegment(string? name, int startIndex, int? expectedSize, IReadOnlyList<string>? debugLabels)
    {
        var length = Count - startIndex;
        if (expectedSize.HasValue && expectedSize.Value != length)
        {
            GD.PushError(
                $"[ObservationBuffer] Observation segment '{name ?? "<unnamed>"}' wrote {length} value(s), " +
                $"but {expectedSize.Value} were expected.");
        }

        if (debugLabels is not null && debugLabels.Count != length)
        {
            GD.PushWarning(
                $"[ObservationBuffer] Observation segment '{name ?? "<unnamed>"}' supplied {debugLabels.Count} debug label(s), " +
                $"but wrote {length} value(s).");
            debugLabels = null;
        }

        if (length <= 0 || string.IsNullOrWhiteSpace(name))
        {
            return;
        }

        var labels = debugLabels is null ? null : CopyLabels(debugLabels);
        _segments.Add(new ObservationSegment(name, startIndex, length, labels));
    }

    private static string[] CopyLabels(IReadOnlyList<string> debugLabels)
    {
        var copy = new string[debugLabels.Count];
        for (var i = 0; i < debugLabels.Count; i++)
        {
            copy[i] = debugLabels[i];
        }

        return copy;
    }
    public override string ToString() => $"[{string.Join(", ", _values)}]";
}
