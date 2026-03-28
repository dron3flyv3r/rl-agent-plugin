using System.Collections.Generic;
using System.Diagnostics;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Lightweight timing profiler. Accumulates per-label stats and prints a summary
/// to the console every <see cref="ReportEvery"/> calls to <see cref="Tick"/>.
///
/// Usage:
///   var t = RLProfiler.Begin();
///   // ... work ...
///   RLProfiler.End("MyLabel", t);
///
/// Toggle with <see cref="Enabled"/>. Zero overhead when disabled.
/// </summary>
internal static class RLProfiler
{
    /// <summary>Set to false to disable all profiling with zero overhead.</summary>
    public static bool Enabled = false;

    /// <summary>How many calls to <see cref="End"/> on the trigger label before printing a summary.</summary>
    public static int ReportEvery = 1;

    /// <summary>The label whose call count triggers the summary print.</summary>
    public static string TriggerLabel = "PPO.BackgroundUpdate";

    private static readonly Dictionary<string, Entry> _entries = new();

    private sealed class Entry
    {
        public long   TotalTicks;
        public long   MinTicks = long.MaxValue;
        public long   MaxTicks;
        public int    Count;
    }

    /// <summary>Returns a high-resolution timestamp to pass to <see cref="End"/>.</summary>
    public static long Begin() => Stopwatch.GetTimestamp();

    /// <summary>Records elapsed time since <paramref name="startTick"/> under <paramref name="label"/>.</summary>
    public static void End(string label, long startTick)
    {
        if (!Enabled) return;

        var elapsed = Stopwatch.GetTimestamp() - startTick;

        if (!_entries.TryGetValue(label, out var entry))
        {
            entry = new Entry();
            _entries[label] = entry;
        }

        entry.TotalTicks += elapsed;
        entry.Count++;
        if (elapsed < entry.MinTicks) entry.MinTicks = elapsed;
        if (elapsed > entry.MaxTicks) entry.MaxTicks = elapsed;

        if (label == TriggerLabel && entry.Count % ReportEvery == 0)
            PrintSummary(entry.Count);
    }

    private static void PrintSummary(int triggerCount)
    {
        var freq = (double)Stopwatch.Frequency / 1000.0; // ticks per ms

        GD.Print($"[RLProfile] ── Summary after {triggerCount} × '{TriggerLabel}' ──────────────");
        foreach (var (label, e) in _entries)
        {
            if (e.Count == 0) continue;
            var avg   = e.TotalTicks / (double)e.Count / freq;
            var min   = e.MinTicks / freq;
            var max   = e.MaxTicks / freq;
            var total = e.TotalTicks / freq;
            GD.Print($"[RLProfile]   {label,-22} calls={e.Count,6}  avg={avg,7:F3}ms  min={min,7:F3}ms  max={max,7:F3}ms  total={total,9:F1}ms");
        }
        GD.Print("[RLProfile] ──────────────────────────────────────────────────────────");
    }
}
