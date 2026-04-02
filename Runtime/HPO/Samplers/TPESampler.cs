using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Tree-structured Parzen Estimator (TPE) sampler — the algorithm at the core of Optuna.
/// <para>
/// For each parameter dimension independently:
/// <list type="number">
///   <item>Splits completed trials into "good" (top <see cref="GoodFraction"/>) and "bad" groups.</item>
///   <item>Fits a 1-D Gaussian KDE to each group in the natural (or log) space.</item>
///   <item>Draws <see cref="NumCandidates"/> candidates from the good KDE.</item>
///   <item>Returns the candidate that maximises <c>log p_good(x) − log p_bad(x)</c>.</item>
/// </list>
/// Falls back to <see cref="RandomSampler"/> when fewer than
/// <see cref="MinTrialsForTPE"/> complete trials are available.
/// </para>
/// </summary>
public sealed class TPESampler : IHPOSampler
{
    /// <summary>Fraction of top trials considered "good". Default 0.25 (top 25%).</summary>
    public float GoodFraction { get; set; } = 0.25f;

    /// <summary>Number of candidates drawn per dimension when scoring.</summary>
    public int NumCandidates { get; set; } = 24;

    /// <summary>Minimum completed trials required before switching from random to TPE.</summary>
    public int MinTrialsForTPE { get; set; } = 5;

    private readonly RandomSampler _fallback;

    public TPESampler(int? seed = null)
    {
        _fallback = new RandomSampler(seed);
    }

    public Dictionary<string, float> Suggest(
        IReadOnlyList<TrialRecord> completedTrials,
        RLHPOStudy study)
    {
        if (completedTrials.Count < MinTrialsForTPE)
            return _fallback.Suggest(completedTrials, study);

        // Sort by objective direction
        var sorted = study.Direction == RLHPODirection.Maximize
            ? completedTrials.OrderByDescending(t => t.ObjectiveValue ?? float.NegativeInfinity).ToList()
            : completedTrials.OrderBy(t => t.ObjectiveValue ?? float.PositiveInfinity).ToList();

        int goodCount = Math.Max(1, (int)MathF.Ceiling(sorted.Count * GoodFraction));
        var good = sorted.Take(goodCount).ToList();
        var bad  = sorted.Skip(goodCount).ToList();

        var result = new Dictionary<string, float>(StringComparer.Ordinal);
        foreach (var param in study.SearchSpace)
        {
            result[param.ParameterName] = param.Kind == RLHPOParameterKind.Categorical
                ? SampleCategorical(good, bad, param)
                : SampleNumeric(good, bad, param);
        }
        return result;
    }

    // ── Numeric (uniform / log) ──────────────────────────────────────────

    private float SampleNumeric(List<TrialRecord> good, List<TrialRecord> bad, RLHPOParameter param)
    {
        bool logScale = param.Kind is RLHPOParameterKind.FloatLog or RLHPOParameterKind.IntLog;
        bool isInt    = param.Kind is RLHPOParameterKind.IntUniform or RLHPOParameterKind.IntLog;

        float low  = logScale ? MathF.Log(Math.Max(param.Low,  1e-10f)) : param.Low;
        float high = logScale ? MathF.Log(Math.Max(param.High, 1e-10f)) : param.High;
        float range = high - low;
        if (range <= 0f)
            return param.Low;

        float[] goodVals = ExtractValues(good, param, logScale);
        float[] badVals  = ExtractValues(bad,  param, logScale);

        if (goodVals.Length == 0)
            return _fallback.SampleOne(param);

        float goodBw = SilvermanBandwidth(goodVals, range);
        float badBw  = badVals.Length > 0 ? SilvermanBandwidth(badVals, range) : goodBw;

        float bestX     = float.NaN;
        float bestScore = float.NegativeInfinity;

        for (int i = 0; i < NumCandidates; i++)
        {
            // Sample candidate from the good KDE
            int srcIdx = i % goodVals.Length;
            float x = goodVals[srcIdx] + _fallback.NextGaussian() * goodBw;
            x = Math.Clamp(x, low, high);

            float score = LogKDE(x, goodVals, goodBw)
                        - (badVals.Length > 0 ? LogKDE(x, badVals, badBw) : 0f);

            if (score > bestScore)
            {
                bestScore = score;
                bestX = x;
            }
        }

        float result = logScale ? MathF.Exp(bestX) : bestX;
        return isInt ? MathF.Round(result) : result;
    }

    // ── Categorical ──────────────────────────────────────────────────────

    private float SampleCategorical(List<TrialRecord> good, List<TrialRecord> bad, RLHPOParameter param)
    {
        if (param.Choices is null || param.Choices.Count == 0)
            return 0f;

        int n = param.Choices.Count;
        var goodCounts = new float[n];
        var badCounts  = new float[n];

        foreach (var t in good)
        {
            if (t.Parameters.TryGetValue(param.ParameterName, out var v))
            {
                int idx = FindChoiceIndex(param, v);
                if (idx >= 0) goodCounts[idx] += 1f;
            }
        }
        foreach (var t in bad)
        {
            if (t.Parameters.TryGetValue(param.ParameterName, out var v))
            {
                int idx = FindChoiceIndex(param, v);
                if (idx >= 0) badCounts[idx] += 1f;
            }
        }

        // Add-one smoothing, then compute log ratio
        float[] scores = new float[n];
        for (int i = 0; i < n; i++)
        {
            float pGood = (goodCounts[i] + 1f) / (good.Count + n);
            float pBad  = (badCounts[i]  + 1f) / (bad.Count  + n);
            scores[i] = MathF.Log(pGood) - MathF.Log(pBad);
        }

        int best = 0;
        for (int i = 1; i < n; i++)
            if (scores[i] > scores[best]) best = i;

        return float.TryParse(
            param.Choices[best],
            NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture,
            out var val) ? val : 0f;
    }

    // ── KDE helpers ──────────────────────────────────────────────────────

    private static float[] ExtractValues(List<TrialRecord> trials, RLHPOParameter param, bool logScale)
    {
        var vals = new List<float>(trials.Count);
        foreach (var t in trials)
        {
            if (!t.Parameters.TryGetValue(param.ParameterName, out var v)) continue;
            vals.Add(logScale ? MathF.Log(Math.Max(v, 1e-10f)) : v);
        }
        return vals.ToArray();
    }

    private static float SilvermanBandwidth(float[] values, float range)
    {
        if (values.Length < 2) return range * 0.1f;
        float mean = 0f;
        foreach (var v in values) mean += v;
        mean /= values.Length;
        float variance = 0f;
        foreach (var v in values) variance += (v - mean) * (v - mean);
        variance /= values.Length;
        float std = MathF.Sqrt(variance);
        if (std < 1e-10f) std = range * 0.01f;
        return 1.06f * std * MathF.Pow(values.Length, -0.2f);
    }

    private static float LogKDE(float x, float[] values, float bandwidth)
    {
        float sum = 0f;
        foreach (var v in values)
        {
            float u = (x - v) / bandwidth;
            sum += MathF.Exp(-0.5f * u * u);
        }
        return MathF.Log(sum / values.Length + 1e-20f);
    }

    private static int FindChoiceIndex(RLHPOParameter param, float value)
    {
        for (int i = 0; i < param.Choices.Count; i++)
        {
            if (float.TryParse(param.Choices[i], NumberStyles.Float,
                    CultureInfo.InvariantCulture, out var parsed)
                && MathF.Abs(parsed - value) < 1e-6f)
                return i;
        }
        return -1;
    }
}
