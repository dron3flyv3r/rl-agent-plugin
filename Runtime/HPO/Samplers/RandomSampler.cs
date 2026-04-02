using System;
using System.Collections.Generic;
using System.Globalization;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Samples each hyperparameter independently and uniformly (or log-uniformly).
/// Serves as the default fallback for early trials in the TPE sampler.
/// </summary>
public sealed class RandomSampler : IHPOSampler
{
    private readonly Random _rng;

    public RandomSampler(int? seed = null)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public Dictionary<string, float> Suggest(
        IReadOnlyList<TrialRecord> completedTrials,
        RLHPOStudy study)
    {
        var result = new Dictionary<string, float>(StringComparer.Ordinal);
        foreach (var param in study.SearchSpace)
            result[param.ParameterName] = SampleOne(param);
        return result;
    }

    internal float SampleOne(RLHPOParameter param)
    {
        switch (param.Kind)
        {
            case RLHPOParameterKind.FloatUniform:
                return (float)(_rng.NextDouble() * (param.High - param.Low) + param.Low);

            case RLHPOParameterKind.FloatLog:
            {
                if (param.Low <= 0f || param.High <= 0f)
                    throw new ArgumentException(
                        $"[RandomSampler] {param.Kind} parameter '{param.ParameterName}' requires positive bounds; got Low={param.Low}, High={param.High}.",
                        nameof(param));
                double logLow  = Math.Log(param.Low);
                double logHigh = Math.Log(param.High);
                return (float)Math.Exp(_rng.NextDouble() * (logHigh - logLow) + logLow);
            }

            case RLHPOParameterKind.IntUniform:
                return (float)_rng.Next((int)Math.Round(param.Low), (int)Math.Round(param.High) + 1);

            case RLHPOParameterKind.IntLog:
            {
                if (param.Low <= 0f || param.High <= 0f)
                    throw new ArgumentException(
                        $"[RandomSampler] {param.Kind} parameter '{param.ParameterName}' requires positive bounds; got Low={param.Low}, High={param.High}.",
                        nameof(param));
                double logLow  = Math.Log(param.Low);
                double logHigh = Math.Log(param.High);
                return (float)Math.Round(Math.Exp(_rng.NextDouble() * (logHigh - logLow) + logLow));
            }

            case RLHPOParameterKind.Categorical:
            {
                if (param.Choices is null || param.Choices.Count == 0)
                    return 0f;
                var choice = param.Choices[_rng.Next(param.Choices.Count)];
                return float.TryParse(choice, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : 0f;
            }

            default:
                return param.Low;
        }
    }

    internal float NextGaussian()
    {
        // Box-Muller transform
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }
}
