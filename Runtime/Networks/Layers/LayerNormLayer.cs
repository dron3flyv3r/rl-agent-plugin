using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Layer normalisation layer. Normalises across the feature dimension, then applies a
/// learned affine transform: y_i = gamma_i * normalised_i + beta_i.
///
/// Gamma is initialised to 1, beta to 0. Gradients are accumulated in
/// <see cref="GradientBuffer.WeightGradients"/> (gamma) and
/// <see cref="GradientBuffer.BiasGradients"/> (beta).
/// </summary>
internal sealed class LayerNormLayer : NetworkLayer
{
    private const float Eps = 1e-5f;

    private readonly int     _size;
    private readonly float[] _gamma; // learned scale
    private readonly float[] _beta;  // learned shift

    // Forward-pass cache (set by Forward, used by Backward / AccumulateGradients / ComputeInputGrad)
    private float[]? _lastNormalized;
    private float    _lastStd;

    public override int InputSize  => _size;
    public override int OutputSize => _size;

    public LayerNormLayer(int size)
    {
        _size  = size;
        _gamma = new float[size];
        _beta  = new float[size];
        Array.Fill(_gamma, 1f);
        // _beta is already zero
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    public override float[] Forward(float[] input, bool isTraining = false)
    {
        var mean = 0f;
        for (var i = 0; i < _size; i++) mean += input[i];
        mean /= _size;

        var variance = 0f;
        for (var i = 0; i < _size; i++) { var d = input[i] - mean; variance += d * d; }
        variance /= _size;

        var std = Mathf.Sqrt(variance + Eps);
        _lastStd = std;

        var normalized = new float[_size];
        var output     = new float[_size];
        for (var i = 0; i < _size; i++)
        {
            normalized[i] = (input[i] - mean) / std;
            output[i]     = _gamma[i] * normalized[i] + _beta[i];
        }

        _lastNormalized = normalized;
        return output;
    }

    public override VectorBatch ForwardBatch(VectorBatch input)
    {
        var output = new VectorBatch(input.BatchSize, _size);
        for (var b = 0; b < input.BatchSize; b++)
        {
            var off  = b * _size;
            var mean = 0f;
            for (var i = 0; i < _size; i++) mean += input.Data[off + i];
            mean /= _size;

            var variance = 0f;
            for (var i = 0; i < _size; i++) { var d = input.Data[off + i] - mean; variance += d * d; }
            variance /= _size;
            var std = Mathf.Sqrt(variance + Eps);

            for (var i = 0; i < _size; i++)
            {
                var norm = (input.Data[off + i] - mean) / std;
                output.Data[off + i] = _gamma[i] * norm + _beta[i];
            }
        }

        return output;
    }

    // ── Single-sample backward (immediate weight update) ──────────────────

    public override float[] Backward(float[] outputGrad, float learningRate, float gradScale = 1f)
    {
        var (inputGrad, gammaGrad, betaGrad) = ComputeGradients(outputGrad);
        for (var i = 0; i < _size; i++)
        {
            _gamma[i] -= learningRate * gammaGrad[i] * gradScale;
            _beta[i]  -= learningRate * betaGrad[i]  * gradScale;
        }

        return inputGrad;
    }

    // ── Batch accumulate + apply ──────────────────────────────────────────

    public override GradientBuffer CreateGradientBuffer() =>
        new(_size, _size); // WeightGradients = gamma grads, BiasGradients = beta grads

    public override float[] AccumulateGradients(float[] outputGrad, GradientBuffer buffer)
    {
        var (inputGrad, gammaGrad, betaGrad) = ComputeGradients(outputGrad);
        for (var i = 0; i < _size; i++)
        {
            buffer.WeightGradients[i] += gammaGrad[i];
            buffer.BiasGradients[i]   += betaGrad[i];
        }

        return inputGrad;
    }

    public override void ApplyGradients(GradientBuffer buffer, float learningRate, float gradScale)
    {
        for (var i = 0; i < _size; i++)
        {
            _gamma[i] -= learningRate * buffer.WeightGradients[i] * gradScale;
            _beta[i]  -= learningRate * buffer.BiasGradients[i]   * gradScale;
        }
    }

    // ── Frozen gradient ───────────────────────────────────────────────────

    public override float[] ComputeInputGrad(float[] outputGrad)
    {
        var (inputGrad, _, _) = ComputeGradients(outputGrad);
        return inputGrad;
    }

    // ── Serialization ─────────────────────────────────────────────────────

    public override void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add((int)RLLayerKind.LayerNorm);
        shapes.Add(_size);
        foreach (var g in _gamma) weights.Add(g);
        foreach (var b in _beta)  weights.Add(b);
    }

    public override void LoadSerialized(
        IReadOnlyList<float> weights, ref int wi,
        IReadOnlyList<int>   shapes,  ref int si,
        bool isLegacy = false)
    {
        if (!isLegacy)
        {
            var typeCode = shapes[si++];
            if (typeCode != (int)RLLayerKind.LayerNorm)
                throw new InvalidOperationException($"Expected LayerNorm layer type ({(int)RLLayerKind.LayerNorm}), got {typeCode}.");
        }

        var serializedSize = shapes[si++];
        if (serializedSize != _size)
            throw new InvalidOperationException($"LayerNorm checkpoint size {serializedSize} does not match network size {_size}.");

        for (var i = 0; i < _size; i++) _gamma[i] = weights[wi++];
        for (var i = 0; i < _size; i++) _beta[i]  = weights[wi++];
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// <summary>
    /// Standard layer-norm backward. Returns (input gradient, gamma gradient, beta gradient).
    /// Formula: dL/dx_i = (1/std) * (dNorm_i − mean(dNorm) − normalised_i · mean(dNorm · normalised))
    /// </summary>
    private (float[] inputGrad, float[] gammaGrad, float[] betaGrad) ComputeGradients(float[] outputGrad)
    {
        var normalized = _lastNormalized ?? throw new InvalidOperationException("Backward called before Forward.");
        var std        = _lastStd;

        var gammaGrad = new float[_size];
        var betaGrad  = new float[_size];
        for (var i = 0; i < _size; i++)
        {
            gammaGrad[i] = outputGrad[i] * normalized[i];
            betaGrad[i]  = outputGrad[i];
        }

        // dNorm_i = dL/dy_i * gamma_i
        var dNorm = new float[_size];
        for (var i = 0; i < _size; i++) dNorm[i] = outputGrad[i] * _gamma[i];

        var meanDNorm  = 0f;
        var meanDNormN = 0f;
        for (var i = 0; i < _size; i++)
        {
            meanDNorm  += dNorm[i];
            meanDNormN += dNorm[i] * normalized[i];
        }
        meanDNorm  /= _size;
        meanDNormN /= _size;

        var inputGrad = new float[_size];
        for (var i = 0; i < _size; i++)
            inputGrad[i] = (dNorm[i] - meanDNorm - normalized[i] * meanDNormN) / std;

        return (inputGrad, gammaGrad, betaGrad);
    }
}
