using System;
using System.Collections.Generic;
using System.Threading;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Runtime.Imitation;

/// <summary>
/// Behavior cloning trainer. Runs supervised gradient updates on a
/// <see cref="PolicyValueNetwork"/> using recorded demonstration frames.
///
/// Designed to run on a background Task (no Godot API calls during training).
/// Progress is reported via the <see cref="BCProgressReport"/> callback.
///
/// Usage:
///   var trainer = new BCTrainer(network, dataset, config);
///   await Task.Run(() => trainer.Train(onProgress, cts.Token));
///   var checkpoint = trainer.BuildCheckpoint(runId);
/// </summary>
internal sealed class BCTrainer
{
    private readonly PolicyValueNetwork _network;
    private readonly DemonstrationDataset _dataset;
    private readonly RLImitationConfig _config;

    public BCTrainer(PolicyValueNetwork network, DemonstrationDataset dataset, RLImitationConfig config)
    {
        _network = network;
        _dataset = dataset;
        _config = config;
    }

    /// <summary>
    /// Runs BC training synchronously. Call from a background Task.
    /// </summary>
    /// <param name="onProgress">Called after each mini-batch (may be called from a non-main thread).</param>
    /// <param name="ct">Cancellation token — checked between batches.</param>
    /// <returns>Mean loss of the final epoch, or NaN if cancelled before any update.</returns>
    public float Train(Action<BCProgressReport> onProgress, CancellationToken ct)
    {
        var frames = new List<DemonstrationFrame>(_dataset.Frames);
        if (frames.Count == 0)
        {
            onProgress(new BCProgressReport(0, _config.Epochs, 0f, 0, "Empty dataset."));
            return float.NaN;
        }

        var rng = new Random();
        var batchSize = Math.Max(1, _config.BatchSize);
        var lastEpochLoss = float.NaN;
        var totalBatches = (int)Math.Ceiling((double)frames.Count / batchSize) * _config.Epochs;
        var batchIndex = 0;

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            if (ct.IsCancellationRequested) break;

            if (_config.ShuffleEachEpoch)
                ShuffleInPlace(frames, rng);

            var epochLoss = 0f;
            var epochBatches = 0;

            for (var start = 0; start < frames.Count; start += batchSize)
            {
                if (ct.IsCancellationRequested) break;

                var end = Math.Min(start + batchSize, frames.Count);
                var batch = BuildBatch(frames, start, end);

                var batchLoss = _network.ApplyBCGradients(batch, _config.LearningRate, _config.MaxGradientNorm);
                epochLoss += batchLoss;
                epochBatches++;
                batchIndex++;

                onProgress(new BCProgressReport(
                    epoch + 1,
                    _config.Epochs,
                    batchLoss,
                    (float)batchIndex / totalBatches,
                    null));
            }

            if (epochBatches > 0)
                lastEpochLoss = epochLoss / epochBatches;
        }

        return lastEpochLoss;
    }

    /// <summary>
    /// Saves the trained network weights as an <see cref="RLCheckpoint"/> ready to write to disk.
    /// Call after <see cref="Train"/> completes.
    /// </summary>
    public RLCheckpoint BuildCheckpoint(string runId)
    {
        var checkpoint = _network.SaveCheckpoint(runId, totalSteps: 0, episodeCount: 0, updateCount: 1);
        checkpoint.ObservationSize         = _dataset.ObsSize;
        checkpoint.DiscreteActionCount     = _dataset.DiscreteActionCount;
        checkpoint.ContinuousActionDimensions = _dataset.ContinuousActionDims;
        return checkpoint;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static List<BCSample> BuildBatch(List<DemonstrationFrame> frames, int start, int end)
    {
        var batch = new List<BCSample>(end - start);
        for (var i = start; i < end; i++)
        {
            var f = frames[i];
            batch.Add(new BCSample
            {
                Observation = f.Obs,
                DiscreteAction = f.DiscreteAction,
                ContinuousActions = f.ContinuousActions,
            });
        }

        return batch;
    }

    private static void ShuffleInPlace<T>(List<T> list, Random rng)
    {
        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}

/// <summary>Progress update emitted after each BC mini-batch.</summary>
public sealed class BCProgressReport
{
    public int Epoch { get; }
    public int TotalEpochs { get; }
    public float BatchLoss { get; }
    /// <summary>Overall training progress in [0, 1].</summary>
    public float Progress { get; }
    /// <summary>Optional status message (e.g., error or completion note).</summary>
    public string? Message { get; }

    public BCProgressReport(int epoch, int totalEpochs, float batchLoss, float progress, string? message)
    {
        Epoch = epoch;
        TotalEpochs = totalEpochs;
        BatchLoss = batchLoss;
        Progress = progress;
        Message = message;
    }
}
