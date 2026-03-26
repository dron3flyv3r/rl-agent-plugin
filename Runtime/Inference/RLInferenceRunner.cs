using System;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Convenience wrapper for running a trained model in code without a full training scene.
///
/// Usage:
/// <code>
///   var runner = new RLInferenceRunner("res://runs/my_run/checkpoint_1000.json");
///   float[] obs = GetMyObservations();
///   PolicyDecision action = runner.Predict(obs);
///   int move  = action.DiscreteAction;       // -1 if continuous-only
///   float[] v = action.ContinuousActions;    // empty if discrete-only
/// </code>
/// </summary>
public sealed class RLInferenceRunner
{
    private readonly IInferencePolicy _policy;

    /// <summary>Number of floats expected in each observation vector.</summary>
    public int ObservationSize { get; }

    /// <summary>Number of discrete action choices (0 if continuous-only).</summary>
    public int DiscreteActionCount { get; }

    /// <summary>Number of continuous action dimensions (0 if discrete-only).</summary>
    public int ContinuousActionDimensions { get; }

    /// <summary>Algorithm string from the checkpoint — "PPO" or "SAC".</summary>
    public string Algorithm { get; }

    /// <summary>
    /// Loads a checkpoint from <paramref name="checkpointPath"/> and builds the correct
    /// inference policy for its algorithm.
    /// </summary>
    /// <param name="checkpointPath">
    /// Absolute path, <c>res://</c>, or <c>user://</c> path to a checkpoint JSON file.
    /// </param>
    /// <param name="fallbackGraph">
    /// Optional network graph used when the checkpoint has no embedded graph metadata
    /// (e.g. very old checkpoints). Pass <c>null</c> to use the built-in default.
    /// </param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the file cannot be read or parsed.
    /// </exception>
    public RLInferenceRunner(string checkpointPath, RLNetworkGraph? fallbackGraph = null)
        : this(LoadOrThrow(checkpointPath), fallbackGraph) { }

    /// <summary>
    /// Builds an inference runner from an already-loaded <see cref="RLCheckpoint"/>.
    /// </summary>
    public RLInferenceRunner(RLCheckpoint checkpoint, RLNetworkGraph? fallbackGraph = null)
    {
        if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));

        ObservationSize            = checkpoint.ObservationSize;
        DiscreteActionCount        = checkpoint.DiscreteActionCount;
        ContinuousActionDimensions = checkpoint.ContinuousActionDimensions;
        Algorithm                  = checkpoint.Algorithm;

        if (ObservationSize <= 0)
            throw new InvalidOperationException(
                $"Checkpoint has invalid ObservationSize ({ObservationSize}). " +
                "The checkpoint may be corrupt or from an incompatible format version.");

        _policy = InferencePolicyFactory.Create(checkpoint, fallbackGraph);
        _policy.LoadCheckpoint(checkpoint);
    }

    /// <summary>
    /// Runs a deterministic forward pass and returns the predicted action.
    /// </summary>
    /// <param name="observation">
    /// Float array of length <see cref="ObservationSize"/>.
    /// </param>
    /// <exception cref="ArgumentNullException">If <paramref name="observation"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// If <paramref name="observation"/>.Length does not match <see cref="ObservationSize"/>.
    /// </exception>
    public PolicyDecision Predict(float[] observation)
    {
        if (observation is null) throw new ArgumentNullException(nameof(observation));
        if (observation.Length != ObservationSize)
            throw new ArgumentException(
                $"Observation length mismatch: expected {ObservationSize}, got {observation.Length}.",
                nameof(observation));

        return _policy.Predict(observation);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static RLCheckpoint LoadOrThrow(string path)
    {
        if (string.IsNullOrEmpty(path))
            throw new ArgumentException("Checkpoint path must not be null or empty.", nameof(path));

        return RLCheckpoint.LoadFromFile(path)
               ?? throw new InvalidOperationException($"Failed to load checkpoint from '{path}'.");
    }
}
