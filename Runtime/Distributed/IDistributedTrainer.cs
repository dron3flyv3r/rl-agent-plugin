namespace RlAgentPlugin.Runtime;

/// <summary>
/// Extended trainer interface for distributed training.
///
/// Both PPO and SAC implement this; the semantics differ:
/// <list type="bullet">
///   <item>PPO — synchronous, rollout-based.  Workers collect exactly one rollout then block
///         waiting for new weights.  Master waits until all workers have delivered before training.</item>
///   <item>SAC — asynchronous, transition-based.  Workers stream transitions continuously;
///         master trains on its own schedule and sends weights back periodically.</item>
/// </list>
/// </summary>
public interface IDistributedTrainer : ITrainer
{
    /// <summary>
    /// True for off-policy algorithms (SAC, DQN, etc.).
    /// Controls worker blocking behaviour and master training gate.
    /// </summary>
    bool IsOffPolicy { get; }

    /// <summary>
    /// PPO: true when the local rollout buffer has reached <c>RolloutLength</c> and is ready to ship.
    /// SAC: true when enough transitions have accumulated to justify sending a batch.
    /// </summary>
    bool IsRolloutReady { get; }

    /// <summary>
    /// Serialises the current experience buffer to a byte array and clears it.
    /// Only call when <see cref="IsRolloutReady"/> is true.
    /// </summary>
    byte[] ExportAndClearRollout();

    /// <summary>
    /// Injects serialised experience received from a worker into this trainer's buffer.
    /// The master calls this once per worker rollout before deciding to train.
    /// </summary>
    void InjectRollout(byte[] data);

    /// <summary>Serialises the current network weights to a compact byte array.</summary>
    byte[] ExportWeights();

    /// <summary>Loads weights that were previously produced by <see cref="ExportWeights"/>.</summary>
    void ImportWeights(byte[] data);
}
