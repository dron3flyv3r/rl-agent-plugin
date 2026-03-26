namespace RlAgentPlugin.Runtime;

public enum RLAsyncRolloutPolicy
{
    /// <summary>
    /// While a background gradient update is in flight, incoming worker rollouts are drained
    /// from the network queue but discarded — they are not injected into the trainer.
    /// After training completes, a fresh batch is gathered before the next update.
    /// Keeps update sizes consistent at the cost of dropping transitions collected during training.
    /// </summary>
    Pause = 0,

    /// <summary>
    /// Incoming rollouts are always injected into the trainer buffer.
    /// The training batch is capped at <c>RolloutLength × (WorkerCount + 1)</c> transitions;
    /// anything beyond the cap is discarded when the update is scheduled.
    /// Retains the most recent data while preventing unbounded buffer growth.
    /// </summary>
    Cap = 1,
}

public enum RLActivationKind
{
    Tanh = 0,
    Relu = 1,
}

public enum RLOptimizerKind
{
    Adam = 0,
    Sgd  = 1,
    /// <summary>No optimizer — frozen / target layers only. No moment vectors allocated; weight updates are no-ops.</summary>
    None = -1,
}

public enum RLLayerKind
{
    Dense    = 0,
    Dropout  = 1,
    LayerNorm = 2,
    Flatten  = 3,
}

public enum RLStoppingCombineMode
{
    /// <summary>Stop when ANY enabled condition becomes true (OR logic).</summary>
    Any = 0,
    /// <summary>Stop only when ALL enabled conditions are true simultaneously (AND logic).</summary>
    All = 1,
}
