namespace RlAgentPlugin.Runtime;

/// <summary>
/// Implement this interface on your scene (or a helper node) to enable MCTS planning.
/// <para>
/// Register the model before training starts: <c>MctsTrainer.SetEnvironmentModel(this);</c>
/// </para>
/// <para>
/// <b>Contract:</b>
/// <list type="bullet">
/// <item><description><see cref="SimulateStep"/> must be <b>deterministic</b> and <b>side-effect-free</b> —
/// it must NOT modify the actual game state, physics, or any scene nodes.</description></item>
/// <item><description>The method is called hundreds of times per action decision (once per MCTS simulation).
/// Keep it fast: avoid instancing nodes, running physics, or calling heavy Godot APIs.</description></item>
/// <item><description>The <paramref name="observation"/> passed in is the same flat float array returned
/// by your agent's <c>CollectObservations</c> — use it to reconstruct "virtual" state.</description></item>
/// </list>
/// </para>
/// <example>
/// <code>
/// // Minimal implementation example
/// public (float[] nextObs, float reward, bool done) SimulateStep(float[] obs, int action)
/// {
///     // Decode current state from obs
///     var x = obs[0] * MaxRange;
///     var goal = obs[1] * MaxRange;
///
///     // Simulate action (no scene changes!)
///     float dx = action == 0 ? -Speed : Speed;
///     var newX = Mathf.Clamp(x + dx, -MaxRange, MaxRange);
///
///     // Build next observation
///     var nextObs = new float[] { newX / MaxRange, goal / MaxRange };
///     var reward = Mathf.Abs(newX - goal) &lt; 0.1f ? 1f : -0.01f;
///     var done = reward > 0f;
///     return (nextObs, reward, done);
/// }
/// </code>
/// </example>
/// </summary>
public interface IEnvironmentModel
{
    /// <summary>
    /// Simulates one environment step from <paramref name="observation"/> with <paramref name="action"/>.
    /// </summary>
    /// <param name="observation">Current state as a flat observation vector.</param>
    /// <param name="action">Discrete action index to simulate.</param>
    /// <returns>
    /// <c>nextObservation</c>: resulting state after the action.<br/>
    /// <c>reward</c>: immediate reward for this transition.<br/>
    /// <c>done</c>: true if this transition ends the episode.
    /// </returns>
    (float[] nextObservation, float reward, bool done) SimulateStep(float[] observation, int action);
}
