using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Configures periodic greedy evaluation runs during training.
/// Attach to <see cref="RLRunConfig.Evaluation"/> to enable.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLEvaluationConfig : Resource
{
    /// <summary>
    /// Run a greedy evaluation every N total environment steps.
    /// Set to 0 to disable evaluation entirely.
    /// </summary>
    [Export(PropertyHint.Range, "0,10000000,1,or_greater")]
    public long EvaluationFrequencySteps { get; set; } = 0;

    /// <summary>
    /// Number of episodes to run per evaluation pass.
    /// Evaluation reports the mean reward and mean episode length across these episodes.
    /// </summary>
    [Export(PropertyHint.Range, "1,1000,1,or_greater")]
    public int EvaluationEpisodes { get; set; } = 10;
}
