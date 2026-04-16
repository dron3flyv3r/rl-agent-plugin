using Godot;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// A clean facade over <see cref="RLAcademy"/> for population-based evolutionary algorithms.
///
/// Exposes only the inspector properties relevant to evolutionary training:
/// <see cref="EvolutionaryConfig"/>, <see cref="RLAcademy.MaxEpisodeSteps"/>,
/// <see cref="RLAcademy.RunConfig"/>, <see cref="RLAcademy.DistributedConfig"/>,
/// <see cref="RLAcademy.Curriculum"/>, <see cref="RLAcademy.EnableSpyOverlay"/>,
/// and <see cref="RLAcademy.EnableCameraDebug"/>.
///
/// <see cref="RLAcademy.TrainingConfig"/> and <see cref="RLAcademy.SelfPlay"/> are hidden
/// from the inspector. <see cref="RLAcademy.TrainingConfig"/> is built automatically
/// from <see cref="EvolutionaryConfig"/> whenever the latter is assigned — no separate
/// <see cref="RLTrainingConfig"/> resource is needed in the scene.
///
/// Drop this node into your evolutionary algorithm scene in place of <see cref="RLAcademy"/>.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLGeneticAcademy : RLAcademy
{
    private RLEvolutionaryConfig? _evConfig;

    // ── Algorithm ─────────────────────────────────────────────────────────────

    [ExportGroup("Algorithm")]
    /// <summary>
    /// The evolutionary algorithm configuration (e.g. <see cref="RLNEATConfig"/>).
    /// Assigning this automatically builds the internal <see cref="RLAcademy.TrainingConfig"/>;
    /// you do not need a separate <see cref="RLTrainingConfig"/> resource in the scene.
    /// </summary>
    [Export]
    public RLEvolutionaryConfig? EvolutionaryConfig
    {
        get => _evConfig;
        set
        {
            _evConfig = value;
            TrainingConfig = value is not null
                ? new RLTrainingConfig { Algorithm = value }
                : null;
            UpdateConfigurationWarnings();
        }
    }

    // ── Hide irrelevant RLAcademy inspector fields ────────────────────────────

    /// <summary>
    /// Hides <c>TrainingConfig</c> and <c>SelfPlay</c> from the inspector.
    /// Both are still accessible at runtime; <c>TrainingConfig</c> is managed automatically
    /// by the <see cref="EvolutionaryConfig"/> setter.
    /// </summary>
    public override void _ValidateProperty(Dictionary property)
    {
        var name = property["name"].AsString();
        if (name is "TrainingConfig" or "SelfPlay")
        {
            var usage = (PropertyUsageFlags)property["usage"].AsInt32();
            property["usage"] = (int)(usage & ~PropertyUsageFlags.Editor);
        }
    }

    // ── Configuration warnings ────────────────────────────────────────────────

    public override string[] _GetConfigurationWarnings()
    {
        if (EvolutionaryConfig is null)
            return new[] { "EvolutionaryConfig is not assigned. Assign an RLEvolutionaryConfig resource (e.g. RLNEATConfig)." };
        return System.Array.Empty<string>();
    }
}
