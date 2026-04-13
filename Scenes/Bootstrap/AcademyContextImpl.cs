using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Internal implementation of <see cref="IAcademyContext"/>.
/// Holds a back-reference to <see cref="TrainingBootstrap"/> and delegates all calls
/// to its <c>internal</c> accessor methods.
/// <para>
/// <see cref="IAcademyContext"/> is defined in <c>Runtime/Core</c> with no dependency on
/// <see cref="TrainingBootstrap"/>. This impl lives in <c>Scenes/Bootstrap</c> alongside
/// <see cref="TrainingBootstrap"/>, so the dependency is one-way and does not create a
/// circular reference at the public API level.
/// </para>
/// </summary>
internal sealed class AcademyContextImpl : IAcademyContext
{
    private readonly TrainingBootstrap _bootstrap;

    internal AcademyContextImpl(TrainingBootstrap bootstrap) => _bootstrap = bootstrap;

    public long TotalSteps => _bootstrap.TotalStepsInternal;

    public IReadOnlyDictionary<string, long> EpisodeCountByGroup
        => _bootstrap.EpisodeCountByGroupInternal;

    public IReadOnlyList<string> GroupIds => _bootstrap.GroupIdsInternal;

    public ITrainer? GetTrainer(string groupId)
        => _bootstrap.GetTrainerInternal(groupId);

    public IReadOnlyList<IRLAgent> GetGroupAgents(string groupId)
        => _bootstrap.GetGroupAgentsInternal(groupId);

    public void RunGroupDecisionPipeline(string groupId)
        => _bootstrap.RunGroupDecisionPipelineInternal(groupId);

    public PhaseAToken EstimateNextValues(string groupId)
        => _bootstrap.EstimateNextValuesInternal(groupId);

    public PhaseBToken RecordTransitionsAndReset(string groupId, PhaseAToken phaseA)
        => _bootstrap.RecordTransitionsInternal(groupId, phaseA);

    public PhaseCToken SampleActions(string groupId, PhaseBToken phaseB)
        => _bootstrap.SampleActionsInternal(groupId, phaseB);

    public void ApplyDecisions(string groupId, PhaseCToken phaseC)
        => _bootstrap.ApplyDecisionsInternal(groupId, phaseC);

    public void TriggerCheckpoint(bool forceWrite = false)
        => _bootstrap.TriggerCheckpointInternal(forceWrite);

    public void LogMetric(string groupId, string metricKey, float value)
        => _bootstrap.LogMetricInternal(groupId, metricKey, value);

    public void SetCurriculumProgress(float progress)
        => _bootstrap.SetCurriculumProgressInternal(progress);

    public void RequestStop(string reason = "Stopped by custom loop.")
        => _bootstrap.RequestStopInternal(reason);
}
