namespace RlAgentPlugin.Runtime;

/// <summary>
/// Optional capability for scene nodes that consume curriculum progress updates.
/// Implement on agents, arena controllers, or game managers that react to difficulty changes.
/// </summary>
public interface IRLCurriculumConsumer
{
    void NotifyCurriculumProgress(float progress);
}
