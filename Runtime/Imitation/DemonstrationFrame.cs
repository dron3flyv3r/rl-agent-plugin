using System;

namespace RlAgentPlugin.Runtime.Imitation;

/// <summary>
/// One recorded step from a demonstration session.
/// Observations are the inputs the policy sees; actions are what the demonstrator chose.
/// Reward and Done are informational — BC training only uses Obs + Action.
/// </summary>
internal struct DemonstrationFrame
{
    public int AgentSlot;
    public float[] Obs;
    /// <summary>Discrete action index, or -1 if the action space is continuous-only.</summary>
    public int DiscreteAction;
    /// <summary>Continuous actions, or empty array if discrete-only.</summary>
    public float[] ContinuousActions;
    public float Reward;
    public bool Done;
}
