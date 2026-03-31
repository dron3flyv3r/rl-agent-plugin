using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// A node in the MCTS search tree.
/// Each node represents a state reached by following a sequence of actions from the root.
/// </summary>
internal sealed class MctsNode
{
    public int VisitCount;
    public double TotalValue;      // sum of backed-up values (double for precision over many visits)

    /// <summary>Mean Q-value estimate for this node.</summary>
    public float Q => VisitCount > 0 ? (float)(TotalValue / VisitCount) : 0f;

    /// <summary>
    /// Child nodes keyed by the action that leads to them.
    /// Null until the first child is expanded.
    /// </summary>
    public Dictionary<int, MctsNode>? Children;

    public MctsNode() { }

    public bool HasChildren => Children is { Count: > 0 };

    public MctsNode GetOrAddChild(int action)
    {
        Children ??= new Dictionary<int, MctsNode>();
        if (!Children.TryGetValue(action, out var child))
        {
            child = new MctsNode();
            Children[action] = child;
        }
        return child;
    }
}
