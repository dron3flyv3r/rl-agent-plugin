using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// UCT (UCB1 applied to Trees) Monte Carlo Tree Search.
/// Builds a search tree from the current observation using an <see cref="IEnvironmentModel"/>,
/// then returns the action with the highest visit count at the root.
/// </summary>
internal sealed class MctsSearch
{
    private readonly IEnvironmentModel _model;
    private readonly int _actionCount;
    private readonly int _numSimulations;
    private readonly int _maxDepth;
    private readonly int _rolloutDepth;
    private readonly float _explorationConstant;
    private readonly float _gamma;
    private readonly Func<float[], float>? _valueEstimator;
    private readonly Random _rng;

    // Reused across simulations to avoid per-call allocation.
    private readonly List<MctsNode> _pathNodes = new();

    public MctsSearch(
        IEnvironmentModel model,
        int actionCount,
        int numSimulations,
        int maxDepth,
        int rolloutDepth,
        float explorationConstant,
        float gamma,
        Func<float[], float>? valueEstimator = null,
        Random? rng = null)
    {
        _model               = model;
        _actionCount         = actionCount;
        _numSimulations      = numSimulations;
        _maxDepth            = maxDepth;
        _rolloutDepth        = rolloutDepth;
        _explorationConstant = explorationConstant;
        _gamma               = gamma;
        _valueEstimator      = valueEstimator;
        _rng                 = rng ?? new Random();
    }

    /// <summary>
    /// Runs MCTS from <paramref name="rootObs"/> and returns the best action index.
    /// </summary>
    public int Search(float[] rootObs)
    {
        var root = new MctsNode();

        for (var sim = 0; sim < _numSimulations; sim++)
            RunSimulation(root, rootObs);

        return BestAction(root);
    }

    // ── Core simulation ──────────────────────────────────────────────────────

    private void RunSimulation(MctsNode root, float[] rootObs)
    {
        _pathNodes.Clear();
        _pathNodes.Add(root);

        var obs      = rootObs;
        var node     = root;
        var depth    = 0;
        var done     = false;
        var discount = 1f;
        var value    = 0f;

        // ── Selection: follow UCT until a non-fully-expanded node or terminal ──
        while (!done && depth < _maxDepth)
        {
            var fullyExpanded = node.HasChildren && node.Children!.Count == _actionCount;
            if (!fullyExpanded) break;

            var action = SelectUct(node);
            var (nextObs, reward, isDone) = _model.SimulateStep(obs, action);

            value   += discount * reward;
            discount *= _gamma;
            done     = isDone;
            obs      = nextObs;
            node     = node.GetOrAddChild(action);
            depth++;
            _pathNodes.Add(node);
        }

        if (done)
        {
            // Terminal: back-propagate accumulated reward.
            Backpropagate(_pathNodes, value);
            return;
        }

        if (depth < _maxDepth)
        {
            // ── Expansion: expand one new child ─────────────────────────────
            var action = SelectExpansion(node);
            var (nextObs, reward, isDone) = _model.SimulateStep(obs, action);

            value   += discount * reward;
            discount *= _gamma;
            done     = isDone;
            obs      = nextObs;
            node     = node.GetOrAddChild(action);
            depth++;
            _pathNodes.Add(node);

            // ── Evaluation: rollout or value network ─────────────────────────
            if (!done)
            {
                var leafValue = Evaluate(obs, depth);
                value += discount * leafValue;
            }
        }

        // ── Backpropagation ──────────────────────────────────────────────────
        Backpropagate(_pathNodes, value);
    }

    // ── UCT child selection ──────────────────────────────────────────────────

    private int SelectUct(MctsNode node)
    {
        var logN      = MathF.Log(node.VisitCount + 1);
        var bestScore = float.NegativeInfinity;
        var best      = 0;

        for (var a = 0; a < _actionCount; a++)
        {
            float score;
            if (node.Children == null || !node.Children.TryGetValue(a, out var child))
            {
                score = float.PositiveInfinity; // unvisited → maximally optimistic
            }
            else
            {
                var exploit = child.Q;
                var explore = _explorationConstant * MathF.Sqrt(logN / (child.VisitCount + 1));
                score = exploit + explore;
            }

            if (score > bestScore)
            {
                bestScore = score;
                best      = a;
            }
        }

        return best;
    }

    /// <summary>Returns a random unexpanded action (one with no child yet).</summary>
    private int SelectExpansion(MctsNode node)
    {
        if (!node.HasChildren)
            return _rng.Next(_actionCount);

        Span<int> unexpanded = stackalloc int[_actionCount];
        var count = 0;
        for (var a = 0; a < _actionCount; a++)
            if (!node.Children!.ContainsKey(a))
                unexpanded[count++] = a;

        return count == 0
            ? _rng.Next(_actionCount)          // fully expanded (shouldn't happen here)
            : unexpanded[_rng.Next(count)];
    }

    // ── Leaf evaluation ──────────────────────────────────────────────────────

    private float Evaluate(float[] obs, int startDepth)
        => _valueEstimator != null ? _valueEstimator(obs) : RandomRollout(obs, startDepth);

    private float RandomRollout(float[] obs, int startDepth)
    {
        var total    = 0f;
        var discount = 1f;
        var limit    = startDepth + _rolloutDepth;

        for (var d = startDepth; d < limit; d++)
        {
            var action = _rng.Next(_actionCount);
            var (nextObs, reward, done) = _model.SimulateStep(obs, action);
            total   += discount * reward;
            discount *= _gamma;
            obs      = nextObs;
            if (done) break;
        }

        return total;
    }

    // ── Backpropagation ──────────────────────────────────────────────────────

    private static void Backpropagate(List<MctsNode> path, float value)
    {
        foreach (var n in path)
        {
            n.VisitCount++;
            n.TotalValue += value;
        }
    }

    // ── Best action ──────────────────────────────────────────────────────────

    /// <summary>Returns the action with the highest visit count at the root.</summary>
    private int BestAction(MctsNode root)
    {
        if (!root.HasChildren) return _rng.Next(_actionCount);

        var bestAction = 0;
        var bestVisits = -1;
        foreach (var (action, child) in root.Children!)
        {
            if (child.VisitCount > bestVisits)
            {
                bestVisits = child.VisitCount;
                bestAction = action;
            }
        }
        return bestAction;
    }
}
