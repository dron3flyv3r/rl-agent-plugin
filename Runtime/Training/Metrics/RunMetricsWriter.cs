using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class RunMetricsWriter
{
    private readonly string _metricsPath;
    private readonly string _statusPath;

    public RunMetricsWriter(string metricsPath, string statusPath)
    {
        _metricsPath = metricsPath;
        _statusPath = statusPath;
    }

    /// <summary>Legacy constructor for backward compatibility with single-group training.</summary>
    public RunMetricsWriter(TrainingLaunchManifest manifest)
        : this(manifest.MetricsPath, manifest.StatusPath) { }

    public void WriteStatus(
        string status,
        string scenePath,
        long totalSteps,
        long episodeCount,
        string message,
        long workerEpisodeCount = 0,
        string resumedFrom = "",
        bool warmStartUsed = false,
        string warmStartSource = "")
    {
        EnsureFileDirectory(_statusPath);
        using var file = FileAccess.Open(_statusPath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            GD.PushError($"[RL] Failed to write status file '{_statusPath}': {FileAccess.GetOpenError()}");
            return;
        }

        var payload = new Godot.Collections.Dictionary
        {
            { "status",        status },
            { "scene_path",    scenePath },
            { "total_steps",   totalSteps },
            { "episode_count", episodeCount },
            { "message",       message },
        };
        if (workerEpisodeCount > 0)
            payload["worker_episode_count"] = workerEpisodeCount;
        if (!string.IsNullOrEmpty(resumedFrom))
            payload["resumed_from"] = resumedFrom;
        payload["warm_start_used"] = warmStartUsed;
        if (!string.IsNullOrWhiteSpace(warmStartSource))
            payload["warm_start_source"] = warmStartSource;

        file.StoreString(Json.Stringify(payload));
    }

    public void AppendMetric(
        float episodeReward,
        int episodeLength,
        float policyLoss,
        float valueLoss,
        float entropy,
        float? clipFraction,
        float? sacAlpha,
        long totalSteps,
        long episodeCount,
        IReadOnlyDictionary<string, float>? rewardComponents = null,
        string policyGroup = "",
        string opponentGroup = "",
        string opponentSource = "",
        string opponentCheckpointPath = "",
        long? opponentUpdateCount = null,
        float? learnerElo   = null,
        float? poolWinRate  = null,
        float? curriculumProgress = null)
    {
        EnsureFileDirectory(_metricsPath);
        var mode = FileAccess.FileExists(_metricsPath)
            ? FileAccess.ModeFlags.ReadWrite
            : FileAccess.ModeFlags.Write;
        using var file = FileAccess.Open(_metricsPath, mode);
        if (file is null)
        {
            GD.PushError($"[RL] Failed to open metrics file '{_metricsPath}': {FileAccess.GetOpenError()}");
            return;
        }

        if (mode == FileAccess.ModeFlags.ReadWrite)
        {
            file.SeekEnd();
        }

        var payload = new Godot.Collections.Dictionary
        {
            { "episode_reward", episodeReward },
            { "episode_length", episodeLength },
            { "policy_loss", policyLoss },
            { "value_loss", valueLoss },
            { "entropy", entropy },
            { "total_steps", totalSteps },
            { "episode_count", episodeCount },
        };

        if (clipFraction.HasValue)
        {
            payload["clip_fraction"] = clipFraction.Value;
        }

        if (sacAlpha.HasValue)
        {
            payload["sac_alpha"] = sacAlpha.Value;
        }

        if (rewardComponents is not null && rewardComponents.Count > 0)
        {
            var rewardBreakdown = new Godot.Collections.Dictionary();
            foreach (var (tag, amount) in rewardComponents)
            {
                rewardBreakdown[tag] = amount;
            }

            payload["reward_components"] = rewardBreakdown;
        }

        if (!string.IsNullOrWhiteSpace(policyGroup))
        {
            payload["policy_group"] = policyGroup;
        }

        if (!string.IsNullOrWhiteSpace(opponentGroup))
        {
            payload["opponent_group"] = opponentGroup;
        }

        if (!string.IsNullOrWhiteSpace(opponentSource))
        {
            payload["opponent_source"] = opponentSource;
        }

        if (!string.IsNullOrWhiteSpace(opponentCheckpointPath))
        {
            payload["opponent_checkpoint_path"] = opponentCheckpointPath;
        }

        if (opponentUpdateCount.HasValue)
        {
            payload["opponent_update_count"] = opponentUpdateCount.Value;
        }

        if (learnerElo.HasValue)
        {
            payload["learner_elo"] = learnerElo.Value;
        }

        if (poolWinRate.HasValue)
        {
            payload["pool_avg_win_rate"] = poolWinRate.Value;
        }

        if (curriculumProgress.HasValue)
        {
            payload["curriculum_progress"] = curriculumProgress.Value;
        }

        file.StoreLine(Json.Stringify(payload));
    }

    /// <summary>
    /// Appends a greedy-evaluation result line to the metrics file.
    /// The line is tagged with <c>"is_eval": true</c> so the dashboard can separate it
    /// from training metrics. The <c>"episode_reward"</c> field is also set to
    /// <paramref name="meanReward"/> for compatibility with existing reward chart rendering.
    /// </summary>
    public void AppendEvalMetric(
        float meanReward,
        float meanEpisodeLength,
        long totalSteps,
        long episodeCount,
        int evalEpisodes,
        string policyGroup = "")
    {
        EnsureFileDirectory(_metricsPath);
        var mode = FileAccess.FileExists(_metricsPath)
            ? FileAccess.ModeFlags.ReadWrite
            : FileAccess.ModeFlags.Write;
        using var file = FileAccess.Open(_metricsPath, mode);
        if (file is null)
        {
            GD.PushError($"[RL] Failed to open metrics file '{_metricsPath}' for eval metric: {FileAccess.GetOpenError()}");
            return;
        }

        if (mode == FileAccess.ModeFlags.ReadWrite)
            file.SeekEnd();

        var payload = new Godot.Collections.Dictionary
        {
            { "is_eval",           true             },
            { "episode_reward",    meanReward       },
            { "eval_mean_reward",  meanReward       },
            { "eval_mean_length",  meanEpisodeLength },
            { "eval_episodes",     evalEpisodes     },
            { "total_steps",       totalSteps       },
            { "episode_count",     episodeCount     },
        };

        if (!string.IsNullOrWhiteSpace(policyGroup))
            payload["policy_group"] = policyGroup;

        file.StoreLine(Json.Stringify(payload));
    }

    private static void EnsureFileDirectory(string filePath)
    {
        var dir = filePath.GetBaseDir();
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        var absDir = ProjectSettings.GlobalizePath(dir);
        var err = DirAccess.MakeDirRecursiveAbsolute(absDir);
        if (err != Error.Ok)
        {
            GD.PushError($"[RL] Failed to create directory '{absDir}': {err}");
        }
    }
}
