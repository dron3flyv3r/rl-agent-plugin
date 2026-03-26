using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

internal static class CheckpointMetadataBuilder
{
    public static RLCheckpoint Apply(RLCheckpoint checkpoint, PolicyGroupConfig config)
    {
        checkpoint.FormatVersion           = RLCheckpoint.CurrentFormatVersion;
        checkpoint.Algorithm               = config.Algorithm == RLAlgorithmKind.SAC
            ? RLCheckpoint.SacAlgorithm
            : RLCheckpoint.PpoAlgorithm;
        checkpoint.ObservationSize         = config.ObservationSize;
        checkpoint.DiscreteActionCount     = config.DiscreteActionCount;
        checkpoint.ContinuousActionDimensions = config.ContinuousActionDimensions;
        checkpoint.NetworkLayers           = BuildNetworkLayers(config.NetworkGraph);
        checkpoint.NetworkOptimizer        = OptimizerToString(config.NetworkGraph.Optimizer);
        checkpoint.DiscreteActionLabels    = BuildDiscreteActionLabels(config.ActionDefinitions);
        checkpoint.ContinuousActionRanges  = BuildContinuousActionRanges(config.ActionDefinitions);
        checkpoint.Hyperparams             = BuildHyperparams(config);
        return checkpoint;
    }

    private static List<RLCheckpointLayer> BuildNetworkLayers(RLNetworkGraph graph)
    {
        var layers = new List<RLCheckpointLayer>(graph.TrunkLayers.Count);
        foreach (var resource in graph.TrunkLayers)
        {
            switch (resource)
            {
                case RLDenseLayerDef dense:
                    layers.Add(new RLCheckpointLayer
                    {
                        Type       = "dense",
                        Size       = dense.Size,
                        Activation = ActivationToString(dense.Activation),
                    });
                    break;
                case RLDropoutLayerDef dropout:
                    layers.Add(new RLCheckpointLayer { Type = "dropout", Rate = dropout.Rate });
                    break;
                case RLLayerNormDef:
                    layers.Add(new RLCheckpointLayer { Type = "layer_norm" });
                    break;
                case RLFlattenLayerDef:
                    layers.Add(new RLCheckpointLayer { Type = "flatten" });
                    break;
                default:
                    GD.PushWarning($"[CheckpointMetadataBuilder] Unknown layer type {resource.GetType().Name} — skipped in checkpoint metadata.");
                    break;
            }
        }
        return layers;
    }

    private static string ActivationToString(RLActivationKind activation) => activation switch
    {
        RLActivationKind.Relu => "relu",
        _                     => "tanh",
    };

    internal static string OptimizerToString(RLOptimizerKind optimizer) => optimizer switch
    {
        RLOptimizerKind.Sgd  => "sgd",
        RLOptimizerKind.None => "none",
        _                    => "adam",
    };

    private static Dictionary<string, string[]> BuildDiscreteActionLabels(IEnumerable<RLActionDefinition> actionDefinitions)
    {
        var labels = new Dictionary<string, string[]>(StringComparer.Ordinal);
        foreach (var action in actionDefinitions)
        {
            if (action.VariableType != RLActionVariableType.Discrete) continue;
            labels[action.Name] = (string[])action.Labels.Clone();
        }
        return labels;
    }

    private static Dictionary<string, RLContinuousActionRange> BuildContinuousActionRanges(IEnumerable<RLActionDefinition> actionDefinitions)
    {
        var ranges = new Dictionary<string, RLContinuousActionRange>(StringComparer.Ordinal);
        foreach (var action in actionDefinitions)
        {
            if (action.VariableType != RLActionVariableType.Continuous) continue;
            ranges[action.Name] = new RLContinuousActionRange
            {
                Dimensions = action.Dimensions,
                Min        = action.MinValue,
                Max        = action.MaxValue,
            };
        }
        return ranges;
    }

    private static Dictionary<string, float> BuildHyperparams(PolicyGroupConfig config)
    {
        var trainer = config.TrainerConfig;
        var values = new Dictionary<string, float>(StringComparer.Ordinal)
        {
            ["learning_rate"] = trainer.LearningRate,
            ["gamma"]         = trainer.Gamma,
        };

        if (config.Algorithm == RLAlgorithmKind.PPO)
        {
            values["rollout_length"]       = trainer.RolloutLength;
            values["epochs_per_update"]    = trainer.EpochsPerUpdate;
            values["ppo_minibatch_size"]   = trainer.PpoMiniBatchSize;
            values["gae_lambda"]           = trainer.GaeLambda;
            values["clip_epsilon"]         = trainer.ClipEpsilon;
            values["max_gradient_norm"]    = trainer.MaxGradientNorm;
            values["value_loss_coefficient"] = trainer.ValueLossCoefficient;
            values["use_value_clipping"]   = trainer.UseValueClipping ? 1f : 0f;
            values["value_clip_epsilon"]   = trainer.ValueClipEpsilon;
            values["entropy_coefficient"]  = trainer.EntropyCoefficient;
        }
        else
        {
            values["replay_buffer_capacity"] = trainer.ReplayBufferCapacity;
            values["sac_batch_size"]         = trainer.SacBatchSize;
            values["sac_warmup_steps"]       = trainer.SacWarmupSteps;
            values["sac_tau"]                = trainer.SacTau;
            values["sac_init_alpha"]         = trainer.SacInitAlpha;
            values["sac_auto_tune_alpha"]    = trainer.SacAutoTuneAlpha ? 1f : 0f;
            values["sac_update_every_steps"] = trainer.SacUpdateEverySteps;
        }

        return values;
    }
}
