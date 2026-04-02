namespace RlAgentPlugin;

public enum RLHPOSamplerKind
{
    Random = 0,
    TPE    = 1,
}

public enum RLHPOPrunerKind
{
    None               = 0,
    Median             = 1,
    SuccessiveHalving  = 2,
}

public enum RLHPOParameterKind
{
    FloatUniform = 0,
    FloatLog     = 1,
    IntUniform   = 2,
    IntLog       = 3,
    Categorical  = 4,
}

public enum RLHPOObjectiveMetric
{
    MeanEpisodeReward = 0,
    MeanEpisodeLength = 1,
    PolicyLoss        = 2,
    ValueLoss         = 3,
    Custom            = 4,
}

public enum RLHPOObjectiveAggregation
{
    Mean         = 0,
    WeightedMean = 1,
    Min          = 2,
    Max          = 3,
}

public enum RLHPODirection
{
    Maximize = 0,
    Minimize = 1,
}

public enum RLHPOTrialState
{
    Pending  = 0,
    Running  = 1,
    Complete = 2,
    Pruned   = 3,
    Failed   = 4,
}
