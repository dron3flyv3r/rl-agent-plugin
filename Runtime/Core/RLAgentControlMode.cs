namespace RlAgentPlugin.Runtime;

public enum RLAgentControlMode
{
    Train = 0,
    Inference = 1,
    Human = 2,
    /// <summary>
    /// Automatically selects the appropriate mode based on context:
    /// <list type="bullet">
    ///   <item><description><b>Start Training</b> — behaves as <see cref="Train"/>.</description></item>
    ///   <item><description><b>Run Project / Run Inference</b> — behaves as <see cref="Inference"/>; loads the model from
    ///   <c>PolicyGroupConfig.InferenceModelPath</c> when a <c>.rlmodel</c> is assigned.</description></item>
    /// </list>
    /// The explicit modes (<see cref="Train"/>, <see cref="Inference"/>, <see cref="Human"/>) always override Auto.
    /// </summary>
    Auto = 3,
}
