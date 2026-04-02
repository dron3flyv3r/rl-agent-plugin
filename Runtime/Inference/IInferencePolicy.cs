namespace RlAgentPlugin.Runtime;

public interface IInferencePolicy
{
    void LoadCheckpoint(RLCheckpoint checkpoint);
    PolicyDecision Predict(float[] observation);
    RecurrentState? CreateZeroRecurrentState() => null;
    PolicyDecision PredictRecurrent(float[] observation, RecurrentState state)
        => Predict(observation);
}
