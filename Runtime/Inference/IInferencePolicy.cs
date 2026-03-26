namespace RlAgentPlugin.Runtime;

public interface IInferencePolicy
{
    void LoadCheckpoint(RLCheckpoint checkpoint);
    PolicyDecision Predict(float[] observation);
}
