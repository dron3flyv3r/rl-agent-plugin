namespace RlAgentPlugin.Runtime;

public interface IObservationSensor
{
    int Size { get; }
    void Write(ObservationBuffer buffer);
}
