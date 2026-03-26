namespace RlAgentPlugin.Runtime;

public enum ObservationStreamKind
{
    /// <summary>A flat vector of float values (position, velocity, etc.).</summary>
    Vector = 0,
    /// <summary>An image encoded as row-major normalized float pixels [0, 1].</summary>
    Image = 1,
}
