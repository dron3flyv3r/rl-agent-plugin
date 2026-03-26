using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

public interface IObservationDebugLabels
{
    IReadOnlyList<string> DebugLabels { get; }
}
