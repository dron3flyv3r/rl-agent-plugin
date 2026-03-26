namespace RlAgentPlugin.Runtime;

public readonly record struct ObservationSegment(string Name, int StartIndex, int Length, string[]? DebugLabels = null);
