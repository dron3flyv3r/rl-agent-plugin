using System;

namespace RlAgentPlugin.Runtime;

internal sealed class OpponentRecord
{
    public string CheckpointPath { get; init; } = string.Empty;
    public string SnapshotKey    { get; init; } = string.Empty;  // "{path}::{updateCount}"
    public float  SnapshotElo    { get; set; }  = 1200f;         // frozen at insertion time
    public int    Wins           { get; set; }  = 1;             // Laplace prior
    public int    Episodes       { get; set; }  = 2;             // Laplace prior
    public float  WinRate        => (float)Wins / Episodes;
    public float  PfspWeight(float alpha) =>
        MathF.Exp(-alpha * (WinRate - 0.5f) * (WinRate - 0.5f));
}
