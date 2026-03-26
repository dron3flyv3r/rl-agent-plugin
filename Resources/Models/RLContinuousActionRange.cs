using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLContinuousActionRange : Resource
{
    [Export] public int Dimensions { get; set; }
    [Export] public float Min { get; set; } = -1f;
    [Export] public float Max { get; set; } = 1f;
}
