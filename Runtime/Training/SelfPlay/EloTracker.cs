using System;

namespace RlAgentPlugin.Runtime;

internal sealed class EloTracker
{
    public const float InitialRating = 1200f;
    public const float KFactor       = 32f;

    public float Rating { get; private set; } = InitialRating;

    public void Update(float opponentElo, float score) // score: 1=win, 0=loss
    {
        var expected = 1f / (1f + MathF.Pow(10f, (opponentElo - Rating) / 400f));
        Rating += KFactor * (score - expected);
    }
}
