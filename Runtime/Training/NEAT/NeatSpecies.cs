using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

internal sealed class NeatSpecies
{
    private static int _nextSpeciesId = 0;

    public int SpeciesId { get; init; }

    /// <summary>
    /// A representative genome sampled from the previous generation.
    /// New genomes are tested against this for compatibility.
    /// </summary>
    public NeatGenome Representative { get; set; } = null!;

    public List<NeatGenome> Members { get; } = new();

    public float BestFitnessEver { get; private set; } = float.MinValue;
    public int   StagnationCounter { get; private set; }
    public int   Age { get; set; }

    public static NeatSpecies Create(NeatGenome representative)
    {
        var s = new NeatSpecies
        {
            SpeciesId      = System.Threading.Interlocked.Increment(ref _nextSpeciesId),
            Representative = representative,
        };
        return s;
    }

    public float AverageAdjustedFitness() =>
        Members.Count == 0 ? 0f : Members.Average(g => g.AdjustedFitness);

    public float SumAdjustedFitness() =>
        Members.Sum(g => g.AdjustedFitness);

    /// <summary>
    /// Checks current best fitness against historical best and increments the
    /// stagnation counter if there was no improvement.
    /// </summary>
    public void UpdateStagnation()
    {
        if (Members.Count == 0) return;
        float currentBest = Members.Max(g => g.Fitness);
        if (currentBest > BestFitnessEver)
        {
            BestFitnessEver   = currentBest;
            StagnationCounter = 0;
        }
        else
        {
            StagnationCounter++;
        }
    }
}
