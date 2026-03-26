using System;
using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

public sealed class VectorBatch
{
    private readonly float[] _data;

    public VectorBatch(int batchSize, int vectorSize)
    {
        if (batchSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        }

        if (vectorSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(vectorSize));
        }

        BatchSize = batchSize;
        VectorSize = vectorSize;
        _data = new float[batchSize * vectorSize];
    }

    public int BatchSize { get; }
    public int VectorSize { get; }
    public float[] Data => _data;

    public static VectorBatch FromRows(IReadOnlyList<float[]> rows)
    {
        if (rows.Count == 0)
        {
            throw new ArgumentException("At least one row is required.", nameof(rows));
        }

        var batch = new VectorBatch(rows.Count, rows[0].Length);
        for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            batch.SetRow(rowIndex, rows[rowIndex]);
        }

        return batch;
    }

    public void SetRow(int rowIndex, float[] values)
    {
        if (values.Length != VectorSize)
        {
            throw new ArgumentException(
                $"Expected row length {VectorSize}, got {values.Length}.",
                nameof(values));
        }

        Array.Copy(values, 0, _data, rowIndex * VectorSize, VectorSize);
    }

    public float[] CopyRow(int rowIndex)
    {
        var row = new float[VectorSize];
        Array.Copy(_data, rowIndex * VectorSize, row, 0, VectorSize);
        return row;
    }

    public float Get(int rowIndex, int columnIndex)
    {
        return _data[(rowIndex * VectorSize) + columnIndex];
    }
}
