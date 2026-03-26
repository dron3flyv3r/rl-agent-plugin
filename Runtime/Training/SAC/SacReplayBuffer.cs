using System;
using System.Threading;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Fixed-capacity ring-buffer replay store for SAC.
/// Thread-safe: <see cref="Add"/> takes a write lock; <see cref="SampleBatch"/> takes a read lock,
/// so concurrent reads are allowed but adding a transition excludes sampling.
/// </summary>
internal sealed class SacReplayBuffer
{
    private readonly Transition[] _buffer;
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.NoRecursion);
    private int _head;
    private int _count;

    public SacReplayBuffer(int capacity)
    {
        _buffer = new Transition[capacity];
    }

    public int Count => _count;
    public int Capacity => _buffer.Length;

    public void Add(Transition transition)
    {
        _rwLock.EnterWriteLock();
        try
        {
            _buffer[_head] = transition;
            _head = (_head + 1) % _buffer.Length;
            if (_count < _buffer.Length)
                _count++;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    public Transition[] SampleBatch(int batchSize, Random rng)
    {
        _rwLock.EnterReadLock();
        try
        {
            var actualBatch = Math.Min(batchSize, _count);
            var batch = new Transition[actualBatch];

            // Fisher-Yates shuffle on indices for sampling without replacement
            var indices = new int[_count];
            for (var i = 0; i < _count; i++)
                indices[i] = i;

            for (var i = 0; i < actualBatch; i++)
            {
                var j = i + rng.Next(_count - i);
                (indices[i], indices[j]) = (indices[j], indices[i]);
                batch[i] = _buffer[indices[i]];
            }

            return batch;
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }
}
