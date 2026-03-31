using System;
using System.Threading;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Generic fixed-capacity ring-buffer replay store.
/// Thread-safe: <see cref="Add"/> takes a write lock; <see cref="SampleBatch"/> takes a read lock,
/// so concurrent reads are allowed but adding an item excludes sampling.
/// Used by SAC, DQN, and any future off-policy algorithm.
/// </summary>
internal sealed class ReplayBuffer<T>
{
    private readonly T[] _buffer;
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.NoRecursion);
    private int _head;
    private int _count;

    public ReplayBuffer(int capacity)
    {
        _buffer = new T[capacity];
    }

    public int Count => _count;
    public int Capacity => _buffer.Length;

    public void Add(T item)
    {
        _rwLock.EnterWriteLock();
        try
        {
            _buffer[_head] = item;
            _head = (_head + 1) % _buffer.Length;
            if (_count < _buffer.Length)
                _count++;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    public T[] SampleBatch(int batchSize, Random rng)
    {
        _rwLock.EnterReadLock();
        try
        {
            var actualBatch = Math.Min(batchSize, _count);
            var batch = new T[actualBatch];

            // Fisher-Yates partial shuffle for sampling without replacement.
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
