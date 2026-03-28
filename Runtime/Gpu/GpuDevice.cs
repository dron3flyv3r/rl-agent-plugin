using System;
using System.Runtime.InteropServices;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Wraps a thread-local Vulkan <see cref="RenderingDevice"/> created via
/// <see cref="RenderingServer.CreateLocalRenderingDevice"/>.
///
/// A local rendering device is isolated from the main render loop and may be
/// created, used, and destroyed on any background thread. All operations
/// (buffer uploads, shader dispatches, Submit/Sync) must be called from the
/// same thread that created this instance.
///
/// Usage:
/// <code>
///   using var gpu = new GpuDevice();
///   // ... allocate buffers, dispatch shaders ...
///   gpu.Submit();
///   gpu.Sync();
/// </code>
/// </summary>
internal sealed class GpuDevice : IDisposable
{
    public RenderingDevice Rd { get; }

    public GpuDevice()
    {
        Rd = RenderingServer.CreateLocalRenderingDevice()
             ?? throw new InvalidOperationException(
                 "[GpuDevice] RenderingServer.CreateLocalRenderingDevice() returned null. " +
                 "Vulkan is not available on this platform or display server.");
    }

    /// <summary>
    /// Submits all pending GPU commands for execution.
    /// Must be called from the thread that created this device.
    /// </summary>
    public void Submit() => Rd.Submit();

    /// <summary>
    /// Blocks until all submitted GPU commands have completed.
    /// Must be called from the thread that created this device.
    /// </summary>
    public void Sync() => Rd.Sync();

    /// <summary>
    /// Submits pending commands and blocks until the GPU is idle.
    /// Convenience wrapper for <see cref="Submit"/> + <see cref="Sync"/>.
    /// </summary>
    public void SubmitAndSync() { Rd.Submit(); Rd.Sync(); }

    // ── Batched compute list helpers ──────────────────────────────────────────

    /// <summary>
    /// Opens a new compute list. Record multiple dispatches into it via
    /// <see cref="DispatchToList"/>, insert <see cref="AddBarrier"/> between
    /// data-dependent dispatches, then close and execute with
    /// <see cref="EndSubmitAndSync"/>.  Only one compute list may be open at a time.
    /// </summary>
    public long BeginComputeList() => Rd.ComputeListBegin();

    /// <summary>
    /// Records a compute dispatch into an open compute list without submitting.
    /// The uniform set must remain valid until <see cref="EndSubmitAndSync"/> returns.
    /// </summary>
    public void DispatchToList(long list, Rid pipeline, Rid uniformSet, byte[] pushConstant, uint elementCount)
    {
        Rd.ComputeListBindComputePipeline(list, pipeline);
        Rd.ComputeListBindUniformSet(list, uniformSet, 0);
        Rd.ComputeListSetPushConstant(list, pushConstant, (uint)pushConstant.Length);
        Rd.ComputeListDispatch(list, (elementCount + 255u) / 256u, 1, 1);
    }

    /// <summary>
    /// Inserts a full memory/execution barrier between two dispatch calls in the
    /// same compute list.  Required whenever dispatch B reads data written by
    /// dispatch A in the same list.
    /// </summary>
    public void AddBarrier(long list) => Rd.ComputeListAddBarrier(list);

    /// <summary>
    /// Closes the open compute list, submits all recorded commands, and blocks
    /// until the GPU is idle.  Free uniform sets created for this list after this
    /// call returns.
    /// </summary>
    public void EndSubmitAndSync() { Rd.ComputeListEnd(); Rd.Submit(); Rd.Sync(); }

    public void Dispose() => Rd.Free();

    // ── Buffer helpers ────────────────────────────────────────────────────────

    /// <summary>Allocates a GPU storage buffer and optionally uploads <paramref name="initialData"/>.</summary>
    public Rid CreateBuffer(int floatCount, float[]? initialData = null)
    {
        var bytes = initialData is not null
            ? FloatsToBytes(initialData, floatCount)
            : new byte[floatCount * sizeof(float)];
        return Rd.StorageBufferCreate((uint)(floatCount * sizeof(float)), bytes);
    }

    /// <summary>Uploads <paramref name="data"/> into an existing GPU buffer.</summary>
    public void UploadBuffer(Rid buffer, float[] data, int count = -1)
    {
        var n = count < 0 ? data.Length : count;
        Rd.BufferUpdate(buffer, 0, (uint)(n * sizeof(float)), FloatsToBytes(data, n));
    }

    /// <summary>Downloads the entire contents of a GPU buffer as a float array.</summary>
    public float[] DownloadBuffer(Rid buffer, int floatCount)
    {
        var bytes = Rd.BufferGetData(buffer, 0, (uint)(floatCount * sizeof(float)));
        return BytesToFloats(bytes, floatCount);
    }

    /// <summary>Fills an existing GPU buffer with zeros.</summary>
    public void ClearBuffer(Rid buffer, int floatCount)
        => Rd.BufferUpdate(buffer, 0, (uint)(floatCount * sizeof(float)), new byte[floatCount * sizeof(float)]);

    // ── Shader helpers ────────────────────────────────────────────────────────

    /// <summary>
    /// Compiles a GLSL compute shader source string to SPIR-V and creates a shader Rid.
    /// Logs any compilation errors via <see cref="GD.PushError"/>.
    /// </summary>
    public Rid CompileComputeShader(string glslSource)
    {
        var src = new RDShaderSource { SourceCompute = glslSource };
        var spirv = Rd.ShaderCompileSpirVFromSource(src);
        if (spirv is null)
            throw new InvalidOperationException("[GpuDevice] ShaderCompileSpirVFromSource returned null.");

        var compileError = spirv.GetStageCompileError(RenderingDevice.ShaderStage.Compute);
        if (!string.IsNullOrEmpty(compileError))
            throw new InvalidOperationException($"[GpuDevice] Compute shader compile error:\n{compileError}");

        return Rd.ShaderCreateFromSpirV(spirv);
    }

    // ── Byte / float conversion ───────────────────────────────────────────────

    public static byte[] FloatsToBytes(float[] data, int count)
    {
        var bytes = new byte[count * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public static float[] BytesToFloats(byte[] bytes, int floatCount)
    {
        var floats = new float[floatCount];
        Buffer.BlockCopy(bytes, 0, floats, 0, floatCount * sizeof(float));
        return floats;
    }

    // ── Push constant helpers ─────────────────────────────────────────────────

    /// <summary>
    /// Creates a 16-byte push constant buffer (Vulkan minimum alignment) with
    /// <paramref name="v0"/> in the first slot. Remaining bytes are zeroed.
    /// </summary>
    public static byte[] PushConstant(uint v0)
    {
        var pc = new byte[16];
        MemoryMarshal.Write(pc.AsSpan(0), v0);
        return pc;
    }

    /// <summary>Creates a 16-byte push constant buffer containing four uint values.</summary>
    public static byte[] PushConstant(uint v0, uint v1, uint v2, uint v3)
    {
        var pc = new byte[16];
        MemoryMarshal.Write(pc.AsSpan(0),  v0);
        MemoryMarshal.Write(pc.AsSpan(4),  v1);
        MemoryMarshal.Write(pc.AsSpan(8),  v2);
        MemoryMarshal.Write(pc.AsSpan(12), v3);
        return pc;
    }

    /// <summary>Creates a 32-byte push constant buffer containing eight uint values.</summary>
    public static byte[] PushConstant(
        uint v0, uint v1, uint v2, uint v3,
        uint v4, uint v5, uint v6, uint v7)
    {
        var pc = new byte[32];
        MemoryMarshal.Write(pc.AsSpan(0),  v0);
        MemoryMarshal.Write(pc.AsSpan(4),  v1);
        MemoryMarshal.Write(pc.AsSpan(8),  v2);
        MemoryMarshal.Write(pc.AsSpan(12), v3);
        MemoryMarshal.Write(pc.AsSpan(16), v4);
        MemoryMarshal.Write(pc.AsSpan(20), v5);
        MemoryMarshal.Write(pc.AsSpan(24), v6);
        MemoryMarshal.Write(pc.AsSpan(28), v7);
        return pc;
    }

    /// <summary>
    /// Creates a push constant buffer from an arbitrary number of uint values,
    /// padded up to the next 16-byte boundary to satisfy Vulkan/Godot pipeline
    /// validation for push-constant structs.
    /// </summary>
    public static byte[] PushConstant(params uint[] values)
    {
        var byteCount = values.Length * sizeof(uint);
        var paddedByteCount = (byteCount + 15) & ~15;
        var pc = new byte[paddedByteCount];
        for (var i = 0; i < values.Length; i++)
            MemoryMarshal.Write(pc.AsSpan(i * sizeof(uint)), values[i]);
        return pc;
    }

    /// <summary>
    /// Creates a 32-byte push constant buffer for Adam update parameters.
    /// Layout: { uint count, float lr, float gradScale, float b1Corr,
    ///           float b2Corr, float beta1, float beta2, float epsilon }.
    /// </summary>
    public static byte[] PushConstantAdam(
        uint count,
        float learningRate,
        float gradScale,
        float b1Corr,
        float b2Corr,
        float beta1,
        float beta2,
        float epsilon)
    {
        var pc = new byte[32];
        MemoryMarshal.Write(pc.AsSpan(0),  count);
        MemoryMarshal.Write(pc.AsSpan(4),  learningRate);
        MemoryMarshal.Write(pc.AsSpan(8),  gradScale);
        MemoryMarshal.Write(pc.AsSpan(12), b1Corr);
        MemoryMarshal.Write(pc.AsSpan(16), b2Corr);
        MemoryMarshal.Write(pc.AsSpan(20), beta1);
        MemoryMarshal.Write(pc.AsSpan(24), beta2);
        MemoryMarshal.Write(pc.AsSpan(28), epsilon);
        return pc;
    }

    /// <summary>
    /// Creates a 32-byte push constant buffer for native C++ encoder Adam update.
    /// Layout: { uint count, float lrCorrected, float gradScale, float beta1,
    ///           float beta2, float epsilon, uint pad0, uint pad1 }.
    /// </summary>
    public static byte[] PushConstantAdamNative(
        uint count,
        float learningRateCorrected,
        float gradScale,
        float beta1,
        float beta2,
        float epsilon)
    {
        var pc = new byte[32];
        MemoryMarshal.Write(pc.AsSpan(0),  count);
        MemoryMarshal.Write(pc.AsSpan(4),  learningRateCorrected);
        MemoryMarshal.Write(pc.AsSpan(8),  gradScale);
        MemoryMarshal.Write(pc.AsSpan(12), beta1);
        MemoryMarshal.Write(pc.AsSpan(16), beta2);
        MemoryMarshal.Write(pc.AsSpan(20), epsilon);
        MemoryMarshal.Write(pc.AsSpan(24), 0u);
        MemoryMarshal.Write(pc.AsSpan(28), 0u);
        return pc;
    }

    // ── Vulkan availability check ─────────────────────────────────────────────

    /// <summary>
    /// Returns true if Vulkan compute is available on this machine.
    /// Safe to call from any thread; disposes the test device immediately.
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            using var test = new GpuDevice();
            return true;
        }
        catch
        {
            return false;
        }
    }
}
