using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// GPU CNN encoder. Phase 5 implements full single-sample forward/backward/apply
/// matching the native HWC CPU encoder:
/// conv forward with ReLU, projection forward, conv backward with ReLU gating,
/// and Adam updates on GPU.
///
/// The projection layer intentionally mirrors the native encoder's current behavior:
/// forward uses ReLU when conv layers are present, while backward applies no gate.
/// </summary>
internal sealed class GpuCnnEncoder : IEncoder, IDisposable
{
    private const float AdamBeta1   = 0.9f;
    private const float AdamBeta2   = 0.999f;
    private const float AdamEpsilon = 1e-8f;

    private sealed class GpuConvLayer
    {
        public int InH;
        public int InW;
        public int InC;
        public int OutH;
        public int OutW;
        public int OutC;
        public int Kernel;
        public int Stride;

        public int InputCount;
        public int OutputCount;
        public int FilterCount;

        public Rid FilterBuf;
        public Rid BiasBuf;
        public Rid PreActBuf;
        public Rid OutputBuf;
        public Rid BatchPreActBuf;
        public Rid BatchOutputBuf;

        public Rid GatedGradBuf;
        public Rid InputGradBuf;
        public Rid BatchGatedGradBuf;
        public Rid BatchInputGradBuf;
        public Rid GradFilterBuf;
        public Rid GradBiasBuf;
        public Rid FilterMoment1Buf;
        public Rid FilterMoment2Buf;
        public Rid BiasMoment1Buf;
        public Rid BiasMoment2Buf;
    }

    public int  OutputSize              { get; }
    public bool SupportsBatchedTraining => true;

    private readonly GpuDevice _gpu;
    private readonly int       _inputSize;
    private readonly int       _projInputSize;
    private readonly bool      _projectionUsesRelu;

    private readonly GpuConvLayer[] _convLayers;

    private readonly Rid _reluBackwardShader;
    private readonly Rid _reluBackwardPipeline;
    private readonly Rid _convForwardShader;
    private readonly Rid _convForwardPipeline;
    private readonly Rid _convBackwardFilterShader;
    private readonly Rid _convBackwardFilterPipeline;
    private readonly Rid _convBackwardInputShader;
    private readonly Rid _convBackwardInputPipeline;
    private readonly Rid _linearForwardShader;
    private readonly Rid _linearForwardPipeline;
    private readonly Rid _linearBackwardShader;
    private readonly Rid _linearBackwardPipeline;
    private readonly Rid _denseAdamShader;
    private readonly Rid _denseAdamPipeline;
    private readonly Rid _nativeAdamShader;
    private readonly Rid _nativeAdamPipeline;
    private readonly Rid _normAccumShader;
    private readonly Rid _normAccumPipeline;
    private readonly Rid _normAccumBuf;

    private readonly Rid _inputBuf;
    private readonly Rid _outputBuf;
    private Rid _batchInputBuf;
    private Rid _batchOutputBuf;
    private readonly Rid _weightBuf;
    private readonly Rid _biasBuf;
    private readonly Rid _outputGradBuf;
    private Rid _batchOutputGradBuf;
    private readonly Rid _projInputGradBuf;
    private Rid _batchProjInputGradBuf;
    private readonly Rid _gradWeightBuf;
    private readonly Rid _gradBiasBuf;
    private readonly Rid _weightMoment1Buf;
    private readonly Rid _weightMoment2Buf;
    private readonly Rid _biasMoment1Buf;
    private readonly Rid _biasMoment2Buf;

    private readonly int _weightCount;
    private readonly GpuLinearGradientToken _gradientToken;
    private int _batchCapacity;

    private float _adamB1Pow = 1f;
    private float _adamB2Pow = 1f;

    public GpuCnnEncoder(int width, int height, int channels, RLCnnEncoderDef def)
    {
        if (def.FilterCounts.Length != def.KernelSizes.Length || def.FilterCounts.Length != def.Strides.Length)
            throw new ArgumentException(
                "[GpuCnnEncoder] FilterCounts, KernelSizes, and Strides must have equal length.",
                nameof(def));

        OutputSize = def.OutputSize;
        _inputSize = width * height * channels;

        _gpu = new GpuDevice();

        _reluBackwardShader       = _gpu.CompileComputeShader(GpuShaderSources.ReluBackward);
        _reluBackwardPipeline     = _gpu.Rd.ComputePipelineCreate(_reluBackwardShader);
        _convForwardShader        = _gpu.CompileComputeShader(GpuShaderSources.ConvForward);
        _convForwardPipeline      = _gpu.Rd.ComputePipelineCreate(_convForwardShader);
        _convBackwardFilterShader = _gpu.CompileComputeShader(GpuShaderSources.ConvBackwardFilter);
        _convBackwardFilterPipeline = _gpu.Rd.ComputePipelineCreate(_convBackwardFilterShader);
        _convBackwardInputShader  = _gpu.CompileComputeShader(GpuShaderSources.ConvBackwardInput);
        _convBackwardInputPipeline = _gpu.Rd.ComputePipelineCreate(_convBackwardInputShader);
        _linearForwardShader      = _gpu.CompileComputeShader(GpuShaderSources.LinearForward);
        _linearForwardPipeline    = _gpu.Rd.ComputePipelineCreate(_linearForwardShader);
        _linearBackwardShader     = _gpu.CompileComputeShader(GpuShaderSources.LinearBackward);
        _linearBackwardPipeline   = _gpu.Rd.ComputePipelineCreate(_linearBackwardShader);
        _denseAdamShader          = _gpu.CompileComputeShader(GpuShaderSources.AdamUpdate);
        _denseAdamPipeline        = _gpu.Rd.ComputePipelineCreate(_denseAdamShader);
        _nativeAdamShader         = _gpu.CompileComputeShader(GpuShaderSources.AdamUpdateNative);
        _nativeAdamPipeline       = _gpu.Rd.ComputePipelineCreate(_nativeAdamShader);
        _normAccumShader          = _gpu.CompileComputeShader(GpuShaderSources.NormSquaredAccumulate);
        _normAccumPipeline        = _gpu.Rd.ComputePipelineCreate(_normAccumShader);
        _normAccumBuf             = _gpu.CreateBuffer(1, new float[1]);

        _inputBuf = _gpu.CreateBuffer(_inputSize);

        _convLayers = new GpuConvLayer[def.FilterCounts.Length];
        var prevH = height;
        var prevW = width;
        var prevC = channels;
        for (var i = 0; i < _convLayers.Length; i++)
        {
            var outC   = def.FilterCounts[i];
            var kernel = def.KernelSizes[i];
            var stride = def.Strides[i];
            var outH   = (prevH - kernel) / stride + 1;
            var outW   = (prevW - kernel) / stride + 1;

            if (outH <= 0 || outW <= 0)
                throw new InvalidOperationException(
                    $"[GpuCnnEncoder] Conv layer {i} produces invalid shape {outH}x{outW}.");

            var filterCount = outC * kernel * kernel * prevC;
            var inputCount  = prevH * prevW * prevC;
            var outputCount = outH * outW * outC;

            var layer = new GpuConvLayer
            {
                InH = prevH,
                InW = prevW,
                InC = prevC,
                OutH = outH,
                OutW = outW,
                OutC = outC,
                Kernel = kernel,
                Stride = stride,
                InputCount = inputCount,
                OutputCount = outputCount,
                FilterCount = filterCount,
                FilterBuf = _gpu.CreateBuffer(filterCount, CreateInitialWeights(filterCount, kernel * kernel * prevC)),
                BiasBuf = _gpu.CreateBuffer(outC, new float[outC]),
                PreActBuf = _gpu.CreateBuffer(outputCount),
                OutputBuf = _gpu.CreateBuffer(outputCount),
                GatedGradBuf = _gpu.CreateBuffer(outputCount),
                InputGradBuf = _gpu.CreateBuffer(inputCount),
                GradFilterBuf = _gpu.CreateBuffer(filterCount),
                GradBiasBuf = _gpu.CreateBuffer(outC),
                FilterMoment1Buf = _gpu.CreateBuffer(filterCount),
                FilterMoment2Buf = _gpu.CreateBuffer(filterCount),
                BiasMoment1Buf = _gpu.CreateBuffer(outC),
                BiasMoment2Buf = _gpu.CreateBuffer(outC),
            };
            _convLayers[i] = layer;

            prevH = outH;
            prevW = outW;
            prevC = outC;
        }

        _projInputSize = prevH * prevW * prevC;
        _projectionUsesRelu = _convLayers.Length > 0;
        _weightCount = _projInputSize * OutputSize;

        _outputBuf         = _gpu.CreateBuffer(OutputSize);
        _weightBuf         = _gpu.CreateBuffer(_weightCount, CreateInitialWeights(_weightCount, _projInputSize));
        _biasBuf           = _gpu.CreateBuffer(OutputSize, new float[OutputSize]);
        _outputGradBuf     = _gpu.CreateBuffer(OutputSize);
        _projInputGradBuf  = _gpu.CreateBuffer(_projInputSize);
        _gradWeightBuf     = _gpu.CreateBuffer(_weightCount);
        _gradBiasBuf       = _gpu.CreateBuffer(OutputSize);
        _weightMoment1Buf  = _gpu.CreateBuffer(_weightCount);
        _weightMoment2Buf  = _gpu.CreateBuffer(_weightCount);
        _biasMoment1Buf    = _gpu.CreateBuffer(OutputSize);
        _biasMoment2Buf    = _gpu.CreateBuffer(OutputSize);

        _gradientToken = new GpuLinearGradientToken(this);

        GD.Print(
            $"[GpuCnnEncoder] Initialised conv_layers={_convLayers.Length}, input={_inputSize}, proj_in={_projInputSize}, output={OutputSize}.");
    }

    public float[] Forward(float[] input)
    {
        if (input.Length != _inputSize)
            throw new ArgumentException(
                $"[GpuCnnEncoder] Forward expected {_inputSize} floats, got {input.Length}.",
                nameof(input));

        _gpu.UploadBuffer(_inputBuf, input, _inputSize);

        var currentInputBuf = _inputBuf;
        var currentInputSize = _inputSize;

        foreach (var layer in _convLayers)
        {
            var uniformSet = BuildUniformSet(_convForwardShader,
                (currentInputBuf, 0, RenderingDevice.UniformType.StorageBuffer),
                (layer.FilterBuf, 1, RenderingDevice.UniformType.StorageBuffer),
                (layer.BiasBuf,   2, RenderingDevice.UniformType.StorageBuffer),
                (layer.PreActBuf, 3, RenderingDevice.UniformType.StorageBuffer),
                (layer.OutputBuf, 4, RenderingDevice.UniformType.StorageBuffer));

            var pc = GpuDevice.PushConstant(
                (uint)layer.InH, (uint)layer.InW, (uint)layer.InC, (uint)layer.OutH,
                (uint)layer.OutW, (uint)layer.OutC, (uint)layer.Kernel, (uint)layer.Stride);
            Dispatch(_convForwardPipeline, uniformSet, pc, (uint)layer.OutputCount);
            _gpu.Rd.FreeRid(uniformSet);

            currentInputBuf = layer.OutputBuf;
            currentInputSize = layer.OutputCount;
        }

        var projUniformSet = BuildUniformSet(_linearForwardShader,
            (currentInputBuf, 0, RenderingDevice.UniformType.StorageBuffer),
            (_weightBuf,      1, RenderingDevice.UniformType.StorageBuffer),
            (_biasBuf,        2, RenderingDevice.UniformType.StorageBuffer),
            (_outputBuf,      3, RenderingDevice.UniformType.StorageBuffer));

        var projPc = GpuDevice.PushConstant(
            (uint)currentInputSize,
            (uint)OutputSize,
            _projectionUsesRelu ? 1u : 0u,
            0u);
        Dispatch(_linearForwardPipeline, projUniformSet, projPc, (uint)OutputSize);
        _gpu.Rd.FreeRid(projUniformSet);

        return _gpu.DownloadBuffer(_outputBuf, OutputSize);
    }

    public void ForwardBatch(float[] inputBatch, int batchSize, float[] outputBatch)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "[GpuCnnEncoder] ForwardBatch batchSize must be > 0.");
        if (inputBatch.Length != batchSize * _inputSize)
            throw new ArgumentException(
                $"[GpuCnnEncoder] ForwardBatch expected {batchSize * _inputSize} input floats, got {inputBatch.Length}.",
                nameof(inputBatch));
        if (outputBatch.Length != batchSize * OutputSize)
            throw new ArgumentException(
                $"[GpuCnnEncoder] ForwardBatch expected {batchSize * OutputSize} output floats, got {outputBatch.Length}.",
                nameof(outputBatch));

        EnsureBatchCapacity(batchSize);
        var tUpload = RLProfiler.Begin();
        _gpu.UploadBuffer(_batchInputBuf, inputBatch, inputBatch.Length);
        RLProfiler.End("GpuCnn.ForwardBatch.Upload", tUpload);

        var currentInputBuf  = _batchInputBuf;
        var currentInputSize = _inputSize;

        // All conv + projection dispatches in a single compute list — one Submit+Sync.
        var tDispatch = RLProfiler.Begin();
        var list = _gpu.BeginComputeList();
        var uniformSets = new List<Rid>();

        foreach (var layer in _convLayers)
        {
            var us = BuildUniformSet(_convForwardShader,
                (currentInputBuf,      0, RenderingDevice.UniformType.StorageBuffer),
                (layer.FilterBuf,      1, RenderingDevice.UniformType.StorageBuffer),
                (layer.BiasBuf,        2, RenderingDevice.UniformType.StorageBuffer),
                (layer.BatchPreActBuf, 3, RenderingDevice.UniformType.StorageBuffer),
                (layer.BatchOutputBuf, 4, RenderingDevice.UniformType.StorageBuffer));
            uniformSets.Add(us);

            var pc = GpuDevice.PushConstant(
                (uint)layer.InH, (uint)layer.InW, (uint)layer.InC, (uint)layer.OutH,
                (uint)layer.OutW, (uint)layer.OutC, (uint)layer.Kernel, (uint)layer.Stride,
                (uint)batchSize);
            _gpu.DispatchToList(list, _convForwardPipeline, us, pc, (uint)(batchSize * layer.OutputCount));
            _gpu.AddBarrier(list); // conv[i] output feeds conv[i+1] or projection input

            currentInputBuf  = layer.BatchOutputBuf;
            currentInputSize = layer.OutputCount;
        }

        var projUs = BuildUniformSet(_linearForwardShader,
            (currentInputBuf, 0, RenderingDevice.UniformType.StorageBuffer),
            (_weightBuf,      1, RenderingDevice.UniformType.StorageBuffer),
            (_biasBuf,        2, RenderingDevice.UniformType.StorageBuffer),
            (_batchOutputBuf, 3, RenderingDevice.UniformType.StorageBuffer));
        uniformSets.Add(projUs);

        var projPc = GpuDevice.PushConstant(
            (uint)currentInputSize,
            (uint)OutputSize,
            _projectionUsesRelu ? 1u : 0u,
            (uint)batchSize);
        _gpu.DispatchToList(list, _linearForwardPipeline, projUs, projPc, (uint)(batchSize * OutputSize));

        _gpu.EndSubmitAndSync();
        foreach (var us in uniformSets) _gpu.Rd.FreeRid(us);
        RLProfiler.End("GpuCnn.ForwardBatch.Dispatch", tDispatch);

        var tReadback = RLProfiler.Begin();
        var downloaded = _gpu.DownloadBuffer(_batchOutputBuf, batchSize * OutputSize);
        Array.Copy(downloaded, outputBatch, downloaded.Length);
        RLProfiler.End("GpuCnn.ForwardBatch.Readback", tReadback);
    }

    public ICnnGradientToken CreateGradientToken()
    {
        ZeroGradients();
        return _gradientToken;
    }

    public float[] AccumulateGradients(float[] outputGrad, ICnnGradientToken token)
    {
        ValidateToken(token);

        if (outputGrad.Length != OutputSize)
            throw new ArgumentException(
                $"[GpuCnnEncoder] AccumulateGradients expected {OutputSize} floats, got {outputGrad.Length}.",
                nameof(outputGrad));

        _gpu.UploadBuffer(_outputGradBuf, outputGrad, OutputSize);

        var projectionInputBuf = GetProjectionInputBuffer();
        var projectionThreadCount = Math.Max(_projInputSize, OutputSize);
        var projUniformSet = BuildUniformSet(_linearBackwardShader,
            (projectionInputBuf, 0, RenderingDevice.UniformType.StorageBuffer),
            (_weightBuf,         1, RenderingDevice.UniformType.StorageBuffer),
            (_outputGradBuf,     2, RenderingDevice.UniformType.StorageBuffer),
            (_projInputGradBuf,  3, RenderingDevice.UniformType.StorageBuffer),
            (_gradWeightBuf,     4, RenderingDevice.UniformType.StorageBuffer),
            (_gradBiasBuf,       5, RenderingDevice.UniformType.StorageBuffer));

        var projPc = GpuDevice.PushConstant((uint)_projInputSize, (uint)OutputSize, (uint)projectionThreadCount, 0u);
        Dispatch(_linearBackwardPipeline, projUniformSet, projPc, (uint)projectionThreadCount);
        _gpu.Rd.FreeRid(projUniformSet);

        if (_convLayers.Length == 0)
            return _gpu.DownloadBuffer(_projInputGradBuf, _inputSize);

        var currentGradBuf = _projInputGradBuf;
        for (var layerIndex = _convLayers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            var layer = _convLayers[layerIndex];

            var reluUniformSet = BuildUniformSet(_reluBackwardShader,
                (currentGradBuf,     0, RenderingDevice.UniformType.StorageBuffer),
                (layer.PreActBuf,    1, RenderingDevice.UniformType.StorageBuffer),
                (layer.GatedGradBuf, 2, RenderingDevice.UniformType.StorageBuffer));
            var reluPc = GpuDevice.PushConstant((uint)layer.OutputCount);
            Dispatch(_reluBackwardPipeline, reluUniformSet, reluPc, (uint)layer.OutputCount);
            _gpu.Rd.FreeRid(reluUniformSet);

            var layerInputBuf = layerIndex == 0 ? _inputBuf : _convLayers[layerIndex - 1].OutputBuf;

            var filterUniformSet = BuildUniformSet(_convBackwardFilterShader,
                (layerInputBuf,      0, RenderingDevice.UniformType.StorageBuffer),
                (layer.GatedGradBuf, 1, RenderingDevice.UniformType.StorageBuffer),
                (layer.GradFilterBuf, 2, RenderingDevice.UniformType.StorageBuffer),
                (layer.GradBiasBuf,   3, RenderingDevice.UniformType.StorageBuffer));
            var convPc = GpuDevice.PushConstant(
                (uint)layer.InH, (uint)layer.InW, (uint)layer.InC, (uint)layer.OutH,
                (uint)layer.OutW, (uint)layer.OutC, (uint)layer.Kernel, (uint)layer.Stride);
            Dispatch(_convBackwardFilterPipeline, filterUniformSet, convPc, (uint)Math.Max(layer.FilterCount, layer.OutC));
            _gpu.Rd.FreeRid(filterUniformSet);

            var inputUniformSet = BuildUniformSet(_convBackwardInputShader,
                (layer.GatedGradBuf, 0, RenderingDevice.UniformType.StorageBuffer),
                (layer.FilterBuf,    1, RenderingDevice.UniformType.StorageBuffer),
                (layer.InputGradBuf, 2, RenderingDevice.UniformType.StorageBuffer));
            Dispatch(_convBackwardInputPipeline, inputUniformSet, convPc, (uint)layer.InputCount);
            _gpu.Rd.FreeRid(inputUniformSet);

            currentGradBuf = layer.InputGradBuf;
        }

        return _gpu.DownloadBuffer(currentGradBuf, _inputSize);
    }

    public void AccumulateGradientsBatch(float[] outputGradBatch, int batchSize, ICnnGradientToken token)
    {
        ValidateToken(token);

        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "[GpuCnnEncoder] AccumulateGradientsBatch batchSize must be > 0.");
        if (outputGradBatch.Length != batchSize * OutputSize)
            throw new ArgumentException(
                $"[GpuCnnEncoder] AccumulateGradientsBatch expected {batchSize * OutputSize} floats, got {outputGradBatch.Length}.",
                nameof(outputGradBatch));

        EnsureBatchCapacity(batchSize);
        var tUpload = RLProfiler.Begin();
        _gpu.UploadBuffer(_batchOutputGradBuf, outputGradBatch, outputGradBatch.Length);
        RLProfiler.End("GpuCnn.BackwardBatch.UploadGrad", tUpload);

        // Linear backward + all conv backward passes in a single compute list.
        var tDispatch = RLProfiler.Begin();
        var list = _gpu.BeginComputeList();
        var uniformSets = new List<Rid>();

        var projectionInputBuf    = GetBatchProjectionInputBuffer();
        var projectionThreadCount = Math.Max(batchSize * _projInputSize, Math.Max(_weightCount, OutputSize));

        var projUs = BuildUniformSet(_linearBackwardShader,
            (projectionInputBuf,     0, RenderingDevice.UniformType.StorageBuffer),
            (_weightBuf,             1, RenderingDevice.UniformType.StorageBuffer),
            (_batchOutputGradBuf,    2, RenderingDevice.UniformType.StorageBuffer),
            (_batchProjInputGradBuf, 3, RenderingDevice.UniformType.StorageBuffer),
            (_gradWeightBuf,         4, RenderingDevice.UniformType.StorageBuffer),
            (_gradBiasBuf,           5, RenderingDevice.UniformType.StorageBuffer));
        uniformSets.Add(projUs);

        var projPc = GpuDevice.PushConstant(
            (uint)_projInputSize, (uint)OutputSize, (uint)batchSize, (uint)projectionThreadCount);
        _gpu.DispatchToList(list, _linearBackwardPipeline, projUs, projPc, (uint)projectionThreadCount);

        if (_convLayers.Length > 0)
        {
            _gpu.AddBarrier(list); // projection input grad feeds first relu backward

            var currentGradBuf = _batchProjInputGradBuf;
            for (var layerIndex = _convLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                var layer = _convLayers[layerIndex];

                var reluUs = BuildUniformSet(_reluBackwardShader,
                    (currentGradBuf,          0, RenderingDevice.UniformType.StorageBuffer),
                    (layer.BatchPreActBuf,    1, RenderingDevice.UniformType.StorageBuffer),
                    (layer.BatchGatedGradBuf, 2, RenderingDevice.UniformType.StorageBuffer));
                uniformSets.Add(reluUs);
                var reluPc = GpuDevice.PushConstant((uint)(batchSize * layer.OutputCount));
                _gpu.DispatchToList(list, _reluBackwardPipeline, reluUs, reluPc, (uint)(batchSize * layer.OutputCount));
                _gpu.AddBarrier(list); // gated grad feeds filter + input backward

                var layerInputBuf = layerIndex == 0 ? _batchInputBuf : _convLayers[layerIndex - 1].BatchOutputBuf;
                var convPc = GpuDevice.PushConstant(
                    (uint)layer.InH, (uint)layer.InW, (uint)layer.InC, (uint)layer.OutH,
                    (uint)layer.OutW, (uint)layer.OutC, (uint)layer.Kernel, (uint)layer.Stride,
                    (uint)batchSize);

                // Filter and input backward are independent readers of gated grad — no barrier between them.
                var filterUs = BuildUniformSet(_convBackwardFilterShader,
                    (layerInputBuf,           0, RenderingDevice.UniformType.StorageBuffer),
                    (layer.BatchGatedGradBuf, 1, RenderingDevice.UniformType.StorageBuffer),
                    (layer.GradFilterBuf,     2, RenderingDevice.UniformType.StorageBuffer),
                    (layer.GradBiasBuf,       3, RenderingDevice.UniformType.StorageBuffer));
                uniformSets.Add(filterUs);
                _gpu.DispatchToList(list, _convBackwardFilterPipeline, filterUs, convPc, (uint)Math.Max(layer.FilterCount, layer.OutC));

                var inputUs = BuildUniformSet(_convBackwardInputShader,
                    (layer.BatchGatedGradBuf, 0, RenderingDevice.UniformType.StorageBuffer),
                    (layer.FilterBuf,         1, RenderingDevice.UniformType.StorageBuffer),
                    (layer.BatchInputGradBuf, 2, RenderingDevice.UniformType.StorageBuffer));
                uniformSets.Add(inputUs);
                _gpu.DispatchToList(list, _convBackwardInputPipeline, inputUs, convPc, (uint)(batchSize * layer.InputCount));

                if (layerIndex > 0)
                    _gpu.AddBarrier(list); // input grad feeds the next relu backward

                currentGradBuf = layer.BatchInputGradBuf;
            }
        }

        _gpu.EndSubmitAndSync();
        foreach (var us in uniformSets) _gpu.Rd.FreeRid(us);
        RLProfiler.End("GpuCnn.BackwardBatch.Dispatch", tDispatch);
    }

    public void ApplyGradients(ICnnGradientToken token, float learningRate, float gradScale)
    {
        ValidateToken(token);
        var tApply = RLProfiler.Begin();

        _adamB1Pow *= AdamBeta1;
        _adamB2Pow *= AdamBeta2;

        // All Adam updates are independent — no barriers needed. One list, one Submit+Sync.
        var list = _gpu.BeginComputeList();
        var uniformSets = new List<Rid>();

        if (_convLayers.Length == 0)
        {
            var b1Corr = 1f - _adamB1Pow;
            var b2Corr = 1f - _adamB2Pow;
            AdamDenseToList(list, uniformSets, _weightBuf, _gradWeightBuf, _weightMoment1Buf, _weightMoment2Buf, _weightCount,
                learningRate, gradScale, b1Corr, b2Corr);
            AdamDenseToList(list, uniformSets, _biasBuf, _gradBiasBuf, _biasMoment1Buf, _biasMoment2Buf, OutputSize,
                learningRate, gradScale, b1Corr, b2Corr);
        }
        else
        {
            var lrCorrected = learningRate * Mathf.Sqrt(1f - _adamB2Pow) / (1f - _adamB1Pow);
            foreach (var layer in _convLayers)
            {
                AdamNativeToList(list, uniformSets, layer.FilterBuf, layer.GradFilterBuf,
                    layer.FilterMoment1Buf, layer.FilterMoment2Buf, layer.FilterCount, lrCorrected, gradScale);
                AdamNativeToList(list, uniformSets, layer.BiasBuf, layer.GradBiasBuf,
                    layer.BiasMoment1Buf, layer.BiasMoment2Buf, layer.OutC, lrCorrected, gradScale);
            }
            AdamNativeToList(list, uniformSets, _weightBuf, _gradWeightBuf,
                _weightMoment1Buf, _weightMoment2Buf, _weightCount, lrCorrected, gradScale);
            AdamNativeToList(list, uniformSets, _biasBuf, _gradBiasBuf,
                _biasMoment1Buf, _biasMoment2Buf, OutputSize, lrCorrected, gradScale);
        }

        _gpu.EndSubmitAndSync();
        foreach (var us in uniformSets) _gpu.Rd.FreeRid(us);
        RLProfiler.End("GpuCnn.ApplyGradients", tApply);
    }

    public float GradNormSquared(ICnnGradientToken token)
    {
        ValidateToken(token);
        var tNorm = RLProfiler.Begin();

        // Zero the 1-float accumulator on the CPU side (host write, no compute needed).
        _gpu.ClearBuffer(_normAccumBuf, 1);

        // Dispatch one NormSquaredAccumulate per gradient buffer into a single compute list.
        // Each dispatch uses exactly 1 workgroup; barriers ensure accumulator writes are ordered.
        var list = _gpu.BeginComputeList();
        var uniformSets = new List<Rid>();

        void Accumulate(Rid gradBuf, int count)
        {
            var us = BuildUniformSet(_normAccumShader,
                (gradBuf,       0, RenderingDevice.UniformType.StorageBuffer),
                (_normAccumBuf, 1, RenderingDevice.UniformType.StorageBuffer));
            uniformSets.Add(us);
            _gpu.DispatchToList(list, _normAccumPipeline, us,
                GpuDevice.PushConstant((uint)count), 1u); // 1u → exactly 1 workgroup
            _gpu.AddBarrier(list); // each += must complete before the next reads the accumulator
        }

        foreach (var layer in _convLayers)
        {
            Accumulate(layer.GradFilterBuf, layer.FilterCount);
            Accumulate(layer.GradBiasBuf,   layer.OutC);
        }
        Accumulate(_gradWeightBuf, _weightCount);
        Accumulate(_gradBiasBuf,   OutputSize);

        _gpu.EndSubmitAndSync();
        foreach (var us in uniformSets) _gpu.Rd.FreeRid(us);

        var result = _gpu.DownloadBuffer(_normAccumBuf, 1)[0];
        RLProfiler.End("GpuCnn.GradNormSquared", tNorm);
        return result;
    }

    public void AppendSerialized(ICollection<float> weights, ICollection<int> shapes)
    {
        shapes.Add(_convLayers.Length);
        foreach (var layer in _convLayers)
        {
            shapes.Add(layer.OutC);
            shapes.Add(layer.Kernel);
            shapes.Add(layer.Kernel);
            shapes.Add(layer.InC);
            shapes.Add(layer.Stride);

            foreach (var v in _gpu.DownloadBuffer(layer.FilterBuf, layer.FilterCount)) weights.Add(v);
            foreach (var v in _gpu.DownloadBuffer(layer.BiasBuf, layer.OutC))          weights.Add(v);
        }

        shapes.Add(_projInputSize);
        shapes.Add(OutputSize);

        foreach (var v in _gpu.DownloadBuffer(_weightBuf, _weightCount)) weights.Add(v);
        foreach (var v in _gpu.DownloadBuffer(_biasBuf, OutputSize))     weights.Add(v);
    }

    public void LoadSerialized(IReadOnlyList<float> weights, ref int wi,
                               IReadOnlyList<int>   shapes,  ref int si)
    {
        var nConv = shapes[si++];
        if (nConv != _convLayers.Length)
            throw new InvalidOperationException(
                $"[GpuCnnEncoder] Checkpoint conv count {nConv} does not match active encoder {_convLayers.Length}.");

        for (var layerIndex = 0; layerIndex < _convLayers.Length; layerIndex++)
        {
            var layer = _convLayers[layerIndex];
            var outC  = shapes[si++];
            var kH    = shapes[si++];
            var kW    = shapes[si++];
            var inC   = shapes[si++];
            var stride = shapes[si++];

            if (outC != layer.OutC || kH != layer.Kernel || kW != layer.Kernel || inC != layer.InC || stride != layer.Stride)
                throw new InvalidOperationException(
                    $"[GpuCnnEncoder] Conv layer {layerIndex} checkpoint shape does not match the active encoder.");

            var loadedFilters = new float[layer.FilterCount];
            var loadedLayerBiases = new float[layer.OutC];
            for (var i = 0; i < loadedFilters.Length; i++) loadedFilters[i] = weights[wi++];
            for (var i = 0; i < loadedLayerBiases.Length; i++) loadedLayerBiases[i] = weights[wi++];

            _gpu.UploadBuffer(layer.FilterBuf, loadedFilters, loadedFilters.Length);
            _gpu.UploadBuffer(layer.BiasBuf, loadedLayerBiases, loadedLayerBiases.Length);
        }

        var projIn  = shapes[si++];
        var projOut = shapes[si++];
        if (projIn != _projInputSize || projOut != OutputSize)
            throw new InvalidOperationException(
                "[GpuCnnEncoder] Checkpoint projection shape does not match the active encoder.");

        var loadedWeights = new float[_weightCount];
        var loadedBiases  = new float[OutputSize];
        for (var i = 0; i < loadedWeights.Length; i++) loadedWeights[i] = weights[wi++];
        for (var i = 0; i < loadedBiases.Length;  i++) loadedBiases[i]  = weights[wi++];

        _gpu.UploadBuffer(_weightBuf, loadedWeights, loadedWeights.Length);
        _gpu.UploadBuffer(_biasBuf, loadedBiases, loadedBiases.Length);
        ResetOptimizerState();
    }

    public void CopyWeightsTo(IEncoder other)
    {
        if (ReferenceEquals(this, other))
            return;

        var weights = new List<float>();
        var shapes  = new List<int>();
        AppendSerialized(weights, shapes);

        var wi = 0;
        var si = 0;
        other.LoadSerialized(weights, ref wi, shapes, ref si);
    }

    public void Dispose()
    {
        FreeBatchBuffers();
        _gpu.Rd.FreeRid(_biasMoment2Buf);
        _gpu.Rd.FreeRid(_biasMoment1Buf);
        _gpu.Rd.FreeRid(_weightMoment2Buf);
        _gpu.Rd.FreeRid(_weightMoment1Buf);
        _gpu.Rd.FreeRid(_gradBiasBuf);
        _gpu.Rd.FreeRid(_gradWeightBuf);
        _gpu.Rd.FreeRid(_projInputGradBuf);
        _gpu.Rd.FreeRid(_outputGradBuf);
        _gpu.Rd.FreeRid(_biasBuf);
        _gpu.Rd.FreeRid(_weightBuf);
        _gpu.Rd.FreeRid(_outputBuf);

        foreach (var layer in _convLayers)
        {
            _gpu.Rd.FreeRid(layer.BiasMoment2Buf);
            _gpu.Rd.FreeRid(layer.BiasMoment1Buf);
            _gpu.Rd.FreeRid(layer.FilterMoment2Buf);
            _gpu.Rd.FreeRid(layer.FilterMoment1Buf);
            _gpu.Rd.FreeRid(layer.GradBiasBuf);
            _gpu.Rd.FreeRid(layer.GradFilterBuf);
            _gpu.Rd.FreeRid(layer.InputGradBuf);
            _gpu.Rd.FreeRid(layer.GatedGradBuf);
            _gpu.Rd.FreeRid(layer.OutputBuf);
            _gpu.Rd.FreeRid(layer.PreActBuf);
            _gpu.Rd.FreeRid(layer.BiasBuf);
            _gpu.Rd.FreeRid(layer.FilterBuf);
        }

        _gpu.Rd.FreeRid(_inputBuf);
        _gpu.Rd.FreeRid(_normAccumBuf);
        _gpu.Rd.FreeRid(_normAccumPipeline);
        _gpu.Rd.FreeRid(_normAccumShader);
        _gpu.Rd.FreeRid(_nativeAdamPipeline);
        _gpu.Rd.FreeRid(_nativeAdamShader);
        _gpu.Rd.FreeRid(_denseAdamPipeline);
        _gpu.Rd.FreeRid(_denseAdamShader);
        _gpu.Rd.FreeRid(_linearBackwardPipeline);
        _gpu.Rd.FreeRid(_linearBackwardShader);
        _gpu.Rd.FreeRid(_linearForwardPipeline);
        _gpu.Rd.FreeRid(_linearForwardShader);
        _gpu.Rd.FreeRid(_convBackwardInputPipeline);
        _gpu.Rd.FreeRid(_convBackwardInputShader);
        _gpu.Rd.FreeRid(_convBackwardFilterPipeline);
        _gpu.Rd.FreeRid(_convBackwardFilterShader);
        _gpu.Rd.FreeRid(_convForwardPipeline);
        _gpu.Rd.FreeRid(_convForwardShader);
        _gpu.Rd.FreeRid(_reluBackwardPipeline);
        _gpu.Rd.FreeRid(_reluBackwardShader);
        _gpu.Dispose();
    }

    internal float[] DebugReadGradientBuffer()
    {
        var totalSize = 0;
        foreach (var layer in _convLayers)
            totalSize += layer.FilterCount + layer.OutC;
        totalSize += _weightCount + OutputSize;

        var flat = new float[totalSize];
        var offset = 0;
        foreach (var layer in _convLayers)
        {
            var fg = _gpu.DownloadBuffer(layer.GradFilterBuf, layer.FilterCount);
            Array.Copy(fg, 0, flat, offset, fg.Length);
            offset += fg.Length;

            var bg = _gpu.DownloadBuffer(layer.GradBiasBuf, layer.OutC);
            Array.Copy(bg, 0, flat, offset, bg.Length);
            offset += bg.Length;
        }

        var pw = _gpu.DownloadBuffer(_gradWeightBuf, _weightCount);
        Array.Copy(pw, 0, flat, offset, pw.Length);
        offset += pw.Length;

        var pb = _gpu.DownloadBuffer(_gradBiasBuf, OutputSize);
        Array.Copy(pb, 0, flat, offset, pb.Length);
        return flat;
    }

    private void ZeroGradients()
    {
        foreach (var layer in _convLayers)
        {
            _gpu.ClearBuffer(layer.GradFilterBuf, layer.FilterCount);
            _gpu.ClearBuffer(layer.GradBiasBuf, layer.OutC);
        }

        _gpu.ClearBuffer(_gradWeightBuf, _weightCount);
        _gpu.ClearBuffer(_gradBiasBuf, OutputSize);
    }

    private void ResetOptimizerState()
    {
        foreach (var layer in _convLayers)
        {
            _gpu.ClearBuffer(layer.FilterMoment1Buf, layer.FilterCount);
            _gpu.ClearBuffer(layer.FilterMoment2Buf, layer.FilterCount);
            _gpu.ClearBuffer(layer.BiasMoment1Buf, layer.OutC);
            _gpu.ClearBuffer(layer.BiasMoment2Buf, layer.OutC);
        }

        _gpu.ClearBuffer(_weightMoment1Buf, _weightCount);
        _gpu.ClearBuffer(_weightMoment2Buf, _weightCount);
        _gpu.ClearBuffer(_biasMoment1Buf, OutputSize);
        _gpu.ClearBuffer(_biasMoment2Buf, OutputSize);
        ZeroGradients();
        _adamB1Pow = 1f;
        _adamB2Pow = 1f;
    }

    private void ValidateToken(ICnnGradientToken token)
    {
        if (!ReferenceEquals(token, _gradientToken))
            throw new InvalidOperationException("[GpuCnnEncoder] Gradient token does not belong to this encoder.");
    }

    private Rid GetProjectionInputBuffer() =>
        _convLayers.Length == 0 ? _inputBuf : _convLayers[_convLayers.Length - 1].OutputBuf;

    private Rid GetBatchProjectionInputBuffer() =>
        _convLayers.Length == 0 ? _batchInputBuf : _convLayers[_convLayers.Length - 1].BatchOutputBuf;

    private void EnsureBatchCapacity(int batchSize)
    {
        if (_batchCapacity >= batchSize)
            return;

        FreeBatchBuffers();

        _batchInputBuf = _gpu.CreateBuffer(batchSize * _inputSize);
        _batchOutputBuf = _gpu.CreateBuffer(batchSize * OutputSize);
        _batchOutputGradBuf = _gpu.CreateBuffer(batchSize * OutputSize);
        _batchProjInputGradBuf = _gpu.CreateBuffer(batchSize * _projInputSize);

        foreach (var layer in _convLayers)
        {
            layer.BatchPreActBuf = _gpu.CreateBuffer(batchSize * layer.OutputCount);
            layer.BatchOutputBuf = _gpu.CreateBuffer(batchSize * layer.OutputCount);
            layer.BatchGatedGradBuf = _gpu.CreateBuffer(batchSize * layer.OutputCount);
            layer.BatchInputGradBuf = _gpu.CreateBuffer(batchSize * layer.InputCount);
        }

        _batchCapacity = batchSize;
    }

    private void FreeBatchBuffers()
    {
        foreach (var layer in _convLayers)
        {
            FreeRidIfValid(layer.BatchInputGradBuf);
            FreeRidIfValid(layer.BatchGatedGradBuf);
            FreeRidIfValid(layer.BatchOutputBuf);
            FreeRidIfValid(layer.BatchPreActBuf);
            layer.BatchInputGradBuf = default;
            layer.BatchGatedGradBuf = default;
            layer.BatchOutputBuf = default;
            layer.BatchPreActBuf = default;
        }

        FreeRidIfValid(_batchProjInputGradBuf);
        FreeRidIfValid(_batchOutputGradBuf);
        FreeRidIfValid(_batchOutputBuf);
        FreeRidIfValid(_batchInputBuf);
        _batchProjInputGradBuf = default;
        _batchOutputGradBuf = default;
        _batchOutputBuf = default;
        _batchInputBuf = default;
        _batchCapacity = 0;
    }

    private void FreeRidIfValid(Rid rid)
    {
        if (rid.IsValid)
            _gpu.Rd.FreeRid(rid);
    }

    private Rid BuildUniformSet(Rid shader,
        params (Rid buffer, int binding, RenderingDevice.UniformType type)[] bindings)
    {
        var uniforms = new Godot.Collections.Array<RDUniform>();
        foreach (var (buffer, binding, type) in bindings)
        {
            var u = new RDUniform { UniformType = type, Binding = binding };
            u.AddId(buffer);
            uniforms.Add(u);
        }
        return _gpu.Rd.UniformSetCreate(uniforms, shader, 0);
    }

    private void Dispatch(Rid pipeline, Rid uniformSet, byte[] pushConstant, uint elementCount)
    {
        var list = _gpu.Rd.ComputeListBegin();
        _gpu.Rd.ComputeListBindComputePipeline(list, pipeline);
        _gpu.Rd.ComputeListBindUniformSet(list, uniformSet, 0);
        _gpu.Rd.ComputeListSetPushConstant(list, pushConstant, (uint)pushConstant.Length);
        _gpu.Rd.ComputeListDispatch(list, (elementCount + 255u) / 256u, 1, 1);
        _gpu.Rd.ComputeListEnd();
        _gpu.SubmitAndSync();
    }

    private void AdamDenseToList(
        long list, List<Rid> uniformSets,
        Rid paramBuf, Rid gradBuf, Rid moment1Buf, Rid moment2Buf,
        int count, float learningRate, float gradScale, float b1Corr, float b2Corr)
    {
        var us = BuildUniformSet(_denseAdamShader,
            (paramBuf,   0, RenderingDevice.UniformType.StorageBuffer),
            (gradBuf,    1, RenderingDevice.UniformType.StorageBuffer),
            (moment1Buf, 2, RenderingDevice.UniformType.StorageBuffer),
            (moment2Buf, 3, RenderingDevice.UniformType.StorageBuffer));
        uniformSets.Add(us);
        var pc = GpuDevice.PushConstantAdam((uint)count, learningRate, gradScale,
            b1Corr, b2Corr, AdamBeta1, AdamBeta2, AdamEpsilon);
        _gpu.DispatchToList(list, _denseAdamPipeline, us, pc, (uint)count);
    }

    private void AdamNativeToList(
        long list, List<Rid> uniformSets,
        Rid paramBuf, Rid gradBuf, Rid moment1Buf, Rid moment2Buf,
        int count, float learningRateCorrected, float gradScale)
    {
        var us = BuildUniformSet(_nativeAdamShader,
            (paramBuf,   0, RenderingDevice.UniformType.StorageBuffer),
            (gradBuf,    1, RenderingDevice.UniformType.StorageBuffer),
            (moment1Buf, 2, RenderingDevice.UniformType.StorageBuffer),
            (moment2Buf, 3, RenderingDevice.UniformType.StorageBuffer));
        uniformSets.Add(us);
        var pc = GpuDevice.PushConstantAdamNative((uint)count, learningRateCorrected,
            gradScale, AdamBeta1, AdamBeta2, AdamEpsilon);
        _gpu.DispatchToList(list, _nativeAdamPipeline, us, pc, (uint)count);
    }

    private static float[] CreateInitialWeights(int count, int fanIn)
    {
        var weights = new float[count];
        var rng = new RandomNumberGenerator();
        rng.Randomize();

        var scale = Mathf.Sqrt(2.0f / Mathf.Max(1, fanIn));
        for (var i = 0; i < weights.Length; i++)
            weights[i] = rng.Randfn(0.0f, scale);

        return weights;
    }
}

internal sealed class GpuLinearGradientToken : ICnnGradientToken
{
    private readonly GpuCnnEncoder _owner;

    public GpuLinearGradientToken(GpuCnnEncoder owner) => _owner = owner;
    public GpuCnnEncoder Owner => _owner;
}
