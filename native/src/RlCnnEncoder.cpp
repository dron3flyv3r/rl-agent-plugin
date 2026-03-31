#include "RlCnnEncoder.h"
#include "rl_simd.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// im2col / col2im
// ─────────────────────────────────────────────────────────────────────────────
// Input layout:  HWC  [inH, inW, inC]
// col layout:    [outH*outW, kernelH*kernelW*inC]

static void im2col(const float* input, float* col,
                   int inH, int inW, int inC,
                   int outH, int outW,
                   int kernelH, int kernelW, int stride)
{
    const int col_step = kernelH * kernelW * inC;
    for (int oh = 0; oh < outH; ++oh)
    for (int ow = 0; ow < outW; ++ow) {
        float* col_row = col + (oh * outW + ow) * col_step;
        for (int kh = 0; kh < kernelH; ++kh) {
            int ih = oh * stride + kh;
            for (int kw = 0; kw < kernelW; ++kw) {
                int iw = ow * stride + kw;
                int col_off = (kh * kernelW + kw) * inC;
                std::memcpy(col_row + col_off,
                            input + ih * inW * inC + iw * inC,
                            inC * sizeof(float));
            }
        }
    }
}

/// col2im: scatter col gradients back into input gradient (accumulate).
static void col2im(const float* col_grad, float* input_grad,
                   int inH, int inW, int inC,
                   int outH, int outW,
                   int kernelH, int kernelW, int stride)
{
    const int col_step = kernelH * kernelW * inC;
    std::memset(input_grad, 0, inH * inW * inC * sizeof(float));
    for (int oh = 0; oh < outH; ++oh)
    for (int ow = 0; ow < outW; ++ow) {
        const float* col_row = col_grad + (oh * outW + ow) * col_step;
        for (int kh = 0; kh < kernelH; ++kh) {
            int ih = oh * stride + kh;
            for (int kw = 0; kw < kernelW; ++kw) {
                int iw = ow * stride + kw;
                int col_off = (kh * kernelW + kw) * inC;
                // accumulate
                float*       dst = input_grad + ih * inW * inC + iw * inC;
                const float* src = col_row + col_off;
                for (int c = 0; c < inC; ++c)
                    dst[c] += src[c];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConvLayer
// ─────────────────────────────────────────────────────────────────────────────

void RlCnnEncoder::ConvLayer::init(int inH_, int inW_, int inC_,
                                   int outC_, int kernel, int stride_,
                                   uint64_t seed, int grad_offset)
{
    inH = inH_; inW = inW_; inC = inC_;
    outC = outC_; kernelH = kernel; kernelW = kernel; stride = stride_;
    outH = (inH - kernel) / stride_ + 1;
    outW = (inW - kernel) / stride_ + 1;

    int fsize = outC * kernelH * kernelW * inC;
    filters.assign(fsize, 0.f);
    biases.assign(outC, 0.f);
    wm.assign(fsize, 0.f); wv.assign(fsize, 0.f);
    bm.assign(outC,  0.f); bv.assign(outC,  0.f);

    cached_input.resize(inH * inW * inC);
    cached_preact.resize(outH * outW * outC);
    col_buf.resize(outH * outW * kernelH * kernelW * inC);

    // He init
    float scale = std::sqrt(2.f / float(kernelH * kernelW * inC));
    Pcg32 rng(seed);
    for (auto& w : filters) w = rng.next_gaussian() * scale;

    grad_filter_offset = grad_offset;
    grad_bias_offset   = grad_offset + fsize;
}

void RlCnnEncoder::ConvLayer::forward(const float* input,
                                      float* preact_out,
                                      float* output)
{
    std::memcpy(cached_input.data(), input, inH * inW * inC * sizeof(float));

    int M = outH * outW;   // rows in col matrix
    int K = kernelH * kernelW * inC;  // cols in col matrix

    // im2col: input [H,W,C] → col [M, K]
    im2col(input, col_buf.data(), inH, inW, inC,
           outH, outW, kernelH, kernelW, stride);

    // GEMM: output [M, outC] = col [M, K] × filters^T [K, outC]
    // i.e., for each spatial position (row of col), dot with each filter.
    for (int m = 0; m < M; ++m) {
        const float* col_row = col_buf.data() + m * K;
        for (int oc = 0; oc < outC; ++oc) {
            float val = biases[oc] +
                        dot_avx(col_row, filters.data() + oc * K, K);
            int idx = m * outC + oc;
            preact_out[idx] = val;
            output[idx] = val > 0.f ? val : 0.f; // ReLU
        }
    }

    std::memcpy(cached_preact.data(), preact_out,
                outH * outW * outC * sizeof(float));
}

void RlCnnEncoder::ConvLayer::accum_grad(const float* output_grad,
                                          float* filter_grads,
                                          float* bias_grads,
                                          float* input_grad_out)
{
    int M = outH * outW;
    int K = kernelH * kernelW * inC;

    // Re-run im2col on cached input (col_buf still holds it from forward)
    // but we need to be safe in case forward was called since — it's still valid
    // because we only call accum_grad before the next forward.

    // col_grad [M, K]: the gradient w.r.t. each im2col patch
    std::vector<float> col_grad(M * K, 0.f);

    for (int m = 0; m < M; ++m) {
        for (int oc = 0; oc < outC; ++oc) {
            int idx      = m * outC + oc;
            // ReLU backward gate
            float relu_g = cached_preact[idx] > 0.f ? output_grad[idx] : 0.f;

            bias_grads[oc] += relu_g;

            const float* col_row  = col_buf.data() + m * K;
            float*       cg_row   = col_grad.data() + m * K;
            float*       fg_row   = filter_grads + oc * K;

            // Filter grad: outer product accumulation
            axpy(fg_row, col_row, relu_g, K);

            // col grad: filter row scaled by relu_g
            axpy(cg_row, filters.data() + oc * K, relu_g, K);
        }
    }

    // col2im: scatter col_grad back into input_grad_out
    col2im(col_grad.data(), input_grad_out,
           inH, inW, inC, outH, outW, kernelH, kernelW, stride);
}

void RlCnnEncoder::ConvLayer::apply_grad(const float* filter_grads,
                                          const float* bias_grads,
                                          float lr, float scale)
{
    b1t *= kB1; b2t *= kB2;
    float lr_c = lr * std::sqrt(1.f - b2t) / (1.f - b1t);

    adam_update(filters.data(), wm.data(), wv.data(), filter_grads,
                (int)filters.size(), lr_c, scale);
    adam_update(biases.data(), bm.data(), bv.data(), bias_grads,
                (int)biases.size(), lr_c, scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearLayer
// ─────────────────────────────────────────────────────────────────────────────

void RlCnnEncoder::LinearLayer::init(int inSize_, int outSize_,
                                     uint64_t seed, int grad_offset)
{
    inSize  = inSize_;
    outSize = outSize_;

    int wsize = outSize * inSize;
    weights.assign(wsize, 0.f);
    biases.assign(outSize, 0.f);
    wm.assign(wsize, 0.f); wv.assign(wsize, 0.f);
    bm.assign(outSize, 0.f); bv.assign(outSize, 0.f);
    cached_input.resize(inSize);

    float scale = std::sqrt(2.f / float(inSize));
    Pcg32 rng(seed);
    for (auto& w : weights) w = rng.next_gaussian() * scale;

    grad_weight_offset = grad_offset;
    grad_bias_offset   = grad_offset + wsize;
}

void RlCnnEncoder::LinearLayer::forward(const float* input, float* output)
{
    std::memcpy(cached_input.data(), input, inSize * sizeof(float));
    for (int o = 0; o < outSize; ++o) {
        float val = biases[o] +
                    dot_avx(input, weights.data() + o * inSize, inSize);
        output[o] = val > 0.f ? val : 0.f; // ReLU
    }
}

void RlCnnEncoder::LinearLayer::accum_grad(const float* output_grad,
                                            float* weight_grads,
                                            float* bias_grads,
                                            float* input_grad_out)
{
    std::memset(input_grad_out, 0, inSize * sizeof(float));
    for (int o = 0; o < outSize; ++o) {
        float g = output_grad[o]; // no ReLU gate — projection is the last layer
        bias_grads[o] += g;
        axpy(weight_grads + o * inSize, cached_input.data(), g, inSize);
        axpy(input_grad_out, weights.data() + o * inSize, g, inSize);
    }
}

void RlCnnEncoder::LinearLayer::apply_grad(const float* weight_grads,
                                            const float* bias_grads,
                                            float lr, float scale)
{
    b1t *= kB1; b2t *= kB2;
    float lr_c = lr * std::sqrt(1.f - b2t) / (1.f - b1t);
    adam_update(weights.data(), wm.data(), wv.data(), weight_grads,
                (int)weights.size(), lr_c, scale);
    adam_update(biases.data(), bm.data(), bv.data(), bias_grads,
                (int)biases.size(), lr_c, scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// RlCnnEncoder — public API
// ─────────────────────────────────────────────────────────────────────────────

void RlCnnEncoder::initialize(int width, int height, int channels,
                              const PackedInt32Array& filter_counts,
                              const PackedInt32Array& kernel_sizes,
                              const PackedInt32Array& strides,
                              int output_size)
{
    if (width <= 0 || height <= 0 || channels <= 0 || output_size <= 0) {
        UtilityFunctions::push_error("[RlCnnEncoder] width, height, channels, and output_size must be > 0.");
        _ready = false;
        return;
    }

    int n = filter_counts.size();
    if (n == 0 || n != kernel_sizes.size() || n != strides.size()) {
        UtilityFunctions::push_error("[RlCnnEncoder] filter_counts, kernel_sizes, and strides must be non-empty and equal length.");
        _ready = false;
        return;
    }

    _conv.clear();
    _conv.resize(n);
    _grad_buf_size = 0;

    int prevC = channels, prevH = height, prevW = width;
    for (int i = 0; i < n; ++i) {
        int oc = filter_counts[i];
        int k  = kernel_sizes[i];
        int s  = strides[i];
        if (oc <= 0 || k <= 0 || s <= 0) {
            UtilityFunctions::push_error("[RlCnnEncoder] conv layer parameters must be > 0.");
            _ready = false;
            _conv.clear();
            return;
        }
        if (prevH < k || prevW < k) {
            UtilityFunctions::push_error("[RlCnnEncoder] conv kernel larger than current feature map.");
            _ready = false;
            _conv.clear();
            return;
        }

        _conv[i].init(prevH, prevW, prevC, oc, k, s,
                      /*seed=*/uint64_t(42 + i * 137),
                      _grad_buf_size);
        if (_conv[i].outH <= 0 || _conv[i].outW <= 0) {
            UtilityFunctions::push_error("[RlCnnEncoder] conv layer produced invalid output shape.");
            _ready = false;
            _conv.clear();
            return;
        }

        _grad_buf_size += oc * k * k * prevC + oc; // filterGrads + biasGrads
        prevH = _conv[i].outH;
        prevW = _conv[i].outW;
        prevC = oc;
    }

    int flat = prevH * prevW * prevC;
    _proj.init(flat, output_size,
               /*seed=*/uint64_t(42 + n * 137),
               _grad_buf_size);
    _grad_buf_size += output_size * flat + output_size; // weightGrads + biasGrads

    _ready = true;
}

PackedFloat32Array RlCnnEncoder::forward(const PackedFloat32Array& input)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlCnnEncoder] forward called before initialize.");
        return PackedFloat32Array();
    }
    if (_conv.empty()) {
        UtilityFunctions::push_error("[RlCnnEncoder] forward called with no conv layers configured.");
        return PackedFloat32Array();
    }
    const int expected_input = _conv.front().inH * _conv.front().inW * _conv.front().inC;
    if (input.size() != expected_input) {
        UtilityFunctions::push_error("[RlCnnEncoder] forward input size mismatch.");
        return PackedFloat32Array();
    }

    // Bounce between two temp buffers as activations flow through layers.
    std::vector<float> buf_a(input.ptr(), input.ptr() + input.size());
    std::vector<float> buf_b;
    std::vector<float> preact_buf;

    for (auto& layer : _conv) {
        int out_n = layer.outH * layer.outW * layer.outC;
        buf_b.resize(out_n);
        preact_buf.resize(out_n);
        layer.forward(buf_a.data(), preact_buf.data(), buf_b.data());
        std::swap(buf_a, buf_b);
    }

    // Projection
    buf_b.resize(_proj.outSize);
    _proj.forward(buf_a.data(), buf_b.data());

    PackedFloat32Array result;
    result.resize(_proj.outSize);
    std::memcpy(result.ptrw(), buf_b.data(), _proj.outSize * sizeof(float));
    return result;
}

PackedFloat32Array RlCnnEncoder::create_gradient_buffer() const
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlCnnEncoder] create_gradient_buffer called before initialize.");
        return PackedFloat32Array();
    }

    PackedFloat32Array buf;
    buf.resize(_grad_buf_size);
    std::memset(buf.ptrw(), 0, _grad_buf_size * sizeof(float));
    return buf;
}

PackedFloat32Array RlCnnEncoder::accumulate_gradients_impl(
    const PackedFloat32Array& output_grad,
    PackedFloat32Array&       grad_buffer)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlCnnEncoder] accumulate_gradients called before initialize.");
        return PackedFloat32Array();
    }
    if (output_grad.size() != _proj.outSize) {
        UtilityFunctions::push_error("[RlCnnEncoder] accumulate_gradients output_grad size mismatch.");
        return PackedFloat32Array();
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlCnnEncoder] accumulate_gradients grad_buffer size mismatch.");
        return PackedFloat32Array();
    }

    float*       gb   = grad_buffer.ptrw();
    const float* grad = output_grad.ptr();

    // Projection backward
    int flat_in = _proj.inSize;
    std::vector<float> proj_input_grad(flat_in);
    _proj.accum_grad(grad,
                     gb + _proj.grad_weight_offset,
                     gb + _proj.grad_bias_offset,
                     proj_input_grad.data());

    // Conv layers backward (reverse order)
    std::vector<float> cur_grad = std::move(proj_input_grad);
    for (int i = (int)_conv.size() - 1; i >= 0; --i) {
        auto& layer = _conv[i];
        int in_n    = layer.inH * layer.inW * layer.inC;
        std::vector<float> next_grad(in_n);
        layer.accum_grad(cur_grad.data(),
                         gb + layer.grad_filter_offset,
                         gb + layer.grad_bias_offset,
                         next_grad.data());
        cur_grad = std::move(next_grad);
    }

    // Return pixel-space gradient (rarely used but matches C# interface)
    PackedFloat32Array pixel_grad;
    pixel_grad.resize((int)cur_grad.size());
    std::memcpy(pixel_grad.ptrw(), cur_grad.data(),
                cur_grad.size() * sizeof(float));
    return pixel_grad;
}

PackedFloat32Array RlCnnEncoder::accumulate_gradients(
    const PackedFloat32Array& output_grad,
    PackedFloat32Array        grad_buffer)
{
    return accumulate_gradients_impl(output_grad, grad_buffer);
}

Array RlCnnEncoder::accumulate_gradients_with_buffer(
    const PackedFloat32Array& output_grad,
    PackedFloat32Array        grad_buffer)
{
    PackedFloat32Array input_grad = accumulate_gradients_impl(output_grad, grad_buffer);
    Array result;
    result.resize(2);
    result[0] = input_grad;
    result[1] = grad_buffer;
    return result;
}

void RlCnnEncoder::apply_gradients(const PackedFloat32Array& grad_buffer,
                                   float lr, float grad_scale)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlCnnEncoder] apply_gradients called before initialize.");
        return;
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlCnnEncoder] apply_gradients grad_buffer size mismatch.");
        return;
    }

    const float* gb = grad_buffer.ptr();

    for (auto& layer : _conv)
        layer.apply_grad(gb + layer.grad_filter_offset,
                         gb + layer.grad_bias_offset,
                         lr, grad_scale);

    _proj.apply_grad(gb + _proj.grad_weight_offset,
                     gb + _proj.grad_bias_offset,
                     lr, grad_scale);
}

float RlCnnEncoder::grad_norm_squared(const PackedFloat32Array& grad_buffer) const
{
    if (!_ready || grad_buffer.size() != _grad_buf_size)
        return 0.f;

    const float* gb  = grad_buffer.ptr();
    float        sum = 0.f;

    for (const auto& layer : _conv) {
        int fsize = (int)layer.filters.size();
        for (int i = 0; i < fsize;         ++i) sum += gb[layer.grad_filter_offset + i] * gb[layer.grad_filter_offset + i];
        for (int i = 0; i < layer.outC;    ++i) sum += gb[layer.grad_bias_offset   + i] * gb[layer.grad_bias_offset   + i];
    }
    int wsize = (int)_proj.weights.size();
    for (int i = 0; i < wsize;        ++i) sum += gb[_proj.grad_weight_offset + i] * gb[_proj.grad_weight_offset + i];
    for (int i = 0; i < _proj.outSize; ++i) sum += gb[_proj.grad_bias_offset  + i] * gb[_proj.grad_bias_offset  + i];

    return sum;
}

PackedFloat32Array RlCnnEncoder::get_weights() const
{
    if (!_ready)
        return PackedFloat32Array();

    int total = 0;
    for (const auto& l : _conv) total += (int)(l.filters.size() + l.biases.size());
    total += (int)(_proj.weights.size() + _proj.biases.size());

    PackedFloat32Array out;
    out.resize(total);
    float* dst = out.ptrw();

    for (const auto& l : _conv) {
        std::memcpy(dst, l.filters.data(), l.filters.size() * sizeof(float));
        dst += l.filters.size();
        std::memcpy(dst, l.biases.data(), l.biases.size() * sizeof(float));
        dst += l.biases.size();
    }
    std::memcpy(dst, _proj.weights.data(), _proj.weights.size() * sizeof(float));
    dst += _proj.weights.size();
    std::memcpy(dst, _proj.biases.data(), _proj.biases.size() * sizeof(float));
    return out;
}

PackedInt32Array RlCnnEncoder::get_shapes() const
{
    if (!_ready)
        return PackedInt32Array();

    // Mirrors C# AppendSerialized shape descriptor:
    // [num_conv_layers, outC, kH, kW, inC, stride, ... (per conv), inSize, outSize]
    PackedInt32Array out;
    out.push_back((int)_conv.size());
    for (const auto& l : _conv) {
        out.push_back(l.outC);
        out.push_back(l.kernelH);
        out.push_back(l.kernelW);
        out.push_back(l.inC);
        out.push_back(l.stride);
    }
    out.push_back(_proj.inSize);
    out.push_back(_proj.outSize);
    return out;
}

void RlCnnEncoder::set_weights(const PackedFloat32Array& weights,
                               const PackedInt32Array&   /*shapes*/)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlCnnEncoder] set_weights called before initialize.");
        return;
    }

    int expected = 0;
    for (const auto& l : _conv) expected += (int)(l.filters.size() + l.biases.size());
    expected += (int)(_proj.weights.size() + _proj.biases.size());
    if (weights.size() != expected) {
        UtilityFunctions::push_error("[RlCnnEncoder] set_weights weight array size mismatch.");
        return;
    }

    const float* src = weights.ptr();

    for (auto& l : _conv) {
        std::memcpy(l.filters.data(), src, l.filters.size() * sizeof(float));
        src += l.filters.size();
        std::memcpy(l.biases.data(), src, l.biases.size() * sizeof(float));
        src += l.biases.size();
    }
    std::memcpy(_proj.weights.data(), src, _proj.weights.size() * sizeof(float));
    src += _proj.weights.size();
    std::memcpy(_proj.biases.data(), src, _proj.biases.size() * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────────────────
// GDExtension binding
// ─────────────────────────────────────────────────────────────────────────────

void RlCnnEncoder::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("initialize",
        "width", "height", "channels",
        "filter_counts", "kernel_sizes", "strides", "output_size"),
        &RlCnnEncoder::initialize);

    ClassDB::bind_method(D_METHOD("forward", "input"),
        &RlCnnEncoder::forward);

    ClassDB::bind_method(D_METHOD("create_gradient_buffer"),
        &RlCnnEncoder::create_gradient_buffer);

    ClassDB::bind_method(D_METHOD("accumulate_gradients", "output_grad", "grad_buffer"),
        &RlCnnEncoder::accumulate_gradients);

    ClassDB::bind_method(D_METHOD("accumulate_gradients_with_buffer", "output_grad", "grad_buffer"),
        &RlCnnEncoder::accumulate_gradients_with_buffer);

    ClassDB::bind_method(D_METHOD("apply_gradients", "grad_buffer", "lr", "grad_scale"),
        &RlCnnEncoder::apply_gradients);

    ClassDB::bind_method(D_METHOD("grad_norm_squared", "grad_buffer"),
        &RlCnnEncoder::grad_norm_squared);

    ClassDB::bind_method(D_METHOD("get_weights"),
        &RlCnnEncoder::get_weights);

    ClassDB::bind_method(D_METHOD("get_shapes"),
        &RlCnnEncoder::get_shapes);

    ClassDB::bind_method(D_METHOD("set_weights", "weights", "shapes"),
        &RlCnnEncoder::set_weights);
}
