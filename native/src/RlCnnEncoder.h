#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <cstdint>
#include <vector>

using namespace godot;

/// <summary>
/// Native CNN encoder exposed as a Godot RefCounted object.
/// Implements the same logical interface as the C# CnnEncoder:
///   Forward / AccumulateGradients / ApplyGradients / Serialization / CopyWeights
///
/// Gradient buffers are passed as flat PackedFloat32Array so no heap allocation
/// happens on the C# side — the array is just handed back each call.
///
/// Memory layout of a gradient buffer (returned by create_gradient_buffer):
///   [ conv0.filterGrads | conv0.biasGrads |
///     conv1.filterGrads | conv1.biasGrads | ...
///     proj.weightGrads  | proj.biasGrads  ]
/// </summary>
class RlCnnEncoder : public RefCounted {
    GDCLASS(RlCnnEncoder, RefCounted)

public:
    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Must be called once before any other method.
    /// Matches CnnEncoder(int width, int height, int channels, RLCnnEncoderDef def).
    void initialize(int width, int height, int channels,
                    const PackedInt32Array& filter_counts,
                    const PackedInt32Array& kernel_sizes,
                    const PackedInt32Array& strides,
                    int output_size);

    // ── Forward / backward ───────────────────────────────────────────────────

    PackedFloat32Array forward(const PackedFloat32Array& input);

    /// Returns a zero-initialised gradient buffer sized for this encoder.
    PackedFloat32Array create_gradient_buffer() const;

    /// Accumulates gradients into grad_buffer given the loss gradient w.r.t.
    /// the encoder output.  Returns the gradient w.r.t. the pixel input.
    PackedFloat32Array accumulate_gradients(const PackedFloat32Array& output_grad,
                                            PackedFloat32Array        grad_buffer);

    /// Returns [input_grad, updated_grad_buffer].
    Array accumulate_gradients_with_buffer(const PackedFloat32Array& output_grad,
                                           PackedFloat32Array        grad_buffer);

    /// Applies one Adam step using the accumulated grad_buffer, then the
    /// caller should zero the buffer before the next mini-batch.
    void apply_gradients(const PackedFloat32Array& grad_buffer,
                         float lr, float grad_scale);

    // ── Gradient clipping support ─────────────────────────────────────────────

    float grad_norm_squared(const PackedFloat32Array& grad_buffer) const;

    // ── Serialization ────────────────────────────────────────────────────────

    /// Returns all weights as a flat array (same order as C# AppendSerialized).
    PackedFloat32Array get_weights() const;

    /// Returns shape descriptors (same order as C# AppendSerialized).
    PackedInt32Array get_shapes() const;

    /// Restores weights from arrays previously returned by get_weights / get_shapes.
    void set_weights(const PackedFloat32Array& weights,
                     const PackedInt32Array&   shapes);

protected:
    static void _bind_methods();

private:
    PackedFloat32Array accumulate_gradients_impl(const PackedFloat32Array& output_grad,
                                                 PackedFloat32Array&       grad_buffer);

    // ── Internal layer structs ────────────────────────────────────────────────

    struct ConvLayer {
        int inH, inW, inC;
        int outC, kernelH, kernelW, stride;
        int outH, outW;

        std::vector<float> filters; // layout: [outC, kernelH, kernelW, inC]
        std::vector<float> biases;  // [outC]

        // Adam moments
        std::vector<float> wm, wv, bm, bv;
        float b1t = 1.f, b2t = 1.f;

        // Forward cache (for backprop)
        std::vector<float> cached_input;
        std::vector<float> cached_preact;
        std::vector<float> col_buf; // im2col scratch, reused across calls

        // Offsets into the flat gradient buffer
        int grad_filter_offset = 0;
        int grad_bias_offset   = 0;

        void init(int inH, int inW, int inC,
                  int outC, int kernel, int stride,
                  uint64_t seed, int grad_offset);

        void forward(const float* input, float* preact_out, float* output);

        void accum_grad(const float* output_grad,
                        float* filter_grads, float* bias_grads,
                        float* input_grad_out);

        void apply_grad(const float* filter_grads, const float* bias_grads,
                        float lr, float scale);
    };

    struct LinearLayer {
        int inSize, outSize;

        std::vector<float> weights; // [outSize, inSize]
        std::vector<float> biases;  // [outSize]

        std::vector<float> wm, wv, bm, bv;
        float b1t = 1.f, b2t = 1.f;

        std::vector<float> cached_input;

        int grad_weight_offset = 0;
        int grad_bias_offset   = 0;

        void init(int inSize, int outSize, uint64_t seed, int grad_offset);

        void forward(const float* input, float* output);

        void accum_grad(const float* output_grad,
                        float* weight_grads, float* bias_grads,
                        float* input_grad_out);

        void apply_grad(const float* weight_grads, const float* bias_grads,
                        float lr, float scale);
    };

    // ── Members ───────────────────────────────────────────────────────────────

    std::vector<ConvLayer> _conv;
    LinearLayer            _proj;
    int                    _grad_buf_size = 0;
    bool                   _ready         = false;
};
