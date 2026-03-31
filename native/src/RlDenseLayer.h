#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <cstdint>
#include <vector>

using namespace godot;

/// Native fully-connected (dense) layer exposed as a Godot RefCounted object.
///
/// Implements the same logical interface as the C# DenseLayer:
///   forward / forward_batch / backward /
///   create_gradient_buffer / accumulate_gradients / apply_gradients /
///   compute_input_grad / grad_norm_squared /
///   copy_weights_from / soft_update_from /
///   get_weights / get_shapes / set_weights
///
/// Activation encoding (matches DenseLayer.AppendSerialized):
///   0 = None (linear),  1 = Tanh,  2 = ReLU
///
/// Optimizer encoding (matches RLOptimizerKind):
///   0 = Adam,  1 = SGD,  -1 = None (frozen)
///
/// Gradient buffer layout: [weight_grads[out*in] | bias_grads[out]]
///
/// Adam uses the exact bias-corrected formula from DenseLayer.cs so that
/// checkpoints and unit-test outputs are numerically identical.
class RlDenseLayer : public RefCounted {
    GDCLASS(RlDenseLayer, RefCounted)

public:
    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void initialize(int in_size, int out_size, int activation, int optimizer);

    // ── Forward ───────────────────────────────────────────────────────────────
    /// Single-sample forward. Caches input and pre-activation for backward.
    PackedFloat32Array forward(const PackedFloat32Array& input);

    /// Batch forward (no gradient caching). flat_input is [batch_size * in_size].
    PackedFloat32Array forward_batch(const PackedFloat32Array& flat_input, int batch_size);

    // ── Single-sample backward (immediate weight update) ──────────────────────
    PackedFloat32Array backward(const PackedFloat32Array& output_grad,
                                float lr, float grad_scale);

    // ── Batch accumulate + apply ───────────────────────────────────────────────
    PackedFloat32Array create_gradient_buffer() const;

    /// Accumulates gradients into grad_buffer (mutated in place) and returns
    /// the input-space gradient. Uses state cached by the most recent forward().
    PackedFloat32Array accumulate_gradients(const PackedFloat32Array& output_grad,
                                            PackedFloat32Array        grad_buffer);

    /// Returns [input_grad, updated_grad_buffer].
    /// Use this API from managed code so buffer updates survive Variant marshaling.
    Array accumulate_gradients_with_buffer(const PackedFloat32Array& output_grad,
                                           PackedFloat32Array        grad_buffer);

    void apply_gradients(const PackedFloat32Array& grad_buffer,
                         float lr, float grad_scale);

    // ── Frozen gradient (SAC dQ/da) ────────────────────────────────────────────
    PackedFloat32Array compute_input_grad(const PackedFloat32Array& output_grad);

    // ── Gradient clipping support ──────────────────────────────────────────────
    float grad_norm_squared(const PackedFloat32Array& grad_buffer) const;

    // ── Target-network copy ────────────────────────────────────────────────────
    void copy_weights_from(Object* source);
    void soft_update_from(Object* source, float tau);

    // ── Serialization ──────────────────────────────────────────────────────────
    /// Returns [weights..., biases...] (same order as DenseLayer.AppendSerialized).
    PackedFloat32Array get_weights() const;

    /// Returns [RLLayerKind::Dense=0, in_size, out_size, activation_code].
    PackedInt32Array   get_shapes()  const;

    /// Restores weights; resets Adam moments to zero (same as LoadSerialized).
    void set_weights(const PackedFloat32Array& weights,
                     const PackedInt32Array&   shapes);

protected:
    static void _bind_methods();

private:
    std::vector<float> _weights;          // [_out * _in], row-major
    std::vector<float> _biases;           // [_out]
    std::vector<float> _wm, _wv;          // Adam moments for weights (empty if not Adam)
    std::vector<float> _bm, _bv;          // Adam moments for biases  (empty if not Adam)
    float _b1t = 1.f, _b2t = 1.f;        // bias-correction accumulators (beta^t)
    std::vector<float> _cached_input;     // [_in]  — set by forward(), used by backward
    std::vector<float> _cached_preact;    // [_out] — set by forward(), used by backward
    int  _in  = 0, _out = 0;
    int  _activation = 0;                 // 0=None, 1=Tanh, 2=ReLU
    int  _optimizer  = 0;                 // 0=Adam, 1=SGD, -1=None
    int  _grad_buf_size = 0;              // _out*_in + _out
    bool _ready = false;

    // ── Private helpers ────────────────────────────────────────────────────────
    inline float  apply_act(float x) const noexcept;
    inline float  act_deriv(float preact) const noexcept;

    /// Computes local_grad = output_grad ⊙ act'(preact) into a caller-supplied buffer.
    void compute_local_grad(const float* output_grad, float* local_grad) const;

    /// Returns input gradient given pre-computed local_grad.
    PackedFloat32Array input_grad_from_local(const float* local_grad) const;

    /// Shared implementation used by both accumulate APIs.
    PackedFloat32Array accumulate_gradients_impl(const PackedFloat32Array& output_grad,
                                                 PackedFloat32Array&       grad_buffer);
};
