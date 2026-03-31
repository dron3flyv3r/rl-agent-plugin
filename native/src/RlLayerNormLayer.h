#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <cstdint>
#include <vector>

using namespace godot;

/// Native layer normalisation layer exposed as a Godot RefCounted object.
///
/// Normalises across the feature dimension, then applies learned affine
/// transform: y_i = gamma_i * normalised_i + beta_i.
///
/// Matches LayerNormLayer.cs exactly:
///   - Gamma initialised to 1, beta to 0.
///   - Plain SGD for gamma/beta (no Adam moments).
///   - Full backward: standard LayerNorm gradient formula.
///
/// Gradient buffer layout: [gamma_grads[size] | beta_grads[size]]
class RlLayerNormLayer : public RefCounted {
    GDCLASS(RlLayerNormLayer, RefCounted)

public:
    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void initialize(int size);

    // ── Forward ───────────────────────────────────────────────────────────────
    PackedFloat32Array forward(const PackedFloat32Array& input);
    PackedFloat32Array forward_batch(const PackedFloat32Array& flat_input, int batch_size);

    // ── Single-sample backward (immediate SGD update) ─────────────────────────
    PackedFloat32Array backward(const PackedFloat32Array& output_grad,
                                float lr, float grad_scale);

    // ── Batch accumulate + apply ───────────────────────────────────────────────
    PackedFloat32Array create_gradient_buffer() const;
    PackedFloat32Array accumulate_gradients(const PackedFloat32Array& output_grad,
                                            PackedFloat32Array        grad_buffer);
    /// Returns [input_grad, updated_grad_buffer].
    Array              accumulate_gradients_with_buffer(const PackedFloat32Array& output_grad,
                                                        PackedFloat32Array        grad_buffer);
    void               apply_gradients(const PackedFloat32Array& grad_buffer,
                                       float lr, float grad_scale);

    // ── Frozen gradient ────────────────────────────────────────────────────────
    PackedFloat32Array compute_input_grad(const PackedFloat32Array& output_grad);

    // ── Gradient clipping support ──────────────────────────────────────────────
    float grad_norm_squared(const PackedFloat32Array& grad_buffer) const;

    // ── Target-network copy ────────────────────────────────────────────────────
    void copy_weights_from(Object* source);
    void soft_update_from(Object* source, float tau);

    // ── Serialization ──────────────────────────────────────────────────────────
    /// Returns [gamma..., beta...] (same order as LayerNormLayer.AppendSerialized).
    PackedFloat32Array get_weights() const;

    /// Returns [RLLayerKind::LayerNorm=2, size].
    PackedInt32Array   get_shapes()  const;

    /// Restores gamma/beta weights.
    void set_weights(const PackedFloat32Array& weights,
                     const PackedInt32Array&   shapes);

protected:
    static void _bind_methods();

private:
    std::vector<float> _gamma;            // [_size], learned scale (init to 1)
    std::vector<float> _beta;             // [_size], learned shift (init to 0)
    std::vector<float> _last_normalized;  // [_size] — set by forward(), used by backward
    float _last_std = 1.f;               // scalar std from most recent forward()
    int   _size     = 0;
    int   _grad_buf_size = 0;             // 2 * _size
    bool  _ready = false;

    static constexpr float kEpsLN = 1e-5f;  // LayerNorm epsilon (matches C# Eps = 1e-5f)

    // ── Private helper ─────────────────────────────────────────────────────────
    /// Computes all three gradient arrays from the cached forward state.
    /// Returns input gradient; gamma_grad and beta_grad written into caller buffers.
    void compute_gradients(const float* output_grad,
                           float* input_grad,
                           float* gamma_grad,
                           float* beta_grad) const;

    /// Shared implementation used by both accumulate APIs.
    PackedFloat32Array accumulate_gradients_impl(const PackedFloat32Array& output_grad,
                                                 PackedFloat32Array&       grad_buffer);
};
