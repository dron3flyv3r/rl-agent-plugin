#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <cstdint>
#include <vector>

/// Native GRU layer exposed as a Godot RefCounted object.
///
/// GRU equations:
///   r = sigmoid(Wr·x + Ur·h_prev + br)          [reset gate]
///   z = sigmoid(Wz·x + Uz·h_prev + bz)          [update gate]
///   n = tanh(Wn·x + Un·(r ⊙ h_prev) + bn)       [new/candidate gate]
///   h' = (1 - z) ⊙ h_prev + z ⊙ n
///
/// Weight layout (get_weights / set_weights):
///   [Wr | Wz | Wn | Ur | Uz | Un | br | bz | bn]
///   Wr,Wz,Wn  — input weights,     each [hidden × input],   row-major
///   Ur,Uz,Un  — recurrent weights, each [hidden × hidden],  row-major
///   br,bz,bn  — biases,            each [hidden]
///
/// get_shapes() returns [RLLayerKind::Gru=5, input_size, hidden_size].
///
/// Gradient buffer layout:
///   [Wr_grads | Wz_grads | Wn_grads |
///    Ur_grads | Uz_grads | Un_grads |
///    br_grads | bz_grads | bn_grads ]
///
/// Single-step inference:
///   forward(input, h_prev) -> h_next  (hidden floats)
///
/// Training (BPTT):
///   forward_sequence(flat_inputs, T, h0) -> [T * hidden]
///   accumulate_sequence_gradients(seq_h_grads, grad_buf, h0)
///     -> Array[input_seq_grads (T*input), grad_buf_updated]
///
class RlGruLayer : public godot::RefCounted {
    GDCLASS(RlGruLayer, godot::RefCounted)

public:
    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void initialize(int input_size, int hidden_size, int optimizer);

    // ── Single-step inference ─────────────────────────────────────────────────
    /// Returns h_next (hidden_size floats).
    godot::PackedFloat32Array forward(const godot::PackedFloat32Array& input,
                                      const godot::PackedFloat32Array& h_prev);

    // ── Training: sequence forward ────────────────────────────────────────────
    /// flat_inputs: [T * input_size], h0: [hidden_size]
    /// Returns [T * hidden_size].
    godot::PackedFloat32Array forward_sequence(const godot::PackedFloat32Array& flat_inputs,
                                               int                        seq_len,
                                               const godot::PackedFloat32Array& h0);

    // ── Training: BPTT ────────────────────────────────────────────────────────
    /// Returns Array of 2 elements:
    ///   [0] PackedFloat32Array — input gradients [T * input_size]
    ///   [1] PackedFloat32Array — updated grad_buffer
    godot::Array accumulate_sequence_gradients(const godot::PackedFloat32Array& seq_h_grads,
                                               int                        seq_len,
                                               godot::PackedFloat32Array  grad_buffer,
                                               const godot::PackedFloat32Array& h0);

    void apply_gradients(const godot::PackedFloat32Array& grad_buffer,
                         float lr, float grad_scale, float grad_clip_norm);

    godot::PackedFloat32Array create_gradient_buffer() const;
    float              grad_norm_squared(const godot::PackedFloat32Array& grad_buffer) const;

    // ── Target-network copy ────────────────────────────────────────────────────
    void copy_weights_from(godot::Object* source);
    void soft_update_from(godot::Object* source, float tau);

    // ── Serialization ──────────────────────────────────────────────────────────
    godot::PackedFloat32Array get_weights() const;
    godot::PackedInt32Array   get_shapes()  const;
    void                      set_weights(const godot::PackedFloat32Array& weights,
                                          const godot::PackedInt32Array&   shapes);

    // ── Accessors ──────────────────────────────────────────────────────────────
    int get_input_size()  const { return _in; }
    int get_hidden_size() const { return _h;  }

protected:
    static void _bind_methods();

private:
    static constexpr int NUM_GATES = 3;  // r, z, n

    // Parameters
    std::vector<float> _w;   // [3 * _h * _in]   gate order: r=0, z=1, n=2
    std::vector<float> _u;   // [3 * _h * _h]
    std::vector<float> _b;   // [3 * _h]

    // Adam moments
    std::vector<float> _wm, _wv;
    std::vector<float> _um, _uv;
    std::vector<float> _bm, _bv;
    float _b1t = 1.f, _b2t = 1.f;

    // Sequence cache
    int _seq_len = 0;
    // _seq_x[t * _in + k]
    // _seq_rz[t * 2 * _h + gate * _h + j]   post-activation r,z (gate: 0=r, 1=z)
    // _seq_n[t * _h + j]                     post-activation n (tanh)
    // _seq_rh[t * _h + j]                    r ⊙ h_prev at step t
    // _seq_h[(t+1) * _h + j]                 h[t+1]; _seq_h[0.._h) = h0
    std::vector<float> _seq_x;
    std::vector<float> _seq_rz;
    std::vector<float> _seq_n;
    std::vector<float> _seq_rh;
    std::vector<float> _seq_h;

    int  _in = 0, _h = 0;
    int  _optimizer  = 0;
    int  _grad_buf_size = 0;
    bool _ready = false;

    static void scale_buffer(float* buf, int n, float scale);
};
