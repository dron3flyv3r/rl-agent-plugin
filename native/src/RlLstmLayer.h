#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <cstdint>
#include <vector>

/// Native LSTM layer exposed as a Godot RefCounted object.
///
/// Weight layout (get_weights / set_weights):
///   [Wi | Wf | Wg | Wo | Ui | Uf | Ug | Uo | bi | bf | bg | bo]
///   Wi,Wf,Wg,Wo  — input weights,     each [hidden × input],   row-major
///   Ui,Uf,Ug,Uo  — recurrent weights, each [hidden × hidden],  row-major
///   bi,bf,bg,bo  — biases,            each [hidden]
///   Gate order: i=input, f=forget, g=cell (tanh), o=output
///
/// get_shapes() returns [RLLayerKind::Lstm=4, input_size, hidden_size].
///
/// Optimizer encoding: 0=Adam, 1=SGD, -1=None (frozen)
///
/// Gradient buffer layout:
///   [Wi_grads | Wf_grads | Wg_grads | Wo_grads |
///    Ui_grads | Uf_grads | Ug_grads | Uo_grads |
///    bi_grads | bf_grads | bg_grads | bo_grads ]
///
/// Single-step inference:
///   forward(input, h_prev, c_prev) -> [h_next | c_next]  (2 * hidden)
///
/// Training (BPTT):
///   forward_sequence(flat_inputs, T, h0, c0) -> [T * hidden] (h at each step)
///   accumulate_sequence_gradients(seq_h_grads, grad_buf, h0, c0)
///     -> Array[input_seq_grads (T*input), grad_buf_updated]
///
class RlLstmLayer : public godot::RefCounted {
    GDCLASS(RlLstmLayer, godot::RefCounted)

public:
    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void initialize(int input_size, int hidden_size, int optimizer);

    // ── Single-step inference ─────────────────────────────────────────────────
    /// Returns [h_next | c_next] (2 * hidden_size floats).
    /// Does NOT cache state — caller is responsible for threading state across steps.
    godot::PackedFloat32Array forward(const godot::PackedFloat32Array& input,
                                      const godot::PackedFloat32Array& h_prev,
                                      const godot::PackedFloat32Array& c_prev);

    // ── Training: sequence forward (caches full sequence for BPTT) ───────────
    /// flat_inputs: [T * input_size], h0/c0: [hidden_size]
    /// Returns [T * hidden_size] (h output at each step).
    /// Caches full sequence internally; call accumulate_sequence_gradients after.
    godot::PackedFloat32Array forward_sequence(const godot::PackedFloat32Array& flat_inputs,
                                               int                        seq_len,
                                               const godot::PackedFloat32Array& h0,
                                               const godot::PackedFloat32Array& c0);

    // ── Training: BPTT ────────────────────────────────────────────────────────
    /// seq_h_grads: [T * hidden_size] — gradient of loss w.r.t. h at each step.
    /// grad_buffer: gradient accumulator (see layout in class doc above).
    /// h0, c0: initial states passed to the matching forward_sequence call.
    ///
    /// Returns Array of 2 elements:
    ///   [0] PackedFloat32Array — input gradients [T * input_size]
    ///   [1] PackedFloat32Array — updated grad_buffer (Variant copy semantics)
    godot::Array accumulate_sequence_gradients(const godot::PackedFloat32Array& seq_h_grads,
                                               godot::PackedFloat32Array        grad_buffer,
                                               const godot::PackedFloat32Array& h0,
                                               const godot::PackedFloat32Array& c0);

    /// Applies accumulated gradients (with Adam or SGD) and optional grad-norm clipping.
    /// grad_clip_norm <= 0 disables clipping.
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
    // ── Parameters ────────────────────────────────────────────────────────────
    // Input weights:     _w[gate * _h * _in  + j * _in  + k]  (gate in 0..3, j in 0.._h-1)
    // Recurrent weights: _u[gate * _h * _h   + j * _h   + k]
    // Biases:            _b[gate * _h         + j]
    std::vector<float> _w;   // [4 * _h * _in]
    std::vector<float> _u;   // [4 * _h * _h]
    std::vector<float> _b;   // [4 * _h]

    // Adam moments
    std::vector<float> _wm, _wv;  // weight moments
    std::vector<float> _um, _uv;  // recurrent moments
    std::vector<float> _bm, _bv;  // bias moments
    float _b1t = 1.f, _b2t = 1.f;

    // ── Sequence cache (set by forward_sequence, read by BPTT) ───────────────
    int _seq_len = 0;
    // _seq_x[t * _in  + k]:              input at step t
    // _seq_gates[t * 4*_h + gate*_h + j]: post-activation gate value at step t
    // _seq_h[(t+1) * _h + j]:            h[t+1]; _seq_h[j] = h0
    // _seq_c[(t+1) * _h + j]:            c[t+1]; _seq_c[j] = c0
    // _seq_tanh_c[t * _h + j]:           tanh(c[t+1]) at step t
    std::vector<float> _seq_x;
    std::vector<float> _seq_gates;
    std::vector<float> _seq_h;
    std::vector<float> _seq_c;
    std::vector<float> _seq_tanh_c;

    // Dimensions
    int  _in = 0, _h = 0;
    int  _optimizer  = 0;
    int  _grad_buf_size = 0;
    bool _ready = false;

    // ── Helpers ────────────────────────────────────────────────────────────────
    /// Compute one LSTM step. Writes h_next into dst_h[0.._h), c_next into dst_c.
    /// gates_out: optional [4*_h] buffer to receive post-activation gate values.
    /// tanh_c_out: optional [_h] buffer to receive tanh(c_next).
    void step(const float* x,
              const float* h_prev, const float* c_prev,
              float* dst_h, float* dst_c,
              float* gates_out,   // may be nullptr
              float* tanh_c_out); // may be nullptr

    /// Scale all values in grad_buffer by `scale` (used for gradient clipping).
    static void scale_buffer(float* buf, int n, float scale);
};
