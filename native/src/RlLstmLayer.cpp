#include "RlLstmLayer.h"
#include "rl_simd.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// Gate indices
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int GATE_I   = 0;   // input gate   (sigmoid)
static constexpr int GATE_F   = 1;   // forget gate  (sigmoid)
static constexpr int GATE_G   = 2;   // cell gate    (tanh)
static constexpr int GATE_O   = 3;   // output gate  (sigmoid)
static constexpr int NUM_GATES = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline float sigmoid(float x) noexcept
{
    return 1.f / (1.f + std::exp(-x));
}

void RlLstmLayer::scale_buffer(float* buf, int n, float scale)
{
    for (int i = 0; i < n; ++i) buf[i] *= scale;
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize
// ─────────────────────────────────────────────────────────────────────────────

void RlLstmLayer::initialize(int input_size, int hidden_size, int optimizer)
{
    if (input_size <= 0 || hidden_size <= 0) {
        UtilityFunctions::push_error("[RlLstmLayer] initialize: input_size and hidden_size must be > 0.");
        return;
    }

    _in        = input_size;
    _h         = hidden_size;
    _optimizer = optimizer;

    const int wsize = NUM_GATES * _h * _in;
    const int usize = NUM_GATES * _h * _h;
    const int bsize = NUM_GATES * _h;

    _w.assign(wsize, 0.f);
    _u.assign(usize, 0.f);
    _b.assign(bsize, 0.f);

    if (optimizer == 0) {  // Adam
        _wm.assign(wsize, 0.f); _wv.assign(wsize, 0.f);
        _um.assign(usize, 0.f); _uv.assign(usize, 0.f);
        _bm.assign(bsize, 0.f); _bv.assign(bsize, 0.f);
    } else {
        _wm.clear(); _wv.clear();
        _um.clear(); _uv.clear();
        _bm.clear(); _bv.clear();
    }
    _b1t = 1.f;
    _b2t = 1.f;

    // He-normal init (same as DenseLayer)
    float w_scale = std::sqrt(2.f / float(std::max(1, input_size)));
    float u_scale = std::sqrt(2.f / float(std::max(1, hidden_size)));

    Pcg32 rng_w(uint64_t(42 + uint64_t(wsize) * 7));
    for (auto& v : _w) v = rng_w.next_gaussian() * w_scale;

    Pcg32 rng_u(uint64_t(137 + uint64_t(usize) * 13));
    for (auto& v : _u) v = rng_u.next_gaussian() * u_scale;

    // Forget gate bias = 1: helps gradient flow early in training
    for (int j = 0; j < _h; ++j)
        _b[GATE_F * _h + j] = 1.f;

    _grad_buf_size = wsize + usize + bsize;
    _ready = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// step — shared single-timestep computation
// ─────────────────────────────────────────────────────────────────────────────

void RlLstmLayer::step(const float* x,
                        const float* h_prev, const float* c_prev,
                        float* dst_h, float* dst_c,
                        float* gates_out,
                        float* tanh_c_out)
{
    // Allocate gate buffer on stack for small hidden sizes, heap otherwise
    std::vector<float> heap_gates;
    float stack_gates[NUM_GATES * 256];
    float* gates;
    if (gates_out != nullptr) {
        gates = gates_out;
    } else if (_h <= 256) {
        gates = stack_gates;
    } else {
        heap_gates.resize(NUM_GATES * _h);
        gates = heap_gates.data();
    }

    for (int g = 0; g < NUM_GATES; ++g) {
        const float* W  = _w.data() + g * _h * _in;
        const float* U  = _u.data() + g * _h * _h;
        const float* B  = _b.data() + g * _h;
        float*       ga = gates + g * _h;

        for (int j = 0; j < _h; ++j) {
            float pre = B[j]
                      + dot_avx(x,      W + j * _in, _in)
                      + dot_avx(h_prev, U + j * _h,  _h);
            ga[j] = (g == GATE_G) ? std::tanh(pre) : sigmoid(pre);
        }
    }

    const float* gi = gates + GATE_I * _h;
    const float* gf = gates + GATE_F * _h;
    const float* gg = gates + GATE_G * _h;
    const float* go = gates + GATE_O * _h;

    // c' = f ⊙ c_prev + i ⊙ g
    for (int j = 0; j < _h; ++j)
        dst_c[j] = gf[j] * c_prev[j] + gi[j] * gg[j];

    // h' = o ⊙ tanh(c')
    for (int j = 0; j < _h; ++j) {
        float tc = std::tanh(dst_c[j]);
        dst_h[j] = go[j] * tc;
        if (tanh_c_out != nullptr) tanh_c_out[j] = tc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// forward — single-step inference
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLstmLayer::forward(const PackedFloat32Array& input,
                                         const PackedFloat32Array& h_prev,
                                         const PackedFloat32Array& c_prev)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] forward called before initialize.");
        return PackedFloat32Array();
    }
    if (input.size() != _in || h_prev.size() != _h || c_prev.size() != _h) {
        UtilityFunctions::push_error("[RlLstmLayer] forward: input / h_prev / c_prev size mismatch.");
        return PackedFloat32Array();
    }

    PackedFloat32Array result;
    result.resize(2 * _h);
    float* dst = result.ptrw();

    step(input.ptr(), h_prev.ptr(), c_prev.ptr(),
         dst,       // h_next [0 .. _h)
         dst + _h,  // c_next [_h .. 2*_h)
         nullptr, nullptr);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_sequence — training forward, caches full sequence for BPTT
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLstmLayer::forward_sequence(const PackedFloat32Array& flat_inputs,
                                                   int                        seq_len,
                                                   const PackedFloat32Array& h0,
                                                   const PackedFloat32Array& c0)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] forward_sequence called before initialize.");
        return PackedFloat32Array();
    }
    if (seq_len <= 0 || flat_inputs.size() != seq_len * _in
        || h0.size() != _h || c0.size() != _h) {
        UtilityFunctions::push_error("[RlLstmLayer] forward_sequence: size mismatch.");
        return PackedFloat32Array();
    }

    const int T = seq_len;
    _seq_len = T;

    _seq_x.resize(T * _in);
    _seq_gates.resize(T * NUM_GATES * _h);
    _seq_h.resize((T + 1) * _h);
    _seq_c.resize((T + 1) * _h);
    _seq_tanh_c.resize(T * _h);

    std::memcpy(_seq_x.data(), flat_inputs.ptr(), T * _in * sizeof(float));
    std::memcpy(_seq_h.data(), h0.ptr(), _h * sizeof(float));  // _seq_h[0] = h0
    std::memcpy(_seq_c.data(), c0.ptr(), _h * sizeof(float));  // _seq_c[0] = c0

    const float* x_ptr = flat_inputs.ptr();
    for (int t = 0; t < T; ++t) {
        step(x_ptr + t * _in,
             _seq_h.data() + t * _h,       // h[t]
             _seq_c.data() + t * _h,       // c[t]
             _seq_h.data() + (t + 1) * _h, // h[t+1]
             _seq_c.data() + (t + 1) * _h, // c[t+1]
             _seq_gates.data() + t * NUM_GATES * _h,
             _seq_tanh_c.data() + t * _h);
    }

    // Return h[1..T]
    PackedFloat32Array result;
    result.resize(T * _h);
    std::memcpy(result.ptrw(), _seq_h.data() + _h, T * _h * sizeof(float));
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// accumulate_sequence_gradients — truncated BPTT
//
// For each timestep t from T-1 downto 0:
//   1. Total dh[t] = external loss grad + recurrent grad from step t+1
//   2. Total dc[t] = recurrent dc from step t+1 + backprop through h'=o⊙tanh(c')
//   3. Compute per-gate pre-act gradients (delta_i, delta_f, delta_g, delta_o)
//   4. Accumulate weight/bias grads and produce input grad at step t
//   5. Propagate dc and dh back to step t-1
// ─────────────────────────────────────────────────────────────────────────────

Array RlLstmLayer::accumulate_sequence_gradients(const PackedFloat32Array& seq_h_grads,
                                                   PackedFloat32Array        grad_buffer,
                                                   const PackedFloat32Array& h0,
                                                   const PackedFloat32Array& c0)
{
    (void)h0; (void)c0;  // cached in _seq_h[0], _seq_c[0] during forward_sequence

    Array error_result;
    error_result.resize(2);
    error_result[0] = PackedFloat32Array();
    error_result[1] = grad_buffer;

    if (!_ready || _seq_len <= 0) {
        UtilityFunctions::push_error("[RlLstmLayer] accumulate_sequence_gradients: forward_sequence not called.");
        return error_result;
    }
    const int T = _seq_len;
    if (seq_h_grads.size() != T * _h || grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlLstmLayer] accumulate_sequence_gradients: size mismatch.");
        return error_result;
    }

    // Gradient buffer sections
    float* gb = grad_buffer.ptrw();
    float* wg = gb;                                // [NUM_GATES * _h * _in]
    float* ug = wg + NUM_GATES * _h * _in;         // [NUM_GATES * _h * _h]
    float* bg = ug + NUM_GATES * _h * _h;          // [NUM_GATES * _h]

    PackedFloat32Array input_grads;
    input_grads.resize(T * _in);
    float* ig_all = input_grads.ptrw();
    std::memset(ig_all, 0, T * _in * sizeof(float));

    const float* ext_dh = seq_h_grads.ptr();

    // Recurrent gradients passed backward through time
    std::vector<float> dh_recur(_h, 0.f);   // recurrent contribution to dh
    std::vector<float> dc_recur(_h, 0.f);   // recurrent contribution to dc

    // Per-gate scratch (delta = pre-act gradient for one gate)
    std::vector<float> delta_i(_h), delta_f(_h), delta_g(_h), delta_o(_h);

    for (int t = T - 1; t >= 0; --t) {
        const float* gates  = _seq_gates.data() + t * NUM_GATES * _h;
        const float* gi     = gates + GATE_I * _h;
        const float* gf     = gates + GATE_F * _h;
        const float* gg     = gates + GATE_G * _h;
        const float* go     = gates + GATE_O * _h;
        const float* tanh_c = _seq_tanh_c.data() + t * _h;
        const float* c_prev = _seq_c.data()       + t * _h;    // c[t]
        const float* h_prev = _seq_h.data()       + t * _h;    // h[t]
        const float* x_t    = _seq_x.data()       + t * _in;
        float*       ig_t   = ig_all              + t * _in;

        // dh_total[t] = external gradient + recurrent from t+1
        // dc_total[t] = recurrent dc from t+1 + backprop through h'=o⊙tanh(c')
        std::vector<float> dh(_h), dc(_h);
        for (int j = 0; j < _h; ++j) {
            dh[j] = ext_dh[t * _h + j] + dh_recur[j];
            dc[j] = dc_recur[j] + dh[j] * go[j] * (1.f - tanh_c[j] * tanh_c[j]);
        }

        // ── Per-gate pre-activation gradients ──────────────────────────────
        // Output gate:  delta_o = dh ⊙ tanh_c ⊙ go ⊙ (1 - go)
        for (int j = 0; j < _h; ++j)
            delta_o[j] = dh[j] * tanh_c[j] * go[j] * (1.f - go[j]);

        // Input gate:   delta_i = dc ⊙ gg ⊙ gi ⊙ (1 - gi)
        for (int j = 0; j < _h; ++j)
            delta_i[j] = dc[j] * gg[j] * gi[j] * (1.f - gi[j]);

        // Forget gate:  delta_f = dc ⊙ c_prev ⊙ gf ⊙ (1 - gf)
        for (int j = 0; j < _h; ++j)
            delta_f[j] = dc[j] * c_prev[j] * gf[j] * (1.f - gf[j]);

        // Cell gate:    delta_g = dc ⊙ gi ⊙ (1 - gg²)
        for (int j = 0; j < _h; ++j)
            delta_g[j] = dc[j] * gi[j] * (1.f - gg[j] * gg[j]);

        // ── Accumulate weight/bias gradients and compute input+recurrent grad ──
        // For each gate k: wg_k += outer(delta_k, x_t)
        //                  ug_k += outer(delta_k, h_prev)
        //                  bg_k += delta_k
        //                  ig_t += W_k^T · delta_k
        //                  new_dh_recur += U_k^T · delta_k
        //
        // Reset recurrent grad accumulators before filling for this step
        std::fill(dh_recur.begin(), dh_recur.end(), 0.f);

        static constexpr int GATE_ORDER[NUM_GATES] = { GATE_I, GATE_F, GATE_G, GATE_O };
        const float* deltas[NUM_GATES] = { delta_i.data(), delta_f.data(),
                                           delta_g.data(), delta_o.data() };

        for (int g = 0; g < NUM_GATES; ++g) {
            const int   gate_idx = GATE_ORDER[g];
            const float* d       = deltas[g];
            float* wg_gate = wg + gate_idx * _h * _in;
            float* ug_gate = ug + gate_idx * _h * _h;
            float* bg_gate = bg + gate_idx * _h;
            const float* W_gate = _w.data() + gate_idx * _h * _in;
            const float* U_gate = _u.data() + gate_idx * _h * _h;

            for (int j = 0; j < _h; ++j) {
                if (d[j] == 0.f) continue;
                axpy(wg_gate + j * _in, x_t,    d[j], _in);
                axpy(ug_gate + j * _h,  h_prev, d[j], _h);
                bg_gate[j] += d[j];
                // Input grad: W^T · delta  (contribution of this gate)
                axpy(ig_t, W_gate + j * _in, d[j], _in);
                // Recurrent grad: U^T · delta
                axpy(dh_recur.data(), U_gate + j * _h, d[j], _h);
            }
        }

        // dc for previous step: dc_prev = dc ⊙ gf
        for (int j = 0; j < _h; ++j)
            dc_recur[j] = dc[j] * gf[j];
    }

    Array result;
    result.resize(2);
    result[0] = input_grads;
    result[1] = grad_buffer;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_gradients
// ─────────────────────────────────────────────────────────────────────────────

void RlLstmLayer::apply_gradients(const PackedFloat32Array& grad_buffer,
                                   float lr, float grad_scale, float grad_clip_norm)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] apply_gradients called before initialize.");
        return;
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlLstmLayer] apply_gradients: grad_buffer size mismatch.");
        return;
    }
    if (_optimizer == -1) return;  // frozen

    // Gradient clipping: scale all grads if their L2 norm exceeds the threshold
    // Work on a mutable copy for clipping (avoid modifying caller's buffer)
    std::vector<float> buf_copy;
    const float* gb;
    if (grad_clip_norm > 0.f) {
        float sq = 0.f;
        const float* p = grad_buffer.ptr();
        for (int i = 0; i < _grad_buf_size; ++i) sq += p[i] * p[i];
        float norm = std::sqrt(sq);
        if (norm > grad_clip_norm) {
            buf_copy.assign(p, p + _grad_buf_size);
            float clip_scale = grad_clip_norm / norm;
            scale_buffer(buf_copy.data(), _grad_buf_size, clip_scale);
            gb = buf_copy.data();
        } else {
            gb = p;
        }
    } else {
        gb = grad_buffer.ptr();
    }

    const int wsize = NUM_GATES * _h * _in;
    const int usize = NUM_GATES * _h * _h;
    const int bsize = NUM_GATES * _h;
    const float* wg = gb;
    const float* ug = wg + wsize;
    const float* bg_ = ug + usize;

    if (_optimizer == 0) {  // Adam
        _b1t *= kB1; _b2t *= kB2;
        adam_update_exact(_w.data(), _wm.data(), _wv.data(), wg,  wsize, lr, grad_scale, _b1t, _b2t);
        adam_update_exact(_u.data(), _um.data(), _uv.data(), ug,  usize, lr, grad_scale, _b1t, _b2t);
        adam_update_exact(_b.data(), _bm.data(), _bv.data(), bg_, bsize, lr, grad_scale, _b1t, _b2t);
    } else {  // SGD
        for (int i = 0; i < wsize; ++i) _w[i] -= lr * wg[i]  * grad_scale;
        for (int i = 0; i < usize; ++i) _u[i] -= lr * ug[i]  * grad_scale;
        for (int i = 0; i < bsize; ++i) _b[i] -= lr * bg_[i] * grad_scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// create_gradient_buffer / grad_norm_squared
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLstmLayer::create_gradient_buffer() const
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] create_gradient_buffer called before initialize.");
        return PackedFloat32Array();
    }
    PackedFloat32Array buf;
    buf.resize(_grad_buf_size);
    std::memset(buf.ptrw(), 0, _grad_buf_size * sizeof(float));
    return buf;
}

float RlLstmLayer::grad_norm_squared(const PackedFloat32Array& grad_buffer) const
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] grad_norm_squared called before initialize.");
        return 0.f;
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlLstmLayer] grad_norm_squared: grad_buffer size mismatch. Expected " +
                                     String::num_int64(_grad_buf_size) + " got " + String::num_int64(grad_buffer.size()) + ".");
        return 0.f;
    }
    const float* p = grad_buffer.ptr();
    float sum = 0.f;
    for (int i = 0; i < _grad_buf_size; ++i) sum += p[i] * p[i];
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// copy_weights_from / soft_update_from
// ─────────────────────────────────────────────────────────────────────────────

void RlLstmLayer::copy_weights_from(Object* source_obj)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] copy_weights_from called before initialize.");
        return;
    }
    RlLstmLayer* src = Object::cast_to<RlLstmLayer>(source_obj);
    if (!src || !src->_ready || src->_in != _in || src->_h != _h) {
        UtilityFunctions::push_error("[RlLstmLayer] copy_weights_from: null/incompatible source.");
        return;
    }
    std::memcpy(_w.data(), src->_w.data(), _w.size() * sizeof(float));
    std::memcpy(_u.data(), src->_u.data(), _u.size() * sizeof(float));
    std::memcpy(_b.data(), src->_b.data(), _b.size() * sizeof(float));
}

void RlLstmLayer::soft_update_from(Object* source_obj, float tau)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] soft_update_from called before initialize.");
        return;
    }
    RlLstmLayer* src = Object::cast_to<RlLstmLayer>(source_obj);
    if (!src || !src->_ready || src->_in != _in || src->_h != _h) {
        UtilityFunctions::push_error("[RlLstmLayer] soft_update_from: null/incompatible source.");
        return;
    }
    float omt = 1.f - tau;
    for (int i = 0; i < (int)_w.size(); ++i) _w[i] = tau * src->_w[i] + omt * _w[i];
    for (int i = 0; i < (int)_u.size(); ++i) _u[i] = tau * src->_u[i] + omt * _u[i];
    for (int i = 0; i < (int)_b.size(); ++i) _b[i] = tau * src->_b[i] + omt * _b[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLstmLayer::get_weights() const
{
    // Layout: [W | U | B]
    const int total = (int)(_w.size() + _u.size() + _b.size());
    PackedFloat32Array out;
    out.resize(total);
    float* dst = out.ptrw();
    std::memcpy(dst,                         _w.data(), _w.size() * sizeof(float));
    std::memcpy(dst + _w.size(),             _u.data(), _u.size() * sizeof(float));
    std::memcpy(dst + _w.size() + _u.size(), _b.data(), _b.size() * sizeof(float));
    return out;
}

PackedInt32Array RlLstmLayer::get_shapes() const
{
    // [RLLayerKind::Lstm=4, input_size, hidden_size]
    PackedInt32Array out;
    out.push_back(4);   // RLLayerKind.Lstm
    out.push_back(_in);
    out.push_back(_h);
    return out;
}

void RlLstmLayer::set_weights(const PackedFloat32Array& weights,
                               const PackedInt32Array& /*shapes*/)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLstmLayer] set_weights called before initialize.");
        return;
    }
    const int expected = (int)(_w.size() + _u.size() + _b.size());
    if (weights.size() != expected) {
        UtilityFunctions::push_error("[RlLstmLayer] set_weights: weight array size mismatch.");
        return;
    }
    const float* src = weights.ptr();
    std::memcpy(_w.data(), src,                         _w.size() * sizeof(float));
    std::memcpy(_u.data(), src + _w.size(),             _u.size() * sizeof(float));
    std::memcpy(_b.data(), src + _w.size() + _u.size(), _b.size() * sizeof(float));

    // Reset Adam moments (matches LoadSerialized behaviour in DenseLayer)
    if (_optimizer == 0) {
        std::fill(_wm.begin(), _wm.end(), 0.f); std::fill(_wv.begin(), _wv.end(), 0.f);
        std::fill(_um.begin(), _um.end(), 0.f); std::fill(_uv.begin(), _uv.end(), 0.f);
        std::fill(_bm.begin(), _bm.end(), 0.f); std::fill(_bv.begin(), _bv.end(), 0.f);
        _b1t = 1.f; _b2t = 1.f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GDExtension binding
// ─────────────────────────────────────────────────────────────────────────────

void RlLstmLayer::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("initialize", "input_size", "hidden_size", "optimizer"),
                         &RlLstmLayer::initialize);

    ClassDB::bind_method(D_METHOD("forward", "input", "h_prev", "c_prev"),
                         &RlLstmLayer::forward);

    ClassDB::bind_method(D_METHOD("forward_sequence", "flat_inputs", "seq_len", "h0", "c0"),
                         &RlLstmLayer::forward_sequence);

    ClassDB::bind_method(D_METHOD("accumulate_sequence_gradients",
                                  "seq_h_grads", "grad_buffer", "h0", "c0"),
                         &RlLstmLayer::accumulate_sequence_gradients);

    ClassDB::bind_method(D_METHOD("apply_gradients", "grad_buffer", "lr", "grad_scale", "grad_clip_norm"),
                         &RlLstmLayer::apply_gradients);

    ClassDB::bind_method(D_METHOD("create_gradient_buffer"),
                         &RlLstmLayer::create_gradient_buffer);

    ClassDB::bind_method(D_METHOD("grad_norm_squared", "grad_buffer"),
                         &RlLstmLayer::grad_norm_squared);

    ClassDB::bind_method(D_METHOD("copy_weights_from", "source"),
                         &RlLstmLayer::copy_weights_from);

    ClassDB::bind_method(D_METHOD("soft_update_from", "source", "tau"),
                         &RlLstmLayer::soft_update_from);

    ClassDB::bind_method(D_METHOD("get_weights"), &RlLstmLayer::get_weights);
    ClassDB::bind_method(D_METHOD("get_shapes"),  &RlLstmLayer::get_shapes);
    ClassDB::bind_method(D_METHOD("set_weights", "weights", "shapes"),
                         &RlLstmLayer::set_weights);

    ClassDB::bind_method(D_METHOD("get_input_size"),  &RlLstmLayer::get_input_size);
    ClassDB::bind_method(D_METHOD("get_hidden_size"), &RlLstmLayer::get_hidden_size);
}
