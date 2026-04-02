#include "RlGruLayer.h"
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

static constexpr int GATE_R = 0;   // reset gate  (sigmoid)
static constexpr int GATE_Z = 1;   // update gate (sigmoid)
static constexpr int GATE_N = 2;   // new gate    (tanh)

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline float sigmoid(float x) noexcept
{
    return 1.f / (1.f + std::exp(-x));
}

void RlGruLayer::scale_buffer(float* buf, int n, float scale)
{
    for (int i = 0; i < n; ++i) buf[i] *= scale;
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize
// ─────────────────────────────────────────────────────────────────────────────

void RlGruLayer::initialize(int input_size, int hidden_size, int optimizer)
{
    if (input_size <= 0 || hidden_size <= 0) {
        UtilityFunctions::push_error("[RlGruLayer] initialize: input_size and hidden_size must be > 0.");
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

    float w_scale = std::sqrt(2.f / float(std::max(1, input_size)));
    float u_scale = std::sqrt(2.f / float(std::max(1, hidden_size)));

    Pcg32 rng_w(uint64_t(42 + uint64_t(wsize) * 11));
    for (auto& v : _w) v = rng_w.next_gaussian() * w_scale;

    Pcg32 rng_u(uint64_t(137 + uint64_t(usize) * 17));
    for (auto& v : _u) v = rng_u.next_gaussian() * u_scale;

    _grad_buf_size = wsize + usize + bsize;
    _ready = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward — single-step inference
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlGruLayer::forward(const PackedFloat32Array& input,
                                        const PackedFloat32Array& h_prev)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlGruLayer] forward called before initialize.");
        return PackedFloat32Array();
    }
    if (input.size() != _in || h_prev.size() != _h) {
        UtilityFunctions::push_error("[RlGruLayer] forward: input / h_prev size mismatch.");
        return PackedFloat32Array();
    }

    const float* x = input.ptr();
    const float* hp = h_prev.ptr();

    // r = sigmoid(Wr·x + Ur·h_prev + br)
    // z = sigmoid(Wz·x + Uz·h_prev + bz)
    std::vector<float> r(_h), z(_h), rh(_h);
    for (int j = 0; j < _h; ++j) {
        float pr = _b[GATE_R * _h + j]
                 + dot_avx(x,  _w.data() + GATE_R * _h * _in + j * _in, _in)
                 + dot_avx(hp, _u.data() + GATE_R * _h * _h  + j * _h,  _h);
        r[j] = sigmoid(pr);

        float pz = _b[GATE_Z * _h + j]
                 + dot_avx(x,  _w.data() + GATE_Z * _h * _in + j * _in, _in)
                 + dot_avx(hp, _u.data() + GATE_Z * _h * _h  + j * _h,  _h);
        z[j] = sigmoid(pz);

        rh[j] = r[j] * hp[j];
    }

    // n = tanh(Wn·x + Un·(r ⊙ h_prev) + bn)
    PackedFloat32Array result;
    result.resize(_h);
    float* dst = result.ptrw();
    for (int j = 0; j < _h; ++j) {
        float pn = _b[GATE_N * _h + j]
                 + dot_avx(x,       _w.data() + GATE_N * _h * _in + j * _in, _in)
                 + dot_avx(rh.data(), _u.data() + GATE_N * _h * _h  + j * _h,  _h);
        float n_j = std::tanh(pn);
        // h' = (1-z) ⊙ h_prev + z ⊙ n
        dst[j] = (1.f - z[j]) * hp[j] + z[j] * n_j;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_sequence — training forward
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlGruLayer::forward_sequence(const PackedFloat32Array& flat_inputs,
                                                  int                        seq_len,
                                                  const PackedFloat32Array& h0)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlGruLayer] forward_sequence called before initialize.");
        return PackedFloat32Array();
    }
    if (seq_len <= 0 || flat_inputs.size() != seq_len * _in || h0.size() != _h) {
        UtilityFunctions::push_error("[RlGruLayer] forward_sequence: size mismatch.");
        return PackedFloat32Array();
    }

    const int T = seq_len;
    _seq_len = T;

    _seq_x.resize(T * _in);
    _seq_rz.resize(T * 2 * _h);   // [r | z] per step
    _seq_n.resize(T * _h);
    _seq_rh.resize(T * _h);        // r ⊙ h_prev per step
    _seq_h.resize((T + 1) * _h);

    std::memcpy(_seq_x.data(), flat_inputs.ptr(), T * _in * sizeof(float));
    std::memcpy(_seq_h.data(), h0.ptr(), _h * sizeof(float));   // _seq_h[0] = h0

    const float* x_ptr = flat_inputs.ptr();
    for (int t = 0; t < T; ++t) {
        const float* x  = x_ptr + t * _in;
        const float* hp = _seq_h.data() + t * _h;
        float*       r  = _seq_rz.data() + t * 2 * _h + GATE_R * _h;
        float*       z  = _seq_rz.data() + t * 2 * _h + GATE_Z * _h;
        float*       rh = _seq_rh.data() + t * _h;
        float*       n  = _seq_n.data()  + t * _h;
        float*       hn = _seq_h.data()  + (t + 1) * _h;

        for (int j = 0; j < _h; ++j) {
            float pr = _b[GATE_R * _h + j]
                     + dot_avx(x,  _w.data() + GATE_R * _h * _in + j * _in, _in)
                     + dot_avx(hp, _u.data() + GATE_R * _h * _h  + j * _h,  _h);
            r[j] = sigmoid(pr);

            float pz = _b[GATE_Z * _h + j]
                     + dot_avx(x,  _w.data() + GATE_Z * _h * _in + j * _in, _in)
                     + dot_avx(hp, _u.data() + GATE_Z * _h * _h  + j * _h,  _h);
            z[j] = sigmoid(pz);

            rh[j] = r[j] * hp[j];
        }

        for (int j = 0; j < _h; ++j) {
            float pn = _b[GATE_N * _h + j]
                     + dot_avx(x,   _w.data() + GATE_N * _h * _in + j * _in, _in)
                     + dot_avx(rh,  _u.data() + GATE_N * _h * _h  + j * _h,  _h);
            n[j] = std::tanh(pn);
            hn[j] = (1.f - z[j]) * hp[j] + z[j] * n[j];
        }
    }

    PackedFloat32Array result;
    result.resize(T * _h);
    std::memcpy(result.ptrw(), _seq_h.data() + _h, T * _h * sizeof(float));
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// accumulate_sequence_gradients — BPTT for GRU
//
// Backward equations (at each step t from T-1 to 0):
//   dh_total = ext_dh[t] + dh_recur
//
//   dn = dh_total ⊙ z[t]
//   dz = dh_total ⊙ (n[t] - h_prev[t])
//
//   delta_n = dn ⊙ (1 - n[t]²)              (tanh deriv)
//   delta_z = dz ⊙ z[t] ⊙ (1 - z[t])        (sigmoid deriv)
//
//   // n = tanh(Wn·x + Un·(r⊙h_prev) + bn)
//   // drh = Un^T · delta_n    (grad to r⊙h_prev)
//   drh[j] = sum_k Un[k,j] * delta_n[k]
//   dr[j]  = drh[j] * h_prev[j]             (grad to r through r⊙h_prev)
//   dh_from_n[j] = drh[j] * r[j]            (grad to h_prev through r⊙h_prev)
//
//   delta_r = dr ⊙ r[t] ⊙ (1 - r[t])        (sigmoid deriv)
//
//   // Grad to h_prev:
//   dh_recur = (1-z[t]) ⊙ dh_total           (direct path through h')
//            + dh_from_n                      (through r⊙h_prev in new gate)
//            + Ur^T · delta_r                 (through r gate pre-act)
//            + Uz^T · delta_z                 (through z gate pre-act)
// ─────────────────────────────────────────────────────────────────────────────

Array RlGruLayer::accumulate_sequence_gradients(const PackedFloat32Array& seq_h_grads,
                                                  int seq_len,
                                                  PackedFloat32Array        grad_buffer,
                                                  const PackedFloat32Array& h0)
{
    Array error_result;
    error_result.resize(2);
    error_result[0] = PackedFloat32Array();
    error_result[1] = grad_buffer;

    if (!_ready || _seq_len <= 0) {
        UtilityFunctions::push_error("[RlGruLayer] accumulate_sequence_gradients: forward_sequence not called.");
        return error_result;
    }
    if (seq_len != _seq_len) {
        UtilityFunctions::push_error("[RlGruLayer] accumulate_sequence_gradients: seq_len does not match the cached forward sequence length.");
        return error_result;
    }

    const float* cached_h0 = _seq_h.data();
    const float* provided_h0 = h0.ptr();
    if (h0.size() != _h) {
        UtilityFunctions::push_error("[RlGruLayer] accumulate_sequence_gradients: h0 size mismatch.");
        return error_result;
    }
    for (int i = 0; i < _h; ++i) {
        if (std::fabs(provided_h0[i] - cached_h0[i]) > 1e-6f) {
            UtilityFunctions::push_error("[RlGruLayer] accumulate_sequence_gradients: provided h0 does not match the cached forward sequence state.");
            return error_result;
        }
    }

    const int T = seq_len;
    if (seq_h_grads.size() != T * _h || grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlGruLayer] accumulate_sequence_gradients: size mismatch.");
        return error_result;
    }

    const int wsize = NUM_GATES * _h * _in;
    const int usize = NUM_GATES * _h * _h;

    float* gb  = grad_buffer.ptrw();
    float* wg  = gb;
    float* ug  = wg + wsize;
    float* bg_ = ug + usize;

    PackedFloat32Array input_grads;
    input_grads.resize(T * _in);
    float* ig_all = input_grads.ptrw();
    std::memset(ig_all, 0, T * _in * sizeof(float));

    const float* ext_dh = seq_h_grads.ptr();

    std::vector<float> dh_recur(_h, 0.f);
    std::vector<float> drh(_h), dr(_h), dh_from_n(_h);
    std::vector<float> delta_r(_h), delta_z(_h), delta_n(_h);

    for (int t = T - 1; t >= 0; --t) {
        const float* r  = _seq_rz.data() + t * 2 * _h + GATE_R * _h;
        const float* z  = _seq_rz.data() + t * 2 * _h + GATE_Z * _h;
        const float* n  = _seq_n.data()  + t * _h;
        const float* rh = _seq_rh.data() + t * _h;   // r ⊙ h_prev
        const float* hp = _seq_h.data()  + t * _h;   // h_prev (= h[t])
        const float* x  = _seq_x.data()  + t * _in;
        float*       ig = ig_all         + t * _in;

        // Total gradient to h[t+1]
        std::vector<float> dh(_h);
        for (int j = 0; j < _h; ++j)
            dh[j] = ext_dh[t * _h + j] + dh_recur[j];

        // dn = dh ⊙ z;   dz = dh ⊙ (n - h_prev)
        for (int j = 0; j < _h; ++j) {
            float dn_j = dh[j] * z[j];
            float dz_j = dh[j] * (n[j] - hp[j]);
            delta_n[j] = dn_j * (1.f - n[j] * n[j]);     // tanh deriv
            delta_z[j] = dz_j * z[j] * (1.f - z[j]);     // sigmoid deriv
        }

        // drh = Un^T · delta_n
        std::fill(drh.begin(), drh.end(), 0.f);
        for (int k = 0; k < _h; ++k) {
            if (delta_n[k] == 0.f) continue;
            axpy(drh.data(), _u.data() + GATE_N * _h * _h + k * _h, delta_n[k], _h);
        }

        // dr = drh ⊙ h_prev;  dh_from_n = drh ⊙ r
        for (int j = 0; j < _h; ++j) {
            dr[j]       = drh[j] * hp[j];
            dh_from_n[j] = drh[j] * r[j];
            delta_r[j]  = dr[j] * r[j] * (1.f - r[j]);   // sigmoid deriv
        }

        // ── Accumulate weight/bias grads ──────────────────────────────────────

        // Gate R: outer(delta_r, x) for W; outer(delta_r, h_prev) for U
        {
            float* wg_r = wg + GATE_R * _h * _in;
            float* ug_r = ug + GATE_R * _h * _h;
            float* bg_r = bg_ + GATE_R * _h;
            for (int j = 0; j < _h; ++j) {
                if (delta_r[j] == 0.f) continue;
                axpy(wg_r + j * _in, x,  delta_r[j], _in);
                axpy(ug_r + j * _h,  hp, delta_r[j], _h);
                bg_r[j] += delta_r[j];
                axpy(ig, _w.data() + GATE_R * _h * _in + j * _in, delta_r[j], _in);
            }
        }

        // Gate Z: outer(delta_z, x) for W; outer(delta_z, h_prev) for U
        {
            float* wg_z = wg + GATE_Z * _h * _in;
            float* ug_z = ug + GATE_Z * _h * _h;
            float* bg_z = bg_ + GATE_Z * _h;
            for (int j = 0; j < _h; ++j) {
                if (delta_z[j] == 0.f) continue;
                axpy(wg_z + j * _in, x,  delta_z[j], _in);
                axpy(ug_z + j * _h,  hp, delta_z[j], _h);
                bg_z[j] += delta_z[j];
                axpy(ig, _w.data() + GATE_Z * _h * _in + j * _in, delta_z[j], _in);
            }
        }

        // Gate N: outer(delta_n, x) for W; outer(delta_n, rh) for U
        {
            float* wg_n = wg + GATE_N * _h * _in;
            float* ug_n = ug + GATE_N * _h * _h;
            float* bg_n = bg_ + GATE_N * _h;
            for (int j = 0; j < _h; ++j) {
                if (delta_n[j] == 0.f) continue;
                axpy(wg_n + j * _in, x,  delta_n[j], _in);
                axpy(ug_n + j * _h,  rh, delta_n[j], _h);  // rh = r ⊙ h_prev
                bg_n[j] += delta_n[j];
                axpy(ig, _w.data() + GATE_N * _h * _in + j * _in, delta_n[j], _in);
            }
        }

        // ── Recurrent gradient to h[t] ────────────────────────────────────────
        // dh_recur = (1-z) ⊙ dh  (direct)
        //          + dh_from_n   (through n gate via r⊙h_prev)
        //          + Ur^T · delta_r
        //          + Uz^T · delta_z
        std::fill(dh_recur.begin(), dh_recur.end(), 0.f);
        for (int j = 0; j < _h; ++j)
            dh_recur[j] = (1.f - z[j]) * dh[j] + dh_from_n[j];

        for (int k = 0; k < _h; ++k) {
            if (delta_r[k] != 0.f)
                axpy(dh_recur.data(), _u.data() + GATE_R * _h * _h + k * _h, delta_r[k], _h);
            if (delta_z[k] != 0.f)
                axpy(dh_recur.data(), _u.data() + GATE_Z * _h * _h + k * _h, delta_z[k], _h);
        }
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

void RlGruLayer::apply_gradients(const PackedFloat32Array& grad_buffer,
                                  float lr, float grad_scale, float grad_clip_norm)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlGruLayer] apply_gradients called before initialize.");
        return;
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlGruLayer] apply_gradients: grad_buffer size mismatch.");
        return;
    }
    if (_optimizer == -1) return;

    std::vector<float> buf_copy;
    const float* gb;
    if (grad_clip_norm > 0.f) {
        const float* p = grad_buffer.ptr();
        float sq = 0.f;
        for (int i = 0; i < _grad_buf_size; ++i) sq += p[i] * p[i];
        float norm = std::sqrt(sq);
        if (norm > grad_clip_norm) {
            buf_copy.assign(p, p + _grad_buf_size);
            scale_buffer(buf_copy.data(), _grad_buf_size, grad_clip_norm / norm);
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
    const float* wg  = gb;
    const float* ug_ = wg + wsize;
    const float* bg_ = ug_ + usize;

    if (_optimizer == 0) {
        _b1t *= kB1; _b2t *= kB2;
        adam_update_exact(_w.data(), _wm.data(), _wv.data(), wg,  wsize, lr, grad_scale, _b1t, _b2t);
        adam_update_exact(_u.data(), _um.data(), _uv.data(), ug_, usize, lr, grad_scale, _b1t, _b2t);
        adam_update_exact(_b.data(), _bm.data(), _bv.data(), bg_, bsize, lr, grad_scale, _b1t, _b2t);
    } else {
        for (int i = 0; i < wsize; ++i) _w[i] -= lr * wg[i]  * grad_scale;
        for (int i = 0; i < usize; ++i) _u[i] -= lr * ug_[i] * grad_scale;
        for (int i = 0; i < bsize; ++i) _b[i] -= lr * bg_[i] * grad_scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// create_gradient_buffer / grad_norm_squared
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlGruLayer::create_gradient_buffer() const
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlGruLayer] create_gradient_buffer called before initialize.");
        return PackedFloat32Array();
    }
    PackedFloat32Array buf;
    buf.resize(_grad_buf_size);
    std::memset(buf.ptrw(), 0, _grad_buf_size * sizeof(float));
    return buf;
}

float RlGruLayer::grad_norm_squared(const PackedFloat32Array& grad_buffer) const
{
    if (!_ready || grad_buffer.size() != _grad_buf_size) return 0.f;
    const float* p = grad_buffer.ptr();
    float sum = 0.f;
    for (int i = 0; i < _grad_buf_size; ++i) sum += p[i] * p[i];
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// copy_weights_from / soft_update_from
// ─────────────────────────────────────────────────────────────────────────────

void RlGruLayer::copy_weights_from(Object* source_obj)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlGruLayer] copy_weights_from called before initialize.");
        return;
    }
    RlGruLayer* src = Object::cast_to<RlGruLayer>(source_obj);
    if (!src || !src->_ready || src->_in != _in || src->_h != _h) {
        UtilityFunctions::push_error("[RlGruLayer] copy_weights_from: null/incompatible source.");
        return;
    }
    std::memcpy(_w.data(), src->_w.data(), _w.size() * sizeof(float));
    std::memcpy(_u.data(), src->_u.data(), _u.size() * sizeof(float));
    std::memcpy(_b.data(), src->_b.data(), _b.size() * sizeof(float));
}

void RlGruLayer::soft_update_from(Object* source_obj, float tau)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlGruLayer] soft_update_from called before initialize.");
        return;
    }
    RlGruLayer* src = Object::cast_to<RlGruLayer>(source_obj);
    if (!src || !src->_ready || src->_in != _in || src->_h != _h) {
        UtilityFunctions::push_error("[RlGruLayer] soft_update_from: null/incompatible source.");
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

PackedFloat32Array RlGruLayer::get_weights() const
{
    const int total = (int)(_w.size() + _u.size() + _b.size());
    PackedFloat32Array out;
    out.resize(total);
    float* dst = out.ptrw();
    std::memcpy(dst,                         _w.data(), _w.size() * sizeof(float));
    std::memcpy(dst + _w.size(),             _u.data(), _u.size() * sizeof(float));
    std::memcpy(dst + _w.size() + _u.size(), _b.data(), _b.size() * sizeof(float));
    return out;
}

PackedInt32Array RlGruLayer::get_shapes() const
{
    PackedInt32Array out;
    out.push_back(5);   // RLLayerKind.Gru
    out.push_back(_in);
    out.push_back(_h);
    return out;
}

void RlGruLayer::set_weights(const PackedFloat32Array& weights,
                              const PackedInt32Array& /*shapes*/)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlGruLayer] set_weights called before initialize.");
        return;
    }
    const int expected = (int)(_w.size() + _u.size() + _b.size());
    if (weights.size() != expected) {
        UtilityFunctions::push_error("[RlGruLayer] set_weights: weight array size mismatch.");
        return;
    }
    const float* src = weights.ptr();
    std::memcpy(_w.data(), src,                         _w.size() * sizeof(float));
    std::memcpy(_u.data(), src + _w.size(),             _u.size() * sizeof(float));
    std::memcpy(_b.data(), src + _w.size() + _u.size(), _b.size() * sizeof(float));

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

void RlGruLayer::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("initialize", "input_size", "hidden_size", "optimizer"),
                         &RlGruLayer::initialize);

    ClassDB::bind_method(D_METHOD("forward", "input", "h_prev"),
                         &RlGruLayer::forward);

    ClassDB::bind_method(D_METHOD("forward_sequence", "flat_inputs", "seq_len", "h0"),
                         &RlGruLayer::forward_sequence);

    ClassDB::bind_method(D_METHOD("accumulate_sequence_gradients",
                                  "seq_h_grads", "seq_len", "grad_buffer", "h0"),
                         &RlGruLayer::accumulate_sequence_gradients);

    ClassDB::bind_method(D_METHOD("apply_gradients", "grad_buffer", "lr", "grad_scale", "grad_clip_norm"),
                         &RlGruLayer::apply_gradients);

    ClassDB::bind_method(D_METHOD("create_gradient_buffer"),
                         &RlGruLayer::create_gradient_buffer);

    ClassDB::bind_method(D_METHOD("grad_norm_squared", "grad_buffer"),
                         &RlGruLayer::grad_norm_squared);

    ClassDB::bind_method(D_METHOD("copy_weights_from", "source"),
                         &RlGruLayer::copy_weights_from);

    ClassDB::bind_method(D_METHOD("soft_update_from", "source", "tau"),
                         &RlGruLayer::soft_update_from);

    ClassDB::bind_method(D_METHOD("get_weights"), &RlGruLayer::get_weights);
    ClassDB::bind_method(D_METHOD("get_shapes"),  &RlGruLayer::get_shapes);
    ClassDB::bind_method(D_METHOD("set_weights", "weights", "shapes"),
                         &RlGruLayer::set_weights);

    ClassDB::bind_method(D_METHOD("get_input_size"),  &RlGruLayer::get_input_size);
    ClassDB::bind_method(D_METHOD("get_hidden_size"), &RlGruLayer::get_hidden_size);
}
