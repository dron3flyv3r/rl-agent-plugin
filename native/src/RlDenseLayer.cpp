#include "RlDenseLayer.h"
#include "rl_simd.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// Activation helpers
// ─────────────────────────────────────────────────────────────────────────────

inline float RlDenseLayer::apply_act(float x) const noexcept
{
    if (_activation == 1) return std::tanh(x);         // Tanh
    if (_activation == 2) return x > 0.f ? x : 0.f;   // ReLU
    return x;                                           // None
}

inline float RlDenseLayer::act_deriv(float preact) const noexcept
{
    if (_activation == 1) { float t = std::tanh(preact); return 1.f - t * t; }
    if (_activation == 2) return preact > 0.f ? 1.f : 0.f;
    return 1.f;
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize
// ─────────────────────────────────────────────────────────────────────────────

void RlDenseLayer::initialize(int in_size, int out_size, int activation, int optimizer)
{
    if (in_size <= 0 || out_size <= 0) {
        UtilityFunctions::push_error("[RlDenseLayer] initialize: in_size and out_size must be > 0.");
        return;
    }

    _in         = in_size;
    _out        = out_size;
    _activation = activation;
    _optimizer  = optimizer;

    int wsize = out_size * in_size;
    _weights.assign(wsize,    0.f);
    _biases.assign(out_size,  0.f);
    _cached_input.resize(in_size);
    _cached_preact.resize(out_size);

    if (optimizer == 0 || optimizer == 2) {  // Adam or AdamW
        _wm.assign(wsize,    0.f); _wv.assign(wsize,    0.f);
        _bm.assign(out_size, 0.f); _bv.assign(out_size, 0.f);
    } else {
        _wm.clear(); _wv.clear();
        _bm.clear(); _bv.clear();
    }
    _b1t = 1.f;
    _b2t = 1.f;

    // He initialisation: N(0, sqrt(2/fan_in))
    float scale = std::sqrt(2.f / float(std::max(1, in_size)));
    Pcg32 rng(uint64_t(42 + uint64_t(wsize) * 7));
    for (auto& w : _weights) w = rng.next_gaussian() * scale;

    _grad_buf_size = wsize + out_size;
    _ready = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// set_weight_decay
// ─────────────────────────────────────────────────────────────────────────────

void RlDenseLayer::set_weight_decay(float wd)
{
    _weight_decay = wd;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward (single-sample, caches state)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::forward(const PackedFloat32Array& input)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] forward called before initialize.");
        return PackedFloat32Array();
    }
    if (input.size() != _in) {
        UtilityFunctions::push_error("[RlDenseLayer] forward input size mismatch.");
        return PackedFloat32Array();
    }

    const float* x = input.ptr();
    std::memcpy(_cached_input.data(), x, _in * sizeof(float));

    for (int o = 0; o < _out; ++o)
        _cached_preact[o] = _biases[o] + dot_avx(x, _weights.data() + o * _in, _in);

    PackedFloat32Array result;
    result.resize(_out);
    float* dst = result.ptrw();
    for (int o = 0; o < _out; ++o)
        dst[o] = apply_act(_cached_preact[o]);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_batch (batch inference, no gradient state)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::forward_batch(const PackedFloat32Array& flat_input,
                                                int batch_size)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] forward_batch called before initialize.");
        return PackedFloat32Array();
    }
    if (batch_size <= 0) {
        UtilityFunctions::push_error("[RlDenseLayer] forward_batch batch_size must be > 0.");
        return PackedFloat32Array();
    }
    if (flat_input.size() != batch_size * _in) {
        UtilityFunctions::push_error("[RlDenseLayer] forward_batch input size mismatch.");
        return PackedFloat32Array();
    }

    const float* x = flat_input.ptr();

    PackedFloat32Array result;
    result.resize(batch_size * _out);
    float* dst = result.ptrw();

    for (int b = 0; b < batch_size; ++b) {
        const float* in_b  = x   + b * _in;
        float*       out_b = dst + b * _out;
        for (int o = 0; o < _out; ++o) {
            float val = _biases[o] + dot_avx(in_b, _weights.data() + o * _in, _in);
            out_b[o] = apply_act(val);
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: compute local gradient (output_grad ⊙ activation')
// ─────────────────────────────────────────────────────────────────────────────

void RlDenseLayer::compute_local_grad(const float* output_grad, float* local_grad) const
{
    if (_activation == 0) {
        std::memcpy(local_grad, output_grad, _out * sizeof(float));
    } else {
        for (int o = 0; o < _out; ++o)
            local_grad[o] = output_grad[o] * act_deriv(_cached_preact[o]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: compute input gradient from local gradient
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::input_grad_from_local(const float* local_grad) const
{
    PackedFloat32Array result;
    result.resize(_in);
    float* ig = result.ptrw();
    std::memset(ig, 0, _in * sizeof(float));
    for (int o = 0; o < _out; ++o)
        axpy(ig, _weights.data() + o * _in, local_grad[o], _in);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// backward (single-sample, immediate weight update)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::backward(const PackedFloat32Array& output_grad,
                                           float lr, float grad_scale)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] backward called before initialize.");
        return PackedFloat32Array();
    }
    if (output_grad.size() != _out) {
        UtilityFunctions::push_error("[RlDenseLayer] backward output_grad size mismatch.");
        return PackedFloat32Array();
    }

    std::vector<float> local_grad(_out);
    compute_local_grad(output_grad.ptr(), local_grad.data());

    PackedFloat32Array in_grad = input_grad_from_local(local_grad.data());

    if (_optimizer == -1) return in_grad;  // None — frozen, no update

    if (_optimizer == 0 || _optimizer == 2) {  // Adam / AdamW
        _b1t *= kB1; _b2t *= kB2;

        // Weight gradient: wg[o*in + i] = local_grad[o] * cached_input[i]
        std::vector<float> wg(_out * _in, 0.f);
        for (int o = 0; o < _out; ++o)
            axpy(wg.data() + o * _in, _cached_input.data(), local_grad[o], _in);

        if (_optimizer == 2) {  // AdamW — decoupled weight decay on weights only
            adamw_update_exact(_weights.data(), _wm.data(), _wv.data(),
                               wg.data(), _out * _in, lr, grad_scale, _b1t, _b2t, _weight_decay);
            adamw_update_exact(_biases.data(), _bm.data(), _bv.data(),
                               local_grad.data(), _out, lr, grad_scale, _b1t, _b2t, 0.f);
        } else {  // Adam
            adam_update_exact(_weights.data(), _wm.data(), _wv.data(),
                              wg.data(), _out * _in, lr, grad_scale, _b1t, _b2t);
            adam_update_exact(_biases.data(), _bm.data(), _bv.data(),
                              local_grad.data(), _out, lr, grad_scale, _b1t, _b2t);
        }
    } else {  // SGD
        for (int o = 0; o < _out; ++o) {
            axpy(_weights.data() + o * _in, _cached_input.data(),
                 -lr * local_grad[o] * grad_scale, _in);
            _biases[o] -= lr * local_grad[o] * grad_scale;
        }
    }

    return in_grad;
}

// ─────────────────────────────────────────────────────────────────────────────
// create_gradient_buffer
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::create_gradient_buffer() const
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] create_gradient_buffer called before initialize.");
        return PackedFloat32Array();
    }

    PackedFloat32Array buf;
    buf.resize(_grad_buf_size);
    std::memset(buf.ptrw(), 0, _grad_buf_size * sizeof(float));
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// accumulate_gradients
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::accumulate_gradients_impl(const PackedFloat32Array& output_grad,
                                                           PackedFloat32Array&       grad_buffer)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] accumulate_gradients called before initialize.");
        return PackedFloat32Array();
    }
    if (output_grad.size() != _out) {
        UtilityFunctions::push_error("[RlDenseLayer] accumulate_gradients output_grad size mismatch.");
        return PackedFloat32Array();
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlDenseLayer] accumulate_gradients grad_buffer size mismatch.");
        return PackedFloat32Array();
    }

    float*       gb = grad_buffer.ptrw();
    float*       wg = gb;               // weight grads at [0 .. out*in - 1]
    float*       bg = gb + _out * _in;  // bias grads at  [out*in .. out*in + out - 1]

    std::vector<float> local_grad(_out);
    compute_local_grad(output_grad.ptr(), local_grad.data());

    // Accumulate: weight grad += local_grad[o] * cached_input (outer product row by row)
    for (int o = 0; o < _out; ++o) {
        axpy(wg + o * _in, _cached_input.data(), local_grad[o], _in);
        bg[o] += local_grad[o];
    }

    return input_grad_from_local(local_grad.data());
}

PackedFloat32Array RlDenseLayer::accumulate_gradients(const PackedFloat32Array& output_grad,
                                                       PackedFloat32Array        grad_buffer)
{
    return accumulate_gradients_impl(output_grad, grad_buffer);
}

Array RlDenseLayer::accumulate_gradients_with_buffer(const PackedFloat32Array& output_grad,
                                                     PackedFloat32Array        grad_buffer)
{
    PackedFloat32Array input_grad = accumulate_gradients_impl(output_grad, grad_buffer);
    Array result;
    result.resize(2);
    result[0] = input_grad;
    result[1] = grad_buffer;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_gradients — exact bias-corrected Adam matching DenseLayer.cs
// ─────────────────────────────────────────────────────────────────────────────

void RlDenseLayer::apply_gradients(const PackedFloat32Array& grad_buffer,
                                    float lr, float grad_scale)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] apply_gradients called before initialize.");
        return;
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlDenseLayer] apply_gradients grad_buffer size mismatch.");
        return;
    }

    if (_optimizer == -1) return;  // None — frozen

    const float* gb = grad_buffer.ptr();
    const float* wg = gb;
    const float* bg = gb + _out * _in;

    if (_optimizer == 0 || _optimizer == 2) {  // Adam / AdamW
        _b1t *= kB1; _b2t *= kB2;
        if (_optimizer == 2) {  // AdamW — decoupled weight decay on weights only
            adamw_update_exact(_weights.data(), _wm.data(), _wv.data(),
                               wg, _out * _in, lr, grad_scale, _b1t, _b2t, _weight_decay);
            adamw_update_exact(_biases.data(), _bm.data(), _bv.data(),
                               bg, _out, lr, grad_scale, _b1t, _b2t, 0.f);
        } else {  // Adam
            adam_update_exact(_weights.data(), _wm.data(), _wv.data(),
                              wg, _out * _in, lr, grad_scale, _b1t, _b2t);
            adam_update_exact(_biases.data(), _bm.data(), _bv.data(),
                              bg, _out, lr, grad_scale, _b1t, _b2t);
        }
    } else {  // SGD
        int wsize = _out * _in;
        for (int i = 0; i < wsize; ++i)
            _weights[i] -= lr * wg[i] * grad_scale;
        for (int o = 0; o < _out; ++o)
            _biases[o] -= lr * bg[o] * grad_scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_input_grad (frozen — no weight update, used for SAC dQ/da)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::compute_input_grad(const PackedFloat32Array& output_grad)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] compute_input_grad called before initialize.");
        return PackedFloat32Array();
    }
    if (output_grad.size() != _out) {
        UtilityFunctions::push_error("[RlDenseLayer] compute_input_grad output_grad size mismatch.");
        return PackedFloat32Array();
    }

    std::vector<float> local_grad(_out);
    compute_local_grad(output_grad.ptr(), local_grad.data());
    return input_grad_from_local(local_grad.data());
}

// ─────────────────────────────────────────────────────────────────────────────
// grad_norm_squared
// ─────────────────────────────────────────────────────────────────────────────

float RlDenseLayer::grad_norm_squared(const PackedFloat32Array& grad_buffer) const
{
    if (!_ready || grad_buffer.size() != _grad_buf_size)
        return 0.f;

    const float* gb  = grad_buffer.ptr();
    float        sum = 0.f;
    for (int i = 0; i < _grad_buf_size; ++i)
        sum += gb[i] * gb[i];
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// copy_weights_from / soft_update_from
// ─────────────────────────────────────────────────────────────────────────────

void RlDenseLayer::copy_weights_from(Object* source_obj)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] copy_weights_from called before initialize.");
        return;
    }
    RlDenseLayer* source = Object::cast_to<RlDenseLayer>(source_obj);
    if (source == nullptr || !source->_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] copy_weights_from source is null or uninitialized.");
        return;
    }
    if (source->_in != _in || source->_out != _out || source->_activation != _activation) {
        UtilityFunctions::push_error("[RlDenseLayer] copy_weights_from shape mismatch.");
        return;
    }

    std::memcpy(_weights.data(), source->_weights.data(), _weights.size() * sizeof(float));
    std::memcpy(_biases.data(),  source->_biases.data(),  _biases.size()  * sizeof(float));
}

void RlDenseLayer::soft_update_from(Object* source_obj, float tau)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] soft_update_from called before initialize.");
        return;
    }
    if (tau < 0.f || tau > 1.f) {
        UtilityFunctions::push_error("[RlDenseLayer] soft_update_from tau must be in [0,1].");
        return;
    }

    RlDenseLayer* source = Object::cast_to<RlDenseLayer>(source_obj);
    if (source == nullptr || !source->_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] soft_update_from source is null or uninitialized.");
        return;
    }
    if (source->_in != _in || source->_out != _out || source->_activation != _activation) {
        UtilityFunctions::push_error("[RlDenseLayer] soft_update_from shape mismatch.");
        return;
    }

    float one_minus_tau = 1.f - tau;
    int wsize = _out * _in;
    for (int i = 0; i < wsize; ++i)
        _weights[i] = tau * source->_weights[i] + one_minus_tau * _weights[i];
    for (int o = 0; o < _out; ++o)
        _biases[o] = tau * source->_biases[o] + one_minus_tau * _biases[o];
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlDenseLayer::get_weights() const
{
    int total = _out * _in + _out;
    PackedFloat32Array out;
    out.resize(total);
    float* dst = out.ptrw();
    std::memcpy(dst,           _weights.data(), _weights.size() * sizeof(float));
    std::memcpy(dst + _out * _in, _biases.data(), _biases.size()  * sizeof(float));
    return out;
}

PackedInt32Array RlDenseLayer::get_shapes() const
{
    // Matches DenseLayer.AppendSerialized shape descriptor:
    // [RLLayerKind::Dense=0, inputSize, outputSize, activationCode]
    PackedInt32Array out;
    out.push_back(0);           // RLLayerKind.Dense
    out.push_back(_in);
    out.push_back(_out);
    out.push_back(_activation); // 0=None, 1=Tanh, 2=ReLU
    return out;
}

void RlDenseLayer::set_weights(const PackedFloat32Array& weights,
                                const PackedInt32Array&   /*shapes*/)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlDenseLayer] set_weights called before initialize.");
        return;
    }
    const int expected_weights = _out * _in + _out;
    if (weights.size() != expected_weights) {
        UtilityFunctions::push_error("[RlDenseLayer] set_weights weight array size mismatch.");
        return;
    }

    const float* src = weights.ptr();
    std::memcpy(_weights.data(), src,           _weights.size() * sizeof(float));
    std::memcpy(_biases.data(),  src + _out * _in, _biases.size()  * sizeof(float));

    // Reset Adam/AdamW moments so they warm up from the new weights (matches LoadSerialized).
    if (_optimizer == 0 || _optimizer == 2) {
        std::fill(_wm.begin(), _wm.end(), 0.f); std::fill(_wv.begin(), _wv.end(), 0.f);
        std::fill(_bm.begin(), _bm.end(), 0.f); std::fill(_bv.begin(), _bv.end(), 0.f);
        _b1t = 1.f; _b2t = 1.f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GDExtension binding
// ─────────────────────────────────────────────────────────────────────────────

void RlDenseLayer::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("initialize", "in_size", "out_size", "activation", "optimizer"),
                         &RlDenseLayer::initialize);

    ClassDB::bind_method(D_METHOD("set_weight_decay", "wd"),
                         &RlDenseLayer::set_weight_decay);

    ClassDB::bind_method(D_METHOD("forward", "input"),
                         &RlDenseLayer::forward);

    ClassDB::bind_method(D_METHOD("forward_batch", "flat_input", "batch_size"),
                         &RlDenseLayer::forward_batch);

    ClassDB::bind_method(D_METHOD("backward", "output_grad", "lr", "grad_scale"),
                         &RlDenseLayer::backward);

    ClassDB::bind_method(D_METHOD("create_gradient_buffer"),
                         &RlDenseLayer::create_gradient_buffer);

    ClassDB::bind_method(D_METHOD("accumulate_gradients", "output_grad", "grad_buffer"),
                         &RlDenseLayer::accumulate_gradients);

    ClassDB::bind_method(D_METHOD("accumulate_gradients_with_buffer", "output_grad", "grad_buffer"),
                         &RlDenseLayer::accumulate_gradients_with_buffer);

    ClassDB::bind_method(D_METHOD("apply_gradients", "grad_buffer", "lr", "grad_scale"),
                         &RlDenseLayer::apply_gradients);

    ClassDB::bind_method(D_METHOD("compute_input_grad", "output_grad"),
                         &RlDenseLayer::compute_input_grad);

    ClassDB::bind_method(D_METHOD("grad_norm_squared", "grad_buffer"),
                         &RlDenseLayer::grad_norm_squared);

    ClassDB::bind_method(D_METHOD("copy_weights_from", "source"),
                         &RlDenseLayer::copy_weights_from);

    ClassDB::bind_method(D_METHOD("soft_update_from", "source", "tau"),
                         &RlDenseLayer::soft_update_from);

    ClassDB::bind_method(D_METHOD("get_weights"),
                         &RlDenseLayer::get_weights);

    ClassDB::bind_method(D_METHOD("get_shapes"),
                         &RlDenseLayer::get_shapes);

    ClassDB::bind_method(D_METHOD("set_weights", "weights", "shapes"),
                         &RlDenseLayer::set_weights);
}
