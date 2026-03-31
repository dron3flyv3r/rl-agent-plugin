#include "RlLayerNormLayer.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cmath>
#include <cstring>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// initialize
// ─────────────────────────────────────────────────────────────────────────────

void RlLayerNormLayer::initialize(int size)
{
    if (size <= 0) {
        UtilityFunctions::push_error("[RlLayerNormLayer] initialize: size must be > 0.");
        return;
    }
    _size = size;
    _gamma.assign(size, 1.f);   // init to 1 (matches C#)
    _beta.assign(size,  0.f);   // init to 0
    _last_normalized.resize(size, 0.f);
    _last_std        = 1.f;
    _grad_buf_size   = 2 * size;
    _ready = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward (single-sample, caches normalised values and std)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLayerNormLayer::forward(const PackedFloat32Array& input)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] forward called before initialize.");
        return PackedFloat32Array();
    }
    if (input.size() != _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] forward input size mismatch.");
        return PackedFloat32Array();
    }

    const float* x = input.ptr();

    float mean = 0.f;
    for (int i = 0; i < _size; ++i) mean += x[i];
    mean /= float(_size);

    float variance = 0.f;
    for (int i = 0; i < _size; ++i) {
        float d = x[i] - mean;
        variance += d * d;
    }
    variance /= float(_size);

    float std_val = std::sqrt(variance + kEpsLN);
    _last_std = std_val;

    PackedFloat32Array result;
    result.resize(_size);
    float* dst = result.ptrw();
    for (int i = 0; i < _size; ++i) {
        float norm = (x[i] - mean) / std_val;
        _last_normalized[i] = norm;
        dst[i] = _gamma[i] * norm + _beta[i];
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_batch (batch inference, no gradient caching)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLayerNormLayer::forward_batch(const PackedFloat32Array& flat_input,
                                                    int batch_size)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] forward_batch called before initialize.");
        return PackedFloat32Array();
    }
    if (batch_size <= 0) {
        UtilityFunctions::push_error("[RlLayerNormLayer] forward_batch batch_size must be > 0.");
        return PackedFloat32Array();
    }
    if (flat_input.size() != batch_size * _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] forward_batch input size mismatch.");
        return PackedFloat32Array();
    }

    const float* x = flat_input.ptr();

    PackedFloat32Array result;
    result.resize(batch_size * _size);
    float* dst = result.ptrw();

    for (int b = 0; b < batch_size; ++b) {
        const float* in_b  = x   + b * _size;
        float*       out_b = dst + b * _size;

        float mean = 0.f;
        for (int i = 0; i < _size; ++i) mean += in_b[i];
        mean /= float(_size);

        float variance = 0.f;
        for (int i = 0; i < _size; ++i) {
            float d = in_b[i] - mean;
            variance += d * d;
        }
        variance /= float(_size);
        float std_val = std::sqrt(variance + kEpsLN);

        for (int i = 0; i < _size; ++i) {
            float norm = (in_b[i] - mean) / std_val;
            out_b[i] = _gamma[i] * norm + _beta[i];
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: compute_gradients
//
// Standard LayerNorm backward (matches LayerNormLayer.cs ComputeGradients):
//   dNorm[i]    = outputGrad[i] * gamma[i]
//   gammaGrad[i] = outputGrad[i] * normalised[i]
//   betaGrad[i]  = outputGrad[i]
//   meanDNorm   = mean(dNorm)
//   meanDNormN  = mean(dNorm[i] * normalised[i])
//   inputGrad[i] = (dNorm[i] - meanDNorm - normalised[i] * meanDNormN) / std
// ─────────────────────────────────────────────────────────────────────────────

void RlLayerNormLayer::compute_gradients(const float* output_grad,
                                          float*       input_grad,
                                          float*       gamma_grad,
                                          float*       beta_grad) const
{
    const float* norm = _last_normalized.data();
    float std_val = _last_std;

    // dNorm[i] = outputGrad[i] * gamma[i]
    float mean_dNorm  = 0.f;
    float mean_dNormN = 0.f;
    for (int i = 0; i < _size; ++i) {
        float dNorm    = output_grad[i] * _gamma[i];
        gamma_grad[i]  = output_grad[i] * norm[i];
        beta_grad[i]   = output_grad[i];
        mean_dNorm    += dNorm;
        mean_dNormN   += dNorm * norm[i];
    }
    mean_dNorm  /= float(_size);
    mean_dNormN /= float(_size);

    for (int i = 0; i < _size; ++i) {
        float dNorm    = output_grad[i] * _gamma[i];
        input_grad[i]  = (dNorm - mean_dNorm - norm[i] * mean_dNormN) / std_val;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// backward (single-sample, immediate SGD update)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLayerNormLayer::backward(const PackedFloat32Array& output_grad,
                                               float lr, float grad_scale)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] backward called before initialize.");
        return PackedFloat32Array();
    }
    if (output_grad.size() != _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] backward output_grad size mismatch.");
        return PackedFloat32Array();
    }

    std::vector<float> gamma_grad(_size);
    std::vector<float> beta_grad(_size);

    PackedFloat32Array in_grad_arr;
    in_grad_arr.resize(_size);
    float* ig = in_grad_arr.ptrw();

    compute_gradients(output_grad.ptr(), ig, gamma_grad.data(), beta_grad.data());

    for (int i = 0; i < _size; ++i) {
        _gamma[i] -= lr * gamma_grad[i] * grad_scale;
        _beta[i]  -= lr * beta_grad[i]  * grad_scale;
    }
    return in_grad_arr;
}

// ─────────────────────────────────────────────────────────────────────────────
// create_gradient_buffer
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLayerNormLayer::create_gradient_buffer() const
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] create_gradient_buffer called before initialize.");
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

PackedFloat32Array RlLayerNormLayer::accumulate_gradients_impl(const PackedFloat32Array& output_grad,
                                                               PackedFloat32Array&       grad_buffer)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] accumulate_gradients called before initialize.");
        return PackedFloat32Array();
    }
    if (output_grad.size() != _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] accumulate_gradients output_grad size mismatch.");
        return PackedFloat32Array();
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] accumulate_gradients grad_buffer size mismatch.");
        return PackedFloat32Array();
    }

    float* gb = grad_buffer.ptrw();
    float* gg = gb;           // gamma grads at [0 .. size-1]
    float* bg = gb + _size;   // beta grads  at [size .. 2*size-1]

    std::vector<float> gamma_grad(_size);
    std::vector<float> beta_grad(_size);

    PackedFloat32Array in_grad_arr;
    in_grad_arr.resize(_size);
    float* ig = in_grad_arr.ptrw();

    compute_gradients(output_grad.ptr(), ig, gamma_grad.data(), beta_grad.data());

    for (int i = 0; i < _size; ++i) {
        gg[i] += gamma_grad[i];
        bg[i] += beta_grad[i];
    }
    return in_grad_arr;
}

PackedFloat32Array RlLayerNormLayer::accumulate_gradients(const PackedFloat32Array& output_grad,
                                                           PackedFloat32Array        grad_buffer)
{
    return accumulate_gradients_impl(output_grad, grad_buffer);
}

Array RlLayerNormLayer::accumulate_gradients_with_buffer(const PackedFloat32Array& output_grad,
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
// apply_gradients (plain SGD — matches LayerNormLayer.cs which has no Adam)
// ─────────────────────────────────────────────────────────────────────────────

void RlLayerNormLayer::apply_gradients(const PackedFloat32Array& grad_buffer,
                                        float lr, float grad_scale)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] apply_gradients called before initialize.");
        return;
    }
    if (grad_buffer.size() != _grad_buf_size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] apply_gradients grad_buffer size mismatch.");
        return;
    }

    const float* gb = grad_buffer.ptr();
    const float* gg = gb;
    const float* bg = gb + _size;
    for (int i = 0; i < _size; ++i) {
        _gamma[i] -= lr * gg[i] * grad_scale;
        _beta[i]  -= lr * bg[i] * grad_scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_input_grad (frozen — no weight update)
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLayerNormLayer::compute_input_grad(const PackedFloat32Array& output_grad)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] compute_input_grad called before initialize.");
        return PackedFloat32Array();
    }
    if (output_grad.size() != _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] compute_input_grad output_grad size mismatch.");
        return PackedFloat32Array();
    }

    std::vector<float> gamma_grad(_size);
    std::vector<float> beta_grad(_size);

    PackedFloat32Array in_grad_arr;
    in_grad_arr.resize(_size);
    float* ig = in_grad_arr.ptrw();

    compute_gradients(output_grad.ptr(), ig, gamma_grad.data(), beta_grad.data());
    return in_grad_arr;
}

// ─────────────────────────────────────────────────────────────────────────────
// grad_norm_squared
// ─────────────────────────────────────────────────────────────────────────────

float RlLayerNormLayer::grad_norm_squared(const PackedFloat32Array& grad_buffer) const
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

void RlLayerNormLayer::copy_weights_from(Object* source_obj)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] copy_weights_from called before initialize.");
        return;
    }

    RlLayerNormLayer* source = Object::cast_to<RlLayerNormLayer>(source_obj);
    if (source == nullptr || !source->_ready || source->_size != _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] copy_weights_from source mismatch.");
        return;
    }

    std::memcpy(_gamma.data(), source->_gamma.data(), _size * sizeof(float));
    std::memcpy(_beta.data(),  source->_beta.data(),  _size * sizeof(float));
}

void RlLayerNormLayer::soft_update_from(Object* source_obj, float tau)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] soft_update_from called before initialize.");
        return;
    }
    if (tau < 0.f || tau > 1.f) {
        UtilityFunctions::push_error("[RlLayerNormLayer] soft_update_from tau must be in [0,1].");
        return;
    }

    RlLayerNormLayer* source = Object::cast_to<RlLayerNormLayer>(source_obj);
    if (source == nullptr || !source->_ready || source->_size != _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] soft_update_from source mismatch.");
        return;
    }

    float omt = 1.f - tau;
    for (int i = 0; i < _size; ++i) {
        _gamma[i] = tau * source->_gamma[i] + omt * _gamma[i];
        _beta[i]  = tau * source->_beta[i]  + omt * _beta[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────────────────────────────────────

PackedFloat32Array RlLayerNormLayer::get_weights() const
{
    PackedFloat32Array out;
    out.resize(2 * _size);
    float* dst = out.ptrw();
    std::memcpy(dst,         _gamma.data(), _size * sizeof(float));
    std::memcpy(dst + _size, _beta.data(),  _size * sizeof(float));
    return out;
}

PackedInt32Array RlLayerNormLayer::get_shapes() const
{
    // Matches LayerNormLayer.AppendSerialized: [RLLayerKind::LayerNorm=2, size]
    PackedInt32Array out;
    out.push_back(2);      // RLLayerKind.LayerNorm
    out.push_back(_size);
    return out;
}

void RlLayerNormLayer::set_weights(const PackedFloat32Array& weights,
                                    const PackedInt32Array&   /*shapes*/)
{
    if (!_ready) {
        UtilityFunctions::push_error("[RlLayerNormLayer] set_weights called before initialize.");
        return;
    }
    if (weights.size() != 2 * _size) {
        UtilityFunctions::push_error("[RlLayerNormLayer] set_weights weight array size mismatch.");
        return;
    }

    const float* src = weights.ptr();
    std::memcpy(_gamma.data(), src,         _size * sizeof(float));
    std::memcpy(_beta.data(),  src + _size, _size * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────────────────
// GDExtension binding
// ─────────────────────────────────────────────────────────────────────────────

void RlLayerNormLayer::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("initialize", "size"),
                         &RlLayerNormLayer::initialize);

    ClassDB::bind_method(D_METHOD("forward", "input"),
                         &RlLayerNormLayer::forward);

    ClassDB::bind_method(D_METHOD("forward_batch", "flat_input", "batch_size"),
                         &RlLayerNormLayer::forward_batch);

    ClassDB::bind_method(D_METHOD("backward", "output_grad", "lr", "grad_scale"),
                         &RlLayerNormLayer::backward);

    ClassDB::bind_method(D_METHOD("create_gradient_buffer"),
                         &RlLayerNormLayer::create_gradient_buffer);

    ClassDB::bind_method(D_METHOD("accumulate_gradients", "output_grad", "grad_buffer"),
                         &RlLayerNormLayer::accumulate_gradients);

    ClassDB::bind_method(D_METHOD("accumulate_gradients_with_buffer", "output_grad", "grad_buffer"),
                         &RlLayerNormLayer::accumulate_gradients_with_buffer);

    ClassDB::bind_method(D_METHOD("apply_gradients", "grad_buffer", "lr", "grad_scale"),
                         &RlLayerNormLayer::apply_gradients);

    ClassDB::bind_method(D_METHOD("compute_input_grad", "output_grad"),
                         &RlLayerNormLayer::compute_input_grad);

    ClassDB::bind_method(D_METHOD("grad_norm_squared", "grad_buffer"),
                         &RlLayerNormLayer::grad_norm_squared);

    ClassDB::bind_method(D_METHOD("copy_weights_from", "source"),
                         &RlLayerNormLayer::copy_weights_from);

    ClassDB::bind_method(D_METHOD("soft_update_from", "source", "tau"),
                         &RlLayerNormLayer::soft_update_from);

    ClassDB::bind_method(D_METHOD("get_weights"),
                         &RlLayerNormLayer::get_weights);

    ClassDB::bind_method(D_METHOD("get_shapes"),
                         &RlLayerNormLayer::get_shapes);

    ClassDB::bind_method(D_METHOD("set_weights", "weights", "shapes"),
                         &RlLayerNormLayer::set_weights);
}
