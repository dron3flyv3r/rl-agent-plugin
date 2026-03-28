namespace RlAgentPlugin.Runtime;

/// <summary>
/// Embedded GLSL compute shader sources.
/// Shaders are stored as C# string constants to avoid file I/O at runtime
/// and to ensure they are always bundled with the plugin.
///
/// All shaders target GLSL 4.50 / Vulkan semantics (no gl_FragCoord etc.).
/// Push-constant structs use std430 layout throughout.
/// </summary>
internal static class GpuShaderSources
{
    public const string ReluBackward = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer OutputGradBuffer {
    float data[];
} output_grad_buf;

layout(set = 0, binding = 1, std430) readonly buffer PreActBuffer {
    float data[];
} preact_buf;

layout(set = 0, binding = 2, std430) writeonly buffer GatedGradBuffer {
    float data[];
} gated_grad_buf;

layout(push_constant, std430) uniform PushConstants {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.count) return;
    gated_grad_buf.data[idx] = preact_buf.data[idx] > 0.0 ? output_grad_buf.data[idx] : 0.0;
}
";

    public const string ConvForward = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float data[];
} input_buf;

layout(set = 0, binding = 1, std430) readonly buffer FilterBuffer {
    float data[];
} filter_buf;

layout(set = 0, binding = 2, std430) readonly buffer BiasBuffer {
    float data[];
} bias_buf;

layout(set = 0, binding = 3, std430) writeonly buffer PreActBuffer {
    float data[];
} preact_buf;

layout(set = 0, binding = 4, std430) writeonly buffer OutputBuffer {
    float data[];
} output_buf;

layout(push_constant, std430) uniform PushConstants {
    uint in_h;
    uint in_w;
    uint in_c;
    uint out_h;
    uint out_w;
    uint out_c;
    uint kernel;
    uint stride;
    uint batch_size;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint sample_out_count = pc.out_h * pc.out_w * pc.out_c;
    uint total = pc.batch_size * sample_out_count;
    if (idx >= total) return;

    uint batch_idx = idx / sample_out_count;
    uint local_idx = idx - batch_idx * sample_out_count;
    uint spatial = local_idx / pc.out_c;
    uint oc = local_idx - spatial * pc.out_c;
    uint oh = spatial / pc.out_w;
    uint ow = spatial - oh * pc.out_w;
    uint input_offset = batch_idx * (pc.in_h * pc.in_w * pc.in_c);
    uint output_offset = batch_idx * sample_out_count;

    float sum = bias_buf.data[oc];
    for (uint kh = 0; kh < pc.kernel; ++kh) {
        uint ih = oh * pc.stride + kh;
        for (uint kw = 0; kw < pc.kernel; ++kw) {
            uint iw = ow * pc.stride + kw;
            uint input_base = input_offset + (ih * pc.in_w + iw) * pc.in_c;
            uint filter_base = (((oc * pc.kernel) + kh) * pc.kernel + kw) * pc.in_c;
            for (uint ic = 0; ic < pc.in_c; ++ic)
                sum += input_buf.data[input_base + ic] * filter_buf.data[filter_base + ic];
        }
    }

    preact_buf.data[output_offset + local_idx] = sum;
    output_buf.data[output_offset + local_idx] = max(sum, 0.0);
}
";

    public const string ConvBackwardFilter = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float data[];
} input_buf;

layout(set = 0, binding = 1, std430) readonly buffer GatedGradBuffer {
    float data[];
} gated_grad_buf;

layout(set = 0, binding = 2, std430) buffer FilterGradBuffer {
    float data[];
} filter_grad_buf;

layout(set = 0, binding = 3, std430) buffer BiasGradBuffer {
    float data[];
} bias_grad_buf;

layout(push_constant, std430) uniform PushConstants {
    uint in_h;
    uint in_w;
    uint in_c;
    uint out_h;
    uint out_w;
    uint out_c;
    uint kernel;
    uint stride;
    uint batch_size;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint filter_count = pc.out_c * pc.kernel * pc.kernel * pc.in_c;

    if (idx < filter_count) {
        uint tmp = idx;
        uint ic = tmp % pc.in_c;
        tmp /= pc.in_c;
        uint kw = tmp % pc.kernel;
        tmp /= pc.kernel;
        uint kh = tmp % pc.kernel;
        uint oc = tmp / pc.kernel;

        float sum = 0.0;
        uint sample_input_count = pc.in_h * pc.in_w * pc.in_c;
        uint sample_grad_count = pc.out_h * pc.out_w * pc.out_c;
        for (uint batch_idx = 0u; batch_idx < pc.batch_size; ++batch_idx) {
            uint input_offset = batch_idx * sample_input_count;
            uint grad_offset = batch_idx * sample_grad_count;
            for (uint oh = 0; oh < pc.out_h; ++oh) {
                uint ih = oh * pc.stride + kh;
                for (uint ow = 0; ow < pc.out_w; ++ow) {
                    uint iw = ow * pc.stride + kw;
                    uint input_idx = input_offset + (ih * pc.in_w + iw) * pc.in_c + ic;
                    uint grad_idx = grad_offset + (oh * pc.out_w + ow) * pc.out_c + oc;
                    sum += gated_grad_buf.data[grad_idx] * input_buf.data[input_idx];
                }
            }
        }
        filter_grad_buf.data[idx] += sum;
    }

    if (idx < pc.out_c) {
        float bias_sum = 0.0;
        uint sample_grad_count = pc.out_h * pc.out_w * pc.out_c;
        for (uint batch_idx = 0u; batch_idx < pc.batch_size; ++batch_idx) {
            uint grad_offset = batch_idx * sample_grad_count;
            for (uint oh = 0; oh < pc.out_h; ++oh)
                for (uint ow = 0; ow < pc.out_w; ++ow)
                    bias_sum += gated_grad_buf.data[grad_offset + (oh * pc.out_w + ow) * pc.out_c + idx];
        }
        bias_grad_buf.data[idx] += bias_sum;
    }
}
";

    public const string ConvBackwardInput = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer GatedGradBuffer {
    float data[];
} gated_grad_buf;

layout(set = 0, binding = 1, std430) readonly buffer FilterBuffer {
    float data[];
} filter_buf;

layout(set = 0, binding = 2, std430) writeonly buffer InputGradBuffer {
    float data[];
} input_grad_buf;

layout(push_constant, std430) uniform PushConstants {
    uint in_h;
    uint in_w;
    uint in_c;
    uint out_h;
    uint out_w;
    uint out_c;
    uint kernel;
    uint stride;
    uint batch_size;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint sample_input_count = pc.in_h * pc.in_w * pc.in_c;
    uint input_count = pc.batch_size * sample_input_count;
    if (idx >= input_count) return;

    uint batch_idx = idx / sample_input_count;
    uint local_idx = idx - batch_idx * sample_input_count;
    uint tmp = local_idx;
    uint ic = tmp % pc.in_c;
    tmp /= pc.in_c;
    uint iw = tmp % pc.in_w;
    uint ih = tmp / pc.in_w;
    uint grad_offset = batch_idx * (pc.out_h * pc.out_w * pc.out_c);

    float sum = 0.0;
    for (uint oc = 0; oc < pc.out_c; ++oc) {
        for (uint kh = 0; kh < pc.kernel; ++kh) {
            if (ih < kh) continue;
            uint dh = ih - kh;
            if ((dh % pc.stride) != 0u) continue;
            uint oh = dh / pc.stride;
            if (oh >= pc.out_h) continue;

            for (uint kw = 0; kw < pc.kernel; ++kw) {
                if (iw < kw) continue;
                uint dw = iw - kw;
                if ((dw % pc.stride) != 0u) continue;
                uint ow = dw / pc.stride;
                if (ow >= pc.out_w) continue;

                uint grad_idx = grad_offset + (oh * pc.out_w + ow) * pc.out_c + oc;
                uint filter_idx = (((oc * pc.kernel) + kh) * pc.kernel + kw) * pc.in_c + ic;
                sum += gated_grad_buf.data[grad_idx] * filter_buf.data[filter_idx];
            }
        }
    }

    input_grad_buf.data[idx] = sum;
}
";

    public const string LinearForward = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float data[];
} input_buf;

layout(set = 0, binding = 1, std430) readonly buffer WeightBuffer {
    float data[];
} weight_buf;

layout(set = 0, binding = 2, std430) readonly buffer BiasBuffer {
    float data[];
} bias_buf;

layout(set = 0, binding = 3, std430) writeonly buffer OutputBuffer {
    float data[];
} output_buf;

layout(push_constant, std430) uniform PushConstants {
    uint input_size;
    uint output_size;
    uint apply_relu;
    uint batch_size;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = pc.batch_size * pc.output_size;
    if (idx >= total) return;

    uint batch_idx = idx / pc.output_size;
    uint oi = idx - batch_idx * pc.output_size;
    uint input_offset = batch_idx * pc.input_size;
    uint output_offset = batch_idx * pc.output_size;

    float sum = bias_buf.data[oi];
    uint wBase = oi * pc.input_size;
    for (uint ii = 0; ii < pc.input_size; ++ii)
        sum += input_buf.data[input_offset + ii] * weight_buf.data[wBase + ii];

    output_buf.data[output_offset + oi] = pc.apply_relu != 0u ? max(sum, 0.0) : sum;
}
";

    public const string LinearBackward = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float data[];
} input_buf;

layout(set = 0, binding = 1, std430) readonly buffer WeightBuffer {
    float data[];
} weight_buf;

layout(set = 0, binding = 2, std430) readonly buffer OutputGradBuffer {
    float data[];
} output_grad_buf;

layout(set = 0, binding = 3, std430) writeonly buffer InputGradBuffer {
    float data[];
} input_grad_buf;

layout(set = 0, binding = 4, std430) buffer WeightGradBuffer {
    float data[];
} weight_grad_buf;

layout(set = 0, binding = 5, std430) buffer BiasGradBuffer {
    float data[];
} bias_grad_buf;

layout(push_constant, std430) uniform PushConstants {
    uint input_size;
    uint output_size;
    uint batch_size;
    uint thread_count;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.thread_count) return;

    uint total_input_count = pc.batch_size * pc.input_size;
    uint weight_count = pc.output_size * pc.input_size;

    if (idx < total_input_count) {
        uint batch_idx = idx / pc.input_size;
        uint ii = idx - batch_idx * pc.input_size;
        uint input_offset = batch_idx * pc.input_size;
        uint output_offset = batch_idx * pc.output_size;
        float x = input_buf.data[input_offset + ii];
        float input_grad = 0.0;
        for (uint oi = 0; oi < pc.output_size; ++oi) {
            uint wi = oi * pc.input_size + ii;
            float go = output_grad_buf.data[output_offset + oi];
            input_grad += weight_buf.data[wi] * go;
        }
        input_grad_buf.data[idx] = input_grad;
    }

    if (idx < weight_count) {
        uint oi = idx / pc.input_size;
        uint ii = idx - oi * pc.input_size;
        float sum = 0.0;
        for (uint batch_idx = 0u; batch_idx < pc.batch_size; ++batch_idx)
            sum += output_grad_buf.data[batch_idx * pc.output_size + oi] * input_buf.data[batch_idx * pc.input_size + ii];
        weight_grad_buf.data[idx] += sum;
    }

    if (idx < pc.output_size) {
        float bias_sum = 0.0;
        for (uint batch_idx = 0u; batch_idx < pc.batch_size; ++batch_idx)
            bias_sum += output_grad_buf.data[batch_idx * pc.output_size + idx];
        bias_grad_buf.data[idx] += bias_sum;
    }
}
";

    public const string AdamUpdate = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer ParamBuffer {
    float data[];
} param_buf;

layout(set = 0, binding = 1, std430) buffer GradBuffer {
    float data[];
} grad_buf;

layout(set = 0, binding = 2, std430) buffer Moment1Buffer {
    float data[];
} moment1_buf;

layout(set = 0, binding = 3, std430) buffer Moment2Buffer {
    float data[];
} moment2_buf;

layout(push_constant, std430) uniform PushConstants {
    uint count;
    float learning_rate;
    float grad_scale;
    float b1_corr;
    float b2_corr;
    float beta1;
    float beta2;
    float epsilon;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.count) return;

    float g = grad_buf.data[idx] * pc.grad_scale;
    float m1 = pc.beta1 * moment1_buf.data[idx] + (1.0 - pc.beta1) * g;
    float m2 = pc.beta2 * moment2_buf.data[idx] + (1.0 - pc.beta2) * g * g;

    moment1_buf.data[idx] = m1;
    moment2_buf.data[idx] = m2;
    param_buf.data[idx] -= pc.learning_rate * (m1 / pc.b1_corr) / (sqrt(m2 / pc.b2_corr) + pc.epsilon);
    grad_buf.data[idx] = 0.0;
}
";

    public const string AdamUpdateNative = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer ParamBuffer {
    float data[];
} param_buf;

layout(set = 0, binding = 1, std430) buffer GradBuffer {
    float data[];
} grad_buf;

layout(set = 0, binding = 2, std430) buffer Moment1Buffer {
    float data[];
} moment1_buf;

layout(set = 0, binding = 3, std430) buffer Moment2Buffer {
    float data[];
} moment2_buf;

layout(push_constant, std430) uniform PushConstants {
    uint count;
    float learning_rate_corrected;
    float grad_scale;
    float beta1;
    float beta2;
    float epsilon;
    uint _pad0;
    uint _pad1;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.count) return;

    float g = grad_buf.data[idx] * pc.grad_scale;
    float m1 = pc.beta1 * moment1_buf.data[idx] + (1.0 - pc.beta1) * g;
    float m2 = pc.beta2 * moment2_buf.data[idx] + (1.0 - pc.beta2) * g * g;

    moment1_buf.data[idx] = m1;
    moment2_buf.data[idx] = m2;
    param_buf.data[idx] -= pc.learning_rate_corrected * m1 / (sqrt(m2) + pc.epsilon);
    grad_buf.data[idx] = 0.0;
}
";

    /// <summary>
    /// Single-workgroup parallel reduction that accumulates the sum of squares of a
    /// gradient buffer into a shared 1-float accumulator.
    ///
    /// Always dispatch with exactly 1 workgroup (elementCount = 1 in DispatchToList).
    /// The stride loop handles buffers of any size.  Insert AddBarrier between
    /// successive calls on different buffers so each write to the accumulator is
    /// visible before the next dispatch reads it.
    ///
    /// Binding 0: gradient buffer (read-only)
    /// Binding 1: 1-float accumulator (read-write, += sdata[0] from thread 0)
    /// Push constant: count — number of floats in the gradient buffer
    /// </summary>
    public const string NormSquaredAccumulate = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer GradBuf { float data[]; } grad;
layout(set = 0, binding = 1, std430) buffer NormBuf { float result; } norm;

layout(push_constant, std430) uniform PC {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} pc;

shared float sdata[256];

void main()
{
    uint tid = gl_LocalInvocationID.x;
    float s = 0.0;
    for (uint i = tid; i < pc.count; i += 256u)
    {
        float v = grad.data[i];
        s += v * v;
    }
    sdata[tid] = s;

    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        barrier();
    }

    if (tid == 0u)
        norm.result += sdata[0];
}
";

    /// Passthrough — copies input buffer to output buffer unchanged.
    /// Used as a smoke test to verify the GPU pipeline is functional.
    public const string Passthrough = @"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float data[];
} input_buf;

layout(set = 0, binding = 1, std430) writeonly buffer OutputBuffer {
    float data[];
} output_buf;

layout(push_constant, std430) uniform PushConstants {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.count) return;
    output_buf.data[idx] = input_buf.data[idx];
}
";
}
