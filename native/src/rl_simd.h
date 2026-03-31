#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Shared SIMD / math utilities for rl-agent native layers.
//
// All functions are static inline — each translation unit gets its own copy.
// This header is intended only for internal use by RlCnnEncoder, RlDenseLayer,
// and RlLayerNormLayer.
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef RL_USE_AVX2
#  include <immintrin.h>
#endif

// ── Adam hyper-parameters ─────────────────────────────────────────────────────

static constexpr float kB1  = 0.9f;
static constexpr float kB2  = 0.999f;
static constexpr float kEps = 1e-8f;

// ── PCG32 PRNG + Box-Muller He init ──────────────────────────────────────────

struct Pcg32 {
    uint64_t state, inc;
    explicit Pcg32(uint64_t seed, uint64_t seq = 1)
        : state(0), inc((seq << 1) | 1)
    { next(); state += seed; next(); }

    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xsh = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xsh >> rot) | (xsh << ((-rot) & 31));
    }

    // Box-Muller normal sample
    float next_gaussian() {
        float u1 = std::max((next() >> 8) * (1.f / (1 << 24)), 1e-7f);
        float u2  = (next() >> 8) * (1.f / (1 << 24));
        return std::sqrt(-2.f * std::log(u1)) *
               std::cos(2.f * 3.14159265358979f * u2);
    }
};

// ── dot_avx: dot product of two float vectors, length K ──────────────────────
// Uses AVX2 FMA when available (2× unrolled to hide FMA latency), scalar fallback.

static inline float dot_avx(const float* __restrict__ a,
                             const float* __restrict__ b,
                             int K)
{
    float sum = 0.f;
    int k = 0;

#ifdef RL_USE_AVX2
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (; k + 16 <= K; k += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),
                               _mm256_loadu_ps(b + k), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8),
                               _mm256_loadu_ps(b + k + 8), acc1);
    }
    for (; k + 8 <= K; k += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),
                               _mm256_loadu_ps(b + k), acc0);

    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 s4  = _mm_add_ps(lo, hi);
    __m128 s2  = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1  = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    sum = _mm_cvtss_f32(s1);
#endif

    for (; k < K; ++k)
        sum += a[k] * b[k];
    return sum;
}

// ── axpy: dst[i] += src[i] * scale, length N ─────────────────────────────────

static inline void axpy(float* __restrict__ dst,
                        const float* __restrict__ src,
                        float scale, int N)
{
    int i = 0;
#ifdef RL_USE_AVX2
    __m256 vs = _mm256_set1_ps(scale);
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(dst + i,
            _mm256_fmadd_ps(_mm256_loadu_ps(src + i), vs,
                            _mm256_loadu_ps(dst + i)));
#endif
    for (; i < N; ++i)
        dst[i] += src[i] * scale;
}

// ── adam_update: fast (approximate) Adam used by RlCnnEncoder ────────────────
// lr_corrected = lr * sqrt(1 - b2t) / (1 - b1t)
// Approximation: uses sqrt(v) in denominator instead of sqrt(v / (1-b2t)).
// Equivalent to exact Adam when v >> eps (i.e. after warmup).

static inline void adam_update(float* params,
                                float* m, float* v,
                                const float* grads,
                                int N, float lr_corrected, float scale)
{
    int i = 0;
#ifdef RL_USE_AVX2
    __m256 vb1   = _mm256_set1_ps(kB1);
    __m256 vb1c  = _mm256_set1_ps(1.f - kB1);
    __m256 vb2   = _mm256_set1_ps(kB2);
    __m256 vb2c  = _mm256_set1_ps(1.f - kB2);
    __m256 veps  = _mm256_set1_ps(kEps);
    __m256 vlr   = _mm256_set1_ps(lr_corrected);
    __m256 vsc   = _mm256_set1_ps(scale);

    for (; i + 8 <= N; i += 8) {
        __m256 g  = _mm256_mul_ps(_mm256_loadu_ps(grads + i), vsc);
        __m256 mi = _mm256_fmadd_ps(vb1c, g, _mm256_mul_ps(vb1, _mm256_loadu_ps(m + i)));
        __m256 vi = _mm256_fmadd_ps(vb2c, _mm256_mul_ps(g, g),
                                    _mm256_mul_ps(vb2, _mm256_loadu_ps(v + i)));
        _mm256_storeu_ps(m + i, mi);
        _mm256_storeu_ps(v + i, vi);
        __m256 update = _mm256_div_ps(
            _mm256_mul_ps(vlr, mi),
            _mm256_add_ps(_mm256_sqrt_ps(vi), veps));
        _mm256_storeu_ps(params + i,
            _mm256_sub_ps(_mm256_loadu_ps(params + i), update));
    }
#endif
    for (; i < N; ++i) {
        float g  = grads[i] * scale;
        m[i]     = kB1 * m[i] + (1.f - kB1) * g;
        v[i]     = kB2 * v[i] + (1.f - kB2) * g * g;
        params[i] -= lr_corrected * m[i] / (std::sqrt(v[i]) + kEps);
    }
}

// ── adam_update_exact: standard bias-corrected Adam matching DenseLayer.cs ───
// Uses the exact formula: params -= lr * m_hat / (sqrt(v_hat) + eps)
// where m_hat = m / (1 - b1t), v_hat = v / (1 - b2t).
// b1t and b2t are the ACCUMULATED accumulators (already advanced for this step).

static inline void adam_update_exact(float* params,
                                      float* m, float* v,
                                      const float* grads,
                                      int N, float lr, float scale,
                                      float b1t, float b2t)
{
    float b1_corr = 1.f - b1t;
    float b2_corr = 1.f - b2t;
    int i = 0;
#ifdef RL_USE_AVX2
    __m256 vb1    = _mm256_set1_ps(kB1);
    __m256 vb1c   = _mm256_set1_ps(1.f - kB1);
    __m256 vb2    = _mm256_set1_ps(kB2);
    __m256 vb2c   = _mm256_set1_ps(1.f - kB2);
    __m256 veps   = _mm256_set1_ps(kEps);
    __m256 vlr    = _mm256_set1_ps(lr);
    __m256 vsc    = _mm256_set1_ps(scale);
    __m256 vb1corr_inv = _mm256_set1_ps(1.f / b1_corr);
    __m256 vb2corr     = _mm256_set1_ps(b2_corr);

    for (; i + 8 <= N; i += 8) {
        __m256 g  = _mm256_mul_ps(_mm256_loadu_ps(grads + i), vsc);
        __m256 mi = _mm256_fmadd_ps(vb1c, g, _mm256_mul_ps(vb1, _mm256_loadu_ps(m + i)));
        __m256 vi = _mm256_fmadd_ps(vb2c, _mm256_mul_ps(g, g),
                                    _mm256_mul_ps(vb2, _mm256_loadu_ps(v + i)));
        _mm256_storeu_ps(m + i, mi);
        _mm256_storeu_ps(v + i, vi);
        // m_hat = mi / b1_corr, v_hat = vi / b2_corr
        __m256 m_hat  = _mm256_mul_ps(mi, vb1corr_inv);
        __m256 v_hat  = _mm256_div_ps(vi, vb2corr);
        __m256 update = _mm256_div_ps(
            _mm256_mul_ps(vlr, m_hat),
            _mm256_add_ps(_mm256_sqrt_ps(v_hat), veps));
        _mm256_storeu_ps(params + i,
            _mm256_sub_ps(_mm256_loadu_ps(params + i), update));
    }
#endif
    for (; i < N; ++i) {
        float g  = grads[i] * scale;
        m[i]     = kB1 * m[i] + (1.f - kB1) * g;
        v[i]     = kB2 * v[i] + (1.f - kB2) * g * g;
        float m_hat = m[i] / b1_corr;
        float v_hat = v[i] / b2_corr;
        params[i] -= lr * m_hat / (std::sqrt(v_hat) + kEps);
    }
}
