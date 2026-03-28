#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// WHT-TQ quantization functions
#include "wht.cuh"

static __device__ void quantize_f32_tq3_0_wht_block(const float * __restrict__ x, block_tq3_0 * __restrict__ y) {
    float rotated[256];

    // Copy and apply WHT rotation to each sub-block of 32
    for (int sb = 0; sb < 8; sb++) {
        for (int j = 0; j < 32; j++) rotated[sb*32+j] = x[sb*32+j];
        wht_rotate_forward_32(rotated + sb*32, WHT_SEED_D, WHT_SEED_D_PRIME, sb);
    }

    // Find amax of rotated data
    float amax = 0.0f;
    for (int j = 0; j < 256; j++) {
        float ax = fabsf(rotated[j]);
        if (ax > amax) amax = ax;
    }
    y->d = __float2half(amax);
    float inv_d = amax > 0.0f ? 1.0f / amax : 0.0f;

    // Clear arrays
    memset(y->al, 0, sizeof(y->al));
    memset(y->signs, 0, sizeof(y->signs));

    // Quantize each element to nearest WHT codebook entry
    for (int j = 0; j < 256; j++) {
        float normalized = rotated[j] * inv_d;
        float abs_norm = fabsf(normalized);
        int sign = (normalized < 0.0f) ? 1 : 0;

        // Find nearest codebook entry (4 entries for TQ3)
        int best = 0;
        float best_err = 1e30f;
        for (int g = 0; g < 4; g++) {
            float err = (abs_norm - WHT_CODEBOOK_4[g]) * (abs_norm - WHT_CODEBOOK_4[g]);
            if (err < best_err) { best_err = err; best = g; }
        }

        y->al[j/4] |= (uint8_t)((best & 0x3) << ((j%4)*2));
        if (sign) y->signs[j/8] |= (uint8_t)(1 << (j%8));
    }
}

static __device__ void quantize_f32_tq4_0_wht_block(const float * __restrict__ x, block_tq4_0 * __restrict__ y) {
    float rotated[256];

    // Copy and apply WHT rotation to each sub-block of 32
    for (int sb = 0; sb < 8; sb++) {
        for (int j = 0; j < 32; j++) rotated[sb*32+j] = x[sb*32+j];
        wht_rotate_forward_32(rotated + sb*32, WHT_SEED_D, WHT_SEED_D_PRIME, sb);
    }

    // Find amax of rotated data
    float amax = 0.0f;
    for (int j = 0; j < 256; j++) {
        float ax = fabsf(rotated[j]);
        if (ax > amax) amax = ax;
    }
    y->d = __float2half(amax);
    float inv_d = amax > 0.0f ? 1.0f / amax : 0.0f;

    // Clear arrays
    memset(y->al, 0, sizeof(y->al));
    memset(y->ah, 0, sizeof(y->ah));
    memset(y->signs, 0, sizeof(y->signs));

    // Quantize each element to nearest WHT codebook entry
    for (int j = 0; j < 256; j++) {
        float normalized = rotated[j] * inv_d;
        float abs_norm = fabsf(normalized);
        int sign = (normalized < 0.0f) ? 1 : 0;

        // Find nearest codebook entry (8 entries for TQ4)
        int best = 0;
        float best_err = 1e30f;
        for (int g = 0; g < 8; g++) {
            float err = (abs_norm - WHT_CODEBOOK_8[g]) * (abs_norm - WHT_CODEBOOK_8[g]);
            if (err < best_err) { best_err = err; best = g; }
        }

        // 3-bit index split: 2-bit lo in al, 1-bit hi in ah
        y->al[j/4] |= (uint8_t)((best & 0x3) << ((j%4)*2));
        if (best & 0x4) y->ah[j/8] |= (uint8_t)(1 << (j%8));
        if (sign) y->signs[j/8] |= (uint8_t)(1 << (j%8));
    }
}

// TQ3_KV: animehacker 3-bit KV cache quantize
static __constant__ const float TQ3_KV_CENTROIDS_D[8] = {
    -2.1573f, -1.3336f, -0.7434f, -0.2428f,
     0.2428f,  0.7434f,  1.3336f,  2.1573f
};

static __constant__ const int8_t TQ3_KV_SIGNS_D[32] = {
    +1,-1,+1,+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,+1,-1,-1,
    +1,-1,-1,+1,+1,-1,+1,-1,-1,+1,+1,+1,-1,-1,+1,-1
};

static __device__ void tq3_wht32_forward_device(float data[32]) {
    // Apply diagonal signs
    for (int j = 0; j < 32; j++) data[j] *= TQ3_KV_SIGNS_D[j];
    // 5-stage butterfly
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step * 2) {
            for (int j = i; j < i + step; j++) {
                float a = data[j], b = data[j + step];
                data[j] = a + b; data[j + step] = a - b;
            }
        }
    }
    // Normalize: 1/sqrt(32)
    for (int j = 0; j < 32; j++) data[j] *= 0.17677669529663688f;
}

static __device__ void quantize_f32_tq3_kv_block(const float * __restrict__ x, block_tq3_kv * __restrict__ y) {
    float rotated[32];
    for (int j = 0; j < 32; j++) rotated[j] = x[j];
    tq3_wht32_forward_device(rotated);

    // Find amax
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
        float ax = fabsf(rotated[j]);
        if (ax > amax) amax = ax;
    }

    float d = amax / 2.1573f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    y->gamma = __float2half(d);

    // Clear arrays
    memset(y->qs, 0, sizeof(y->qs));
    memset(y->qr, 0, sizeof(y->qr));

    // Quantize with threshold-based centroid assignment
    for (int j = 0; j < 32; j++) {
        float xn = rotated[j] * id;
        int idx;
        if      (xn < -1.7455f) idx = 0;
        else if (xn < -1.0385f) idx = 1;
        else if (xn < -0.4931f) idx = 2;
        else if (xn <  0.0f)    idx = 3;
        else if (xn <  0.4931f) idx = 4;
        else if (xn <  1.0385f) idx = 5;
        else if (xn <  1.7455f) idx = 6;
        else                    idx = 7;
        y->qs[j/4] |= (uint8_t)((idx & 3) << (2*(j%4)));
        y->qr[j/8] |= (uint8_t)(((idx >> 2) & 1) << (j%8));
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}

static __device__ void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t * src = (const int32_t *)cxi;
    int32_t * dst = (int32_t *)cdsti;
    *dst = *src;
}
