// TurboQuant block structures and lookup tables
// PolarQuant (angular grid quantization) + QJL (1-bit sign error correction)
//
// Reference: TurboQuant (ICLR 2026), PolarQuant (AISTATS 2026), QJL (AAAI)

#pragma once

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifndef ggml_half
typedef unsigned short ggml_half;
#endif

#ifndef GGML_FP32_TO_FP16
// Simplified fp16 conversion for standalone use
static inline ggml_half ggml_fp32_to_fp16_simple(float f) {
    union { float f; unsigned int u; } fu;
    fu.f = f;
    unsigned int u = fu.u;
    unsigned int sign = (u >> 16) & 0x8000;
    int exp = ((u >> 23) & 0xFF) - 127 + 15;
    unsigned int mant = (u >> 13) & 0x3FF;
    if (exp <= 0) return (ggml_half)sign;
    if (exp >= 31) return (ggml_half)(sign | 0x7C00);
    return (ggml_half)(sign | ((unsigned int)exp << 10) | mant);
}
static inline float ggml_fp16_to_fp32_simple(ggml_half h) {
    unsigned int sign = ((unsigned int)h & 0x8000) << 16;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    if (exp == 0) return 0.0f;
    exp = exp - 15 + 127;
    union { unsigned int u; float f; } r;
    r.u = sign | (exp << 23) | (mant << 13);
    return r.f;
}
#define GGML_FP32_TO_FP16(x) ggml_fp32_to_fp16_simple(x)
#define GGML_FP16_TO_FP32(x) ggml_fp16_to_fp32_simple(x)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// Block size constants
// =====================================================================

// Both TQ types use 256-element super-blocks (consistent with K-quants)
#define QK_TQ3 256
#define QK_TQ4 256

// GPU kernel constants (defined in CUDA/HIP sections of ggml-common.h)
// QR = QK / number of values before dequantization
// QI = number of 32-bit integers before dequantization
//
// For TQ3_0: each element is 3 bits (2-bit angle + 1-bit sign)
//   QR = 256 / (256/4) = 4  (4 elements decoded per byte of angle data)
//   QI = QK / (4 * QR) = 256 / 16 = 16
//
// For TQ4_0: each element is 4 bits (3-bit angle + 1-bit sign)
//   QR = 2 (2 elements decoded per byte of lo-angle data)
//   QI = QK / (4 * QR) = 256 / 8 = 32

#define QR_TQ3 4
#define QI_TQ3 (QK_TQ3 / (4 * QR_TQ3))   // = 16

#define QR_TQ4 2
#define QI_TQ4 (QK_TQ4 / (4 * QR_TQ4))   // = 32

// =====================================================================
// Polar angle grid lookup tables
// =====================================================================

// 4-entry grid for TQ3_0 (2-bit angle index)
// Uniform centroids in [0, 1] for amax-normalized quantization.
// All positive; the QJL sign bit handles negation (8 effective levels).
static const float TQ_POLAR_GRID_4[4] = {
    1.0f/8.0f,   // 0.125  — centroid of [0, 1/4)
    3.0f/8.0f,   // 0.375  — centroid of [1/4, 2/4)
    5.0f/8.0f,   // 0.625  — centroid of [2/4, 3/4)
    7.0f/8.0f,   // 0.875  — centroid of [3/4, 1]
};

// 8-entry grid for TQ4_0 (3-bit angle index)
// Uniform centroids in [0, 1] for amax-normalized quantization.
// All positive; the QJL sign bit handles negation (16 effective levels).
static const float TQ_POLAR_GRID_8[8] = {
    1.0f/16.0f,   // 0.0625  — centroid of [0, 1/8)
    3.0f/16.0f,   // 0.1875  — centroid of [1/8, 2/8)
    5.0f/16.0f,   // 0.3125  — centroid of [2/8, 3/8)
    7.0f/16.0f,   // 0.4375  — centroid of [3/8, 4/8)
    9.0f/16.0f,   // 0.5625  — centroid of [4/8, 5/8)
    11.0f/16.0f,  // 0.6875  — centroid of [5/8, 6/8)
    13.0f/16.0f,  // 0.8125  — centroid of [6/8, 7/8)
    15.0f/16.0f,  // 0.9375  — centroid of [7/8, 1]
};

// =====================================================================
// block_tq3_0 — ~3.0625 bpw
//
// PolarQuant with 4-entry angular grid + QJL 1-bit sign correction.
//
// Per element: 2-bit angle index + 1-bit sign = 3 bits
// Overhead: fp16 radius per 256 elements = 0.0625 bpw
//
// Dequantization:
//   x[i] = d * TQ_POLAR_GRID_4[al[i]] * (1 - 2*sign[i])
//
// Memory layout (98 bytes):
//   [d: 2B] [al: 64B (2-bit packed)] [signs: 32B (1-bit packed)]
// =====================================================================

typedef struct {
    ggml_half d;                     // super-block scale (absolute max)
    uint8_t   al[QK_TQ3 / 4];       // 2-bit angle grid indices, packed 4 per byte
                                     // al[i] byte contains indices for elements 4*i .. 4*i+3
                                     // bits [1:0] = elem 4*i, [3:2] = elem 4*i+1, etc.
    uint8_t   signs[QK_TQ3 / 8];    // 1-bit QJL sign corrections, packed 8 per byte
                                     // bit j of signs[i] = sign for element 8*i+j
} block_tq3_0;

static_assert(sizeof(block_tq3_0) == sizeof(ggml_half) + QK_TQ3/4 + QK_TQ3/8,
              "wrong tq3_0 block size/padding");
// 2 + 64 + 32 = 98 bytes for 256 values = 3.0625 bpw

// =====================================================================
// block_tq4_0 — ~4.0625 bpw
//
// PolarQuant with 8-entry angular grid + QJL 1-bit sign correction.
//
// Per element: 3-bit angle index (2 lo + 1 hi) + 1-bit sign = 4 bits
// Overhead: fp16 radius per 256 elements = 0.0625 bpw
//
// The 3-bit angle index is split into lo (2-bit) and hi (1-bit) arrays,
// following the Q5_0/Q5_1 pattern for GPU-friendly memory access.
//
// Dequantization:
//   idx = al[i] | (ah[i] << 2)
//   x[i] = d * TQ_POLAR_GRID_8[idx] * (1 - 2*sign[i])
//
// Memory layout (130 bytes):
//   [d: 2B] [al: 64B (2-bit lo)] [ah: 32B (1-bit hi)] [signs: 32B (1-bit)]
// =====================================================================

typedef struct {
    ggml_half d;                     // super-block scale (absolute max)
    uint8_t   al[QK_TQ4 / 4];       // 2-bit angle grid indices (low bits), packed 4 per byte
                                     // bits [1:0] = elem 4*i, [3:2] = elem 4*i+1, etc.
    uint8_t   ah[QK_TQ4 / 8];       // 1-bit angle grid index (high bit), packed 8 per byte
                                     // bit j of ah[i] = high bit for element 8*i+j
    uint8_t   signs[QK_TQ4 / 8];    // 1-bit QJL sign corrections, packed 8 per byte
                                     // bit j of signs[i] = sign for element 8*i+j
} block_tq4_0;

static_assert(sizeof(block_tq4_0) == sizeof(ggml_half) + QK_TQ4/4 + QK_TQ4/8 + QK_TQ4/8,
              "wrong tq4_0 block size/padding");
// 2 + 64 + 32 + 32 = 130 bytes for 256 values = 4.0625 bpw

// =====================================================================
// Quantization helpers
// =====================================================================

// Find the closest polar grid entry for a normalized value
static inline int tq3_find_nearest_angle(float normalized_val) {
    int best = 0;
    float best_dist = fabsf(normalized_val - TQ_POLAR_GRID_4[0]);
    for (int i = 1; i < 4; i++) {
        float dist = fabsf(normalized_val - TQ_POLAR_GRID_4[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

static inline int tq4_find_nearest_angle(float normalized_val) {
    int best = 0;
    float best_dist = fabsf(normalized_val - TQ_POLAR_GRID_8[0]);
    for (int i = 1; i < 8; i++) {
        float dist = fabsf(normalized_val - TQ_POLAR_GRID_8[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

// =====================================================================
// Bit packing helpers
// =====================================================================

// Pack a 2-bit value into the angle-lo array
//   byte_idx = elem / 4,  shift = (elem % 4) * 2
static inline void tq_pack_2bit(uint8_t * al, int elem, uint8_t val) {
    int byte_idx = elem / 4;
    int shift    = (elem % 4) * 2;
    al[byte_idx] = (al[byte_idx] & ~(0x3 << shift)) | ((val & 0x3) << shift);
}

static inline uint8_t tq_unpack_2bit(const uint8_t * al, int elem) {
    int byte_idx = elem / 4;
    int shift    = (elem % 4) * 2;
    return (al[byte_idx] >> shift) & 0x3;
}

// Pack/unpack a 1-bit value (for ah[] and signs[])
static inline void tq_pack_1bit(uint8_t * arr, int elem, uint8_t val) {
    int byte_idx = elem / 8;
    int bit_idx  = elem % 8;
    if (val) {
        arr[byte_idx] |=  (1 << bit_idx);
    } else {
        arr[byte_idx] &= ~(1 << bit_idx);
    }
}

static inline uint8_t tq_unpack_1bit(const uint8_t * arr, int elem) {
    int byte_idx = elem / 8;
    int bit_idx  = elem % 8;
    return (arr[byte_idx] >> bit_idx) & 0x1;
}

// =====================================================================
// Reference quantize / dequantize (scalar, for validation)
// =====================================================================

// Quantize a block of QK_TQ3 floats into block_tq3_0
static inline void quantize_block_tq3_0(const float * x, block_tq3_0 * block) {
    // Step 1: Compute block absolute max
    float amax = 0.0f;
    for (int i = 0; i < QK_TQ3; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) amax = ax;
    }
    block->d = GGML_FP32_TO_FP16(amax);

    float inv_d = amax > 0.0f ? 1.0f / amax : 0.0f;

    // Zero out arrays
    memset(block->al, 0, sizeof(block->al));
    memset(block->signs, 0, sizeof(block->signs));

    // Step 2: For each element, find nearest grid entry + sign correction
    for (int i = 0; i < QK_TQ3; i++) {
        float normalized = x[i] * inv_d;  // in [-1, 1]

        // Try all grid entries in both polarities
        int best_idx = 0;
        int best_sign = 0;
        float best_err = 1e30f;
        int j;
        for (j = 0; j < 4; j++) {
            float g = TQ_POLAR_GRID_4[j];
            float e_pos = (normalized - g) * (normalized - g);
            float e_neg = (normalized + g) * (normalized + g);
            if (e_pos < best_err) { best_err = e_pos; best_idx = j; best_sign = 0; }
            if (e_neg < best_err) { best_err = e_neg; best_idx = j; best_sign = 1; }
        }

        tq_pack_2bit(block->al, i, (uint8_t)best_idx);
        tq_pack_1bit(block->signs, i, (uint8_t)best_sign);
    }
}

// Dequantize a block_tq3_0 into QK_TQ3 floats
static inline void dequantize_block_tq3_0(const block_tq3_0 * block, float * x) {
    float d = GGML_FP16_TO_FP32(block->d);

    for (int i = 0; i < QK_TQ3; i++) {
        uint8_t angle_idx = tq_unpack_2bit(block->al, i);
        uint8_t sign      = tq_unpack_1bit(block->signs, i);
        float grid_val    = TQ_POLAR_GRID_4[angle_idx];
        x[i] = d * grid_val * (1.0f - 2.0f * sign);
    }
}

// Quantize a block of QK_TQ4 floats into block_tq4_0
static inline void quantize_block_tq4_0(const float * x, block_tq4_0 * block) {
    // Step 1: Compute block absolute max
    float amax = 0.0f;
    for (int i = 0; i < QK_TQ4; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) amax = ax;
    }
    block->d = GGML_FP32_TO_FP16(amax);

    float inv_d = amax > 0.0f ? 1.0f / amax : 0.0f;

    // Zero out arrays
    memset(block->al, 0, sizeof(block->al));
    memset(block->ah, 0, sizeof(block->ah));
    memset(block->signs, 0, sizeof(block->signs));

    // Step 2: For each element, find nearest grid entry + sign correction
    for (int i = 0; i < QK_TQ4; i++) {
        float normalized = x[i] * inv_d;  // in [-1, 1]

        // Find closest grid point, considering sign flip
        // Try all 8 grid entries in both polarities, pick minimum error
        int best_idx = 0;
        int best_sign = 0;
        float best_err = 1e30f;
        int j;
        for (j = 0; j < 8; j++) {
            float g = TQ_POLAR_GRID_8[j];
            float e_pos = (normalized - g) * (normalized - g);
            float e_neg = (normalized + g) * (normalized + g);
            if (e_pos < best_err) { best_err = e_pos; best_idx = j; best_sign = 0; }
            if (e_neg < best_err) { best_err = e_neg; best_idx = j; best_sign = 1; }
        }

        // Split 3-bit index into 2-bit lo + 1-bit hi
        tq_pack_2bit(block->al, i, (uint8_t)(best_idx & 0x3));
        tq_pack_1bit(block->ah, i, (uint8_t)((best_idx >> 2) & 0x1));
        tq_pack_1bit(block->signs, i, (uint8_t)best_sign);
    }
}

// Dequantize a block_tq4_0 into QK_TQ4 floats
static inline void dequantize_block_tq4_0(const block_tq4_0 * block, float * x) {
    float d = GGML_FP16_TO_FP32(block->d);

    for (int i = 0; i < QK_TQ4; i++) {
        uint8_t lo        = tq_unpack_2bit(block->al, i);
        uint8_t hi        = tq_unpack_1bit(block->ah, i);
        uint8_t angle_idx = lo | (hi << 2);
        uint8_t sign      = tq_unpack_1bit(block->signs, i);
        float grid_val    = TQ_POLAR_GRID_8[angle_idx];
        x[i] = d * grid_val * (1.0f - 2.0f * sign);
    }
}

#ifdef __cplusplus
}
#endif
