#pragma once

// Walsh-Hadamard Transform (WHT) building blocks for TurboQuant KV cache compression.
// Provides WHT-32 butterfly transforms, random diagonal sign generation,
// and Lloyd-optimal codebooks for post-WHT Gaussian data.

#include <cstdint>

// Fixed seeds for the D @ H @ D' rotation structure
#define WHT_SEED_D       0x9E3779B9u  // golden ratio
#define WHT_SEED_D_PRIME 0xDEADBEEFu

// ====================== WHT-32 butterfly transform ======================

// In-place forward WHT-32 on data[0..31] in registers.
// 5 butterfly stages for N=32, normalized by 1/sqrt(32).
__device__ __forceinline__ void wht_forward_32(float data[32]) {
    for (int stride = 16; stride >= 1; stride >>= 1) {
        for (int i = 0; i < 32; i++) {
            if ((i & stride) == 0) {
                float a = data[i];
                float b = data[i | stride];
                data[i]          = a + b;
                data[i | stride] = a - b;
            }
        }
    }
    // Normalize by 1/sqrt(32)
    const float norm = 0.17677669529663689f;
    for (int i = 0; i < 32; i++) data[i] *= norm;
}

// In-place inverse WHT-32 on data[0..31] in registers.
// WHT is self-inverse (up to scaling), so same butterflies.
__device__ __forceinline__ void wht_inverse_32(float data[32]) {
    for (int stride = 16; stride >= 1; stride >>= 1) {
        for (int i = 0; i < 32; i++) {
            if ((i & stride) == 0) {
                float a = data[i];
                float b = data[i | stride];
                data[i]          = a + b;
                data[i | stride] = a - b;
            }
        }
    }
    const float norm = 0.17677669529663689f;
    for (int i = 0; i < 32; i++) data[i] *= norm;
}

// ====================== Random diagonal sign generation ======================

// Wang hash for deterministic PRNG
__device__ __forceinline__ uint32_t wang_hash(uint32_t seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

// Apply random +/-1 diagonal to 32 elements.
// Uses sign bits packed in a uint32_t: bit i set => negate data[i].
__device__ __forceinline__ void apply_diagonal_signs(float data[32], uint32_t signs_bits) {
    for (int i = 0; i < 32; i++) {
        if ((signs_bits >> i) & 1u) {
            data[i] = -data[i];
        }
    }
}

// Generate sign bits for a sub-block deterministically
__device__ __forceinline__ uint32_t generate_signs(uint32_t base_seed, int sub_block_idx) {
    return wang_hash(base_seed ^ (uint32_t)sub_block_idx);
}

// Full forward rotation: D @ H @ D'
__device__ __forceinline__ void wht_rotate_forward_32(float data[32], uint32_t seed_d, uint32_t seed_d_prime, int sub_block_idx) {
    uint32_t signs_d_prime = generate_signs(seed_d_prime, sub_block_idx);
    uint32_t signs_d = generate_signs(seed_d, sub_block_idx);
    apply_diagonal_signs(data, signs_d_prime);
    wht_forward_32(data);
    apply_diagonal_signs(data, signs_d);
}

// Full inverse rotation: D'^T @ H^T @ D^T = D' @ H @ D (since D, H are symmetric)
__device__ __forceinline__ void wht_rotate_inverse_32(float data[32], uint32_t seed_d, uint32_t seed_d_prime, int sub_block_idx) {
    uint32_t signs_d = generate_signs(seed_d, sub_block_idx);
    uint32_t signs_d_prime = generate_signs(seed_d_prime, sub_block_idx);
    apply_diagonal_signs(data, signs_d);
    wht_inverse_32(data);
    apply_diagonal_signs(data, signs_d_prime);
}

// ====================== Lloyd-optimal codebooks ======================
// After WHT rotation, normalized coordinates are approximately N(0, 1/sqrt(d)).
// For sign-separated quantization (positive grid + sign bit):

// Optimal centroids for half-normal |N(0,1)| distribution
// 4-level (TQ3: 2-bit index + 1-bit sign = 3 bits)
__device__ static const float WHT_CODEBOOK_4[4] = {
    0.1450f, 0.4528f, 0.7913f, 1.2771f
};

// 8-level (TQ4: 3-bit index + 1-bit sign = 4 bits)
__device__ static const float WHT_CODEBOOK_8[8] = {
    0.0737f, 0.2225f, 0.3740f, 0.5322f,
    0.7025f, 0.8941f, 1.1249f, 1.4709f
};
