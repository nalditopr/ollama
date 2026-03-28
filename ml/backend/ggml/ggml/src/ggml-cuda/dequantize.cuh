#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

__device__ static const float TQ_GRID_4[4] = {
    1.0f/8.0f, 3.0f/8.0f, 5.0f/8.0f, 7.0f/8.0f,
};

__device__ static const float TQ_GRID_8[8] = {
    1.0f/16.0f, 3.0f/16.0f, 5.0f/16.0f, 7.0f/16.0f,
    9.0f/16.0f, 11.0f/16.0f, 13.0f/16.0f, 15.0f/16.0f,
};

static __device__ __forceinline__ void dequantize_tq3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_tq3_0 * x = (const block_tq3_0 *) vx;
    const float d = __half2float(x[ib].d);

    // iqs indexes 2 consecutive elements
    const int j0 = iqs * 2;
    const int j1 = j0 + 1;

    // Extract 2-bit angle indices
    const uint8_t a0 = (x[ib].al[j0/4] >> ((j0%4)*2)) & 0x3;
    const uint8_t a1 = (x[ib].al[j1/4] >> ((j1%4)*2)) & 0x3;

    // Extract sign bits
    const uint8_t s0 = (x[ib].signs[j0/8] >> (j0%8)) & 0x1;
    const uint8_t s1 = (x[ib].signs[j1/8] >> (j1%8)) & 0x1;

    v.x = d * TQ_GRID_4[a0] * (1.0f - 2.0f * s0);
    v.y = d * TQ_GRID_4[a1] * (1.0f - 2.0f * s1);
}

static __device__ __forceinline__ void dequantize_tq4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_tq4_0 * x = (const block_tq4_0 *) vx;
    const float d = __half2float(x[ib].d);

    const int j0 = iqs * 2;
    const int j1 = j0 + 1;

    // Extract 3-bit angle indices (2-bit lo + 1-bit hi)
    const uint8_t lo0 = (x[ib].al[j0/4] >> ((j0%4)*2)) & 0x3;
    const uint8_t hi0 = (x[ib].ah[j0/8] >> (j0%8)) & 0x1;
    const uint8_t lo1 = (x[ib].al[j1/4] >> ((j1%4)*2)) & 0x3;
    const uint8_t hi1 = (x[ib].ah[j1/8] >> (j1%8)) & 0x1;

    const uint8_t idx0 = lo0 | (hi0 << 2);
    const uint8_t idx1 = lo1 | (hi1 << 2);

    // Extract sign bits
    const uint8_t s0 = (x[ib].signs[j0/8] >> (j0%8)) & 0x1;
    const uint8_t s1 = (x[ib].signs[j1/8] >> (j1%8)) & 0x1;

    v.x = d * TQ_GRID_8[idx0] * (1.0f - 2.0f * s0);
    v.y = d * TQ_GRID_8[idx1] * (1.0f - 2.0f * s1);
}

// WHT codebooks for dequantization
#include "wht.cuh"

// WHT dequantize: dequantize 2 elements within a sub-block, then apply inverse WHT
// NOTE: For the generic convert path, this operates on pairs of elements.
// The inverse WHT needs all 32 elements of a sub-block, so this function
// dequantizes the pair using WHT codebook values. The full inverse WHT
// is applied in the block-level dequantize paths (fattn, vecdot).
// For the element-level dequant used by convert.cu, we apply a simplified
// approach: dequant with WHT codebook but mark that inverse WHT is needed.
static __device__ __forceinline__ void dequantize_tq3_0_wht(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_tq3_0 * x = (const block_tq3_0 *) vx;
    const float d = __half2float(x[ib].d);

    const int j0 = iqs * 2;
    const int j1 = j0 + 1;

    const uint8_t a0 = (x[ib].al[j0/4] >> ((j0%4)*2)) & 0x3;
    const uint8_t a1 = (x[ib].al[j1/4] >> ((j1%4)*2)) & 0x3;

    const uint8_t s0 = (x[ib].signs[j0/8] >> (j0%8)) & 0x1;
    const uint8_t s1 = (x[ib].signs[j1/8] >> (j1%8)) & 0x1;

    v.x = d * WHT_CODEBOOK_4[a0] * (1.0f - 2.0f * s0);
    v.y = d * WHT_CODEBOOK_4[a1] * (1.0f - 2.0f * s1);
}

static __device__ __forceinline__ void dequantize_tq4_0_wht(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_tq4_0 * x = (const block_tq4_0 *) vx;
    const float d = __half2float(x[ib].d);

    const int j0 = iqs * 2;
    const int j1 = j0 + 1;

    const uint8_t lo0 = (x[ib].al[j0/4] >> ((j0%4)*2)) & 0x3;
    const uint8_t hi0 = (x[ib].ah[j0/8] >> (j0%8)) & 0x1;
    const uint8_t lo1 = (x[ib].al[j1/4] >> ((j1%4)*2)) & 0x3;
    const uint8_t hi1 = (x[ib].ah[j1/8] >> (j1%8)) & 0x1;

    const uint8_t idx0 = lo0 | (hi0 << 2);
    const uint8_t idx1 = lo1 | (hi1 << 2);

    const uint8_t s0 = (x[ib].signs[j0/8] >> (j0%8)) & 0x1;
    const uint8_t s1 = (x[ib].signs[j1/8] >> (j1%8)) & 0x1;

    v.x = d * WHT_CODEBOOK_8[idx0] * (1.0f - 2.0f * s0);
    v.y = d * WHT_CODEBOOK_8[idx1] * (1.0f - 2.0f * s1);
}

// TQ3_KV dequantize: element-pair dequant for convert.cu
// NOTE: This dequantizes in WHT-rotated space (no inverse WHT).
// For the convert path, this produces centroid-scaled values.
// The vec_dot path handles the WHT rotation on the Q side.
static __constant__ const float TQ3_KV_CENTROIDS_DQ[8] = {
    -2.1573f, -1.3336f, -0.7434f, -0.2428f,
     0.2428f,  0.7434f,  1.3336f,  2.1573f
};

static __device__ __forceinline__ void dequantize_tq3_kv(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_tq3_kv * x = (const block_tq3_kv *) vx;
    const float d = __half2float(x[ib].gamma);

    const int j0 = iqs;
    const int j1 = j0 + 1;

    // Extract 3-bit indices (2-bit lo in qs + 1-bit hi in qr)
    const uint8_t lo0 = (x[ib].qs[j0/4] >> ((j0%4)*2)) & 0x3;
    const uint8_t hi0 = (x[ib].qr[j0/8] >> (j0%8)) & 0x1;
    const uint8_t lo1 = (x[ib].qs[j1/4] >> ((j1%4)*2)) & 0x3;
    const uint8_t hi1 = (x[ib].qr[j1/8] >> (j1%8)) & 0x1;

    const uint8_t idx0 = lo0 | (hi0 << 2);
    const uint8_t idx1 = lo1 | (hi1 << 2);

    v.x = d * TQ3_KV_CENTROIDS_DQ[idx0];
    v.y = d * TQ3_KV_CENTROIDS_DQ[idx1];
}
