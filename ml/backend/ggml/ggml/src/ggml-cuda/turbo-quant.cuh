#pragma once

#include "common.cuh"

/*
 * TurboQuant CUDA backend — KV cache compression
 *
 * turbo3: Lucien2468 3-bit uniform quantization, block_size=32, 3.5 bpw
 *   - Simple round+clamp: (val-4)*d, no codebook, no rotation, no WHT
 *   - FA works via graph-level F32 cast before attention
 * turbo4: 3-bit angle grid + 1-bit sign, block_size=128, 4.0625 bpw
 */

#define QR_TURBO3  2   /* 2 elements per dequantize call */
#define QK_TURBO4  128

/* ===== Declarations ===== */

void turbo_quant_init_cuda(cudaStream_t stream);
void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

/* Inverse WHT32 on output data (post-matmul). Used by TQ3_KV/TQ4_WHT types, not turbo3. */
void turbo3_inverse_wht32_cuda(float * data, int64_t n_elements, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);
