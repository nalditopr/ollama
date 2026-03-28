/*
 * TurboQuant CUDA kernels
 * turbo3: Lucien2468 3-bit uniform quantization (val-4)*d
 * turbo4: 3-bit angle grid + 1-bit sign
 */

#include "turbo-quant.cuh"
#include "convert.cuh"

#include <cstdint>

void turbo_quant_init_cuda(cudaStream_t stream) {
    GGML_UNUSED(stream);
}

/* ===== turbo3 dequantize: Lucien2468 3-bit uniform (val-4)*d ===== */

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0(
        const void * __restrict__ vx, dst_t * __restrict__ yy, const int64_t k) {
    const int64_t i = blockIdx.x;
    const int tid = threadIdx.x;
    if (i * QK_TURBO3 >= k) return;

    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float d = __half2float(x[i].d);
    const int base = tid * 4;
    dst_t * out = yy + i * QK_TURBO3 + base;

    // Each thread handles one group of 8 values (one 3-byte pack produces 8 values,
    // but we split across 2 threads: 4 values each)
    const int group = base / 8;
    const int half = (base % 8) / 4;
    const uint8_t * qs = x[i].qs + group * 3;

    float v[4];
    if (half == 0) {
        v[0] = ((qs[0]      ) & 7) - 4.0f;
        v[1] = ((qs[0] >> 3 ) & 7) - 4.0f;
        v[2] = (((qs[0] >> 6) & 3) | ((qs[1] & 1) << 2)) - 4.0f;
        v[3] = ((qs[1] >> 1 ) & 7) - 4.0f;
    } else {
        v[0] = ((qs[1] >> 4 ) & 7) - 4.0f;
        v[1] = (((qs[1] >> 7) & 1) | ((qs[2] & 3) << 1)) - 4.0f;
        v[2] = ((qs[2] >> 2 ) & 7) - 4.0f;
        v[3] = ((qs[2] >> 5 ) & 7) - 4.0f;
    }

    for (int l = 0; l < 4 && base + l < QK_TURBO3; l++) {
        out[l] = ggml_cuda_cast<dst_t>(v[l] * d);
    }
}

/* ===== turbo4 dequantize block kernel ===== */

template<typename dst_t>
static __global__ void dequantize_block_turbo4_0(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    constexpr float grid[8] = {
        1.0f/16.0f, 3.0f/16.0f, 5.0f/16.0f, 7.0f/16.0f,
        9.0f/16.0f, 11.0f/16.0f, 13.0f/16.0f, 15.0f/16.0f
    };
    const int64_t i = blockIdx.x;
    const int tid = threadIdx.x;
    if (i * QK_TURBO4 >= k) return;

    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float d = __half2float(x[i].d);
    const int base = tid * 4;
    dst_t * out = y + i * QK_TURBO4 + base;

    for (int l = 0; l < 4 && base + l < QK_TURBO4; l++) {
        const int j = base + l;
        const uint8_t lo = (x[i].al[j/4] >> ((j%4)*2)) & 0x3;
        const uint8_t hi = (x[i].ah[j/8] >> (j%8)) & 0x1;
        const uint8_t s  = (x[i].signs[j/8] >> (j%8)) & 0x1;
        out[l] = ggml_cuda_cast<dst_t>(d * grid[lo|(hi<<2)] * (1.0f - 2.0f * s));
    }
}

/* Inverse WHT32 kernel kept for TQ3_KV/TQ4_WHT types (not used by turbo3 anymore) */

static __global__ void turbo3_inverse_wht32_kernel(float * __restrict__ data, const int64_t n_elements) {
    const int8_t signs[32] = {
        +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
        +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1
    };

    const int64_t block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (block_idx * 32 + tid >= n_elements) return;

    __shared__ float shmem[32];
    shmem[tid] = data[block_idx * 32 + tid];
    __syncthreads();

    for (int step = 1; step < 32; step <<= 1) {
        int partner = tid ^ step;
        float a = shmem[tid];
        float b = shmem[partner];
        __syncthreads();
        if (tid < partner) {
            shmem[tid]     = a + b;
            shmem[partner] = a - b;
        }
        __syncthreads();
    }

    const float inv_sqrt32 = 0.17677669529663688f;
    data[block_idx * 32 + tid] = shmem[tid] * inv_sqrt32 * signs[tid];
}

void turbo3_inverse_wht32_cuda(float * data, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % 32 == 0);
    const int nb = n_elements / 32;
    turbo3_inverse_wht32_kernel<<<nb, 32, 0, stream>>>(data, n_elements);
}

/* ===== TURBO_WHT graph op (for post-matmul inverse WHT on output) ===== */

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t ne = ggml_nelements(src0);
    cudaStream_t stream = ctx.stream();

    if (src0_d != dst_d) {
        cudaMemcpyAsync(dst_d, src0_d, ne * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    /* direction=1 → inverse WHT32 (for V output), direction=0 → forward (not used here) */
    GGML_ASSERT(ne % 32 == 0);
    turbo3_inverse_wht32_cuda(dst_d, ne, stream);
    /* Note: forward WHT would need a separate kernel. For now only inverse is used. */
}

/* ===== Row dequantize launchers ===== */

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    dequantize_block_turbo3_0<<<nb, 8, 0, stream>>>(vx, y, k); // 32 elems / 4 per thread = 8
}

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    dequantize_block_turbo4_0<<<nb, 64, 0, stream>>>(vx, y, k); // 256 elems / 4 per thread = 64
}

/* Explicit template instantiations */
template void dequantize_row_turbo3_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);

template void dequantize_row_turbo4_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);
