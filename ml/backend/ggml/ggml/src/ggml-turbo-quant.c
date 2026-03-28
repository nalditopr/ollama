/*
 * TurboQuant: KV cache compression
 *
 * TURBO3_0: Lucien2468 3-bit uniform quantization (3.5 bpw)
 * TURBO4_0: 3-bit angle grid + 1-bit sign (4.0625 bpw)
 *
 * For use as --cache-type-k turbo3 --cache-type-v turbo3 in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ---------- TURBO3_0: Lucien2468 3-bit uniform quantization ---------- */
/* round(x / d) clamped to [-4, 3], stored as unsigned [0,7] in 3-bit packing.
 * Dequant: (val - 4) * d, where d = amax / 4.0.
 * No codebook, no rotation, no WHT. */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % 32 == 0);
    for (int64_t i = 0; i < k / 32; i++) {
        float max = 0.0f;
        for (int j = 0; j < 32; j++) {
            float av = fabsf(x[i * 32 + j]);
            if (av > max) max = av;
        }
        float d = max / 4.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);
        for (int group = 0; group < 4; group++) {
            int base = (int)(i * 32) + group * 8;
            uint8_t v[8];
            for (int j = 0; j < 8; j++) {
                int q = (int)roundf(x[base + j] * id);
                int c = q < -4 ? -4 : (q > 3 ? 3 : q);
                v[j] = (uint8_t)(c + 4);
            }
            y[i].qs[group * 3 + 0] = (v[0] & 7) | ((v[1] & 7) << 3) | ((v[2] & 3) << 6);
            y[i].qs[group * 3 + 1] = ((v[2] & 4) >> 2) | ((v[3] & 7) << 1) | ((v[4] & 7) << 4) | ((v[5] & 1) << 7);
            y[i].qs[group * 3 + 2] = ((v[5] & 6) >> 1) | ((v[6] & 7) << 2) | ((v[7] & 7) << 5);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % 32 == 0);
    for (int64_t i = 0; i < k / 32; i++) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        for (int group = 0; group < 4; group++) {
            const uint8_t *qs = x[i].qs + group * 3;
            float *out = y + i * 32 + group * 8;
            out[0] = ((qs[0]      ) & 7) - 4.0f;
            out[1] = ((qs[0] >> 3 ) & 7) - 4.0f;
            out[2] = (((qs[0] >> 6) & 3) | ((qs[1] & 1) << 2)) - 4.0f;
            out[3] = ((qs[1] >> 1 ) & 7) - 4.0f;
            out[4] = ((qs[1] >> 4 ) & 7) - 4.0f;
            out[5] = (((qs[1] >> 7) & 1) | ((qs[2] & 3) << 1)) - 4.0f;
            out[6] = ((qs[2] >> 2 ) & 7) - 4.0f;
            out[7] = ((qs[2] >> 5 ) & 7) - 4.0f;
            for (int j = 0; j < 8; j++) out[j] *= d;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit angle grid + 1-bit sign (simple amax quantization) ---------- */
/* QK_TURBO4=256 elements, same layout as TQ4_0: d + al[64] + ah[32] + signs[32] = 130 bytes */

static const float TURBO4_GRID_CPU[8] = {
    1.0f/16.0f, 3.0f/16.0f, 5.0f/16.0f, 7.0f/16.0f,
    9.0f/16.0f, 11.0f/16.0f, 13.0f/16.0f, 15.0f/16.0f
};

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_TURBO4;
        float amax = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) {
            float av = fabsf(src[j]);
            if (av > amax) amax = av;
        }
        y[block].d = GGML_FP32_TO_FP16(amax);
        float inv_d = amax > 0.0f ? 1.0f / amax : 0.0f;

        memset(y[block].al, 0, QK_TURBO4 / 4);
        memset(y[block].ah, 0, QK_TURBO4 / 8);
        memset(y[block].signs, 0, QK_TURBO4 / 8);

        for (int j = 0; j < QK_TURBO4; j++) {
            float normalized = src[j] * inv_d;
            int best_idx = 0, best_sign = 0;
            float best_err = 1e30f;
            for (int g = 0; g < 8; g++) {
                float gv = TURBO4_GRID_CPU[g];
                float ep = (normalized - gv) * (normalized - gv);
                float en = (normalized + gv) * (normalized + gv);
                if (ep < best_err) { best_err = ep; best_idx = g; best_sign = 0; }
                if (en < best_err) { best_err = en; best_idx = g; best_sign = 1; }
            }
            y[block].al[j / 4] |= (uint8_t)((best_idx & 0x3) << ((j % 4) * 2));
            if (best_idx & 0x4) y[block].ah[j / 8] |= (uint8_t)(1 << (j % 8));
            if (best_sign)       y[block].signs[j / 8] |= (uint8_t)(1 << (j % 8));
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float d = GGML_FP16_TO_FP32(x[block].d);
        for (int j = 0; j < QK_TURBO4; j++) {
            uint8_t lo = (x[block].al[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi = (x[block].ah[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx = lo | (hi << 2);
            uint8_t sign = (x[block].signs[j / 8] >> (j % 8)) & 0x1;
            y[block * QK_TURBO4 + j] = d * TURBO4_GRID_CPU[idx] * (1.0f - 2.0f * sign);
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
