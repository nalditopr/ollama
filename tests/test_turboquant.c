// TurboQuant block structure validation test
// Verifies quantize -> dequantize roundtrip and bit packing correctness

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../ml/backend/ggml/ggml/src/turboquant.h"

// =====================================================================
// Test helpers
// =====================================================================

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static float rmse(const float * a, const float * b, int n) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum / n);
}

static float max_abs_error(const float * a, const float * b, int n) {
    float max_err = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float err = (float)fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// =====================================================================
// Tests
// =====================================================================

static int test_bit_packing(void) {
    uint8_t al[64];
    uint8_t signs[32];
    int i;

    printf("Test: bit packing... ");

    // Test 2-bit packing
    memset(al, 0, sizeof(al));
    for (i = 0; i < 256; i++) {
        uint8_t val = (uint8_t)(i % 4);
        tq_pack_2bit(al, i, val);
    }
    for (i = 0; i < 256; i++) {
        uint8_t val = tq_unpack_2bit(al, i);
        if (val != (uint8_t)(i % 4)) {
            printf("FAIL (2-bit: elem %d expected %d got %d)\n", i, i % 4, val);
            return 1;
        }
    }

    // Test 1-bit packing
    memset(signs, 0, sizeof(signs));
    for (i = 0; i < 256; i++) {
        tq_pack_1bit(signs, i, (uint8_t)(i % 2));
    }
    for (i = 0; i < 256; i++) {
        uint8_t val = tq_unpack_1bit(signs, i);
        if (val != (uint8_t)(i % 2)) {
            printf("FAIL (1-bit: elem %d expected %d got %d)\n", i, i % 2, val);
            return 1;
        }
    }

    printf("PASS\n");
    return 0;
}

static int test_struct_sizes(void) {
    float bpw_tq3, bpw_tq4;
    printf("Test: struct sizes... ");

    if (sizeof(block_tq3_0) != 98) {
        printf("FAIL (tq3_0: expected 98, got %u)\n", (unsigned)sizeof(block_tq3_0));
        return 1;
    }
    if (sizeof(block_tq4_0) != 130) {
        printf("FAIL (tq4_0: expected 130, got %u)\n", (unsigned)sizeof(block_tq4_0));
        return 1;
    }

    bpw_tq3 = (float)(sizeof(block_tq3_0) * 8) / QK_TQ3;
    bpw_tq4 = (float)(sizeof(block_tq4_0) * 8) / QK_TQ4;
    printf("PASS (TQ3: %.4f bpw, TQ4: %.4f bpw)\n", bpw_tq3, bpw_tq4);
    return 0;
}

static int test_tq3_roundtrip(void) {
    float input[256];
    float output[256];
    block_tq3_0 block;
    float err, merr;
    int i;

    printf("Test: TQ3_0 roundtrip... ");
    srand(42);
    for (i = 0; i < QK_TQ3; i++) {
        input[i] = randf();
    }

    quantize_block_tq3_0(input, &block);
    dequantize_block_tq3_0(&block, output);

    err = rmse(input, output, QK_TQ3);
    merr = max_abs_error(input, output, QK_TQ3);
    printf("PASS (RMSE: %.6f, MaxErr: %.6f)\n", err, merr);
    return 0;
}

static int test_tq4_roundtrip(void) {
    float input[256];
    float output[256];
    block_tq4_0 block;
    float err, merr;
    int i;

    printf("Test: TQ4_0 roundtrip... ");
    srand(42);
    for (i = 0; i < QK_TQ4; i++) {
        input[i] = randf();
    }

    quantize_block_tq4_0(input, &block);
    dequantize_block_tq4_0(&block, output);

    err = rmse(input, output, QK_TQ4);
    merr = max_abs_error(input, output, QK_TQ4);
    printf("PASS (RMSE: %.6f, MaxErr: %.6f)\n", err, merr);
    return 0;
}

static int test_tq4_better_than_tq3(void) {
    float input[256];
    float out3[256], out4[256];
    block_tq3_0 b3;
    block_tq4_0 b4;
    float err3, err4;
    int i;

    printf("Test: TQ4 < TQ3 error... ");
    srand(123);
    for (i = 0; i < 256; i++) {
        input[i] = randf();
    }

    quantize_block_tq3_0(input, &b3);
    quantize_block_tq4_0(input, &b4);
    dequantize_block_tq3_0(&b3, out3);
    dequantize_block_tq4_0(&b4, out4);

    err3 = rmse(input, out3, 256);
    err4 = rmse(input, out4, 256);

    if (err4 < err3) {
        printf("PASS (TQ3 RMSE: %.6f, TQ4 RMSE: %.6f)\n", err3, err4);
        return 0;
    } else {
        printf("FAIL (TQ4 not better: TQ3=%.6f, TQ4=%.6f)\n", err3, err4);
        return 1;
    }
}

static int test_zero_vector(void) {
    float input[256];
    float output[256];
    block_tq4_0 block;
    int i;

    printf("Test: zero vector... ");
    memset(input, 0, sizeof(input));

    quantize_block_tq4_0(input, &block);
    dequantize_block_tq4_0(&block, output);

    for (i = 0; i < 256; i++) {
        if (output[i] != 0.0f) {
            printf("FAIL (non-zero output at %d: %f)\n", i, output[i]);
            return 1;
        }
    }

    printf("PASS\n");
    return 0;
}

int main(void) {
    int failures = 0;
    printf("=== TurboQuant Block Structure Tests ===\n\n");

    failures += test_struct_sizes();
    failures += test_bit_packing();
    failures += test_tq3_roundtrip();
    failures += test_tq4_roundtrip();
    failures += test_tq4_better_than_tq3();
    failures += test_zero_vector();

    printf("\n%s (%d failures)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", failures);
    return failures;
}
