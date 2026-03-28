// TurboQuant performance benchmark
// Measures quantization/dequantization throughput, compression ratio, and error metrics
// Compares TQ3_0 (3.0625 bpw) vs TQ4_0 (4.0625 bpw)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "../ml/backend/ggml/ggml/src/turboquant.h"

// =====================================================================
// Configuration
// =====================================================================

// 4096 x 4096 = 16M floats, simulating a large model layer
#define BENCH_ROWS      4096
#define BENCH_COLS      4096
#define BENCH_TOTAL     (BENCH_ROWS * BENCH_COLS)

// Number of blocks (each block = 256 elements)
#define NUM_BLOCKS      (BENCH_TOTAL / QK_TQ3)

// Number of iterations for timing
#define QUANT_ITERS     10
#define DEQUANT_ITERS   20

// =====================================================================
// High-resolution timer (QueryPerformanceCounter)
// =====================================================================

static LARGE_INTEGER qpc_freq;

static void timer_init(void) {
    QueryPerformanceFrequency(&qpc_freq);
}

static double timer_now_sec(void) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)qpc_freq.QuadPart;
}

// =====================================================================
// Helpers
// =====================================================================

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static float compute_rmse(const float *a, const float *b, int n) {
    double sum = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return (float)sqrt(sum / n);
}

static float compute_peak_abs_error(const float *a, const float *b, int n) {
    float peak = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float err = (float)fabs((double)a[i] - (double)b[i]);
        if (err > peak) peak = err;
    }
    return peak;
}

// =====================================================================
// Benchmark routines
// =====================================================================

typedef struct {
    const char *name;
    float bpw;
    double quant_time_sec;
    double dequant_time_sec;
    double quant_throughput_gbs;
    double dequant_throughput_gbs;
    float compression_ratio;
    float rmse;
    float peak_error;
} bench_result_t;

static void bench_tq3(const float *input, float *output,
                       block_tq3_0 *blocks, bench_result_t *result) {
    double t0, t1;
    int iter, b;
    double input_bytes  = (double)BENCH_TOTAL * sizeof(float);
    double output_bytes = (double)BENCH_TOTAL * sizeof(float);

    result->name = "TQ3_0";
    result->bpw  = (float)(sizeof(block_tq3_0) * 8) / QK_TQ3;

    // -- Quantization benchmark --
    t0 = timer_now_sec();
    for (iter = 0; iter < QUANT_ITERS; iter++) {
        for (b = 0; b < NUM_BLOCKS; b++) {
            quantize_block_tq3_0(input + b * QK_TQ3, &blocks[b]);
        }
    }
    t1 = timer_now_sec();
    result->quant_time_sec = (t1 - t0) / QUANT_ITERS;
    result->quant_throughput_gbs = (input_bytes / result->quant_time_sec) / (1024.0 * 1024.0 * 1024.0);

    // -- Dequantization benchmark --
    // Ensure blocks are quantized
    for (b = 0; b < NUM_BLOCKS; b++) {
        quantize_block_tq3_0(input + b * QK_TQ3, &blocks[b]);
    }
    t0 = timer_now_sec();
    for (iter = 0; iter < DEQUANT_ITERS; iter++) {
        for (b = 0; b < NUM_BLOCKS; b++) {
            dequantize_block_tq3_0(&blocks[b], output + b * QK_TQ3);
        }
    }
    t1 = timer_now_sec();
    result->dequant_time_sec = (t1 - t0) / DEQUANT_ITERS;
    result->dequant_throughput_gbs = (output_bytes / result->dequant_time_sec) / (1024.0 * 1024.0 * 1024.0);

    // -- Compression ratio --
    {
        double compressed_bytes = (double)NUM_BLOCKS * sizeof(block_tq3_0);
        result->compression_ratio = (float)(input_bytes / compressed_bytes);
    }

    // -- Error metrics (use last dequant pass) --
    result->rmse       = compute_rmse(input, output, BENCH_TOTAL);
    result->peak_error = compute_peak_abs_error(input, output, BENCH_TOTAL);
}

static void bench_tq4(const float *input, float *output,
                       block_tq4_0 *blocks, bench_result_t *result) {
    double t0, t1;
    int iter, b;
    double input_bytes  = (double)BENCH_TOTAL * sizeof(float);
    double output_bytes = (double)BENCH_TOTAL * sizeof(float);

    result->name = "TQ4_0";
    result->bpw  = (float)(sizeof(block_tq4_0) * 8) / QK_TQ4;

    // -- Quantization benchmark --
    t0 = timer_now_sec();
    for (iter = 0; iter < QUANT_ITERS; iter++) {
        for (b = 0; b < NUM_BLOCKS; b++) {
            quantize_block_tq4_0(input + b * QK_TQ4, &blocks[b]);
        }
    }
    t1 = timer_now_sec();
    result->quant_time_sec = (t1 - t0) / QUANT_ITERS;
    result->quant_throughput_gbs = (input_bytes / result->quant_time_sec) / (1024.0 * 1024.0 * 1024.0);

    // -- Dequantization benchmark --
    for (b = 0; b < NUM_BLOCKS; b++) {
        quantize_block_tq4_0(input + b * QK_TQ4, &blocks[b]);
    }
    t0 = timer_now_sec();
    for (iter = 0; iter < DEQUANT_ITERS; iter++) {
        for (b = 0; b < NUM_BLOCKS; b++) {
            dequantize_block_tq4_0(&blocks[b], output + b * QK_TQ4);
        }
    }
    t1 = timer_now_sec();
    result->dequant_time_sec = (t1 - t0) / DEQUANT_ITERS;
    result->dequant_throughput_gbs = (output_bytes / result->dequant_time_sec) / (1024.0 * 1024.0 * 1024.0);

    // -- Compression ratio --
    {
        double compressed_bytes = (double)NUM_BLOCKS * sizeof(block_tq4_0);
        result->compression_ratio = (float)(input_bytes / compressed_bytes);
    }

    // -- Error metrics --
    result->rmse       = compute_rmse(input, output, BENCH_TOTAL);
    result->peak_error = compute_peak_abs_error(input, output, BENCH_TOTAL);
}

// =====================================================================
// Report
// =====================================================================

static void print_separator(void) {
    printf("+-----------+----------+----------+-----------+-----------+----------+----------+\n");
}

static void print_result(const bench_result_t *r) {
    printf("| %-9s | %6.2f   | %6.2f   | %7.3f   | %7.3f   | %.6f | %.6f |\n",
           r->name, r->bpw, r->compression_ratio,
           r->quant_throughput_gbs, r->dequant_throughput_gbs,
           r->rmse, r->peak_error);
}

// =====================================================================
// Main
// =====================================================================

int main(void) {
    float         *input   = NULL;
    float         *output  = NULL;
    block_tq3_0   *tq3_buf = NULL;
    block_tq4_0   *tq4_buf = NULL;
    bench_result_t r3, r4;
    int i;

    printf("=== TurboQuant Performance Benchmark ===\n\n");
    printf("Data size:  %d x %d = %d floats (%.1f MB)\n",
           BENCH_ROWS, BENCH_COLS, BENCH_TOTAL,
           (double)BENCH_TOTAL * sizeof(float) / (1024.0 * 1024.0));
    printf("Block size: %d elements\n", QK_TQ3);
    printf("Blocks:     %d\n", NUM_BLOCKS);
    printf("Quant iters:   %d\n", QUANT_ITERS);
    printf("Dequant iters: %d\n\n", DEQUANT_ITERS);

    timer_init();

    // Allocate
    input   = (float *)malloc((size_t)BENCH_TOTAL * sizeof(float));
    output  = (float *)malloc((size_t)BENCH_TOTAL * sizeof(float));
    tq3_buf = (block_tq3_0 *)malloc((size_t)NUM_BLOCKS * sizeof(block_tq3_0));
    tq4_buf = (block_tq4_0 *)malloc((size_t)NUM_BLOCKS * sizeof(block_tq4_0));

    if (!input || !output || !tq3_buf || !tq4_buf) {
        fprintf(stderr, "ERROR: allocation failed (need ~%.0f MB)\n",
                ((double)BENCH_TOTAL * 2 * sizeof(float) +
                 (double)NUM_BLOCKS * (sizeof(block_tq3_0) + sizeof(block_tq4_0)))
                / (1024.0 * 1024.0));
        return 1;
    }

    // Fill with random data (simulating typical weight distribution)
    printf("Generating random input data...\n");
    srand(12345);
    for (i = 0; i < BENCH_TOTAL; i++) {
        input[i] = randf() * 0.1f;  // small weights typical of transformer layers
    }

    // Benchmark TQ3_0
    printf("Benchmarking TQ3_0...\n");
    bench_tq3(input, output, tq3_buf, &r3);

    // Benchmark TQ4_0
    printf("Benchmarking TQ4_0...\n");
    bench_tq4(input, output, tq4_buf, &r4);

    // Report
    printf("\n=== Results ===\n\n");
    print_separator();
    printf("| Type      | BPW      | Ratio    | Q GB/s    | DQ GB/s   | RMSE     | PeakErr  |\n");
    print_separator();
    print_result(&r3);
    print_result(&r4);
    print_separator();

    printf("\nDetailed timing:\n");
    printf("  TQ3_0 quantize:   %.3f ms/iter\n", r3.quant_time_sec * 1000.0);
    printf("  TQ3_0 dequantize: %.3f ms/iter\n", r3.dequant_time_sec * 1000.0);
    printf("  TQ4_0 quantize:   %.3f ms/iter\n", r4.quant_time_sec * 1000.0);
    printf("  TQ4_0 dequantize: %.3f ms/iter\n", r4.dequant_time_sec * 1000.0);

    printf("\nMemory footprint for %d floats (%.1f MB FP32):\n",
           BENCH_TOTAL,
           (double)BENCH_TOTAL * sizeof(float) / (1024.0 * 1024.0));
    printf("  TQ3_0: %.2f MB  (%.1fx compression)\n",
           (double)NUM_BLOCKS * sizeof(block_tq3_0) / (1024.0 * 1024.0),
           r3.compression_ratio);
    printf("  TQ4_0: %.2f MB  (%.1fx compression)\n",
           (double)NUM_BLOCKS * sizeof(block_tq4_0) / (1024.0 * 1024.0),
           r4.compression_ratio);

    printf("\nError comparison (TQ4_0 should be lower than TQ3_0):\n");
    printf("  TQ3_0 RMSE: %.8f  Peak: %.8f\n", r3.rmse, r3.peak_error);
    printf("  TQ4_0 RMSE: %.8f  Peak: %.8f\n", r4.rmse, r4.peak_error);
    if (r4.rmse < r3.rmse) {
        printf("  -> TQ4_0 has %.1f%% lower RMSE (expected: more bits = better quality)\n",
               (1.0 - r4.rmse / r3.rmse) * 100.0);
    } else {
        printf("  WARNING: TQ4_0 RMSE is not lower than TQ3_0!\n");
    }

    printf("\nQ4_0 reference characteristics (for comparison):\n");
    printf("  Q4_0: 4.5 bpw, ~7.1x compression, typical RMSE ~0.002-0.01 for similar data\n");
    printf("  TQ4_0: %.4f bpw (%.1f%% smaller than Q4_0 at 4.5 bpw)\n",
           r4.bpw, (1.0 - r4.bpw / 4.5) * 100.0);

    free(input);
    free(output);
    free(tq3_buf);
    free(tq4_buf);

    printf("\nBenchmark complete.\n");
    return 0;
}
