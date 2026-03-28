// Compute optimal TurboQuant polar grid values using Lloyd's algorithm
// on the distribution of normalized vector components.
//
// The normalized components of a d-dimensional Gaussian vector follow
// approximately N(0, 1/d). With d=256, sigma = 1/sqrt(256) = 0.0625.
//
// Since TurboQuant uses a sign bit (QJL), the grid values should all be
// positive. The sign bit doubles each grid entry to cover both polarities.
// Optimal grid = centroids of equal-probability regions of |N(0, sigma)|
// (the half-normal distribution).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define DIM 256
#define NUM_VECTORS 100000
#define NUM_SAMPLES (NUM_VECTORS * DIM)  // 25.6M samples

// Box-Muller transform for Gaussian random numbers
static double rand_normal(void) {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// Standard normal CDF (approximation)
static double phi(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// Standard normal PDF
static double npdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * 3.14159265358979323846);
}

// Compute the conditional mean of |X| where X ~ N(0, sigma^2) and a <= |X| < b
// E[|X| | a <= |X| < b] = sigma * [phi_pdf(a/sigma) - phi_pdf(b/sigma)] / [Phi(b/sigma) - Phi(a/sigma) - Phi(-a/sigma) + Phi(-b/sigma)]
// For half-normal: P(|X| < t) = 2*Phi(t/sigma) - 1 for t >= 0
// E[|X| | a <= |X| < b] = sigma * 2 * [npdf(a/sigma) - npdf(b/sigma)] / [2*Phi(b/sigma) - 2*Phi(a/sigma)]
//                        = sigma * [npdf(a/sigma) - npdf(b/sigma)] / [Phi(b/sigma) - Phi(a/sigma)]
static double half_normal_centroid(double a, double b, double sigma) {
    double as = a / sigma;
    double bs = b / sigma;
    double num = npdf(as) - npdf(bs);
    double den = phi(bs) - phi(as);
    if (den < 1e-15) return (a + b) / 2.0;
    return sigma * num / den;
}

// Compute optimal grid using analytical half-normal quantiles
static void compute_optimal_grid_analytical(int n_grid, double sigma, double *grid) {
    // Split the half-normal distribution [0, inf) into n_grid equal-probability regions
    // P(|X| < t) = 2*Phi(t/sigma) - 1 for t >= 0
    // For the half-normal, the CDF is F(t) = 2*Phi(t/sigma) - 1
    // We want boundaries b_0=0, b_1, ..., b_{n-1}, b_n=inf
    // such that F(b_{k+1}) - F(b_k) = 1/n for each k

    double *boundaries = (double *)malloc((n_grid + 1) * sizeof(double));
    boundaries[0] = 0.0;
    boundaries[n_grid] = 1e10;  // ~infinity

    for (int k = 1; k < n_grid; k++) {
        // F(b_k) = k/n_grid
        // 2*Phi(b_k/sigma) - 1 = k/(double)n_grid
        // Phi(b_k/sigma) = (1 + k/(double)n_grid) / 2
        double target_phi = (1.0 + (double)k / (double)n_grid) / 2.0;

        // Inverse normal CDF using bisection
        double lo = 0.0, hi = 10.0 * sigma;
        for (int iter = 0; iter < 200; iter++) {
            double mid = (lo + hi) / 2.0;
            if (phi(mid / sigma) < target_phi) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        boundaries[k] = (lo + hi) / 2.0;
    }

    // Compute centroids for each region
    for (int k = 0; k < n_grid; k++) {
        grid[k] = half_normal_centroid(boundaries[k], boundaries[k + 1], sigma);
    }

    free(boundaries);
}

// Refine grid using Lloyd's algorithm on actual samples
static void lloyd_refine(double *grid, int n_grid, double *samples, int n_samples, int max_iter) {
    double *new_grid = (double *)malloc(n_grid * sizeof(double));
    double *sum = (double *)malloc(n_grid * sizeof(double));
    int *count = (int *)malloc(n_grid * sizeof(int));

    for (int iter = 0; iter < max_iter; iter++) {
        // Reset accumulators
        for (int k = 0; k < n_grid; k++) {
            sum[k] = 0.0;
            count[k] = 0;
        }

        // Assign each sample to nearest grid point (using absolute value, sign handled separately)
        for (int i = 0; i < n_samples; i++) {
            double val = fabs(samples[i]);
            int best = 0;
            double best_dist = fabs(val - grid[0]);
            for (int k = 1; k < n_grid; k++) {
                double dist = fabs(val - grid[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = k;
                }
            }
            sum[best] += val;
            count[best]++;
        }

        // Update centroids
        double max_change = 0.0;
        for (int k = 0; k < n_grid; k++) {
            if (count[k] > 0) {
                new_grid[k] = sum[k] / count[k];
            } else {
                new_grid[k] = grid[k];
            }
            double change = fabs(new_grid[k] - grid[k]);
            if (change > max_change) max_change = change;
            grid[k] = new_grid[k];
        }

        if (max_change < 1e-12) break;
    }

    free(new_grid);
    free(sum);
    free(count);
}

// Compute RMSE for a given grid (with sign bit)
static double compute_rmse(double *grid, int n_grid, double *samples, int n_samples) {
    double mse = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double val = samples[i];
        double abs_val = fabs(val);
        // Find nearest grid point for |val|
        int best = 0;
        double best_dist = fabs(abs_val - grid[0]);
        for (int k = 1; k < n_grid; k++) {
            double dist = fabs(abs_val - grid[k]);
            if (dist < best_dist) {
                best_dist = dist;
                best = k;
            }
        }
        // Reconstruct: sign(val) * grid[best]
        double recon = (val >= 0.0 ? 1.0 : -1.0) * grid[best];
        double err = val - recon;
        mse += err * err;
    }
    return sqrt(mse / n_samples);
}

// Compute RMSE for OLD grid (which has both positive and negative entries, with sign bit)
static double compute_rmse_old_grid(const double *grid, int n_grid, double *samples, int n_samples) {
    double mse = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double val = samples[i];
        double best_err = 1e30;
        // Try each grid entry in both polarities (sign bit)
        for (int k = 0; k < n_grid; k++) {
            double g = grid[k];
            double e_pos = (val - g) * (val - g);
            double e_neg = (val + g) * (val + g);
            if (e_pos < best_err) best_err = e_pos;
            if (e_neg < best_err) best_err = e_neg;
        }
        mse += best_err;
    }
    return sqrt(mse / n_samples);
}

int main(void) {
    double sigma = 1.0 / sqrt((double)DIM);

    printf("=== TurboQuant Polar Grid Optimization ===\n\n");
    printf("Block dimension: %d\n", DIM);
    printf("Sigma (std of normalized components): %.10f\n", sigma);
    printf("Number of sample vectors: %d\n", NUM_VECTORS);
    printf("Total samples: %d\n\n", NUM_SAMPLES);

    // Generate normalized samples
    printf("Generating %d normalized random vectors...\n", NUM_VECTORS);
    srand(12345);

    double *samples = (double *)malloc(NUM_SAMPLES * sizeof(double));
    if (!samples) {
        fprintf(stderr, "Failed to allocate sample array\n");
        return 1;
    }

    for (int v = 0; v < NUM_VECTORS; v++) {
        double vec[DIM];
        double norm_sq = 0.0;
        for (int j = 0; j < DIM; j++) {
            vec[j] = rand_normal();
            norm_sq += vec[j] * vec[j];
        }
        double norm = sqrt(norm_sq);
        double inv_norm = (norm > 0.0) ? 1.0 / norm : 0.0;
        for (int j = 0; j < DIM; j++) {
            samples[v * DIM + j] = vec[j] * inv_norm;
        }
    }

    // Compute statistics
    double mean = 0.0, var = 0.0, min_abs = 1e30, max_abs = 0.0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        double v = samples[i];
        mean += v;
        var += v * v;
        double a = fabs(v);
        if (a < min_abs) min_abs = a;
        if (a > max_abs) max_abs = a;
    }
    mean /= NUM_SAMPLES;
    var = var / NUM_SAMPLES - mean * mean;

    printf("Sample statistics:\n");
    printf("  Mean:     %.10f (expected ~0)\n", mean);
    printf("  Variance: %.10f (expected %.10f = 1/%d)\n", var, 1.0 / DIM, DIM);
    printf("  Std dev:  %.10f (expected %.10f)\n", sqrt(var), sigma);
    printf("  |min|:    %.10f\n", min_abs);
    printf("  |max|:    %.10f\n\n", max_abs);

    // Old grids (current values)
    double old_grid_4[4] = {-0.8660254038, -0.2886751346, 0.2886751346, 0.8660254038};
    double old_grid_8[8] = {
        -0.9238795325, -0.7071067812, -0.3826834324, -0.1305261922,
         0.1305261922,  0.3826834324,  0.7071067812,  0.9238795325
    };

    // ================================================================
    // 4-entry grid optimization (for TQ3_0)
    // ================================================================
    printf("--- 4-entry grid (TQ3_0, with sign bit = 8 effective levels) ---\n\n");

    double grid_4[4];
    compute_optimal_grid_analytical(4, sigma, grid_4);
    printf("Analytical (equal-probability half-normal centroids):\n");
    for (int i = 0; i < 4; i++) printf("  grid_4[%d] = %.10f\n", i, grid_4[i]);

    // Refine with Lloyd's on actual samples
    lloyd_refine(grid_4, 4, samples, NUM_SAMPLES, 1000);
    printf("\nAfter Lloyd's refinement:\n");
    for (int i = 0; i < 4; i++) printf("  grid_4[%d] = %.10f\n", i, grid_4[i]);

    double rmse_old_4 = compute_rmse_old_grid(old_grid_4, 4, samples, NUM_SAMPLES);
    double rmse_new_4 = compute_rmse(grid_4, 4, samples, NUM_SAMPLES);
    printf("\nOld grid RMSE (TQ3): %.10f\n", rmse_old_4);
    printf("New grid RMSE (TQ3): %.10f\n", rmse_new_4);
    printf("Improvement: %.2f%%\n\n", 100.0 * (rmse_old_4 - rmse_new_4) / rmse_old_4);

    // ================================================================
    // 8-entry grid optimization (for TQ4_0)
    // ================================================================
    printf("--- 8-entry grid (TQ4_0, with sign bit = 16 effective levels) ---\n\n");

    double grid_8[8];
    compute_optimal_grid_analytical(8, sigma, grid_8);
    printf("Analytical (equal-probability half-normal centroids):\n");
    for (int i = 0; i < 8; i++) printf("  grid_8[%d] = %.10f\n", i, grid_8[i]);

    // Refine with Lloyd's on actual samples
    lloyd_refine(grid_8, 8, samples, NUM_SAMPLES, 1000);
    printf("\nAfter Lloyd's refinement:\n");
    for (int i = 0; i < 8; i++) printf("  grid_8[%d] = %.10f\n", i, grid_8[i]);

    double rmse_old_8 = compute_rmse_old_grid(old_grid_8, 8, samples, NUM_SAMPLES);
    double rmse_new_8 = compute_rmse(grid_8, 8, samples, NUM_SAMPLES);
    printf("\nOld grid RMSE (TQ4): %.10f\n", rmse_old_8);
    printf("New grid RMSE (TQ4): %.10f\n", rmse_new_8);
    printf("Improvement: %.2f%%\n\n", 100.0 * (rmse_old_8 - rmse_new_8) / rmse_old_8);

    // ================================================================
    // Output final values in C format
    // ================================================================
    printf("=== FINAL GRID VALUES (for code update) ===\n\n");

    printf("// TQ3_0: 4 positive grid values (sign bit handles negation)\n");
    printf("// Format for turboquant.h (all positive, sign bit doubles to 8 levels)\n");
    printf("static const float TQ_POLAR_GRID_4[4] = {\n");
    for (int i = 0; i < 4; i++) {
        printf("    %.10ff,%s\n", grid_4[i], (i < 3) ? "" : "");
    }
    printf("};\n\n");

    printf("// TQ4_0: 8 positive grid values (sign bit handles negation)\n");
    printf("static const float TQ_POLAR_GRID_8[8] = {\n");
    for (int i = 0; i < 8; i++) {
        printf("    %.10ff,%s\n", grid_8[i], (i < 7) ? "" : "");
    }
    printf("};\n\n");

    // INT8 versions for MMQ (scaled by 127)
    printf("// INT8 versions for CUDA MMQ (round(grid * 127)):\n");
    printf("static __device__ const int8_t TQ_GRID_4_I8[4] = {");
    for (int i = 0; i < 4; i++) {
        int v = (int)(grid_4[i] * 127.0 + 0.5);
        printf("%d%s", v, (i < 3) ? ", " : "");
    }
    printf("};  // round(grid * 127)\n");

    printf("static __device__ const int8_t TQ_GRID_8_I8[8] = {");
    for (int i = 0; i < 8; i++) {
        int v = (int)(grid_8[i] * 127.0 + 0.5);
        printf("%d%s", v, (i < 7) ? ", " : "");
    }
    printf("};  // round(grid * 127)\n");

    free(samples);
    return 0;
}
