# TurboQuant Block Layout Design

## Background

TurboQuant is a two-stage compression scheme:

1. **PolarQuant** — Converts each block of values to polar coordinates (radius + angles).
   The radius (norm) is the only per-block scalar. Angle indices map into a
   **fixed precomputed grid** — no per-sub-block scales needed ("zero overhead").

2. **QJL** — 1-bit sign correction per element via Quantized Johnson-Lindenstrauss
   transform. Reduces residual quantization error with minimal storage.

Target: 3–4 effective bits per weight, matching the paper's claims.

## Design Constraints

- Must fit GGML's block-based quantization model
- Super-block size QK_K = 256 (consistent with K-quants, IQ types)
- GPU-friendly memory layout (avoid unaligned access, enable coalesced reads)
- Lookup table for polar grid (similar to IQ types' grid tables)
- Clean bit packing for CUDA `dp4a` compatibility

## Block Structures

### block_tq3_0 — ~3.0625 bpw

```
QK = 256 values per super-block

Layout (98 bytes total):
┌──────────────────────────────────────────────────┐
│ ggml_half d              (2 bytes)  — radius     │
│ uint8_t   al[64]         (64 bytes) — 2-bit      │
│                           angle grid indices      │
│ uint8_t   signs[32]      (32 bytes) — 1-bit      │
│                           QJL sign corrections    │
└──────────────────────────────────────────────────┘

Per-element breakdown:
  - 2 bits: angle index → 4-entry polar grid lookup
  - 1 bit:  QJL sign correction (+1 / -1)
  Total: 3 bits + 0.0625 bits overhead (fp16 d / 256) = 3.0625 bpw
```

Dequantization: `x[i] = d * polar_grid_4[al[i]] * (signs[i] ? -1 : +1)`

### block_tq4_0 — ~4.0625 bpw

```
QK = 256 values per super-block

Layout (130 bytes total):
┌──────────────────────────────────────────────────┐
│ ggml_half d              (2 bytes)  — radius     │
│ uint8_t   al[64]         (64 bytes) — 2-bit      │
│                           angle grid indices (lo) │
│ uint8_t   ah[32]         (32 bytes) — 1-bit      │
│                           angle grid indices (hi) │
│ uint8_t   signs[32]      (32 bytes) — 1-bit      │
│                           QJL sign corrections    │
└──────────────────────────────────────────────────┘

Per-element breakdown:
  - 3 bits: angle index (2 lo + 1 hi) → 8-entry polar grid lookup
  - 1 bit:  QJL sign correction (+1 / -1)
  Total: 4 bits + 0.0625 bits overhead (fp16 d / 256) = 4.0625 bpw

The lo/hi split follows the Q5_0/Q5_1 pattern for clean GPU access.
```

Dequantization: `x[i] = d * polar_grid_8[al_lo[i] | (ah[i] << 2)] * (signs[i] ? -1 : +1)`

## Polar Grid Lookup Tables

Like the IQ types use precomputed importance-weighted grids, TurboQuant uses
precomputed **polar angle grids**. The key insight from the paper is that
normalized vectors in high dimensions have angles that concentrate around
known distributions, enabling fixed grids that work across all datasets
("data-oblivious").

### 4-entry grid (for TQ3_0)

Maps 2-bit index → normalized polar coordinate value.

```c
// Placeholder — actual values derived from angular concentration analysis
static const float tq_polar_grid_4[4] = {
    -0.8660254f,  // -sqrt(3)/2  ≈ cos(5π/6)
    -0.2886751f,  // -1/sqrt(12) ≈ cos(π/2 + π/12)
     0.2886751f,  //  1/sqrt(12)
     0.8660254f,  //  sqrt(3)/2  ≈ cos(π/6)
};
```

### 8-entry grid (for TQ4_0)

Maps 3-bit index → normalized polar coordinate value.

```c
// Placeholder — actual values derived from angular concentration analysis
static const float tq_polar_grid_8[8] = {
    -0.9238795f,  // cos(7π/8)
    -0.7071068f,  // cos(3π/4) = -sqrt(2)/2
    -0.3826834f,  // cos(5π/8)
    -0.1305262f,  // cos(π/2 + δ)
     0.1305262f,  // cos(π/2 - δ)
     0.3826834f,  // cos(3π/8)
     0.7071068f,  // cos(π/4) = sqrt(2)/2
     0.9238795f,  // cos(π/8)
};
```

These are initialized as uniform angular spacing. The actual optimal grid points
should be computed from the angular concentration patterns described in the
PolarQuant paper (AISTATS 2026). The grid values can be refined empirically
by minimizing reconstruction error on reference model weights.

## QJL Error Correction

The QJL stage corrects residual error from PolarQuant quantization:

1. Compute residual: `r[i] = x[i] - d * polar_grid[angle_idx[i]]`
2. Apply random JL projection (sign-flip matrix): `projected = JL * r`
3. Store sign bits: `signs[i] = (projected[i] > 0) ? 1 : 0`

During dequantization, the sign bit acts as a ±1 correction factor applied
to the polar-reconstructed value. This is mathematically equivalent to
choosing between two reconstruction candidates and picking the one closer
to the original.

**Simplified model for GGML**: Rather than a full JL projection (which
requires storing/regenerating the random matrix), we use the sign of the
residual directly as the correction factor:

```
quantize:   signs[i] = (x[i] >= 0) ? 0 : 1  // relative to polar reconstruction
dequant:    x[i] = d * polar_grid[idx] * (1 - 2*signs[i])
```

This captures the dominant error mode (sign errors in the polar reconstruction)
with zero memory overhead for the projection matrix.

## GPU Kernel Considerations (RTX 5090 / SM 100)

### Memory Layout
- `al[]` is contiguous for coalesced 32-bit reads (4 elements per byte, warp reads 32 bytes = 128 elements)
- `ah[]` and `signs[]` are 1-bit packed, natural for `__popc()` / bitwise ops
- Total block fits in 130 bytes (TQ4_0) — fits in 2-3 cache lines

### Dot Product Strategy
For the query (Q8_1) × key (TQ4_0) dot product:
1. Load angle indices → lookup grid values (shared memory LUT, 8 floats = 32 bytes)
2. Apply sign correction via XOR with sign bits
3. Scale by `d`
4. Accumulate with `dp4a` after converting to int8 representation

Alternative: fused dequant + FMA without int8 intermediate (may be faster on SM 100
with its enhanced FP throughput).

### Flash Attention Integration
For KV cache quantization (the primary use case):
- Keys stored as TQ4_0/TQ3_0 blocks
- During attention: dequantize K on-the-fly in registers
- The small LUT (4 or 8 entries) fits in registers or shared memory
- QJL signs are cheap — single AND + conditional negate

## Comparison with Existing Types

| Type     | bpw    | Scale overhead | Grid/LUT | Error correction |
|----------|--------|----------------|----------|-----------------|
| Q4_0     | 4.5    | fp16 per 32    | None     | None            |
| Q4_K     | 4.5    | fp16 + 6-bit   | None     | None            |
| IQ4_XS   | 4.25   | fp16 + 6-bit   | 256-entry| None            |
| IQ3_XXS  | 3.0625 | fp16 per 256   | 256-entry| None            |
| **TQ4_0**| 4.0625 | fp16 per 256   | 8-entry  | 1-bit QJL       |
| **TQ3_0**| 3.0625 | fp16 per 256   | 4-entry  | 1-bit QJL       |

Key advantages over IQ types:
- Much smaller lookup tables (4/8 vs 256 entries) → better GPU cache utilization
- QJL correction recovers accuracy that would otherwise need larger grids
- Data-oblivious: no dataset-dependent grid optimization needed
- Simpler quantization: no importance-matrix computation required
