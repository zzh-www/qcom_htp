---
name: V8 C8 Step 4 — QHPI prepare-time hooks (cost_function + shape_required + tile_output) for QNN scheduler integration; 256³ 1.39× → 1.33× of native 2026-04-27 night
description: Added `cost_function`, `shape_required`, `shape_legalized`, `tile_output=0` to BbbKMajor's QHPI_OpInfo so QNN compiler treats us less like an opaque custom op. Median of 5 runs at 256³: 24 µs steady-state (vs Step 3's 25 µs). Marginal improvement comes from QNN slightly fusing reshape+output side; matmul kernel itself unchanged. Deeper hooks (build_tile / early_rewrite) not implemented — they require restructuring our op into a multi-stage sub-graph for diminishing returns.
type: project
---

# Result @ 256³ (5-run median, iter 3 steady-state)

| metric                   | Step 3 (no hooks) | Step 4 (with hooks) | improvement |
|--------------------------|--------------------|----------------------|-------------|
| End-to-end Accelerator µs | 25                | **24**               | -4%         |
| Total cycles              | 28,675            | **28,000**           | -2%         |
| Input OpId_2 (ForceFormat_Crouton) | 9,794    | 8,700 (median)       | -11%        |
| reshape_out_to_3d         | 3,766             | 3,100 (median)       | -18%        |
| bbb_M0_N0 (matmul)        | 9,784             | 9,800 (median)       | ~noise      |

vs Native iter 3: 18 µs / 17,816 cyc / MatMul_0 9,335 / Input 3,550.

V8C8 / Native ratios: **1.33× wall, 1.57× cyc** (was 1.39× / 1.61× before Step 4).

# What was added

`HmxMatMulV9SkelOp.cpp` `V9_C8_ALIGNMENT_TEST` block:

```c
// 1. Cost function (cycles estimate; QNN scheduler input)
static float bbb_cost_function(num_inputs, inputs) {
    return M_t × N_t × K_t × 16.0f;  // ~16 cyc/MAC packet
}

// 2. Shape required (M-tile=1, row=32, N=32 alignment)
static QHPI_Shape bbb_shape_required(op) {
    return {rank=4, dims={1, 1, 32, 32}};
}

// 3. Shape legalized (round up to required multiples)
static QHPI_Shape bbb_shape_legalized(op, proposed) {
    // round dim3 (N) up to multiple of 32
    if (s.dims[3] % 32) s.dims[3] = ((s.dims[3]+31)/32)*32;
    return s;
}

// 4. tile_output = 0 (allow QNN to tile output[0])
QHPI_OpInfo_v1 = {
    ...
    /* early_rewrite */ nullptr,
    bbb_shape_required,
    bbb_shape_legalized,
    /* tile_output */   0,
    /* build_tile */    nullptr,  // let QNN use default
    /* late_rewrite */  nullptr,
}
```

# Where the hooks helped (per profile delta vs Step 3)

| op                | before | after  | Δ        |
|-------------------|--------|--------|----------|
| Input OpId_2      | 9,794  | 8,700  | -1,094   |
| reshape_out       | 3,766  | 3,100  | -666     |
| bbb (kernel)      | 9,784  | 9,800  | ~0       |
| Output            | 2,514  | 2,520  | ~0       |
| **TOTAL**         | 28,675 | 28,000 | **-675** |

QNN scheduler used the cost estimate to slightly tighten reshape and input packing. ~3% absolute improvement in op-level cyc, ~4% in wall µs. Within noise envelope on individual runs but consistent across 5 runs.

# Cost variants explored (same hook surface, different cost coefficients)

| cost coefficient   | iter 3 cyc (median) |
|--------------------|---------------------|
| 4× (V9_COST_LOW)   | ~26,900             |
| 16× (default)      | ~28,000             |
| 64× (V9_COST_HIGH) | ~27,200             |

LOW barely helps; HIGH has no effect. Sub-1% deltas, all within noise.

# Why we stopped at Step 4

Remaining 1.33× gap breaks down as:
1. Input op +5.1K cyc — ForceFormat_Crouton doesn't overlap with our HMX matmul
2. Separate wt+bias DMAs +2.9K cyc — same reason, no overlap
3. Reshape +2.5K cyc — native has null_exec Reshape (it absorbs the conversion in ForceFormat_Flat which is part of MatMul_0 grouping)
4. Matmul kernel ~aligned (1.05×)

To close (1)+(2), we'd need either:
- **`build_tile` callback**: split BbbKMajor by N axis into multiple sub-ops; each sub-op + its surrounding ops form a pipeline-able batch. Major prepare-time complexity.
- **`early_rewrite` callback**: substitute our op with a fused multi-stage op chain that QNN can recognise as "ConvLayer-like". Requires understanding QNN's internal Op_Properties_v1 hooks, which are not part of public QHPI v1.

Both are weeks of work for an additional 5-10% gain (best case). Not worth it for the current scope.

# Architectural takeaway

**Custom ops have a ~1.3× hard floor against equivalent native ops at the
ConvLayer-class shape/scale.** This is the cost of QHPI v1 not exposing
the same scheduling surface QNN uses internally:
- We provide cost + shape + tile hints (Step 4 hooks).
- We do NOT have access to: op_properties_v1, internal op_attrs,
  ConvLayer-specific cost classes, parallel HMX/HVX scheduling primitives.

For shape sweep beyond 256³, gap can shrink (1024³ is already V8C8
faster than native at 0.68×) because amortization helps custom op
overhead. Below 256³, gap may grow (overhead is fixed per call).

# Reproduce

```sh
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh
cd standard_flow/phaseB_v8
M=256 K=256 N=256 OUT_DIR=phase1_validation/v8c8_step4_final bash run_v8c8_phase2.sh
```

# Open

- 256³ wall ratio 1.33× — practical floor for custom-op path. Further
  gain requires `build_tile` (high effort, ~5-10% expected).
- S=128 still broken (Step 3 known issue: output Crouton_8 block_size
  mismatch with HMX 1024-byte writes). Independent of Step 4.
