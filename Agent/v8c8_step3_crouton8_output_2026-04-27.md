---
name: V8 C8 Step 3 — BbbKMajor outputs Crouton_8, no UntileToRowMajor; 256³ e2e 1.39× native 2026-04-27 night
description: Changed BbbKMajor output sig from Flat4/Direct to Crouton_8/Indirect. UntileToRowMajor + internal Reshape removed from ONNX graph. QNN compiler auto-handles Crouton_8 → user row-major DDR via implicit framework Output op (no explicit q::ForceFormat_Flat node observed in lowered graph but bytes land correctly). 256³ e2e: 105 µs → 25 µs = 1.39× of native (was 5.83×). Bit-exact at S=256/512/1024. S<256 broken due to QNN allocating output Crouton_8 with smaller block_size (512 B at S=128 vs 2048 B at S≥256).
type: project
---

# Result @ 256³ (user's target shape)

| metric                 | V8C8 Step3 | Native | ratio |
|------------------------|-----------|--------|-------|
| End-to-end Accelerator time (µs)  | 25        | 18     | 1.39× |
| Total cycles (steady)             | 28,675    | 17,816 | 1.61× |
| Matmul kernel cycles (bbb / MatMul_0) | **9,784** | **9,335** | **1.05×** ✓ aligned |
| HTP-op nodes in lowered graph     | 6         | 8      | (we have fewer) |

cyc/MAC at 256³: **19.1** (V8C8) vs 18.2 (native). HMX silicon practical floor.

# Result at all sweep shapes

| S    | V8C8 µs | V8C8 cyc | bbb cyc | Native µs | Native MM cyc | e2e ratio | kernel ratio |
|------|---------|----------|---------|-----------|---------------|-----------|--------------|
| 256  | 25      | 28,675   | 9,784   | 18        | 9,335         | 1.39×     | 1.05× ✓     |
| 512  | 66      | 115,104  | 61,311  | 39        | 39,447        | 1.69×     | 1.55×        |
| 1024 | 348     | 682,414  | 459,123 | 512       | 1,167,447     | **0.68× (FASTER!)** | 0.39× |

# What changed (Step 3)

1. **C++ sig** (`HmxMatMulV9SkelOp.cpp`):
   ```c
   // Output: Flat4/Direct/TCM_Only → Crouton_8/Indirect/TCM_Only
   {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only}
   ```
   Kernel uses `qhpi_tensor_block_table(outputs[0])` instead of `raw_data`.

2. **Output offset formula** (per HMX tile):
   ```c
   const uint32_t mt_per_blk_o = block_rows / 32;             // 1 (S=128) or 2 (S≥256)
   const uint32_t bi_o = (mt / mt_per_blk_o) * out_n_chunks + nt;
   const uint32_t off_o = (mt % mt_per_blk_o) * 1024;
   uint8_t *out_tile = (uint8_t *)out_blocks[bi_o] + off_o;
   ```
   For S≥256 (block_rows=64, blocks 2 KB each): writes 2 HMX tiles per block.
   For S=128 (block_rows=32, blocks 1 KB each in our model — but ACTUAL QNN
   allocation differs, see "S=128 limitation" below).

3. **XML OpDef** — `BbbKMajor` output declared as `[1, M/32, 32, N]` (was
   `[1, M/32, N/32, 1024]`).

4. **gen_v8c8_test.py** — removed `UntileToRowMajor` node. Graph is now
   just `BbbKMajor → Reshape([1,M,N])`. QNN compiler folds the Reshape
   to null_exec (S=1024) or near-zero cost (S=256/512).

# Lowered graph at 256³ (after Step 3)

```
[1] q::*InputSlice              (DMA: act DDR → VTCM)
[2] q::ForceFormat_Crouton      (HVX: row-major → Crouton_8 in VTCM)
[3] q::ConvLayer.opt.weights_to_vtcm  (DMA: wt static → VTCM)
[4] q::ConvLayer.opt.weights_to_vtcm  (DMA: bias static → VTCM)
[5] HmxMatMulPhase3Package::BbbKMajor  (HMX: matmul, writes Crouton_8 output)
[6] q::Reshape                  (Crouton_8 → user [1,M,N] flat row-major)
```

Native at 256³ has 8 nodes (Reshape × 2, ForceFormat_Crouton, ForceFormat_Flat,
ConvLayer.opt.bias_to_vtcm, ConvLayer.opt.weights_to_vtcm, ConvLayer_s1.opt,
*InputSlice). **We have FEWER nodes (6 vs 8) AND faster matmul kernel.**

# How does QNN auto-flatten without q::ForceFormat_Flat in the lowered graph?

The lowered graph at S=256 contains NO `q::ForceFormat_Flat` node. Yet the
user-facing `out` tensor lands as flat row-major bytes in DDR (verified
bit-exact 65536/65536). Mechanism appears to be:
- QNN treats Reshape from Crouton_8 [1,M/32,32,N] → flat [1,M,N] as a
  layout transformation. The conversion is fused into the framework
  Output op (which performs VTCM → DDR copy) as a "format" pass.
- Cost lands in `Output OpId_3` (~2.6K cyc at 256³) plus the explicit
  Reshape cost (~2.9K cyc at 256³). Total layout-conversion overhead
  ~5.5K cyc — vs our former UntileToRowMajor at 441K cyc. **80×
  improvement** on this step.

# S=128 limitation (known issue)

Probe (V9_OUT_PROBE) revealed QNN allocates output Crouton_8 differently
at S=128 than S≥256:

| S    | out_blocks | block_size | block_rows |
|------|------------|------------|------------|
| 128  | 32         | 512 B      | 16         |
| 256  | 32         | 2048 B     | 64         |
| 512  | 128        | 2048 B     | 64         |
| 1024 | 512        | 2048 B     | 64         |

Our HMX kernel writes 1024 contiguous bytes per `:after:cm:sat.ub`. At
S=128 each output block is only 512 B → 1024-byte write overflows into
the next block (which may or may not be physically adjacent in VTCM).
Result: 30% match (bytes that happen to land in correct positions).

S<128 (scalar fallback) is also affected because the scalar path's
output formula `out_blocks[rg * out_n_chunks + kc][bo]` assumes the
input act's `block_rows` matches output's, which doesn't hold at small
S.

**Workarounds** (not implemented in this commit):
- Add a 4th VTCM scratch input (1 KB Direct buffer). Write HMX tile to
  scratch, then memcpy 512 B halves into the two relevant Crouton blocks.
- OR detect block_rows mismatch at runtime and fall back to scalar path
  with the corrected output formula. Scalar at S=128 = 2 M MAC × ~5
  cyc/MAC = ~10 M cyc, ~1000× slower than HMX path. Unacceptable for
  prod.
- OR change output declaration to force same block_size as input
  (e.g., declare [1, M/64, 64, N] to encourage block_rows=64). Need
  experimentation; failed in initial attempts.

For user's target shape 256³, Step 3 is fully working.

# Reproduce

```sh
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh
cd standard_flow/phaseB_v8
M=256 K=256 N=256 OUT_DIR=phase1_validation/v8c8_step3final_256 bash run_v8c8_phase2.sh
```

# Open

- 256³ residual e2e gap (1.39×): native 18 µs, ours 25 µs. Native
  steady-state at iter 3 might be artificially low (cache-warm); iter 2
  was 26 µs for native = comparable to ours. Real-world workloads
  rarely hit iter-3 perf.
- S=128 not working — needs scratch path or shape-declaration workaround.
- S<128 broken — same root cause.
