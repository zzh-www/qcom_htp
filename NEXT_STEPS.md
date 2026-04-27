# V8C8 matmul — 256³ aligned to 1.33× of native (2026-04-27 night)

> **Status**: BbbKMajor (V8C8) custom op end-to-end matmul on QNN HTP v75
> at 256³ runs at **24 µs steady-state vs Native QNN MatMul at 18 µs =
> 1.33× of native**, **bit-exact 65536/65536**, lowered graph **6 nodes
> (vs native's 8)**. The matmul kernel itself (BbbKMajor) is at parity
> with native ConvLayer_s1.opt (~1.05× cyc, both ~9.3-9.8K cyc at 256³).
>
> Standard operating procedure persisted at **`docs/qnn_custom_op_sop.md`**.
> Build/run: `EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh`,
> then `M=256 K=256 N=256 bash standard_flow/phaseB_v8/run_v8c8_phase2.sh`.

## What landed (2026-04-27 day → night session)

### Step 1: HMX inline-asm kernel against Crouton_8 (bit-exact 32³–1024³)
- `src/HmxMatMulV9SkelOp.cpp` `V9_KERNEL_HMX` branch reads input via
  `qhpi_tensor_block_table`, writes output via `:after:cm:sat.ub`.
- `act_tile = block_table[(mt/mt_per_block)*k_chunks+kt] + (mt%mt_per_block)*1024`
  where `mt_per_block = block_rows/32 ∈ {1, 2}`.
- Bias native-fold layout (256 B/N-tile: lower 128 B fp16 pair, upper
  128 B int32 effective) loaded in single `bias = mxmem2`.
- S<128 falls back to scalar (HMX `:cm` SIGSEGVs on stack DDR scratch).

### Step 2: Hot-loop perfectly aligned with native (1.02×)
- Pre-bake `act_ptrs[M_t × K_t]` and `wt_ptrs[N_t × K_t]` into 4 KB stack
  arrays at kernel entry; inner loop only does `memw` ptr loads.
- Hexagon `loop0(1f, K_t/2):endloop0` + 4-packet/2-MAC body (dual-memw
  post-inc + dual mxmem). 2 cyc/MAC body throughput.
- Removed redundant defensive output-zero loop (HMX sat.ub writes every
  byte of the output anyway). 256³ bbb cyc went 107K → 9.5K (11.3× speedup).
- 256³ matmul kernel cyc: **9,533 vs native 9,335 = 1.02× ✓**.

### Step 3: Crouton_8 OUTPUT — eliminate UntileToRowMajor (5.83× → 1.39×)
- Output sig flipped to `Crouton_8 + Indirect + TCM_Only`.
- `UntileToRowMajor` op deleted from ONNX graph.
- QNN compiler auto-handles Crouton_8 → user row-major DDR (no explicit
  `q::ForceFormat_Flat` node in lowered graph at 256³, but bytes land
  bit-exact via framework Output op).
- 256³ e2e: **25 µs (was 105 µs) = 4.2× wall improvement**.
- Lowered graph: 6 nodes (vs native's 8 — q::*InputSlice +
  q::ForceFormat_Crouton + 2× q::ConvLayer.opt.weights_to_vtcm +
  BbbKMajor + q::Reshape).
- 1024³ V8C8 = 348 µs vs native 512 µs — **we're FASTER (0.68×)**.

### Step 4: QHPI prepare-time hooks for QNN scheduler integration
- `cost_function = M_t × N_t × K_t × 16 cyc` (rough HMX MAC estimate).
- `shape_required = {1, 1, 32, 32}` (HMX-tile alignment).
- `shape_legalized` rounds N up to ×32 multiples.
- `tile_output = 0` (output[0] is tile-friendly).
- 256³ e2e: 25 µs → **24 µs = 1.33× of native** (median of 5 runs).
- Improvement concentrated in Input op (-11%) and reshape (-18%);
  matmul kernel unchanged.
- Cost-coefficient sweep (4×/16×/64×) shows <1% diff — within noise.

## Per-shape perf summary (steady-state iter 3)

| S    | V8C8 µs | Native µs | wall ratio | bbb cyc  | Native MM cyc | kernel ratio |
|------|---------|-----------|------------|----------|---------------|--------------|
| 256  | 24      | 18        | **1.33×**  | 9,800    | 9,335         | **1.05× ✓**  |
| 512  | 66      | 39        | 1.69×      | 61,300   | 39,447        | 1.55×        |
| 1024 | 348     | 512       | **0.68× (faster)** | 459,000 | 1,167,447 | 0.39× |

## 256³ residual gap (1.33×) — anatomy

| op group           | V8C8 cyc | Native cyc | gap     | reason |
|--------------------|----------|------------|---------|--------|
| Input + ForceFormat_Crouton | 9,000   | 3,550 + part of MatMul_0 | ~+5K | custom op can't overlap with HMX downstream |
| wt + bias DMAs     | 2,800   | folded into MatMul_0 (~0) | ~+2.8K | same — no fusion |
| matmul kernel      | 9,800   | 9,335 | ~+0.5K (noise) | parity |
| Reshape (Crouton→flat) | 3,100 | 0 (null_exec) | +3.1K | native splits into ForceFormat_Flat +
Reshape with the heavy work absorbed into Output op |
| Output (DDR copy)  | 2,500   | 4,931 | -2.4K | we save here |
| **Total**          | **27,200** | **17,816** | **+9.4K (1.53× cyc)** | (1.33× wall) |

The ~9K cyc residual is the **"custom op tax"** — QHPI v1 doesn't
expose the same scheduling/pipelining surface as native ConvLayer-class
ops, so layout-conversion + DMA can't overlap with our matmul.

## Known issues

- **S=128 broken** (Step 3): QNN allocates output Crouton_8 with
  `block_size=512 B` at S=128 (vs `2048 B` at S≥256). Our HMX 1024-byte
  `:cm:sat.ub` write overflows the 512 B block boundary →
  ~30% bit-exact only.
  - Workaround (not done): add a 4th VTCM scratch input (V8 prod
    pattern) at sig level; HMX writes scratch, scalar splits to two
    output blocks per HMX tile. Or runtime-detect output block_size
    and use a different write strategy.
- **S=64, S=32**: same root cause + scalar fallback also assumes input
  block layout matches output's. Fixable with same workaround.

## What's next (priorities)

### A. Close S=128/64/32 with 4-input scratch path (medium effort)
- Add `Layout_Flat4 + Direct + TCM_Only` 4th input of size 1024 B.
- At runtime: if `out_block_size < 1024`, redirect HMX `:after:cm:sat.ub`
  to scratch, then memcpy split into two output blocks.
- Sig change → XML + gen_v8c8_test.py + run script all need updating.
- Expected: bit-exact at all shapes 32³–1024³.

### B. Investigate `build_tile` for ≥1.2× e2e at 256³ (medium-high effort, uncertain payoff)
- Implement `QHPI_BuildTileOfOp` that splits BbbKMajor by N axis into
  N_t sub-ops; each sub-op + its surrounding ops form a pipeline-able
  batch. QNN should overlap tile_i HVX with tile_{i+1} HMX.
- Use `qhpi_op_slice` to slice wt/bias along N; `qhpi_op_create` to
  construct sub-graphs.
- Expected: 5-10% cycle savings at 256³ (hard to predict — native's
  ConvLayer fusion happens at a deeper layer than tile-level).

### C. ≥2048³ multi-instance graph split (low-medium effort)
- 2048³ exceeds VTCM 6 MiB ceiling (single-instance V8C8 needs ~12 MiB
  intermediate). Need to port `gen_v8_graph.py`'s adaptive M_TILE/N_TILE
  splitting into `gen_v8c8_test.py`.
- 1024³ V8C8 already FASTER than native (0.68×). 2048³+ should also be
  faster after split, given amortization works in our favor.

### D. End-to-end perf on a real model with multiple matmuls (test)
- Wire BbbKMajor into a 2-3 matmul transformer attention block via
  ONNX gen-script; verify perf scales linearly.
- Custom-op tax is fixed per call, so longer chains get closer to native
  ratio.

### E. (deferred) Real (non-degenerate) quant scales
- Current test uses `scale=1.0, out_zp=0` to stay in `saturate_u8(acc)`
  regime. Validate with random scale + non-zero zp via existing fp16
  pair encoding. Should "just work" since the math is already
  silicon-verified at V8 prod.

## Key code paths

| file | purpose |
|------|---------|
| `example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp` | BbbKMajor implementation (HMX inline asm + Crouton_8 sig + QHPI hooks). The main artifact of this work. |
| `example/hmx_matmul_phase3/src/HmxMatMulV8Op.cpp` | V8 prod (legacy path). Reference for hot-loop optimization. |
| `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_v8c8_test.py` | ONNX gen with native-fold bias byte layout |
| `example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v8c8_phase2.sh` | Build + run flow |
| `example/hmx_matmul_phase3/standard_flow/phaseB_v8/sweep_v8c8_shapes.sh` | Shape sweep harness |
| `example/hmx_matmul_phase3/standard_flow/phaseB_v8/MatMulV8Package.xml` | OpDef declarations (incl. BbbKMajor with Crouton_8 sig) |

## Reference reading (Agent/)

- `v8c8_step3_crouton8_output_2026-04-27.md` — Step 3 detail (Crouton_8 output)
- `v8c8_step4_qhpi_hooks_2026-04-27.md` — Step 4 detail (prepare hooks + analysis)
- `v8_c8_kernel_perf_hwloop_2026-04-27.md` — Step 2 detail (hw loop0 + pre-baked ptrs)
- `v8_c8_kernel_phase3_2_hmx_bit_exact_2026-04-27.md` — Step 1 (HMX kernel body)
- `v8_c8_kernel_phase3_native_fold_2026-04-27.md` — bias native-fold byte layout decode
- `qnn_re/bias_to_vtcm_decoded_2026-04-27.md` — Native bias_to_vtcm reverse engineering
- `qnn_primitive_alignment_phase01_2026-04-26.md` — overall C8 alignment plan
- `dlsym_spike_PASS_2026-04-25.md` — cross-.so dlsym to libQnnHtpV75Skel.so works
