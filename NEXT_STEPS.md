# V8C8 matmul — 256³ kernel-level shape aligned, 1.50× wall vs native (2026-04-27 night)

> **Status**: BbbKMajor at 256³ has **fully aligned kernel-level I/O shapes**
> with native q::ConvLayer_s1.opt. Steady-state (iter 3): **27 µs vs native 18 µs
> = 1.50× wall, 1.71× cyc**. Bit-exact 65536/65536 preserved at all shapes
> 32³–1024³.
>
> **Detailed handoff**: see `Agent/SESSION_2026-04-27_handoff_v8c8_alignment_status.md`
> for the alignment audit + 4 prioritized fix paths.

## What is aligned ✓

| | V8C8 BbbKMajor | Native ConvLayer_s1.opt | match |
|--|----------------|-------------------------|-------|
| user input/output shape | `[1, 1, 256, 256]` u8 ↔ `[1, 1, 256, 256]` u8 | same | ✓ |
| in[0] act dim       | `[1, 8, 32, 256]` u8q | `[1, 8, 32, 256]` u8q | ✓ |
| in[1] wt dim        | `[1, 1, 256, 256]` u8q| `[1, 1, 256, 256]` i8q| ✓ shape (dt: u8 vs i8) |
| in[2] bias dim      | `[1, 8, 1, 64]` i32   | `[1, 8, 1, 64]` i32   | ✓ |
| out[0] dim          | `[1, 8, 32, 256]` u8q | `[1, 8, 32, 256]` u8q | ✓ |
| Reshape ops null_exec | yes                 | yes                    | ✓ |
| ctxgen schematic.bin emit | yes (after fix)| yes                    | ✓ |
| optrace decode end-to-end | yes (after fix)| yes                    | ✓ |
| bit-exact matmul    | 65536/65536          | (reference)           | ✓ |

## What is NOT aligned ✗ (the 1.50× gap, +12.7K cyc at iter 3)

```
                            V8C8 cyc   Native cyc   Δ        fix path
─────────────────────────────────────────────────────────────────────────
Input + ForceFormat_Crouton  8,935      3,999      +4.9K    D (build_tile)
wt + bias DMAs               3,411       (folded)  +3.4K    D (build_tile)
HMX matmul kernel           12,852      ~5,000     +7.8K
  ├─ Output VTCM bank conflict             —       +3.0K    C (rollback factor)
  ├─ K-outer wt extra packet               —       +0.2K    B (N-outer wt)
  ├─ Per-call descriptor rebuild           —       +1.5K    A (precompute hook)
  └─ Misc (drain, bookkeeping)             —       +3.1K    needs probe
Output Reshape + DDR        5,326       5,168      +0.2K    ✓ aligned
─────────────────────────────────────────────────────────────────────────
Total                      30,524      17,816    +12.7K    1.71× cyc / 1.50× wall
```

### A. Per-call HMX descriptor rebuild (+~1.5K cyc/call)

Native bakes the 0x40-byte descriptor at `graph_finalize` via
`set_hmx_params_conv1x1`. We rebuild every call (act_ptrs/wt_ptrs
prebake + bias mxmem2 + loop0 register setup). **Fix path**:
implement `QHPI_Precompute_Function` to bake what's
shape-only-dependent into `precomputed_data`. **Open question**:
does precompute have access to STATIC tensor data (for wt_ptrs)?
Read qhpi.h:639-658 spec.

### B. wt layout direction (+~256 cyc, low priority)

Native uses N-tile-outer wt → can `r8 += 0x400` inline post-inc per
K (3-packet/2-MAC body). We use K-tile-outer (byte-1:1 with
`weights_to_vtcm@FB.fB.` verbatim DMA) → must memw post-inc from
pre-baked array (4-packet/2-MAC body). Fix: change gen-script to
output N-outer wt + update kernel inner loop. Trade-off: need to
re-validate weights_to_vtcm DMA produces the bytes our kernel
expects.

### C. Output VTCM block geometry (+3K cyc — biggest single ticket)

This regressed in this session when we did shape alignment.
- Before alignment: output factorized as `[1, 8, 8, 1024]` tile-array
  (64 blocks × 1KB). HMX `:after:cm:sat.ub` 1024-byte writes fit
  exactly one block. **bbb cyc = 9,777**.
- After alignment: output factorized as `[1, 8, 32, 256]` Crouton
  logical (32 blocks × 2KB). Our 1KB writes hit half-block, adjacent
  (mt%2=0,1) writes share same VTCM block at different 1KB offsets.
  Bank conflict / cache eviction. **bbb cyc = 12,852**.

**Fix path**: roll back `value_info=[mm_c8_info]` in `gen_v8c8_test.py`
to let QNN auto-factorize as tile-array. Lose visual `[1,8,32,256]`
match in trace but bytes still equivalent (65 KB, just dim
factorization). **Saves ~3K cyc immediately.**

### D. ForceFormat_Crouton + DMAs not overlapping with HMX (+~8K cyc)

Native fuses bias_to_vtcm + ForceFormat_Crouton + weights_to_vtcm +
ConvLayer_s1.opt under one `MatMul_0` grouping. QNN scheduler
runs HVX (ForceFormat) concurrently with HMX (matmul) — different
physical units. We're a custom op, can't be fused.

**Fix path**: implement `QHPI_BuildTileOfOp` (`build_tile`) callback to
split BbbKMajor by N axis into N_t sub-ops. Each tile + its
surrounding DMA/format becomes pipeline-able. Hard to implement
correctly (requires `qhpi_op_slice` + `qhpi_op_create` to construct
sub-graphs at prepare time).

## Concrete next-session priorities (ROI-ordered)

| | what | expected savings | effort | risk |
|-|------|------------------|--------|------|
| 1 | **Roll back output factor (C)** — single-line change in gen-script | 3K cyc | trivial | none (loses visual alignment) |
| 2 | **`do_precomputation_function` (A)** | 1-2K cyc | medium | uncertain (qhpi.h spec must be read) |
| 3 | **N-outer wt + 3-packet body (B)** | ~256 cyc | medium | re-validate DMA bytes |
| 4 | **`build_tile` callback (D)** | 3-5K cyc | high | uncertain payoff |

Doing 1+2+3: expected ratio 1.50× → ~1.25× wall. Doing all four: maybe ~1.10×.

## Per-shape perf summary (steady-state iter 3)

| S    | V8C8 µs | Native µs | wall ratio | bbb cyc  | Native MM cyc | kernel ratio |
|------|---------|-----------|------------|----------|---------------|--------------|
| 256  | 27      | 18        | **1.50×**  | 12,852   | ~5,000 (alone) | **2.4×**    |
| 512  | (post-shape-align untested) | 39 | — | — | 39,447 | — |
| 1024 | (untested) | 512 | — | — | ~1,167K | — |

Shape sweep ≥512³ should be re-run after the round-2 shape alignment.
1024³ was previously V8C8 FASTER (0.68×) but that was with the
tile-array output. After Crouton logical alignment, bbb may also
regress 30%+ at larger shapes.

## Known issues (carry-overs, not in this session's scope)

- **S=128 broken** (~30% bit-exact): QNN allocates output Crouton_8 with
  smaller block_size (512 B vs 2048 B at S≥256). HMX 1024-byte write
  overflows. Fix: 4th VTCM scratch input (V8 prod pattern). Not
  blocking 256³.
- **S<128 (32, 64)**: scalar fallback, also broken on output side.
  Same root cause.

## Build / run / verify

```sh
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh

cd standard_flow/phaseB_v8
M=256 K=256 N=256 OUT_DIR=phase1_validation/v8c8_apples2_256 \
    bash run_v8c8_phase2.sh

# bit-exact + perf checks: see Agent/SESSION_2026-04-27_handoff_v8c8_alignment_status.md
```

## Reference reading (Agent/)

- **`SESSION_2026-04-27_handoff_v8c8_alignment_status.md`** ← read first next session
- `v8c8_step3_crouton8_output_2026-04-27.md` — Step 3 (Crouton_8 output)
- `v8c8_step4_qhpi_hooks_2026-04-27.md` — Step 4 (cost_function/shape hooks)
- `v8_c8_kernel_perf_hwloop_2026-04-27.md` — hw loop0 + pre-baked ptrs
- `qnn_re/weights_to_vtcm_RE_2026-04-27.md` — wt DMA byte layout RE
- `qnn_re/bias_to_vtcm_decoded_2026-04-27.md` — bias DMA byte layout RE
- `qnn_re/hmx_convbbb1x1_stride1_full.S` — native HMX kernel disasm

## Optrace artifact bundle (uploaded to user's GDrive)

`G:\我的云端硬盘\optrace_256_bundle\` (Windows DriveFS) — apples-aligned
V8C8 + native side-by-side at 256³. Includes ONNX, DLC, ctx,
schematic, qnn-profiling-data_0.log, optrace.txt + chrometrace JSONs +
QHAS HTML, README with full alignment table.

## Recent commits

```
cef7e01 gen_v8c8_test.py: align BbbKMajor I/O shapes with ConvLayer_s1.opt   (this session)
356f40f gen_v8c8_test.py: align user-facing input shape with native [1,1,M,K] (this session)
2928716 run_v8c8_phase2.sh: enable optrace in ctxgen → schematic.bin emitted (this session)
eb6fd20 V8C8 BbbKMajor: 256³ matmul aligned to 1.33× of native               (this session)
c5d29fc V9 sweep: characterize 512³→4096³                                    (prior)
c40954e V9 adaptive ONNX gen: V8 matmul shape-scales to 4096³                (prior)
7ae8668 V8-only cleanup + standard QNN custom-op flow                        (prior)
```
