---
name: Session 2026-04-27 night handoff — V8C8 vs Native alignment status + remaining work
description: Hand-off doc for next session. V8C8 BbbKMajor at 256³ kernel-level I/O shapes now match native ConvLayer_s1.opt; user-facing graph I/O also match. Steady-state perf 27 µs vs native 18 µs = 1.50×. Documents what IS aligned, what IS NOT, and three concrete fix paths with expected savings.
type: project
---

# TL;DR for next session

**Where we are at 256³, iter 3 steady-state**:
- V8C8 BbbKMajor: **27 µs / 30,524 cyc** (bit-exact 65536/65536 vs ref)
- Native QNN MatMul: **18 µs / 17,816 cyc**
- **Ratio: 1.50× wall, 1.71× cyc**

**Where we were misled earlier**: at one point we claimed "kernel cyc 1.05× aligned" comparing V8C8 bbb to **native MatMul_0 group cycles** (which include surrounding ops). That was wrong. The honest comparison is:
- V8C8 bbb (HMX matmul kernel alone): 12,852 cyc
- Native ConvLayer_s1.opt (HMX matmul kernel alone, from chrometrace iter 1): 5,407 cyc
- **Kernel-only ratio: 2.4×**

We HAVE aligned: lowered graph I/O shapes (after this session's fix), bit-exact correctness, end-to-end DMA pipeline structure.
We HAVE NOT aligned: HMX kernel internals (descriptor caching, wt layout direction, output VTCM block geometry).

# What IS aligned ✓

## 1. User-facing tensor shapes (commit 356f40f)
```
V8C8                              Native
─────                             ──────
[1, 1, 256, 256] u8 input         [1, 1, 256, 256] u8 input
→ q::Reshape (null_exec, 0 cyc)   → q::Reshape (null_exec, 0 cyc)
→ ForceFormat_Crouton (HVX)       → ForceFormat_Crouton (HVX)
→ HMX matmul                      → HMX matmul
→ q::Reshape (null_exec, 0 cyc)   → q::Reshape (null_exec, 0 cyc)
[1, 1, 256, 256] u8 output        [1, 1, 256, 256] u8 output
```

Both Reshape ops are null_exec on both sides.

## 2. Kernel I/O dim factorization (commit cef7e01)
| pos      | V8C8 BbbKMajor             | Native ConvLayer_s1.opt    | match |
|----------|----------------------------|----------------------------|-------|
| in[0] act| `[1, 8, 32, 256]` u8q      | `[1, 8, 32, 256]` u8q      | ✓     |
| in[1] wt | `[1, 1, 256, 256]` u8q     | `[1, 1, 256, 256]` i8q     | ✓ shape (dt: u8 vs i8) |
| in[2] bias|`[1, 8, 1, 64]` i32        | `[1, 8, 1, 64]` i32        | ✓     |
| in[3]    | -                          | `[1]` i32 (descriptor meta)| (native QNN-internal) |
| in[4]    | -                          | `[1, 1, 1, 2]` i32 (meta)  | (native QNN-internal) |
| out[0]   | `[1, 8, 32, 256]` u8q      | `[1, 8, 32, 256]` u8q      | ✓     |

## 3. ctxgen schematic flow (commit 2928716)
`run_v8c8_phase2.sh` now passes `--profiling_level detailed --profiling_option optrace` to ctxgen, which emits `v8c8_schematic.bin`. End-to-end optrace decode (chrometrace JSON, QHAS HTML, runtrace) now works.

## 4. End-to-end correctness
- All shapes 32³–1024³ bit-exact at HMX path (S<128 uses scalar fallback, also bit-exact).
- `bbb_M0_N0` lowered into 5-node graph (vs native's 8). Graph compactness OK.
- Cold-start (iter 1): closes by iter 3.

# What is NOT aligned ✗ (the 1.50× gap)

## A. HMX descriptor: native baked at prepare-time, we rebuild every call

**Native** calls `libQnnHtpV75Skel.so::set_hmx_params_conv1x1(...)` during
`graph_finalize` (host side, prepare-time). The 0x40-byte descriptor
(act/wt/bias VTCM addresses, strides, Rt masks, output start) gets
embedded in the ctx-binary. At runtime ConvLayer_s1.opt fires HMX
directly with the cached descriptor.

**Us** in `HmxMatMulV9SkelOp.cpp::hmx_matmul_v9_kernel` (V9_KERNEL_HMX
branch), every call we:
```c
// Pre-bake act_ptrs[M_t × K_t] = 64 entries × 4 B    (~200 cyc)
int32_t act_ptrs_all[32 * 32];
for (mt) for (kt) act_ptrs_all[mt*K_t + kt] = ...;

// Pre-bake wt_ptrs[N_t × K_t]                        (~200 cyc)
int32_t wt_ptrs_all[32 * 32];
for (nt) for (kt) wt_ptrs_all[nt*K_t + kt] = ...;

// HMX state setup
asm("mxclracc");
asm("bias = mxmem2(...)");                            (~5 cyc × N_t)

// Inner loop register load + loop0 setup            (~5 cyc × N_t × M_t)
```

**Estimated cost**: ~1.5–2K cyc per call that native pays 0.

**Why we can't fix easily**: QHPI v1 doesn't expose
`set_hmx_params_conv1x1` or any equivalent prepare-time descriptor
hook. We could implement `do_precomputation_function` (QHPI provides
this) to bake act_ptrs/wt_ptrs into `precomputed_data` at graph load,
runtime kernel reads them — but this only works if the data is shape-
dependent only (it is, for our static-shape benchmark). **Try this in
next session.**

## B. wt layout direction: native N-outer (inline post-inc), we K-outer (need pre-baked array)

**Native** wt VTCM layout is `[1, N_tiles, K_tiles, 1024]` N-tile-outer.
For fixed nt, adjacent kt tiles are 1024 bytes apart → fits Hexagon
immediate post-inc:
```asm
{ r8 = add(r8, #0x400);              ; +1024 inline, fits in same packet as MAC
  activation.ub = mxmem(r6, r24):cm;
  weight.b      = mxmem(r8, r25) }   ; MAC #1
```
**3-packet body / 2 MAC = 1.5 cyc/MAC body.**

**Us** wt is `[1, K_tiles, N_tiles, 1024]` K-tile-outer (this layout was
chosen because it's byte-1:1 with native's `q::ConvLayer.opt.weights_to_vtcm@FB.fB.`
verbatim DMA output, validated in `Agent/qnn_re/weights_to_vtcm_RE_2026-04-27.md`).
For fixed nt, adjacent kt stride = `N_t * 1024` = 8 KB at 256³, too
large for immediate post-inc. We pre-bake wt_ptrs[K_t] into stack and
use memw post-inc:
```asm
{ r6  = memw(r1++#8); r8  = memw(r3++#8) }    ; load act ptr + wt ptr
{ r23 = memw(r1+#-4); r9  = memw(r3+#-4) }    ; second pair
{ activation.ub = mxmem(r6, r24):cm; weight.b = mxmem(r8, r25) }
{ activation.ub = mxmem(r23, r24):cm; weight.b = mxmem(r9, r25) }:endloop0
```
**4-packet body / 2 MAC = 2 cyc/MAC body. Extra 1 packet per K-pair.**

At 256³: 256 K-pairs total → **+256 cyc**. Smaller than I initially estimated;
not the dominant cost.

**How to fix**:
- **Option B1**: Change `gen_v8c8_test.py` wt to N-outer layout
  `wt_packed[N_t, K_t, 1024]` (transpose first two dims), update kernel
  to use inline `r8 += 0x400` post-inc. Cost: lose byte-equivalence
  with native `weights_to_vtcm@FB.fB.` (we'd need to handle our own
  pre-pack since QNN's verbatim DMA copies our N-outer bytes verbatim
  too — should still work). Saves ~256 cyc.
- **Option B2**: Keep K-outer, accept 2 cyc/MAC body. ~256 cyc cost.

## C. Output VTCM block geometry — Crouton_8 logical [1, 8, 32, 256] vs tile-array [1, 8, 8, 1024]

**This regressed the kernel by ~3K cyc** when we did shape alignment in
this session.

Before alignment (output sig declared as `[1, M/32, 32, N]` with the
Crouton_8 layout enum but no value_info): QNN lowered output to
`[1, 8, 8, 1024]` (tile-array) at 256³. Each "block" in QNN's
allocation = 1024 contiguous bytes = exactly 1 HMX `:after:cm:sat.ub`
write. Perfect fit. **bbb cyc = 9,777**.

After alignment (added value_info `[1, M/32, 32, N]` to ONNX node so
it stays in Crouton_8 logical form): QNN lowered output to
`[1, 8, 32, 256]` (matches native). Allocation became 32 blocks ×
2048 B (block_rows=64 from probe). Our 1024-byte HMX writes fit as
half-block, with adjacent (mt%2=0, mt%2=1) writes hitting the same
block at different 1KB offsets. **bbb cyc = 12,852** (+3K).

The +3K is likely VTCM bank conflict or cache eviction pattern — when
HMX writes the second half of a block ~30 cyc after the first half,
the cache line is in flight.

**How to fix**:
- **Option C1**: Roll back the value_info change → output factorizes as
  tile-array → bbb back to ~9.8K. Lose visual shape alignment in
  trace (but bytes are still 1:1 with same total = 65 KB).
- **Option C2**: Keep Crouton_8 logical alignment, investigate VTCM
  bank conflict via probe (Hexagon perf counters?). Possibly tune the
  HMX write Rt mask or interleaving. Risky / unknown payoff.

## D. ForceFormat_Crouton + DMAs not overlapping with HMX

**Native** groups bias_to_vtcm + ForceFormat_Crouton + weights_to_vtcm +
ConvLayer_s1.opt under a single `MatMul_0` grouping. QNN's compiler
schedules HVX (ForceFormat) concurrently with HMX (matmul) — they're
different physical units. Result: MatMul_0 wall ≈ max(HVX time, HMX
time) ≈ 9,531 cyc at iter 3.

**Us** BbbKMajor is opaque to QNN. ForceFormat_Crouton runs serially
under `Input` grouping; bias/wt DMAs run as separate nodes. No
overlap.

| | V8C8 cyc | Native cyc | gap |
|--|---------|-----------|-----|
| Input + ForceFormat_Crouton | 8,935 | 3,999 (just InputSlice) | +4.9K |
| wt + bias DMAs | 3,411 | folded | +3.4K |
| Total non-matmul work overhead | ~12K cyc ungrouped | ~4K cyc grouped + concurrent | **+8K** |

**How to fix**:
- **Option D1**: Implement `QHPI_BuildTileOfOp` (`build_tile`) callback.
  Split BbbKMajor by N axis into N_t sub-ops. Each sub-op + its
  surrounding DMA/format can be pipeline-able by QNN. Complex to
  implement correctly (need `qhpi_op_slice` + `qhpi_op_create`). Expected
  savings: ~3-5K cyc (uncertain).
- **Option D2**: Implement `early_rewrite` to inline custom-op dependencies.
  Probably can't replicate native's deep fusion; QNN compiler reserves
  that for ConvLayer-class ops.

# Cycle-count summary (256³ iter 3)

```
                       V8C8       Native      gap     fix
─────────────────────────────────────────────────────────────────
Input + Format_Crouton   8,935      3,999    +4.9K    D (build_tile)
wt + bias DMAs           3,411          0    +3.4K    D (build_tile)
HMX matmul kernel       12,852      ~5,000   +7.8K    A+B+C
  - Output VTCM bank                          +3.0K     C (rollback layout)
  - K-outer wt extra packet                   +0.2K     B (N-outer wt)
  - Per-call descriptor rebuild               +1.5K     A (precompute hook)
  - misc (bookkeeping, ALU)                   +3.1K     ?
Output Reshape + DDR     5,326      5,168    +0.2K    ✓ aligned
Reshape (null_exec)          0          0     0       ✓ aligned
─────────────────────────────────────────────────────────────────
Total                   30,524     17,816   +12.7K
                                            (1.71× cyc)
                                            (1.50× wall)
```

# Concrete next-session priorities

## Priority 1: Implement `do_precomputation_function` (Option A)

QHPI provides `QHPI_Precompute_Function` for shape-dependent setup
that doesn't change between inferences. Move act_ptrs/wt_ptrs prebake
into precompute, runtime kernel just reads `precomputed_data`.

**Code touch points**:
- `HmxMatMulV9SkelOp.cpp`:
  - Add `precompute_data_size = 2 × 32 × 32 × sizeof(int32_t)` (= 8 KB)
  - Add `do_precomputation_function = bbb_precompute` that fills act_ptrs/wt_ptrs
  - Add `function_with_precomputed_data = bbb_kernel_precomputed`
  - Set `function = NULL` (mutually exclusive)
- New `bbb_kernel_precomputed` reads pre-built ptr tables from
  precomputed data, skips runtime prebake.

**Expected savings**: ~1-2K cyc/call.

**Risk**: QHPI precompute may not have access to runtime VTCM addresses
(act blocks come from block_table at runtime, can't pre-bake those).
Need to verify what's available in precompute context. If only wt
ptrs can be precomputed (they're STATIC), savings cut in half (~500 cyc).

## Priority 2: Roll back output factorization to tile-array (Option C1)

Most impactful single change: removes the +3K bank-conflict cost.

**Code touch points**:
- `gen_v8c8_test.py`: remove `value_info=[mm_c8_info]` from
  `make_graph()` call so QNN auto-factorizes output as tile-array.
- Update `Agent/v8c8_step5_*.md` doc explaining the trade-off.

**Cost**: trace's output dim shows `[1, 8, 8, 1024]` instead of native's
`[1, 8, 32, 256]`. Bytes still byte-1:1 (65 KB). Visual alignment
imperfect but performance ~3K cyc better.

**Decision needed from user**: visual alignment or 3K cyc? Recommend
asking before implementing.

## Priority 3: Try N-outer wt layout (Option B1)

Lower priority since saves only ~256 cyc.

**Code touch points**:
- `gen_v8c8_test.py`:
  - Change `wt_packed` shape from `[1, K_t, N_t, 1024]` to
    `[1, N_t, K_t, 1024]` (just transpose first two dims).
- `HmxMatMulV9SkelOp.cpp`:
  - Update wt_ptrs prebake: `wt_ptrs_all[nt*K_t + kt] = wt_pack + (nt*K_t + kt)*1024;`
  - Inner loop: replace memw post-inc with `r8 += 0x400` inline if
    possible. Match V8 prod's 3-packet body.

**Expected savings**: ~256 cyc.

## Priority 4 (if A+B+C done): build_tile callback (Option D1)

Expensive in development time. Only attempt if A+B+C don't close enough.

# Open questions for next session

1. Does QHPI's `do_precomputation_function` get access to STATIC tensor
   raw_data? (Needed to pre-bake wt_ptrs at prepare time.) Read
   `qhpi.h:639-658` for spec; if unclear, write a probe.

2. The "+3.1K misc" in HMX kernel is unaccounted. Possible sources:
   - HMX bias state warmup latency
   - sat.ub drain pipeline depth (~10-20 cyc each, × 64 tiles = 640-1280 cyc)
   - VTCM bank conflicts between multiple (mt, nt) tiles writing same row
   Need a per-tile cyc breakdown probe (insert MIPS counter reads).

3. Is the dtype mismatch (in[1] wt: u8q ours vs i8q native) costly? If
   we declared as i8q, QNN might insert auto +128 cast. Worth probing.

# Reference: All commits in this work stream

```
cef7e01 gen_v8c8_test.py: align BbbKMajor I/O shapes with ConvLayer_s1.opt   ← shape align (+3K cyc)
356f40f gen_v8c8_test.py: align user-facing input shape with native [1,1,M,K]
2928716 run_v8c8_phase2.sh: enable optrace in ctxgen → schematic.bin emitted
eb6fd20 V8C8 BbbKMajor: 256³ matmul aligned to 1.33× of native (Step 1-4)
c5d29fc V9 sweep: characterize 512³→4096³, VTCM overflow threshold = 2048³
c40954e V9 adaptive ONNX gen: V8 matmul shape-scales to 4096³
7ae8668 V8-only cleanup + standard QNN custom-op flow (ONNX→DLC→ctx-binary→optrace)
```

# Repro

```sh
cd /home/zzh/work/qcom_htp/example/hmx_matmul_phase3

# Build
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh

# Run + decode optrace
cd standard_flow/phaseB_v8
M=256 K=256 N=256 OUT_DIR=phase1_validation/v8c8_apples2_256 \
    bash run_v8c8_phase2.sh

# Verify shapes
python3 -c "
import json
d = json.load(open('phase1_validation/v8c8_apples2_256/ctx/v8c8_bottom_mapping.json'))
nodes = d['graph']['nodes']
tensors = d['graph']['tensors']
for nid, n in nodes.items():
    if 'BbbKMajor' in n.get('type', ''):
        for i, inp in enumerate(n.get('input_names', [])):
            t = tensors[inp]
            print(f'in[{i}]: {t[\"dims\"]} dt={hex(t[\"data_type\"])}')
        for out in n.get('output_names', []):
            t = tensors[out]
            print(f'out[0]: {t[\"dims\"]} dt={hex(t[\"data_type\"])}')
"

# Bit-exact check
python3 -c "
import numpy as np
b = np.fromfile('phase1_validation/v8c8_apples2_256/device_out/out.raw', dtype=np.float32)
out = np.round(b).astype(int).clip(0,255).astype(np.uint8).reshape(256,256)
ref = np.load('phase1_validation/v8c8_apples2_256/v8c8.onnx.out_ref_u8.npy')
print(f'match: {(out==ref).sum()}/65536')
"
```

# Optrace bundle

Last full bundle: `/tmp/optrace_256_bundle/` and
`/tmp/optrace_256_bundle.tar.gz` (also synced to user's GDrive at
`G:\我的云端硬盘\optrace_256_bundle\` via DriveFS).

Contains apples-aligned V8C8 + native side-by-side: ONNX, DLC, ctx,
schematic, qnn-profiling-data_0.log, optrace.txt + chrometrace JSONs +
QHAS HTML.
