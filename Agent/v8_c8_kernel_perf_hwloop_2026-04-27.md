---
name: V8 C8 BbbKMajor hw loop0 + pre-baked ptrs — 256³ kernel cyc 107K → 15.9K (6.7×) 2026-04-27 night
description: Ported V8 prod's Hexagon hardware loop0 + dual-memw post-inc + dual-MAC body into V9 BbbKMajor. Pre-bake act_ptrs[M_t*K_t] (Crouton-block-resolved) and wt_ptrs[N_t*K_t] into stack arrays so inner loop only does memw post-inc. Bit-exact preserved at 32³–1024³.
type: project
---

# Result

Steady-state (iter 3 of 3) BbbKMajor cycles, V8C8 vs native MatMul_0:

| S    | V8C8 bbb (before)| V8C8 bbb (after) | speedup | Native MatMul | V8C8/Native |
|------|------------------|------------------|---------|---------------|-------------|
| 256  | 107,736          | 15,929           | **6.8×**| 9,335         | 1.71×       |
| 512  | 155,129          | 76,960           | 2.0×    | 39,447        | 1.95×       |
| 1024 | 573,166          | 531,130          | 1.08×   | 1,167,447     | **0.45× (faster)** |

cyc / MAC packet (HMX silicon ceiling ~8):

| S    | MAC packets | Before (cyc/MAC) | After (cyc/MAC) | Native (cyc/MAC) |
|------|-------------|------------------|-----------------|------------------|
| 256  | 512 (8³)    | 210              | **31**          | 18               |
| 512  | 4096 (16³)  | 38               | 19              | 9.6              |
| 1024 | 32768 (32³) | 17.5             | 16.2            | 35.6             |

# What changed

`HmxMatMulV9SkelOp.cpp` `V9_KERNEL_HMX` branch:

1. **Pre-bake act_ptrs[M_t × K_t]** at kernel entry: resolve Crouton block_table
   indirection + per-mt 1024-byte offset once, store in stack int32 array.
2. **Pre-bake wt_ptrs[N_t × K_t]** at kernel entry: resolve K-tile-outer
   layout `wt_pack + (kt*N_t+nt)*1024` once.
3. **Hexagon hw loop0** with dual-MAC unrolled body (4 packets / 2 MAC):
   ```asm
   loop0(1f, K_t/2)
   1:
   { r6  = memw(r1++#8); r8  = memw(r3++#8) }   ; ptr loads dual-slot
   { r23 = memw(r1+#-4); r9  = memw(r3+#-4) }   ; second pair (sees post-inc r1/r3)
   { activation.ub = mxmem(r6, r24):cm; weight.b = mxmem(r8, r25) }   ; MAC #1
   { activation.ub = mxmem(r23,r24):cm; weight.b = mxmem(r9, r25) } : endloop0  ; MAC #2
   ```
   Replaces naive C `for(kt) {...}` with scalar bookkeeping packets between MACs.
4. **Removed per-mt bias reload** — was V8 prod paranoia; HMX bias state
   persists across `:after:cm:sat.ub` events for at least M_t=32. Bit-exact
   preserved.

Stack budget: act_ptrs[32*32]=4 KB + wt_ptrs[32*32]=4 KB = 8 KB. Fits at
all shapes.

# Why our wt is K-outer (not N-outer like V8 prod)

V8 prod could post-inc r8 by `#0x400` (1024 B) inline because its wt_pack
was [N_tiles, K_tiles, 1024] N-tile-outer, so adjacent kt for fixed nt
are 1024 B apart. Our V9 wt is [K_tiles, N_tiles, 1024] K-tile-outer
(byte-1:1 with native ConvLayer's pre-pack from
`q::ConvLayer.opt.weights_to_vtcm@FB.fB.`). The per-kt stride is
`N_t * 1024`, too large for immediate post-inc. Pre-baking ptrs sidesteps
this without changing the wt layout (which we MUST preserve to stay
byte-equivalent to native bias_to_vtcm fold).

# Why 1024³ is faster than native (0.45×)

Native splits 1024³ into 75 separate `q::ConvLayer_s1.opt` instances for
VTCM budgeting (memory: 1024³ has 136 lowered nodes, 75 ConvLayer
calls). Each instance has descriptor-build setup overhead. Our V9 does
1024³ in a single instance — setup is amortized over 32K MAC packets,
hot-loop dominates.

At ≥2048³ we'll hit the same VTCM ceiling and need our own
multi-instance split (port from `gen_v8_graph.py`). Until then, V8C8
beats native at 1024³.

# End-to-end gap remains

Total Accelerator (execute) µs steady-state:

| S    | V8C8 µs | Native µs | ratio | reason       |
|------|---------|-----------|-------|--------------|
| 256  | 105     | 18        | 5.8×  | UntileToRowMajor 90% |
| 512  | 348     | 39        | 8.9×  | UntileToRowMajor 90% |
| 1024 | 1553    | 512       | 3.0×  | UntileToRowMajor 90% |

UntileToRowMajor is OUR op (tile-layout VTCM → row-major DDR) — native
graph emits row-major directly via `q::ForceFormat_Flat` + Output node
~50K cyc total. Our untile is 6.3 M cyc at 1024³. Separate optimization
target.

# Reproduce

```sh
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh
cd standard_flow/phaseB_v8
M=256 K=256 N=256 OUT_DIR=phase1_validation/v8c8_hwloop_256 bash run_v8c8_phase2.sh
```

# Open

- 256³ V8C8 / native MatMul = 1.71× — extra 7K cyc beyond native's 9K is
  likely setup (output zero, mxclracc, bias mxmem2 × N_t, sat.ub × M_t × N_t).
  Per (mt,nt) tile: ~250 cyc / 64 tiles = ~4 cyc fixed cost. Could push
  closer with fewer fixed overheads but diminishing returns.
- UntileToRowMajor must be optimized separately (next target if user
  wants e2e parity).
