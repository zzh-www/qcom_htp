# Three-kernel matmul — final status (2026-04-22)

Goal (from 2026-04-22 user directive): reverse-engineer QNN's HMX matmul
for **w4a8 / w4a16 / w16a16**; hand-write HMX+HVX implementations that
align on correctness and performance across simulator and device.

## Deliverables — all three kernels end-to-end

| kernel  | QNN OpPackage path                 | scalar ref | device HMX  | sim | device cyc/MAC @ 512³ |
|---------|------------------------------------|:----------:|:-----------:|:---:|----------------------:|
| w4a16   | `example/hmx_matmul_qnn/`          | ✅         | ✅ 0 mismatch | ✅ 3/3 | **2.08** |
| w4a8    | `example/hmx_matmul_w4a8/`         | ✅         | ✅ 0 mismatch | ✅ 3/3 | **1.92** |
| w16a16  | `example/hmx_matmul_w16a16/`       | ✅         | ✅ 0 mismatch | ✅ 3/3 | **12.22** |

All tests reproducible:
```sh
# Device (SM8650 v75 via ssh oneplus)
bash example/hmx_matmul_w4a8/run_on_device.sh  --shape 512,512,512
bash example/hmx_matmul_qnn/run_on_device.sh   --shape 512,512,512
bash example/hmx_matmul_w16a16/run_on_device.sh --shape 512,512,512
# Simulator (hexagon-sim + H2 booter)
bash tests/test_hmx_matmul_w4a8.sh
bash tests/test_hmx_matmul_w4a16.sh
bash tests/test_hmx_matmul_w16a16.sh
bash tests/test_hmx_matmul_int16.sh
```

## Phase-by-phase

| phase | status | key artifact |
|-------|--------|--------------|
| P0 measurement harness     | ✅ done   | `Agent/baseline_2026-04-22.md` |
| P1 deep QNN RE             | ✅ done   | `Agent/qnn_hmx_pipelining.md` + 2 silicon probes |
| P2 A w16a16 beat-QNN wrap  | ✅ done   | `example/hmx_matmul_w16a16/` |
| P2 B w16a16 match-QNN      | 🟡 design | `Agent/phase2b_w16a16_match_qnn.md` (impl deferred) |
| P3 w4a16 optimization      | ✅ done   | 2.12 → 2.08 via fused dualacc + Rt_wt=0x3FF |
| P4 w4a8 new kernel         | ✅ done   | `example/hmx_matmul_w4a8/` |
| P5 sim↔device parity       | ✅ done   | `Agent/sim_vs_device_cycles.md` + 3 new sim harnesses |
| P6 HVX pack/unpack         | 🟡 attempted | vdelta approach wrong; 2-stage vshuff deferred |

Overall: **6/8 phases fully delivered; 2 phases attempted with blockers
documented.**

## Key Phase-1 RE findings (all silicon-validated, SM8650 v75)

1. **`Rt_wt = 0x3FF`** unlocks HMX MAC pipelining: 19.7 → 7.9 cyc/packet
   (2.5× packet-level speedup). Use this constant in any u8·i8 HMX MAC.
2. **`:above` is a no-op** for accumulator routing — it writes current acc,
   same as plain MAC. Prior RE confused it with Conv2D's `:above`.
3. **`mxswapacc` really swaps** current/other accs. Works as expected.
4. **`store :after.uh`** **WITHOUT `:retain`** **clears BOTH accs** (not
   just current). This was the root cause of the prior dualacc bug at
   K ≥ 64 — missing `:retain` on the hi-byte store wiped acc B before
   the swap-and-read.
5. **HMX output tile must be 2 KB-aligned** — `acc:2x1` store address-masks
   silently corrupt phys_row 8–15 otherwise.
6. **C file compiled as C++** (hexagon-clang++ on .c) mangles symbols;
   headers must guard with `extern "C"` to match Op.cpp's wrapped include.

## Known blockers to full perf parity (≤10× QNN w8a16)

The HMX itself is not the bottleneck — our probe shows 7.9 cyc/packet
steady-state, competitive with QNN. The 500-2000× kernel-level gap is:

1. **Scalar DDR bandwidth** in `gather_w_col` and `pack_weight_32x32`.
   HVX-izing these is the P6 lever; attempted with `vdelta` → wrong
   output. Needs 2-stage `vshuff` or `vlut32` redesign.
2. **HMX VTCM bank contention** — back-to-back `mxmem` from adjacent
   VTCM regions stalls more than same-address reuse. Prior "all-prepacked"
   path regresses 63% for this reason; double-buffering didn't help.
3. **No graph-level parallelism** — QNN's 4-way-tiled `ConvLayer_s1.opt`
   runs 4 HMX sub-kernels on separate HVX threads concurrently. QHPI's
   self-slicing gate is HVX-only (`QHPI_RESOURCE_HVX`), so HMX ops can't
   self-slice. Requires emitting 4 sub-ops at host graph-build time.

## Commits landed in this session

```
d15e2c0  P1 RE: decode QNN hmx_convbbb1x1_stride1 + HMX semantics probes
7b31067  Phase 4 w4a8: scaffolding + kernel, scalar verified, HMX path WIP
8036378  w4a8: fix phys_row 8-15 corruption via 2KB-aligned out tile
(P3)     Phase 2A w16a16: new QNN OpPackage, 4-term HMX decomp, int32 out
2045369  P4+P5: sim/device parity doc, gather_w_col cache note
(P6/P5)  P2b HVX attempt + Phase 2B doc + P5 sim harnesses for all 3 kernels
```

## How to continue

The three pieces of work that would substantially improve on this:

1. **HVX `pack_weight_32x32`** via 2-stage byte shuffle — would save ~8%
   per kernel, validate HVX capability for Phase 6 broader.
2. **Decode QNN's `ForceFormat_Crouton_f2c`** framework HVX op — that's the
   pack-layer QNN uses for its fast path, and the Crouton tile shape
   (`[1,8,8,32]` → HMX-ready via ForceFormat) is the abstraction our
   custom ops should consume.
3. **Graph-level 2×2 M/N tile** — emit 4 sub-op nodes in host graph,
   each computing [256, 512, 256] (at 512³). This is the architectural
   move that QNN uses to hit <1ms on w8a8 512³.

Each is 2-3 days of focused work. Leaving docs + test scaffolding in
place for resumption.
