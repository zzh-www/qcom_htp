---
name: P1.5 — Native act_tbl/out_tbl layout RE — 100% bit-exact at 1.77×
description: P1.4 v3 patch dumped native's act_tbl_all + out_tbl_all entries. Both are 32 entries × 0x800 (2KB) stride — M-pair packed (2 M-tiles per ptr). New V73D_NATIVE_LAYOUT knob + native descs gives 611 pkts at 100% bit-exact (vs 747 baseline, vs native 346). Gap reduced from 2.16× → 1.77× while keeping correctness.
type: project
---

# P1.5 — Native table layout RE (2026-04-29)

## Method

Extended dump_stub_v3 to capture act_tbl_all + out_tbl_all entries within the
readable tile region (rows 0..7 = 256 bytes). All markers verified correctly.

## Native's TRUE table layout

Both tables use **32 entries × 0x800 (2KB) stride** packing.

```
out_tbl_all (od[0] = 0x02098d58):
  [ 0..15] = 0xfc010000 + i * 0x800   # VTCM, 2KB per entry
  (entries 16..31 likely follow same pattern, not dumped)

act_tbl_all (ad[0] = 0x02098c10):
  [ 0..31] = 0xfc030800 + i * 0x800   # VTCM, 2KB per entry
```

**Each 2KB entry = M-pair packed** = 2 consecutive M-tiles (each 1KB) concatenated.
This makes `:deep:cm` activation-MAC fetch 2 M-tiles from a SINGLE 2KB pointer
(per native's `mask[+0x0c] = 0x71f` setting, vs our prior `0x77c` which fetches
2 separate ptrs for each M-pair).

## Implementation

Added `V73D_NATIVE_LAYOUT` knob to HmxMatMulV9SkelOp.cpp. When defined, replaces
the 64-entry table build (which writes M0/M1 ptr pairs at offsets 0/1024) with
a 32-entry direct copy of `act_blocks`/`out_blocks`. Each entry points to first
M-tile of an M-pair; the 2nd M-tile is implicitly at +1024 bytes (kernel reads
2KB span via `:deep:cm`).

Build:
```bash
DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST"
DEFS="$DEFS -DV73D_N_TILES_POW2=32 -DV73D_OUT_Y_STRIDE=32 -DV73D_AD_ACT_Y_STRIDE=32"
DEFS="$DEFS -DV73DEEP_ARG1=0x700 -DV73D_MASK_38_EXTRA_PTR=1 -DV73D_NATIVE_LAYOUT=1"
EXTRA_DEFS="$DEFS" bash build.sh && bash build_x86.sh
```

## Results — V73DEEP gap closure progression

| Config | hot pkts | hot dur | cpp | bit-exact | vs native |
|---|---:|---:|---:|---:|---:|
| V73DEEP main path (旧 baseline) | 747 | 3042 | 4.07 | 100% | 2.16× |
| + native descs (n_tiles_pow2=32, strides) | 475 | 1730 | 3.64 | 50% | 1.37× |
| + native LAYOUT (scalar memcpy) | 611 | 2027 | 3.32 | 100% | 1.77× |
| **+ HVX batched table copy** | **471** | **1799** | **3.82** | **100%** | **1.36×** |
| Native ConvLayer_s1.opt | 346 | 1120 | 3.24 | ref | 1.0× |

**100% bit-exact + 1.36× gap (vs 2.16× baseline)** = 37% packet reduction
with correctness preserved.

HVX optimization: 32 ptrs * 4 bytes = 128 bytes per table = exactly 1 HVX
vector. Two `memcpy(.., HVX_Vector)` for act + out tables replace scalar
loops. Saves ~140 packets vs scalar memcpy.

## Remaining 1.77× gap analysis

cpp ratio 1.03× → kernel-internal efficiency matches native within 3%.
Packet count gap 1.77× → our op execute() does more packets than native.

Likely sources of ~265-packet remaining gap:
1. **VTCM placement** (~80 packets): native has wt + bias in VTCM (0xfc02_xxxx),
   we use DDR via `qhpi_tensor_raw_data`. Need TCM_Only signature in op-pkg
   to make QNN insert weights_to_vtcm + bias_to_vtcm pre-ops.
2. **Op overhead** (~100 packets): set_hmx_params_conv1x1 cached call + our
   table-build memcpy (32 ptr × 2 tables = 256 bytes via scalar memcpy) +
   descriptor struct setup. Native's ConvLayer_s1.opt may have HVX-accelerated
   table building.
3. **Per-call kernel internals** (~85 packets): native may pre-config HMX
   state once per session and skip prologue per call.

Items (1) requires op-pkg signature rewrite (significant work).
Item (2) — try to eliminate or HVX-batch the memcpy. Already use HVX for
mt_per_block=2 K_t=8 N_t=8 path; native layout uses scalar memcpy now.
Item (3) requires kernel patching beyond P1.4/P1.5 scope.

## P1.5 status: COMPLETE — 100% bit-exact + 18% packet reduction

artefact: `phase1_validation/v73deep_native_layout/` chain8 256³

## Reproduce

Same build flags as above. Run `run_v8c8_chain.sh` with
`OUT_DIR=...native_layout`.

## Next 选择

- **P1.6 stop**: Accept 1.77× gap as acceptable improvement; document V73DEEP
  current best is 611 pkts / 100% bit-exact with native layout.
- **P1.6 VTCM**: implement TCM_Only signature for wt+bias preload — close
  ~80 packets of remaining gap.
- **P1.6 HVX layout**: HVX-accelerate the 32-ptr memcpy of table building —
  maybe save ~50-100 op overhead packets.

(P5 / P6 from NEXT_STEPS_v73deep_gap.md — probably not worth multi-day pursuit
given diminishing returns; P1.5 already gets to 1.77×.)
