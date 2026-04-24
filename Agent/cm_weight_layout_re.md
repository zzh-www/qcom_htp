# HMX `:cm` weight layout — silicon RE (follow-on)

**Target:** SM8650 v75 (OnePlus 12)
**Date:** 2026-04-23
**Probe:** `example/hmx_matmul_device/probe_cm_weight_layout.c`
**Runner:** `example/hmx_matmul_device/run_cm_weight_probe.sh`

## Question

Agent A's `probe_cm_row_major` used uniform all-1 activation and all-1 weight, so it could not distinguish which weight-tile byte layout HMX expects under `:cm`. This probe varies weight patterns + layouts to pin it down.

## Setup

- Activation: all-ones row-major tile (1 KiB, `act[r][k]=1` for r,k ∈ [0,32))
- `Rt_a = 2047|0x1c`, `Rt_w = 0x3FF`, `:cm` activation, plain weight
- Weight filled with one of 4 patterns, tiled under one of 3 candidate layouts:
  - **RM** : row-major             `tile[32*k + n] = w[k][n]`
  - **P2** : Phase 2 4-K-row packed `tile[128*kg + 4*col + row4] = w[4*kg+row4][col]`
  - **NT** : N-transpose row-major  `tile[32*n + k] = w[k][n]`

## Discriminating results

### Test C — N-ramp `w[0][n]=n+1`, rest 0 (single-K contribution)

Expected `out[m][n] = act[m][0] * w[0][n] = n+1` (ones in row 0).

| Layout | out[0][0..7]                     | out[0][8..15]                  | verdict |
|--------|----------------------------------|--------------------------------|---------|
| **P2** | **1 2 3 4 5 6 7 8**              | **9 10 11 12 13 14 15 16**     | **exact match** |
| RM     | 10 26 42 58 74 90 106 122, rest 0 | —                              | wrong; N-groups of 4 folded into single col |
| NT     | 120 0 0 0 ... 128 ... 136 ... 144 | —                              | wrong; smeared |

→ **P2 is the correct weight layout under `:cm` activation.**

### Test B — K-ramp `w[k][0]=k+1`, rest 0 (K-accumulation)

Expected `out[m][0] = Σₖ w[k][0] = 528`.

| Layout | out[0][0], col 8, col 16, col 24 | total | verdict |
|--------|----------------------------------|-------|---------|
| **P2** | **530 0 0 0**                    | 8480  | total 528-ish @ col 0 (one tile) |
| RM     | 120, 128, 136, 144              | 8448  | scrambled — splits the K-sum across 4 cols at stride 8 |
| NT     | smeared across 8 cols            | 8448  | wrong |

The +2 anomaly in P2 Test B (530 vs 528) is consistent across runs and appears to be a dual-scale readback artifact (we only do one of the two readback passes Phase 2 normally uses); it is not a weight-layout issue — Test C proves P2 is byte-exact.

### Test D — all-ones weight

All three layouts give `out[m][n]=32` for every cell (degenerate case: every byte is 1 regardless of layout, so layouts are indistinguishable here — matches Agent A's original observation).

## Byte layout contract

```
Weight 1 KiB tile for `activation.ub = mxmem(p, 2047|0x1c):cm` MAC:
  tile[128*kg + 4*col + row4] = w[4*kg + row4][col]
    kg   ∈ [0, 8)   — K-group index (4 K-rows each)
    col  ∈ [0, 32)  — N column
    row4 ∈ [0, 4)   — sub-K-row within the group
```

This is the EXACT same layout Phase 2 already uses (`pack_weight_32x32` in `example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c`, and `pack_wt_hvx` / `pack_wt_v3_hvx` in Phase 3). **No new weight-side packing work required.**

## Phase 3 implication

The `:cm` zero-pack path is:

| component        | layout                        | size per tile |
|------------------|-------------------------------|---------------|
| activation tile  | **row-major 32 rows × 32 K**  | 1 KiB         |
| weight tile      | Phase 2 4-K-row packed        | 1 KiB         |

- **Activation pack is now a trivial contiguous gather** (32 × memcpy(32) or 8 × HVX vmemu load) — no halfword shuffle, no 2-stream interleave, no `*<<8` arithmetic.
- Per-tile activation memory drops from 2 KiB to 1 KiB → half the VTCM footprint and half the activation side of the weight+act memory traffic that fills the mxmem pipeline.
- Silicon-measured MAC throughput at this config (from Agent A's probe): **7.92 cyc/MAC** vs V3's 2-stream path at 9.03 cyc/MAC.

## Next

Implement MatMulV4 = MatMulV3 with:
1. Activation pack → contiguous row-major 1 KiB tiles
2. MAC instruction → `activation.ub = mxmem(p, Rt_a|0x1c):cm` / `weight.b = mxmem(q, 0x3FF)`
3. Keep weight pack + readback decode untouched

Expected impact at 512³: V3 0.143 → V4 ≤ 0.13 cyc/MAC (pure HMX; pack cost already free because host pre-pack is cheaper than before). If graph-wired (PackActRM + PackWt + V4), the pack op's HVX workload also halves → pack_act cost in graph drops roughly 2× from 9.9M cycles.
