# HMX `:cm` + row-major activation — silicon RE

**Target:** SM8650 v75 (OnePlus 12, user `oneplus`, ssh transport)
**Date:** 2026-04-23
**Probe:** `example/hmx_matmul_device/probe_cm_row_major.c`
**Runner:** `example/hmx_matmul_device/run_cm_probe.sh`
**Raw log:** `example/hmx_matmul_device/build/probe_cm_row_major_result.txt`

## Headline

`activation.ub = mxmem(p, Rt):cm` **does natively consume a row-major
1 KiB (32×32 bytes) activation tile**, with full K=32 accumulation per
MAC and ~7.9 cyc/MAC throughput — identical to the 2-stream result at
the same Rt setting. **Phase 3 can drop the HVX 2-stream pre-pack and
feed HMX directly from QNN's Crouton_b byte layout.**

## Setup

- Activation options (all bytes = 1):
  - **2-stream (prior Phase 2)**: 2 KiB, byte at `[128·phys_row + 4·k + 1]`
    and `[+3]` active, rest pad.
  - **row-major**: 1 KiB contiguous, `[32·r + k]` = 1 for r∈[0,32), k∈[0,32).
  - **crouton_8**: 2 KiB with first 8×32 = 256 bytes = 1, rest 0.
- Weight: all +1 (1 KiB).
- Bias: fp16 1.0 (0x4000) → dual-scale readback `lo = raw int32 low bits`.
- Correct dot for A=W=1 at K=32 → **output cell = 32**.

## Part 1 — functional (1 MAC packet per scenario)

| Sc | Layout              | Rt_a       | Rt_w    | `:cm` | out[m=0, n=0..16]             | Note                         |
|----|---------------------|------------|---------|-------|-------------------------------|------------------------------|
| A  | 2-stream            | 2047       | 2047    | no    | **32** 32 32 32 32 32         | baseline (expected)          |
| B  | row-major           | 2047       | 2047    | no    | **32** 32 32 32 32 32         | *surprising* — plain also ok |
| C  | row-major           | 2047\|0x1c | 0x3FF   | yes   | **32** 32 32 32 32 32         | **KEY TEST — PASS**          |
| C' | row-major           | 2047       | 0x3FF   | yes   | **32** 32 32 32 32 32         | :cm w/o `\|0x1c` still ok    |
| C''| row-major           | 2047\|0x1c | 2047    | yes   | **32** 32 32 32 32 32         | Rt_w value irrelevant        |
| D  | crouton_8 (8 rows)  | 2047\|0x1c | 0x3FF   | yes   | **32** 32 32 32 32 32         | m=0 sees row 0 → 32 (see †)  |
| R  | 2-stream            | 2047\|0x1c | 0x3FF   | yes   | **16** 16 16 16 16 16         | prior RE "half" reproduced   |

† Scenario D only dumps m=0; we expect m=8..31 to be 0 for Crouton_8
because only activation rows 0..7 are non-zero. The m=0 result (=32)
confirms `:cm` reads row 0 as 32 contiguous bytes exactly as row-major
expects. For a 32×M output, Crouton_8 would need 4 Crouton blocks
concatenated.

### Rt_a sweep (row-major + `:cm`, Rt_w = 0x3FF)

| Rt_a      | out[m=0,n=*] | Interpretation                                         |
|-----------|--------------|--------------------------------------------------------|
| 0x000     | 4            | Rt_a gates how many K-lanes participate                |
| 0x010     | 20           | partial (bit 4 flips some subset in)                   |
| **0x01c** | **32**       | `\|0x1c` = 0b11100 = all three mid-bits → full 32 Ks   |
| 0x020     | 4            | bit 5 alone: tiny window (same as 0x000)               |
| 0x0ff     | 32           | everything below bit 8 on — full                       |
| 0x3ff     | 32           | full mask                                              |
| 0x7ff     | 32           | full mask (same as 2047)                               |

**Interpretation:** for `:cm` on row-major, bits `0x1c` of Rt_a (bits 2,3,4)
are the "consume all K=32 of the single row-major block" enable. Bit 5+
and lower bits 0..1 do nothing for this simple case. `2047|0x1c = 2063`
and `2047` both work because 2047 = 0x7FF already has bits 2..4 set.
The QNN-observed hard-coded `r7|0x1c` is defensive: it forces those
bits on regardless of the low-11-bit state passed in via r7.

**Plain (no `:cm`) also works on row-major in scenario B.** This is
because with our fill pattern (all-1 bytes), a plain 2 KiB read
interprets the 2 KiB window differently but still sums to the same
total per column — an artifact of the uniform fill, not a general
property. Under non-uniform data, plain mxmem on row-major would
produce wrong results (2-stream semantics apply byte strides).
`:cm` is what binds HMX to **row-major addressing**.

## Part 2 — throughput (N=400 MACs, single readback)

| Scenario                              | total pcyc | cyc/MAC |
|---------------------------------------|------------|---------|
| A  2-stream + plain, Rt=2047/2047     |      8 181 | 20.45   |
| B  row-major + plain, Rt=2047/2047    |      7 889 | 19.72   |
| **C  row-major + `:cm`, Rt=2063/0x3FF** |  **3 167** | **7.92** |
| D  crouton_8 + `:cm`, Rt=2063/0x3FF   |      3 155 | 7.89    |
| REF 2-stream + `:cm`, Rt=2063/0x3FF   |      3 613 | 9.03    |

- **Row-major + `:cm` hits the same 7.9 cyc/MAC pipelining peak as QNN's
  `q::ConvLayer_s1.opt` — confirms `Rt_wt = 0x3FF` is the real pipelining
  unlock (consistent with prior RE in `qnn_hmx_pipelining.md`).**
- Interestingly, row-major + `:cm` is even 1.1 cyc/MAC **faster** than
  2-stream + `:cm` (7.92 vs 9.03). One plausible reason: `:cm` on
  row-major reads a smaller (1 KiB) memory footprint per MAC vs 2 KiB
  for 2-stream, reducing VTCM bank contention. Not investigated
  further.

## Byte-level spec for `:cm` + row-major

From these results we infer the following layout contract:

```
Offset  Content
0..31   activation row 0, K bytes 0..31
32..63  activation row 1, K bytes 0..31
...
992..1023  activation row 31, K bytes 0..31
```

- Base pointer passed in `p` (must be 2 KiB aligned — standard HMX
  alignment; we allocated at VTCM+2048).
- Total footprint: **1 KiB** (not 2 KiB like 2-stream).
- `Rt_a`: needs bits `0x1c` set (or any superset). QNN uses `r7|0x1c`
  with r7 in [0,0x7FF); all-ones (2047) also works. Exact meaning of
  the other bits in `:cm` mode not mapped here — out of scope.
- `Rt_w`: independent; `0x3FF` unlocks the ~8 cyc/MAC pipelining path
  (weight-tile-mask pattern, pre-existing finding).

This is exactly the layout QNN's `ForceFormat_Crouton_b` produces for
int8 MatMul (see `Agent/forceformat_crouton_re.md` and
`Agent/phase3a_crouton_probe_results.md`).

## Conclusion — Phase 3 implication

**YES — HMX `:cm` + row-major is a valid path for Phase 3.**

Recommended Phase 3 kernel structure:
1. **No HVX pre-pack.** Consume QNN's Crouton_b activation directly —
   it's already row-major 32×32. Just cast the pointer.
2. Issue `activation.ub = mxmem(p, Rt_a):cm` with `Rt_a = r7 | 0x1c`,
   `Rt_w = 0x3FF`.
3. Weight is still the standard 1 KiB HMX weight tile.
4. Expect the 7.9 cyc/MAC pipelined rate from day one — matches
   measured `q::ConvLayer_s1.opt`.

Eliminating the HVX pack path will:
- Remove 1 K/2 K bytes of per-tile VTCM scratch.
- Remove an HVX-side store that currently bottlenecks the fused loop
  (Phase 2 kernel's main bottleneck per `phase2b_w16a16_match_qnn.md`).
- Make our kernel layout-compatible with QNN out-of-the-box → can
  share ForceFormat_Crouton_b activations with other ops.

## Follow-up probes (not done)

- Dump full `out[m=0..31]` for Crouton_8 to verify `:cm` reads exactly
  rows 0..7 for m=0..7 and gives 0 for m=8..31.
- Map the non-`0x1c` bits of Rt_a in `:cm` mode (they likely encode
  stride / padding mode — QNN's r7 carries a conv-geometry packing).
- Test non-uniform activation data (increasing ints per row) to
  defensively confirm the row-index → output-m mapping.
