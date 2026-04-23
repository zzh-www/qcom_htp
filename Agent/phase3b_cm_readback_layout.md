# Phase 3B — `:cm` readback layout RE (2026-04-23, DEFINITIVE)

## Headline finding

**`:cm` MAC produces only 16 output rows per invocation, not 32.** The
readback layout is:

```
out_lo halfwords 0..511:    non-zero, contain 16-row × 32-col result
out_lo halfwords 512..1023: zero (HMX never writes)
```

Measured at M=K=N=32 with all-1 input (expected 32 per cell):
- 512 halfwords = 32
- 512 halfwords = 0
- First 20 zero positions: 512, 513, 514, ..., 531 (continuous block)
- out_lo[511] is the last non-zero; out_lo[512] is first zero

For 32 rows output, need TWO MAC invocations (or a different Rt_a
encoding that enables all 32 rows — unknown).

## Decode formula (half tile)

For the 16×32 output tile written by ONE `:cm` MAC at some Rt_a config:
halfwords arranged in stride-2 per col, stride-64 per row:

```
mn_tile[ir, jc]  =  out_lo[ir*32 + jc]   for ir=0..15, jc=0..31
```

Wait, that's 16 rows × 32 cols = 512 halfwords, which is exactly the
non-zero range. But actually the raw dump showed halfwords 0..511 ALL = 32.
For 16 rows × 32 cols = 512 halfwords in a stride-1 layout, OK:

```
mn_tile[ir, jc] = out_lo[ir * 32 + jc]   for ir=0..15, jc=0..31
```

**TO BE VERIFIED next session**: is it really `ir*32+jc` (pure row-major
16×32) or `ir*64+2*jc` (16 rows × 32 cols stride-2 = 1024 halfwords but
only even positions used = 512 halfwords)?

My current test output oBuf[0..3] = 32 suggests positions 0,1,2,3 all = 32.
Under `ir*32+jc` that's right (first 4 cells). Under stride-2 `ir*64+2*jc`
that would be positions 0, 2, 4, 6 (not all 0..3).

So layout is **pure row-major 16×32**: `out_lo[ir*32+jc]` for ir in 0..15.

## Perf implication

Agent A's 7.92 cyc/MAC result: 400 MACs × 32 K × "MACs per packet" cyc.
Per "MAC packet" = 1 issue = 512 output-MACs (16×32, not 32×32 like Phase 2).
So 7.92 cyc/MAC was counting differently than Phase 2. Re-normalized to
output-MACs-per-packet, `:cm` throughput vs Phase 2 2-stream:

- Phase 2 2-stream: 32×32=1024 output-MACs/packet × 9.03 cyc/MAC = 9246 cyc/packet
- :cm row-major (half): 16×32=512 output-MACs/packet × 7.92 cyc/MAC = 4055 cyc/packet
- For 32 rows: 2 × :cm = 8110 cyc/packet → **12% faster than 2-stream**

Still a win, but modest (not the 1.1 cyc/MAC perceived benefit). Real
architectural value of `:cm` is **zero-pack consumption of row-major data**,
not raw throughput.

## Phase 3B action items (progress)

1. ✅ **2-pass MAC with activation offset +512** correctly produces rows
   0-15 and 16-31. Confirmed bit-exact for constant all-1 input (1024/1024).
2. ✅ **Dual-scale readback** wired (4 stores: top-lo, top-hi, bot-lo, bot-hi).
3. ❌ **Random data still mismatches** (non-zero max_err 13K-22K). Likely
   weight layout mismatch for `:cm` (row-major weight test insufficient —
   Agent A only tested all-1 which hides layout discrimination).

## Remaining puzzle

For random uint8 act × int8 weight @ 512³:
- cyc/MAC = 0.32 (very fast, mechanism clearly running)
- oBuf[0..3] = 13837 13837 5706 5706 (pairs duplicate suggests stride issue)
- oRef[0..3] = 3332 362 -1168 362

Three hypotheses:
1. Weight tile format for `:cm` is DIFFERENT from plain mxmem. Agent A's
   test used uniform weight (all-1) so didn't discriminate. Try Phase 2's
   packed weight format.
2. Dual-scale readback for `:cm` may have different scaling (not the same
   `(hi<<8)|(lo&0xff)` as Phase 2).
3. The 2-pass structure may need to store to the SAME output buffer with
   `:retain` (as is Phase 2's dualacc pattern) so both passes contribute.

Next-session action: probe weight-layout sensitivity — use non-uniform
weight (e.g., 1..32 in first row, then zero) + all-1 activation, verify
which weight-tile layout produces expected pattern via HMX.

Expected outcome: bit-exact matmul_v2 + cyc/MAC ≈ 2× Phase 2 at larger K
(since 2 MACs per tile instead of 1; Phase 2's dualacc is similar).

## Files at current state

- `example/hmx_matmul_phase3/src/HmxMatMulV2Op.cpp` currently dumps
  raw out_lo as int32 for diagnostics. Needs decode-fix for next session.
- `example/hmx_matmul_phase3/kernel/hmx_core_v2.c` has single-scale
  readback + row-major weight + `:cm` activation.
