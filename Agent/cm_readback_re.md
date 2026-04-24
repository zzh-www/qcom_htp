# HMX `:cm` readback layout — silicon RE (definitive)

**Target:** SM8650 v75 (OnePlus 12)
**Date:** 2026-04-23
**Probes:**
  - `example/hmx_matmul_device/probe_cm_readback.c` (T1..T4)
  - `example/hmx_matmul_device/probe_cm_singlecell.c` (single (M,N) sweep)
**Raw logs:** `example/hmx_matmul_device/build/probe_cm_*_result.txt`

## Headline (DO NOT FORGET)

**Under `:cm` activation, ONE MAC + `:after.uh acc:2x1` readback only
populates 16 of the 32 output rows** — specifically the **ODD-indexed
activation rows (M = 1, 3, 5, …, 31)**. EVEN-indexed rows
(M = 0, 2, 4, …, 30) are *not* present in either the LO (bias 0x4000)
or the HI (bias 0x2000) readback. They are silently dropped (or
require a different MAC config we have not found).

**Implication**: the `:cm` "zero-pack row-major" kernel V4 is broken by
design. To get full 32-row output you would need *two* `:cm` MACs per
K-step (one for odd rows, one for even with rows shifted by 1) which
doubles HMX work. There is no real per-MAC win vs the V3 2-stream
plain path.

The "7.92 cyc/MAC :cm vs 9.03 cyc/MAC plain" gap Agent A measured in
`cm_row_major_re.md` was *per instruction*. Each `:cm` MAC instruction
performs only half the MACs of a plain instruction (16 rows × 32 K vs
32 rows × 32 K). On equal-work basis the two paths are essentially
equivalent. V3's 0.143 cyc/MAC at 512³ vs V4's 0.145 confirms this.

## Index mapping (for the ODD rows that *do* show up)

For `acc[M][N]` where M ∈ {1, 3, 5, ..., 31}:

```
m_odd_idx = (M - 1) / 2            // 0..15
idx       = (m_odd_idx / 2) * 64
          + 2 * N
          + (m_odd_idx & 1)
```

Verified with single-cell probe at N ∈ {0, 1, 7, 15, 31} for all M.
Examples:
  M=1, N=0  → idx=0
  M=3, N=0  → idx=1
  M=5, N=0  → idx=64
  M=31, N=31 → idx=511

The `idx` range used is 0..511 (half the 1 KiB rb buffer). The other
512 entries stay zero in our experiments.

## Dual-scale combine (for the ODD rows)

LO bias = 0x4000 (fp16 1.0), HI bias = 0x2000 (fp16 0.5):
- `lo[idx] = acc[M][N] + ⌊acc[M_paired][N] / 256⌋`  (we believe)
- `hi[idx] = ⌊acc[M_paired][N] / 256⌋`

Where `M_paired` is the EVEN partner — but since we cannot recover the
low byte of `acc[M_paired]`, the EVEN-row data is effectively
unreachable from this readback.

For our purposes (recovering the ODD rows only):
  `acc[M][N] = (int16)lo[idx] − (int16)hi[idx]`

V3's combine `((int16)hi << 8) | (lo & 0xFF)` is a different
arithmetic formula intended for the 2-stream plain path; it does not
apply here.

## Why this kills the `:cm` zero-pack idea

Phase 3 V4 was meant to:
  1. Replace V3's 2-stream activation pack (2 KiB/tile, halfword shuffle)
     with row-major activation (1 KiB/tile, contiguous gather)
  2. Use `:cm` MAC to consume the row-major tile directly
  3. Reuse V3's dual-scale decode

Step 3 is broken because of the readback layout. Step 1+2 alone do
work for 16 of the 32 output rows but require doubling MAC work to
fill the other 16. Net: no speedup, more complexity.

## What's still open

If a future probe finds a `:cm` MAC variant that produces full 32-row
output (e.g., a combination of `:cm` + some other modifier, or a
different activation Rt encoding that we haven't tried), V4 could be
revived. Candidates to try (none verified):
- `Rt_a = 0x800 | 0x1c` or other high-bit variants
- `:after.uh` *without* `:2x1` (single-stream readback?)
- `acc:2x2` / `acc:1x1` / other `:after` sub-modes
- Dual MAC pair using `:cm` and `:above` together (RE'd in Phase 1
  for a different purpose)

For now we accept V3 at 0.143 cyc/MAC as the working baseline and
move on to other directions in NEXT_STEPS.md.
