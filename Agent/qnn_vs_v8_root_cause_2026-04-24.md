# QNN vs V8 — the real root cause is output layout (2026-04-24)

## What QNN actually does

Disassembled `hmx_convbbb1x1_stride1 @ 0x2ea740` in `libQnnHtpV75Skel.so`
(Hexagon v75).  The MatMul hot loop:

```asm
loop1 (outer, per output tile):
  r3 = r3 + 0x100           ; advance bias ptr (+256 B)
  bias = mxmem2(r3)         ; load 2 biases (512 B)
  mxclracc

  loop0 (inner, K/2 iterations — 2-MAC unroll):
    { p0 = cmp.eq(r26, #2); r26 -= 2
      r6  = memw(r1++#8)    ; act ptr 0
      r23 = memw(r1+#4) }   ; act ptr 1
    { r8 += 0x400           ; wt ptr += 1 KiB
      activation.ub = mxmem(r6, r24):cm
      weight.b      = mxmem(r8, r25) }
    { r8 += r25+1           ; wt ptr += 1 KiB
      activation.ub = mxmem(r23, r24):cm
      weight.b      = mxmem(r8, r25) }

  ; post-K:
  r10 = memw(r0++m0)        ; load NEXT output tile ptr from a list
  mxmem(r10, r11):after:cm:sat.ub = acc   ; write 1 KiB directly to r10
```

**Key observations:**

1. **Same HMX instruction we use** — `:after:cm:sat.ub`.  No exotic
   pair-mode MAC that would speed up the MAC loop.

2. **Rt_wt = 0x3FF and Rt_act = r7|0x1C** — exactly the values V8 already
   uses.

3. **2-MAC unrolled inner loop** — QNN issues 2 MAC packets per iter
   with alternating activation pointers but incrementing weight ptrs.
   P5 probe confirmed this gives ~7.98 cyc/MAC = same as 1-MAC (8.03).
   It is a LOOP UNROLL, not a pipelining mechanism.  Equivalent to
   compiler auto-unrolling our K loop by 2 — no architectural advantage.

4. **`sat.ub` writes DIRECTLY to the output ptr `r10`** (1 KiB = one full
   32×32 tile, contiguous DDR write).  No intermediate VTCM staging, no
   32-row scatter at stride N.  `r10` is loaded from a list via
   `memw(r0++m0)` — QNN walks a precomputed list of tile pointers.

5. **Output layout is [tiles, 32, 32]** (tile-layout), not `[M, N]`
   row-major.  This is what avoids the scatter.

## The real V8 bottleneck

Our V8 output is row-major `[M, N]`.  HMX sat.ub writes 1 KiB contiguous;
to place it at stride N in DDR we do 32 × 32-B scalar memcpys — each
fetch-modify-writes a DDR cache line (no write-combining because adjacent
rows are N bytes apart, different cache lines).

Measured breakdown at 512³:
- **mmv8 with inline scatter**: 1.6 M cyc total.  Per tile ≈ 6,280 cyc:
  - MAC work: ~130 cyc
  - sat.ub drain: ~?
  - **scalar scatter to DDR: majority of overhead**

When we tested splitting into `mmv8 → oTile VTCM → untile → oT DDR`:
- mmv8 alone at 512³: **281 K cyc** (≈1,100 cyc/tile, very close to
  QNN's MAC + sat.ub cost)
- untile alone at 512³: 1.6-2.3 M cyc (pure scatter from VTCM to DDR)
- Total: 2.0 M cyc — **worse** than the inline version because the
  inline scatter was overlapping with HMX pipeline drain.

So **the scatter IS the ~5000 cyc/tile overhead**, and it is
fundamentally DDR-latency-bound for row-major strided partial writes.

## Results from this round's experiments

| Experiment                                              | 32³ total | 512³ total | 1024³ total |
|---------------------------------------------------------|----------:|-----------:|------------:|
| Baseline (inline scatter, post pack_act fix)            |    14 K   |    1.69 M  |    7.9 M    |
| Split into mmv8+untile (tile-layout intermediate)       |    20 K   |    2.02 M  |    8.9 M    |
| mxswapacc pair-mode                                     |    14 K   |    1.59 M  |    7.5 M    |
| pair-mode + `:after:retain:cm:sat.ub`                   | **broke** |  **broke** |  **broke**  |

**Best single-op baseline stays at 1.69 M / 7.9 M cycles.**  mxswapacc
gives ≈1% noise-level improvement at 512³ but breaks correctness
because `:retain` is not honored by `:cm:sat.ub`.

## Conclusion

**To match QNN's per-tile throughput, V8 must accept tile-layout
output** (avoiding the row-major scatter entirely).  This is feasible
when V8 is an INTERMEDIATE op feeding another tile-aware op — which is
the standard QNN internal pattern (Crouton_b-chained pipeline).

For a **standalone MatMul producing `[M, N]` row-major DDR output**, the
~1.6 M cyc scatter at 512³ is a hard floor set by DDR partial-write
latency, not HMX.  Neither pair-mode nor mmv8 internal optimization can
lower it.

**Two realistic paths forward:**

1. **Expose tile-layout output option** in V8 for chained usage.  User
   opts in by tagging the output tensor as tile-layout; saves ~1.3 M
   cyc at 512³, brings V8 very close to QNN's per-tile throughput for
   in-graph intermediates.

2. **Accept current perf** for the row-major-output use case.  V8 at
   1.69 M / 512³ is 4.7× slower than V6 (which uses HVX requant +
   HMX-concurrent HVX scatter — a different pipeline that achieves
   the overlap V8 cannot).

## Files touched in this investigation (reverted after measurement)

- `example/hmx_matmul_phase3/src/HmxMatMulV8Op.cpp` — pair-mode + :retain
  attempt; tile-layout direct output attempt.  All reverted.
- `example/hmx_matmul_phase3/kernel/untile_to_rowmajor_hvx.c` — new op
  for the split experiment.  Registered in the interface but unused.
- `example/hmx_matmul_phase3/src/run_matmul_v8_graph.cpp` — graph split
  experiment; reverted.

Keeping the untile op source around as a reference; it can be re-enabled
if we later want the split/tile-layout path.
