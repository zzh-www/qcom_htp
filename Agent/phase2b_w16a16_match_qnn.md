# Phase 2B — w16a16 match-QNN 48-tile replica

Phase 2B of the three-kernel plan: produce a kernel that **matches QNN's
exact 48-tile flat-layout structure for w16a16 at 512³**, proving we
understand QNN's tiling strategy bitwise and cycle-wise.

## Prior art findings

From `Agent/qnn_matmul_dtype_comparison.md` (2026-04-19 profiling):

| size | tile count | compute cycles | matmul_kernel |
|-----:|-----------:|---------------:|---------------|
|   32³ |     1     |     1,931      | q::ConvLayer_s1.opt |
|  128³ |     4     |    11,552      | q::ConvLayer_s1.opt |
|  256³ |    12     |    84,031      | (systemservice-lumped, profiler bug) |
|  512³ |    **48** |   564,055      | (systemservice-lumped) |

- **Layout**: flat `[1,1,M,N]` (not Crouton). Rules out Crouton packer.
- **Tile count scaling**: 1 → 4 → 12 → 48. Ratios {1, 4, 12, 48} =
  {1, 4·1, 4·3, 12·4}. Non-square, suggesting QNN tiles both M and N
  but also K at some threshold.
- **HMX utilization @ 512³**: 85.6% (from QHAS). HMX runs hot;
  per-tile MACs are fewer than for fp16 (which only has 2 tiles).

## Reverse engineering plan

1. **Capture QNN's intermediate tile data** via qnn-net-run:
   ```sh
   qnn-net-run --save_tensor_data_dir ... --input_list ...
   ```
   Dumps every intermediate tensor including post-tile outputs.

2. **Decode the 48-tile partition**:
   - Run 512³ w16a16 built-in. Examine `chrometrace_htp.json` per-op
     shapes for each of the 48 HMX kernel events.
   - Hypothesis: QNN splits M×K=512×512 into 4 M-slices, K×N=512×512
     into 4 K-slices × 3 N-slices = 12 (k,n) combinations per M-slice →
     4·12 = 48. Or some other decomposition; need data.

3. **Per-tile geometry**:
   - Each tile has a specific (M_range, K_range, N_range).
   - Measure per-tile cycles; our kernel should match per-tile ± 10%.

4. **Our replica**:
   - Kernel enumerates the 48 tiles in the same order QNN uses.
   - Each tile calls the 32×32×32 base HMX MAC (same one used in
     `hmx_int16x16_matmul_mn` but stripped of K-accumulation).
   - Output accumulates into an intermediate int32 buffer, then
     final combine at end.

## Why this is lower-priority than beat-QNN

- **Practical value**: lower. QNN's 48-tile path is 2900× slower than
  our straightforward per-(m,n) kernel at 512³ numerically, but QNN
  uses fewer lifetime cycles *elapsed* (564K vs our 1.64G) because its
  48 HMX tiles fire in parallel across HVX threads / the scheduler.
  Matching QNN gets us literal parity but not *better* than QNN.
- **Engineering effort**: high. QNN's tile enumeration is
  undocumented; requires reversing the qnn-graph-prepare output +
  cross-referencing each HMX event in optrace. At least 2 days.
- **User's "both tracks"**: the "beat-QNN" track is already delivered
  (our w16a16 works end-to-end); the "match-QNN" track is now
  bootstrapped via this design doc.

## Deferred — implementation plan when picked up

```
example/hmx_matmul_w16a16_match_qnn/
├── build.sh          — mirror hmx_matmul_w16a16/
├── src/
│   ├── Hmx{W16A16_MATCH}MatMulOp.cpp    — dispatcher with 48-tile enum
│   └── run_{match}_matmul.cpp            — host harness
└── kernel/
    └── hmx_int16x16_match_qnn.c          — 48-tile enumeration kernel
```

The 48-tile enumeration table `static const tile_t qnn_512_tiles[48]`
comes from the per-op shape data captured in step 1 above.

## Verification plan (when done)

1. Side-by-side: our replica vs QNN w16a16 at 512³:
   - Per-tile cycle count within 10%
   - Same tile count (exactly 48)
   - Same (M_range, K_range, N_range) per tile
   - Bit-exact output
2. Document in `Agent/phase2b_replica_results.md`.

Status: **design only** — implementation deferred pending fresh turn
budget.
