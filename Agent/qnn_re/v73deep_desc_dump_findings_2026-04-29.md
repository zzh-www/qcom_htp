---
name: V73DEEP descriptor-dump findings
description: Runtime dump of our V73DEEP V9 op-pkg descriptors at 256³ (Option 1 from wrapper_3dc2a8_TRACE.md follow-up). Confirms our hand-built descriptors match what the kernel expects but does NOT pinpoint the 747→346 packet gap.
type: project
---

# V73DEEP descriptor-dump findings (2026-04-29)

Build: `EXTRA_DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST -DV9_DESC_DUMP"`
Run: `bash example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v9_desc_dump.sh`
Output: `phase1_validation/v73deep_desc_dump/device_out/out.raw`
Parser: `scripts/parse_v73deep_desc_dump.py`

## Confirmed via dump (256³, M_t=N_t=K_t=8, mt_per_block=2)

```
Mask buffer (set_hmx_params_conv1x1(0x70b, 0, 0, 0, 0x20)):
  +0x04 = 0x700      out_rt_mask (sat.ub Rt)
  +0x0c = 0x77c      act_rt_base (with arg5=0x20 deep flag set)
  +0x18 = 0x7ff      alt_rt (last-K Rt_wt — 0x7FF in deep mode)
  +0x30 = 0x20       deep flag (bit 5)

od (hmx_conv_out_desc_t):
  out_tile_ptr_table       (out_tbl_all)
  out_table_stride_dwords  = 8         → m0 = 28 (8*4 - 4)
  out_y_stride_words       = 0
  n_tiles_pow2             = 64        → r20 = 8 (loop1 trip = M_t)
  m_total_minus_step       = 8         → r17 = 4 → no K-iter restart
  k_total_bytes            = 256       → r13 = 8 → 4 outer iters (-=2)

ad (hmx_conv_act_desc_t):
  act_ptr_pairs            (act_tbl_all)
  n_act_pairs              = 8         → r28 = 4 (loop0 trip = K_t/2)
  act_table_y_stride_words = 0

extra_param[16] = {1, 0, 0, ...}       fast path (r24=1), drain init r25=0
```

All values match what the V73DEEP kernel disasm at 0x2ebe40 expects. Output
is 100% bit-exact at 256/512/1024/2048/4096³ shapes.

## What the dump does NOT explain

747 vs native 346 packets. Our hand-built descriptors match the kernel's
expected fields and our work-trip counts (4 outer × 8 loop1 × 4 loop0 ×
2 MAC = 256 MAC packets + drains + bias) match what would correspond to
~358 packets per the disasm packet structure.

Empirical: 747 measured, 358 expected. The "extra ~390 packets" is unaccounted.

Possible explanations (none confirmed):
1. HMX engine stalls counted as committed packets — but PMU PROBE confirms
   chrometrace pkts ≈ PMU committed packets, so this is the real count.
2. Kernel has additional internal loops/code we're not seeing in our
   "loop1=8 × loop0=4" model.
3. Native uses a DIFFERENT call path (e.g., wrapper-loop with smaller per-call
   work) where per-call work + per-call wrapper-overhead < our single big call.
   But our memory says multi-call experiments (4 calls, 8 calls per-M-tile)
   were worse, not better.

## Static-RE attempt rules these out

- `extra_param` consumption: in fast path (extra_param[0]==1) only 2 entries
  consumed (r24=ep[0], r25=ep[1]). Our 16-entry array is plenty.
- p1=tstbit(n_act_pairs, #0): n_act_pairs=8 → p1=0 → drain code skips the
  extra MAC at 2ebf54. So we're NOT triggering the redundant MAC.
- Loop trips: r20=8, r28=4, r13 starting at 8 dec by 2 → 4 iters. All match
  our work model.

## Tools left in tree

- `example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp` — V9_DESC_DUMP branch
  in V73DEEP block. Dumps mask/od/ad/extra_param/act_tbl_all/out_tbl_all/
  derived shape values to the output Crouton_8 buffer (5 rows × 128 bytes).
- `scripts/parse_v73deep_desc_dump.py` — parses the dump.
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v9_desc_dump.sh` —
  build+run+parse runner.

## Next-step options (still open after this run)

1. **PMU-profile the kernel with V9_PMU_PROBE** to break down per-region
   packet count (probe-around-prologue, probe-around-loop, probe-around-drain)
   — would localize the 390 missing packets to a specific code region.
2. **Patch native kernel @ 0x2ebe40 prologue** to record packet counts at
   each loop boundary (NEXT_STEPS Phase 2 — runtime kernel patching).
3. **Inline the kernel body** (NEXT_STEPS Phase 3) — just write an HMX kernel
   asm-by-asm replica without prologue overhead.

The dump infra is reusable for future descriptor experiments.
