---
name: htp-cycle-metric
description: Measure HTP op cycles so the number is IDENTICAL in unit to QNN optrace — for comparing a hand-written/bare-metal Hexagon op against a QNN op (or against the shipped GdnSolve). Use whenever you need an apples-to-apples cycle comparison, a "my op vs QNN is Nx" claim, or you read a cyc/head number and aren't sure it's wall vs aggregate vs domain. Pins the proven fact (QNN QHAS per-op `cycles` == the C15:14 PCYCLE register, no conversion), the per-head metric definitions QHAS uses, and the three traps that produce ~2× wrong numbers (tiler tile-count, counter mixing, mean-tile/head). Full manual: docs/cycle_metric_alignment.md.
---

# HTP Cycle Metric (optrace-consistent)

Use this to make any HTP op cycle measurement directly comparable to QNN optrace.
Process/measurement guide only. Full evidence + reproduce: `docs/cycle_metric_alignment.md`.

## The one fact (device-proven, ratio 0.9963)

**QNN QHAS per-op `cycles` (and `start_cycle`/`end_cycle`, `htp_resources[].cycles_used`) ARE the
`C15:14` PCYCLE register.** Proven by reading `C15:14` inside a QNN custom op and comparing to QHAS for
the same instance: 18,159,963 vs 18,227,561. **One counter, NO conversion factor.** So a bare-metal
`C15:14` read and an optrace cycle number are in the SAME unit — compare them directly.

Clock: PCYCLE/µs = graph PCYCLE span / `QNN accelerator (execute) time` µs ≈ **1422 ≈ 1.42 GHz** (v75
TURBO). Use this as a sanity check — if your `cycles/µs` ≫ 1.42e3 you are reading a different counter.

## The per-head metrics (define which one you mean)

For a QNN-tiled, multi-thread op, read `out_s/optrace/chrometrace_qnn_htp_analysis_summary.json`:
- **compute-busy / head = `max(HVX htp_resources[].cycles_used) / H`** — the DOMAIN cycle (busiest unit),
  i.e. kernel/algo efficiency, no inter-op gaps. (HMX tid=256; HVX threads tids=512..515.)
- **wall / head = `(max end_cycle − min start_cycle) / H`** — latency incl. QNN per-tile dispatch bubbles.

For a bare-metal op: read `C15:14` (`pcyc()`) around the whole spawn→join; `(t1−t0)/H` ≈ **wall/head**
(parallelism already folded in via /H). Subtract the fixed spawn/join/power overhead (isolate via an
H-sweep linear fit) for the compute-only figure. To get an op-internal total directly, use the
`-DGDN_BR_PROBE_TOTAL` pattern (read `C15:14` inside the op, write it to output head 0).

## Three traps that give ~2× wrong numbers (do NOT)

1. **mean-tile / heads.** The old shipped "70–83K cyc/head" was `mean(tile cycles)/8` — WRONG. The central
   tiler splits H=32 into **24 tile-instances** (not 4), ~6 run serially per HVX thread. Dividing one
   tile's `cycles` by `H` ignores that serialization → ~2× undercount. Never use `mean(tile cycles)/heads`.
2. **Counter mixing.** PCYCLE (QHAS/C15:14, ~1.42 GHz) ≠ the accelerator counter the repo note "1µs =
   4209 acc-cyc" refers to ≠ `chrometrace_runtrace.json` phase counters (~1.78 GHz). Never convert between
   them or compare across them.
3. **sum vs max of per-thread `cycles_used`.** `sum` = work volume (Σ over threads); the real compute time
   is `max` (threads run in parallel). Use max for domain cycles. (`sum(htp_op_instances[].cycles)` is
   likewise work volume, not wall.)

## Recipe for "my op vs QNN op" (apples-to-apples)

1. Pick ONE metric (compute-busy/head OR wall/head) and use it on BOTH sides.
2. Read both in PCYCLE (optrace `cycles_used`/`start_cycle` for the QNN side; `C15:14` for bare-metal).
3. Cross-check the clock once (PCYCLE span / accelerator-execute µs ≈ 1.42e3).
4. If the two ops are the SAME C++ (e.g. a custom op built both ways), build both and read `C15:14`
   inside the op — that is the cleanest direct proof (no tiler/counter ambiguity).

## Worked result (C=256, H=32, 4-thread, all PCYCLE)
| per head | shipped `GdnSolve` | bare-metal BR (VTCM-resident) |
|---|---|---|
| compute-busy/head | 146,963 | ~161,040 |
| wall/head | 190,356 | 156,287 |
→ parity (~1.0–1.1×). The "2–3×" once feared was an artifact of trap #1.

## Reproduce / references
- Full manual + exact commands: `docs/cycle_metric_alignment.md`.
- Decode optrace: `scripts/decode_qnn_optrace.py`; flow + QHAS field meanings: skill `qnn-htp-profiling`.
- Op-internal probe: `-DGDN_BR_PROBE_TOTAL` in `example/gdn_native/solve_br_op/src/GdnSolveBROp.cpp`.
