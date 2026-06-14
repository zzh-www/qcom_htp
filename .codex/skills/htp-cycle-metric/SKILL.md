---
name: htp-cycle-metric
description: The single home for how to COUNT and REPORT HTP cycles unambiguously. ONE vocabulary only — the QNN optrace fields (num_dominant_path_cycles, per-unit cycles_used, graph-wall span); all self-invented口径 names (op-latency/unit-busy/per-call-wall/domain/feed-inclusive/①②③④) are DELETED as meaningless. Use whenever you make a cycle/perf claim or an "ours vs QNN is Nx" comparison. Pins: QNN cycles == C15:14 PCYCLE (no conversion, bare-metal maps onto the same fields); THE rule = compare same field + same shape + same scenario; why num_dominant_path ≠ cycles_used (same op, two values: 370 vs 1388 on native 64³); the traps that give 2–6× wrong numbers; and the 64³-custom-op-under-optrace recipe. Full manual: docs/cycle_metric_alignment.md.
---

# HTP Cycle Metric (unify on QNN optrace fields)

Make every HTP cycle number a QNN optrace field — nothing else. Process guide; full evidence: `docs/cycle_metric_alignment.md`.

## The one fact (device-proven, ratio 0.9963)

**QNN optrace cycles (`num_dominant_path_cycles`, `start/end_cycle`, `htp_resources[].cycles_used`) ARE the
`C15:14` PCYCLE register.** Proven by reading `C15:14` inside a QNN custom op vs QHAS for the same instance:
18,159,963 vs 18,227,561. **One counter, no conversion.** So a bare-metal `C15:14` read and an optrace
number are the SAME unit — and a bare-metal number is reported AS one of the QNN fields below, never as a
new口径. Clock sanity: PCYCLE/µs = graph PCYCLE span / `QNN accelerator (execute) time` µs ≈ **1422–1594**
(v75 TURBO); if your cycles/µs ≫ that you are reading a different counter.

## 口径 = the QNN optrace fields, and ONLY these (no self-invented names)

From `out/optrace/chrometrace_qnn_htp_analysis_summary.json`. There are exactly THREE:

| field | from | meaning |
|---|---|---|
| **`num_dominant_path_cycles`** | per-op `htp_op_types[].num_dominant_path_cycles_htp_0`; graph-level `data.dominant_path_htp_0` | the **critical dependency chain** = what's left after ALL overlappable work (parallel units, internal pipelining) is hidden. A **lower bound / "ideal-schedule" number**, not an occupancy. |
| **`cycles_used`** (per-unit HMX/HVX) | `htp_overall_summary[].htp_resources[].cycles_used` | the unit **actually occupied** — incl. its own pipeline bubbles = the **real per-unit cost** |
| **graph wall** | `max(end_cycle) − min(start_cycle)` | end-to-end makespan — actual, incl. all scheduling gaps (the final verdict) |

**Bare-metal `C15:14` maps DIRECTLY onto these — it is not a separate口径**: a back-to-back per-call read =
that op's `cycles_used` (occupancy); a single spawn→join span = graph wall. **DELETED vocabulary (do not
use): "op-latency / unit-busy / per-call wall / feed-inclusive / domain cyc / ①②③④".** Say the QNN field name.

## THE rule: compare ONLY same field + same shape + same scenario

Every phantom "Nx gap" in this project came from breaking this:
- **Cross-field** (e.g. `num_dominant_path` vs `cycles_used`): they DIFFER for the same op — native 64³
  ConvLayer = **370 `num_dominant_path` but 1388 `cycles_used`**. Comparing 370 to our 1320 `cycles_used`
  invented a phantom 3.6×. Same field: 1320 vs 1388 → parity.
- **Cross-scenario** (single op vs batch-amortized): single `[1,1,64,64]` = **11176 `cycles_used`**; the
  same conv inside a `[1,32,64,64]` supertile (n_tiles=8) amortizes to **1388 `cycles_used`/conv**. Two
  scenarios, one order of magnitude apart — never mix.
- **Cross-impl**: pick ONE field, read it on BOTH sides at the SAME shape+scenario. A bare-metal number
  (always `cycles_used` or graph-wall) must NOT be compared to an optrace `num_dominant_path`.

## `num_dominant_path` is an OVERLAP concept — composable, but NOT scalable from wall

`num_dominant_path` is the **critical dependency chain after ideal overlap** (every parallelizable /
pipelinable cycle hidden). Three consequences, all important:

1. **It is composable, not a black-box** — the graph-level dominant path (`data.dominant_path_htp_0`) =
   Σ of the per-op `num_dominant_path` ALONG the critical chain. So you can *build* a solve's dominant-path
   from its op dominant-paths + the dependency graph; you do NOT need optrace to "measure" it as one number.
   (A bare-metal op that never runs under optrace still HAS a dominant-path — compose it.)
2. **It is a lower bound, not occupancy** — one op, two values: native 64³ ConvLayer `num_dominant_path`=370
   but `cycles_used`=1388. **Nobody runs at the dominant-path**; the real per-unit cost is `cycles_used`.
   The gap (370→1388) = the op's own internal pipeline bubbles. They converge only on large shapes where
   fill/drain amortizes (256³: 74670 dompath ≈ 86202 cycles_used); on small shapes they split.
3. **You CANNOT scale it from `graph wall`** — `graph wall − dominant_path` = the *un-hidden* gaps
   (op-internal bubbles + cross-op scheduling idle). That gap depends on the schedule, not a fixed ratio.
   The relationship is a nesting of bounds: **`graph wall` (actual) ≥ critical-chain `dominant_path`
   (ideal-overlap floor)**, and that headroom IS the schedulable optimization space. Use dominant-path to
   answer "if the schedule were perfect, how fast?" — use `cycles_used`/wall for "what it costs now".

(Reading the busy/throughput figure as "the kernel cost" once wrongly killed the int16 GDN-inverse merge —
it's producer-bound, HMX idle-mostly, so its marginal HMX add is small.)

## Traps that give 2–6× wrong numbers (do NOT)

1. **Cross-field / cross-scenario** (above) — the #1 source; always state field + shape + scenario.
2. **mean-tile / heads.** Old "70–83K cyc/head" = `mean(tile cycles)/8` — WRONG (tiler splits H=32 into ~24
   serial tile-instances). Never `mean(tile cycles)/heads`.
3. **Counter mixing.** PCYCLE (optrace/C15:14, ~1.42–1.59 GHz) ≠ "1µs=4209 acc-cyc" ≠
   `chrometrace_runtrace.json` phase counters (~1.78 GHz). Never convert/compare across them.
4. **sum vs max of per-thread `cycles_used`.** `sum` = work volume (Σ over threads); real time = `max`
   (threads parallel). Use max.
5. **Blaming the tool for your config.** If optrace "crashes" or a number looks impossible, suspect YOUR
   setup first (wrong profile/descriptor/shape) and diff against a working example — see 64³ recipe.

## Getting a hand-written op measured under the SAME QNN optrace (apples-to-apples)

1. Build the op into a QNN custom-op package; run it under `qnn-net-run --profiling_level detailed
   --profiling_option optrace`; decode with `scripts/decode_qnn_optrace.py` (needs the ctx `*schematic.bin`).
2. Read the SAME field on both your op and native at the SAME shape. Cross-check the clock once.
3. **64³ w16a16 recipe (cron#71, the right way):** `native_record_256` profile (FORMULA_DESC = descriptor
   computed from M_t/N_t/K_t) + `MODE=chain_qdq` + `W16A16_NATIVE_ORACLE_DIR` (native oracle → weight
   sidecar) + device **HTP-only** op package (the CPU package fails to register). One-shot CI path:
   `SHAPES="64,64,64" bash scripts/w16a16_shape_sweep.sh`.
   - ❌ Wrong (crashes optrace): the `accepted` profile hard-codes 256³ descriptor fields (out_y/n_tiles=256);
     on a 64³ tensor the kernel reads/writes 256-wide → out-of-bounds → execute fault (plain execute
     tolerates it; optrace exposes it). That is a config error, **not** a "QNN can't trace small shapes" limit.
   - Result (both under optrace, same shape): our `HmxU16I16ToU16MatMul` 3789 `num_dominant_path` / 11152
     `cycles_used` vs native ConvLayer 3543 / 11176 → parity. Detail: `[[reference_64cube_conv_occupancy_vs_latency]]`.

## Worked result (C=256, H=32, 4-thread, all PCYCLE, same field)
| field / head | shipped `GdnSolve` | bare-metal BR (VTCM-resident) |
|---|---|---|
| `cycles_used` (HVX max) /head | 146,963 | ~161,040 |
| graph-wall /head | 190,356 | 156,287 |
→ parity (~1.0–1.1×). The "2–3×" once feared was trap #2.

## Reproduce / references
- Full manual: `docs/cycle_metric_alignment.md`. Decode: `scripts/decode_qnn_optrace.py`; field meanings: skill `qnn-htp-profiling`.
- 64³ case + recipe: `[[reference_64cube_conv_occupancy_vs_latency]]`. Op-internal probe: `-DGDN_BR_PROBE_TOTAL` in `example/gdn_native/solve_br_op/src/GdnSolveBROp.cpp`.
