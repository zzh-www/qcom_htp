---
name: htp-cycle-metric
description: The single home for how to COUNT and REPORT HTP cycles so a number is unambiguous and IDENTICAL in unit to QNN optrace — for comparing a hand-written/bare-metal Hexagon op against a QNN op (or the shipped GdnSolve), or any time you read/report a cyc number and aren't sure it's latency vs throughput vs wall vs aggregate vs domain. Use whenever you make a cycle/perf claim, an "my op vs QNN is Nx" comparison, or need the right way to report a kernel/op/graph cycle figure. Pins: QNN QHAS per-op `cycles` == the C15:14 PCYCLE register (no conversion); the four口径 (op latency = `num_dominant_path_cycles` / unit throughput = busy / per-call feed-inclusive / graph wall) and which to pick (by whether the unit is the saturated bottleneck); the value+口径+context report template; a "一看就懂" timeline; and the four traps that produce 2–6× wrong numbers (tiler tile-count, counter mixing, mean-tile/head, per-op latency-vs-throughput). Full manual: docs/cycle_metric_alignment.md.
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

## Which number? The four口径 — ALWAYS say which (+ report template)

A cycle figure is meaningless without its口径. Two axes: KIND (latency vs throughput) × GRANULARITY (op / call / graph).

| 口径 | read from | answers | use when |
|---|---|---|---|
| **op latency** | `num_dominant_path_cycles` | one op's critical-path time (data resident) | dependency chain / unit **idle-mostly** |
| **unit throughput (busy)** | `by_htp_type` Σ, or `max(htp_resources.cycles_used)`=DOMAIN | how long the unit is **occupied** | that unit is the **saturated bottleneck** |
| **per-call feed-inclusive** | single-call wall (kernel + load/format) | implementation quality | comparing impls; **= op latency only if feed hidden** |
| **graph wall (/head)** | `(max end_cycle − min start_cycle)/H` | end-to-end latency | the **final verdict** |

**Pick latency vs throughput by whether that unit is the saturated bottleneck.** Device gotcha: native int16 64³
= **256 latency vs 1167 throughput (6×)** — the 4 byte-passes pipeline, so latency ≪ throughput. (This trap once
flipped a GDN verdict — see trap #4.)

**Report template = value + 口径 + context:**
- ✅ `int16 64³ kernel latency = 256 PCYCLE (dominant-path, data resident)`
- ✅ `compute-busy/head = 146,963 (max HVX cycles_used / H)`  ·  `wall/head = 190,356`
- ❌ `int16 matmul = 10.4K` (no口径)  ·  ❌ `int16 = 6× u8i8` (that's throughput; irrelevant to an idle unit)

## Timeline (口径 一看就懂)

`■`=compute(kernel)　`▓`=feed/load (or producer)　time → (schematic, not to scale; numbers are the truth)
```
A. 同一个核, 数据在不在 VTCM 决定你看到哪个数:
     数据已就位:  ■264■                            = op latency (纯 kernel)
     还得先搬:    ▓▓ feed ~8K ▓▓ ■264■             = per-call feed-inclusive (核仅占 3%)
B. feed 藏不藏住 = matmul 之间有没有依赖:
     可批量/独立:  HVX ▓pack▓▓pack▓▓pack▓          ← 提前喂, 并行
                  HMX ■176■■176■■176■  → 连跑 ≈ latency+握手 (≈313/eq)
     依赖链:       ▓feed 8K▓■264■▓feed 8K▓■264■ … → 串行 = feed-inclusive (8.6K/eq)
C. 同一个核, latency vs throughput (int16 4 byte-pass 流水):
     ■■p1■■  ■■p2■■  ■■p3■■  ■■p4■■  (重叠)
     └ 关键路 latency 256 = 1.45× u8i8 ┘   总 MAC 吞吐(HMX-busy) ≈ 1167 = 6×  ← 别拿吞吐当核成本
```

## Four traps that give 2–6× wrong numbers (do NOT)

1. **mean-tile / heads.** The old shipped "70–83K cyc/head" was `mean(tile cycles)/8` — WRONG. The central
   tiler splits H=32 into **24 tile-instances** (not 4), ~6 run serially per HVX thread. Dividing one
   tile's `cycles` by `H` ignores that serialization → ~2× undercount. Never use `mean(tile cycles)/heads`.
2. **Counter mixing.** PCYCLE (QHAS/C15:14, ~1.42 GHz) ≠ the accelerator counter the repo note "1µs =
   4209 acc-cyc" refers to ≠ `chrometrace_runtrace.json` phase counters (~1.78 GHz). Never convert between
   them or compare across them.
3. **sum vs max of per-thread `cycles_used`.** `sum` = work volume (Σ over threads); the real compute time
   is `max` (threads run in parallel). Use max for domain cycles. (`sum(htp_op_instances[].cycles)` is
   likewise work volume, not wall.)
4. **latency vs throughput per op (can be ~6× wrong).** One op has TWO numbers:
   `num_dominant_path_cycles` = **latency** (critical path, data resident) vs `by_htp_type`/`cycles_used` =
   **throughput / busy** (occupancy). They diverge when internal passes pipeline — device: native int16 64³
   = **256 latency but ~1167 throughput (6×)**. Pick by whether the unit is the **saturated bottleneck**:
   idle-mostly / dependency-chain → latency; back-to-back / bottleneck → throughput. Reading throughput as
   "the kernel cost" once wrongly killed the int16 GDN-inverse merge (it's producer-bound, HMX idle-mostly →
   latency 1.45×, not throughput 6×). Full case: `Agent/current/int16_matmul_cycle_model.md`.

## Recipe for "my op vs QNN op" (apples-to-apples)

1. Pick ONE metric (compute-busy/head OR wall/head) and use it on BOTH sides.
2. Read both in PCYCLE (optrace `cycles_used`/`start_cycle` for the QNN side; `C15:14` for bare-metal).
3. Cross-check the clock once (PCYCLE span / accelerator-execute µs ≈ 1.42e3).
4. If the two ops are the SAME C++ (e.g. a custom op built both ways), build both and read `C15:14`
   inside the op — that is the cleanest direct proof (no tiler/counter ambiguity).

### 🔒 THE cross-implementation rule (this is trap #4's most expensive form — a GDN loop burned ~12 iterations on it)
Your bare-metal `C15:14`-around-the-call is **wall / feed-inclusive (口径④)**: matmul + every byte
moved into the HMX array. QNN optrace's *headline per-op* numbers — `num_dominant_path_cycles`,
`htp_op_instances[].cycles`, `by_htp_type` busy — are **latency / busy (口径②③)**: the matmul-only
slice, feed excluded (QNN bills feed to *separate* sidecar ops like `weights_to_vtcm`/`ForceFormat`).
**Putting "native 327" next to "ours 10,838" is comparing ② to ④ → a phantom 5–33× "gap" that is pure
口径 mismatch.** (Real case: a w16a16 64³ kernel that is *byte-identical* to native and uses a
*field-identical* descriptor still reads 10,838 wall vs native's "327" — because native's single-op
*graph-wall* is ALSO ~11,034. Same kernel ⇒ same wall, always; mxmem timing is data-independent.)
- **To compare two implementations, the ONLY admissible number is `graph-wall ÷ N`** (口径①, the whole
  op/graph span `max(end_cycle)−min(start_cycle)` over the matmul count) — on BOTH sides. Never compare
  an optrace per-op latency/busy figure to a bare-metal wall.
- A faster *batch* wall (e.g. native 128-batch = 2,020 wall/matmul vs single-op 11,034) is real and IS a
  口径① comparison — but it's overlap/amortization across ops, NOT the kernel being faster. Single-op
  wall is the kernel's true cost; batch wall measures the *schedule*.
- **A latency-floor fact ("the matmul is ~256 cyc") is NOT a gate on your wall measurement.** If your
  per-call wall reads ≫300, that is the *correct feed-inclusive wall*, not a "口径 error to fix". Don't
  go hunting for a missing fast-path because a floor number says 300 — first confirm whether the
  fast-path (e.g. descriptor-driven M-fan-out) even exists for your dtype (objdump the kernel; it may be
  byte-weight-only). Verify the kernel/descriptor are actually different before attributing a gap to them.

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
