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
| **`cycles_used`** (per-unit HMX/HVX) | `htp_overall_summary[].htp_resources[].cycles_used` | the unit **actually occupied** (incl. fill/data-move + bubbles) = real cost of ONE isolated op. **⚠️ NOT sequence throughput: when ops pipeline, retire-rate ≪ cycles_used/N — see trap #6. For per-op throughput use `start_cycle` retire interval.** |
| **graph wall** | `max(end_cycle) − min(start_cycle)` | end-to-end makespan — actual, incl. all scheduling gaps (the final verdict) |

**Bare-metal `C15:14` maps DIRECTLY onto these — it is not a separate口径**: a back-to-back per-call read =
that op's `cycles_used` (occupancy); a single spawn→join span = graph wall. **DELETED vocabulary (do not
use): "op-latency / unit-busy / per-call wall / feed-inclusive / domain cyc / ①②③④".** Say the QNN field name.

## THE rule: compare ONLY same field + same shape + same scenario

Every phantom "Nx gap" in this project came from breaking this:
- **Cross-field** (e.g. `num_dominant_path` vs `cycles_used`): they DIFFER for the same op — native 64³
  ConvLayer = **370 `num_dominant_path` but 1388 `cycles_used`**. Don't compare 370 to our 1320 `cycles_used`.
  **⚠️ "same field 1320 vs 1388 → parity" is a trap (cron#76) AND "ours 1577 vs native 290 = 5× gap" is ALSO a
  trap (cron#78 correction): the native "290" is the HMX SUB-OP `cycles` inside a 32-conv batch where fill latency
  overlaps the adjacent op — it is NOT the conv's wall. Apples-to-apples = SAME scenario: native SINGLE conv
  (`mm_1x1x64x64`) ConvLayer = **1970 `cycles`/15 cyc-pkt ≈ our 1576** (we're NOT 5× slower); the batch's 290 only
  exists because `mm_64`'s graph is HVX-bound (HMX Σbusy 8.6%) and optrace charges the overlapped fill elsewhere.
  Never compare our single-conv to native's batch-sub-op. See trap #6.**
- **Cross-scenario** (single op vs batch-amortized): single `[1,1,64,64]` = **11176 `cycles_used`**; the
  same conv inside a `[1,32,64,64]` supertile (n_tiles=8) amortizes to **1388 `cycles_used`/conv**. Two
  scenarios, one order of magnitude apart — never mix.
- **Cross-impl**: pick ONE field, read it on BOTH sides at the SAME shape+scenario. A bare-metal number
  (always `cycles_used` or graph-wall) must NOT be compared to an optrace `num_dominant_path`.

## Report perf by the THREE QNN categories (NEVER one lumped number) — cron#79

QNN optrace tags every op with a **unit** and groups into exactly three kinds of work. **Report your op's
perf the same way** — split into the three categories, each tagged with its **QNN op name + QNN field** —
so a comparison is automatically same-category + same-field. Lumping them into one "wall" or "busy" number
is how the cross-口径 mis-compares happen.

| category | the work | unit | native ops | report field |
|---|---|---|---|---|
| **真算 (MAC)** | irreducible compute | **HMX** (tid256) | `q::ConvLayer_s1.opt` | `cycles_used` (occupancy) + packets/cyc-pkt + (native) `num_dominant_path` |
| **装料 (prep)** | feed: weight/act/bias format | **HVX** (tid512+) | `convert_weights_to_signed`,`Cast`,`ForceFormat_Crouton`,`Reshape`,`bias_*` | per-op `cycles` |
| **卸料+输入 (IO)** | DMA in/out | **DMA** (tid256) | `q::*InputSlice`,`q::*OutputSlice` | per-op `cycles` |
| *(waste)* | idle / scheduling gap | — | *(a gap, not an op)* | reported separately, NEVER folded into a category |

Rules that fall out (encode these):
- **真算 / 装料 / 卸料 are reported SEPARATELY.** Never collapse "MAC + its feed" into one per-call number
  and compare it to a pure-MAC sub-op (that is the native-263-vs-our-1577 trap). Compare 真算 to 真算,
  装料 to 装料.
- **真算 = HMX-busy** (`cycles_used` / per-call occupancy); **装料/卸料 = the HVX/DMA op `cycles`.**
- **Throughput = retire-interval or graph-wall÷N, NEVER `cycles_used`/N** (trap #6).
- Every number = **value + QNN field + shape + scenario** (single vs batch; wall vs HMX-busy).
- A bare-metal op that has **no native counterpart** (e.g. a solve's renorm/acc) is flagged "装料-alg,
  solve-specific, no native op" — never silently dropped or mislabeled.

**Canonical worked example = the pure-HMX GDN solve** (`docs/cycle_metric_alignment.md` §"Canonical 口径
map"): each instrument segment ↔ QNN op ↔ category ↔ field; side-by-side native via
`scripts/gdn_solve_qnn_aligned_report.py` (reads native optrace + our device stdout, prints both in the
same 3-category table). Our 真算-MAC = 1577 cyc / 111 pkt / 14 cyc-pkt **≈ native SINGLE ConvLayer 1970 /
130 / 15** (NOT the batch warm sub-op 263). The aligned report is also the device-side FARF (`GDN_PURE
QNN-ALIGN ...`) and host print (`[真算-MAC] / [装料-*] / [卸料-IO]`), gated by nothing extra (production
bit-exact).

## `num_dominant_path` is an OVERLAP concept — composable, but NOT scalable from wall

`num_dominant_path` is the **critical dependency chain after ideal overlap** (every parallelizable /
pipelinable cycle hidden). Three consequences, all important:

1. **It is composable, not a black-box** — the graph-level dominant path (`data.dominant_path_htp_0`) =
   Σ of the per-op `num_dominant_path` ALONG the critical chain. So you can *build* a solve's dominant-path
   from its op dominant-paths + the dependency graph; you do NOT need optrace to "measure" it as one number.
   (A bare-metal op that never runs under optrace still HAS a dominant-path — compose it.)
2. **It is a lower bound, not occupancy — the optrace `cycles` of an op DEPENDS on its batch context (cron#78).**
   The SAME native 64³ ConvLayer (`convhhh1x1_stride1`) reports **`cycles`=1970 when run SINGLE** (`mm_1x1x64x64`,
   15 cyc-pkt) but **`cycles`=263 when run in a 32-conv batch** (`mm_64`, 2.1 cyc-pkt) — a 7.5× swing for ONE op,
   because in the batch the op's fill latency overlaps the PREVIOUS op's drain so optrace charges less to THIS op.
   ⇒ **An op's `cycles` is NOT an intrinsic per-op cost; it's scenario-dependent.** (cron#76 read the batch 263/290
   and mislabeled it the conv's throughput vs our 1577 → phantom 5× gap. WRONG: our 1576 ≈ native SINGLE 1970.)
   Also: `mm_64`'s graph is HVX-bound (HMX Σbusy only 8.6% of span); the conv's true sustained wall ≈ graph-span/N
   ≈ 4087, not 290. And bare-metal C15:14 ≈ the SINGLE-op `cycles` (our 1576 ≈ native single 1970; both incl. the
   un-overlapped fill). They converge on large single shapes where fill/drain amortizes within the op (256³: 74670
   dompath ≈ 86202 cycles_used). **The lever to cut a kernel's `cycles` is fewer/leaner packets, NOT cross-conv
   double-buffer pipelining — measured ZERO gain from software fill‖drain (HMX has 1 acc, no bank). cron#78.**
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
6. **`cycles_used`/N as per-op SEQUENCE throughput (cron#76, cost me 2 wrong verdicts).** `htp_resources[].cycles_used`
   is the unit's occupied-cycle counter and **includes overlappable fill / data-movement** (the HMX resource line
   carries `dram_read`/`vtcm_read`). When consecutive ops PIPELINE (op N+1's fill hides under op N's MAC), the real
   **per-op throughput (retire rate) ≪ `cycles_used`/N**. Device proof: native `mm_64` = 32× 64³ ConvLayer, HMX
   `cycles_used`=54076 → 54076/32 = **1690/conv (OVERSTATED)**, but the actual `start_cycle` retire interval between
   consecutive convs (fed, in-cluster) = **~290/conv**, and Σ per-op `cycles` = 11176 ≈ 290×32 (agrees with retire,
   NOT with cycles_used). Using 1690 wrongly concluded "we're at native parity (our 1320 ≈ native 1388)"; the truth
   is **native retires a 64³ conv every ~290 cyc back-to-back; a single one (or our hand-driven loop) costs ~1300-1577**.
   **For sequence throughput use the `start_cycle` retire interval (start[i+1]−start[i] of consecutive same-unit ops)
   or graph-wall/N — never `cycles_used`/N (single-op `cycles_used` likewise does NOT predict a fed sequence).**
   `cycles_used` is still correct for "is this ONE unit the busy bottleneck over the graph"; just don't divide it by
   op-count and call it throughput.
   **⚠️ cron#77 mechanism correction (PMU + M-fanout, device): the ~290 is NOT "native pipelines the 8 mxmem walks
   within one conv" — the 8 walks (4 output sub-tiles × 2 byte-passes) accumulate, the byte-passes are serial. The
   ~290 = native's QNN supertile OVERLAPS consecutive ConvLayer ops (fill[i+1] under drain[i]): the mm_64 timeline
   shows conv0 cold=2725, conv1-7 warm=279-337, each a SEPARATE op, retire 290 < dominant_path 370. And the gap is
   cyc/PACKET not packet count: at the right n_tiles our hand-written op fires FEWER packets than native (111 vs 130
   at n_tiles=8 — the "587 packets" once seen was n_tiles=64 = 8× over-walk), but stalls at ~14 cyc/pkt vs native ~2.0.
   Critically, this ~290 is UNREACHABLE by batching the byte-identical convhhh kernel: an M-fanout sweep (B=1/2/4/8
   independent 64-row blocks in ONE invocation) gives per-64³ = 1574/1454/1387/1354 → converges to the NTSWEEP
   walk-floor ~1320, NOT 290 — batching only amortizes the per-call prologue, it does NOT make one kernel invocation's
   walks overlap across convs the way QNN's supertile scheduler does. So "our kernel == native" holds for ONE isolated
   op (cycles_used 11152≈11176) but a fed SEQUENCE of our single-conv invocations cannot hit native's overlapped 290.**
   **⚠️⚠️ cron#78 FINAL correction — the "290 = native conv throughput we should reach / we're 5× slower" framing is
   ITSELF wrong (the trap-6 core "don't use cycles_used/N" stays valid; this fixes the secondary claim):** the native
   "290/263" is the HMX SUB-OP `cycles` in the 32-conv batch where fill overlaps the adjacent op — NOT the conv's wall.
   Device proof: the SAME native convhhh run SINGLE (`mm_1x1x64x64`) reports `cycles`=**1970 (15 cyc-pkt) ≈ our 1576**.
   `mm_64`'s graph is HVX-bound (HMX Σbusy 11176 = only 8.6% of the 130K span); the conv's true sustained wall ≈
   graph-span/32 ≈ **4087**, not 290. So our consumer is NOT 5× behind native — comparing our single-conv to native's
   batch-sub-op was a cross-scenario error. And the recoverable lever is **fewer/leaner packets** (a from-scratch
   dilate micro does the same 16 tile-MACs in 363 cyc/81 pkt/4.5 cyc-pkt vs convhhh 1576/113/14), NOT cross-conv
   double-buffer pipelining — from-scratch dilate+deep micro-kernels measured **SERIAL == PIPE (zero fill‖drain gain;
   HMX has 1 int32 acc, no bank)**. Reproduce: `-DGP_PKTPROBE` build, stats[18/19] (dilate SERIAL/PIPE), stats[12-17]
   (PMU: THREAD_IDLE≈1564/COPROC_BUSY=0), stats[20-23] (SMT spinners don't shrink the conv).**

7. **consumer-thread-occupancy ≠ HMX-unit-MAC (cron#81, cost 1 inverted verdict).** A kernel's
   back-to-back per-call `cycles_used`/occupancy is the *thread* running the kernel — it is **NOT** the
   HMX unit's MAC ("真算"). Device proof: the SAME 16 tile-MAC costs **363 cyc** in a lean from-scratch
   dilate micro (81 pkt, 4.5 cyc/pkt) but **1576 cyc** in convhhh (113 pkt, 14 cyc/pkt) — a 4.3× gap that
   is non-MAC kernel bloat (bias staircase + M-loop + THREAD_IDLE/WAIT between MAC packets), NOT the HMX
   computing. So calling the pure-HMX consumer's 1576/conv (1.31M total) "真算 / HMX compute" is wrong:
   the irreducible 真算 floor = ~363/conv ≈ 0.28M; the other ~1.0M is liftable. Two follow-on rules:
   - A **Σ work-volume** (a stage summed over all threads, e.g. wt-pack 2.52M Σ = 4 producers = 0.63M each)
     is NOT a wall — **never rank wall levers by a Σ %**; compare it to the roofline floor max(feed/P,
     consumer-occupancy), not to the wall.
   - **slowest-prod-life (critical-path)** includes spin = the producer *waiting the HMX*; that spin is
     consumer-dependent, so "lmax > consumer ⇒ producer-bound" is a trap. Decide bound-by from the
     roofline floor max(parallel-feed/P, serial-consumer-occupancy), not from lmax alone.
   Full level taxonomy (6 levels: HMX-unit-MAC / consumer-occupancy / Σ-work-volume / critical-path /
   roofline-floor / wall): `docs/cycle_metric_alignment.md` §"Canonical cycle taxonomy".

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
   - Result (both under optrace, same shape, SINGLE op): our `HmxU16I16ToU16MatMul` 3789 `num_dominant_path` / 11152
     `cycles_used` vs native ConvLayer 3543 / 11176 → single-op parity. **⚠️ cron#76: single-op `cycles_used` parity
     does NOT mean SEQUENCE parity** — in a fed batch native pipelines its 8 mxmem walks and retires a 64³ conv every
     **~290 cyc** (`start_cycle` interval, `mm_64`), while our hand-driven mxmem stays at ~1577/call (165/walk serial).
     So native is ~4.5× faster per conv in a sequence even though one isolated op looks equal. Detail: `[[reference_64cube_conv_occupancy_vs_latency]]`.

## Worked result (C=256, H=32, 4-thread, all PCYCLE, same field)
| field / head | shipped `GdnSolve` | bare-metal BR (VTCM-resident) |
|---|---|---|
| `cycles_used` (HVX max) /head | 146,963 | ~161,040 |
| graph-wall /head | 190,356 | 156,287 |
→ parity (~1.0–1.1×). The "2–3×" once feared was trap #2.

## Reproduce / references
- Full manual: `docs/cycle_metric_alignment.md`. Decode: `scripts/decode_qnn_optrace.py`; field meanings: skill `qnn-htp-profiling`.
- 64³ case + recipe: `[[reference_64cube_conv_occupancy_vs_latency]]`. Op-internal probe: `-DGDN_BR_PROBE_TOTAL` in `example/gdn_native/solve_br_op/src/GdnSolveBROp.cpp`.
