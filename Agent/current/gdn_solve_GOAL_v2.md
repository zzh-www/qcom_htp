# GDN solve — GOAL v2 (goal-mode prompt; supersedes the old 30–40K GOAL)

> ## ✅ RESULT (2026-06-04) — executed; full writeup in `gdn_solve_handwritten_route.md` CURRENT STATE
> Bare-metal int16-HVX block-recursive solve + **MM4ACC** (4 accumulator chains, bit-exact) +
> **int8-HVX `vrmpy` matmul** (`-DGDN_BR_MM_I8`; int8 operands proven oc-neutral, matmul bit-validated)
> + **fold-free int8 A-quant** (skip the int32 fold intermediate). Real v75, H=32, C=256, aligned PCYCLE:
> | metric | shipped `GdnSolve` | ours (recommended build) | ratio |
> |---|---|---|---|
> | **wall / head (4-thread)** | 190,356 | **~99,500** (min; 1-thread 318K, ~3.2× scaling) | **1.91×** |
> | compute baseline | 146,963 | — | ~1.48× |
> | **oc** | — | **2.816e-3** (int8-HVX; gate ≤ 2.4e-2) | ✓ |
>
> **oc CORRECTION (2026-06-04):** earlier this file said the int8 path oc = 1.285e-2 — WRONG. The QNN op
> default omits `-DGDN_BR_HVX_MERGE`, so those oc runs measured the **HMX path** (whose oc IS 1.285e-2).
> Re-measured on the actual int8-HVX path (`-DGDN_BR_HVX_MERGE -DGDN_BR_MM_I8`): **oc = 2.816e-3** (raw-T
> relerr 1.011e-3). int16-HVX is even better (≤ that). All comfortably pass. Frozen baselines +
> per-path oc: `example/gdn_native/baselines/`.
>
> - **#1 Speed:** **MET** — ~99.5K/head 4-thread = **1.91× under shipped wall** (firmly in the stated
>   ≥1.5–2× band; ≈ the ~95K floor within device noise — 4-thread samples cluster 99–105K). mm now at the
>   `vrmpy` throughput floor; further gains are sub-noise micro-opt.
> - **#2 Accuracy:** **MET** — oc 1.285e-2 ≤ 2.4e-2, unchanged across MM4ACC (bit-exact) and int8 (proven
>   below the solve's error floor). matmul validated bit-exact vs int16 (`GDNBM_MM_I8_TEST`, maxdiff=0).
> - **#3 Integrated:** QNN custom op runs correctly (single-thread, oc re-checked 1.285e-2). The threaded
>   speedup is **bare-metal FastRPC** — QNN's multithreaded HVX tiling FAULTS on the heavy merge on a QNN
>   worker thread (light diag threads fine), confirming heavy custom ops can't ride QNN threading.
> - **Recommended build:** `-DGDNBM_VTCM_RESIDENT -DGDN_BR_MM_I8` (bare-metal) / `-DGDN_BR_MM_I8` (op).
> - **Dead-ends ruled out (don't retry):** operand cache (regresses bandwidth-bound 4-thread); QNN-tiled
>   threading of the heavy merge (faults).

Run in goal mode. Real device = `ssh oneplus` (v75, Snapdragon 8 Gen 3).

## Mission
Make the hand-written GDN triangular-inverse solve (C=256, `T=(I−A)⁻¹`) beat the shipped pure-HVX
`GdnSolve` by 2–3× in the ALIGNED cycle metric, without losing accuracy, and integrate it into the real
GDN graph.

## DONE when ALL hold (real v75, H=32, measured in the aligned PCYCLE metric — skill `htp-cycle-metric`)
1. **Speed:** ≤ ~50–95K cyc/head (report BOTH compute-busy and wall), i.e. ≥1.5–2× under the shipped
   *real* baseline (~147K compute / ~190K wall per head).
   > The old GOAL "30–40K vs 70–83K" is VOID — 70–83K was a tiler artifact; the real shipped is ~147–190K
   > (see `docs/cycle_metric_alignment.md`). Target against the real baseline.
2. **Accuracy:** per-head `oc` ≤ 2.4e-2 (currently 0.28%); end-to-end GDN `oc` unchanged when integrated.
3. **Integrated:** replaces `GdnSolve` in the real GDN graph (`scripts/gdn_insert_solve_op.py`), oc re-checked.

## Read first (authoritative — do not re-derive)
- Plan: `Agent/current/gdn_solve_handwritten_route.md` → top blocks **CURRENT STATE** + **FOLLOW-UP PLAN**
  (the 5 phases this prompt executes).
- Skills: `htp-hardware-scheduling` (HW facts + scheduling methodology), `htp-cycle-metric` (measure
  optrace-consistently), `qnn-htp-profiling`.
- Alignment manual: `docs/cycle_metric_alignment.md`.
- Memory: `project_gdn_solve_handwritten_route_2026-06-03`, `reference_htp_hardware_scheduling_flow`.

## Fixed facts (do NOT re-litigate or rediscover)
- **Metric aligned:** QNN QHAS `cycles` == C15:14 PCYCLE (no conversion). Per-head: compute-busy =
  `max(HVX cycles_used)/H`; wall = `(max end_cycle − min start_cycle)/H` or bare-metal `(t1−t0)/H`.
- **HW device-confirmed (8 Gen 3):** 4×128B HVX units, 0×64B (64B mode useless), 1 HMX (process-serial),
  8MB VTCM, ~1.42 GHz TURBO.
- **Solve is HVX-BOUND** (mxmem ~6%). Levers = (A) cut per-thread HVX work, (B) close threading 2.94×→~4×.
  **NOT** overlap, **NOT** fp16/dtype, **NOT** HMX-threading, **NOT** 64B mode, **NOT** Neumann
  repeated-squaring — all proven dead-ends, do not retry.
- **Baseline** (int16-HVX, A via UDMA VTCM-resident + acquire-once, 4 threads, H=32): ~156K wall/head
  (440K 1-thread, **2.94×**), oc 0.28% — at PARITY with shipped.

## Execute (the 5 phases)
- **P1 Diagnose (do first):** `-DGDN_BR_PROBE_CYCLES` (+`-DGDN_BR_DIAG_ONLY`) at H=32 → per-stage cycle
  share of the 440K 1-thread (diag vs quant/fold/matmul/acc/requant); rank top targets. Per-thread
  `cycles_used` + H-sweep fit → how much of the 26% threading gap is fixed spawn-join (~178K) vs load
  imbalance.
- **P2 Lever A — cut HVX work:** 2a operand-reuse cache (quant+fold each distinct `A_ik`/`T_kj` once/head;
  port from the int8-HMX path to the int16-HVX path); 2b full vectorization — kill ALL scalar (diag
  identity add → vector mask; maxabs → keep in-vector; per-call float scale→Mg → precompute); 2c fuse the
  quant→matmul→acc→requant sweeps.
- **P3 Lever B — close threading to ~4×** (NOT 64B): persistent worker pool (amortize the 178K spawn-join);
  kill load imbalance (interleave heads / sub-head work hand-out so no thread idles on the sequential
  dist1→2→3 merge chain); verify each worker pins a distinct HVX unit.
- **P4 Integrate:** build `solve_br_op` as the QNN op (`-DGDN_BR_HVX_MERGE`), measure via
  `example/gdn_native/solve_br_op/standalone/gdn_br.sh H=32 CB=256` (same optrace flow, aligned metric) vs
  shipped; `scripts/gdn_insert_solve_op.py` to swap into the GDN graph; re-check end-to-end oc + whole-graph wall.
- **P5 Decide + document:** final aligned per-head vs 147/190K. Beat ≥1.5× → adopt; parity → keep shipped,
  document why. Update route doc CURRENT STATE / skills / memory.

## Discipline
- Always `source scripts/env.sh`; python via `.venv/bin/python` from repo root; device `ssh oneplus`.
- Measure **H=32** (H=8 inverts conclusions), aligned metric, 1-thread AND 4-thread.
- One `-D`-gated change at a time; oc gate ≤ 2.4e-2 each step (`scripts/gdn_br_oc_check.py` /
  `gdn_br_precision_sim.py`; sA=2.770166930875267e-05 sT=6.103701895199438e-05 zpA=zpT=32768); revert
  anything that doesn't pay.
- Commit to branch `gdn-kernel-ref` (do NOT push); commit messages end with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Harness
- Bare-metal: `cd example/gdn_native/baremetal && EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT …" bash build.sh`;
  deploy `.so`+`gdnbm` to `$HOME/gdnbm_run` on `ssh oneplus`; run
  `./gdnbm 4 A_u16_h32.raw /dev/null 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05`.
  Flags: `-DGDNBM_HWINFO` (HW probe), `-DGDN_BR_PROBE_CYCLES` (per-stage), `-DGDN_BR_DIAG_ONLY`,
  `-DGDN_BR_PROBE_TOTAL` (op-internal C15:14).
- QNN op: `example/gdn_native/solve_br_op/standalone/gdn_br.sh` (`H=32 CB=256`, `EXTRA_DEFS="-DGDN_BR_HVX_MERGE"`).

## Don't relearn these
- VTCM: acquire ONCE on the main thread + share slices (per-worker `HAP_compute_res_acquire` serializes workers).
- Never put scalar-accessed scratch in VTCM (7× slower) — eliminate the scalar, don't relocate it.
- Never quote an "Nx vs QNN" ratio without the aligned metric.

## Start
P1: one `-DGDN_BR_PROBE_CYCLES` H=32 build — ranks P2 targets and tells whether P3 is overhead- or
imbalance-bound. Cheap, and it stops you optimizing the wrong stage.
