# GOAL — pure-handwritten GDN triangular inverse (2–3× over the shipped HVX op)

Self-contained goal spec for goal-mode execution. Full context + per-head op decomposition + measured
per-op costs: `Agent/current/gdn_solve.md`. Device = `ssh oneplus`.

## North star
Rewrite the fused glue of the device-correct `example/gdn_native/solve_br_op/` so its **steady compute at
C=256 beats the shipped pure-HVX `GdnSolve` by 2–3×**, without losing accuracy.

## DONE when ALL hold (real v75)
1. **Speed:** `solve_br_op` C=256 steady ≤ **30–40K cyc/head 4-thread** (= 2–3× under shipped 70–83K),
   i.e. single-thread ≤ **~100K/head** (from 380K) + a working ≥3× thread scale-up. Metric = steady
   **DOMAIN** cycles (busiest unit = real time), NOT wall, NOT summed work-volume.
2. **Accuracy:** per-head T relerr ≤ **2.4e-2** vs `np.linalg.inv` (no worse than current BR 2.378e-2);
   end-to-end GDN `oc` unchanged when integrated.
3. **Correct + integrated:** device bit-exact preserved every step; replaces `GdnSolve` for prefill C in
   the GDN graph (`scripts/gdn_insert_solve_op.py`), oc re-checked.

## Scoreboard (cyc/head, C=256)
| | single-thread | 4-thread (product) | relerr |
|---|---|---|---|
| shipped HVX `GdnSolve` (beat) | 434,130 | 70–83K | 4.8e-5 |
| current `solve_br_op` (start) | ~380K | untuned | 2.378e-2 |
| **GOAL** | **≤ ~100K** | **≤ 30–40K (2–3×)** | **≤ 2.4e-2** |

## Task sequence (each = one bounded change → validate → no regress)
1. **Vectorize `gdn_pack_act_crouton8`** (GdnSolveBROp.cpp:372): kill the 256 scalar uint64 VTCM stores
   (actpack ~3,900→~hundreds/merge). Biggest single lever, no restructure.
2. **Operand-reuse cache:** quant+pack each distinct A_ik (act) / T_kj (wt) ONCE in VTCM (10 uses→6
   distinct ≈ 1.67×). Restructure recursion ~853–903 + merge ~656; mind 64 KB surface spacing.
3. **Cache the A fold** (`gdn_fold_block_raw` 10×→6×, BSS).
4. Speed remaining HVX glue (`gdn_quant_*`, depack) toward QNN-native leanness.
5. **Thread with own HVX contexts** toward ~3–4× (HMX stays on MAIN — worker HMX FAULTS).
6. Integrate into GDN graph; re-check oc end-to-end.

## Validate every step
```
cd example/gdn_native/solve_br_op/standalone
CB=256 H=8 bash gdn_br.sh                                   # relerr + aggregate cyc
CB=256 H=8 EXTRA_DEFS=-DGDN_BR_PROBE_CYCLES bash gdn_br.sh  # per-stage cyc (decode T head 0)
```
Baseline to track: 380K/head single-thread, relerr 2.378e-2, aggregate ~3.03M / 8 heads.

## Guardrails (don't relearn the hard way)
- **HVX∥HMX overlap is NOT a lever** — HMX is ~free and custom (QHPI) ops can't overlap (compiler
  `is_plugin_op` excludes supertiling; see `docs/qnn_htp_scheduling_and_custom_op_limits.md`). The win =
  cut HVX glue volume + own-context threading.
- HMX must run on the MAIN callback thread; HMX surfaces VTCM-only (spaced 64 KB).
- **2-pass scale is necessary** (1-pass → relerr 1.76, DEAD END).
- Keep changes gated/measurable; revert any cut that regresses relerr or doesn't pay.
