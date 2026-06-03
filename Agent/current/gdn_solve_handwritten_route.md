# GDN triangular inverse — pure-handwritten route (target + per-head op decomposition)

## GOAL (for goal-mode execution)

**North star:** rewrite the fused glue of the device-correct `example/gdn_native/solve_br_op/` so its
steady compute at C=256 beats the shipped pure-HVX `GdnSolve` by **2–3×**, without losing accuracy.

**DONE when ALL hold (real v75, `ssh oneplus`):**
1. **Speed:** `solve_br_op` C=256 steady compute ≤ **30–40K cyc/head 4-thread** (= 2–3× under the shipped
   70–83K/head), i.e. single-thread ≤ **~100K/head** (from current 380K) *and* a working ≥3× thread
   scale-up. Metric = steady DOMAIN cycles (per [[feedback_kernel_efficiency_compute_cycles_not_wall]]),
   not wall, not work-volume.
2. **Accuracy:** per-head T relerr ≤ **2.4e-2** vs `np.linalg.inv` (no worse than the current BR
   2.378e-2), and end-to-end GDN `oc` unchanged when integrated.
3. **Correct + integrated:** device bit-exact preserved at every step; replaces `GdnSolve` for prefill C
   in the real GDN graph (`scripts/gdn_insert_solve_op.py`), oc re-checked.

**Scoreboard (cyc/head, C=256):**
| | single-thread | 4-thread (product) | relerr |
|---|---|---|---|
| shipped HVX `GdnSolve` (beat this) | 434,130 | 70–83K | 4.8e-5 |
| current `solve_br_op` (start) | ~380K | (untuned) | 2.378e-2 |
| **GOAL** | **≤ ~100K** | **≤ 30–40K (2–3×)** | **≤ 2.4e-2** |

**Task sequence (each = one bounded change → `gdn_br.sh CB=256` bit-exact+relerr+PROBE_CYCLES, no regress):**
1. Vectorize `gdn_pack_act_crouton8` (line 372): kill the 256 scalar uint64 VTCM stores (actpack
   3,900→~hundreds/merge). Biggest single lever, no restructure.
2. Operand-reuse cache: quant+pack each distinct A_ik (act) / T_kj (wt) ONCE in VTCM (10 uses→6 distinct,
   ~1.67×). Restructure recursion ~853–903 + merge ~656; mind 64 KB surface spacing.
3. Cache the A fold (`gdn_fold_block_raw` 10×→6×, BSS).
4. Speed the remaining HVX glue (quant `gdn_quant_*`, depack) toward QNN-native leanness.
5. Thread with own HVX contexts toward ~3–4× (HMX stays on main — worker HMX FAULTS); pick BL/thread
   layout so heads pipeline.
6. Integrate into the GDN graph; re-check oc end-to-end.

**Guardrails / invariants (do NOT relearn the hard way):**
- HVX∥HMX overlap is NOT a lever — HMX is ~free and custom ops can't overlap
  ([[project_gdn_hvx_hmx_overlap_impossible_2026-06-03]]). The win = cut HVX glue volume + threading.
- HMX must run on the MAIN callback thread (spawned-worker HMX faults). VTCM-only for HMX surfaces.
- 2-pass scale is necessary (1-pass → relerr 1.76, DEAD END).
- Keep changes gated/measurable; revert any cut that regresses relerr or doesn't pay (like cut #1).

---

# GDN triangular inverse — pure-handwritten route (analysis + per-head op decomposition)

The QNN custom-op route for `T=(I−A)⁻¹` at prefill C=256 is at its floor (shipped int16-packed HVX
`GdnSolve`, ~70–83K steady cyc/head) and cannot get HVX∥HMX overlap (custom ops are excluded from
supertiling — see `docs/qnn_htp_scheduling_and_custom_op_limits.md`). This doc defines the **pure-
handwritten** target (no QNN runtime tax, self-managed VTCM/threads) and decomposes the implementation
into its constituent ops, **per single head**, at C=256.

Frequency is held constant throughout (same v75 HVX/HMX at burst, ~1.05 GHz), so **cycles ↔ time are
1:1** and the whole analysis is in steady compute cycles.

## Target (steady compute, 32×256×256, frequency-invariant)

Measured baseline (real v75, 4 HVX threads, `solve_op/standalone/gdn_shape.sh CS=256 H=32`), steady =
warm-tile op cycles, cold-start/power-on/VTCM-acquire/dispatch excluded:

| | steady compute (32 heads) | per head |
|---|---|---|
| **current pure-HVX baseline (to beat)** | 2.2–2.65M cyc ≈ **0.53–0.63 ms** | 70–83K cyc ≈ 17–20 µs |
| **TARGET — 2–3× lower** | **0.75–1.3M cyc ≈ 0.18–0.31 ms** | **23–42K cyc** |

Calibration: profile-viewer `16,726,879 cyc ↔ 3974 µs` ⇒ 1 µs = 4209 accelerator-cycles (4-thread
aggregate). The 0.53–0.63 ms is **pure compute only** — the end-to-end execute wall was ~4.2 ms
(power-on ~10.5 ms once, VTCM-acquire 678 µs, RPC/dispatch), all of which pure-handwritten also avoids
but which is NOT part of this compute-vs-compute target.

**Go/no-go bar:** below ~2× is not worth the build (keep the shipped simple bit-exact HVX op); 2–3×
makes it worth it. Structural ceiling is ~5.6× (diagonal-only HVX MAC ≈ 12.6K/head + free HMX), so the
2–3× target leaves margin for the irreducible glue — plausible, but the whole bet rides on the glue floor
(measure it before building: see last section).

## Why pure-handwritten can reach it (the levers QNN denies)

1. **VTCM-resident, no DDR re-read** — kills the `fold` tax (A read once, not re-read per merge).
2. **Own HVX contexts** — lift the thread ceiling from QNN's ~2.5× (backend holds the contexts) toward ~4×.
3. **No per-op dispatch / Cast / ForceFormat / scratch-claim tax.**
4. HVX∥HMX overlap is NOT a lever here (HMX is ~free); the win is purely cutting HVX work + threading.

## Algorithm (per head, C=256, nb=4 blocks of BL=64)

`T=(I−A)⁻¹`, A strictly-lower. Block-partition into a 4×4 grid of 64×64 blocks `A_ij`, `T_ij`:
- **Diagonal:** `T_ii = (I − A_ii)⁻¹`, i=0..3.
- **Off-diagonal (i>j):** `T_ij = T_ii @ S_ij`, where `S_ij = Σ_{k=j}^{i−1} A_ik @ T_kj`.

The 6 off-diagonal blocks expand to **16 64³ matmuls** per head:
`T10=T11@(A10·T00)`, `T21=T22@(A21·T11)`, `T32=T33@(A32·T22)` (2 mm each),
`T20=T22@(A20·T00+A21·T10)`, `T31=T33@(A31·T11+A32·T21)` (3 mm each),
`T30=T33@(A30·T00+A31·T10+A32·T20)` (4 mm). Total 2·3+3·2+4 = 16.

## Per-head op decomposition (what to hand-write, and why each exists)

Data: each block is 64×64 = 4096 elems. "×N" = invocations per head. M5 col = measured single-thread
QNN cost (the tax-inflated reference); Target col = pure-handwritten VTCM-resident + vectorized.

| # | op (kernel) | unit | meaning — why it exists | ×/head | M5 cyc | target |
|---|---|---|---|---|---|---|
| 1 | `solve_tri64` | HVX | **the irreducible part**: forward-substitution inverse of a 64×64 unit-lower-triangular `(I−A_ii)`, int16. HMX cannot do a triangular solve (data-dependent back-substitution), so the diagonals stay on HVX. | 4 | 92K* | 12.6K |
| 2 | `quant_i16_to_i8` | HVX | HMX is 8-bit; every matmul operand (an `A_ik` block, or an intermediate `T_kj`/`S`) must be quantized int16→int8: compute scale, mul, round, clamp. | ~28 | 90K | ~10K |
| 3 | `pack_act_crouton8` | HVX | reorder an int8 operand into HMX **crouton8** activation layout (each K-tile a separate contiguous 32×32-blocked tile, row4/byte interleave). HMX reads activation only in this VTCM layout. | 16 | }40K | ~22K |
| 4 | `pack_wt_kmajor` | HVX | reorder the other int8 operand into the HMX **K-major 32×32** weight stream. HMX reads weights only in this layout. For `T_ij=T_ii@S`, the merge-1 output `S` is re-packed here as merge-2's weight. | 16 | }24K | ~8K |
| 5 | `effective_bias` | HVX | per output column `eff[n] = −128·Σ_k wt[k,n] + bias_q[n]` — folds the activation zero-point (act stored as u8 = signed+128) and bias into the HMX drain so a **signed** int matmul comes out correct. | 16 | 9K | ~6K |
| 6 | `hmx_matmul64` | **HMX** | the v73deep u8×i8 conv1x1: `P_int=Σ(act_u8−128)·wt_i8`, drain `out_u8 = clip(FLOOR(P_int·scale_f16/512) + (baseline≫7), 0,255)`, with `baseline=16384` injecting a +128 zero-point so the signed result fits u8 (recover `int8=out_u8−128`). **~free.** | 16 | 14K | 14K |
| 7 | `depack_crouton8` | HVX | de-pack the u8 crouton8 HMX output back to natural [64,64] order (closed-form byte map) so HVX can accumulate it. | 16 | 30K | ~15K |
| 8 | `requant_u8_to_i16` | HVX | recover `int8 = out_u8−128`, rescale to an int16 code — bring the HMX output back into the int16 accumulation domain. | 16 | 36K | ~15K |
| 9 | `acc_i32` | HVX | accumulate the inner sum `S_ij = Σ_k A_ik@T_kj` in int32 codes before the final `T_ii@(·)` matmul (avoids requant per term). | 6 | 12K | ~12K |
| 10 | `assemble_T` | HVX | write the 4 diagonal + 6 off-diagonal 64×64 blocks into the full `T[256,256]` in the downstream layout (uint16/int8). | 1 | 4K | ~4K |
| — | (`fold` / DDR re-read of A) | HVX | **TAX, not real work** — M5 re-read A blocks from DDR per merge; VTCM-resident kills it. | — | 63K | ~0 |

`*` M5's `solve_tri64` 92K is itself mostly the DDR-fold tax; the pure MAC of 4× 64-solve is ~12.6K.

**Single-thread target total ≈ 12.6 + 10 + 30 + 6 + 14 + 15 + 15 + 12 + 4 ≈ 120K/head.** Then own-context
threading at ~3–4× → **~30–40K/head** → lands (just) inside the 23–42K target band. The margin is thin,
so the two load-bearing assumptions — (a) the glue floor (#2–#8) really collapses to ~95K single-thread
VTCM-resident, and (b) threading really reaches ~3–4× — are exactly what must be proven before building.

## MEASURED single-thread per-op costs (2026-06-03, real v75, hvx_threads=1)

At `hvx_threads=1` the decoded per-op cycle IS the clean single-thread steady cost (no 4-thread
aggregation ambiguity). Measured from `gdn_concurrency_probe.sh` (C=64 B=64 combined graph for the
matmul glue; C=256 B=32 solve graph for the baseline). cyc/head = op total ÷ heads.

**Per-op single-thread cost (C=64 block = one 64×64):**

| op# | my op | QNN op measured | cyc/head (1-thread) |
|---|---|---|---|
| #1 | `solve_tri64` | `GdnSolve` (C=64) | **21,390** /block |
| #2/#8 | quant/requant | `q::Cast` | 503 |
| #3/#7 | crouton pack/depack | `q::ForceFormat_Crouton` | 436 |
| #4 | k-major wt pack | `convert_weights_to_signed.shuffled` | 874 |
| #5 | effective_bias | `bias_weight_update` + `bias_scale_shuff` | 659 |
| #6 | hmx matmul | `q::ConvLayer_s1.opt` | **237** |
| — | reshape/slice glue | `Reshape` + `Slice_contig.tcm` | 1,004 |

**One native 64³ matmul end-to-end ≈ 3,713 cyc/head single-thread — kernel is only 237 (6%); the other
94% (3,476) is glue** (Cast 503 + ForceFormat 436 + convert_wt 874 + bias 659 + Reshape/Slice 1,004).
This empirically confirms: HMX compute is negligible; the cost is Cast + layout-repack + bias glue.

**Single-thread C=256 baselines (the apples-to-apples "to beat"):**

| C=256 single-thread, per-head | cyc/head | vs full-solve |
|---|---|---|
| **full triangular solve (shipped HVX op)** | **434,130** | 1.0× |
| block-recursive built from native ops (extrapolated 4×#1 + 16×3,713) | **~145,000** | **~3.0×** |

⇒ The block-recursive ALGORITHM already wins ~3× single-thread when the merges use lean native-op glue
(not M5's 441K hand-glue-in-one-op). M5 lost on *implementation tax*, not the algorithm. Note: this is
single-thread; the shipped product number (70–83K/head) is the 4-thread steady metric (the 1→4-thread
factor is ~5×, measured separately, not a clean ×4 — must be re-measured for BR, don't assume).

## Fusion floor & plan (b) — what hand-fusion removes from the 145K

Classify the per-merge glue (3,476/merge) by fusibility, given a single fused kernel with all 64×64
blocks **VTCM-resident in one int8/crouton layout**:

| glue component | /merge | fate under fusion |
|---|---|---|
| `q::Cast` (502) | 503 | **eliminated** — keep int8 resident; requant only at the solve↔merge boundary, not per matmul |
| `Reshape`+`Slice_contig` | 1,004 | **eliminated** — pure per-op tensor-boundary artifacts; a fused kernel has none |
| `convert_weights_to_signed` | 874 | **mostly eliminated** — done once per *distinct* weight; `T_kj` is reused across merges (6 distinct weights vs 16 uses ≈ 2.7×) |
| `bias_weight_update`+`bias_scale_shuff` | 659 | **eliminated/precomputed** — effective bias once per weight |
| `ForceFormat_Crouton` | 436 | **mostly eliminated** — if the diagonal solve writes its output directly in crouton and intermediates stay crouton-resident; floor ~100–200 |
| `ConvLayer_s1` kernel | 237 | **irreducible** (the matmul) |

Fused per-merge irreducible ≈ ForceFormat(~150) + kernel(237) + minimal requant(~100) ≈ **~500/merge** →
16 × ≈ **8K**; one-time per-weight prep (convert+bias for ~6 distinct weights) ≈ **~9K**; merges total
fused ≈ **~17K**. Diagonal: fuse the 4 solves (shared setup, A resident, drop the per-op GdnSolve overhead
~13K/block beyond the ~8K pure forward-subst) → ≈ **~35K** for 4 blocks.

**Fused block-recursive single-thread ≈ ~50–55K cyc/head** (35K diag + 17K merges) → **~8× under the
434K single-thread baseline.** The ~40K of eliminable per-merge glue (Cast+Reshape+Slice+redundant
convert/bias) is the entire fusion prize; the kernel (237×16≈4K) and the diagonal forward-subst are the
irreducible floor.

**Fusion plan:** one custom op (or bare-metal kernel) holding all 16 blocks in VTCM in int8/crouton;
(1) 4 diagonal `solve_tri64` writing **directly in crouton int8**; (2) merges read blocks in place,
weights `convert`-ed once and cached, effective bias precomputed, **no Cast/Reshape/Slice between merges**;
(3) accumulate `S_ij` in int32, final `T_ii@S` per off-diag block; (4) one requant pass at the end.

## ✅ GO/NO-GO — RESOLVED: GO (2026-06-03)

Decided from measured device data + sim kernel floor (no full sim-glue build needed; the device
`ForceFormat_Crouton` is the authoritative crouton pack/depack floor — more real than a sim with no
memory system). The merge glue measured three ways:

| implementation | glue /merge | × over floor |
|---|---|---|
| **irreducible floor** (ForceFormat 436 + kernel 237) | **~673** | 1× |
| QNN native ops (Cast+ForceFormat+convert+bias, incl eliminable boundaries) | 3,476 | 5× |
| **current hand-fused `solve_br_op`** (M5: quant90+hmxpack64+depack30+requant36 ÷16) | **~13,750** | **~20×** |
Sim kernel floor (no QNN): u8i8 64³ = 417 pcyc single-call / ~109 steady; device kernel 237.

**Key finding: the current hand-fused glue is ~20× above the floor (and ~4× above even QNN's native
ops). M5's "BR loses" was an IMPLEMENTATION failure (bad glue), not algorithmic.**

Single-thread C=256 verdict (vs the measured 434,130 baseline):

| fused-glue quality | BR 1-thread cyc/head | vs 434K |
|---|---|---|
| current hand glue (M5) | ~441K | ~1.0× (loses — M5's result) |
| written to native-op leanness (3,476/merge) | 35K diag + 16×3,476 + prep ≈ **~100K** | **~4.3×** |
| written to the irreducible floor (673/merge) | 35K + 16×673 + 9K ≈ **~55K** | **~8×** |

Product 4-thread conversion (baseline 4-thread = 70–83K/head): even at a pessimistic 2.5× BR threading,
~55–100K / 2.5 ≈ 22–40K → still **2–3.8×**; at 4× → 3–7×. **Clears the 2–3× target with margin.**

**GO. The work is rewriting the fused glue, not the algorithm or the kernel** — kill per-op
Cast/Reshape/Slice boundaries, reuse convert/bias once per distinct weight, have the diagonal solve write
directly in crouton int8, keep all blocks VTCM-resident. Target the ~673–3,476/merge band (current is
~13,750). The HMX kernel (237) and diagonal forward-subst are the irreducible floor.

**Implementation log (rewrite of `solve_br_op` glue):**
- **Cut #1 — 2-pass→1-pass scale (DUD, 2026-06-03, reverted).** Replacing the 2-pass scale estimation
  with the analytic `loose` bound directly gave relerr **1.76** (off-diag blocks → 0; `loose` ≈ 100× the
  true max|P|) for only **~7%** cycle saving (3.03M→2.87M / 8 heads). The 2-pass is small (~27K/head)
  AND necessary — NOT a target. (The doc's earlier "loose ≈ 1.29e-2" claim is wrong.) Baseline at C=256
  H=8 single-thread: aggregate **~3.03M cyc / 8 heads ≈ 380K/head**, relerr 2.378e-2.
- **Packer hotspot found (2026-06-03).** `gdn_pack_act_crouton8` (line 372) packs with **256 scalar
  uint64 stores into VTCM** (NOT HVX) — scalar→VTCM latency is high, which is why `actpack` ≈ 39K/10 ≈
  3,900/merge while the already-HVX-vectorized `gdn_pack_w8_kmajor` (vshuff) `wtpack` ≈ 16K/10 ≈ 1,600,
  and QNN's tuned `ForceFormat_Crouton` is 436. So the per-merge glue is expensive because OUR HVX glue
  kernels are slow, NOT (as the native-op graph suggested) inter-op Cast/Reshape/Slice — `solve_br_op` is
  one fused op and never had those. ⇒ the (b) "~3,476/merge native level" needs the PACKERS to match
  QNN's, not just operand caching.
- **Prioritized cuts for the next focused pass (each ~one bounded change + device bit-exact + relerr):**
  1. **Vectorize `gdn_pack_act_crouton8`** to HVX vector stores / BSS-stage + bulk HVX copy (kill the
     scalar-VTCM-store penalty). Biggest single lever (~3,900→~hundreds /merge). No restructure.
  2. **Operand-reuse caching** — quant+pack each distinct A_ik (act) / T_kj (wt) ONCE into a VTCM cache
     (10 uses → 6 distinct ≈ 1.67×). Bigger restructure (recursion ~853–903 + merge ~656 + VTCM layout,
     mind the 64 KB surface spacing).
  3. **Cache the A fold** (`gdn_fold_block_raw` 10×→6×, BSS not VTCM, cheap).
  No single cut is transformative (~7–15% each); the 8× target is the COMBINATION. This is multi-cut
  focused work, not a one-shot. Baseline to track: 380K/head single-thread, relerr 2.378e-2.
- **DEAD END confirmed:** 2-pass→1-pass scale (cut #1, relerr 1.76).

### Focused-pass RESULTS (2026-06-03, all device bit-exact, relerr held at 2.378e-2 throughout)

C=256 H=8 single-thread aggregate **3,043,707 → 1,924,321 cyc/8 heads (−37%, 380K → ~240K/head)**. Each
step validated on real v75 (`gdn_br.sh CB=256`), relerr unchanged (every cut is deterministic / provably
identity, so bit-exactness is preserved — the metric is purely cycles).

| # | cut | commit | per-stage effect (cyc/head) | aggregate |
|---|---|---|---|---|
| #1  | vectorize `gdn_pack_act_crouton8` (256 scalar-VTCM stores → HVX ror+mask) | `2155888`+ | actpack 40,106→15,691 | 3,043,707→2,853,179 |
| #1b | drop redundant scalar diag upper-tri zeroing (forward-subst already leaves code 0 → requant writes zpT) | | requant 55,500→16,885 | 2,853,179→2,554,594 |
| #1c | vectorize depack (its scalar-copy twin: combine nt0/nt1 crouton halves → aligned HVX stores) | | hmxdepack 28,804→7,497 | 2,554,594→2,403,004 |
| #2  | operand-reuse cache: quant+fold each distinct operand once/head (A_ik act 10→6, T_kj wt 10→6, T_ii act 6→3) | | quant 89,526→58,491; fold 15,298→9,231 | 2,403,004→2,094,659 |
| #2b | cache PACKED HMX surfaces in VTCM (crouton act / k-major wt / eff) + constant loose bound (both operands rail to ±127 ⇒ loose = 64·127·127) | | actpack→9.5K, wtpack→11.9K, eff→6.6K, pint 38K→24K | 2,094,659→**1,924,321** |

| #3 | maxabs-fusion: producers (fold/solve) track the exact maxabs in-register; quant takes it and skips its 128-vec maxabs scan (bit-exact). Covers A_ik act + diag act/wt (12 of 21 quants) | | quant 58.4K→~54K | 1,924,321→**1,889,690** |

**Current per-head single-thread breakdown (clean ~236K):** quant ~54K · diag ~48K · pint 24K · requant 17K
· hmxkern 15.6K · acc 12K · wtpack 12K · actpack 9.5K · fold 9K · depack 7.6K · eff 6.6K · widen 3.7K.

**SINGLE-THREAD IS NEAR ITS CLEAN FLOOR (380K → 236K = −38%).** Remaining cuts are ≤2% each:
- **maxabs-fusion gave only ~2%** — the per-element quant-loop (fixed-point mul/round/clamp/narrow over
  4096 elems), NOT the maxabs scan, is the quant cost; irreducible without changing precision.
- **The "off-diag T is already int8 → skip its wt quant as an identity" cut is INVALID** (verified by
  reasoning, not tried): the re-quant deliberately *re-normalizes* a sub-127 block to the full ±127 range
  (`sQ=maxval/127`), raising precision. Skipping it regresses relerr. Only the maxabs (not the rescale) is
  fusable, and that is already done.
- The diagonal triangular solve (~48K) is HVX-bound (data-dependent back-substitution) and near floor.

Pushing single-thread toward ~100K would need a deeper algorithmic change to the quant-loop or the solve,
not more glue cleanup.

### THREADING (Task 5) — RESOLVED: simple head-parallel is IMPOSSIBLE for this HMX op (2026-06-03)

Device-verified on real v75 (unsigned PD, `gdn_br.sh` gated builds):
- **spawn + `qurt_hvx_lock` on a worker WORKS** — `GDN_BR_THREAD_TEST` returns create_rc=0, hvx_ok=1, HVX
  sentinel correct, join_rc=0; `GDN_BR_USE_THREADS -DGDN_BR_NT=2 -DGDN_BR_DIAG_ONLY -DGDN_BR_NO_HMX_LOCK`
  runs **bit-exact** (diag relerr 1.077e-4) with 2 workers. ⇒ HVX head-parallelism is fully viable.
- **`qurt_hmx_lock()` AND `qurt_hmx_lock2(QURT_HMX_SHARED_LOCK)` on a spawned worker FAULT the graph** —
  and it is the *lock call itself*, not the kernel: `DIAG_ONLY` with the worker HMX-lock but no kernel
  still faults. v75 has 2 HMX units + a shared-lock mode, but neither is grantable to a dynamically
  spawned thread in the QNN-managed PD. HMX is bindable ONLY to the main callback thread.

**Implication for the BR route (important):** BR is an HMX-resource op (`multithreaded=false`), so QNN will
NOT auto-thread it, AND manual threading can't put HMX on workers. So **BR is stuck at single-thread
~236K/head**, while the shipped pure-HVX `GdnSolve` threads freely via QNN to **70–83K/head (4-thread)**.
⇒ as-is, **BR single-thread LOSES ~3× to the shipped op on the 4-thread metric.** The GO decision's
"own-context threading 3–4×" assumption is **REFUTED** — consistent with
[[project_gdn_hvx_hmx_overlap_impossible_2026-06-03]] (custom ops can't get HMX concurrency).

### TASK 6 (end-to-end oc) — DECISIVE: BR is also an ACCURACY dead-end (2026-06-03)

`scripts/gdn_br_oc_check.py` injects the **device** BR-T into the fp64 GDN chunk forward, isolating the
solve's contribution to oc (sanity: exact-T injected → 8.6e-9 vs the internal solve; exact-solve oc vs the
real golden `o` → 2.6e-3, so the forward is faithful). Result on the real p29_L00 C=256 chunk:

**device BR-T → oc relerr 0.73 (73%). UNUSABLE.** Per-block T error explains it:

| | T00 | T10 | T20 | T30 | T11 | T21 | T31 | T22 | T32 | T33 |
|---|---|---|---|---|---|---|---|---|---|---|
| ‖exact‖_fro | 8.47 | 1.36 | **0.65** | **0.36** | 8.52 | 1.30 | **0.62** | 8.48 | 1.29 | 8.58 |
| relerr(BR) | 1e-4 | 0.19 | **1.11** | **0.35** | 1e-4 | 0.21 | **1.01** | 1e-4 | 0.16 | 1e-4 |

Diagonal blocks are excellent (1e-4) but **off-diagonal blocks have relerr 0.16–1.11, compounding with merge
depth** (1-merge ≈0.2, 2-merge ≈1.0). The whole-T relerr (3.3%) is diagonal-norm-dominated and **HID** this —
the GOAL's "T relerr ≤2.4e-2" proxy is misleading; **oc is the real metric and it fails catastrophically.**
Root cause is fundamental: **HMX is 8-bit**, so every merge quantizes its operands to ±127 and the small
off-diagonal results (‖·‖<1.4) drown in 8-bit accumulation noise across 2–3 merges. An HMX-based triangular
solve cannot produce accurate off-diagonals; the shipped int16-HVX `GdnSolve` gets 4.8e-5 precisely because
it stays int16.

### CORRECTION (2026-06-03): the 73% was a device 8-bit SCALE bug, NOT an algorithm limit

Host sim `scripts/gdn_br_precision_sim.py` (block-recursive solve, per-block symmetric quant, real chunk →
oc): **the BR ALGORITHM is accuracy-viable.**

| merge precision | oc relerr |
|---|---|
| clean 8-bit (exact-max scale) | **9.5e-3** |
| clean 8-bit + diagonal-subtraction (user's idea) | 6.0e-3 |
| **int16** | **3.4e-5** |
| int16 + diagonal-subtraction | 2.5e-5 |

So clean 8-bit already gives ~1% oc and **int16 gives 3.4e-5** — the device's 73% is a regression vs the
algorithm's own ceiling. Localized (splice device error band-by-band into exact T → oc): the **dist-1
sub-diagonal blocks (T₁₀,T₂₁,T₃₂) drive it** — splicing device dist-1 alone → oc 0.875. oc is hypersensitive
to dist-1, and the device's **2-pass loose-bound HMX scale** gets dist-1 to only 0.19 relerr vs clean
8-bit's 0.056. ROOT CAUSE = the coarse 2-pass scale estimation on the oc-critical near-diagonal merges
(the user's "scale" instinct was right). int16 removes the sensitivity entirely.

### ACCURACY RESOLVED ON DEVICE (2026-06-03): the bug was the kernel FLOOR drain

The 73% oc was NOT inherent to 8-bit and NOT just the scale. Sim isolated it: per-merge int8 **round** →
oc 1.0e-2, but int8 **FLOOR** → oc 0.66 (the v73deep drain FLOORs `(P_int+eff)*scale_f16/512`, biasing the
small off-diag codes toward 0; oc is hypersensitive). **Fix (shipped, device-verified):** inject +0.5 LSB
into the effective bias on the OUTPUT pass (`rdelta=round(256/scale_f16)`) so the drain rounds-to-nearest,
plus a scale **refine pass** (the loose bound left pass-1 codes ~1 → 21% dist-1 scale error). Device result:
**oc 0.73 → 0.67 (refine) → 1.3e-2 (round)**; off-diag block relerr 0.52→0.18; whole-T relerr 2.69e-2→1.36e-2
(now under the 2.4e-2 gate). **oc 1.3% matches the GDN native-quant baseline (1.35e-2) → Task 6 PASSES at
8-bit.** Cost: the refine pass adds ~16K/head (H=8 single-thread 236K→252K). The user's "it's the scale /
the diagonal 1" instinct drove this investigation to the FLOOR root cause.

**GOAL status now:** (2) accuracy ✅ MET (oc 1.3%, relerr 1.36e-2); (3) integration oc ✅ validated
(`scripts/gdn_br_oc_check.py`); (1) speed ❌ still blocked — HMX op is single-thread-only.

### PURE-HANDWRITTEN ROUTE (chosen 2026-06-03): escape QNN, self-manage HVX/HMX/threads

User direction for SPEED: drop the QNN custom-op framework and run a **bare-metal FastRPC HAP** that owns
its threads + HMX + VTCM. This is GOAL.md's original "pure-handwritten" vision and the only way past the
threading wall.

**Why it can work (the QNN HMX-on-worker fault was an API/PD-management artifact, not a HW limit):**
- raw `qurt_hmx_lock()` from a spawned worker FAULTED inside QNN's backend-managed unsigned PD.
- the PROPER API is **`HAP_compute_res_hmx_lock3(ctx, HAP_COMPUTE_RES_HMX_SHARED, &mutex, timeout)`** /
  `_hmx_unlock3` — "lock/unlock HMX directly from the threads using HMX" (HAP_compute_res.h). v75 has 2 HMX
  units + a SHARED mode. In our own HAP we acquire HMX per-thread via this API → worker HMX should work.

**Feasibility CONFIRMED (all present):** `qaic` (ipc/fastrpc/qaic/Ubuntu/qaic), host `libcdsprpc.so`
(android_aarch64), `HAP_compute_res_hmx_lock3`, device = `pineapple`/cdsp. Template = SDK
`examples/calculator` (IDL→qaic→stub+skel; manual hexagon-clang skel build mirrors the QNN op build.sh).

**Roadmap:**
1. **DECISIVE PROBE — DONE 2026-06-03: GO ✅.** `example/gdn_native/baremetal/` (qaic IDL → hexagon skel +
   aarch64 host, unsigned PD on cdsp/pineapple; `run.sh`). Spawned qurt workers EACH acquire HMX + run HVX:
   **2 and 4 workers all succeed** (ctx nonzero, HVX sentinel set). Key findings:
   - `HAP_compute_res_hmx_lock3` is **NOT_SUPPORTED (0x80000404)** on this device. The working model is
     **per-thread `HAP_compute_res_acquire(attr with set_hmx_param(1))`** (older HMX-in-acquire path) — call
     it from each worker; `set_hmx_param` is required or acquire returns 0.
   - **The QNN "HMX bound to main callback thread" fault was a QNN-PD-management artifact, NOT a hw/qurt
     limit.** In our own HAP, worker HMX works. ⇒ threaded HVX+HMX is viable → the route is GO.
   - (Workers share one process-wide ctx value; whether HMX runs truly concurrently on both units vs shares
     is TBD, but every worker CAN run HMX, and the 93%-HVX glue parallelizes regardless.)
2. Port the solve: diagonal int16-HVX forward-subst (carry over) + the merges. Merges can stay HMX (now
   threadable via hmx_lock3) OR go int16-HVX; keep the **FLOOR→round drain fix** (rdelta) either way.
3. Self-managed VTCM (no QNN scratch tensor), own thread pool (heads partitioned), no per-op QNN tax.
4. Measure on device: oc (reuse `gdn_br_oc_check.py`) + 4-thread steady cycles vs the 30–40K GOAL band.

Carry-overs from the QNN op: the diagonal solve, fold, the −38% glue vectorizations (pack/depack ror+mask),
operand-reuse caching, and the **FLOOR→round drain fix** (the single most important correctness fix).

### VERDICT (for the QNN-op variant): accuracy SOLVED (oc 1.3%); speed needs the pure-handwritten route above

- **HMX is wrong on BOTH counts:** it can't thread (HMX bound to main thread → 3× slower than shipped) AND
  8-bit forces the coarse-scale accuracy cliff. Abandon the HMX merge engine, not the algorithm.
- **int16-HVX block-recursive solve is the path:** merges as int16-HVX matmuls (reuse the diagonal solve's
  `Q6_Ww_vmpyacc_WwVhRh` MAC) → oc 3.4e-5 (sim-proven) AND pure-HVX → threadable (HVX-on-worker works).
  MAC count ≈ 16×64³ ≈ 4.2M dense vs the shipped naive forward-subst's ~8.4M serial → plausibly faster too.
  Remaining unknown = SPEED (build + measure on device); accuracy and threadability are now both cleared.
- The −38% single-thread glue wins (#1–#3) targeted the HMX-merge glue and don't transfer to an int16-HVX
  matmul merge (different engine); the diagonal solve, fold, and overall recursion DO carry over.
- Fallback if int16-HVX isn't fast enough: keep shipped `GdnSolve` (4.8e-5, threadable, 70–83K).

**(superseded) Only remaining path to a competitive BR: HVX-worker + main-thread-HMX marshalling** — N HVX workers do
the 93% HVX glue (quant/pack/depack/requant/fold/acc/diag) for their heads; main is an HMX server running
the 32 kernel calls/head. Rough ceiling: HVX 221K/head ×8 ÷4 workers ≈ 442K wall ≈ **~55K/head 4-thread**
(beats shipped 70–83K, still MISSES the 30–40K GOAL band; needs the HVX glue cut further too). It is a
large producer/consumer subsystem (qurt signals, per-worker VTCM surfaces, HMX-server loop) with real
sync-overhead risk — a major build with uncertain payoff. **Decision point for the user before investing.**

**Metric/probe note:** per-stage cyc decoded from T head-0 (op writes g_c_* there under
`-DGDN_BR_PROBE_CYCLES`); read uint16 codes back via `code=round(T_float/sT)+32768`, pair into uint32.
The PROBE build was silently failing earlier because the getters referenced `g_c_fold` declared *after*
them — declarations now hoisted above the getters.

## (superseded) Go/no-go experiment — kept for reference

Measure the **glue floor** with zero QNN, in hexagon-sim (pcycles), reusing `scripts/gdn_hmx_matmul_sim.py`
/ `gdn_blockrec_sim.py`: for ONE 64³ merge, time the full VTCM-resident chain #2→#8 (quant → pack_act +
pack_wt → hmx_matmul64 → depack → requant), single-thread pcyc. Then `×16 + solve_tri64×4 (12.6K)`:
- extrapolated single-thread **< ~140K** → 3–4× threading clears 23–42K → **build it**.
- **> ~250K** → even 4× misses → **don't build; keep the shipped HVX op.**

This is the one number M5 never isolated (it only ever measured the glue *inside* QNN, tax included).

## References
- target baseline measurement: `example/gdn_native/solve_op/standalone/gdn_shape.sh` (CS=256 H=32).
- scheduling limit (why not QNN): `docs/qnn_htp_scheduling_and_custom_op_limits.md`.
- block-recursive analysis + M1–M5 device results + the merge choreography: `gdn_solve_prefill_128_256.md`.
- owned HMX kernel (op #6): `example/gdn_native/solve_br_op/` + `project_v73deep_BREAKTHROUGH_2026-04-28`.
