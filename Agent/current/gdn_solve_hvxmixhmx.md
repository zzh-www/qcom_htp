# GDNSolveHVXMixHMX — HVX-feed + HMX-matmul producer/consumer pipeline (matmul-portion squeeze)

## 命名定义（GDN 求逆三种实现）

全仓库统一用这三个名指代 GDN 三角求逆的三种实现，**勿用其他叫法**（如 "HVX-merge"、"HMX-feed"、
"HMX 加速版"、"HVX 直算版"、"pure HMX" 等旧叙述名一律弃用）。注意：这是**叙述性名词**的统一；
代码标识符（宏 `GDN_BR_HVX_MERGE`、函数 `gdn_merge_packed` / `gdn_merge_hvx` 等）保持原样不变。

| 标准名 | 定义 | 代码路径 | 性能 |
|---|---|---|---|
| **GDNSolveHVX** | 纯 HVX 实现（基线，已出货）。64³ 矩阵乘用 HVX int16 vrmpy；对角块求逆（diag forward-subst）用 HVX。 | 定义 `GDN_BR_HVX_MERGE` 时走的 `gdn_merge_hvx` | ~135K cyc/head（4-thread per-head） |
| **GDNSolveHVXMixHMX** | HVX 喂数 + HMX 算 64³ 矩阵乘；对角块求逆仍用 HVX（与基线共享）。 | 默认（不定义 `GDN_BR_HVX_MERGE`）走的 `gdn_merge_packed`；baremetal 里 `GDNBM_FEED_PIPE`/`GDNBM_FEED_4P` 的生产者-消费者流水线（4 个 HVX 生产者打包喂 1 个 HMX 消费者） | 矩阵乘子环节实测 **578 cyc/matmul = 2.77×** 于基线 vrmpy（1601，单次 HMX run）。端到端**实测**（忠实 multi-pass，`-DGDNBM_FEED_MULTIPASS`）：真实现状（scalar-bias 3-pass）**1.29×**；把 multi-pass 的 bias-pack 向量化后 **2.06×**；再 +#1c（省 PASS2）**2.35×**（均 oc 安全）。优化后 diag 成新 floor。详见文末实测节 |
| **GDNSolveHMX** | 全程 HMX（连三角求逆本身也用 HMX，如 divide-conquer）。 | 已探索并否决 | 多算 28–60× 矩阵乘，实测慢 4–6×，三者中最差 |

本文档主要讲 **GDNSolveHVXMixHMX**。

**Goal:** speed up the GDN triangular-inverse *merge-matmul* portion by running the 64³ matmuls on the
(idle) HMX unit, fed by HVX producers — instead of the GDNSolveHVX baseline's HVX `vrmpy` matmul. Bare-metal only
(QNN custom ops get NO HVX∥HMX overlap).

**Result (real v75 `ssh oneplus`, C=256-shaped 64³ matmuls, chain8-style steady, DUMMY data):**
matmul-portion throughput **1208 → ~578 cyc/matmul = 2.09×**, and **2.77× vs the shipped-best vrmpy
4-thread (1601 cyc/matmul throughput)**. (Was ~635; the depack squeeze + single-barrier below took it to ~578.)
This is the practical optimum on 4 HVX units — feed-bound + average-VTCM-bandwidth-bound (NOT the old 388;
the 4P pure-HMX consumer floor is ~418, see the scaling-curve section).

## The pipeline

`P HVX producer threads` pack each 64³ matmul's operands into a VTCM ring (crouton act + k-major wt +
bias), a `main-thread HMX consumer` drains the ring (the `our_v73deep_kernel` mxmem). Static-stripe job
distribution, CAS-free volatile slot flags (CAS `__sync` is unreliable on this target — use volatile +
static striping). Reproduce: `example/gdn_native/baremetal`, flags below.

## Clean ceilings (single-thread, wall-based — the only trustworthy timing)

| quantity | cyc/matmul | note |
|---|---|---|
| HMX kernel (continuous, descriptors set once) | **215** | mxmem floor |
| **consumer drain ceiling** (zero+kernel+depack, hot loop) | **388** | the pipeline's theoretical floor |
| producer 1-head pack floor (eff+act+wt+bias) | **3139** | the wall — pack is 8× the HMX consumer |

⚠️ **Per-thread `pcyc()` sub-timings are GARBAGE** — `C15:14` PCYCLE is per-hardware-thread and QuRT
migrates SW threads. Only total wall + single-thread isolated loops are reliable.

## Optimization journey (each step measured)

| step | cyc/mm | lever |
|---|---|---|
| orig pipeline (3 producers, 1-head, scalar eff+bias) | 1208 | |
| **+ vectorized eff+bias** | 833 (1.45×) | `gdn_pack_bias`'s 128 SCALAR int32 VTCM writes were a hidden 35% (eff+bias 1093→215). Scalar VTCM writes = the documented pathology. Replaced with HVX col-sum + **4 vector stores** (`fp_pack_effbias`). Also un-stuck 2-head. |
| + 2-head producers (3P) | 713 | act/wt pack ILP now visible (was masked by the scalar eff+bias) |
| **+ 4th producer (3-stage)** + 2-head | **~635 (1.9×)** | consumer made PURE-HMX → frees its HVX unit → a 4th HVX producer becomes legal |
| **+ fused one-pass depack** (`fp_depack`) | ~612 | kill the `surf_sub` VTCM round-trip: fold base-subtract into the de-crouton loop (read `surf` once, sub inline). Isolated depack 512→~420 (~1.2×); bit-exact (dep_mism=0). |
| **+ 2-head interleaved depack** (`fp_depack2`) | ~592 | interleave the pair's two depack streams (ILP, mirrors `fp_pack_act2`); distinct out buffers avoid store aliasing |
| + drop producer out-zero | ~592 (neutral) | verified `our_v73deep_kernel` fully OVERWRITES the out surface (ovr_mism=0) → pre-zero unnecessary. No wall gain (was overlapped) but removes 32 vec-stores/matmul; kept as verified-safe simplification. |
| **+ 2-head interleaved eff+bias** (`fp_pack_effbias2`) | **~585 (2.06×)** | interleave the two column-sum chains; consumer spin 184→168/matmul (producer genuinely faster); bit-exact (eff_mism=0) |

## Three findings that unlocked it (all device-verified)

1. **Diagnosis inversion:** consumer ceiling **388 ≪ pipeline 1196** ⇒ the pipeline is **producer-feed-bound**,
   consumer idle ~68% — NOT consumer-bound. (Earlier "consumer-bound" was wrong; the clean ceiling
   measurement settled it.) And 2-head not helping at first = the feed wasn't compute-bound *because*
   the scalar eff+bias dominated.
2. **The HMX kernel needs only `hmx_lock`, NOT `qurt_hvx_lock`** (verified: pure-HMX consumer → `open rc=0`,
   no fault). So the consumer can be HVX-lock-free ⇒ it doesn't consume an HVX unit ⇒ 4 HVX producers
   + 1 pure-HMX consumer = legal (vs the 3+1 cap when the consumer locks HVX for zero/depack).
3. **HVX-unit rule (root cause of the thread cap):** `#HVX-LOCKED threads ≤ #HVX units (=4)`, because
   `qurt_hvx_lock` blocks when no unit is free (`qurt_hvx.h:145-148`). Oversubscribing (5 HVX-locking
   threads) hangs and **SSRs the cDSP** (`gdnbm_open` → `rc=0x80000406` until reboot; non-root can't
   force-clear). Landed in `.codex/skills/htp-hardware-scheduling` + `docs/htp_hardware_scheduling_flow.md`.

## 4-producer 3-stage state machine (CAS-free)

Slot states `g_fp_ready[k]`: `0=free` / `1=packed` / `2=hmx-done`. 4 HVX producers (static stripe) each:
spin until target slot ∉ {1}; if state==2 **depack** the previous occupant (cold cross-thread read of the
HMX output) and zero; pack (2-head pair); set state 1. Main pure-HMX consumer (in job order): spin
state==1; run kernel; set state 2 (a producer reuses+depacks it later).

## Where it's still bound + why we stopped

Producer-bound at **~585** (vs the 388 HMX floor); consumer still spins ~168/585 (~29% idle). Reaching 388
would need ~8 producers (impossible — 4 HVX units) or much cheaper packing. **This is the practical limit
of the 4-HVX-unit silicon for the matmul portion** — the depack squeeze (fused + 2-head, ~635→~585)
trimmed the producer's non-pack cost; what remains is the act/wt crouton-pack + cross-thread depack across
4 producers, which can't be cut further without more HVX units.

**Two levers that did NOT pay off (measured, then reverted/neutral):**
- *Pack-before-depack reorder* (to hide the cross-thread `.out` read latency under pack compute): no-op,
  587/601/611 vs 592/598/601 — within noise. The cross-thread VTCM read isn't an exposed gap; reverted.
  (Note: l2fetch is the WRONG tool here — `.out` is VTCM-resident, not L2/DDR-cached.)
- *Drop the producer out-zero*: wall-neutral (was already overlapped), but kept because it's a verified-safe
  simplification removing real work.

### Scaling curve (P=2/3/4 sweep) — corrected bottleneck model (2026-06-04)

Swept producer count P in 4P mode (`./gdnbm <P> ...`); each row min-of-2, real v75:

| P (HVX producers) | cyc/matmul | consumer spin | consumer busy (=cyc−spin) | per-producer cost (=cyc×P) |
|---|---|---|---|---|
| 2 | 1034 | 667 (64%) | 367 | 2068 |
| 3 | 721  | 334 (46%) | 387 | 2163 |
| 4 | **585** | 167 (29%) | 418 | 2340 |
| 5 (extrapolated) | ~495 | — | ~432 | ~2476 |
| 6 (extrapolated) | ~435 | — | ~446 | ~2612 |

**Three reads of this curve:**
1. **Producer-feed-bound, confirmed independently:** consumer spin falls 64%→29% as P grows 2→4; adding
   producers directly buys throughput. Scaling P=2→4 is 1034→585 = **1.77× (88% of the ideal 2.0×)** — mildly
   sublinear, not a wall yet.
2. **Light VTCM/L2 bandwidth contention (the sublinearity):** *both* the per-producer cost (2068→2340) and
   the consumer's own busy time (367→418) rise with P — each added producer slows every thread ~13% / ~14%.
   This is the bandwidth fingerprint the route doc predicted; the bound is VTCM/L2 *bandwidth*, not capacity.
3. **The real floor is the PURE-HMX consumer at ~418, NOT 388.** The 4P consumer does only kernel(215) +
   descriptor-build + 2× `__sync_synchronize` (~203 fixed/iter) — it can never go below ~418 even with
   infinite producers. The old "388 ceiling" was the *non-parallel* consumer's zero+kernel+depack hot loop;
   it does NOT apply to the 4P pure-HMX consumer. Extrapolating the two lines, the producer-feed rate would
   only cross the consumer floor at **P≈6** (~440 cyc/matmul) — so even a 6×128B SKU lands near ~440, not 388.
   On this 4-HVX-unit part, ~585 is the practical optimum.

**Next lever IF the consumer ever becomes the bound (P≥5 hardware):** cut the consumer's ~203 fixed/iter —
hoist the `od`/`ad` descriptor build out (only `outtab`/`acttab` change per slot; pre-store them in the slot)
and replace the 2 full barriers with one-way volatile acquire. Pointless today (consumer has 167 slack).

### Can we re-schedule the 4 HVX to "use bandwidth better"? — NO (tested 2026-06-04)

The ~13% contention is **conserved average-bandwidth, NOT burst collision** — so reordering the work can
only *move* the contention between threads, never reduce the total. Two experiments settle it:
- **Phase-stagger** (even producers depack→pack, odd pack→depack, so ~half read / ~half write VTCM at any
  instant): no-op on throughput (575→578). It DID cut consumer spin 168→158 (producers fed sooner) but the
  consumer's own busy time ROSE 412→427 by the same amount — the contention just shifted producer→consumer.
  This is the signature of an average-bandwidth bound, not a peak-collision one.
- *(small real win, kept)* one acquire barrier after the spin instead of one per depack: ~585 → **~578**.

**Implication:** scheduling/relayout (phase-stagger, bank-aware slot placement, read/write specialization)
cannot beat ~578 here — the VTCM traffic is already the HMX-mandated minimum *for one HMX run*. The ONLY way
down is to **reduce VTCM traffic itself** (algorithm-level). See the next section for the real lever.

## Algorithm-level traffic-reduction map (2026-06-04) — the microbench HIDES a 2-3× factor

**Headline finding:** the microbench measures ONE HMX run per "matmul", but the real `gdn_merge_packed`
(GdnSolveBROp.cpp:854) runs the HMX kernel **2-3× per logical 64³ matmul** — a dynamic-quant gain search:
- **PASS 1** (`gdn_hmx_run_only`, loose gain g1) → `gdn_surf_maxabs(out)` → **result thrown away**, only max|P| kept
- **PASS 2** (refine gain gr, almost always taken since code1>0) → maxabs → **thrown away**
- **PASS 3** (tight gain g2 = 127/maxP) → `gdn_depack_out_fast` → the ONLY useful output

Why: int8 HMX output must be scaled to fill [-127,127], which needs max|P|, which needs P itself → chicken-
and-egg, "solved" by running the matmul 2 extra times just to measure its own output magnitude. oc is
hypersensitive to this scale (comment: dist-1 blocks oc 0.73 vs 0.01 if scale is ~20% off), so the passes
can't just be dropped.

**VTCM traffic per LOGICAL matmul (corrected):** ~3 runs × (read act+wt+bias 8.5K + write out 4K) + 2×
maxabs-read out (8K) + depack-read out (4K) ≈ **~50K**, of which **PASS1+2 ≈ 33K (~66%) is pure
scale-probing that is discarded.** So the real GDNSolveHVXMixHMX matmul-portion is ~2.5-3× the microbench's 578 — and
the #1 traffic lever is killing the probe passes, NOT the depack round-trip or scheduling.

**Optimization points, ranked by traffic saved:**
1. **Kill / cheapen the multi-pass gain search (≈2-3× — by far the biggest).** Options, each oc-gated:
   (a) *predict* max|P| from input norms (‖A_i,:‖·‖T_:,j‖ bound) → 0 probe runs, but oc-risky (scale must be
   ~1% accurate); (b) replace the PASS-1/2 *full HMX runs* with a cheap HVX sub-sampled dot-product estimate
   of max|P| (keeps accuracy, drops 2 HMX runs + 2 out-writes); (c) tighten PASS-1's initial gain (norm-based)
   so PASS 2 is unnecessary → 3 passes → 2 (saves 1/3). **Must validate oc vs golden** (scale-sensitive).
2. **HMX-accumulate the inner-product** `S_ij=Σ_k A_ik·T_kj` (10 matmuls/head): accumulate K terms into one
   `out` → depack once per block, not per term. Bounded by (i) `extra_param` accumulate-mode availability
   (today ep={1,0}=overwrite, verified) and (ii) per-term scale alignment; NB=4 is shallow so ~small.
3. **crouton-out → k-major-wt direct transcode** (out codes are reused as the next stage's wt): skip the
   row-major intermediate. Tangled — codes also feed int32 widen + requant. Small-medium.
4. **Already correct, keep:** operand cache (act A_ik / wt T_kj packed once, reused across i/j — amortizes the
   pack writes); intermediate scratch in DDR not VTCM (keeps it OFF the contended VTCM bus).

**Note:** the microbench's single-run 578 is the right number for the *steady GDNSolveHVXMixHMX throughput of one run*;
the real-solve speedup vs GDNSolveHVX (151K/head) must account for the 2-3× multi-pass factor — i.e.
GDNSolveHVXMixHMX only wins once lever #1 lands. This reframes "is GDNSolveHVXMixHMX worth it": the squeeze done so far
is necessary but not sufficient; the multi-pass gain search is the gate.

### SIM VERDICT on lever #1 (scripts/gdn_solve_maxp_probe.py, real golden p29_L00, 2026-06-05)

Replayed the real block-recursive forward-subst, measured actual max|P| vs norm predictors at all 512 merge
matmuls. Output-quant relerr (lower=better): **PASS3-exact (true max|P|, today) p50~0.022 / p90~0.037**;
**pure norm-prediction as the final scale p50~0.078 / p90~0.168** (K=1 Holder bound, never saturates but
3-4x worse; block-recursive propagation would push oc up — too lossy to BE the output scale).

**BUT lever #1c is SAFE and lands ~33%:** the Holder bound's output FILL is ~37 (>=5-bit), enough for PASS-1's
integer maxabs to be accurate in ONE shot. So replace PASS-1's LOOSE constant (fill~1 -> needs PASS-2) with
the Holder norm-predicted gain -> PASS-1 measures max|P| accurately -> **drop PASS-2**; PASS-3 still uses the
MEASURED max|P|, so **oc is unchanged** (relerr stays 0.022, not the predicted 0.078). Saves 1 HMX run +
1 maxabs/matmul ~= 33% of the path's VTCM traffic. The Holder norm (act row-1-norm max x wt max) is ~free:
`fp_pack_effbias` already computes wt column-sums; act row-sums piggyback on the act-pack.

**STRATEGIC fact uncovered:** the shipped/baremetal GDNSolveHVX path (`gdn_merge_hvx`, int16) has NO multi-pass
at all — int16 output doesn't need the int8 gain search. Multi-pass is a tax UNIQUE to GDNSolveHVXMixHMX (int8-out).
So even with #1c (3->2 passes) GDNSolveHVXMixHMX still carries 1 probe pass GDNSolveHVX doesn't. The
GDNSolveHVXMixHMX-vs-GDNSolveHVX decision weighs: HMX raw matmul is faster, but pays a ~2x (post-#1c) probe-traffic tax GDNSolveHVX
avoids. #1c lives in `gdn_merge_packed` (the QNN `solve_br_op` DEFAULT path; baremetal forces GDNSolveHVX);
oc-validate via `solve_op/standalone` net-run vs golden.

## Bigger-picture context (does the solve even need to be this fast?)

- **Whole C=256 GDN chunk ≈ 2 ms is hopeful** — IF the chunk is fused into one bare-metal kernel and the
  **HVX-bound solve overlaps the HMX-bound big matmuls (A/U/W/P/v_new/oc/S_out) across heads**. Then
  wall ≈ HMX work (~1.9 ms), and the solve's HVX work HIDES under it. So the solve may not need to be
  squeezed at all — but this 2.52× is banked. See `reference_htp_hardware_scheduling_flow`.
- Alternatives explored and REFUTED for the *inverse itself*: Taylor/Neumann/squaring/Newton iterative
  matmul (‖A‖≫1 explodes), GDNSolveHMX (divide-conquer; 28–60× more matmuls → more glue, HMX never
  saturates, ~4–6× slower). Hybrid (HVX forward-subst diagonals + HMX/vrmpy merges) is optimal. See
  `gdn_solve_taylor_probe.py` / `gdn_solve_divconq_probe.py`.
- Hardware what-ifs: more HVX units → sublinear (~1.4× at 8, bandwidth + single-HMX walls); a 6×128B SKU
  exists. **VTCM capacity adds ~0** at C=256 (8MB ≫ working set; the bound is VTCM/L2 *bandwidth*).

## Reproduce

```bash
cd example/gdn_native/baremetal
# ceilings + producer floor + vectorized-eff+bias gain:
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_HMX_BENCH" bash build.sh   # gdnbm 1 ... -> stats[2]=HMX kern, [3]/[4]=pack OLD/NEW
# the pipeline (matmul-portion throughput), pick one:
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE"                 bash build.sh   # 3P 1-head  -> ./gdnbm 3 ...
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_2H" bash build.sh   # 3P 2-head (713)
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P" bash build.sh   # 4P 3-stage 2-head (~635)
#   deploy build/{libgdnbm_skel.so,gdnbm} to $HOME/gdnbm_run; pkill -9 gdnbm BEFORE each run; cap threads ≤4
#   ./gdnbm <P> A_u16_h32.raw /dev/null 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05  -> stats[0]=cyc/matmul
```

## Depack-squeeze DONE (2026-06-04) — plateaued ~635 → ~585

The matmul-portion squeeze the user requested (2026-06-04) is complete; landed in `gdnbm_imp.cpp`:
- **`fp_depack`** (one-pass fused depack) + **`fp_depack2`** (2-head interleaved) — replace the producer's
  `gdn_depack_out_fast` call. `GdnSolveBROp.cpp:gdn_depack_out_fast` is UNCHANGED (still the bit-exact path
  for the real solve op).
- **`fp_pack_effbias2`** (2-head interleaved eff+bias) — replaces the two serial `fp_pack_effbias` in
  `fp_pack_slot2`.
- Producer out-surface **zero removed** (verified kernel overwrites).
- **HMX_BENCH stats fixed + extended** (stats[3] double-write gone): stats[5]=consumer ceiling,
  stats[6]=old two-pass depack, stats[7]=new fused depack, stats[8]=depack-mismatch (==0),
  stats[9]=overwrite+effbias2-mismatch (==0). All three correctness gates PASS on device.

**Verdict:** plateaued — 1c (reorder) was a no-op, lever 2 (no-zero) wall-neutral, lever 3 (~2%). Going
below ~585 needs >4 HVX producers (impossible) or cheaper crouton pack. The remaining stretch lever (act/wt
crouton-pack lane waste, 32/128 masked) is near the floor and not worth it.

## DEFERRED (later sessions, not now)

- **Integration into the real solve_br_op + oc-vs-golden:** numbers are DUMMY-data microbench. The new
  primitives are now bit-verified in isolation (HMX_BENCH: `fp_depack`==`gdn_depack_out_fast`,
  `fp_pack_effbias2`==2×`fp_pack_effbias`, kernel-overwrite all 0-mismatch), but the END-TO-END 3-stage
  pipeline (CAS-free slot machine + cross-thread depack) is still only wired in the throughput microbench —
  it must be integrated into the real block-recursive `solve_br_op` and oc re-checked vs golden.
- **Whole-GDN-chunk fused kernel with HVX∥HMX cross-head overlap** (~2 ms path) where the solve's HVX
  hides under the HMX big matmuls — likely the higher-value direction once the squeeze plateaus.

## End-to-end GDNSolveHVXMixHMX-vs-GDNSolveHVX comparison (2026-06-05) — before committing to #1c

User asked to compare the two paths end-to-end before investing in #1c. Measured the shipped GDNSolveHVX path
(baremetal default, `-DGDN_BR_PROBE_CYCLES`, 1-thread, one head, real A_u16_h32):

  GDNSolveHVX per-head (1-thread): mm=210949 (39%) diag=103369 (19%) fold=100719 (18%) quant=76336 (14%)
  requant=29236 acc=12300 zero=8193  => TOTAL 545829 cyc/head  (÷4 head-parallel ≈ 136K, matches the 151K
  4-thread number with bandwidth contention).

Key structure: matmul (mm) is 39%, but **diag forward-subst (103K, shared by BOTH paths) + operand prep
(fold+quant = 177K) together dwarf it**. GDNSolveHVXMixHMX replaces mm + the prep with HVX-feed + HMX-matmul; it CANNOT
avoid diag.

Rough per-head (4-HVX-unit budget), GDNSolveHVXMixHMX pipeline at 578 cyc/run (single-run microbench):
  - matmul+pack only: 512 mm × 578 / 32 head = 9.3K/head (1 pass) -> ×3 multi-pass = 27.7K (×2 w/ #1c = 18.5K)
  - vs GDNSolveHVX mm+fold+quant ≈ 97K/head (4-thread) -> HMX matmul is ~2-3.5× cheaper IF the microbench were faithful.

**But the microbench UNDER-models the real GDNSolveHVXMixHMX path in 3 ways, all favoring GDNSolveHVX:**
  1. **multi-pass (2-3×)** — the 578 is ONE run; real is 2-3 (lever #1c cuts to 2).
  2. **fold+quant NOT in the pipeline** — producers pack DUMMY pre-quantized data; the real producer must
     also fold u16->int + quantize (~44K/head of work) before packing.
  3. **multi-pass breaks the 4P architecture** — PASS-1/2 need maxabs (HVX) + a serial dependency (PASS-3
     waits on the gain), so the consumer can't stay PURE-HMX; it must take an HVX unit -> back to 3 producers.

**Verdict: GDNSolveHVXMixHMX's win is real in the matmul kernel itself (215 vs vrmpy ~13K/mm) but is heavily diluted by
the shared diag floor (~26K/head), operand prep, and the multi-pass tax. Optimistic end-to-end ~1.3-2× over
GDNSolveHVX's 135K; pessimistic ~parity.** The decisive unknown is whether multi-pass + fold/quant can hide in
the pipeline without serializing the HMX. #1c (norm-predicted PASS-1 -> drop PASS-2) attacks the traffic but
not the serial/architecture risk.

### FAITHFUL multi-pass MEASURED (2026-06-05, `-DGDNBM_FEED_MULTIPASS`, P=3, real v75)

Built a faithful consumer (3 HMX runs + 2 maxabs + serial gain dep, bias re-packed each pass), depack still
offloaded to a producer. Consumer LOCKS HVX (maxabs) so P=3 producers (3+1 = 4 HVX units). All consumer-bound
(spin≈58 — producers idle, so fold/quant CAN hide in their slack). cyc/matmul (min of 2):

| multi-pass config | cyc/matmul | matmul/head (512mm/32h) | end-to-end/head¹ | vs GDNSolveHVX (135K) |
|---|---|---|---|---|
| single-run ideal (old microbench) | 578 | 9.3K | — | (didn't model multi-pass) |
| **scalar-bias 3-pass = real gdn_merge_packed TODAY** | **4190** | 67K | ~105K | **1.29×** |
| **vectorized-bias 3-pass** | **1720** | 27.5K | ~65K | **2.06×** |
| **vectorized-bias 2-pass (#1c)** | **1218** | 19.5K | ~57K | **2.35×** |

¹ end-to-end = matmul + diag (26K, shared HVX floor) + requant/acc (12K).

**DECISIVE FINDING: the real gdn_merge_packed TODAY is only 1.29× — and the bottleneck is NOT #1c, it's the
scalar `gdn_pack_bias` (128 scalar VTCM writes, the documented pathology) re-run every pass = ~60% of the
multi-pass cost.** Swapping in the vectorized `fp_pack_effbias` (already built+verified for the pipeline)
alone jumps it to **2.06×**; adding #1c (drop PASS-2) reaches **2.35×**. Both are oc-SAFE (vectorized bias is
numerically equivalent; #1c keeps the measured max|P| so oc is unchanged).

### LEVER #1 (vectorized bias) LANDED in the real op + verified (2026-06-05)

Replaced the scalar `gdn_pack_bias` (128 scalar VTCM writes) in `GdnSolveBROp.cpp` with 4 HVX vector stores
(bias tile = [ctrl×32 | eff[0:32]+rdelta | ctrl×32 | eff[32:64]+rdelta]). Verified via `solve_br_op/standalone/gdn_br.sh`
(C=128, H=16, real device): **T relerr BIT-IDENTICAL** (8.538e-3 mean / 1.646e-2 max, off-diag 6.424e-2 —
same to all digits as the scalar baseline → numerically equivalent, oc unchanged). aggregate_cyc
1,060,341 → 980,516 (−7.5%; C=128 is diag-dominated with only 2 matmuls/head — C=256's 16 matmuls/head get
the full ~2× per the MULTIPASS microbench). Only affects the HMX path (`gdn_pack_bias` is dead in `#if 0` for
everything else; GDNSolveHVX's `gdn_merge_hvx` untouched). Next: lever #2 (#1c, drop PASS-2).

**So the ranked GDNSolveHVXMixHMX levers are now: (1) vectorize the multi-pass bias-pack (1.29→2.06×, biggest,
~free, numerically equivalent); (2) #1c drop PASS-2 (2.06→2.35×, oc-safe). After both, diag (26K, ~45% of the
57K) becomes the new floor — shared with GDNSolveHVX, so the next frontier is the diagonal forward-subst.**
fold+quant hides in the consumer-bound producers' slack, so it doesn't add. These are device-measured
throughput on DUMMY data; the real integration into `gdn_merge_packed` + oc-vs-golden is the remaining step.
