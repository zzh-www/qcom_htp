# GDNSolveHVXMixHMX — HVX-feed + HMX-matmul producer/consumer pipeline (matmul-portion squeeze)

## ⛔ 工作方式（权威，置顶，勿违反）

**只跑微基准（microbench）。不要再跑完整 GDNSolveHVXMixHMX 全 solve。**

- 当前**只有微基准是跑通且可信的**（matmul-portion 4P、HMX_BENCH ceilings、P-sweep）。
- 完整 solve 路径 `-DGDNBM_HMX_MERGE_PATH` 在 **open 阶段就 SSR**（`gdnbm_open rc=0x80000406`，
  2026-06-05 复现确认；HVX 基线同设备 rc=0x0 正常 → 是该 build 自身问题，不是 DSP 挂了）。它的
  "434K/head" 数字**无法在当前代码复现**，已作废，别再引用、别再尝试跑它。
- **下一步不是修这个全 solve，而是据微基准结论彻底重写整个 solve**（2-head pack / fused depack /
  2-head eff+bias / #1c / 正确多线程一次性进新实现）。在重写之前，唯一可信的性能依据是微基准。
- 因此所有"GDNSolveHVXMixHMX vs GDNSolveHVX 全 solve 快/慢"的结论**暂缺**，等重写完成后用同 harness、
  同线程数才能下；现在不要据残缺/SSR 的全 solve 数字下任何结论。

## 命名定义（GDN 求逆三种实现）

全仓库统一用这三个名指代 GDN 三角求逆的三种实现，**勿用其他叫法**（如 "HVX-merge"、"HMX-feed"、
"HMX 加速版"、"HVX 直算版"、"pure HMX" 等旧叙述名一律弃用）。注意：这是**叙述性名词**的统一；
代码标识符（宏 `GDN_BR_HVX_MERGE`、函数 `gdn_merge_packed` / `gdn_merge_hvx` 等）保持原样不变。

| 标准名 | 定义 | 代码路径 | 性能 |
|---|---|---|---|
| **GDNSolveHVX** | 纯 HVX 实现（基线，已出货）。64³ 矩阵乘用 HVX int16 vrmpy；对角块求逆（diag forward-subst）用 HVX。**正确的多线程方式：4 个 HVX worker 按 head 并行（#HVX-locked ≤ 4 单元）。** | 定义 `GDN_BR_HVX_MERGE` 时走的 `gdn_merge_hvx` | baremetal 实测 **4-thread 157K cyc/head**（1-thread 414K）。这是当前已验证的基准。 |
| **GDNSolveHVXMixHMX** | HVX 喂数 + HMX 算 64³ 矩阵乘；对角块求逆仍用 HVX。**HMX=1 单元 process-serial，绝不 thread HMX**（skill `htp-hardware-scheduling`）；正确多线程 = HMX consumer 在 main + HVX 并行。 | 默认（不定义 `GDN_BR_HVX_MERGE`）走的 `gdn_merge_packed`；baremetal `GDNBM_FEED_PIPE`/`GDNBM_FEED_4P` 微基准流水线 | **已验证**：matmul 子环节 578 cyc/matmul（2.77× vs vrmpy 微基准）；向量化 bias 已进完整 solve（bit-identical）；multi-pass 真实成本 4190→1218（向量化bias+#1c，oc 安全）。**未做**：把微基准 pack/depack 优化集成进完整 solve + 正确多线程实现。**完整-solve 多线程性能 = 未知，尚无 vs 基线的公平结论**（见文末 "Full-solve status"）。 |
| **GDNSolveHMX** | 全程 HMX（连三角求逆本身也用 HMX，如 divide-conquer）。 | 已探索并否决 | 多算 28–60× 矩阵乘，实测慢 4–6×，三者中最差 |

本文档主要讲 **GDNSolveHVXMixHMX**。

## VERIFY 复测结果（2026-06-05，real v75 `ssh oneplus`，本次 session 全部重跑）

微基准全部复现，与文档记录一致或更好；全 solve 路径 SSR（见置顶禁令）。

| 量 | 文档记录 | 本次实测 | 判定 |
|---|---|---|---|
| GDNSolveHVX 4-thread | 157K | **128–146K**（127541/128317/131675/140586/146189） | ✅ 复现，实测**更快**，157K 偏保守/过时 |
| GDNSolveHVX 1-thread | 414K | **406K**（405895） | ✅ 复现，略快 |
| matmul-portion 4P (cyc/matmul) | ~578/585 | **587/588/603** | ✅ 复现（噪声内） |
| 4P consumer spin /matmul | ~167 | **166/181** | ✅ 复现 |
| P-sweep P2/P3/P4 | 1034/721/585 | **1030–1043 / 713–730 / 588–612** | ✅ 复现，缩放曲线吻合 |
| P-sweep spin (64%→46%→29%) | 667/334/167 | **666–703 / 344–350 / 172–183** | ✅ producer-feed-bound 结论成立 |
| HMX kernel floor (stats[2]) | 215 | **214** | ✅ 复现 |
| bit-exact 门 dep_mism / ovr_mism | 0 / 0 | **0 / 0** | ✅ 复现 |
| consumer ceiling (HMX_BENCH stats[5]) | 388 | **749** | ⚠️ 对不上，但文档自己已声明 388 被 4P 实际 floor ~418/585 取代，属过时次要 ceiling，不影响结论 |
| producer pack floor OLD/NEW (stats[3]/[4]) | ~3139 | **2561 / 2333** | ℹ️ 比记录低；向量化 eff+bias 省 ~228 cyc（isolated 单 head） |
| 全 solve nthreads=1 `-DGDNBM_HMX_MERGE_PATH` | 434K | **SSR `rc=0x80000406`（open 即挂，无法复现）** | ❌ 作废，见置顶禁令 |

**结论：微基准侧的性能优化分析（producer-feed-bound、HMX kernel 215、4P≈585、缩放曲线、bit-exact）全部
站得住；全 solve 的任何数字（含 434K）不可信。**

## NEXT SESSION — START HERE (集成状态 + 计划, 2026-06-05)

**真实基线（已验证，apples-to-apples 用它）**：GDNSolveHVX baremetal **4-thread = 128–146K cyc/head**
（本次实测；旧记 157K 偏保守。1-thread 406K）。这是当前最优。
**禁止跑全 solve（SSR + 即将重写）——只跑微基准，见置顶禁令。**

**GDNSolveHVXMixHMX 在基准代码 (`gdn_merge_packed`) 的集成状态：**
- ✅ **已集成**：向量化 `gdn_pack_bias`（128 scalar→4 vector，bit-identical，commit `2cc2884`）；
  **#1c** 省 PASS2（范数预估 gain，`gdn_effective` 顺带算 colabsmax，oc 安全，**工作区未 commit**）。
- ❌ **未集成**（只在 baremetal 微基准 `gdnbm_imp.cpp`）：2-head pack `fp_pack_act2/wt2`、
  fused depack `fp_depack`、2-head eff+bias `fp_pack_effbias2`。完整 solve 的
  `gdn_get_act_A`/`gdn_get_wt_T` 仍用旧 1-head `gdn_pack_act_crouton8`/`gdn_pack_w8_kmajor` +
  `gdn_depack_out_fast`。
- ❌ **未做**：diag 优化；GDNSolveHVXMixHMX 正确多线程实现。

**完整 solve 现状（实测，别再用错误估算）**：GDNSolveHVXMixHMX nthreads=1 C=256 = **434K/head**（旧 1-head
pack）；per-stage diag 132K / fold 122K（虚高，旧 pack）/ merge 61K / quant 53K / other 59K。
⚠️ 之前文档里的 "65K 三块构成 / 2.06×→2.35× / 输 2.77× / 不能并发" **都是错误估算或不公平比较，已废弃**
（原因见文末 "Full-solve status — CORRECTED"）。

**计划（按优先级）：**
1. **集成微基准 pack/depack 优化进 `gdn_merge_packed`/getter**（2-head `fp_pack_act2/wt2` + fused
   `fp_depack` + `fp_pack_effbias2`）→ 完整 solve 的 fold(122K) 应大降。重测 nthreads=1，`gdn_br.sh` 验 oc 不退化。
2. **commit #1c**（已在工作区，oc 安全），C=256 重测确认收益。
3. **正确多线程 GDNSolveHVXMixHMX** —— 按 skill `htp-hardware-scheduling`：**HMX=1 单元 process-serial，
   绝不 thread HMX**；HMX consumer 在 MAIN，HVX work 多线程（head 并行）。测 4-thread。
4. **apples-to-apples vs GDNSolveHVX**（同 baremetal harness、同线程数、C=256）才能下"快/慢"结论。
5. （独立）**diag 优化分析** —— diag 两路线共享，优化它绝对提速两者（相对倍数几乎不变，但 prefill 墙钟有用）。

**禁止重犯**（见 memory `feedback_read_scheduling_skill_before_hmx_hvx`）：读 skill 先；never thread HMX（→SSR
`rc=0x80000406` 需 reboot）；公平比较（同线程数）前不下结论；`pkill -9 gdnbm` between device runs。

**Reproduce（baremetal，`ssh oneplus` via `scripts/dssh.sh`）：**
```bash
cd example/gdn_native/baremetal
# GDNSolveHVX 基线（4-thread 157K）：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDN_BR_PROBE_CYCLES" bash build.sh                       # ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 32768 32768 2.77e-05 6.10e-05
# GDNSolveHVXMixHMX 完整 solve（nthreads=1 SSR-safe；多线程是 TODO，勿乱开多 HMX worker）：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_HMX_MERGE_PATH -DGDN_BR_PROBE_CYCLES" bash build.sh # ./gdnbm 1 ...
# matmul-portion 微基准（578）：
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_FEED_PIPE -DGDNBM_FEED_4P" bash build.sh            # ./gdnbm 4 ...
```
PROBE stats（HMX_MERGE_PATH/HVX 两路通用）：[3]=diag [4]=mergeRun [5]=mergeGlue [6]=fold [7]=quant [8]=other [9]=wall/head。

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

## DEFERRED / TODO (the real next steps)

- **Integrate the matmul-portion microbench wins into the real `gdn_merge_packed` + getters** (2-head
  `fp_pack_act2/wt2`, fused `fp_depack`, `fp_pack_effbias2`, #1c) — they are validated in isolation but the
  full-solve path still uses the OLD 1-head pack + `gdn_depack_out_fast`. Re-check oc vs golden after.
- **Implement GDNSolveHVXMixHMX multithreading per the skill** then measure (see status below).

## Full-solve status (2026-06-05) — CORRECTED (prior "loses 2.77× / can't parallelize" was WRONG)

⚠️ An earlier version of this doc concluded "GDNSolveHVXMixHMX loses to GDNSolveHVX by 2.77× and can't
parallelize." **That was wrong — removed.** The errors:
1. **Unfair comparison** — GDNSolveHVXMixHMX nthreads=1 (434K) vs GDNSolveHVX nthreads=4 (157K). Not apples
   to apples. GDNSolveHVXMixHMX's 4-thread number was never correctly measured.
2. **Un-integrated pack** — the full-solve `gdn_merge_packed` still uses the OLD 1-head pack, so its
   fold/pack is inflated (fold 122K). The 2-head pack / fused-depack microbench wins are NOT yet integrated.
3. **Skill violation → SSR misread as "can't parallelize"** — I spawned 4 HMX workers, which the skill
   `htp-hardware-scheduling` explicitly forbids (HMX = 1 unit, process-serial, NEVER thread HMX; put the HMX
   consumer on MAIN, HVX work multithreaded). That oversubscription SSR'd the cDSP; it is an implementation
   bug, NOT evidence that GDNSolveHVXMixHMX can't be threaded. (There is 1 HMX unit, not 2.)

### Confirmed-effective (device-verified, real)
- **matmul-portion microbench**: HMX matmul + 2-head pack + fused depack + 2-head eff+bias =
  **578 cyc/matmul** (2.77× vs the vrmpy microbench 1601). commit `2cc2884`.
- **vectorized `gdn_pack_bias` LANDED in the full solve**: 128 scalar VTCM writes → 4 HVX vector stores,
  **T relerr BIT-IDENTICAL** (gdn_br.sh C=128: 8.538e-3, same to every digit; HMX path only). commit `2cc2884`.
- **multi-pass real cost** (MULTIPASS microbench, P=3, consumer-bound): scalar-bias 3-pass **4190** →
  vectorized-bias 3-pass **1720** (vectorizing the per-pass bias saves ~60% — scalar `gdn_pack_bias` was the
  pathology) → 2-pass #1c **1218** cyc/matmul.
- **#1c** (norm-predicted PASS-1 gain via `gdn_effective` colabsmax → drop PASS-2): in `gdn_merge_packed`;
  sim + gdn_br.sh verified **oc unchanged** (off-diag 21.2%→21.5%, baseline already 21%). Working tree (uncommitted).

### Full-solve current measurement (reference only — NOT GDNSolveHVXMixHMX's potential)
GDNSolveHVXMixHMX, nthreads=1, C=256, **OLD 1-head pack**, baremetal `-DGDNBM_HMX_MERGE_PATH`: 434K/head —
diag 132K / fold 122K (inflated by old pack) / merge 61K / quant 53K / other 59K. Single-thread +
un-integrated, so this number does NOT represent the path's potential.

### Baseline reference (GDNSolveHVX, this baremetal harness, C=256)
nthreads=4 = **157K/head**; nthreads=1 = 414K/head. (Shipped QNN-op numbers + metric alignment:
[[project_gdn_solve_handwritten_route_2026-06-03]], `docs/cycle_metric_alignment.md`.)

### Honest verdict
**GDNSolveHVXMixHMX's full-solve multithreaded performance vs GDNSolveHVX is UNKNOWN** — it needs (1) the
microbench pack/depack wins integrated into `gdn_merge_packed`, and (2) correct multithreading per the skill
(HMX-on-main consumer + parallel HVX, never threaded HMX). Only then is a fair verdict possible. What IS
proven: the matmul-portion is 2.77× faster on HMX, and the vectorized bias + #1c cut the multi-pass tax
~3.4× (4190→1218) oc-safely.
