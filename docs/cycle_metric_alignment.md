# Cycle-metric alignment: shipped `GdnSolve` (QNN optrace) vs bare-metal solve (C15:14)

**Problem this solves.** The two GDN-solve performance numbers were measured with different
instruments and could not be compared:

- shipped **`GdnSolve`** (pure-HVX forward-substitution) reported as *"~70–83K steady cyc/head"* from
  QNN optrace (`chrometrace_qnn_htp_analysis_summary.json`);
- hand-written **bare-metal** solve reported as *"~151–172K cyc/head"* from a `C15:14` (PCYCLE) read
  around the qurt spawn→join.

Every prior "Nx faster/slower" claim was therefore untrustworthy. The repo docs explicitly refused to
quote a ratio ("⚠️ No speed-ratio vs shipped yet"). This doc establishes the exact, device-proven
alignment.

## TL;DR — the result

1. **QNN's QHAS per-op `cycles` field IS the `C15:14` (PCYCLE) counter.** Proven by reading `C15:14`
   from *inside* a QNN custom op and comparing to what QHAS reports for that same op-instance:
   **C15:14 = 18,159,963 vs QHAS = 18,227,561 → ratio 0.9963** (≈1.000; the 0.4% is the few packets
   outside the probe wrap). **No conversion factor is needed** — both sides already count the same
   register. (The repo note "1µs = 4209 acc-cyc" refers to a *different*, faster accelerator counter,
   not PCYCLE; do not use it to convert these numbers.)

2. **The old "70–83K cyc/head" shipped number was a measurement artifact**, not a real per-head wall.
   It was computed as `mean(GdnSolve tile 'cycles') / 8 heads`. But the central tiler splits H=32 into
   **24 tile-instances** (not 4), spread over 4 HVX threads (~6 serial tiles per thread). Dividing a
   single tile's cycles by 8 heads ignores that each thread runs several tiles serially, under-counting
   by ~2×.

3. **Aligned, apples-to-apples (C=256, H=32, 4-thread, all C15:14/PCYCLE):**

   | metric (per head) | shipped `GdnSolve` (fwd-subst) | bare-metal BR (VTCM-resident) |
   |---|---|---|
   | compute-busy / head | **146,963** (max-thread `cycles_used`/H) | ~**161,040** (overhead-removed) |
   | full wall / head | **190,356** (PCYCLE span/H) | **156,287** (`(t1−t0)`/H) |

   **True gap: ~1.0–1.1× — parity, NOT the previously feared 2–3×.** The shipped op is *not* 2× faster
   than the bare-metal; the 70–83K baseline was ~2× too optimistic. (They are different algorithms:
   shipped = forward-substitution, HMX idle; bare-metal = block-recursive with int8-HMX merges.)

## What each number actually measures

### Shipped `GdnSolve` — QHAS / optrace

Decoded from `out_s/optrace/chrometrace_qnn_htp_analysis_summary.json` (QHAS) and `chrometrace.json`:

- **`htp_resources[].cycles_used`** — per hardware unit (HMX tid=256; the 4 HVX threads tids=512..515).
  This is **per-thread busy PCYCLEs** (compute only, no inter-op gaps). The units run in parallel, so
  the op's real compute time = the **busiest** thread's `cycles_used` (DOMAIN cycle), **not** their sum.
  For C=256 H=32: HVX threads = [4.45M, 4.70M, 4.14M, 3.41M], HMX = 0.18M (2.9% util → HVX-bound).
  **Compute-busy per head = max(HVX cycles_used)/H = 4,702,802/32 = 146,963.**
- **`htp_resources[].start_cycle/end_cycle`** — absolute free-running `C15:14` PCYCLE values
  (e.g. `8012397961807`). **Graph wall = max(end) − min(start) = 6,091,377 PCYCLE** ⇒ **190,356/head**.
  This includes the QNN per-tile dispatch bubbles between tiles within a thread (~28% over compute-busy).
- **`htp_op_instances[].cycles`** — per **tile-instance** PCYCLEs. 24 instances for H=32 (`dims:[1,8,256,256]`
  is the op's declared tile shape, but the central tiler subdivides further). `sum(cycles)` = work volume
  (16.7M ≈ Σ per-thread busy); a single tile's `cycles`/8 is **not** a per-head wall.
- Clock sanity: graph PCYCLE span / `QNN accelerator (execute) time` µs = 6,091,377 / 4285 µs =
  **1422 cyc/µs ≈ 1.42 GHz** (v75 TURBO). NOTE the `chrometrace_runtrace.json` phase counters use a
  *different* (higher, ~1.78 GHz) reference — do not mix them with QHAS/PCYCLE.

### Per-op: latency (dominant-path) vs throughput (busy) — the trap that flipped an int16 verdict

One op has **two** legitimate cycle numbers; they answer different questions and can differ several-fold:

- **`htp_op_instances[].num_dominant_path_cycles`** = the op's **latency** = critical-path PCYCLEs for that op
  instance (data resident). This is "how long this op takes." Use it for **dependency chains** and when the
  unit is **idle-mostly** (the op's latency, not its occupancy, sets the wall).
- **`by_htp_type` cycle sum / per-unit `cycles_used`** = **throughput / occupancy** = total cycles the unit is
  busy across the op's internal work. Use it only when the unit is **saturated** (back-to-back, the bottleneck).

They diverge when an op **pipelines internal passes**. Device example (native int16 64³ MatMul vs u8i8):

| | u8i8 | int16 (4 byte-pass) | ratio |
|---|---|---|---|
| latency (dominant-path) | 176 | **256** | **1.45×** |
| throughput (HMX-busy / by_htp_type) | ~194 | ~1167 | **6.0×** |

The 4 int16 byte-passes pipeline, so latency (256) ≪ throughput (1167). **Picking the wrong one flips verdicts:**
reading the **6× throughput** as "the int16 kernel cost" wrongly killed the int16 GDN-inverse merge; the inverse
is producer-bound (HMX ~7% busy = idle-mostly), so the merge's relevant cost is its **1.45× latency**, not its
throughput. **Rule: pick latency vs throughput by whether that unit is the saturated bottleneck.** Full worked
case: `Agent/current/int16_matmul_cycle_model.md`.

### Bare-metal solve — `C15:14`

`gdnbm_solve` (`example/gdn_native/baremetal/src/gdnbm_imp.cpp`) reads `pcyc()` = `C15:14` (PCYCLE)
immediately before the qurt `thread_create` loop and immediately after the `thread_join` loop. So
`stats[0] = t1−t0` = **wall PCYCLEs for all H heads across the 4 parallel threads**, and the harness
prints `stats[0]/H`. This is wall (parallelism already folded in via /H), and it includes a fixed
spawn/join/power-vote overhead.

H-sweep fit (4-thread, plain build) isolates that overhead:
`wall = 178,206 (fixed spawn/join/power) + 644,161 · (H/4 heads-per-thread)`.
⇒ overhead-removed compute = `644,161/4 = 161,040 cyc/head`.

## Same-code cross-proof (the rigorous alignment)

The bare-metal device code and the QNN `solve_br_op` are the **same C++** (`gdnbm_imp.cpp` `#include`s
`GdnSolveBROp.cpp`). Building that one op **both ways** and adding a `C15:14` probe (`-DGDN_BR_PROBE_TOTAL`,
writes `C15:14` total into output head 0) gives the conversion directly:

| run of the IDENTICAL BR solve (single calling thread, H=32) | counter | value | /head |
|---|---|---|---|
| as a **QNN custom op** (optrace flow) | `C15:14` read inside op | 18,159,963 | 567,499 |
| as a **QNN custom op** (same run)     | **QHAS `cycles`**       | 18,227,561 | 569,611 |
| as **bare-metal** (`nthreads=1`)      | `C15:14` (`t1−t0`)      | 20,703,031 | 646,970 |

- **QHAS `cycles` / `C15:14` = 0.9963** → they are the same counter. (This is the key alignment.)
- bare-metal NT=1 is ~14% higher than the QNN op only because bare-metal reads `A` from **uncached
  FastRPC DDR** while the QNN backend stages `A` in TCM; the `-DGDNBM_VTCM_RESIDENT` build closes this.

## Reproduce

```bash
source scripts/env.sh

# (1) Shipped GdnSolve, C=256 H=32 — wall µs + QHAS, decode optrace
CS=256 H=32 bash example/gdn_native/solve_op/standalone/gdn_shape.sh
#   then read QHAS htp_resources from:
#   example/gdn_native/solve_op/standalone/out_s/optrace/chrometrace_qnn_htp_analysis_summary.json
#   compute-busy/head = max(HVX cycles_used)/32 ; wall/head = (max end_cycle - min start_cycle)/32

# (2) Bare-metal BR solve (VTCM-resident, 4-thread) — C15:14 wall
#   generate A_u16_h32.raw (quantize the same fp32 A at sA=2.770166930875267e-05, zp=32768):
.venv/bin/python - <<'PY'
import numpy as np
a0=np.fromfile("example/gdn_native/solve_op/standalone/A_ref.raw",dtype=np.float32).reshape(-1,64,64)
H,C=32,256; reps=(C+63)//64
A=np.stack([np.tril(np.tile(a0[i%a0.shape[0]],(reps,reps))[:C,:C]*0.7,-1) for i in range(H)]).astype(np.float32)
sA=2.770166930875267e-05
np.clip(np.round(A/sA)+32768,0,65535).astype(np.uint16).tofile(
    "example/gdn_native/baremetal/A_u16_h32.raw")
PY
cd example/gdn_native/baremetal
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT" bash build.sh
W=$(ssh oneplus 'echo $HOME/gdnbm_run'); ssh oneplus "mkdir -p $W"
ssh oneplus "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
ssh oneplus "cat > $W/gdnbm" < build/gdnbm; ssh oneplus "chmod +x $W/gdnbm"
ssh oneplus "cat > $W/A_u16_h32.raw" < A_u16_h32.raw
ssh oneplus "cd $W && LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
  ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
  ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05"
#   -> "NNNNNN cyc/head (4-thread)" = C15:14 wall / 32

# (3) Same-code cross-proof: BR op as a QNN op, read C15:14 inside it
EXTRA_DEFS="-DGDN_BR_HVX_MERGE -DGDN_BR_PROBE_TOTAL" H=32 CB=256 \
  bash example/gdn_native/solve_br_op/standalone/gdn_br.sh
#   recover the probe (output head 0 is dequantized fp32; codes = round(f/sT)+zpT, reassemble u32 pairs):
.venv/bin/python - <<'PY'
import numpy as np
f=np.fromfile("example/gdn_native/solve_br_op/standalone/out_s/Result_0/T.raw",
              dtype=np.float32).reshape(32,256,256)[0].ravel()
codes=np.clip(np.round(f/6.103701895199438e-05)+32768,0,65535).astype(np.uint16)
u=np.frombuffer(codes[:16].tobytes(),dtype=np.uint32)
print("C15:14 total", int(u[0])|(int(u[2])<<32), "heads", int(u[1]))   # vs QHAS aggregate from gdn_br.sh
PY
```

## Practical rule for future GDN-solve comparisons

- **Compare per-head in PCYCLE = QHAS `cycles` = `C15:14`** (one counter, no conversion).
- For the shipped (QNN-tiled, multi-thread) op, the per-head numbers are:
  - **compute-busy/head = max(HVX `cycles_used`)/H** (kernel/algo efficiency — the DOMAIN cycle),
  - **wall/head = (max `end_cycle` − min `start_cycle`)/H** (latency incl. QNN dispatch bubbles).
  - **Never** use `mean(tile cycles)/heads_per_tile` — the tiler emits more tiles than `H/8`.
- For the bare-metal op, `(t1−t0)/H` ≈ wall/head; subtract the H-sweep-fit overhead (~178K) for the
  compute-only figure.
- Cross-check the clock once: graph PCYCLE span / `QNN accelerator (execute) time` µs ≈ 1.42e3 cyc/µs
  (v75 TURBO). If `cycles/µs` ≫ that, you are reading an aggregate/other counter.

---

# Canonical cycle taxonomy (AUTHORITATIVE — cron#81 SPEC; cite this, don't re-copy)

**The bug this kills (the SECOND-level口径 error, after the 290-vs-1576 one).** Even after cron#78
corrected "native 290 = we're 4.5× slow" → "our 1576 ≈ native single 1970, NOT slow", the docs kept
calling that **1576/conv (1.31M total)** the consumer's "真算 / HMX compute". It is NOT. The HMX *unit*
only MACs ~363/conv; the other ~1213/conv is liftable non-MAC kernel bloat. Mislabeling occupancy as
"真算" produced the chain of wrong verdicts: "producer-bound", "consumer only worth ~11%", "165/walk =
pure mxmem", "#1 lever = producer feed". All are口径 errors. The taxonomy below is the single source of
truth; every pure-HMX number is tagged with WHICH level it is.

## The 6 cycle levels (the only legal vocabulary for a per-stage number)

A solve number must say which of these it is — they differ by **4–5×** and conflating them is what
repeatedly inverted verdicts. Unit = PCYCLE (= C15:14 = optrace; this baseline self-checks ~1593 PCYCLE/µs).

| level | what it is | pure-HMX value (device, cron#77 baseline `Agent/current/perf_baseline_cron77.txt`) | NEVER call it |
|---|---|---|---|
| **HMX-unit MAC (真算)** | the HMX *unit* actually MACing — the only irreducible work | **~363/conv** (lean from-scratch dilate micro, 4.5 cyc/pkt, SAME 16 tile-MAC, cron#78); total ~0.28M (768×363) | the consumer's cost (it's a fraction of it) |
| **Consumer kernel occupancy (整核)** | the consumer *thread* running the matmul kernel (convhhh) per call = thread-wall | **~1576/conv** = MAC(~363) + non-MAC bloat(~1213: bias staircase + M-loop + 14 cyc/pkt packet-stall); total `cbusy` **1.31M** | **"HMX compute / 真算" — this is the classic confusion.** It is thread-occupancy, mostly liftable. |
| **Σ work-volume** | a stage's cycles summed across ALL threads/calls — NOT a wall | e.g. wt-vec Σ 2.05M = 4 producer threads = 0.51M/thread | a wall; **never rank wall levers by a Σ %** |
| **Critical-path / thread-life** | the slowest single-thread span — correlates with wall | slowest-prod-life lmax = **1.53M** (= feed 1.00M + spin 0.50M + skew) | "producer work" (it includes spin = waiting the HMX) |
| **Roofline floor** | max(parallel producer-feed/P, serial consumer-occupancy) | current = max(1.00M, **1.31M**) = **1.31M**; after a lean consumer = max(1.00M, 0.28M) = **1.00M** (🟡 derived) | settled (the lean-consumer branch is DERIVED) |
| **Wall** | 32-head TOTAL, VTCM-only, reps2-8 median | **1.738M** (the final verdict number) | — |

**The crucial distinction (memorize): consumer-thread-occupancy ≠ HMX-unit-MAC.** Same 16 tile-MAC →
lean dilate micro 363 cyc vs convhhh 1576 cyc (4.3×). The 1213-cyc difference is kernel bloat
(THREAD_IDLE/WAIT between MAC packets), **not** the HMX doing math. So:
- **真算 (the irreducible floor) = ~363/conv ≈ 0.28M**, NOT 1576/1.31M.
- A Σ work-volume number (e.g. wt-pack 2.52M Σ) is the sum over 4 producers (0.63M/thread); it sits
  *below* the roofline floor, so its **%-of-Σ does NOT set its wall priority**.
- latency vs throughput (which field to read) follows the existing htp-cycle-metric rule (pick by whether
  that unit is the saturated bottleneck); unchanged.

## Current pure-HMX numbers, every one tagged to a level + 🟢verified/🟡derived (cron#81 SPEC)

| quantity | value | level / 口径 | tag |
|---|---|---|---|
| wall | 1.738M | Wall (32-head total, VTCM-only, reps2-8 median) | 🟢 |
| HMX-unit MAC (真算)/conv | ~363 | HMX-unit MAC (lean kernel, same 16 tile-MAC, cron#78) | 🟢 |
| HMX-unit MAC total | ~0.28M | 768 × 363 | 🟢/🟡 (×768 not yet measured on the solve path) |
| consumer occupancy/conv | ~1576 | Consumer kernel occupancy (convhhh, 14 cyc/pkt) | 🟢 |
| consumer occupancy total (cbusy) | 1.31M | = MAC 0.28M + bloat ~1.0M | 🟢 |
| producer feed/thread (feed_pt) | 1.00M | DERIVED: (verified-Σ 的合计)/P — Σ_HVX≈4.02M = wt-vec 2.054M + wt-bias 0.456M + act 0.112M + renorm/acc 1.255M + outcopy 0.139M (各 Σ 均逐字见 perf_baseline_cron77.txt;outcopy=卸料 HVX,producer 线程也做), ÷4 | 🟡 DERIVED |
| spin/thread (spin_pt) | 0.50M | DERIVED: SPIN Σ 2.007M (🟢 逐字见 perf_baseline_cron77.txt `[waste] SPIN idle-wait Σcyc=2007041`) ÷4 = waiting-HMX, **consumer-dependent** | 🟡 DERIVED |
| slowest-prod-life (lmax) | 1.53M | Critical-path (feed 1.00M + spin 0.50M + skew) | 🟢 |
| roofline floor (current) | 1.31M | max(feed/P 1.00M, consumer-occupancy 1.31M) | 🟢 |
| roofline floor (after lean consumer) | 1.00M | max(feed/P 1.00M, MAC 0.28M) | 🟡 derived |
| wall ceiling (after lean consumer) | ~1.0M + glue ≈ **−42%** | derived | 🟡 derived |
| wall − current floor = schedulable slack | ~0.43M | glue + spin-bubble + imbalance | 🟢 |

## The verdict, restated to this taxonomy (and what it RETRACTS)

**Current pure-HMX is consumer-occupancy-bound AT THE FLOOR 1.31M — and it is BLOAT-bound, not
MAC-bound.** The #1 lever is **leaning the consumer kernel** (strip the ~1.0M of non-MAC bloat →
consumer 0.28M → floor 1.00M); wt-pack feed is the SECOND tier (it only becomes binding after the
consumer is leaned). The following earlier claims are RETRACTED (kept struck for the audit trail):

- ❌ ~~"producer-bound (lmax 1.53M > consumer 1.30M)"~~ → lmax includes 0.50M spin = waiting-HMX, not
  pure producer work. **Correction: currently consumer-occupancy-bound at floor 1.31M (and bloat-bound,
  not MAC-bound).**
- ❌ ~~"leaning the consumer is worth only ~11% of wall"~~ → spin is consumer-dependent. **Correction:
  ceiling ~42% (🟡 derived).**
- ❌ ~~"真算 = 1576/conv (1.31M)" / "165/walk = pure mxmem"~~ → 1576 is the bloated occupancy; 165/walk
  includes stall. **Correction: 真算 = 363/conv ≈ 0.28M.**
- ❌ ~~"#1 lever = producer feed (wt-pack vgather)"~~ → wt-pack is below the floor (Σ 2.52M = 0.63M/thread;
  0.51M is the wt-**vec** sub-item per-thread, not wt-pack). **Correction: #1 lever = lean the consumer
  kernel; wt-pack is the second tier, binding only after.**

## Honesty rule (so this doesn't mislead in the OTHER direction)

The *decomposition* of the verdict ("consumer-occupancy-bound / #1 = lean consumer kernel") is
**🟢 verified**. But the **42% ceiling is 🟡 DERIVED** — it assumes the lean from-scratch micro (cron#78's
363 cyc) actually drops onto the solve's consumer path, which has NOT been done (the 363 is a
from-scratch micro, not yet landed in the solve consumer). Tag it 🟡; **do not write it as a new
"settled" number.**

---

# Canonical 口径 map: our pure-HMX solve ↔ QNN optrace (cron#79)

**The bug this kills.** Our solve's instrumentation used an ad-hoc taxonomy
(`actcopy/kmajor/scatter/renorm/spin/outcopy + lmax`; the host print even used the DELETED
`①②③④` names) that did NOT map to QNN's op categories or fields — so numbers got cross-compared
(the canonical example: native `mm_64`'s ConvLayer **batch warm sub-op `cycles`=263** vs our **per-call
occupancy 1577** — a phantom 4.5× "gap" that is a cross-scenario error; the real apples-to-apples is
native SINGLE ConvLayer `mm_1x1x64x64` = **1970 cyc / 15.15 cyc-pkt ≈ our 1577**). From cron#79 every
solve perf number is reported in the **three QNN categories**, each tagged with its **QNN op name +
QNN field**, so a comparison is automatically same-category + same-field.

## The three categories (the only legal grouping)

| category | meaning | unit | native ops (mm_64) | our solve segment |
|---|---|---|---|---|
| **真算 (MAC)** | the only irreducible work | **HMX** (tid256) | `q::ConvLayer_s1.opt` | consumer `w16a16_mm_run` (`g_cbusy`) |
| **装料 (prep)** | feed: weight/act/bias format | **HVX** (tid512-515) | `convert_weights_to_signed`,`Cast`,`ForceFormat_Crouton`,`Reshape`,`bias_weight_update`,`bias_scale_shuff`,`Slice_contig.tcm` | `gp_pack_wt_bias` (vec+bias), `gp_cv_to_surf`, renorm/acc |
| **卸料 + 输入 (IO)** | DMA in/out | **DMA** (tid256) | `q::*InputSlice`, `q::*OutputSlice` | `gp_surf_to_cv` (卸料), bulk DDR↔VTCM (输入/卸料, EXCL from wall) |
| *(waste)* | idle, NOT a category | — | *(scheduling gap)* | `SPIN` (producer waits the 1 HMX) |

## Segment ↔ QNN op ↔ field (exact)

| our segment (stat) | QNN op | category | QNN field we report | field we CAN'T (and why) |
|---|---|---|---|---|
| `gp_cv_to_surf` (`t_pack`→stats[19] actcopy) | `q::ForceFormat_Crouton` | 装料-act | cycles (Σ, per-conv) | per-instance / cyc-pkt: bare-metal has no per-op DMA split |
| `gp_pack_wt_bias` vec (`t_gather`→stats[17] wt_vec) | `convert_weights_to_signed`+`Cast` | 装料-wt | cycles (Σ, per-conv) | cyc-pkt (HVX feed not PMU-split per segment) |
| `gp_pack_wt_bias` bias (`t_bias`→stats[18] wt_bia) | `bias_weight_update`+`bias_scale_shuff` | 装料-bias | cycles (Σ, per-conv) | — |
| consumer `w16a16_mm_run` (`g_cbusy`→stats[3]) | `q::ConvLayer_s1.opt` | **真算-MAC** | **cycles_used** (Σ); per-call occupancy stats[5]; **packets/cyc-pkt** via `-DGP_PKTPROBE` | `num_dominant_path`: bare-metal C15:14 = occupancy, not dominant-path |
| renorm/acc (`other`→stats[10]) | *(none — solve-specific)* | 装料-alg | cycles (Σ) | no native counterpart (honestly flagged) |
| `gp_surf_to_cv` (`t_depack`→stats[28] outcopy) | `q::*OutputSlice` | 卸料-IO | cycles (Σ) | — |
| bulk DDR↔VTCM (`bulk_ld`/`bulk_st`→stats[15/16]) | `q::*InputSlice`+`q::*OutputSlice` | 输入/卸料 | cycles (one-time, **EXCL from wall**) | — |
| `SPIN` (`spin`→stats[4]) | *(gap, not an op)* | waste | cycles (Σ) | — |

## The fields, and how to read each (skill `htp-cycle-metric`)

- **cycles** — per-op-type `cycles` on the HVX/DMA side; for HMX = `cycles_used` (occupancy of ONE op,
  incl. fill). Bare-metal C15:14 back-to-back = this. **Per-conv = Σ ÷ N_conv** (N_conv = H×24 @ Newton=0).
- **num_dominant_path_cycles** — critical chain after ideal overlap (a lower bound). Native exposes it;
  bare-metal C15:14 is occupancy, so we report occupancy and **never** compare our occupancy to native's
  dominant-path.
- **cyc/pkt** (`cycles_per_packet`) — stall indicator. Our MAC = **14 cyc/pkt** (111 pkt/call, `-DGP_PKTPROBE`)
  vs native ConvLayer SINGLE **15.15** / batch **2.04**. Same packets, same kernel; the batch's 2.04 is fill
  overlapping the adjacent op, not a cheaper MAC.
- **THROUGHPUT** — `graph-wall ÷ N` or the `start_cycle` retire interval. **NEVER `cycles_used`/N** — it
  OVERSTATES (trap #6: native HMX `cycles_used`/32 = 1690 ≠ retire ~290).
- Every number = **value + QNN field + shape + scenario** (single conv vs batch; wall vs HMX-busy).

## Apples-to-apples 真算-MAC (device-measured, cron#79, all in QNN fields)

| scenario | QNN op | cycles | packets | cyc/pkt | what it is |
|---|---|---|---|---|---|
| **ours, single per-call** | `w16a16_mm_run` | **1577** (occupancy) | **111** | **14.2** | our consumer MAC, back-to-back resident |
| native, single conv | `mm_1x1x64x64` ConvLayer | **1970** | 130 | **15.15** | **apples-to-apples → we are NOT slower** |
| native, batch warm sub-op | `mm_64` ConvLayer | **263** (canonical) | 130 | 2.04 | HMX sub-op `cycles`, warm-steady (fill overlaps adjacent op) — **NOT a conv wall** |
| native, batch true per-conv | graph-span ÷ 32 | ~4087 | — | — | batch is HVX-bound (HMX Σbusy = 8.6% of span) |

> **口径 note (cron#78 canonical, avoid re-mixing):** native `mm_64` ConvLayer per-conv has TWO numbers —
> **warm-steady = 263 cyc (canonical; self-consistent with cyc/pkt 2.04; THIS is what we compare our warm
> per-conv to)** vs **mean-incl-cold = 349 (= Σ11176 ÷ 32, includes conv0 cold-start 2725; NOT a throughput
> 口径)**. A third, distinct口径 is the `start_cycle` retire interval ~290 (throughput, see build doc) —
> never fold 290 into the 263/349 comparison.

⇒ **Our consumer is NOT behind native; the gap to native's batch warm 263 is a cross-scenario illusion.**
The recoverable lever is fewer/leaner packets (a from-scratch dilate micro = 363 cyc / 81 pkt / 4.5 cyc-pkt
vs convhhh 1577/111/14), NOT cross-conv pipelining (HMX has 1 acc; from-scratch SERIAL==PIPE, cron#78).

## Reproduce

```bash
# native side (decode the two reference optraces into the canonical category table):
uv run python scripts/gdn_solve_qnn_aligned_report.py --native-only
# full side-by-side (parse a device-run stdout):
uv run python scripts/run_w16a16_head_phase4.py --threads 4 --heads 32 --scale 0.05 > /tmp/run.txt 2>&1
uv run python scripts/gdn_solve_qnn_aligned_report.py --our-log /tmp/run.txt
# our consumer-MAC packets + cyc/pkt (diagnostic build; skips solve, production unaffected):
EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE -DGP_PKTPROBE" bash example/gdn_native/baremetal/build.sh
```
The aligned host print + per-category stats are gated by nothing extra — they reuse the always-on
`gp_pcyc` probes, so the production build is bit-exact (oc 4.238e-3, PACKCHK=0, verified cron#79).
