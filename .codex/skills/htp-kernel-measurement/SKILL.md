---
name: htp-kernel-measurement
description: The reusable recipe for putting ANY new hand-written HTP kernel onto the unified, trustworthy QNN-standard measurement (P2 tooling). Use BEFORE/AFTER timing a new GDN-solve-class kernel (bare-metal FastRPC, custom QNN op, dspqueue kernel) — it pins the build-flag convention, the reps2-N median + clock self-check anchors, the ONE canonical timeline tool (scripts/htp_timeline.py), the ONE perf-report generator (scripts/htp_perf_report.py, honest-tail serial floor §3 + §4 stats[] dictionary), and the gated PMU真值 utilization (GP_PMU_UTIL, §6 tier-1) that does NOT pollute the wall口径. Authoritative SPEC = docs/cycle_metric_alignment.md § "HTP Kernel Measurement Standard" (§1 anchors / §2 4-questions→QNN field / §3 honest-tail / §4 stats[] dict / §5 timeline stage spec / §6 utilization tiers / §7 report template). This skill is the HOW; that section is the WHAT.
---

# HTP Kernel Measurement (the reusable harness recipe)

The contract is `docs/cycle_metric_alignment.md` § "HTP Kernel Measurement Standard". This skill is the
step-by-step for getting a NEW kernel onto it so every result is unified + trustworthy by default.

## 0. Anchors you must satisfy (§1) — non-negotiable
- **reps2-N median** (typ. reps2-8). NEVER min, NEVER a single rep. (`GDNBM_REPS=N`.)
- **VTCM-only** wall: bulk DDR↔VTCM moves OUTSIDE the timed window.
- **32-head TOTAL wall** = `stats[0]` makespan. Never per-head.
- **PCYCLE = C15:14 = QHAS = optrace** (no conversion). Clock self-check: `wall/µs ≈ 1592–1594`
  (v75 TURBO). If `cycles/µs` ≫ that, you read the wrong counter — re-measure.
- **same-thermal ACAC paired** for any A/B delta (interleave A,C,A,C…; never vs a historical constant).

## 1. Build-flag convention
- Production solve = `EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE" bash example/gdn_native/baremetal/build.sh`
  (lean + wt-cache + B are all default-ON; escapes are `-DGP_NO_LEANMM / -DGP_NO_WTCACHE /
  -DGP_NO_INPLACE_RENORM`).
- Diagnostic builds are GATED, default-OFF, and MUST leave the production byte stream unchanged:
  - `-DGP_TRACE` — emit the timeline event blob into the T output (timeline only).
  - `-DGP_PMU_UTIL` — post-solve PMU真值 pass (§6 tier-1) into `stats[20..23]` (clean-wall-safe).
  - `-DGP_LEANCHK_LIVE` — per-conv shadow lean-vs-native bit-exact check.
- **Prove production unchanged** for any new gated flag: preprocess the production-default TU and
  `grep -c` your flag's symbols (must be 0); then `git stash` the source and diff the two preprocessed
  streams after normalizing `__LINE__` literals in FARF/HAP_debug strings — they must be IDENTICAL
  (the only legitimate residual is `__LINE__` shift inside debug-log strings, never executed logic).

## 2. Run + steady-state median
```bash
# pure-HMX production, reps6 median + oc + QNN-aligned breakdown:
uv run python scripts/run_w16a16_head_phase4.py --deploy --threads 4 --heads 32 --scale 0.05 --reps 6
#   -> "WALL reps2-6 MEDIAN=... (steady; never min)" + "oc mean=..." + raw stats[0..19].
# device ssh: ALWAYS `source scripts/dssh.sh oneplus` first (ControlMaster mux; skill device-ssh-exec).
```
Read the steady median, NOT rep1 (warm-up) and NOT min. Confirm the clock self-check line.

## 3. Timeline — ONE canonical tool (scripts/htp_timeline.py, P2.1)
The §5 stage-id→class table lives ONLY in `scripts/htp_timeline.py` (do not re-define it elsewhere).
```bash
# capture: -DGP_TRACE build, run, the T output IS the trace blob (magic 0x47545203).
scripts/htp_timeline.py single   purehmx  trace.raw  out.svg     # one impl
scripts/htp_timeline.py single   hvxmix   trace.raw  out.svg     # ship|ares|hvxmix titled variants
scripts/htp_timeline.py aggregate ship.raw ares.raw pure.raw out.svg   # 3 impls stacked per thread
scripts/htp_timeline.py ascii    purehmx  trace.raw  [width]     # ASCII
```
Hard rules it enforces (§5): MM (stage 3) drawn ONLY on the consumer; stage 5 is a PACK LEAF in BOTH
taxonomies; a prominent caveat band says single-rep = STRUCTURE only (utilization off the perf 表,
never off the figure). The old 5 scripts (`gdn_{pipe,pure_perfetto,hvxmix_perfetto,perfetto,3impl_
aggregate}_timeline.py`) are DEPRECATED wrappers that forward here.
Archived regression blobs: `Agent/current/trace_blobs_3impl/{ship,w16n8,purehmx}_trace.raw`.

## 4. Perf report — ONE generator (scripts/htp_perf_report.py, P2.2)
Feed a small JSON of P-points (raw `stats[]` + `path` = `hvxmix`|`pure`); it emits the §7 template.
```bash
scripts/htp_perf_report.py spec.json [out.txt]
```
It handles the §4 disambiguation automatically (`stats[4]` = feed_Σ on HVXMix but spin_Σ on pure-HMX;
`lmax` = `stats[8]` on HVXMix vs `stats[11]` on pure-HMX), and computes the **serial floor as the
per-P honest-tail `tail(P)=wall(P)−实测feed_Σ(P)/P`** (§3, NOT the constant-K regression intercept —
that inflates 3× when feed_Σ grows with P, e.g. ARES). pure-HMX feed_Σ is a production decomposition
value (not in `stats[]`) — supply it via `"pure_feed": {"4": <feed_Σ>}`.
Spec keys: `title / device / reproduce / points[] / pure_feed / pmu`. Cross-check: feeding the ARES raw
stats reproduces honest-tail @P4 = 0.254M; pure-HMX @P4 = 0.349M.

## 5. Utilization (§6) — trust order
1. **PMU真值** (tier-1): build `-DGP_PMU_UTIL`, run; the host prints `PMU_UTIL (§6 tier-1): per-call
   COPROC_BUSY/THREAD_IDLE/CYCLES_1T/PKT` and writes them to `stats[20..23]`. Feed them to the report
   via `"pmu": {"4": {"coproc_busy": <frac>, "thread_idle": <frac>}}`. The PMU pass runs AFTER the clean
   wall is captured (separate back-to-back kernel loop while the HMX lock is still held) so it CANNOT
   pollute `stats[0]` — verify with an ACAC paired clean-vs-PMU wall comparison (must be in thermal
   noise). NTSWEEP `stats[20..23]` are gated off under `-DGP_PMU_UTIL` to free the slots.
2. **steady-derived (non-PMU)**: `(feed/P)/lmax` (HVX util/thread) or `cbusy/wall` (HMX util). Label it
   "steady-derived (non-PMU)" explicitly — the report generator does this automatically.
3. ❌ **single-rep trace utilization% = VOID** — never quote it.

## 6. Bit-exact / correctness gate FIRST
Correctness before perf: `oc` (device T vs fp64 inv) + `PACKCHK=0` + `GP_LEANCHK_LIVE max|d|=0
checked=768`. The strongest gate = default(B on) vs escape(int32 oracle) 32-head 4MB T md5 byte-identical.
A perf number without its correctness gate is not a result.

## Pointers
- SPEC (authoritative): `docs/cycle_metric_alignment.md` § "HTP Kernel Measurement Standard".
- Tools: `scripts/htp_timeline.py`, `scripts/htp_perf_report.py`.
- pure-HMX loop doc: `Agent/current/pure_hmx_solve_build.md`; route doc: `Agent/current/gdn_solve.md`.
- P2 tooling regression artifacts: `Agent/current/p2_tooling/`.
- device ssh: skill `device-ssh-exec` (`source scripts/dssh.sh` first, always).
