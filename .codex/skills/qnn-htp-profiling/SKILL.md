---
name: qnn-htp-profiling
description: Measure QNN HTP op/graph PERFORMANCE correctly via optrace in qcom_htp. Use this BEFORE making ANY HTP perf claim (op cycles, graph latency, "faster/slower", parallelism/threads). It pins the one metric that matters (WALL µs + per-op Dominant Path Cycles) and the exact ctxgen→net-run→decode flow, so you never again misread qnn-profile-viewer aggregate "Duration (cycles)" as wall.
---

# QNN HTP Profiling (perf, not pictures)

Use this skill whenever you need a **performance number** from the HTP backend — op cycles, graph
latency, "X is faster/slower than Y", or "does this op use multiple HVX threads". For drawing the
op-flow graph as an SVG, use `qnn-optrace-svg` instead; this skill is about the *numbers*.

**Read this skill BEFORE quoting any perf number.** Skipping it once produced a fully-inverted
conclusion ("op is 1.4× slower / single-thread / needs manual qurt") that was actually **5.5× faster
and already multithreaded** — a pure metric misread.

## Cardinal rule: ALWAYS read BOTH a coarse and a fine view, together

Never conclude from a single number. Two views, cross-referenced every time:

**Coarse (high level) = pure COMPUTE time** — `Accelerator (execute) time` (µs) = compute-busy time (op
in-flight, excludes inter-op gaps), plus `Accelerator (execute) time (cycles)` for total work (this
cycle field is the thread-AGGREGATE — ~4× a 4-thread op's wall; use it for *work volume*, never as wall).
This says how much real work the graph does.

**Fine (detailed) = optrace** — decode to `chrometrace.json` and read per-op `args["Dominant Path
Cycles"]` (`ph=X`, `pid=0`; NOT `Duration (cycles)`). This says *where* the time goes per op, what is
compute vs inter-op overhead, and whether an op was tiled/parallelized (a tiled op appears as multiple
instances). See Flow B.

**Low level (root cause / kernel optimization) = HTP op → HVX/HMX → packet** — for any "why is it slow"
or kernel-tuning question you MUST go down to the hardware-unit layer (QHAS): which unit (HMX/HVX/DMA),
its utilization, and **`cycles_per_packet`** per HTP sub-op, plus VTCM/DRAM traffic and the dominant
path. Per-op cycles alone never tell you this. See Flow C.

**Why both are mandatory (they can disagree):** `Accelerator (execute) time` (compute) vs `QNN
accelerator (execute) time` (full-graph WALL incl. per-op dispatch/scheduling overhead) can tell
opposite stories. Real case — GdnSolve op vs 363-node int8-matmul solve:

| | compute (`Accelerator time`) | full wall (`QNN accelerator time`) |
|---|---|---|
| baseline 363 int8-matmul ops | **3.2 ms** (less compute) | **68 ms** (65 ms is pure per-op overhead) |
| 1 GdnSolve HVX op | 4.7 ms (more compute) | **11.6 ms** |

Coarse alone ("baseline computes less") and wall alone ("solveop 5.5× faster") are each half-truths;
**together** they give the real story: solveop does *more* compute but wins latency 5.5× by collapsing
363 ops' dispatch overhead into 1 op. Always state both. Fields nest
`NetRun > RPC > QNN accelerator > Accelerator`; `QNN accelerator` µs = full-graph wall (latency),
`Accelerator` µs = compute-busy.

**Sanity tell-tale:** if `cycles / µs` exceeds the core clock (v75 ≈ 1.0–1.4 GHz), the cycle number is
aggregate, not wall. e.g. 12M cycles / 3000 µs = 4 "GHz" ⇒ aggregate over ~4 threads, real wall ≈ 3 ms.

## Flow A — graph wall µs (quick headline)

Net-run with optrace already gives the wall:

```bash
# device run already uses: --profiling_level detailed --profiling_option optrace
ssh "$DEVICE" "cd $W && ... ../qnn-net-run ... --profiling_option optrace ..."
# pull qnn-profiling-data_0.log (binary — use tar, not cat), then:
qnn-profile-viewer --input_log prof.bin | grep "QNN accelerator (execute) time"   # <- WALL µs
```
Compare two configs by their WALL µs. (Same graph, same #inferences, burst, warm run.)

## Flow B — per-op Dominant Path Cycles (who owns the wall, tiling/threads)

DPC needs the **optrace schematic**, which only ctxgen-with-optrace emits (into CWD, ignoring
`--output_dir`):

```bash
# 1) ctxgen WITH optrace -> drops <name>_schematic.bin in CWD
qnn-context-binary-generator --dlc_path g.dlc --backend .../libQnnHtp.so \
    [--op_packages <x86_pkg>:<Provider>] --config_file _cfg.json \
    --profiling_level detailed --profiling_option optrace \
    --binary_file g_ctx --output_dir ctx_ot
ls *_schematic.bin                       # <- pairs with the run log below

# 2) device net-run on THAT ctx, with optrace (binary log: pull via tar)
#    ... qnn-net-run --retrieve_context g_ctx.bin ... --profiling_option optrace ...

# 3) decode log + schematic -> optrace/chrometrace.json  (source scripts/env.sh first!)
.venv/bin/python scripts/decode_qnn_optrace.py <out_dir> \
    --profile-log <out_dir>/qnn-profiling-data_0.log --schematic <name>_schematic.bin

# 4) read Dominant Path Cycles per op (ph=X, pid=0). Tiled ops show as MULTIPLE instances.
.venv/bin/python - <<'PY'
import json
ev=json.load(open("<out_dir>/optrace/chrometrace.json"))["traceEvents"]
rows=[(int(e["args"]["Dominant Path Cycles"]), e["name"])
      for e in ev if e.get("ph")=="X" and e.get("pid")==0 and "Dominant Path Cycles" in e.get("args",{})]
rows.sort(reverse=True)
print("total DPC", f"{sum(d for d,_ in rows):,}")
for d,n in rows[:10]: print(f"{d:>10,}  {n}")
PY
```

The reader lib is `$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so`
(decode_qnn_optrace.py wires it up). Standard artifacts land in `<out_dir>/optrace/`:
`chrometrace.json` (timing), `chrometrace_htp.json` (node flow — feed to `qnn-optrace-svg`),
`chrometrace_qnn_htp_analysis_summary.json` (QHAS — the low-level layer, see Flow C).

## Flow C — low level: HTP op → HVX / HMX → packet (MANDATORY for any "why is it slow" claim)

Bottom-level perf analysis MUST reach the hardware-unit layer, not stop at per-op cycles. It lives in
`optrace/chrometrace_qnn_htp_analysis_summary.json` (QHAS). Read:

- **`data.htp_overall_summary…htp_resources`** — per HW unit (HMX, the 4 HVX threads): `utilization`,
  `cycles_used`, plus graph `percent_idle`, `total_vtcm`, `total_dram`, `peak_vtcm_alloc`. Tells you
  WHICH unit the work is on and whether you're compute-bound, idle, or memory-bound.

  **DOMAIN CYCLE = real performance; TOTAL (summed) cycle = work volume only.** The units run in PARALLEL
  (HMX ∥ the 4 HVX threads), so the op's REAL TIME = the BUSIEST domain's `cycles_used` (HMX, and the MAX
  over the 4 HVX threads), NOT their sum. Summing all threads/units gives *work volume* (~4× the HVX
  domain when 4 threads are busy) and over-counts parallel work — quoting it mis-ranks approaches. When
  comparing kernels/algorithms, lead with **domain cycle = max(HMX_cycles_used, max HVX-thread
  cycles_used)**. Two units overlap, so e.g. a matmul whose HVX-glue domain (45k) exceeds its HMX domain
  (32k) is HVX-bound — the HMX∥HVX overlap saves nothing, real time ≈ the HVX domain. (A pure-HVX custom
  op has HMX domain = 0; its real time = the busiest HVX thread.) Cross-ref
  [[feedback_kernel_efficiency_compute_cycles_not_wall]]; worked example `example/gdn_native/solve_op/BENCHMARKS.md`.
- **`data.htp_op_instances`** — per HTP sub-op: booleans **`hmx` / `hvx` / `dma`**, **`cycles_per_packet`**,
  `cycles`, `num_dominant_path_cycles`, `vtcm_read/write`, `dram_read/write`. A custom op shows as many
  instances (its tiles). `cycles_per_packet` is the key efficiency number — compare to native ops.
- **`data.dominant_path_htp_0`** — the actual critical path as an ordered list of HTP ops (where the
  wall really goes).

**But the summary is DERIVED — the raw `chrometrace.json` trace timeline is the ground truth.** The QHAS
summary gives aggregates/averages (total cycles, mean cyc/pkt, instance counts) and can hide what
actually happened. To know whether N op-instances ran in *parallel* or *serial*, where the gaps are, and
where the real wall went, read the raw `ph=X` events' **timeline** — `ts`, `dur`, and the track
(`pid`/`tid`) — and check overlap:
```python
g=[e for e in ev if e["ph"]=="X" and "MyOp" in e["name"]]
span=max(e["ts"]+e["dur"] for e in g)-min(e["ts"] for e in g)
busy=sum(e["dur"] for e in g)               # sum >> span  ⇒  instances overlap (parallel)
# sweep ts/(ts+dur) for max-concurrent to see the real thread count
```
e.g. GdnSolve: summary says "24 instances"; the trace shows them **overlapping ~4-wide** (busy 23.5M ≫
span 3.6M, max-concurrent = 4 HVX threads) ⇒ genuinely parallel, wall ≈ span not sum. The summary alone
could not have told you that. (Gotcha: each event is duplicated on a process-track and a thread-track —
de-dup or halve the concurrency count.)

Example (GdnSolve HVX op): HMX util 2.5% (idle — it's HVX, not matmul), 4 HVX threads at 59–80%, op
tiled into **24 HTP instances**, **6.9 cyc/packet** vs native `mul_op` 2.5 / `L2Norm` 3.5 — i.e. the
kernel has ~2–3× packet-efficiency headroom (acc-chain + scalar dequant/quant stalls), invisible at the
per-op-cycles level. This is the layer where kernel optimization decisions get made. Pairs with
[[feedback_perf_gap_must_check_trace]]: never claim "the gap is in X" without the packet count + cyc/pkt
from here.

## Reading it

- **Same op appearing N times** in the DPC list = the central tiler split it into N tiles (parallelism
  is real). Their DPC may be partly sequential on the critical path; the graph **wall µs** is the truth.
- DPC-sum ≈ wall only when ops are sequential on the dominant path; don't assume sum = wall.
- To test if a custom op *self-parallelizes at all*, isolate it (`A→Op→T` standalone graph, no
  downstream consumer) and read its wall; or probe `qhpi_num_slices` by writing it into an output elem.

## Pitfalls (each cost real time)

- Reading qnn-profile-viewer per-op `(cycles)` as wall → it's aggregate. Use DPC or wall µs.
- `cat`-ing the binary `qnn-profiling-data_0.log` over ssh corrupts it → pull with `tar`.
- Running decode without `source scripts/env.sh` → `QNN_SDK_ROOT not set`.
- `.venv/bin/python` is repo-root-relative → use the absolute venv path when cwd ≠ repo root.
- ctxgen without `--profiling_option optrace` → no `*_schematic.bin` → decoder gives
  "No Valid Input Schematics".
- Cold first-inference includes load/warmup; compare warm runs, `--perf_profile burst`.

## Related

- `qnn-optrace-svg` — render `chrometrace_htp.json` as a fixed-layout SVG (diagram, not numbers).
- `example/qnn_matmul_profile/profile_all.sh` + `parse_chrometrace.py` — a full multi-config profiler.
- `scripts/decode_qnn_optrace.py` — the log+schematic → chrometrace decoder used above.
