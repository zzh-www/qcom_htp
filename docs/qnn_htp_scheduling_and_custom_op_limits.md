# QNN HTP Scheduling Model & Custom‑Op Concurrency Limits

How the QNN HTP backend (v75) schedules ops onto the HVX and HMX units, why HVX∥HMX
overlap happens for **native** ops but **never** for **custom (QHPI/plugin)** ops, and what
that means for designing custom kernels.

Derived from: reverse‑engineering `libQnnHtp.so` (x86 graph compiler) and
`libQnnHtpV75Skel.so` (on‑device executor); device measurements on real v75 (`ssh oneplus`);
and the public headers `tools/qnn-sdk/include/QNN/HTP/core/qhpi.h` and `.../QnnHtpGraph.h`.

---

## TL;DR

1. **HTP scheduling is compile‑time static.** The graph compiler bakes the entire execution
   order into **per‑unit runlists** (vector=HVX, matrix=HMX, eltwise) inside the context
   binary. The device just **replays** them in fixed order. There is **no runtime
   dataflow dispatch** — an op does not "start when its inputs are ready."
2. **HVX∥HMX overlap exists, but only inside compiler‑built `supertile` fusion regions**,
   synchronized by compile‑time‑inserted DMA checkpoints.
3. **Custom QHPI/plugin ops are excluded from supertiling** by an explicit compiler predicate
   (`is_plugin_op`). A custom op is therefore a **serial barrier**: it never overlaps an HMX
   op, and a single custom op anywhere in a producer→consumer chain breaks fusion **even into
   a native HMX consumer**.
4. **Measured (real v75, O3, 4 HVX threads):** native op graphs overlap HVX∥HMX **17–22 %**;
   every custom‑op configuration overlaps **0 %**. It is *not* "v75 can't overlap" — it is
   specifically the plugin exclusion.
5. The only HVX∥HMX concurrency reachable from a custom op is **manual `qurt` worker threads
   inside one op** (HMX on the main callback thread, HVX work on workers), capped ≈2.5×.

---

## 1. Hardware & thread model (v75)

The HTP has independent execution units, surfaced in optrace as distinct thread IDs:

| Unit | optrace tid(s) | Runs |
|---|---|---|
| HMX (matrix) | `256` | matmul/conv MAC engine (`q::ConvLayer*`, `q::*MatMul*`) |
| HVX (vector) | `512`–`515` | vector ops, layout (`ForceFormat_Crouton`), requant, custom HVX kernels |
| (scalar/eltwise) | varies | small elementwise, `$Shape`, `$Const` |

The units *can* run concurrently in hardware. Whether they *do* is decided entirely by the
compiler (Section 2–4), not by the device at runtime.

---

## 2. The execution model: static, compile‑time runlists

**Prepare phase — x86 `libQnnHtp.so` (the graph compiler).** Compiles the graph into a
static schedule and bakes it into the context binary. Observed symbols/strings:

- `GraphPrepare::schedule_for_alloc`, `show_runlist`/`dump_runlist`,
  `python_pprint_vec_runlist` / `_mtx_runlist` / `_elt_runlist` — the schedule is split into
  **separate runlists per unit** (vector / matrix / eltwise).
- `linear order`, `Update linear order`, `Use position in depth first topological sort as
  metric`, `Number of nodes moved in linear order` — a single linearized topological order.
- `Serializer::before_runlists`, `make_runlist_segment_descs` — runlists serialized into the
  pickle.

**Execute phase — device `libQnnHtpV75Skel.so` (the executor).** Replays the runlists.
Observed symbols/strings:

- `Graph::compile_exec_list(RunListSet&, ListType)`, `Graph::setup_runlists`,
  `Graph::exec_vec_worker` / `exec_mtx_worker` / `exec_elt_worker` / `exec_bkgrnd_worker`,
  `Started %d vec workers, %d matrix workers, %d eltwise workers`,
  `VXU %d: Num HVX threads / Num HMX threads`.
- **No** `dispatch` / `ready_queue` / `scoreboard` / `dataflow` / work‑steal symbols exist.

⇒ Each unit has a worker thread that runs **its** runlist **in order**. There is no dynamic
scheduler that notices "op B's inputs are ready, run it now on the idle HMX unit."

**Cross‑unit synchronization** is also baked in at compile time, via DMA checkpoint tags:

- `GraphPrepare::make_dma_checkpoint_op`, `@DmaCheckpointSet` / `@DmaCheckpointWait`,
  `@Spill` / `@Fill`, `rewrite_op_for_spillfill`. A producer records a completion tag; a
  consumer waits on it. (Device errors `Checkpoint info not found for HVX thread op`,
  `non-zero checkpoint ... may indicate a race condition` confirm this is the handshake.)

---

## 3. How HVX∥HMX overlap actually happens

Two compiler mechanisms, both decided at prepare time:

### 3.1 Supertile fusion (the main one)
The compiler fuses a tiled producer→consumer (or a single large tiled op) into a **supertile**
streamed region: consumer `tile[i]` runs as soon as producer `tile[i]` is done, overlapping
producer `tile[i+1]` — and if the two tiles land on different units, that is HVX∥HMX overlap.
Symbols: `GraphPrepare::create_supertiles`, `make_one_supertile`, `is_SuperTileOp_ptr`,
class `SuperTileOp`; strings `early fuse 0x%llx ...`, `Skip supertile: Layer has few number
of ops`, `ForceFormat_Crouton Free!` / `non-free ForceFormat crouton->crouton`.

Preconditions for a fused/streamed region: ops have tiling **rules**, producer/consumer share
a **crouton** layout, the handoff stays **VTCM‑resident** (so `ForceFormat` is "free").

### 3.2 Background HMX worker
The matrix runlist can run as a **background** worker overlapping the foreground vector worker,
placed by an ML parallelism‑reorder pass. Symbols/strings: `exec_bkgrnd_worker`,
`update_bkgrnd_worker_counts`, `request_bkgrnd_yield`, `continue_execution_bkgrnd_thread`,
`Using ML based parallelism`, `runlist_reorder_for_parallelism_enable`,
`Using runlist reorder num_hmx / num_hvx`, `Can't run HMX in background due to lack of HMX
thread(s)`. No public knob enables it directly; in practice it did **not** engage for a
plugin+native independent pair (Section 6).

---

## 4. The custom‑op limitation: `is_plugin_op`

The compiler carries an explicit predicate that excludes plugin ops from supertiling:

```
no supertiling for 0x%llx %s, util=%f, rules=[%d,%d,%d,%d], is_plugin_op=%d
early fuse 0x%llx set to input ... has_rules: %d, is_plugin: %d
```

Plugin class hierarchy (RTTI): `PluginOpBase` → `PluginOp` / `PluginOpWithCompiler`.
A native op carries tiling **rules** + a compiler, so it can be sliced and fused into a
supertile. A **plugin op has `is_plugin_op=1`** and (unless it is a `PluginOpWithCompiler`
with rules) is **excluded** — it lands in a unit runlist as a plain item outside any streamed
region. Additional plugin restrictions seen: `invalid tiling for multioutput plugin op`,
`invalid output position for plugin operator`.

**Consequences:**
- A custom op never overlaps an HMX op — it is a serialization barrier.
- A plugin **producer** breaks the supertile of a downstream **native** HMX consumer, so even
  `customHVX → nativeMatMul` runs serial.
- The only escapes — `PluginOpWithCompiler` (register tiling rules so the op enters supertiles)
  and the internal background‑HMX path — are **not reachable through the public `qhpi.h`**.

---

## 5. What the public QHPI interface gives you (`qhpi.h`)

| Field / API | Effect | Does it give HVX∥HMX? |
|---|---|---|
| `resources = QHPI_RESOURCE_HVX / HMX / MAIN` | which unit the kernel runs on | no |
| `multithreaded` (self‑slicing) + `qhpi_num_slices/slice_number` | runs the SAME op on N **HVX** threads over slices | **HVX∥HVX only.** Self‑slice on an HMX op is rejected (`Can't set self_slicing on non-HVX op`). |
| `shape_required` / `build_tile` (central tiler) | splits an op into per‑region tile‑ops | tiles the op, but tile‑ops still run in the op's runlist; does **not** put it in a supertile |
| `early_rewrite` / `late_rewrite` | rewrite to a subgraph at prepare time | can fold to constants / native ops (then those natives can supertile) |
| `QHPI_RESOURCE_HVX\|HMX` on one kernel | — | **rejected** at prepare (`invalid resource flag 0x6`) |

Notes:
- Driving HMX from a **spawned worker thread faults** — the backend grants HMX only to the
  **main** callback thread.
- Therefore the only HVX∥HMX concurrency a custom op can achieve is **inside one op, by hand**:
  run the HMX kernel on the main callback thread and spawn `qurt` worker threads for the HVX
  work (`qurt_thread_create` + `qurt_hvx_lock` succeed inside an HMX‑resource callback). This
  is intra‑op manual threading, capped at the measured **≈2.5×** HVX‑worker ceiling (v75 HVX
  context contention) — not graph‑level streaming.

---

## 6. Evidence (device measurements, real v75, O3 + `hvx_threads=4`)

Two distinct overlap metrics, both read from op **timestamps** (tid 256 vs 512–515) — keep them
separate or you will draw the wrong conclusion:

- **custom‑op ∩ HMX** — does the *custom op's own compute* overlap the HMX unit? (the real question)
- **HVX‑unit ∩ HMX** — does *any* HVX op overlap HMX? This also counts a native matmul's **own**
  layout glue (`ForceFormat`, `convert_weights_to_signed`) overlapping its **own** matmul — the
  native within‑op pipeline, which is present regardless of any custom op.

| Graph | Op nature | custom‑op ∩ HMX | HVX‑unit ∩ HMX |
|---|---|---|---|
| u8i8 8‑matmul chain | **all native** (supertiled) | — (no custom op) | **22 %** |
| u8i8 single‑matmul ref | all native | — | 17 % |
| `GdnSolve → native MatMul` (dependent) | plugin producer | **0 %** | 33 % (matmul's own glue) |
| `GdnSolve ∥ native MatMul` (independent, C=64) | plugin + native, no dep | **0 %** | 53 % (matmul's own glue) |
| `GdnSolve ∥ native MatMul` (independent, C=128) | plugin + native, no dep | **0 %** | — |
| `GdnSolveDiag → GdnMergeHmx` (two custom ops) | plugin → plugin | **0 %** | 0 % |

The custom op `GdnSolve` runs on HVX in `[25,256..561,381]`; the HMX matmul runs in
`[588,410..642,797]` — strictly **after**, **0 % intersection**, whether the matmul depends on the
solve or not. Meanwhile the matmul's *own* `ForceFormat` glue (native) does overlap the matmul
(the 33–53 % "HVX‑unit" figure) — proving the native pipeline is alive in the very same graph; the
custom op just cannot be folded into it. Representative timelines (1 char ≈ a fixed cycle slice):

```
ALL-NATIVE u8i8 8-matmul chain  — 22% (native glue ∥ native matmul; mechanism works):
HMX 256 |......MMMMMMMMM.MMM....MMMMMMMM         MMMMMMMMMMMMMMMMMMM   .....|
HVX 512 |                 ####################################             |   <- ForceFormat runs DURING matmuls

CUSTOM GdnSolve(HVX) || native MatMul(HMX), INDEPENDENT  — GdnSolve∩HMX = 0%:
HVX 512 | SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS#M#MMMM#.|   solve...then matmul GLUE
HMX 256 |·····(idle: InputSlice prefetch only)·······················|  |MMMM|   matmul starts after solve
                                                                          ^ overlaps its OWN glue, not the solve
```

The takeaway: the HMX matmul is placed entirely after the custom op finishes — a *scheduling*
decision (the matmul is only ~16–21 K cyc; size is not the reason), exactly as the `is_plugin_op`
exclusion predicts. The custom op's compute gets **0 %** HMX overlap.

---

## 7. Reading overlap correctly (and a warning)

**Read op timestamps, not a derived ratio.** Decode the device profile to a chrometrace and
compare the HMX‑unit (tid 256) op start/end against the HVX‑unit (tids 512–515) span. Genuine
overlap means HMX ops and HVX ops have intersecting `[ts, ts+dur)` intervals.

**Do not trust an `overlap = (S + M − X)/max(S, M)` style formula** built from separate
solve‑only / matmul‑only / combined runs. A matmul that reads a **fresh uint16 activation from
DDR** hits a pathological ~**3 × 10⁹‑cycle** path; if that inflated `M` is used as the
reference, the ratio pins at ≈1.0 regardless of real overlap. (This artifact produced a
spurious "98 % overlap" conclusion that the timestamp method later refuted — real overlap was
0 %.) Keep intermediates on‑chip / use int8 activations to avoid the pathology when isolating a
matmul.

---

## 8. Practical guidance for custom‑op design

- **If a design depends on HVX∥HMX overlap, both sides must be native ops.** Express the whole
  computation in native QNN ops so the compiler can supertile it. A single custom op anywhere
  in the chain serializes it.
- **Custom op intra‑op parallelism is fine and useful**: self‑slice across HVX threads
  (`multithreaded=true`) for HVX∥HVX, or manual `qurt` workers for HVX‑glue while HMX runs on
  main. Do not expect graph‑level HVX∥HMX from splitting into two custom ops.
- **Never conclude perf for a parallel‑dependent approach from a non‑parallel build, and always
  read the real per‑thread timeline before claiming overlap.**
- Graph config knobs that affect scheduling (none of which rescue plugin ops):
  `QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS`, the finalize optimization level
  (`FINALIZE_OPTIMIZATION_FLAG`, "O" = 3 for max), `vtcm_mb`. Set via the netrun/ctxgen backend
  extensions config, e.g. `{"graphs":[{"graph_names":[...],"O":3,"hvx_threads":4,"vtcm_mb":8}]}`.

---

## 9. Reproduce

All on real v75 via `ssh ${DEVICE:-oneplus}`. Each command runs on device, decodes the optrace,
and prints the ASCII timeline + overlap (last line `OVERLAP <pct>%`). No manual post‑processing.

```bash
# Negative (custom op): the probe builds + runs + decodes + measures in one shot.
#   prints custom-op (GdnSolve) ∩ HMX = 0%  and  HVX-unit ∩ HMX (matmul's own glue).
cd example/gdn_native/solve_op/standalone
C=64  B=64 OPT=3 HVXT=4 bash gdn_concurrency_probe.sh                 # solve + independent GdnSolve||MatMul
C=128 B=32 OPT=3 HVXT=4 bash gdn_concurrency_probe.sh
C=64  B=64 OPT=3 HVXT=4 RUN_COMBINED=1 bash gdn_concurrency_probe.sh  # also the dependent chain

# Positive control (all native): 17–22% HVX∥HMX overlap (run_native_chain.sh auto-decodes).
cd example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline
SIZE=256 CHAIN=8 OUT_NAME=ovl_pos_s256_chain8 bash run_native_chain.sh
python ../../../../scripts/gdn_overlap_from_trace.py ovl_pos_s256_chain8   # -> OVERLAP 22%

# Re-analyze any decoded trace (timeline + overlap), optionally isolating a custom op vs HMX:
python scripts/gdn_overlap_from_trace.py <out_dir|chrometrace.json> [--producer GdnSolve]
```

Probe artifacts: `example/gdn_native/solve_op/standalone/gdn_concurrency_probe.sh` (harness, sets
`O`/`hvx_threads`/`vtcm_mb`), `scripts/gdn_overlap_probe.py` (builds the `solve` / `indep_native` /
`combined` ONNX graphs), `scripts/gdn_overlap_from_trace.py` (timeline + overlap from a chrometrace),
`scripts/decode_qnn_optrace.py` (device log → chrometrace). Output dirs (`out_*`, `ov_*`, `*.raw`,
`*.onnx`) are git‑ignored and regenerated by the harness.

---

## 10. References

- RE evidence + the full measurement table: memory
  `project_gdn_hvx_hmx_overlap_impossible_2026-06-03`.
- GDN solve context (where this limit kills the block‑recursive HMX route):
  `Agent/current/gdn_solve.md`.
- Public interfaces: `tools/qnn-sdk/include/QNN/HTP/core/qhpi.h`,
  `tools/qnn-sdk/include/QNN/HTP/QnnHtpGraph.h`.
- Custom‑op authoring SOP: `docs/qnn_custom_op_sop.md`.
- Binaries inspected: `tools/qnn-sdk/lib/x86_64-linux-clang/libQnnHtp.so`,
  `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so`.
