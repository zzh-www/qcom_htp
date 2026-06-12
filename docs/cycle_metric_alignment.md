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
