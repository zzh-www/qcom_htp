---
name: Route 1 Session Summary 2026-04-25 — final state
description: End-of-session retrospective. dlsym validated, PackActCrouton production-ready, pack_act_rm vshuff rewrite SUCCEEDED (bit-exact, 1.26-1.68× per-instance). V9 graph total speedup ~1-10%. mmv8 dominates at large shapes; N_TILE tuning marginal.
type: project
---

# Route 1 wholesale-copy QNN MatMul — 2026-04-25 final state

## What landed in the codebase

### 1. dlsym mechanism — empirically validated cross-.so

Custom op-pkg .so can declare `extern "C"` for libQnnHtpV75Skel.so GLOBAL
exports; toolchain emits `R_HEX_JMP_SLOT`, DSP loader resolves at load.
Confirmed by V8 byte-identical with vs without an embedded skel call.
Sub-agent caveats about FastRPC isolation refuted. ⇒
`Agent/dlsym_spike_PASS_2026-04-25.md`.

### 2. PackActCrouton — `convert_to_crouton_b` wrapper

`example/hmx_matmul_phase3/kernel/pack_act_crouton_skel.c`. Bit-exact at
13 (M, K) shapes (M=32..1024, K=128..4096). Op registered, ONNX/DLC/ctx
flow tested. ⇒ `Agent/pack_act_crouton_skel_2026-04-25.md`. Discovered:
`aux=16` for h-stride=128, `channel_groups=4` constant (not K/32),
`height_tiles ≤ 16` per call (chunked dispatch for M_grp ≥ 16).

### 3. pack_act_rm_hvx rewrite with vshuff(32, 64) — **PRODUCTION**

`example/hmx_matmul_phase3/kernel/pack_act_rm_hvx.c` updated. Was
vmux+vror+vor (12 HVX ops/tile), now 2-pass vshuff with single-bit Rt
(32 then 64), matching the topology already in `pack_wt_v3_hvx.c` (~3
HVX ops/tile). **Bit-exact** — md5 V8 4096³ output identical to golden,
0 diff bytes.

**Per-instance pack_act speedup**:

| Shape | Baseline cyc/inst | vshuff cyc/inst | Speedup |
|-------|------------------:|----------------:|--------:|
| 512³  |        96,436    |       57,499    | 1.68×   |
| 1024³ |        84,349    |       52,706    | 1.60×   |
| 2048³ |       123,238    |       74,741    | 1.65×   |
| 4096³ |       565,842    |      450,351    | 1.26×   |

**Total V9 graph speedup** (whole-graph cycles, 3 inferences avg):

| Shape | Baseline cyc | vshuff cyc | Speedup |
|-------|-------------:|-----------:|--------:|
| 512³  |     540,407  |    489,258 | 1.10×   |
| 1024³ |   3,077,876  |  2,970,310 | 1.04×   |
| 2048³ |  21,124,549  | 20,527,874 | 1.03×   |
| 4096³ | 161,528,782  |159,757,726 | 1.01×   |

Modest total because pack_act is only 4-22% of total cycles (the rest
is mmv8 / slice / concat).

### 4. RE artifacts (knowledge persisted)

- `Agent/sig_convert_to_crouton_b_2026-04-25.md` — descriptor + outer loop
- `Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md` — 3-descriptor architecture
- `Agent/qnn_skel_primitive_symbols_2026-04-25.md` — symbol catalog
- `Agent/v9_matmul_ctxgen_segfault_2026-04-25.md` — MatMulV9 blocker doc

## Key insight: where the QNN-vs-V9 gap actually lives

**At 4096³, mmv8 = 90.6% of cumulative cycles.** Pack_act is 4.6%.
So pack_act optimization caps at ~5% total speedup — not a closing-the-gap lever.

mmv8 cyc/cell @ 4096³ = 25.5 (V9) vs ~5.4 (QNN). 4.7× HMX work itself.

N_TILE sweep at 4096³ (vshuff baseline 78.2-80.2 ms wall):
- N_TILE=64:  78.2 ms (1024 inst, spill 31MB / fill 237MB)
- N_TILE=128: 78.8 ms (512 inst, spill 28MB / fill 271MB)
- N_TILE=256: 80.2 ms (256 inst, spill 63MB / fill 452MB)

**Only ~3% spread.** Smaller tile reduces spill but adds per-instance
overhead; the two cancel. So tile-size tuning isn't the lever either.

## Where the lever IS (next session)

Per QNN's HMX silicon ceiling 5.4 cyc/cell, V9's mmv8 is 4.7× above
ceiling. Possible causes:

1. **HMX-HVX overlap gap** — QNN runs pack_act on HVX in parallel with
   HMX MAC; V9 has the ops marked `multithreaded=true` but the
   serialization may not be as tight. Profile chrometrace to verify.
2. **mmv8 inner loop not actually at ceiling** — measure single-instance
   cycles for known shape, divide by MACs, compare to 5.4.
3. **Per-instance setup overhead** — bias load, mxclracc, address
   computation. Each instance has ~200K cyc fixed cost. With 256-1024
   instances, that's 50-200M cyc fixed overhead on top of the MAC work.

Best next-session bets:
- **Reduce per-instance overhead in mmv8** (e.g., pre-bake act/wt ptr
  tables, cache bias across mt iterations more aggressively)
- **Resolve MatMulV9 (BbbKMajor) ctxgen segfault** to unlock direct
  use of `hmx_convbbb1x1_stride1` which has QNN's pre-baked descriptor
  scheme built in
- **Profile chrometrace** for HMX/HVX overlap analysis to see if
  V9's parallel scheduling actually achieves what QNN does

## Code state

| File                                      | Status                            |
|-------------------------------------------|-----------------------------------|
| `kernel/pack_act_rm_hvx.c`                | NEW vshuff(32,64) path, bit-exact |
| `kernel/pack_act_crouton_skel.c`          | NEW, PackActCrouton, bit-exact    |
| `kernel/pack_act_crouton_hvx.c`           | Sub-agent B's HVX impl (unwired)  |
| `kernel/crouton_pack_spike_hvx.c`         | dlsym test op (kept)              |
| `src/HmxMatMulV9SkelOp.cpp`               | BbbKMajor op, ctxgen-blocked      |
| `standard_flow/phaseB_v8/MatMulV8Package.xml` | + 3 op defs                  |
| `gen_v8_graph.py`                         | + `--n-tile` override flag       |
| `gen_pack_act_crouton_test.py`            | NEW sweep tester                  |
| `test_pack_act_crouton.sh`                | NEW end-to-end test               |
| `gen_v9_test.py`, `gen_v9_no_mm.py`,
  `gen_spike_onnx.py`                      | NEW test ONNX gens                |
| `htp_backend_ext.json`                    | + v9_model/v9_no_mm/v9_test names |
| `sweep_v9_baseline/`                      | Baseline (vmux+vror+vor) results  |
| `sweep_v9/`                               | New (vshuff) results              |
| `ntile_sweep/n{64,128,256}_4096/`         | N_TILE tuning runs                |

V8 / V9 graph still produces byte-identical golden output. Production
op-pkg on device (`~/qnn_run/libQnnHmxMatMulPhase3_htp.so`) is the
vshuff variant.
