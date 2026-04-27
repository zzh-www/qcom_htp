---
name: V8 C8 framework complete (untile + shape sweep) 2026-04-27
description: V8C8 path now has full graph chain BbbKMajor → UntileToRowMajor → Reshape, output is row-major rank-3 [1,M,N] u8. Shape sweep 32³→1024³ all pass with consistent 5-node lowered graph. 2048³ fails ctxgen at err 1002 (12.6 MiB > VTCM ~6 MiB) — multi-instance graph split needed for ≥2048³, deferred until kernel body is wired. Bias path was completed earlier today (native fold layout); now kernel body is the only remaining blocker.
type: project
---

See **`Agent/v8_c8_bias_native_fold_2026-04-27.md`** for the bias work that preceded this.

## What this round added

After the bias-fold work landed, the remaining "framework alignment" items
from the session pickup plan were Output untile + Shape adaptive. Both done:

### 1. Output untile chained in the ONNX graph

`gen_v8c8_test.py` now emits a 3-node ONNX:
```
act_raw → BbbKMajor (tile-out) → UntileToRowMajor → Reshape → out [1,M,N]
```

Reused the existing `UntileToRowMajor` op from V8 production (kernel +
registration already in place under `register_untile_to_rowmajor_op`).
Sigs line up: BbbKMajor outputs `Flat4 + TCM_Only`, untile expects same;
untile produces `Flat4 + DDR_OR_TCM`, becomes graph output.

QNN compiler **absorbs the Reshape** entirely. Lowered graph at every
shape is exactly **5 nodes**:

```
q::*InputSlice  → q::ForceFormat_Crouton  → q::ConvLayer.opt.weights_to_vtcm
                → BbbKMajor               → UntileToRowMajor
```

That's *more compact than native ConvLayer* (typically 8 nodes) because:
- Combined wt+bias static eliminates separate `bias_to_vtcm` and second `weights_to_vtcm`
- UntileToRowMajor produces row-major directly → no `q::ForceFormat_Flat` inserted
- Reshape collapsed during graph-opt

### 2. Shape sweep 32³ → 1024³ (single instance)

| Shape | Lowered nodes | Device | Output size | Marker |
|------:|:-------------:|:------:|:-----------:|:------:|
| 32³   | 5 | OK | 4096 B (32·32·4) | OK |
| 64³   | 5 | OK | 16384 B          | OK |
| 128³  | 5 | OK | 65536 B          | OK |
| 256³  | 5 | OK | 262144 B         | OK |
| 512³  | 5 | OK | 1048576 B        | OK |
| 1024³ | 5 | OK | 4194304 B        | OK |
| 2048³ | — | FAIL ctxgen err 1002 | — | — |

Same 5-node lowered structure at every working shape — graph topology
is **shape-invariant** for single-instance V8C8.

### 2048³ failure mode (expected)

`Operator named HmxMatMulPhase3Package::BbbKMajor (0x11) not sufficiently tiled to fit in TCM. Requires 12648448 bytes`

Single-instance VTCM budget at S³ ≈ `act(S²) + combined_static(S²+8S) + tile_out(S²)` → ~3·S² bytes. At 2048³ that's 12 MiB; VTCM usable is ~6 MiB. **By design**: the V8 production handles ≥2048³ by graph-splitting into multiple BbbKMajor instances with smaller M_TILE/N_TILE (gen_v8_graph.py recipe). Will port that when wiring real kernel — orthogonal to today's framework alignment.

## Files & runners

- `gen_v8c8_test.py` — ONNX gen, 3-op chain, M/K/N args, persists wRaw/bias_q/scale npy.
- `run_v8c8_phase2.sh` — per-shape device test; reads `M=` / `K=` / `N=` / `OUT_DIR=` env. Decode now handles row-major rank-3 fp32 dequant.
- `sweep_v8c8_shapes.sh` — wraps run_v8c8_phase2.sh over `SHAPES="..."`; emits `phase1_validation/v8c8_sweep_summary.txt`.
- No C++ / XML changes needed (UntileToRowMajor was already registered for V8 production).

## State of the V8 C8 alignment program

Phase 1 (sig + ForceFormat_Crouton):                ✅
Phase 2 (combined wt+bias static, err 6006 fix):    ✅
Phase 2.1 (native bias-fold layout in VTCM):        ✅ (this morning)
Phase 2.2 (output untile, row-major user output):   ✅ (this round)
Phase 2.3 (shape sweep 32³–1024³ single instance):  ✅ (this round)
Phase 3   (real HMX kernel body):                   ⏳ — next
Phase 3.1 (multi-instance for ≥2048³):              ⏳ — port from gen_v8_graph.py recipe
Phase 4   (bit-exact validation vs native):         ⏳

## Reproduce

```bash
cd example/hmx_matmul_phase3
EXTRA_DEFS=-DV9_C8_ALIGNMENT_TEST bash build.sh
EXTRA_DEFS=-DV9_C8_ALIGNMENT_TEST bash build_x86.sh

cd standard_flow/phaseB_v8
# single shape
M=512 K=512 N=512 OUT_DIR=/tmp/v8c8_512 bash run_v8c8_phase2.sh
# sweep
SHAPES="32 64 128 256 512 1024" bash sweep_v8c8_shapes.sh
```
