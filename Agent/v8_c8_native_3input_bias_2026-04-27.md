---
name: V8 C8 native-aligned 3-input bias (separate static, auto-DMA, no combined-static workaround) 2026-04-27
description: Replaced the combined wt+bias workaround with a proper 3-input op signature (act + wt + bias), bias is now a separate STATIC tensor with native shape [N] int32. QNN auto-inserts a typed weights_to_vtcm DMA for it. 6-node lowered graph at every shape 32³–1024³. Architecture mirrors native; the FOLD remains a kernel-body responsibility (QHPI v1 has no "bias slot" role to trigger native's q::ConvLayer.opt.bias_to_vtcm fold dispatcher — that's reserved for q::ConvLayer_s1.opt).
type: project
---

# Why this revision

The previous "combined wt+bias static" was a **workaround** for the slot-2 dispatch trap (err 6006). That workaround folded bias bytes into the weight static so we could ride the slot-1 weights_to_vtcm path. Functionally OK but architecturally wrong — bias was no longer a separate input.

User explicitly asked: don't use the workaround, properly use bias_to_vtcm like native.

# Discovery process

## What the previous session tried for slot-2 (all failed err 6006)

- bias as `[1,1,8,128]` u16 (V8-prod fp16-pair format)
- bias as `[1,1,2,1024]`
- bias as `[1,8,8,1024]` (matching wt)
- Layout_Any vs Layout_Flat4
- TCM_Only / DDR_OR_TCM / DDR_Only

**Never tried**: `[N]` int32 — which is what `qairt-dlc-info` shows native actually uses (`B sFxp_32 [256] STATIC`).

## What worked

3-input sig with bias as `[N]` int32 + `QHPI_Int32 + Layout_Any + Direct + TCM_Only`:

```cpp
static QHPI_Tensor_Signature_v1 sig_inputs_v9[] = {
    {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
    {QHPI_Int32,  QHPI_Layout_Any,       QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
};
```

Critical: `QHPI_Int32` (raw signed int32), NOT `QHPI_QInt32` (quantized). The latter caused `Op preparation failed with err:-1` because QNN's compiler delivers raw `Int32_TCM` for an int32 STATIC tensor.

# Lowered graph (6 nodes at every shape 32³–1024³)

```
q::*InputSlice                                               ← act DDR slice
q::ForceFormat_Crouton                                       ← act → Crouton_8
q::ConvLayer.opt.weights_to_vtcm@FB.fB.                      ← wt u8 (slot-1 byte variant)
q::ConvLayer.opt.weights_to_vtcm@Fi.fi.                      ← bias i32 (slot-2 int32 variant)
HmxMatMulPhase3Package::BbbKMajor                            ← our op (3 inputs)
HmxMatMulPhase3Package::UntileToRowMajor                     ← row-major output
```

vs **native** (8 nodes for the same 256³ MatMul):
```
q::*InputSlice
q::ConvLayer.opt.bias_to_vtcm        ← THE FOLD DISPATCHER (different from weights_to_vtcm)
q::Reshape                           ← reshapes folded bias [N] → [N_t, 256]?
q::ConvLayer_s1.opt                  ← native compute
q::ForceFormat_Crouton
q::ForceFormat_Flat
q::Reshape
q::ConvLayer.opt.weights_to_vtcm
```

# The fundamental limitation we hit

**Only `q::ConvLayer_s1.opt` triggers `q::ConvLayer.opt.bias_to_vtcm`** (the weight-aware fold dispatcher). For our custom op, slot-2 dispatches to the generic typed `weights_to_vtcm@Fi.fi.` (raw int32 verbatim DMA) — no fold, no native byte layout.

Why: QHPI v1 `Tensor_Signature` has only `(element_type, layout, storage, mem_placement)` — no role/kind field for "this is bias". The compiler can't know slot-2 of our `BbbKMajor` is a bias that needs the fold. Native ConvLayer's bias-recognition lives inside `q::ConvLayer_s1.opt`'s OWN metadata, which we can't access (`qhpi_op_create("q::*")` rejects internal prefix names — confirmed by previous session's RE).

# What this means for VTCM bytes

- **Native**: bias_to_vtcm produces 256 B/N-tile = `(fp16 scale, fp16 baseline) × 32 || int32 (-act_zp × Σwt + bias_q) × 32`. HMX kernel reads both halves directly.
- **Ours**: weights_to_vtcm@Fi.fi. produces raw int32 [N] (4 B per channel, no fold). **HMX kernel must do the fold itself** at runtime — either compute Σwt on-chip or pre-baked elsewhere.

# What this fix achieves

- ✅ **Architectural alignment with native** — bias is its own STATIC tensor; no combined-static hack.
- ✅ Separate auto-DMA per input, dispatched by type variant (`@FB.fB.` for u8 wt, `@Fi.fi.` for i32 bias).
- ✅ ONNX schema matches native (`bias` as int32 [N] STATIC).
- ✅ All shapes 32³ → 1024³ pass with 6-node lowered graph.
- ⚠️ **The fold is now kernel responsibility** — no longer pre-baked in gen-script (option still available if simpler).

# Files changed

- `src/HmxMatMulV9SkelOp.cpp` — V9_C8_ALIGNMENT_TEST sig: 3 inputs (was 2). Bias entry uses `QHPI_Int32 + Layout_Any + Direct + TCM_Only`. Comment block updated to describe the 3-input form.
- `standard_flow/phaseB_v8/MatMulV8Package.xml` — BbbKMajor OpDef: 3 inputs declared (act 4D, wt 4D, bias 1D `[N]`). Supplemental dtype: `QNN_DATATYPE_INT_32` and `QNN_DATATYPE_SFIXED_POINT_32` for in[2].
- `standard_flow/phaseB_v8/gen_out/.../ConverterOpPackage.cpp` — `BbbKMajorShapeInference` now requires `numOfInputs >= 3`. Comment block describes the 3-input layout. **NOTE**: outer copy at `ConverterOpPackage/ConverterOpPackage.cpp` is what the runner loads; the inner copy at `ConverterOpPackage/ConverterOpPackage/ConverterOpPackage.cpp` is the Makefile source. They were synced manually — should clean this up later.
- `standard_flow/phaseB_v8/gen_v8c8_test.py` — emits separate `wt_packed` (u8 [1, K_t, N_t, 1024]) and `bias` (int32 [N]) initializers; no combined static. Removed `build_native_fold_bias()` (fold no longer host-side; kernel responsibility now).
- `standard_flow/phaseB_v8/quant_overrides.json` — added `bias` entry under `param_encodings`.

# Sweep result

```
shape   ctx_nodes   device   out_size    marker
32³     6           OK       4096        OK
64³     6           OK       16384       OK
128³    6           OK       65536       OK
256³    6           OK       262144      OK
512³    6           OK       1048576     OK
1024³   6           OK       4194304     OK
2048³   FAIL ctxgen err 1002 (single-instance VTCM ceiling, unchanged)
```

Marker decode at every shape: `0xA5, 9 (Crouton_8), 2 (Flat4), 3 (num_inputs), 4 (rank), [1, M_t, N_t, ...], 0x5A`. Critical: `num_inputs=3` confirms 3-input sig is active.

# State + next steps

V8 C8 framework is now structurally aligned with native (3 separate inputs + separate auto-DMAs + output untile + row-major reshape). Next is HMX kernel body, with one extra design decision now baked in:

**Bias fold strategy choice (kernel-side):**
- (a) Read raw int32 [N] bias from VTCM, compute `effective_int32[c] = -ACT_ZP × Σ_k W[k,c] + bias_q[c]` on-chip (HVX scalar accum over N channels) before HMX MAC loop. Negligible cost (256 multiply-adds for 256³ vs 16M MACs).
- (b) Pre-bake fold in gen_v8c8_test.py and ship as folded bytes via int32 [N_t * 64] (reshape to fit). Simplest kernel.

(a) is cleaner architecturally because gen-script doesn't need to know weights. (b) keeps host responsibility but conflicts with the goal of a clean separate-bias input. Recommend (a).
