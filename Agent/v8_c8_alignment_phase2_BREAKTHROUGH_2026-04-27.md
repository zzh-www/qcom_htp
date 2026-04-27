---
name: V8 C8 alignment Phase 2 — full native-style auto-insertion path WORKING 2026-04-27 PM
description: Cracked weights_to_vtcm + ForceFormat_Crouton auto-insertion. Device runs cleanly with the complete ConvLayer-style lowering chain (InputSlice → ForceFormat_Crouton → weights_to_vtcm → BbbKMajor). Two key insights — (1) the native weight layout decoded exactly from ctx-binary diff, (2) the bias/scratch statics need MemLoc_DDR_OR_TCM to suppress a malformed auto-DMA descriptor. Replicates how native ConvLayer feeds Crouton-blocked weights.
type: project
---

# V8 C8 alignment Phase 2 — BREAKTHROUGH (2026-04-27 PM)

User directive: 正面了解 q::ConvLayer.opt.weights_to_vtcm + 替代之法。

## TL;DR — what works now

256³ test ONNX: `act_raw → BbbKMajor(act, wt, bias, scratch) → out`.

ctxgen lowers to **4 nodes** (mirrors native ConvLayer's pattern):

```
q::*InputSlice → q::ForceFormat_Crouton → BbbKMajor
                 q::ConvLayer.opt.weights_to_vtcm (for wt only)
```

Device run: **"Finished Executing Graphs"**, no err 6006, no SSR. NOOP-body kernel writes a 16-byte marker; output 65536 bytes preserved (FP32-dequantized to 262144 bytes by qnn-net-run).

## The decoded native ConvLayer weight layout

Diffed `phaseA_native/s256_w8a8/ctx/matmul_native_ctx.bin` (with random-weight regenerated for disambiguation). Source weight `[K, N]` int8. Dest blob 65536 bytes. **Bit-exact 65536/65536**:

```
dst[(k_tile * N_tiles + n_tile) * 1024  +  (r // 4) * 128  +  c * 4  +  (r % 4)]
  = src_KN[k_tile * 32 + r,  n_tile * 32 + c]

K_tiles = K/32, N_tiles = N/32
1024-byte tile = 32×32 weight block, stored as 8 row-groups × 32 cols × 4 row-within-group
Outer order: K_tile (slow), N_tile (fast)
```

For 256³: 8×8 = 64 tiles × 1024 bytes = 65536 ✓.

## The bisect: which static breaks weights_to_vtcm?

| inputs | static MemLoc | ctxgen | device |
|---|---|---|---|
| 1: act only | TCM_Only | InputSlice + BbbKMajor | ✓ Finished |
| 2: act + wt | TCM_Only | InputSlice + 1× weights_to_vtcm + BbbKMajor | ✓ Finished |
| 3: act + wt + bias | TCM_Only | InputSlice + 2× weights_to_vtcm + BbbKMajor | ✗ err 6006 |
| 3: act + wt + bias | bias=DDR_OR_TCM | InputSlice + 1× weights_to_vtcm + BbbKMajor | ✓ Finished |
| 4: full + Crouton_8 in[0] | wt TCM_Only / bias+scratch DDR_OR_TCM | InputSlice + ForceFormat_Crouton + 1× weights_to_vtcm + BbbKMajor | **✓ Finished** |

**Conclusion**: `q::ConvLayer.opt.weights_to_vtcm` builds a DMA descriptor specialized for ConvLayer-style WEIGHT tensors (shape `[1, K_t, N_t, 1024]` with last-dim 1024). For our **wt** with that shape it works perfectly. For **bias** (`[1, 1, 8, 128]` u16 = 2KB) the descriptor sizes are wrong → out-of-bounds reads on the skel side → err 6006.

The fix is **memloc-decouple**: keep wt in TCM_Only (use the auto-DMA), use DDR_OR_TCM on bias and scratch (suppresses auto-insertion entirely; we read them from DDR ourselves at runtime, or re-add a manual `q::*ToVtcmCache` later).

## The QHPI sig that unlocks it

```cpp
static QHPI_Tensor_Signature_v1 sig_inputs_v9[] = {
    {QHPI_QUInt8,  QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},   // act → triggers ForceFormat_Crouton
    {QHPI_QUInt8,  QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},   // wt  → triggers weights_to_vtcm (works for [1, K_t, N_t, 1024] shape)
    {QHPI_QUInt16, QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_DDR_OR_TCM}, // bias → suppress auto-DMA (would 6006)
    {QHPI_QUInt8,  QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_DDR_OR_TCM}, // scratch → same
};
```

ONNX shapes:
- act_raw `[1, M/32, 32, K]` (Crouton_8 logical shape; QNN's ForceFormat_Crouton converts the runtime row-major bytes for us)
- wt_flat `[1, K_t, N_t, 1024]`, **bytes pre-packed** with the decoded layout above
- bias `[1, 1, N/32, 128]` u16 (V8 production format unchanged)
- scratch `[1, 1, 1, 2048]`

## Other RE findings

- `q::ConvLayer.opt.weights_to_vtcm` string lives in `libQnnHtp.so` / `libHtpPrepare.so` / `libQnnHtpPrepare.so` (host + ARM Prepare libs). **Zero occurrences in `libQnnHtpV75Skel.so`** — confirms it's a prepare-time DMA descriptor builder, not a runtime DSP kernel. The descriptor is serialized into the ctx-binary and replayed at execute time.
- The 7 `.data.rel.ro` references found in libHtpPrepare.so are cost-function tables (`hnnx::cost_func_from_str` polynomial scorer at 0x0f1aef0 / 0x0f1a360 fallback), not the kernel/descriptor builder.
- The error string `"Internal error handing: Dma execution failed on the skel side. result = %d transport error = 0"` lives only in `aarch64-android/libQnnHtp.so` (printed when FastRPC propagates an error from skel).

## Files changed

- `src/HmxMatMulV9SkelOp.cpp` — Crouton_8 + Indirect on in[0], DDR_OR_TCM on bias/scratch (under `V9_C8_ALIGNMENT_TEST`).
- `standard_flow/phaseB_v8/MatMulV8Package.xml` — BbbKMajor in[0] shape `[1, M/32, 32, K]`; in[1] shape `[1, K/32, N/32, 1024]`.
- `standard_flow/phaseB_v8/gen_v8c8_test.py` — pre-pack weight bytes via decoded formula.
- `standard_flow/phaseB_v8/gen_out/.../ConverterOpPackage.cpp` — BbbKMajorShapeInference updated for new dim layout.
- `build_x86.sh` — added `${EXTRA_DEFS:-}` so flags propagate to the host-side metadata .so.
- `Agent/qnn_re/weights_to_vtcm_RE_2026-04-27.md` — RE log of static analysis (cost tables found, descriptor builder unfound).
- `Agent/v8_c8_alignment_phase2_2026-04-27.md` — earlier failed-attempt log.
- `Agent/v8_c8_alignment_phase2_BREAKTHROUGH_2026-04-27.md` — this file.

## Next steps to take this to production

1. **Wire up bias + scratch DDR loading inside the kernel** — they're currently DDR_OR_TCM (in DDR for now). Either (a) HVX-copy them into a VTCM scratch at kernel entry, or (b) keep them in DDR and have the HMX kernel fetch via `mxmem(addr_in_ddr)`-style instructions.
2. **Implement BbbKMajor's HMX kernel body** with proper Crouton_8 act unpacking — the kernel sees activation in Crouton-blocked layout (because ForceFormat_Crouton runs before us), and weight in tile-major layout (because weights_to_vtcm DMA put it there).
3. **Bit-exact validation** vs V8 production output on the same shape.
4. **Perf comparison** vs V8 production on 256³…4096³ — with HVX unloaded from pack ops, hopefully closer to native ConvLayer speeds.

## Not-to-redo learnings

- Do NOT chain experiments after 2-3 zero-improvement results in the SAME direction — this morning's pure layout-permutation probes were a waste. Switching to ctx-binary diffing was the move.
- The "static input + raw act + single-op" topology is fundamentally different from V8 production's multi-op pipeline. Auto-insertions reason about the consumer chain, not just the immediate static.
