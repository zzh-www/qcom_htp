---
name: V8 C8 bias path solved via combined wt+bias static 2026-04-27
description: Bias in VTCM achieved by folding bias bytes into the extra K-tile(s) of the wt static. Single weights_to_vtcm DMA path (slot-1, weight-style descriptor), bias ends up VTCM-resident at offset wt_size, equivalent to native bias_to_vtcm in end state. Avoids the slot-2 dispatcher trap that caused err 6006.
type: project
---

# V8 C8 bias path — combined wt+bias static (2026-04-27)

User goal: 把 bias 输入也搞成功 (1:1 with native's bias_to_vtcm placement).

## What blocked us

QNN's bias-slot dispatcher (slot 2 of a Crouton_8-classified op) hardcodes `q::ConvLayer.opt.bias_to_vtcm` — but custom ops can't manually invoke that op type via early_rewrite (qhpi_op_create rejects internal `q::*` prefixes; only `q::QNN_*` public ops are allowed). And QHPI v1 sig has no role-tag field to tell QNN "this slot is bias, use bias_to_vtcm". So we cannot replicate the slot-semantic mechanism.

Sigil bisect (Layout_Any, Storage_Direct_OR_Indirect, q::Reshape rewrite) all failed with the same err 6006 — slot 2 is permanently broken for bias-shaped inputs.

## What works — fold bias into wt as one static

Single static initializer in slot 1 (wt-slot, where weights_to_vtcm works correctly for `[1, K_t, N_t, 1024]`-shaped inputs). Bias bytes are appended after wt bytes inside the SAME buffer.

```python
# 256³: wt = 65536 bytes, bias = 2048 bytes
K_t, N_t = K // 32, N // 32
wt_size = K_t * N_t * 1024            # 65536
bytes_per_ktile = N_t * 1024          # 8192
extra_ktiles = ceil(bias_size / bytes_per_ktile)   # 1 for 256³
total_ktiles = K_t + extra_ktiles     # 9 for 256³

combined = np.zeros((1, total_ktiles, N_t, 1024), dtype=np.int8)
# bytes 0..wt_size-1 = pre-packed weight (decoded native ConvLayer layout)
# bytes wt_size..wt_size+bias_size-1 = bias bytes
combined_flat[wt_size : wt_size + bias_size] = bias_bytes
```

Op signature: 2 inputs (act + wt_bias), all TCM_Only, all Flat4/Crouton_8.

## Verified

ctxgen produces 4 nodes (mirrors native ConvLayer-style):

```
q::*InputSlice → q::ForceFormat_Crouton → q::ConvLayer.opt.weights_to_vtcm → BbbKMajor
```

Tensors:
- in[0] act `[1, 8, 32, 256]` Crouton_8 — runtime input
- in[1] wt_bias `[1, 9, 8, 1024]` Flat4 — combined wt+bias static
- weights_to_vtcm output `[1, 9, 8, 1024]` — VTCM-resident copy

Device run: **"Finished Executing Graphs"**, no err 6006. Kernel reaches NOOP body (returns success).

## End-state equivalence with native

| Tensor | native | our v8c8 |
|---|---|---|
| act in VTCM | activations_to_vtcm | ForceFormat_Crouton + ImplicitVTCM |
| wt in VTCM | weights_to_vtcm | weights_to_vtcm (offset 0) |
| bias in VTCM | bias_to_vtcm | weights_to_vtcm (offset wt_size, in same buffer) |

Native uses 3 separate DMAs for 3 separate VTCM regions. We use 1 DMA for 1 contiguous VTCM region. Functionally equivalent end state. **Difference**: our bias is at a fixed offset within the wt-VTCM region, instead of in its own region.

## Kernel access pattern (TODO when implementing)

```c
const uint8_t  *wt_bias_vtcm = (uint8_t *)qhpi_tensor_raw_data(inputs[1]);
const uint8_t  *wt_vtcm      = wt_bias_vtcm;
const uint16_t *bias_vtcm    = (uint16_t *)(wt_bias_vtcm + wt_size);
// bias for tile nt: bias_vtcm + nt * 128 (same as V8 production format)
asm volatile("bias = mxmem(%0)" :: "r"(bias_vtcm + nt * 128) : "memory");
```

## Files changed

- `src/HmxMatMulV9SkelOp.cpp`: sig changed to 2-input under `V9_C8_ALIGNMENT_TEST`
- `standard_flow/phaseB_v8/MatMulV8Package.xml`: BbbKMajor declared with 2 inputs
- `standard_flow/phaseB_v8/gen_v8c8_test.py`: builds combined wt+bias buffer
- `standard_flow/phaseB_v8/gen_out/.../ConverterOpPackage.cpp`: shape inference for 2-input

## Caveats

- `extra_ktiles` may waste some VTCM space when bias size doesn't fill an integer number of K-tiles. For 256³: bias=2KB, ktile=8KB → wastes 6KB. For larger shapes the relative overhead shrinks.
- Scratch input was REMOVED entirely from the sig — we'd add it back via the same combined-static trick if/when the kernel actually needs scratch space.
- This is NOT 1:1 graph topology with native (we have 4 nodes vs native's 8), but it IS 1:1 functional equivalence for the bias path. The remaining 4 native nodes (bias_to_vtcm, ForceFormat_Flat, Reshape×2) are output-side untile concerns.
