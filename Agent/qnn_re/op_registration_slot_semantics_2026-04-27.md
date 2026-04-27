---
name: ConvLayer slot-semantic registration RE — QHPI v1 sig has no per-slot role field; early_rewrite is the only mechanism 2026-04-27
description: RE'd libHtpPrepare.so to find how native ConvLayer tells QNN that in[2] is bias (use bias_to_vtcm) vs in[1] is weight (use weights_to_vtcm). The slot-role distinction lives in INTERNAL op-tables not exposed via QHPI v1. Only mechanism custom ops have to control auto-DMA insertion is `early_rewrite` — same pattern MaxPool uses to inject q::QNN_SlicePad_shape.
type: project
---

# Op-registration slot-semantics RE (2026-04-27)

User: 「RE libHtpPrepare.so 的 op-registration 表，看 native ConvLayer 是怎么标注 slot 语义的」.

## What we found in libHtpPrepare.so

Strings present in the host-side prepare lib confirm the helper-op family:

```
q::ConvLayer.opt.weights_to_vtcm        ← for weights (in[1])
q::ConvLayer.opt.bias_to_vtcm           ← for bias    (in[2])
q::ConvLayer.opt.activations_to_vtcm    ← for act     (in[0])
q::flat_to_vtcm                          ← generic Flat tensor → VTCM
q::constant_flat_to_vtcm                 ← static Flat tensor → VTCM
q::crouton_to_vtcm                       ← generic Crouton tensor → VTCM
q::constant_crouton_to_vtcm              ← static Crouton tensor → VTCM
```

Each op has multiple specialization variants tagged with `@<sig>` (e.g. `@F5e.f5e.`, `@CH.fH.`, `@CB.cB.`). Each variant is paired with a polynomial cost coefficient blob; the optimizer picks variants by cost.

## What we did NOT find directly

- No exported symbol named anything like `make_bias_to_vtcm` / `insert_bias_dma` / `dispatch_to_vtcm`.
- No `.data.rel.ro` direct pointer to the canonical `q::ConvLayer.opt.bias_to_vtcm` string. The 4 RIP-rel refs to that string ALL live in `_ZNK12GraphPrepare23serialize_blob_epilogue` — i.e., serialization (writing bottom_mapping.json), not the dispatcher.
- The dispatch logic (which decides bias_to_vtcm vs weights_to_vtcm based on op-type + slot-index) is templated/inlined in graph rewrite passes; finding the exact function would require deeper static analysis (init_array walk, vtable diff, or interactive debug). Not pursued — there's already a viable workaround at the QHPI level.

## What this means for custom ops

`QHPI_Tensor_Signature_v1` has fields `(element_type, layout, storage, mem_placement)` — **no role/kind/slot-semantic field**. There is no QHPI v2 either (none in the headers). The public API does NOT let us tell QNN "this slot is bias, use bias_to_vtcm".

So when our `BbbKMajor` declares `Crouton_8 + Indirect` on in[0], QNN classifies the op as ConvLayer-like, then for each static input picks the BEST `weights_to_vtcm` variant by polynomial cost. The bias slot's shape doesn't match any `weights_to_vtcm@<sig>` variant well, so it falls into a fallback that builds a malformed DMA descriptor → err 6006.

## How to "replicate the slot-semantic mechanism"

Single mechanism available: `QHPI_OpInfo_v1::early_rewrite`. This callback fires before the graph optimizer's auto-insertion runs, lets the op REWRITE its own subgraph, and the result becomes part of the graph. Same pattern the SDK MaxPool example uses to inject `q::QNN_SlicePad_shape` between an Act tensor and the kernel.

For `BbbKMajor`, the early_rewrite would:

```cpp
// 1. Split BbbKMajor (raw, with early_rewrite) → BbbKMajor_kernel (no rewrite)
//    so we don't infinite-recurse.

static const QHPI_Op *bbb_early_rewrite(const QHPI_Op *op) {
    QHPI_OpRef act     = qhpi_op_input(op, 0);
    QHPI_OpRef wt      = qhpi_op_input(op, 1);
    QHPI_OpRef bias    = qhpi_op_input(op, 2);
    QHPI_OpRef scratch = qhpi_op_input(op, 3);

    // 2. Manually insert q::ConvLayer.opt.bias_to_vtcm for bias.
    //    Or use q::constant_flat_to_vtcm as a safer generic fallback.
    QHPI_OutputDef bias_def = qhpi_op_output(bias.op, bias.output_number);
    const QHPI_Op *bias_dma = qhpi_op_create(
        op, "q::ConvLayer.opt.bias_to_vtcm",  // or "q::constant_flat_to_vtcm"
        1, &bias, 1, &bias_def);
    QHPI_OpRef bias_via_dma = qhpi_op_reference(bias_dma, 0);

    QHPI_OutputDef scratch_def = qhpi_op_output(scratch.op, scratch.output_number);
    const QHPI_Op *scratch_dma = qhpi_op_create(
        op, "q::constant_flat_to_vtcm",
        1, &scratch, 1, &scratch_def);
    QHPI_OpRef scratch_via_dma = qhpi_op_reference(scratch_dma, 0);

    // 3. Build the actual kernel op (different name → no recursion).
    QHPI_OpRef new_inputs[4] = {act, wt, bias_via_dma, scratch_via_dma};
    QHPI_OutputDef out_def = qhpi_op_output(op, 0);
    return qhpi_op_create(op, "HmxMatMulPhase3Package::BbbKMajor_kernel",
                          4, new_inputs, 1, &out_def);
}
```

When QNN's optimizer then runs auto-insertion, it sees the bias and scratch inputs ALREADY pass through `to_vtcm` ops, so it skips re-inserting `weights_to_vtcm` for them. Only the wt input (which we still want auto-DMA'd via `weights_to_vtcm`) gets the auto-insertion. We control which DMA helper handles which static.

## Caveats / not yet validated

- We have NOT empirically verified that `qhpi_op_create("q::ConvLayer.opt.bias_to_vtcm", ...)` accepts our `[1, 1, N/32, 128]` u16 bias shape. Bias_to_vtcm has its own variants with their own input-shape constraints (the `@<sig>` strings). If our bias doesn't match any variant, the manual insertion could produce its OWN error.
- Safer first step: try `q::constant_flat_to_vtcm` for bias — it's the GENERIC variant for static Flat tensors, doesn't make ConvLayer-bias-specific assumptions.
- The early_rewrite recipe ALSO requires registering a separate `BbbKMajor_kernel` op type (different from `BbbKMajor`) with the actual kernel function and no early_rewrite. Otherwise we get infinite recursion.

## TL;DR answer

「能不能复制 slot 语义到我们 OpDef」: **不能直接通过 QHPI sig** —— `QHPI_Tensor_Signature_v1` 没有 role/kind 字段。**但可以通过 `early_rewrite` 间接达成** —— 像 MaxPool 例子那样在 op 注册时挂一个 rewrite 回调，回调里手动 `qhpi_op_create("q::ConvLayer.opt.bias_to_vtcm", ...)` 把 bias 输入接到指定 DMA op 上，然后用一个 different-named kernel op 完成 lowering。这正是 QNN-sdk MaxPool 例子里 `poolmax_to_ref` 的模式。

当前 V8 C8 Phase 2 用 `MemLoc_DDR_OR_TCM` workaround 已经跑通端到端，所以 early_rewrite 路径属于「未来想要更精细控制时再做」的优化。
