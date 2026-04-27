---
name: q::ConvLayer.opt.weights_to_vtcm RE — descriptor builder lives in libHtpPrepare 2026-04-27
description: RE the auto-inserted weights_to_vtcm op that fails our V8 C8 Phase 2 with err 6006. Found that it's a prepare-time DMA descriptor builder (not a runtime kernel), and concluded the descriptor depends on the DOWNSTREAM op's weight layout expectation, which mismatches when our op declares Crouton_8 act but flat u8 weight.
type: project
---

# RE of `q::ConvLayer.opt.weights_to_vtcm` (2026-04-27)

User asked: stop being afraid of the symbol, RE it directly.

## Where the string lives

| Library | File | Strings: `q::ConvLayer.opt.weights_to_vtcm` | `@F5e.f5e.` variant |
|---|---|---|---|
| Host x86 | `lib/x86_64-linux-clang/libQnnHtp.so` | 3 | 3 |
| Host x86 | `lib/x86_64-linux-clang/libHtpPrepare.so` | 3 | 3 |
| ARM device | `lib/aarch64-android/libQnnHtpPrepare.so` | 3 | 3 |
| ARM device | `lib/aarch64-android/libQnnHtp.so` | 0 | 0 |
| **DSP** skel | `lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so` | **0** | **0** |

## Key finding — it's PREPARE-time, not RUNTIME

The string does NOT appear in `libQnnHtpV75Skel.so`. The skel-side runtime has no kernel literally called `weights_to_vtcm`. So this op cannot be a DSP kernel that gets dispatched at execute time.

Instead, `q::ConvLayer.opt.weights_to_vtcm` is a **graph-prepare-time** op: it executes during `qnn-context-binary-generator`, builds a serialized DMA descriptor, and embeds it into the context binary. At `graphExecute` time the skel just runs the pre-baked DMA — if the descriptor is malformed (e.g., bogus strides), the DSP DMA engine faults and FastRPC bubbles up err 6006 = "Dma execution failed on the skel side".

Confirmed by the error string lookup:
- `"Internal error handing: Dma execution failed on the skel side. result = %d"` lives only in `aarch64-android/libQnnHtp.so` (ARM-side runtime), printed when FastRPC returns the result code from skel.
- The `0x1776` (= 6006) immediate appears 43 times in the skel disasm, scattered across DMA error paths in `libQnnHtpV75Skel.so` ranges 0x1ad000–0x1f6000 + 0x349000–0x34a000.

## What we did find — the cost-function table

Disassembly of `libHtpPrepare.so` finds 7 pointer-references to `ConvLayer.opt.weights_to_vtcm@F5e.f5e.` strings inside `.data.rel.ro`. Each pointer is the head of an 8-qword struct:

```
+0x00  string ptr  ("ConvLayer.opt.weights_to_vtcm@F5e.f5e.")
+0x08  string len  (38 bytes)
+0x10  ptr to ""   (empty per-instance string)
+0x18  0
+0x20  fn ptr      <-- 0x0f1aef0  (regular)  or  0x0f1a360  (fallback)
+0x28  ptr to binary cost-coefficient blob (per-variant)
+0x30  1
+0x38  another rodata ptr
```

Disassembling `0x0f1aef0` showed it is **not** the kernel — the demangled symbol is `hnnx::cost_func_from_str(string_view)`. It evaluates a polynomial cost using the coefficients at `+0x28`. Used by the graph optimizer to score which variant to pick.

The `&fallback` variant uses `0x0f1a360` (slightly different cost path).

So the 5 + 2 entries are the *optimizer's score tables* for 7 different specializations of `weights_to_vtcm`. They are not the descriptor builder.

## Why our V8 C8 fails but V8 production succeeds

Both setups auto-insert `q::ConvLayer.opt.weights_to_vtcm × 3` for the static initializers (wt, bias, scratch). The static signatures are **identical** in both cases (Flat4 + Direct + TCM_Only) — the *only* thing that changes between V8 production (works) and V8 C8 (err 6006) is the **activation** signature: Flat4 vs Crouton_8 + Indirect.

This means `weights_to_vtcm`'s descriptor-building logic looks at more than just the static tensor it's DMA'ing — it also inspects the **downstream consumer op's expected weight format**, which is determined by the activation layout sig.

- Activation = Flat4 → downstream consumer uses raw weights → DMA = simple linear copy → works for our flat `[1,1,K,N]` u8.
- Activation = Crouton_8 → downstream consumer is treated like native ConvLayer_s1.opt → DMA = transpose+block into Crouton-tile weight format → requires source to be `[H, W, Cin, Cout]` with native ConvLayer's specific blocking → our flat u8 mismatches → out-of-bounds reads → err 6006.

The reason the native flow works: ONNX MatMul → QNN's built-in Conv lowering preprocesses the weight tensor into the expected blocked layout before `weights_to_vtcm` is invoked. For our custom op, QNN doesn't know our static input is "a weight", so it skips that preprocessing.

## Was further RE viable?

To find the actual descriptor builder we'd need to:
1. Look in `libHtpPrepare.so` `.init_array` static initializers, find the op-registration site that pairs the name with the kernel function pointer (likely templated and not directly named).
2. OR: capture native ConvLayer's runtime DMA descriptor by tracing on device — the descriptor is what gets embedded in the context binary, so it's also recoverable from a native ctx-bin diff.

Both are heavier than what's needed to unblock V8 C8 work. The bigger insight from this RE — *the DMA descriptor format is fixed for the ConvLayer pipeline and assumes built-in weight preprocessing* — is itself the answer.

## Concrete fix paths

| Option | What | Cost | Risk |
|---|---|---|---|
| **A. Composition** | wrap `PackActCrouton + BbbKMajor` in one op-pkg op; declare all sigs Flat4 to skip auto-insert path | low — code already exists | none, V8 prod already works with this pattern |
| B. Bake Crouton weight | preprocess our static weight into native ConvLayer's blocked layout before ONNX, declare layout `Weights8x4` or similar in QHPI sig | medium — need to RE the exact blocking | requires getting weight-layout exactly right |
| C. Trace native + replay | run native ConvLayer 256³ on device with a logging hook, dump the DMA descriptor it builds, embed it manually | high — invasive runtime instrumentation | one-shot only, breaks on any QNN version bump |

**Recommendation: A**. We already have working `PackActCrouton` (verified bit-exact in 13 shapes per `Agent/sig_convert_to_crouton_b_2026-04-25.md`) and a working `BbbKMajor` HMX kernel. Composing them in a single op-pkg avoids the auto-insert path entirely.

## Path A sketch

1. Inside `BbbKMajor` kernel body, on first invocation: call `PackActCrouton` (already exposed via dlsym to `convert_to_crouton_b`) on `inputs[0]` (raw u8 act) writing into a private VTCM scratch.
2. Run the existing `mmv8` HMX inner loop using the Crouton'd act + flat wt.
3. Output as Flat4 row-major (existing V8 path).

This collapses the 4-op V8 graph (pack_act_rm + pack_wt + mmv8 + tcm2ddr) into a single op without touching QNN's lowering pipeline.

## Artefacts produced this session

- This file: `Agent/qnn_re/weights_to_vtcm_RE_2026-04-27.md`
- Cost-table dumps confirmed: VMA 0x61ffdd0 / 0x622ea80 / 0x626a410 / 0x62a6d70 / 0x6308560 (regular variant, fn ptr 0x0f1aef0); 0x622fa50 / 0x626b6a0 (fallback variant, fn ptr 0x0f1a360).
- Function `0x0f1aef0` decoded as `hnnx::cost_func_from_str` — polynomial cost evaluator, not the kernel.

Time spent: ~1 hour of static analysis. Further RE viable but composition path is faster to a fix.
