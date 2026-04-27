---
name: V9 MatMul ctxgen segfault — debugging notes (open)
description: ctxgen segfaults at "Graph Optimizations" stage when MatMulV9 op (Crouton K-major input) is in graph; PackActCrouton-only graph works. Multiple workarounds tried, root cause unidentified.
type: project
---

# MatMulV9 ctxgen segfault — open issue (2026-04-25)

## Symptom

`qnn-context-binary-generator` segfaults (SIGSEGV / core dumped) immediately
after the "Starting stage: Graph Optimizations" log line, ONLY when a
graph containing the V9 `MatMulV9` op is fed in. Reproduced at multiple
shapes (32×128×128, 64×128×128) and after multiple op-pkg config variants.

## What works (control)

- V8 graph (`PackActivationU8RowMajor + PackWeightToHmxTileV3 + MatMulV8 +
  TcmDramCopy`) at all shapes, including 32×128×128.
- V9 graph WITHOUT MatMulV9 (`PackActCrouton + TcmDramCopy`) → ctxgen passes.
- Standalone PackActCrouton tests at 13 shapes → all bit-exact PASS.

## What was tried (still segfaults)

| Attempt                                            | Result    |
|----------------------------------------------------|-----------|
| Default V9 op definition                           | segfault  |
| Empty (no-op) host stub for MatMulV9 kernel        | segfault  |
| Match V9 input rank-4 last-dim to 1024 (vs 128)    | segfault  |
| Add v9_test/v9_model graph names to htp_backend_ext.json | segfault |
| Multithreaded=false (V8's setting)                 | segfault  |
| M_t = 1 (axis-squeeze hypothesis) → tried M=64 (M_t=2) | segfault |

## Hypotheses (un-verified)

1. **QNN optimizer pass that lowers MatMul-named ops** — there's likely a
   compiler pass that recognizes "MatMul" in op name and tries layout
   conversion or fusion. With our K-major input layout, the pass
   mis-handles the dim semantics. Test: rename to `MmKMajor` (no "MatMul"
   substring) and retest.
2. **Multiple TCM_Only inputs trigger an allocator-routing bug** —
   MatMulV9 has 4 TCM_Only inputs, optimizer may try to fuse VTCM
   allocations and trip on the K-major shape.
3. **QHPI_RESOURCE_HMX combined with NONTRIVIAL layout** — V8 also has
   this combination, but the input dim ordering differs. Test: change to
   QHPI_RESOURCE_HVX.
4. **Static initializer (transposed weight) being processed by an opt
   pass** — try with no static initializer.

## Pivot

Spending more time debugging is high-risk-low-reward. Take the WIN we have
(PackActCrouton bit-exact for general M×K) and pivot to:

- **Path A (recommended)**: Keep V8's pack_act_rm + mmv8 stack, use V9's
  graph topology (M_ROUNDS × N_ROUNDS) — proven working at sweep_v9 level.
  Improve pack_act_rm internally with HVX vshuff(-32) (sub-agent B's
  `pack_act_crouton_hvx.c` adapted to write row-major-32 output instead
  of Crouton format). This sidesteps the matmul kernel entirely while
  closing the pack_act perf gap.

- **Path B**: Rename MatMulV9 to something not containing "MatMul" and
  retest. If hypothesis 1 is correct, this single change fixes it.

- **Path C**: Defer MatMulV9 to a separate session; ship Path A's gains
  first.

## Files involved (state at impasse)

- `example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp` — kernel + reg
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/MatMulV8Package.xml`
  — OpDef + SupplementalOpDef entries for MatMulV9
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_out/HmxMatMul...
  /ConverterOpPackage/ConverterOpPackage.cpp` — SI/DI for MatMulV9
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_v9_test.py` —
  test ONNX generator (currently produces v9_model.onnx)
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/htp_backend_ext.json`
  — graph_names array includes "v9_model"

## What's preserved as wins

- **PackActCrouton bit-exact at 13 shapes** — `Agent/pack_act_crouton_skel_2026-04-25.md`
- **dlsym empirically validated cross-.so** — `Agent/dlsym_spike_PASS_2026-04-25.md`
- **Full RE for hmx_convbbb1x1_stride1** — `Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md`
  (still useful if Path B works or for descriptor reuse later)
- **Sub-agent B's HVX vshuff(-32) impl** — reusable for Path A
