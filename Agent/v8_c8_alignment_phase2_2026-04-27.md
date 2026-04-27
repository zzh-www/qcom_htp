---
name: V8 C8 alignment Phase 2 retry — err 6006 confirmed independent of MemLoc 2026-04-27
description: Phase 2 device runtime keeps failing with err 6006 (Dma execution failed on skel side) regardless of whether statics are TCM_Only / DDR_OR_TCM / DDR_Only. Auto-inserted q::ConvLayer.opt.weights_to_vtcm × 3 always present. Going forward via composition, not auto-conversion.
type: project
---

# V8 C8 alignment Phase 2 retry (2026-04-27 evening)

## Going-in state

- Phase 1 (memory `v8_c8_alignment_phase1_2026-04-27.md`) verified Crouton_8 + Storage_Indirect on in[0] makes ctxgen pass and auto-insert `q::ForceFormat_Crouton` (256³).
- Phase 2 had failed device runtime with opaque "Graph Execution failure". Plan: add diagnostic markers + try MemLoc combos to surface which auto-inserted op fails.

## Setup

Edited `src/HmxMatMulV9SkelOp.cpp`:
- Crouton_8 + Indirect on in[0]; statics swept across {TCM_Only, DDR_OR_TCM, DDR_Only}.
- NOOP body that writes a 16-byte prologue (`0xA5, layout_in, layout_out, num_in, rank, dims, …, 0x5A`) so we can prove from output bytes alone whether the kernel ran.
- XML `BbbKMajor` in[0] shape `[1, M/32, 32, K]`.
- Build: `EXTRA_DEFS=-DV9_C8_ALIGNMENT_TEST bash build.sh && bash build_x86.sh`.
- Runner: `standard_flow/phaseB_v8/run_v8c8_phase2.sh` (NEW). Replaces `qnn_run/libQnn…Phase3_htp.so` so the proven `../libQnn…` invocation pattern picks our build.

## Result — err 6006 every variant

| Static MemLoc | ctxgen nodes | Device |
|---|---|---|
| DDR_OR_TCM (probe 1) | 5 nodes: 1 InputSlice + 3 weights_to_vtcm + 1 BbbKMajor | err 6006 |
| DDR_Only    (probe 2) | identical 5 nodes | err 6006 |

Critical findings:
- `q::ConvLayer.opt.weights_to_vtcm × 3` **always** auto-inserts when in[0]=Crouton_8, regardless of MemLoc on the statics.
- `q::ForceFormat_Crouton` did **not** auto-insert this time (vs. Phase 1 memory says it did). Graph went `InputSlice → BbbKMajor` directly. Suspect ONNX shape `[1, M/32, 32, K]` already matches the Crouton_8 logical shape, so QNN believes no conversion is needed.
- Output marker bytes never made it back — kernel never executed; failure is in the auto-DMA before our op runs.
- Real error from filtered logcat:
  ```
  E QNN     : QnnDsp <E> Internal error handing: Dma execution failed on the skel side. result = 6006 transport error = 0
  E QNN     : QnnDsp <E> Graph v8c8 failed in execution with err 6006
  ```
- Followed by SIGSEGV in qnn-net-run linker64 thread (unrelated cleanup crash after err 6006).

## Hypothesis (still unproven)

`q::ConvLayer.opt.weights_to_vtcm` is the SAME helper QNN uses for native ConvLayer_s1.opt — it expects native-format Crouton-blocked weight tensors with a specific descriptor. When in[0]=Crouton_8 triggers it on our flat `[1,1,K,N]` weight, the descriptor sizes are wrong → DMA fails on skel side. MemLoc declarations don't suppress its insertion.

## What was NOT tried (saved for next session)

1. **Composition path** (recommended): drop the auto-conversion approach. Have a wrapper op call `PackActCrouton` internally, then run BbbKMajor — same end-state as auto-insertion but with a path under our control. Avoids fighting the compiler.
2. **Minimal 1-input op** (`Identity_C8`): zero statics, only act in/out as Crouton_8 + Indirect. Confirms whether the framework's Crouton path itself works for our pkg, or whether something is broken even without statics.
3. **RE `q::ConvLayer.opt.weights_to_vtcm`** in `libQnnHtpV75Skel.so`: dump the descriptor it generates, see what weight format it actually expects. If we feed Crouton-blocked weights it might just work.
4. **Drop in[3] (scratch)** to see if auto-DMA fails on a specific tensor (count of weights_to_vtcm goes 3→2 if scratch was the offender).

## Reverted state

`HmxMatMulV9SkelOp.cpp` sig: all Flat4 + Direct + TCM_Only (V8 baseline). `MatMulV8Package.xml` BbbKMajor in[0] shape: `[1, K/32, M/32, 1024]` (V8 P2). Device side `qnn_run/libQnnHmxMatMulPhase3_htp.so` is replaced with the post-revert build; `qnn_run/phaseB/v8_ctx.bin` runs clean (verified 2026-04-27).

## Artefacts

- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v8c8_phase2.sh` — runner with logcat capture and marker decode.
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/phase1_validation/v8c8_test/{run.log, ctx/, device_out/}` — last failed run.
- Native 256³ baseline still 716 µs / 45587 cyc / 4 HVX threads.

## Recommendation

For 2026-04-28+ session: go with composition (option 1). It's strictly under our control and doesn't depend on QNN compiler behavior we can't introspect. Phase 1 architectural validation already shows we can declare Crouton_8 in our op-pkg; we just can't yet make the auto-DMA path work for the statics.
