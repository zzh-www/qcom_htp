---
name: V9 MatMul ctxgen segfault — ROOT CAUSE FIXED 2026-04-26
description: Segfault was const_prop calling PackActCrouton kernel on x86 host where the dlsym `convert_to_crouton_b` resolves to libQnnHtp.so x86 sim and crashes; fixed by adding pure-C reference impl on x86. ctxgen now passes; V9 stack runs end-to-end on device.
type: project
---

# V9 ctxgen segfault — fixed 2026-04-26

Supersedes `Agent/v9_matmul_ctxgen_segfault_2026-04-25.md`.

## Root cause

Segfault was NOT in MatMulV9 / BbbKMajor as previously assumed.

**Backtrace from gdb on `qnn-context-binary-generator` x86:**
```
#0  convert_to_crouton_b ()              ← libQnnHtp.so x86 sim
#1  pack_act_crouton_kernel ()           ← OUR custom op
#2  ?? ()
#3  GraphPrepare::const_prop()           ← QNN constant propagation
#4  GraphPrepare::const_prop_and_cse()
#5  GraphPrepare::do_prepare1()
#6  GraphPrepare::prepare()
```

**What happened**:
1. `gen_v9_test.py` builds an ONNX graph with `pack_wt = PackActCrouton(wt_raw_T)` where `wt_raw_T` is a static initializer.
2. QNN's `GraphPrepare::const_prop` pass invokes the custom op on the x86 host to fold the static input through the op into a baked output (so `pack_wt` runs zero times at runtime).
3. The PackActCrouton kernel had a single source path that calls dlsym `convert_to_crouton_b` (libQnnHtpV75Skel.so on Hexagon device).
4. On x86, the loader resolves `convert_to_crouton_b` against libQnnHtp.so (which has its own simulator stub). That stub expects HVX/VTCM runtime context and segfaults when called from QNN's const-prop pass.

**Why prior workarounds didn't help**: All attempts (renaming `MatMulV9 → BbbKMajor`, no-op MatMulV9 kernel, etc.) targeted the wrong op. The crash is in `PackActCrouton`, not `MatMulV9`.

## Fix

`example/hmx_matmul_phase3/kernel/pack_act_crouton_skel.c`: gate the dlsym path behind `#if defined(__hexagon__)`. On x86 host, run a generalized pure-C `crouton_pack_b_reference_full(out, a, M, K, h_start, h_end)` derived from the bit-exact reference in `crouton_pack_spike_hvx.c`.

Result: ctxgen passes cleanly. As a bonus, QNN's const_prop now correctly folds static-weight `PackActCrouton` at compile time, exactly like it folds `PackWeightToHmxTileV3` for V8.

## Validation

- `gen_v9_test.py --M 32 --K 128 --N 128` → ctxgen exit 0
- v9_no_mm graph (PackActCrouton + TcmDramCopy) → ✅ runs on device
- BbbKMajor with `-DV9_KERNEL_NOOP` (zeroes output) → ✅ runs on device — proves graph framework / tensor binding / VTCM allocation all work for V9
- V8 sweep at 512³ → still works (412K cyc, no regression)

## Open: BbbKMajor inline-asm MAC crashes at runtime

With the real inline-asm MAC enabled, V9 device run reports `Graph Execution failure`. Hypothesis: the weight tile from `PackActCrouton(transposed_weight)` is in row-major Crouton format (M-spatial × K-depth bytes contiguous), but HMX `weight.b = mxmem(addr, Rt=0x3FF)` expects a **double-vshuff bit-interleaved layout** that V8's `PackWeightToHmxTileV3` produces (see `pack_wt_v3_hvx.c:48-50` `Q6_Vb_vshuff_Vb` ×2).

Either (a) the wrong-format weight triggers an HMX execution exception, or (b) something else (alignment, bias addressing) faults.

## Path forward for V9 perf

The actual goal of V9 is to call QNN's `hmx_convbbb1x1_stride1` directly via dlsym (proven viable per `Agent/dlsym_spike_PASS_2026-04-25.md`; the §7.3 "NO across .so" claim in `Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md` was refuted by the convert_to_crouton_b spike).

### Scaffold landed 2026-04-26 (compile flag gated, OFF by default)

`HmxMatMulV9SkelOp.cpp` now has the dlsym call structure under `EXTRA_DEFS=-DV9_USE_DLSYM`:
- 3 descriptor structs declared (`hmx_conv_out_desc_t`, `hmx_conv_act_desc_t`, `hmx_conv_mask_desc_t`)
- `extern "C" void hmx_convbbb1x1_stride1(...)` — toolchain emits `R_HEX_JMP_SLOT` reloc against libQnnHtpV75Skel.so (verified via `hexagon-llvm-readelf -r`)
- per-kernel-call descriptor synthesis with best-guess field values (covered shape: M_tiles even, single ConvLayer instance per call, 1×1 conv)
- `EXTRA_DEFS="-DV9_USE_DLSYM -DV9_DLSYM_SKIP_CALL"` lets you build descriptors without invoking the kernel — confirmed safe, no crash

### What works

- ctxgen passes
- Graph framework loads .so with `R_HEX_JMP_SLOT` against `hmx_convbbb1x1_stride1` (confirmed exported as GLOBAL DEFAULT @ 0x2ea740 in libQnnHtpV75Skel.so)
- Descriptor build at runtime with `V9_DLSYM_SKIP_CALL` → no crash, runs to "Finished Executing Graphs"

### What doesn't (yet)

With actual `hmx_convbbb1x1_stride1` call (no skip): "Graph Execution failure". Crash is **inside** the kernel — descriptor values are wrong. Most likely culprits:
1. `mask_desc.out_check` / `act_check` failing the `bitsclr(_, 0x7e0)` alignment probe → tail-dispatches to `_unaligned` variant which may also fault on bad descriptors.
2. `m_total_minus_step` / `n_tiles_pow2` / `k_total_bytes` wrong → loop counter underflow → addresses past-the-end.
3. `act_ptr_pairs` / `out_tile_ptr_table` content interpreted differently than guessed (e.g. pointer pair stride, post-inc offset).
4. `alt_rt` value unknown — set to 0x3FF (same as base) for now; may need a different value for last-K.

### Next steps to actually validate the call

Empirical probing needed because the descriptor field VALUES (not just layout) come from a QNN graph-finalize-time formula that isn't reverse-engineered yet:

1. **Instrument-and-dump approach**: Build a custom op that hooks just before `q::ConvLayer_s1.opt`'s call to `hmx_convbbb1x1_stride1` and dumps the 3 descriptors to a known buffer. Run a tiny QNN MatMul (e.g. M=64, K=128, N=64) and read back the captured descriptors to learn correct field values.
2. **Pure-disasm approach**: Read more of the kernel disassembly past the prologue (currently RE'd only first ~30 packets of 492 B). Specifically, decode the loop1 update of `r17 -= r22` and the `r22 = 2 << r19` derivation to nail down `m_total_minus_step` semantics exactly.
3. **Brute-force approach**: Sweep field values around the best-guess and compare output against V8 reference. Risky — many degrees of freedom.

### Files modified for the scaffold

- `example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp` — `V9_USE_DLSYM`, `V9_DLSYM_SKIP_CALL` paths
- `example/hmx_matmul_phase3/build.sh` — `EXTRA_DEFS` env injection (already added during NOOP debugging)

To exercise:
```
EXTRA_DEFS="-DV9_USE_DLSYM" bash build.sh && bash build_x86.sh
# push libs, run gen_v9_test.py + ctxgen + qnn-net-run as in test_v9_small.sh
```

Estimated remaining effort: 1-2 more sessions for descriptor probing + bit-exact validation, then perf measurement at 2048³ (separate gen_v9_graph.py multi-instance generator like gen_v8_graph.py).

## Files modified 2026-04-26

- `example/hmx_matmul_phase3/kernel/pack_act_crouton_skel.c` — added `crouton_pack_b_reference_full` + x86 host gate (the actual fix)
- `example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp` — added `V9_KERNEL_NOOP` diagnostic compile flag for next-session debugging
- `example/hmx_matmul_phase3/build.sh` — added `EXTRA_DEFS` env var for build-time flag injection
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/test_v9_small.sh` — end-to-end V9-vs-V8 small-shape comparison harness (currently fails at V9 device run; useful once BbbKMajor MAC is fixed)
