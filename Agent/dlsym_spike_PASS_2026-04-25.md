---
name: dlsym Phase 1 spike PASS — libQnnHtpV75Skel.so internal exports callable
description: Empirical proof that an op-pkg .so can call libQnnHtpV75Skel.so internal GLOBAL DEFAULT exports across .so boundaries via R_HEX_JMP_SLOT relocations
type: project
---

# Phase 1 dlsym spike — DEFINITIVE PASS (2026-04-25)

## Outcome

**Cross-.so call works.** A custom op-pkg HTP-side `.so` (our
`libQnnHmxMatMulPhase3_htp.so`) can directly call internal exports of
`libQnnHtpV75Skel.so` loaded into the same FastRPC process on DSP.

This green-lights Route 1 (wholesale reproduce QNN MatMul by calling its
internal compute primitives directly).

## Setup

- Added `extern "C" void convert_to_crouton_b(...)` declaration in
  `example/hmx_matmul_phase3/kernel/pack_act_rm_hvx.c`.
- Added `spike_call_once()` that on the first slice executes
  `convert_to_crouton_b` against a static 32×128 input with hand-crafted
  descriptor (per `Agent/sig_convert_to_crouton_b_2026-04-25.md`).
- Rebuild produced `convert_to_crouton_b` as `R_HEX_JMP_SLOT` UND symbol
  in our op-pkg `.so` — proving toolchain emits a proper PLT-style import.
- Pushed op-pkg + ran existing 4096³ V9 graph → completed without error.
- Disabled the spike call → reran same ctx → byte-identical V8 output (md5 match).

## Empirical evidence

| Test                                    | Result            | Implication                              |
|-----------------------------------------|-------------------|------------------------------------------|
| op-pkg with new UND symbol loads        | ✅ no error       | DSP loader accepts cross-.so import      |
| V8 graph executes                       | ✅ Finished       | Symbol resolved at load (or first call)  |
| `cmp` post-spike vs no-spike output     | ✅ 0 diffs        | Call ran AND didn't corrupt state        |
| md5 with-spike == md5 no-spike          | ✅ identical      | Same as above, conclusive                |

## What this rules in / rules out

**Rules IN**:
- `convert_to_crouton_b` — confirmed callable
- `hmx_convbbb1x1_stride1` — same export type & ABI safety per sub-agent A
- `concat_*_crouton_*` family — same mechanism
- `extract_tile_vmemu_u8/u16/u32` — same
- `scatter_tile_u8/u16/u32` — same
- `vmemcpy_2d_asm` and family — same

**Refutes** (sub-agent caveats based on FastRPC documentation, not testing):
- ✗ "FastRPC doesn't merge .so namespaces"
- ✗ "Our op-pkg has no import reloc"
- ✓ Both are wrong — `R_HEX_JMP_SLOT` reloc + `qhpi_*` UND list + working call all empirically demonstrate the toolchain & loader handle cross-.so imports correctly.

**Remaining valid caveat**:
- Symbol names are not stable ABI. QNN SDK upgrades may rename or
  inline these. Pin the SDK version we test against.

## How to reproduce

1. Build op-pkg with the spike code in `kernel/pack_act_rm_hvx.c`
   (the conditional `if (qhpi_slice_number(handle) == 0) spike_call_once();`).
2. `bash build.sh && bash build_x86.sh` (existing scripts).
3. Push `libQnnHmxMatMulPhase3_htp.so` to device `~/qnn_run/`.
4. Run any V8 ctx, e.g. the `phaseB/v8_ctx.bin`.
5. If V8 finishes without crash and output md5 matches the no-spike
   baseline, dlsym + call both work.

## Bit-exact output validation — DONE 2026-04-25

End-to-end ONNX→DLC→ctxgen→qnn-net-run with `CroutonPackSpike` op for
fixed `32×128` u8 input → 4 × 1024 B Crouton blocks output.

**Result**: stats `[skel_done, ref_done, n_diffs, max_diff] = [1, 1, 0, 0]`.
Python-side independent comparison: `n_diff = 0/4096`. **Bit-exact.**

**Key discovery (via sub-agent disasm of outer loop, recorded in
`Agent/sig_convert_to_crouton_b_2026-04-25.md` §8)**: the `aux` argument
(r1) is NOT a fast-path-only flag. It controls the height-tile-stride
exponent via:

```
r25 = ((h & mask) << (4 - ct0(aux))) << 7
```

For our use case (M=32, K=128, height_tiles=8, channel_groups=4),
**`aux = 16`** gives `ct0(16)=4 → r25 stride = 1<<0 = 1 → r25 = h*128`,
which is the per-spatial-tile offset we want. Initial `aux=0` made the
function gate out the inner loop after iter 0 (skipping h=1..7).

Now **Route 1 is fully validated**: cross-.so dynamic link works AND
descriptor RE is correct.

## Next steps (Phase 3 plan)

1. **PackActCrouton custom op** — wraps `convert_to_crouton_b` for general
   `(M, K)` activation packing. Output: `[M_groups, 4_lanes, 1024]`
   Crouton blocks. Reference impl in same file for testing.
2. **MatMulSkelHmx custom op** — wraps `hmx_convbbb1x1_stride1`. Sets up
   the 3-descriptor architecture (out_desc, act_desc, mask_desc) per
   `Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md`. Reads Crouton-format
   activation, Crouton-format weight, mxmem2-layout bias, writes 1KB tile-
   layout output via scatter table.
3. **PackWtCrouton custom op** — pack weights into the format
   `hmx_convbbb1x1_stride1` expects. May be `convert_to_crouton_b` again
   (same pack works for weights if we transpose K↔N first).
4. **gen_v10_graph.py** — emits ONNX with `plan_matmul_graph`-style tile
   topology, mirroring QNN MatMul lowered structure (see
   `Agent/qnn_matmul_design_principles_2026-04-25.md` §4).
5. **Bit-exact unit tests** at 32³ first (smallest valid Crouton config).
6. **Sweep 32/128/256/512/1024/2048/4096³** vs QNN, iterate until ≤1.15× cyc.
