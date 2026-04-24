# hmx_matmul_phase3 — V8 quantized matmul (u8 × i8 → u8)

Pure-HMX replica of QNN's `q::ConvLayer_s1.opt` with the same `:after:cm:sat.ub`
silicon semantics. Writes tile-layout output directly from HMX — no scatter.

## Directory

```
src/                               V8 runtime code (5 files)
  HmxMatMulPhase3Interface.cpp     OpPackage interface, 5 ops registered
  HmxMatMulV8Op.cpp                MatMulV8 HMX kernel
  run_matmul_v8_graph.cpp          Host harness — hand-built QNN graph, ground truth
kernel/                            HVX kernels (.c)
  pack_act_rm_hvx.c                PackActivationU8RowMajor
  pack_wt_v3_hvx.c                 PackWeightToHmxTileV3
  tcm_dram_copy_hvx.c              TcmDramCopy (VTCM → DDR bulk)
  untile_to_rowmajor_hvx.c         UntileToRowMajor (optional DDR out)
build.sh                           hexagon-v75 + aarch64-android build
build_x86.sh                       x86_64 build (ctxgen host-side)
run_v8_graph_on_device.sh          quick device test via host harness
standard_flow/                     ONNX → DLC → ctx-binary → optrace flow
  phaseA_native/                     (symlink to QNN-native baseline)
  phaseB_v8/                         V8 custom-op full pipeline
_archive/                          Old V2-V7 / int4 / probe code — not built
```

## Two ways to run V8

### A. Host harness (fastest iteration, DIAG bit-exact checks)
```bash
bash build.sh
bash run_v8_graph_on_device.sh --shape 512,512,512
```

### B. Standard QNN flow (ONNX → DLC → context binary → qnn-net-run optrace)
See `docs/qnn_custom_op_sop.md` for the SOP. Worked example:
`standard_flow/phaseB_v8/` (V8 matmul, reproduces to `device_out/chrometrace.json`).

## Current perf (512³ u8×i8)

| Metric | V8 (this) | QNN native w8a8 | V8/QNN |
|---|---:|---:|---:|
| HMX timeline cycles | 317,009 | 66,513 | 4.8× slower |
| MatMul core cycles  | 363,944 | 24,474 | 14.9× |
| pack_act cycles     | 520,132 | 173,244 (built-in InputSlicePad+Crouton) | 3.0× |

V8 is a pedagogical replica; QNN's built-in path uses Crouton format that V8
(staying row-major by design) can't match. See `Agent/v8_vs_native_optrace_2026-04-25.md`
for the full cycle breakdown.

## Git history

All older kernels / host harnesses moved to `_archive/` via `git mv` in the
2026-04-25 V8-only cleanup — blame and log chains preserved. See
`_archive/README.md` for what's there and how to re-enable.
