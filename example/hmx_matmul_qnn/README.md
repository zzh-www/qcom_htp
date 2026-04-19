# hmx_matmul_qnn — int4×int16 MatMul as a QNN custom OpPackage

A QHPI-based QNN OpPackage that implements `MatMulInt4xInt16` as a new op type
(`HmxInt4MatMulPackage::MatMulInt4xInt16`), targeting the gap left by QNN's
built-in MatMul which rejects int4 weights (see
`example/qnn_matmul_profile/README.md` "Unsupported" table).

## Status (2026-04-19)

- ✅ Kernel (`kernel/hmx_int4_matmul.c`) — HMX dual-scale-readback tile
  kernel, 2 MAC packets + 2 converts per 32×32×32 tile (vs 12 for int16×int16).
- ✅ QHPI OpPackage (`src/HmxInt4MatMul{Interface,Op}.cpp`) — builds, loads
  via `backendRegisterOpPackage`, kernel matches in graph prepare.
- ✅ Host harness (`src/run_int4_matmul.cpp`) — builds a single-node QNN
  graph, does warmup + N-iteration steady-state timing, validates bit-exact
  against a scalar reference, reports `cycles_per_MAC`.
- ✅ **End-to-end HMX path verified on SM8650 v75 (ssh oneplus).** All shapes
  32³/128³/256³/512³ are bit-exact (0 mismatches). Root-caused the
  earlier `graphExecute err 1003` to QHPI-thread stack overflow (~17 KiB of
  on-stack tile buffers + decomposition arrays); fix was moving them to
  static globals (single-slice kernel, safe).
- 📊 Iteration-1 baseline (weight-pack-hoist): **8.33 cycles/MAC** at 512³
  (`~1.12 G cycles per execute`). Stable across scales (std dev <0.01%).
  ~565× away from the HMX theoretical floor (~0.015 cycles/MAC) — room
  dominated by scalar pack/unpack/decompose/combine. See
  `Agent/int4_matmul_optimization_log.md` for the trajectory + next-step
  plan (K-accumulation inside the HMX kernel to amortize per-tile overhead
  16× at 512³).

## Build + run

```sh
source scripts/env.sh

bash example/hmx_matmul_qnn/build.sh                        # HMX by default
bash example/hmx_matmul_qnn/run_on_device.sh                # 32x32x32
SHAPE=512,512,512 bash example/hmx_matmul_qnn/run_on_device.sh

# Scalar reference (bisection aid when an HMX change regresses):
SCALAR_ONLY=1 bash example/hmx_matmul_qnn/build.sh
```

Flags mirror `example/qnn_matmul_profile/profile_all.sh`:
`--device`, `--connect ssh|adb`, `--arch`, `--shape M,K,N`. M, K, N must be
multiples of 32. Harness does 1 warmup + 5 steady-state iters, reports
min / avg / max cycles and cycles_per_MAC.

Sample output (512³, HMX, iter 1):
```
[Steady] cycles: avg=1118570420  min=1118569312  max=1118571540  (n=5)
[Steady] MACs=134217728  cycles_per_MAC=8.33
[Check] mismatches=0/262144 max_abs_err=0
```

## Design — the offset-shift gotcha

The op signature declares `QHPI_QUInt16` and `QHPI_QUInt8` for the two data
tensors (matching what QNN supplies after graph optimization). The host test
writes signed int16/int8 data but declares the tensors as `QNN_DATATYPE_INT_16`
/ `QNN_DATATYPE_INT_8`. QNN's optimizer inserts a `Cast@FH.Fh` that turns each
signed value into its unsigned-quantized bit pattern:

- int16 → quint16: stored value = original_int16 + 32768
- int8  → quint8:  stored value = original_int8  + 128

The kernel un-shifts at the gather step (`gather_a_tile`, `gather_w_tile`
in `HmxInt4MatMulOp.cpp`) before calling the HMX kernel, which then expects
ordinary signed int16 activation and sign-extended-int4-as-int8 weight.

This was the difference between 1024/1024 mismatches and 0/1024 mismatches in
the scalar path — documented inline in the code.

## Integration with profile_all.sh (deferred)

`example/qnn_matmul_profile/profile_all.sh` goes ONNX → `qairt-converter` →
DLC → device. For the converter to route a MatMul-shaped op to our custom
package, we need either:

1. An XML op package config (generated via `qnn-op-package-generator` with a
   hand-written config) that `--op_package_lib` can reference, OR
2. A separate host-side graph-builder (like `run_int4_matmul.cpp`) that emits
   a DLC directly and skips `qairt-converter`.

Neither is wired up yet. `profile_all.sh` currently continues to fail on
`w4a16` at compose time — that's a future task.

## Files

```
src/
  HmxInt4MatMulInterface.cpp   QnnOpPackage_Interface_t + qhpi_init()
  HmxInt4MatMulOp.cpp          kernel dispatch (HMX or scalar via SCALAR_ONLY)
  run_int4_matmul.cpp          host test harness (builds graph, executes,
                               validates vs numpy-like reference)
kernel/
  hmx_int4_matmul.h            32×32×32 tile API
  hmx_int4_matmul.c            HMX dual-scale-readback kernel (adapted from
                               example/hmx_matmul_int16/int16_matmul_hmx.c)
build.sh                       hexagon-clang++ for DSP, NDK clang++ for ARM,
                               3 artifacts: HTP .so, CPU .so, host test binary
run_on_device.sh               push + execute via ssh/adb
```

## Debugging trail (resolved 2026-04-19)

- **Err 1003 (`QNN_COMMON_ERROR_SYSTEM`) on graphExecute** → QHPI kernel
  stack overflow. Fix: moved 4 × 1024-element int32 arrays
  (`A_h`, `A_l`, `P_hi`, `P_lo`, `col_sum_w`) in `hmx_int4_matmul.c` and
  three tile buffers in `HmxInt4MatMulOp.cpp` to `static` globals (safe
  because `multithreaded=false` in the kernel signature — single-slice
  sequential invocations).
- **Err 6006 (`QNN_GRAPH_ERROR_SET_PROFILE`)** when reusing a second
  profile handle mid-graph → HTP binds the profile at finalize time.
  Fix: reuse the same profile across every `graphExecute` call and read
  its per-iter events back (they replace, not accumulate).
- **1024/1024 mismatches at first run** → QNN's graph optimizer inserts
  `Cast@FH.Fh` that shifts int16 → quint16 by +32768 (and int8 → quint8
  by +128). Fix: kernel reads raw `uint*_t*` and subtracts the zero-offset
  at the gather step; documented inline in `gather_a_tile` / `gather_w_tile`.
