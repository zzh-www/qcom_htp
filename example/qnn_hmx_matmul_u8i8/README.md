# qnn_hmx_matmul_u8i8

Minimal QNN custom-op package for one representative MatMul:
`HmxU8I8ToU8MatMul`, a u8 x i8 -> u8 path backed by the owned V73DEEP HMX
Conv1x1 inline-asm kernel replica.

## Layout

```text
src/
  QnnHmxMatMulU8I8Interface.cpp   OpPackage interface/provider
  HmxU8I8ToU8MatMulOp.cpp         QHPI precompute + HMX descriptor bridge
  v73deep_conv1x1_kernel.h        owned V73DEEP kernel entry
  v73deep_conv1x1_kernel.inc      1132-byte inline-asm body
build.sh                          hexagon-v75 + aarch64 op-package build
build_x86.sh                      x86_64 op-package build for ctxgen
standard_flow/
  native_baseline/                QNN native MatMul baseline
  custom_u8i8/                    custom op ONNX -> DLC -> ctx flow
```

## Build

```bash
bash example/qnn_hmx_matmul_u8i8/build.sh
bash example/qnn_hmx_matmul_u8i8/build_x86.sh
```

The outputs are:

```text
build/hexagon-v75/libQnnHmxMatMulU8I8_htp.so
build/aarch64/libQnnHmxMatMulU8I8_cpu.so
build/x86_64-linux-clang/libQnnHmxMatMulU8I8.so
```

## Standard Flow

```bash
cd example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8
SKIP_DEVICE=1 bash run_u8i8_chain.sh
```

Drop `SKIP_DEVICE=1` when the target device is available.  Perf decode:

```bash
python scripts/perf_hmx_u8i8_matmul.py \
  example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/out/u8i8_chain_256
```

For the complete design and debugging recipe, see
`docs/qnn_custom_op_matmul_e2e.md`.
