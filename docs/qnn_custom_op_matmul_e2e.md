# QNN HMX U8/I8 MatMul Custom Op E2E

这篇文档是当前自定义 MatMul 的唯一端到端说明。实现主线是：

```text
example/qnn_hmx_matmul_u8i8
```

目标不是保留所有历史实验，而是保留一个最有代表性的 custom op:

```text
HmxU8I8ToU8MatMul
```

它用 QNN custom-op 机制接入，运行 owned V73DEEP HMX Conv1x1 inline-asm kernel replica，实现 u8 activation x i8 weight -> u8 output。

## Current Names

| Surface | Current name |
|---|---|
| Directory | `example/qnn_hmx_matmul_u8i8` |
| Package | `QnnHmxMatMulU8I8Package` |
| Provider | `QnnHmxMatMulU8I8InterfaceProvider` |
| Main op | `HmxU8I8ToU8MatMul` |
| XML | `standard_flow/custom_u8i8/QnnHmxMatMulU8I8Package.xml` |
| HTP lib | `libQnnHmxMatMulU8I8_htp.so` |
| CPU lib | `libQnnHmxMatMulU8I8_cpu.so` |
| x86 lib | `libQnnHmxMatMulU8I8.so` |
| Perf tool | `scripts/perf_hmx_u8i8_matmul.py` |

## What Is Kept

Active files:

```text
example/qnn_hmx_matmul_u8i8/
  build.sh
  build_x86.sh
  src/
    QnnHmxMatMulU8I8Interface.cpp
    HmxU8I8ToU8MatMulOp.cpp
    v73deep_conv1x1_kernel.h
    v73deep_conv1x1_kernel.inc
  standard_flow/
    native_baseline/
    custom_u8i8/
      QnnHmxMatMulU8I8Package.xml
      gen_u8i8_chain.py
      run_u8i8_chain.sh
      converter/ConverterOpPackage.cpp
      htp_config.json
      htp_backend_ext.json
```

`native_baseline` is only for QNN native MatMul comparison.  The custom path has one op and one kernel body.

## QNN Calling Architecture

当前实验结论是：QNN 对 custom op 的调用不是直接进入内置 `q::*` primitive，而是经过一个明确的 custom-op boundary。

```text
ONNX hmx::HmxU8I8ToU8MatMul
  -> XML declares op surface
  -> converter op package provides shape/type inference
  -> qairt-converter emits DLC custom node
  -> ctxgen loads x86 op package and resolves QHPI precompute registration
  -> HTP runtime inserts surrounding built-ins:
       q::*InputSlice
       q::ForceFormat_Crouton
       q::ConvLayer.opt.weights_to_vtcm
  -> QHPI graph-load precompute records direct bias/weight pointers,
     activation/output Crouton block tables, and tile counts
  -> QHPI hot callback stitches the tiny native descriptors on stack
  -> owned V73DEEP inline asm runs
```

这个 boundary 是 gap 的核心背景，但 perf 读法要小心：native MatMul 会拆成 `bias_to_vtcm`、`weights_to_vtcm`、`DmaCheckpointSet`、`ConvLayer_s1.opt` 等多个 HTP 事件。早期 custom 版本把 QHPI lookup、shape recovery、pointer-table copy 都计在 `HmxU8I8ToU8MatMul` 热事件里；当前默认实现用 QHPI precompute 把这些准备工作提前到 graph-load/context-load 阶段，hot event 基本只剩 descriptor stitching + HMX body。

## Op Signature

QHPI runtime signature:

```text
in[0] bias
  dtype: Int32
  layout: Flat4
  storage: Direct
  memory: TCM_Only
  logical shape: [1, N/32, 1, 64]

in[1] weight
  dtype: QUInt8
  layout: Flat4
  storage: Direct
  memory: TCM_Only
  ONNX logical shape: [1, 1, K, N]
  byte layout: K-major HMX tiles

in[2] activation
  dtype: QUInt8
  layout: Crouton_8
  storage: Indirect
  memory: TCM_Only
  logical shape: [1, M/32, 32, K]

in[3] scratch
  dtype: QUInt8
  layout: Flat4
  storage: Direct
  memory: TCM_Only
  logical shape: [1, 1, 1, 2048]

out[0] output
  dtype: QUInt8
  layout: Crouton_8
  storage: Indirect
  memory: TCM_Only
  logical shape: [1, M/32, 32, N]
```

The generator currently assumes square shapes: `M == K == N`, multiples of 32.

## Weight And Bias Format

Weight bytes are pre-packed by `gen_u8i8_chain.py` into K-major HMX tiles:

```text
tile order: [K_t, N_t, 1024]
within tile:
  dst = (k_row / 4) * 128 + n_col * 4 + (k_row % 4)
```

The ONNX initializer is still shaped as `[1, 1, K, N]` so QNN's static weight transfer path sees a native-looking tensor.  The bytes are already in the exact tile order expected by the V73DEEP kernel path.

Bias bytes are folded at generation time into the native 256-byte-per-N-tile form:

```text
per N tile:
  bytes 0..127   : 32 x (fp16 scale, fp16 baseline)
  bytes 128..255 : 32 x int32 effective_bias

effective_bias[c] = -ACT_ZP * sum_k(weight[k,c]) + bias_q[c]
ACT_ZP = 128
```

This is why runtime can pass `bias_bytes` directly to the owned kernel.

## Runtime Descriptor Defaults

`HmxU8I8ToU8MatMulOp.cpp` has the production descriptor choices built in.  Normal build does not need extra flags.

Core defaults:

```text
kernel body: owned V73DEEP inline asm
input order: bias, weight, activation, scratch
activation/output: Crouton_8 indirect TCM block tables
weight layout: K-major packed tiles
out_table_stride_dwords = N_t
out_y_stride_words = M_t * 4
n_tiles_pow2 = M_t * 4
m_total_minus_step = 8
k_total_bytes = N_t * 32
n_act_pairs = K_t
act_table_y_stride_words = M_t * 4
set_hmx_params_conv1x1(mask, 0x700, 0, 0, 0, 0x20)
extra_param = {1, 0}
mask[0x38] = extra_param pointer
```

The QHPI precompute function derives `S` from the activation block table length and stores the prepared state:

```text
precomputed_data:
  bias_bytes       -> QNN-prepared Direct TCM bias
  wt_pack          -> QNN-prepared Direct TCM K-major weight
  act_qhpi_table   -> QNN Crouton activation block table
  out_qhpi_table   -> QNN Crouton output block table
  M_t/N_t/K_t      -> recovered tile counts
```

The hot callback no longer queries QHPI tensors or copies pointer tables. It builds the small native descriptors on stack, patches the mask descriptor, then calls:

```text
our_v73deep_kernel(out_desc, act_desc, weight, bias, mask_desc, extra_param)
```

## Diagnostics

Only these diagnostic build macros are kept:

| Macro | Purpose |
|---|---|
| `HMX_U8I8_DESC_DUMP` | write compact descriptor/mask/table metadata into output and skip kernel |
| `HMX_U8I8_SKIP_KERNEL` | build descriptors, mark output, skip kernel body |
| `HMX_U8I8_PROBE_CYCLES` | write kernel/descriptor/table/QHPI setup cycle split |

Example:

```bash
EXTRA_DEFS=-DHMX_U8I8_PROBE_CYCLES \
  bash example/qnn_hmx_matmul_u8i8/build.sh
```

Then decode:

```bash
python scripts/perf_hmx_u8i8_matmul.py <out_dir> --probe-cycles
```

## Build

From repo root:

```bash
bash example/qnn_hmx_matmul_u8i8/build.sh
bash example/qnn_hmx_matmul_u8i8/build_x86.sh
```

Expected outputs:

```text
example/qnn_hmx_matmul_u8i8/build/hexagon-v75/libQnnHmxMatMulU8I8_htp.so
example/qnn_hmx_matmul_u8i8/build/aarch64/libQnnHmxMatMulU8I8_cpu.so
example/qnn_hmx_matmul_u8i8/build/x86_64-linux-clang/libQnnHmxMatMulU8I8.so
```

## Converter And Context Binary

Device not required:

```bash
cd example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8
SKIP_DEVICE=1 bash run_u8i8_chain.sh
```

Main generated artifacts:

```text
out/u8i8_chain_256/u8i8.onnx
out/u8i8_chain_256/u8i8.dlc
out/u8i8_chain_256/ctx/u8i8_ctx.bin
out/u8i8_chain_256/ctx/u8i8_schematic.bin
out/u8i8_chain_256/ctx/u8i8_bottom_mapping.json
```

The expected ctx node mix for chain8 is:

```text
QnnHmxMatMulU8I8Package::HmxU8I8ToU8MatMul x 8
q::ConvLayer.opt.weights_to_vtcm x 3
q::*InputSlice x 1
q::ForceFormat_Crouton x 1
```

## Device Run

With a reachable device:

```bash
cd example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8
DEVICE=oneplus bash run_u8i8_chain.sh
```

Useful knobs:

```text
M=256 K=256 N=256
CHAIN=8
MODE=chain | independent
OUT_DIR=<custom output dir>
SKIP_DEVICE=1
```

The script pushes:

```text
libQnnHmxMatMulU8I8_htp.so
libQnnHmxMatMulU8I8_cpu.so
u8i8_ctx.bin
runtime_inputs_u8/act_u8i8*.raw
```

and runs `qnn-net-run` with:

```text
QnnHmxMatMulU8I8InterfaceProvider
```

## Perf Method

Use the custom perf reader:

```bash
python scripts/perf_hmx_u8i8_matmul.py \
  example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/out/u8i8_chain_256
```

Compare with native:

```bash
python scripts/perf_hmx_u8i8_matmul.py <custom_out_dir> \
  --compare <native_out_dir>
```

Metric interpretation:

| Metric | Meaning | Current use |
|---|---|---|
| cycles / `dur` | HTP op timeline duration | user-visible hot-op latency |
| packets / `pkts` | committed Hexagon packets, inferred from cycles/cpp | best gap metric |
| `cpp` | cycles per packet | sanity check for stalls/issue behavior |

For the current 256^3 hot-op gap, packets are the most useful number, but the native side must be aggregated at QNN-op level.

Latest verified hot-op numbers on device, 2026-05-05:

| Path | cycles | packets | cpp | Meaning |
|---|---:|---:|---:|---|
| native kernel-only `ConvLayer_s1.opt` | 1147 | 346 | 3.315 | HMX body event only |
| native QNN-op aggregate `MatMul_*` | 1444 | 451 | 3.201 | `bias_to_vtcm + weights_to_vtcm + DmaCheckpointSet + ConvLayer_s1.opt` |
| custom `HmxU8I8ToU8MatMul` precompute default | 1257 | 349 | 3.600 | QHPI-prepared state + tiny descriptor stitching + HMX body |

Gap:

```text
custom - native kernel-only = +110 cycles, +3 packets
custom - native QNN-op aggregate = -187 cycles, -102 packets
```

The corrected explanation is now stronger than the earlier wrapper diagnosis:
the HMX body was already correct, and the apparent `+125 pkts` came from
comparing the old custom callback against only native `ConvLayer_s1.opt`.
Native QNN hides setup in sidecar HTP ops (`bias_to_vtcm`, `weights_to_vtcm`,
`DmaCheckpointSet`) and in graph-load preparation.  After moving the custom
lookup/shape/table preparation into QHPI precompute and consuming QNN's existing
Crouton block tables directly, the hot custom op is only `+3 pkts` from native
kernel-only.  The old wrapper path measured `1794 cycles / 471 pkts`; that is
now historical evidence for where the gap was, not the current implementation.

## Do Not Reopen Without New Evidence

These branches were intentionally removed from the active implementation:

- old row-major custom kernels
- old pack/copy/untile helper kernels
- spike/probe-only kernels
- dlsym/native-symbol swap paths
- old unaligned kernel variants
- PMU-heavy normal measurement paths
- descriptor parameter sweep branches
- compatibility aliases for old public names

New DLC/context binaries should be regenerated with the current XML and op package.

## Validation Checklist

Run from repo root:

```bash
python - <<'PY'
from pathlib import Path
patterns = [
    "Bbb" + "KMajor",
    "MatMul" + "V8",
    "Phase" + "3",
    "V8" + "C8",
    "MatMul" + "V8" + "Package",
    "HmxMatMul" + "Phase" + "3" + "Package",
]
roots = [Path("example/qnn_hmx_matmul_u8i8"), Path("docs"), Path("Agent"), Path("scripts/perf_hmx_u8i8_matmul.py")]
hits = []
for root in roots:
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if any(p in text for p in patterns):
            hits.append(str(path))
if hits:
    print("\n".join(hits))
    raise SystemExit(1)
PY

bash example/qnn_hmx_matmul_u8i8/build.sh
bash example/qnn_hmx_matmul_u8i8/build_x86.sh

cd example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8
SKIP_DEVICE=1 bash run_u8i8_chain.sh
```

Expected:

```text
old-name rg: no matches
build.sh: HTP and CPU libs generated
build_x86.sh: x86 lib generated
run_u8i8_chain.sh: DLC and context binary generated
ctx nodes include QnnHmxMatMulU8I8Package::HmxU8I8ToU8MatMul
```
