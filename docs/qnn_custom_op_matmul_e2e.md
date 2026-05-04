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
  -> ctxgen loads x86 op package and resolves QHPI registration
  -> HTP runtime inserts surrounding built-ins:
       q::*InputSlice
       q::ForceFormat_Crouton
       q::ConvLayer.opt.weights_to_vtcm
  -> QHPI calls hmx_u8i8_to_u8_matmul_kernel()
  -> wrapper reads tensor handles/block tables
  -> wrapper builds native HMX descriptors
  -> owned V73DEEP inline asm runs
```

这个 boundary 是当前 gap 的核心背景：我们可以运行同一个低层 HMX body，但 custom op wrapper 仍然要付 QHPI callback、tensor lookup、descriptor/table build 的成本；native built-in path 的内部 wrapper 和 scheduler 集成更深。

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

The wrapper derives `S` from the activation block table length, builds activation/output pointer tables, patches the mask descriptor, then calls:

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

For the current 256^3 hot-op gap, packets are the most useful number.

Latest verified hot-op numbers on device, 2026-05-05:

| Path | cycles | packets | cpp |
|---|---:|---:|---:|
| native QNN MatMul lowered to `ConvLayer_s1.opt` | 1147 | 346 | 3.315 |
| custom `HmxU8I8ToU8MatMul` | 1679 | 471 | 3.565 |

Gap:

```text
custom - native = +532 cycles, +125 packets
```

The simple explanation is: the HMX body has been replicated; the remaining gap is the custom-op invocation/wrapper and descriptor/table setup path around that body.  A post-cleanup scalar table-copy regression measured 550 packets; restoring the old Hexagon-only 128B HVX copy path for the canonical 256^3 C8 table layout brings the hot op back to 471 packets, essentially aligned with the earlier 468-packet state.  Cycles/cpp move run-to-run; packet count is the more stable gap signal.

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
