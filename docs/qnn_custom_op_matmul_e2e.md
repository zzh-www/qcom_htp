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

## Core Design Principle

简而言之：

```text
HMX only computes.
Everything except MAC compute should already be prepared in TCM/VTCM form.
```

Native QNN 的 MatMul 不是让最终 kernel 自己理解普通矩阵，而是先把 MatMul 变成 HMX-native Conv1x1 输入状态：

```text
ordinary MatMul tensors
  |
  |  QNN / converter / graph-load / runtime sidecars
  |  - pack weight into K-major HMX tiles
  |  - fold bias into native bias blocks
  |  - format activation/output as Crouton_8 TCM blocks
  |  - move static data via weights_to_vtcm / bias_to_vtcm
  |  - prepare pointer tables, mask state, and tile counts
  v
HMX-native prepared state
  |
  |  hot event
  |  - stitch tiny native descriptors
  |  - call V73DEEP HMX body
  v
MAC compute only
```

Therefore the custom op rule is:

```text
do not query tensor metadata, recover shape, repack data, copy QNN tables,
or patch mask state inside the profiled hot callback.
```

Those tasks belong in `gen_u8i8_chain.py`, converter/QNN layout planning, QNN sidecar ops, or QHPI precompute.  The hot `HmxU8I8ToU8MatMul` callback should be a compute event, not an input-preparation event.

## Kernel Design And Computation Flow

The kernel is designed as a MatMul-shaped frontend over QNN's native HMX Conv1x1 compute contract.  The important idea is not "write a generic matrix multiply in C++".  The idea is:

```text
MatMul semantics
  A[M, K] x W[K, N] -> Y[M, N]

Conv1x1 view used by QNN/HMX
  M output positions
  K input channels
  N output channels
  1x1 filter
```

The math is the same:

```text
Y[m, n] = sum_k A[m, k] * W[k, n]
```

but the hardware path is different.  HMX wants fixed tiles, fixed descriptor fields, fixed pointer-table traversal, and a native accumulator/convert pipeline.  Therefore the runtime kernel does not see ordinary tensor metadata.  It sees a native Conv1x1 ABI:

```text
r0 = out_desc
r1 = act_desc
r2 = packed_weight
r3 = folded_bias_record
r4 = mask_desc
r5 = extra_param
```

The hot custom-op callback is only an ABI adapter:

```text
prepared QHPI state
  - Direct TCM bias pointer
  - Direct TCM packed-weight pointer
  - copied activation/output Crouton pointer tables
  - recovered M_t/N_t/K_t tile counts
  - pre-initialized HMX mask
        |
        v
hot callback
  - build out_desc on stack
  - build act_desc on stack
  - create extra_param[2] = {1, 0}
  - call our_v73deep_kernel(...)
        |
        v
owned V73DEEP inline asm
```

The HMX body then runs the native Conv1x1 compute flow:

```text
prologue
  read out_desc / act_desc / mask_desc
  derive loop counts, strides, runtime masks

main K-MAC loop
  load activation block pointer from act table
  load activation tile from TCM
  load K-major weight tile
  feed both into HMX deep MAC pipeline

convert / store
  load native bias/scale record
  convert accumulator to u8
  store u8 tile to Crouton output block

epilogue
  clear HMX accumulator
  restore registers
  return
```

The core HMX packets are:

```text
activation.ub = mxmem(...):deep:cm
weight.b      = mxmem(...):deep
```

Those packets are the actual compute feed.  They are followed by:

```text
bias = mxmem2(...)
cvt.ub = acc(...)
mxmem(...):cm = cvt
```

So the accumulator is not manually read in C++.  The HMX convert path applies the prepared bias/scale/baseline record and writes quantized u8 output.

Quantization follows the same principle: no dynamic scale math in the hot kernel.  The native bias record is prepared per N tile:

```text
256 bytes per N tile:

bytes 0..127
  32 x (fp16 scale, fp16 baseline)

bytes 128..255
  32 x int32 effective_bias
```

The HMX convert formula can be understood as:

```text
out_u8 = clamp(acc * scale_fp16 / 512 + baseline)
```

For the current replica test flow:

```text
runtime scale = 1.0
scale_fp16    = 512.0
baseline      = 0
ACT_ZP        = 128
```

The activation zero point is folded into the int32 bias before runtime:

```text
(act_u8 - ACT_ZP) @ W + bias_q
  = act_u8 @ W + (-ACT_ZP * sum_k(W[k, c]) + bias_q[c])

effective_bias[c] = -ACT_ZP * sum_k(W[k, c]) + bias_q[c]
```

If a future flow uses real per-channel quantization, it should still use the same kernel ABI:

```text
runtime_scale[c] = input_scale * weight_scale[c] / output_scale
scale_fp16[c]    = 512.0 * runtime_scale[c]
baseline[c]      = output_zp << 7
effective_bias[c] = bias_q[c] - input_zp * sum_k(W[k, c])
```

The HMX hot body should remain unchanged.  Only the prepared bias/scale record changes.

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
     copies small activation/output Crouton block tables, pre-initializes
     the HMX mask, and stores tile counts
  -> QHPI hot callback stitches the tiny native descriptors on stack
  -> owned V73DEEP inline asm runs
```

这个 boundary 是 gap 的核心背景，但 perf 读法要小心：native MatMul 会拆成 `bias_to_vtcm`、`weights_to_vtcm`、`DmaCheckpointSet`、`ConvLayer_s1.opt` 等多个 HTP 事件。早期 custom 版本把 QHPI lookup、shape recovery、pointer-table copy 都计在 `HmxU8I8ToU8MatMul` 热事件里；当前默认实现用 QHPI precompute 把这些准备工作提前到 graph-load/context-load 阶段，hot event 基本只剩 descriptor stitching + HMX body。

## Implementation Mapping

当前实现按上面的原则分层：

| Stage | File / owner | What happens |
|---|---|---|
| generator | `standard_flow/custom_u8i8/gen_u8i8_chain.py` | emits K-major HMX weight bytes and folded native bias bytes |
| converter / ctxgen | XML + converter op package | declares Direct/Crouton tensor layouts and resolves QHPI precompute registration |
| QNN sidecar ops | QNN runtime | handles format/input movement and static `weights_to_vtcm` events |
| QHPI precompute | `HmxU8I8ToU8MatMulOp.cpp` | records Direct TCM bias/weight pointers, copies small Crouton pointer tables, pre-initializes mask state, stores tile counts |
| hot callback | `HmxU8I8ToU8MatMulOp.cpp` | builds two small descriptors on stack and enters `our_v73deep_kernel` |
| HMX body | `v73deep_conv1x1_kernel.inc` | performs the V73DEEP MAC compute body |

The important split is this:

```text
prepared outside hot path:
  bias layout
  weight layout
  activation/output block-table values
  shape/tile counts
  mask state

inside hot path:
  out_desc
  act_desc
  extra_param[2]
  call owned V73DEEP body
```

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
mask is pre-initialized during QHPI precompute; r5 passes extra_param directly
```

The QHPI precompute function derives `S` from the activation block table length and stores the prepared state:

```text
precomputed_data:
  bias_bytes       -> QNN-prepared Direct TCM bias
  wt_pack          -> QNN-prepared Direct TCM K-major weight
  act_qhpi_table   -> copied activation block table for small validated shapes
  out_qhpi_table   -> copied output block table for small validated shapes
  M_t/N_t/K_t      -> recovered tile counts
```

The hot callback no longer queries QHPI tensors, copies pointer tables, or patches the mask. It builds the small native descriptors on stack, then calls:

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

For the current 256^3 hot-op gap, `dur` is the final latency number.  `pkts` explains whether custom is doing extra work; since custom is now `-9 pkts` versus native kernel-only, the remaining cycle delta is not an extra-instruction problem.

Latest verified hot-op numbers on device, 2026-05-05:

| Path | cycles | packets | cpp | Meaning |
|---|---:|---:|---:|---|
| native kernel-only `ConvLayer_s1.opt` | 1140 | 346 | 3.296 | HMX body event only |
| native QNN-op aggregate `MatMul_*` | 1452 | 451 | 3.220 | `bias_to_vtcm + weights_to_vtcm + DmaCheckpointSet + ConvLayer_s1.opt` |
| custom `HmxU8I8ToU8MatMul` precompute default | 1165 | 337 | 3.458 | QHPI-prepared state + copied pointer tables + tiny descriptor stitching + HMX body |

Gap:

```text
custom - native kernel-only = +25 cycles, -9 packets
custom - native QNN-op aggregate = -287 cycles, -114 packets
```

The corrected explanation follows the design principle above.  The old wrapper
path measured `1794 cycles / 471 pkts` because it did input/layout preparation
inside the profiled callback.  Comparing that callback against only native
`ConvLayer_s1.opt` created the apparent `+125 pkts` gap.

Native QNN was not doing less total work; it was accounting for preparation in
sidecar HTP ops (`bias_to_vtcm`, `weights_to_vtcm`, `DmaCheckpointSet`) and in
graph-load preparation.  After moving custom lookup/shape/table/mask work into
QHPI precompute and using precomputed local Crouton table copies, the hot custom
op is `9 pkts` lower than native kernel-only.  That means the remaining cycle
delta is not an extra-instruction or extra-compute problem.

Probe split for the retained default shape:

```text
HMX_U8I8_PROBE_CYCLES:
  kernel = 1074 cycles
  desc   =   29 cycles
  table  =    0 cycles
  qhpi   =    0 cycles
```

The owned inline-asm body is therefore not slower than native: `1074 < 1140`.
The remaining `+25 cycles` in chrometrace is the public QHPI callback/profiling
envelope plus tiny descriptor glue and issue/locality effects.  In this public
custom-op architecture there is no further user-code matmul work to remove.

Cycle-gap follow-up: explicit dcfetch of QNN tables reduced some cycles but cost
too many packets; graph-load table copies are the retained version.  Putting
descriptor/mask/extra state in the existing Direct-TCM scratch is bit-exact but
much slower (`~1700 cycles`, high cpp), and pre-writing those records during
QHPI precompute fails at device context creation. Raising the owned kernel entry
alignment above 64B, static mask/extra variants, and precomputed descriptor
records all worsened cycles.

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
