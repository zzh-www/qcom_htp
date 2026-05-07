# W4A16 QNN Native Path

This note describes the native QNN implementation path for the canonical
256^3 W4A16 artifact:

`example/qnn_matmul_profile/output_native_w4a16_conv_ref_256/`

Use this as the first reference before changing the custom
`HmxU16I4ToU16MatMul` path.  The custom op should be aligned to this native
path, not to an analytic formula.

The older
`example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/`
artifact is historical only: it used float runtime input/output and did not
record the required layout-preservation converter flags.

## Current Direction

Treat W4A16 as a native-implementation analysis task before treating it as a
custom-op tuning task.  The current output mismatch is not explained by the
math reference, and the embedded deep HMX body is already byte-identical to the
native skel slice.  The unresolved part is the QNN native route into that body:
converter/ctxgen carrier lowering, runtime tensor-object metadata, descriptor
builder state, mask-helper arguments, and wrapper descriptor-table looping.

Until those native fields are decoded or dumped, do not spend more cycles on
single-field descriptor, mask, control-word, or row4-order sweeps.  A future
custom change should first name the exact native boundary it reproduces.

Current pivot: analyze QNN native first, then align.  The public QHPI tensor
surface visible after native Conv is now known to be a layout-restored export
surface, not the internal HNH compute surface.  Converter output-layout flags
do not move a custom diagnostic op inside the native `ConvLayer_s1.opt`
boundary.  The next useful work is therefore native wrapper/descriptor evidence,
not another custom input/output layout probe.

## Native-First Rule

Do not start from another descriptor/mask/packing sweep.  First identify which
native boundary is being copied:

1. source model and quantization override;
2. converter graph-before tensor contract;
3. ctxgen-lowered HTP nodes and sidecars;
4. `ConvLayer_s1.opt` HMX tensor boundary;
5. skel wrapper, descriptor builder, mask helper, and deep body;
6. performance scope: kernel-only, native Conv group, or end-to-end execute.

Only after a custom change can name the native boundary it is reproducing should
it be considered an alignment attempt.

## 2026-05-08 Native Path Audit

Current audited conclusion: QNN native W4A16 is a layered implementation path,
not a single tensor layout.  The verified route is:

```text
float ONNX Conv + quant_overrides
  -> converter graph-before: UFixed16 activation, UFixed8 W4 carrier metadata
  -> ctxgen: SFixed8 W4 sidecar, Int32 bias/control sidecar, HNH UFixed16 surface
  -> ConvLayer_s1.opt: native wrapper builds descriptors from tensor objects
  -> hmx_v73_convhnh1x1_stride1 / hmx_v73_convhnh1x1deep_stride1
```

The visible final HTP tensor contract is already known:

```text
activation  UFixed16 [1,8,32,256]
weight      SFixed8  [1,1,128,256]
bias        Int32    [1,8,1,128]
control     Int32    [1]
extra ctrl  Int32    [1,1,1,3]
output      UFixed16 [1,8,32,256]
```

That contract is necessary but not sufficient.  The native wrapper passes QNN
internal tensor-object table pointers and metadata-derived descriptor fields to
the HMX body.  Bottom mapping exposes the visible tensor type/shape but not the
runtime values behind:

- `activation.data` and `output.data`;
- `activation.meta[0x04/0x18/0x1c/0x20]` and output counterparts;
- the post-builder stack record at `base = r29 + 0x30`;
- the W4 mask-helper dynamic arguments;
- the wrapper descriptor-advance tuple `r23/r24/r27`.

A custom-side QHPI tensor-object dump was added to `HMX_W4A16_DESC_DUMP` as a
negative boundary check.  The captured custom activation/output QHPI object
words are opaque handle-like values, unlike the readable pointer/metadata words
from the post-Conv `HmxW4A16TensorDump` diagnostic.  They cannot replace the
native `ConvLayer_s1.opt` wrapper metadata listed above.

The post-Conv tensor-dump diagnostic proves a different boundary: QNN exposes a
layout-restored public QHPI surface, `UFixed16 [1,256,1,256]`, to a custom op
placed after native Conv.  It does not expose the internal HNH compute surface.
The host output-layout sweep confirmed that converter output-layout flags do
not move a custom diagnostic op inside `ConvLayer_s1.opt`.

Therefore the next alignment evidence must be a native wrapper/descriptor
record, or a static derivation that accounts for the same metadata fields.  A
custom run that changes only y-stride, mask lane, row4 order, output rank, or
public QHPI table expansion is not native-path evidence unless it names which
field in the native wrapper record it reproduces.

## Evidence Map

| Evidence | Path | What it proves |
|---|---|---|
| source ONNX | `conv.onnx` | Native starts as a float `Conv`, not a custom or integer MatMul op. |
| quant override | `quant_overrides.json` | A/Y use A16 quantization, W uses symmetric 4-bit param encoding. |
| converter graph-before | `ctx/conv_bottom_mapping_graph_before.json` | High-level lowering is `Convert -> Transpose -> Conv2d -> CastInt4ToInt8 -> Transpose -> Convert`; W is still `UFixed8` metadata here. |
| final HTP graph | `ctx/conv_bottom_mapping.json` | Ctxgen expands the graph to 210 HTP nodes and introduces `weights_to_vtcm`, `bias_to_vtcm`, `ForceFormat_Crouton`, and `ConvLayer_s1.opt`. |
| context binary | `ctx/conv_ctx.bin` | Holds prepared static records for the W4 sidecar, bias sidecar, and small control tensors. |
| native output oracle | `device_out/Y.raw` | The custom output oracle; analytic formulas are secondary diagnostics. |
| optrace products | `optrace/` | Standard performance source: timeline, profile text, QHAS summary, decoded `summary.json`. |
| native Conv tensor dump | `example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/out/native_conv_tensor_dump_256/` | Diagnostic graph that keeps native W4 Conv lowering and records the QHPI tensor surface exposed to a custom op after Conv/layout restore. |
| host layout flag sweep | rerun with `gen_native_conv_tensor_dump.py` plus converter output-layout flags | Confirms `HmxW4A16TensorDump` still sees `UFixed16 [1,256,1,256]`; output-layout flags do not expose the internal `[1,8,32,256]` HNH surface to a custom op. |
| skel evidence | `Agent/qnn_re/skel_text_full.S` | Shows the HNH wrapper, descriptor builder, W4 mask helper, and final deep-body entry. |

## Implementation Path At A Glance

Native W4A16 is not a direct call from a high-level MatMul tensor into the HMX
body.  The implementation path is:

```text
float ONNX Conv
  + quant_overrides: A/Y A16, W 4-bit symmetric
    -> converter graph-before
       Convert(A) -> Transpose -> Conv2d -> Transpose -> Convert(Y)
       W appears as CastInt4ToInt8 / UFixed8 at this stage
    -> ctxgen HTP graph
       weights_to_vtcm produces SFixed8 [1,1,128,256]
       bias_to_vtcm produces Int32 [1,8,1,128]
       ForceFormat_Crouton produces UFixed16 [1,8,32,256]
    -> ConvLayer_s1.opt runtime event
       QNN wrapper builds native descriptors and mask state
    -> hmx_v73_convhnh1x1_stride1
       mask bit 5 selects hmx_v73_convhnh1x1deep_stride1
```

The same path as a stage table:

| Stage | Native owner | Boundary to inspect |
|---|---|---|
| ONNX | model generator | `A FLOAT [1,256,1,256]`, `W FLOAT [256,256,1,1]`, `Y FLOAT [1,256,1,256]`. |
| quantization | `quant_overrides.json` | A/Y A16 scale/offset; W symmetric 4-bit scale/offset. |
| converter | DLC / graph-before | Conv still sees an 8-bit W carrier plus 4-bit encoding metadata. |
| ctxgen | QNN HTP lowering | Static W4 and bias records become native sidecar tensors. |
| runtime HTP graph | `q::*` built-ins | Format conversion and sidecar movement are separate events around the hot Conv. |
| HMX wrapper | skel `0x3ddc60` path | Small native descriptors and mask words are built from QNN tensor metadata. |
| deep body | `hmx_v73_convhnh1x1deep_stride1` | HMX consumes only prepared descriptors and sidecars. |

That stage boundary matters.  The graph-before weight tensor is still an
8-bit carrier with 4-bit quantization metadata; the signed packed HNH sidecar
exists only after ctxgen's native Conv lowering.  A custom op that merely asks
for `SFixed8` in XML/converter is not guaranteed to enter the same lowering
path.

## Native Runtime Spine

The runtime path under `ConvLayer_s1.opt` has a specific spine in the skel
binary.  This is the path a custom implementation must reproduce before output
or performance parity is a meaningful claim:

```text
q::ConvLayer_s1.opt optrace event
  -> HTP op trampoline calls 0x3ddc60 at 0x3f2bd0
     args are QNN tensor objects/control, not flat buffers
  -> wrapper 0x3ddc60 computes metadata products from tensor objects
  -> descriptor builder 0x3d9920 writes a stack record at base = r29 + 0x30
     base+0x08  prepared W4 pointer
     base+0x0c  prepared bias/control pointer
     base+0x10  activation descriptor
     base+0x28  output descriptor
     base+0x48  mask descriptor
  -> builder calls set_hmx_params_convw4b1x1 at 0x289380 for the W4 1x1 mask
  -> wrapper selects HNH 1x1 call site 0x3dde78
     r0 = base+0x28, r1 = base+0x10, r2 = W, r3 = bias,
     r4 = base+0x48, r5 = control pointer
  -> hmx_v73_convhnh1x1_stride1 at 0x2fcd80
     mask word +0x30 bit 5 jumps to deep body 0x2fdb80
  -> wrapper tail 0x3de060 advances descriptor table pointers and may repeat
```

Two consequences follow from this spine:

1. Matching the deep-body bytes or W4 sidecar bytes is necessary but not
   sufficient.  Native also relies on QNN's internal tensor-object metadata and
   table pointers.
2. A custom "call twice" patch is not native by itself.  Native loop count and
   descriptor advances come from tensor metadata (`r23/r24/r27`), not from a
   visible `[1,8,32,256]` shape alone.

## Known And Unknown Native State

| State | Current status | Evidence / blocker |
|---|---|---|
| Source model route | Proven | Float ONNX `Conv` plus quant overrides; W4 is introduced by quant metadata, not an ONNX int4 tensor. |
| Graph-before W carrier | Proven | `ctx/conv_bottom_mapping_graph_before.json` shows `UFixed8 [1,1,256,256]` around `CastInt4ToInt8`; the signed compact carrier is not present yet. |
| Ctxgen sidecar lowering | Proven at tensor surface | Final HTP graph has `weights_to_vtcm`, `bias_to_vtcm`, `ForceFormat_Crouton`, and `ConvLayer_s1.opt`; W becomes `SFixed8 [1,1,128,256]`. |
| Prepared W4 bytes | Proven | `W4_PACK_ORDER=native_nmajor_k4_lohi` reproduces the 32768-byte native sidecar. |
| Prepared no-bias/control bytes | Proven | `native_a16_nobias` reproduces the 2048-byte repeated `0x80008000` sidecar. |
| Deep HMX body bytes | Proven | Extracted `hmx_v73_convhnh1x1deep_stride1` is 804 bytes and diffs empty against the embedded custom inc. |
| Wrapper call shape | Proven statically | `0x3ddc60 -> 0x3d9920 -> 0x3dde78 -> 0x2fcd80 -> 0x2fdb80`, with descriptor fields rooted at `base = r29 + 0x30`. |
| Wrapper descriptor loop | Proven statically | Tail `0x3de060` advances output and activation descriptor table bases using `r24` and `r27` until `r26 == r23`. |
| Runtime tensor-object metadata values | Unknown | Offsets such as `activation.meta[0x04]`, `[0x18]`, `[0x1c]`, `[0x20]` and output counterparts are not exposed by bottom mapping. |
| Native internal table layout | Unknown | `activation.data` / `output.data` point to QNN internal HNH tables; current custom code expands public QHPI tables instead. |
| Full W4 mask-helper argument tuple | Partially decoded | `arg1=0x70b` and the helper branch are known, but coupled dynamic args are metadata-derived and not yet dumped. |
| Public custom path to signed W4 carrier | Unsolved | Current custom boundary still reaches QHPI as `QUInt8` for W4 even when bytes match. |

The next useful evidence is therefore a native runtime metadata/descriptor dump,
not another guessed constant.  The minimum dump should capture wrapper args,
tensor-object metadata words, `activation.data` / `output.data`, the
post-builder stack record at `base+0x08..0x80`, mask words, and the final
`r23/r24/r27` loop tuple for the canonical 256^3 native artifact.

## Source Model

The ONNX model is intentionally simple:

| Item | Native source form |
|---|---|
| Input | `A`, `FLOAT`, `[1,256,1,256]` |
| Weight initializer | `W`, `FLOAT`, `[256,256,1,1]` |
| Op | `Conv`, `pads=[0,0,0,0]`, `strides=[1,1]` |
| Output | `Y`, `FLOAT`, `[1,256,1,256]` |

W4 is introduced by `quant_overrides.json`, not by an ONNX int4 tensor:

| Tensor | Native quantization contract |
|---|---|
| `A`, `Y` | 16-bit signed-style integer contract, scale `3.051850947599719e-05`, offset `-32768` in the override, QNN carrier `UFixed16` (`data_type=1046`) |
| `W` | 4-bit symmetric integer contract, scale `1/7`, offset `-8` in the override |

The converter graph-before view contains:

```text
Convert(A float -> UFixed16)
Transpose A_0231
Conv2d conv1x1
CastInt4ToInt8 CastInt4ToInt8_1d   # optimized out
Transpose Y_0231
Convert(Y UFixed16 -> float)
```

In the graph-before JSON, the Conv weight path is still `UFixed8`
(`data_type=1032`) with `[1,1,256,256]` logical dimensions.  The important
point is that native Conv starts from float weights plus a 4-bit param encoding.
Ctxgen then owns the signed W4 carrier and sidecar lowering.

## Lowered HTP Path

After ctxgen, the graph is expanded into 210 HTP nodes.  Most nodes are
format-conversion work around the actual Conv core:

| HTP node class | Count |
|---|---:|
| `q::SlicePad_shape_inplace` | 48 |
| `q::Quantize` | 32 |
| `q::*InputSlice` | 32 |
| `q::Dequantize` | 32 |
| `q::ForceFormat_Flat` | 32 |
| `q::Transpose.2D` | 16 |
| `q::Transpose_impl` | 8 |
| `q::Reshape` | 2 |
| `q::Concat` | 2 |
| `q::ForceFormat_Crouton` | 2 |
| `q::ConvLayer.opt.weights_to_vtcm` | 1 |
| `q::ConvLayer.opt.bias_to_vtcm` | 1 |
| `q::ConvLayer_s1.opt` | 1 |
| `q::ConvLayer.opt.activations_to_vtcm` | 1 |

The graph-before view has only 6 nodes and 15 tensors.  The lowered HTP view has
210 nodes and 260 tensors.  That expansion is the native implementation, not
incidental tracing noise: it is where QNN creates the sidecar movement,
Crouton/HNH surfaces, and final HMX call boundary.

The core Conv tensor contract at the final HMX node is:

| Node / tensor | Native type and shape | Notes |
|---|---|---|
| `weights_to_vtcm` input | `SFixed8` (`data_type=776`), `[1,1,128,256]` | Static prepared W4 carrier. |
| `weights_to_vtcm` output | `SFixed8` (`data_type=776`), `[1,1,128,256]` | Written to VTCM, 32768 bytes. |
| `bias_to_vtcm` input | `Int32` (`data_type=50`), `[1,8,1,128]` | Static bias/control sidecar, 4096 bytes. |
| `bias_to_vtcm` output | `Int32` (`data_type=50`), `[1,8,1,128]` | Written to VTCM. |
| `ConvLayer_s1.opt` input 0 | `UFixed16` (`data_type=1046`), `[1,8,32,256]` | HMX activation surface after `ForceFormat_Crouton`. |
| `ConvLayer_s1.opt` input 1 | `SFixed8` (`data_type=776`), `[1,1,128,256]` | Prepared W4 sidecar. |
| `ConvLayer_s1.opt` input 2 | `Int32` (`data_type=50`), `[1,8,1,128]` | Prepared bias/control sidecar. |
| `ConvLayer_s1.opt` input 3 | `Int32` (`data_type=50`), `[1]` | Small control tensor. |
| `ConvLayer_s1.opt` input 4 | `Int32` (`data_type=50`), `[1,1,1,3]` | Extra small control tensor. |
| `ConvLayer_s1.opt` output | `UFixed16` (`data_type=1046`), `[1,8,32,256]` | HMX output surface. |

`ConvLayer_s1.opt` reports `mem_vtcm_read=165888` and
`mem_vtcm_write=131072`, matching:

```text
activation  [1,8,32,256] u16  = 131072 bytes
weight      [1,1,128,256] i4   =  32768 bytes
bias/control                 =   4096 bytes
output      [1,8,32,256] u16  = 131072 bytes
```

## Prepared Sidecars

The canonical native context binary is:

```text
ctx/conv_ctx.bin
size:   90112 bytes
sha256: b48db57c34c02741ded507eda349a4ca7e094c92302d28e573eddbaeef177e91
```

Current clean-context sidecar note:

| Region | Size | SHA256 | First bytes |
|---|---:|---|---|
| `conv_ctx.bin+0xbd00` candidate | 32768 | `6f1016e71e87b727032c17528eaae834dab48017afbe5feae93703cd01a25bf6` | current best imported candidate, not a complete semantic fix |

The historical float-I/O artifact had a byte-for-byte generated W4 region at
`conv_ctx.bin+0xcc00`; keep that only as a packing-order diagnostic.  The W4
sidecar order decoded from that historical region is:

```text
N32 tile -> K8 group -> n-in-tile -> k4 pair
byte pairs: (k+0,k+4), (k+1,k+5), (k+2,k+6), (k+3,k+7)
twos-complement nibbles
```

`W4_PACK_ORDER=native_nmajor_k4_lohi` reproduces this 32768-byte native stream
byte-for-byte for the old artifact.  It does not reproduce the full clean
native context contract.

## Runtime And Performance Path

The standard native optrace directory is:

`example/qnn_matmul_profile/output_native_w4a16_conv_ref_256/optrace/`

For the captured run:

| Scope | Cycles / time |
|---|---:|
| HTP timeline span | `313032` cycles |
| Sum of pid0 events | `542923` cycles |

The `conv1x1` group in `optrace/summary.json` breaks down as:

| Event | Cycles | Packets | Notes |
|---|---:|---:|---|
| `q::ConvLayer.opt.weights_to_vtcm` | `3323` | n/a | Static W4 sidecar movement. |
| `q::ConvLayer.opt.bias_to_vtcm` | `878` | n/a | Static bias/control movement. |
| `q::ForceFormat_Crouton` | `7341` | n/a | Conv format work. |
| `q::ConvLayer_s1.opt` | `7893` | n/a | Closest native HMX kernel-only comparator. |
| native `conv1x1` QNN-op aggregate | `37287` | n/a | Full native Conv group in qnn-op grouping. |

So there are three different performance scopes:

1. Kernel-only native comparator: `q::ConvLayer_s1.opt = 7893` cycles.
2. Native Conv group comparator: all `conv1x1` HTP events, about `37287` cycles
   in the qnn-profile grouped stat.
3. End-to-end execute comparator: qnn-net-run execute stats, including input
   conversion, transposes, output dequant/export, RPC, and accelerator wait.

Custom W4A16 performance claims must name which scope they compare against.

## Native Conv Tensor-Dump Diagnostic

The current native-first diagnostic flow is:

```bash
bash example/qnn_hmx_matmul_w4a16/build_x86.sh
bash example/qnn_hmx_matmul_w4a16/build.sh
OUT_DIR=example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/out/native_conv_tensor_dump_256 \
STRICT_OPTRACE=1 \
bash example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_native_conv_tensor_dump.sh
```

The flow is intentionally separate from the custom MatMul runner:

1. `gen_native_conv_tensor_dump.py` clones the canonical native `conv.onnx`,
   keeps the original Conv output `Y`, appends a side-branch
   `hmx::HmxW4A16TensorDump(Y -> D)`, and copies native input/quant files.
2. `qairt-converter` uses the same native W4 carrier contract as the canonical
   artifact: no `--pack_4_bit_weights`, and `--preserve_io_datatype A Y` so
   the original input/output stay float while dump output `D` is exported as
   native UFixed16.
3. `qnn-context-binary-generator` keeps the native W4 Conv lowering and adds
   one custom QHPI dump op.
4. Device run pulls `device_out/D.raw`, decodes standard optrace into
   `optrace/`, and parses `device_out/tensor_dump.{json,txt}`.

The latest captured diagnostic artifact is:

`example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/out/native_conv_tensor_dump_256/`

Ctxgen confirms both boundaries are present:

| Boundary | Tensor contract |
|---|---|
| `q::ConvLayer_s1.opt` input activation | `UFixed16 [1,8,32,256]` |
| `q::ConvLayer_s1.opt` weight sidecar | `SFixed8 [1,1,128,256]` |
| `q::ConvLayer_s1.opt` bias/control sidecar | `Int32 [1,8,1,128]` in this side-branch diagnostic graph |
| `q::ConvLayer_s1.opt` small control | `Int32 [1]` plus an added diagnostic `Int32 [1,1,1,3]` side input in this graph |
| `q::ConvLayer_s1.opt` output | `UFixed16 [1,8,32,256]` |
| `HmxW4A16TensorDump` input/output | `UFixed16 [1,256,1,256]` |

`tensor_dump.json` reports the QHPI surface exposed to the custom dump op:

| Field | Value |
|---|---|
| magic/version | `0x48385444` / `1` |
| element/layout/placement | `2` / `10` / `1` |
| quant | zero offset `32768`, step `3.0518509447574615e-05` |
| logical shape | `[1,256,1,256]` |
| padded shape | `[1,256,1,256]` |
| block shape | `[1,8,2,32]` |
| block table length | `256` |
| first block pointers | `0x04480000`, `0x04480800`, `0x04481000`, ... |

This is not yet the internal `ConvLayer_s1.opt` wrapper descriptor dump, and
the added side branch still perturbs part of ctxgen's prepared sidecar contract
(`bias_to_vtcm` is `[1,8,1,128]` here, while the canonical artifact records
`[1,8,1,64]`).  It shows the tensor object that QNN exposes to a custom op
after native Conv and layout restoration.  Use it to align the public QHPI
contract and to prevent custom descriptor guesses from drifting away from
QNN's real tensor layout.  The remaining hard target is still the native
wrapper state under `0x3ddc60`: wrapper args, tensor-object metadata words,
post-builder stack descriptors, mask words, and the `r23/r24/r27`
descriptor-loop tuple.

For this diagnostic graph, optrace is a trace-validation artifact rather than
a performance target.  The dump op and output export dominate runtime:

| Scope | Cycles |
|---|---:|
| HTP timeline span | `362211` |
| Sum of pid0 events | `692866` |
| `HmxW4A16TensorDump` | `209080` |
| diagnostic `conv1x1` QNN-op aggregate | `32532` |
| diagnostic `q::ConvLayer_s1.opt` | `6122` |

Continue using the canonical native artifact's `optrace/summary.json` for
performance acceptance.  Use this diagnostic artifact for native-path tensor
evidence.

Do not use the dumped post-Conv tensor surface as the custom HNH compute
surface.  A follow-up diagnostic added `OP_INPUT_LAYOUT=native_conv_surface` to
feed `HmxU16I4ToU16MatMul` with activation/output `UFixed16 [1,256,1,256]`,
matching the tensor dump's public QHPI shape.  Host conversion and ctxgen pass,
and the custom boundary receives native no-bias/control shape
`Int32 [1,8,1,64]`, but device execution fails before a valid output or
optrace:

`example/qnn_matmul_profile/output_codex_w4a16_native_conv_surface_real_256/`

That failure is a useful boundary result: the layout-restored QHPI tensor
visible after native Conv is an export/custom-op surface, not the internal HNH
descriptor surface consumed by `ConvLayer_s1.opt`.  Continue targeting the
wrapper metadata and stack descriptors under `0x3ddc60`.

A host-only converter layout sweep gives the same conclusion without running a
device kernel.  The sweep regenerated the native Conv plus tensor-dump graph and
ran qairt-converter/ctxgen with these output-layout variants:

| Variant | Layout flags | Dump-op input | Native Conv boundary |
|---|---|---|---|
| `default` | none | `UFixed16 [1,256,1,256]` | `UFixed16 [1,8,32,256]`, `SFixed8 [1,1,128,256]`, `Int32 [1,8,1,128]` |
| `d_nchw` | `D` source/desired `NCHW` | `UFixed16 [1,256,1,256]` | unchanged |
| `d_nhwc` | `D` source/desired `NHWC` | `UFixed16 [1,256,1,256]` | unchanged |
| `d_nchw_to_nhwc` | `D` source `NCHW`, desired `NHWC` | `UFixed16 [1,256,1,256]` | unchanged |
| `yd_nchw` | `Y` and `D` source/desired `NCHW` | `UFixed16 [1,256,1,256]` | unchanged |
| `yd_nhwc` | `Y` and `D` source/desired `NHWC` | `UFixed16 [1,256,1,256]` | unchanged |
| `yd_nchw_to_nhwc` | `Y` and `D` source `NCHW`, desired `NHWC` | `UFixed16 [1,256,1,256]` | unchanged |

This closes the converter-layout hypothesis: QNN only exposes the
layout-restored `Y/D` surface to the custom diagnostic op.  The native HNH
state must be recovered under `ConvLayer_s1.opt` itself.

## Skel Execution Path

Native `ConvLayer_s1.opt` eventually reaches the HNH HMX wrapper in
`Agent/qnn_re/skel_text_full.S`:

| Address | Role |
|---|---|
| `0x3ddc60` | V73 HNH wrapper reached by the native ConvLayer path. |
| `0x3d9920` | Descriptor builder called by the wrapper. |
| `0x289380` | `set_hmx_params_convw4b1x1`, native W4 mask helper. |
| `0x3dde78` | Final call site into `hmx_v73_convhnh1x1_stride1`. |
| `0x2fcd80` | `hmx_v73_convhnh1x1_stride1` entry. |
| `0x2fdb80` | `hmx_v73_convhnh1x1deep_stride1` deep HMX body. |

The owned custom body is byte-identical to the current SDK skel slice for this
symbol:

```bash
python3 scripts/extract_hmx_kernel_bytes.py \
  --symbol hmx_v73_convhnh1x1deep_stride1 \
  --out /tmp/w4a16_convhnh1x1deep.inc
diff -u /tmp/w4a16_convhnh1x1deep.inc \
  example/qnn_hmx_matmul_w4a16/src/v73deep_conv1x1_kernel.inc
```

The current audit produces an empty diff and extracts 804 bytes from
`0x2fdb80`.  Treat the embedded deep body as matched before spending more time
on kernel-byte replacement.

The wrapper uses stack storage rooted at `base = r29 + 0x30`:

| Stack field | Meaning at final call |
|---|---|
| `base+0x08` | Prepared weight pointer. |
| `base+0x0c` | Prepared bias/control pointer. |
| `base+0x10` | Activation descriptor passed as `r1`. |
| `base+0x28` | Output descriptor passed as `r0`. |
| `base+0x48` | Mask descriptor passed as `r4`. |

The final HMX call shape at `0x3dde78` is:

```text
r0 = base + 0x28   # output descriptor
r1 = base + 0x10   # activation descriptor
r2 = weight pointer
r3 = bias/control pointer
r4 = base + 0x48   # mask descriptor
r5 = original wrapper arg5 / control pointer
```

`hmx_v73_convhnh1x1_stride1` checks mask word `+0x30`; when bit 5 is set it
jumps directly to `hmx_v73_convhnh1x1deep_stride1`.  The deep body reads:

| Input | Fields consumed by deep body |
|---|---|
| activation descriptor | `+0x0` pointer table, `+0x4` pair count, `+0x8` table y stride |
| output descriptor | `+0x0` pointer table, `+0x4` table stride, `+0x8` y stride, `+0xc` tile/count selector, `+0x10` inner loop span, `+0x14` byte span |
| mask descriptor | `+0x0..0x18` mask words, plus `+0x30` used by the pre-entry selector |
| control pointer | first 32-bit word via `memw(r5++#0x4)` |

### Wrapper Descriptor Advance Loop

`0x3ddc60` does more than make a single descriptor and call the body.  After a
body call returns, the common tail at `0x3de060` advances descriptor table
pointers and can loop back to the HMX call-selection block:

```text
0x3de060: r26 += 1
0x3de064: r2 = memw(r29+#0x40)          # base+0x10, activation desc +0x00
0x3de068: memw(r29+#0x58) += r24        # base+0x28, output desc +0x00
0x3de06c: r2 = addasl(r2, r27, #2)      # activation table pointer += r27 * 4
0x3de070: p0 = cmp.eq(r26, r23)
0x3de074: if (!p0) jump:t 0x3ddd90
0x3de078: memw(r29+#0x40) = r2
```

So the native call state includes three loop fields in addition to the deep-body
descriptor fields:

| Loop field | Native role |
|---|---|
| `r23` | Number of wrapper HMX calls for this tensor metadata state. |
| `r24` | Output table-pointer advance, in bytes, applied to `out_desc+0x00`. |
| `r27` | Activation table-pointer advance, in 32-bit table entries, applied to `act_desc+0x00`. |

For the direct HNH 1x1 branch that reaches `0x3dde78`, the static source of
those fields is:

| Loop field | Static source before / after `0x3d9920` |
|---|---|
| `r23` | `activation.meta[0x04]`, loaded after the descriptor builder at `0x3ddd2c..0x3ddd30`. |
| `r24` | `4 * ((output.meta[0x20] >> 5) * (output.meta[0x1c] >> 2) * (output.meta[0x18] >> 3))`, computed before the descriptor builder as `r26`, then shifted by `vaslw(r27:26,#2)`. |
| `r27` | `(activation.meta[0x20] >> 5) * (activation.meta[0x1c] >> 2) * (activation.meta[0x18] >> 3)`, computed before the descriptor builder. |

Those metadata offsets are QNN internal tensor-object fields, not the public
bottom-mapping dimensions.  `ctx/conv_bottom_mapping.json` confirms the native
visible tensor shapes, but it does not expose these runtime metadata words or
the native internal `activation.data` / `output.data` table layout.

This is why a static custom descriptor dump is necessary but not sufficient.
The deep body consumes only `base+0x10`, `base+0x28`, `base+0x48`, and `r5`,
but native may call that body with advanced descriptor table bases.  The current
custom adapter has no native-derived values for `r23/r24/r27`; its
`HMX_W4A16_INTERNAL_SPLIT_N128` diagnostic only proved that writing both N
halves is possible, not that the native loop metadata was reproduced.

The native descriptor builder call around `0x3d9c54` reaches
`set_hmx_params_convw4b1x1(base+0x48, 0x70b, r28, 0, r4, r21, r6)`, with later
arguments derived from tensor metadata and wrapper flags.  Single-lane mask
sweeps are therefore weak evidence unless they reproduce the whole builder
state.

### Descriptor Builder Field Decode

Current static decode of `0x3d9920` for the HNH 1x1 W4 path uses these builder
arguments:

| Builder register | Source from `0x3ddc60` wrapper | Role |
|---|---|---|
| `r0` | `base = r29 + 0x30` | Destination stack record. |
| `r1` | wrapper arg0 | Output tensor object. |
| `r2` | wrapper arg1 | Activation tensor object. |
| `r3` | wrapper arg2 | Prepared weight tensor object. |
| `r4` | wrapper arg3 | Prepared bias/control tensor object. |
| `r5` | wrapper arg4 | Wrapper flags used by mask selection. |
| stack arg0 | wrapper arg5 | Control pointer later passed as final HMX `r5`. |

Be careful when checking this in the Hexagon text: stores without `.new` use the
old register value from before the packet.  With that packet rule, the HMX
descriptor fields consumed by `0x2fdb80` decode as:

| Native field | Builder store | Decoded source |
|---|---|---|
| weight pointer | `base+0x08` | `weight.data + encoded_offset(weight.meta[0x24..0x27], weight.meta[0x04/0x08/0x0c])`. |
| bias/control pointer | `base+0x0c` | `bias.data`. |
| activation `+0x00` | `base+0x10` | `activation.data`; this is QNN's internal HNH table/data pointer, not a public QHPI table copy. |
| activation `+0x04` | `base+0x14` | `activation.meta[0x20] >> 5`. |
| activation `+0x08` | `base+0x18` | `(activation.meta[0x20] >> 5) * (activation.meta[0x1c] >> 2)`. |
| output `+0x00` | `base+0x28` | `output.data`; native passes QNN's internal output table/data pointer. |
| output `+0x04` | `base+0x2c` | `output.meta[0x20] >> 5`. |
| output `+0x08` | `base+0x30` | `(output.meta[0x20] >> 5) * (output.meta[0x1c] >> 2)`. |
| output `+0x0c` | `base+0x34` | `output.meta[0x0c]`. |
| output `+0x10` | `base+0x38` | `output.meta[0x08]`. |
| output `+0x14` | `base+0x3c` | `output.meta[0x10]`. |

The builder also writes `base+0x1c..0x24`, but the direct deep HNH body does not
read those fields on this path.  They are useful only as wrapper/debug evidence,
not as the first custom alignment target.

The W4 mask branch at `0x3d9c54` calls
`set_hmx_params_convw4b1x1(base+0x48, 0x70b, r28, 0, r4, r21, r6)`.  Only the
second argument is a literal.  `r28`, `r4`, `r21`, and `r6` are all derived from
weight/activation metadata and wrapper flags, so custom constants such as one
fixed final mask argument are at best probes.  A production match must recreate
the metadata state that feeds the whole helper call.

### W4 Mask Helper Decode

The helper body at `0x289380` explains why some previous mask probes were
structurally weak.

For `arg1=0x70b`, the helper computes:

```text
extractu(arg1, width=6, offset=5) = 0x38
mask[0x04/4] starts as 0x38 << 5 = 0x700
mask[0x30/4] is the final flags argument
```

When `arg1` bit 3 is set, final flags bit `0x20` is set, and final flags bit
`0x40` is clear, the helper takes the early branch that writes:

```text
mask words:
[0]  = 0
[1]  = 0x700
[2]  = selector from coupled arg4/arg5 metadata
[3]  = 0x77c
[6]  = 0x3ff
[12] = 0x20
```

In this branch `arg2` does not affect those words; both `arg2=128` and
`arg2=256` produce the same default mask tuple
`[0, 0x700, 0, 0x77c, 0, 0, 0x3ff, 0, 0, 0, 0, 0, 0x20, 0, 0, 0]` when the
coupled metadata selectors are zero.  That matches the observed no-op result of
the `HMX_W4A16_MASK_ARG2=128` probe.

`arg4` and `arg5` are not independent lane toggles either.  They combine into
the selector stored at mask word `[2]` through a shift chosen from the
`arg1`-derived `0x38` class.  Therefore the useful target is not another
single-lane sweep; it is the native metadata state that produces the whole
helper argument tuple.

### Custom-Versus-Native Boundary

The native HMX boundary for the 256 case is `ConvLayer_s1.opt` with:

```text
activation  UFixed16 [1,8,32,256]
weight      SFixed8  [1,1,128,256]
bias        Int32    [1,8,1,128]
control     Int32    [1]
extra ctrl  Int32    [1,1,1,3]
output      UFixed16 [1,8,32,256]
```

The custom native-surface probe reaches most of this visible tensor shape, but
not the full native route:

| Boundary | Native Conv path | Current custom path |
|---|---|---|
| Graph-before weight | `UFixed8 [1,1,256,256]` after `CastInt4ToInt8`, then private ctxgen W4 lowering. | Custom initializer is already a packed sidecar; bottom mapping still records `UFixed8 [1,1,128,256]` despite XML/converter requesting `SFixed8`. |
| Final HMX weight | `SFixed8 [1,1,128,256]`, produced by `weights_to_vtcm`. | Raw bytes can match native K4 sidecar, but carrier metadata remains custom `UFixed8` at graph boundary. |
| Activation/output | QNN internal HNH Crouton table/data pointers. | Custom copies and expands public QHPI block tables. |
| Bias/control sidecar | `Int32 [1,8,1,128]` in the clean native oracle. | Current custom generated/imported probes still differ at the graph-control boundary. |
| Small control tensor | `Int32 [1]`. | Ctxgen maps current custom control as `[1,1,1,1]`; direct use is worse than local control. |

The fastest custom `native_op` probes are not this boundary: their custom op
input/output tensors are logical `[1,1,256,256]`, and QNN inserts a large
`ForceFormat_Crouton` around that shape.  The custom `tiled` probes are closer
to native at the tensor surface because activation/output become
`UFixed16 [1,8,32,256]`.  The older compact `[1,8,1,64]` bias-sidecar probes
remain historical diagnostics; the clean native oracle currently exposes
`[1,8,1,128]` at the final HNH boundary.

The confirmed remaining deltas are therefore:

1. Custom W4 reaches QHPI as `QUInt8 [1,1,128,256]`; native reaches HNH as
   `SFixed8 [1,1,128,256]`.
2. Custom manually expands public QHPI Crouton block tables; native passes the
   internal `activation.data` / `output.data` tables produced by QNN's HNH
   tensor metadata.
3. Custom mask setup uses fixed probe defaults plus the skel helper, while
   native derives several helper inputs from the same metadata that creates the
   descriptor fields.

The boundary-aware analyzer makes this distinction explicit.  On existing
artifacts:

- `output_codex_w4a16_native_op_k4pack_256` is fast, but its activation,
  output, bias, control, and weight carrier all mismatch native HNH boundary
  metadata.
- `output_codex_w4a16_k4pack_biascompact_256` reaches the native
  activation/output/bias surface; its report now shows only the W4 carrier
  (`QUInt8` versus native `SFixed8`) and the control tensor shape as graph
  boundary mismatches.
- `output_codex_w4a16_k4_nobias_native_surface_256` additionally reproduces the
  native W4 bytes and native no-bias/control bytes, but still fails
  (`1014/65536`).  This keeps the focus on builder-derived mask/table metadata
  and the QHPI carrier/control boundary.

The current closest native-surface custom run has a stronger failure signature
than the scalar exactness count alone:

```text
native-exact: 1014/65536
output-top-value: value=32767 count=32768 n32=[8192,8192,8192,8192,0,0,0,0]
native-top-value: value=65535 count=5895 n32=[721,674,788,638,774,821,655,824]
```

So half of the N32 output groups, columns 0..127, are stuck at the A16
zero-ish value `32767`; the nontrivial writes are only in N32 groups 4..7 and
still do not match native.  A W8-style compact row4 table order changes neither
that signature nor exactness.  This points back to the native output
descriptor/table/mask/body boundary, not to a final analytic formula mismatch.

The first output-table diagnostic confirms the table-consumption part of that
claim.  Building with `HMX_W4A16_OUT_TABLE_N_ROTATE=4` moves the stuck
`32767` half from N32 groups 0..3 to groups 4..7:

```text
artifact: output_codex_w4a16_outrot4_k4_nobias_native_surface_256
native-exact: 1219/65536
output-top-value: value=32767 count=32768 n32=[0,0,0,0,8192,8192,8192,8192]
```

So the current W4 HNH body call consumes the upper four output-table entries
per row4 group.  A full-descriptor two-call diagnostic,
`HMX_W4A16_INTERNAL_SPLIT_N128`, can make both N halves non-`32767`, but remains
incorrect and slow:

```text
artifact: output_codex_w4a16_fulldesc_splitn128_k4_nobias_native_surface_256
native-exact: 2346/65536
custom-optrace main: 204675 cycles
```

The W8-style split that changes each split call to `k_total_bytes=128` is worse
(`587/65536`) and the `k_total_bytes=256` split variant fails graph execution.
Therefore the next native-first target is not just "call twice"; it is the
native wrapper's full per-half metadata state: output table selection, weight
carrier/offset, control pointer, mask-helper inputs, and the wrapper
`r23/r24/r27` descriptor-advance loop together.

A later native-field probe tested the static builder interpretation that
descriptor `act_desc+0x08` and `out_desc+0x08` should be the table stride `8`
rather than the current custom `256`.  On the closest native-surface artifact,
that produces only `1419/65536` exact with `94156` custom main-op cycles, and
the same half-written `32767` N32 signature remains.  Combining `+0x08 = 8`
with `HMX_W4A16_INTERNAL_SPLIT_N128` fails graph execution before a valid
optrace.  This closes descriptor y-stride as an isolated native-field fix; the
native target remains the whole wrapper record, not a single scalar.

## Alignment Consequences

The custom path is not native-equivalent just because the HMX body is called or
the W4 bytes match.  To line up with native, the custom path must reproduce all
of these at the `ConvLayer_s1.opt` boundary:

1. Activation and output surfaces as QNN exposes them to HNH:
   `UFixed16 [1,8,32,256]`.
2. Weight sidecar as a signed native carrier:
   `SFixed8 [1,1,128,256]`, not custom `QUInt8`.
3. Bias/control sidecar:
   `Int32 [1,8,1,64]`.
4. Descriptor builder output:
   activation descriptor, output descriptor, mask words, and control pointer
   semantics together, not one field at a time.
5. Performance accounting:
   compare custom hot op to `q::ConvLayer_s1.opt` for kernel-only, and only use
   native group/end-to-end numbers when the same surrounding format work exists.

Dead-end implication from the current probes:

- Native-K4 byte packing is solved as a byte-order problem, but it is not the
  semantic fix.
- The clean-native sidecar import now proves the custom output value multiset
  can match native exactly, but the rows are rotated by one 32-row block.
  Existing `ACT_PHYSICAL_ONLY`, `OUT_PHYSICAL_ONLY`, and
  `ROW4_BLOCK_ORDER_MOD8` probes do not repair that rotation; the physical-table
  probes also destroy the value multiset.
- ONNX `INT8` initializer plus quant-overrides does not produce the native
  `SFixed8` QHPI carrier.
- `desc32`, y-stride-only, pointer-offset, control-word, and one-lane mask
  sweeps should not be repeated until the full native builder state is decoded.

## Runtime Descriptor Dump Attempts

2026-05-08 loader-safe isolated runs clarified the patched-skel path:

- Replacing `~/qnn_loader_probe_w4a16/libQnnHtpV75Skel.so` with an invalid file
  makes native context execution fail at Device Creation.  This proves the local
  isolated skel override path is active when `LD_LIBRARY_PATH=.:/vendor/lib64`
  and `ADSP_LIBRARY_PATH=.` are used.
- Replacing the same isolated skel with
  `/tmp/libQnnHtpV75Skel_hmx_entry_probe.so`, which patches
  `hmx_v73_convhnh1x1_stride1` at `0x2fcd80`, changes the native output SHA from
  canonical `147b7752a5f8c55f59c8539d65dcffe69214e01f27f157f7ccd540d9377822a8`
  to `372fecf39290f38b9d345e1e3e3cbf2fb986ee78283946a4b87934787593a0ca`.
  The first output word contains the probe magic `0x484d5850`, confirming that
  the patched `0x2fcd80` HNH entry is on the current native path.
- The probe writes into the internal ConvLayer output tile, and native
  post-Conv output ops transform that tile before `Y.raw` is emitted.  Therefore
  the public `Y.raw` dump is not yet a direct linear descriptor record.
  A stride-sampled read still exposes useful fields, including plausible W/B
  pointers and output descriptor scalars, but a reliable native descriptor
  parser must either invert this output transform or patch a wrapper-visible
  public-output location.
- Use `scripts/parse_w4a16_native_entry_probe.py` for the current
  entry probe.  The default `--layout crouton512` mode uses the mapping
  recovered by the corrected pattern probe:
  `public = (i % 32) * 128 + ((i // 32) // 2) * 16 + ((i // 32) & 1)`.
  This is the current way to parse the v3 native entry record from public
  `Y.raw`.
- A direct `r31` entry probe supersedes the earlier call-site guess for the
  clean 256^3 artifact: `hmx_v73_convhnh1x1_stride1` returns to `0x03de46c`,
  so the active call is `0x3de464` inside the simple prebuilt-record wrapper at
  `0x3de3c0`.  The older `0x3dde78` call in the `0x3ddc60` builder wrapper is
  not the call taken by this artifact.
- The v3 entry record reports native mask words
  `[0,0x700,0,0x77c,0,0,0x3ff,0,0,0,0,0,0xa0,0,control_ptr,0]`, output table
  entries `0x046a0000..0x046a7800`, activation table entries
  `0x046c9000..0x046d0800`, and control words `[1,0x401,0x20c,0]`.
  Matching only mask word `[12]` with `HMX_W4A16_MASK_ARG6=0xa0` leaves the
  imported-sidecar custom result unchanged, so the fix is not a single final
  mask flag.

The earlier "patched skel was not loaded" conclusion is superseded by the
invalid-skel loader test above.  For the v3 entry probe, the descriptor dump is
now parseable through the `crouton512` map; future probes still need to prove
their public-output placement before their samples are treated as linear data.

## Next Native-First Work

Before new custom changes, decode or instrument the native path in this order:

1. For the current clean 256^3 artifact, treat the simple wrapper at `0x3de3c0`
   as the active native path.  Its input is a prebuilt record, not the
   `0x3d9920` builder stack path.
2. Compare the v3 native entry record against the custom descriptor dump before
   changing custom code.  Any difference should be classified as carrier
   lowering, tensor table layout, descriptor scalar, mask helper input, control
   ABI, or wrapper loop state.
3. If a future artifact returns to `0x3ddc60`, dump that wrapper's inputs and
   post-builder stack separately instead of mixing the two paths.
4. Only then choose the smallest custom change that reproduces a named native
   boundary.

Secondary work remains useful, but should not replace the runtime dump:

- Recover the exact QNN converter/ctxgen route that turns float `W` plus 4-bit
  param encoding into `SFixed8 [1,1,128,256]`.
- Decode the relation between QNN's native Crouton block tables and the HNH
  pointer tables passed through `base+0x10` / `base+0x28`.
- Decode the dynamic mask-helper inputs for the native HNH 1x1 branch after the
  tensor metadata fields above are available.

## Alignment Checklist

Use this checklist before accepting a future W4A16 custom change:

1. It compares against `device_out/Y.raw` from the native artifact, not only an
   analytic formula.
2. It names the native performance scope being compared: `ConvLayer_s1.opt`,
   whole `conv1x1` group, or qnn-net-run execute.
3. It preserves standard performance products under `<out_dir>/optrace/`.
4. It emits `analysis/w4a16_native_compare.{json,txt}` and records boundary
   tensor contracts plus N32 value distribution.
5. It explains whether the changed boundary is converter/ctxgen carrier,
   sidecar bytes, QNN tensor table metadata, mask helper inputs, or deep body.
6. It avoids repeating closed byte-order, single-lane mask, control-word,
   desc32/y-stride, and row4-order probes unless new native evidence invalidates
   the current conclusions.
