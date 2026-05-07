# W4A16 QNN Native Path

This note describes the native QNN implementation path for the canonical
256^3 W4A16 artifact:

`example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/`

Use this as the first reference before changing the custom
`HmxU16I4ToU16MatMul` path.  The custom op should be aligned to this native
path, not to an analytic formula.

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

The important point is that native Conv starts from float weights plus a 4-bit
param encoding.  Ctxgen then owns the signed W4 carrier and sidecar lowering.

## Lowered HTP Path

After ctxgen, the graph is expanded into 210 HTP nodes.  Most nodes are
format-conversion work around the actual Conv core:

| Grouping | HTP node class | Count |
|---|---:|---:|
| input convert | `q::Quantize` | 32 |
| input convert | `q::*InputSlice` | 32 |
| `A_0231` | `q::Transpose_impl` | 8 |
| `A_0231` | `q::SlicePad_shape_inplace` | 8 |
| `conv1x1` | `q::ConvLayer.opt.weights_to_vtcm` | 1 |
| `conv1x1` | `q::ConvLayer.opt.bias_to_vtcm` | 1 |
| `conv1x1` | `q::ForceFormat_Crouton` | 2 |
| `conv1x1` | `q::ConvLayer_s1.opt` | 1 |
| `Y_0231` | `q::Transpose.2D` | 16 |
| output convert | `q::ForceFormat_Flat` | 32 |
| output convert | `q::Dequantize` | 32 |

The core Conv tensor contract at the final HMX node is:

| Node / tensor | Native type and shape | Notes |
|---|---|---|
| `weights_to_vtcm` input | `SFixed8` (`data_type=776`), `[1,1,128,256]` | Static prepared W4 carrier. |
| `weights_to_vtcm` output | `SFixed8` (`data_type=776`), `[1,1,128,256]` | Written to VTCM, 32768 bytes. |
| `bias_to_vtcm` input | `Int32` (`data_type=50`), `[1,8,1,64]` | Static no-bias/control sidecar, 2048 bytes. |
| `bias_to_vtcm` output | `Int32` (`data_type=50`), `[1,8,1,64]` | Written to VTCM. |
| `ConvLayer_s1.opt` input 0 | `UFixed16` (`data_type=1046`), `[1,8,32,256]` | HMX activation surface after `ForceFormat_Crouton`. |
| `ConvLayer_s1.opt` input 1 | `SFixed8` (`data_type=776`), `[1,1,128,256]` | Prepared W4 sidecar. |
| `ConvLayer_s1.opt` input 2 | `Int32` (`data_type=50`), `[1,8,1,64]` | Prepared bias/control sidecar. |
| `ConvLayer_s1.opt` input 3 | `Int32` (`data_type=50`), `[1]` | Small control tensor. |
| `ConvLayer_s1.opt` output | `UFixed16` (`data_type=1046`), `[1,8,32,256]` | HMX output surface. |

`ConvLayer_s1.opt` reports `mem_vtcm_read=165888` and
`mem_vtcm_write=131072`, matching:

```text
activation  [1,8,32,256] u16  = 131072 bytes
weight      [1,1,128,256] i4   =  32768 bytes
bias/control                 =   2048 bytes
output      [1,8,32,256] u16  = 131072 bytes
```

## Prepared Sidecars

The canonical native context binary is:

```text
ctx/conv_ctx.bin
size:   94208 bytes
sha256: 7e66eb07a341473e714e4a540daa6ba14ddecf6ea2b81efb8a253c61c5dea382
```

Useful sidecar regions:

| Region | Size | SHA256 | First bytes |
|---|---:|---|---|
| `conv_ctx.bin+0xc400` bias/control | 2048 | `e595cebf33d435d88cc1e2d0d7382a122ed389f76f97b41ec9e62d736662bdf3` | `00 80 00 80 ...` |
| `conv_ctx.bin+0xcc00` W4 weight | 32768 | `b0dbe7545ae03e7c5f9a2a4da06ba27fee7164b58948709ea69795845c261297` | `d9 ea fb 0c b6 c7 d9 ea ...` |

The W4 sidecar order decoded so far is:

```text
N32 tile -> K8 group -> n-in-tile -> k4 pair
byte pairs: (k+0,k+4), (k+1,k+5), (k+2,k+6), (k+3,k+7)
twos-complement nibbles
```

`W4_PACK_ORDER=native_nmajor_k4_lohi` reproduces this 32768-byte native stream
byte-for-byte.  That proves byte packing, but not the full native runtime
contract.

## Runtime And Performance Path

The standard native optrace directory is:

`example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/optrace/`

For the captured run:

| Scope | Cycles / time |
|---|---:|
| HTP timeline span | `178332` cycles |
| Sum of pid0 events | `477257` cycles |
| qnn-net-run execute stat 1 | `4416 us` QNN execute, `476779` accelerator cycles |
| qnn-net-run execute stat 2 | `2746 us` QNN execute, `356723` accelerator cycles |
| qnn-net-run execute stat 3 | `2537 us` QNN execute, `308367` accelerator cycles |

The `conv1x1` group in `optrace/summary.json` breaks down as:

| Event | Cycles | Packets | Notes |
|---|---:|---:|---|
| `q::ConvLayer.opt.weights_to_vtcm` | `3139` | `56` | Static W4 sidecar movement. |
| `q::ConvLayer.opt.bias_to_vtcm` | `886` | `44` | Static bias/control movement. |
| `DmaCheckpointSet` | `28` | `4` | Sidecar synchronization. |
| `q::ConvLayer.opt.bias_to_vtcm` | `163` | `22` | Small follow-up sidecar event. |
| `q::ForceFormat_Crouton` | `5472` | `1618` | Conv input/output format work. |
| `q::ConvLayer_s1.opt` | `7702` | `977` | Closest native HMX kernel-only comparator. |
| `q::Reshape` | `18749` | `9562` | Conv output reshape / movement. |
| `q::ForceFormat_Crouton` | `2327` | `1618` | Additional format work. |

So there are three different performance scopes:

1. Kernel-only native comparator: `q::ConvLayer_s1.opt = 7702` cycles.
2. Native Conv group comparator: all `conv1x1` HTP events, about `38466` cycles
   in the qnn-profile grouped stat.
3. End-to-end execute comparator: qnn-net-run execute stats, including input
   conversion, transposes, output dequant/export, RPC, and accelerator wait.

Custom W4A16 performance claims must name which scope they compare against.

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

### Custom-Versus-Native Boundary

The native HMX boundary for the 256 case is `ConvLayer_s1.opt` with:

```text
activation  UFixed16 [1,8,32,256]
weight      SFixed8  [1,1,128,256]
bias        Int32    [1,8,1,64]
control     Int32    [1]
output      UFixed16 [1,8,32,256]
```

The fastest custom `native_op` probes are not this boundary: their custom op
input/output tensors are logical `[1,1,256,256]`, and QNN inserts a large
`ForceFormat_Crouton` around that shape.  The custom `tiled` probes are closer
to native at the tensor surface because activation/output become
`UFixed16 [1,8,32,256]`.  The `native_a16_w4compact` bias sidecar can also match
native's compact `[1,8,1,64]` shape, but its folded-bias content is not the
native no-bias/control sidecar.  For the no-bias native oracle, use
`native_a16_nobias` to reproduce the repeated `0x80008000` 2048B sidecar.

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
- ONNX `INT8` initializer plus quant-overrides does not produce the native
  `SFixed8` QHPI carrier.
- `desc32`, y-stride-only, pointer-offset, control-word, and one-lane mask
  sweeps should not be repeated until the full native builder state is decoded.

## Next Native-First Work

Before new custom changes, decode or instrument the native path at one of these
boundaries:

1. The exact QNN converter/ctxgen route that turns float `W` plus 4-bit param encoding
   into `SFixed8 [1,1,128,256]`.
2. The exact relation between QNN's native Crouton block tables and the HNH
   pointer tables passed through `base+0x10` / `base+0x28`.
3. The dynamic mask helper inputs for the native HNH 1x1 branch, after the
   tensor metadata fields above are reproduced or dumped.

Only after one of those native boundaries is understood should the custom path
be changed.
