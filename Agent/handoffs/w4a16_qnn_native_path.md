# W4A16 QNN Native Path

This note describes the native QNN implementation path for the canonical
256^3 W4A16 artifact:

`example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/`

Use this as the first reference before changing the custom
`HmxU16I4ToU16MatMul` path.  The custom op should be aligned to this native
path, not to an analytic formula.

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

## Evidence Map

| Evidence | Path | What it proves |
|---|---|---|
| source ONNX | `conv.onnx` | Native starts as a float `Conv`, not a custom or integer MatMul op. |
| quant override | `quant_overrides.json` | A/Y use A16 quantization, W uses symmetric 4-bit param encoding. |
| converter graph-before | `ctx/conv_bottom_mapping_graph_before.json` | High-level lowering is `Convert -> Transpose -> Conv2d -> CastInt4ToInt8 -> Transpose -> Convert`; W is still `UFixed8` metadata here. |
| final HTP graph | `ctx/conv_bottom_mapping.json` | Ctxgen expands the graph to 210 HTP nodes and introduces `weights_to_vtcm`, `bias_to_vtcm`, `ForceFormat_Crouton`, and `ConvLayer_s1.opt`. |
| context binary | `ctx/conv_ctx.bin` | Holds the prepared 2048B no-bias/control sidecar and 32768B native W4 sidecar. |
| native output oracle | `device_out/Y.raw` | The custom output oracle; analytic formulas are secondary diagnostics. |
| optrace products | `optrace/` | Standard performance source: timeline, profile text, QHAS summary, decoded `summary.json`. |
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
       bias_to_vtcm produces Int32 [1,8,1,64]
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

The graph-before view has only 6 nodes and 15 tensors.  The lowered HTP view has
210 nodes and 260 tensors.  That expansion is the native implementation, not
incidental tracing noise: it is where QNN creates the sidecar movement,
Crouton/HNH surfaces, and final HMX call boundary.

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
bias        Int32    [1,8,1,64]
control     Int32    [1]
output      UFixed16 [1,8,32,256]
```

The custom native-surface probe reaches most of this visible tensor shape, but
not the full native route:

| Boundary | Native Conv path | Current custom path |
|---|---|---|
| Graph-before weight | `UFixed8 [1,1,256,256]` after `CastInt4ToInt8`, then private ctxgen W4 lowering. | Custom initializer is already a packed sidecar; bottom mapping still records `UFixed8 [1,1,128,256]` despite XML/converter requesting `SFixed8`. |
| Final HMX weight | `SFixed8 [1,1,128,256]`, produced by `weights_to_vtcm`. | Raw bytes can match native K4 sidecar, but carrier metadata remains custom `UFixed8` at graph boundary. |
| Activation/output | QNN internal HNH Crouton table/data pointers. | Custom copies and expands public QHPI block tables. |
| Bias/control sidecar | `Int32 [1,8,1,64]`, native no-bias bytes for this oracle. | `native_a16_nobias` reproduces raw bytes. |
| Small control tensor | `Int32 [1]`. | Ctxgen maps current custom control as `[1,1,1,1]`; direct use is worse than local control. |

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
