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

1. `0x3d9920` descriptor builder fields for the 256^3 HNH path, especially
   `base+0x10..0x24`, `base+0x28..0x3c`, and all `base+0x48` mask words.
2. The QNN converter/ctxgen route that turns float `W` plus 4-bit param encoding
   into `SFixed8 [1,1,128,256]`.
3. The exact relation between QNN's native Crouton block tables and the HNH
   pointer tables passed through `base+0x10` / `base+0x28`.

Only after one of those native boundaries is understood should the custom path
be changed.
