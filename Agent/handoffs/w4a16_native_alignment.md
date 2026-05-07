# w4a16 Native Alignment Handoff

Current status: `example/qnn_hmx_matmul_w4a16`
(`HmxU16I4ToU16MatMul`, i4 weight x u16 activation -> u16 output) is not yet
aligned with QNN native for the canonical 256^3 native-contract path.

Acceptance rule for this family:

1. Correctness must compare against a real QNN native output artifact for the
   same graph/input. Analytic formulas are diagnostic references only.
2. Performance must use the standard optrace artifact set under
   `<out_dir>/optrace/`, not ad-hoc `/tmp/_optrace*` files.
3. Report both custom main-op cycles and native comparable kernel cycles.
   For native W4A16 Conv, the closest kernel-only event is
   `q::ConvLayer_s1.opt`; full native graph timeline also includes
   transpose/quantize/dequantize work.

## Standard Flow

Build the real-kernel packages explicitly:

```bash
EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w4a16/build.sh

EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w4a16/build_x86.sh
```

Run the current best diagnostic custom flow:

```bash
OUT_DIR="$PWD/example/qnn_matmul_profile/output_codex_w4a16_native_nmajor_kpair_hilo_row4_nativebias_256" \
M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq \
NATIVE_OUTPUT=1 STRICT_OPTRACE=1 \
W4_PACK_ORDER=native_nmajor_kpair_hilo \
VERIFY_NATIVE_RAW="$PWD/example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/device_out/Y.raw" \
VERIFY_NATIVE_TRANSPOSE=1 \
GEN_EXTRA_ARGS="--bias-layout native_a16 --a16-quant-contract native --reference-contract native --final-output-rank 3d" \
bash example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh
```

The runner creates the durable performance products under `<OUT_DIR>/optrace/`:

- `summary.json`
- `profile.txt`
- `manifest.json`
- `chrometrace.json`
- `chrometrace_htp.json`
- `chrometrace_runtrace.json`
- `chrometrace_qnn_htp_analysis_summary.json`
- `chrometrace_qnn_htp_analysis_summary.html`

Use `summary.json` for scripted comparisons and `chrometrace*.json/html` for
timeline inspection.

## Current Evidence

Durable native oracle:

| Artifact | Path |
|---|---|
| native output | `example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/device_out/Y.raw` |
| native optrace | `example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/optrace/` |
| refreshed custom probe | `example/qnn_matmul_profile/output_codex_w4a16_native_nmajor_kpair_hilo_row4_nativebias_after_desc_overrides_256/` |

Latest refreshed custom result:

| Check | Result |
|---|---:|
| output vs native | `4229/65536`, maxdiff `65535` |
| custom main-op cycles | `95356` |
| custom timeline span | `141978` |
| custom sum pid0 event cycles | `142717` |

Native performance reference from
`output_codex_native_w4a16_same_custom_256/optrace/summary.json`:

| Native event/view | Cycles |
|---|---:|
| `q::ConvLayer_s1.opt` | `7702` |
| `q::ConvLayer.opt.weights_to_vtcm` | `3139` |
| `q::ConvLayer.opt.bias_to_vtcm` | `1049` |
| native `conv1x1` QNN-op aggregate | `38466` |
| native graph timeline span | `178332` |

The custom main op is therefore still far slower than the native W4 kernel
event and does not satisfy the correctness gate.

## Findings

Native W4A16 Conv evidence:

- Native Conv ONNX stores float weights as `[N,K,1,1]` and the DLC keeps full
  W4 codes before ctxgen.
- Ctxgen lowers native Conv through `weights_to_vtcm` to prepared W4 sidecars:
  weight tensor `SFixed8` (`data_type=776`) with dims `[1,1,128,256]`.
- Native no-bias/control sidecar is `Int32` (`data_type=50`) with dims
  `[1,8,1,64]`; the observed control block bytes at `conv_ctx.bin+0xc400`
  repeat `00 80 00 80`.

Custom W4A16 evidence:

- The custom int8 carrier path reaches ctxgen as `UFixed8`
  (`data_type=1032`) weight dims `[1,1,128,256]`; it does not trigger native
  Conv's W4 packer.
- A `native_full_codes` custom probe kept a full `[1,1,256,256]` byte tensor
  and still did not trigger the native W4 packer.
- The best observed pack/layout is `native_nmajor_kpair_hilo` with
  `--bias-layout native_a16`; it remains only `4229/65536` exact.
- The custom output is heavily saturated: about `27977` zeros and `27614`
  `65535` values in the refreshed best probe, versus native's `3309` zeros
  and `5895` `65535` values.

Dead ends already checked:

- Weight pointer offsets `64,128,256,512,1024` do not improve alignment;
  the best was `4235/65536` at offset `256`.
- Bias/control pointer offsets `64,128,256,512,1024` do not improve alignment.
- Native no-bias control layout `[1,8,1,64]` with repeated `0x80008000`
  reduces saturation but worsens exactness (`1755/65536` in the refreshed
  comparable probe).
- `HMX_W4A16_USE_SKEL_KERNEL` calls the external skel wrapper but preserves the
  same low exactness class.
- `DESC_M_TILES_OVERRIDE=32` lowers custom cycles to about `12.8k`, but only
  computes a small fraction of the output (`~531/65536` exact). It is not a
  performance fix.
- Independent `act/out y_stride=8` overrides do not change correctness or
  cycles. Combining `desc_m=32` with `y_stride=8` remains a partial-coverage
  result.
- `MASK_ARG1` sweep over the `0x700` family did not beat `0x70b` class
  correctness. Several bits raise cycles to `~214k`.
- Converting the prepacked weight initializer to float with an 8-bit symmetric
  override did not produce native `SFixed8`; ctxgen converted it to `UFixed16`.

## Code State

`HmxU16I4ToU16MatMulOp.cpp` now has guarded descriptor override hooks matching
the w8a16 diagnostic style:

- `HMX_W4A16_ACT_N_PAIRS_OVERRIDE`
- `HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE`
- `HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE`
- `HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE`
- `HMX_W4A16_DESC_M_TOTAL_MINUS_STEP_OVERRIDE`
- `HMX_W4A16_DESC_K_TOTAL_BYTES_OVERRIDE`
- existing `HMX_W4A16_DESC_M_TILES_OVERRIDE`,
  `HMX_W4A16_WEIGHT_PTR_OFFSET`, and `HMX_W4A16_BIAS_PTR_OFFSET`

Defaults preserve the prior runtime behavior; these hooks are for focused ABI
probes only.

## Next Work

1. Decode the exact native W4A16 hnh wrapper/descriptor builder used by
   `q::ConvLayer_s1.opt`, not only the older bbb-oriented helper. The relevant
   disassembly anchors are in `Agent/qnn_re/skel_text_full.S` around
   `hmx_v73_convhnh1x1_stride1`, `hmx_v73_convhnh1x1deep_stride1`, and native
   calls to `set_hmx_params_convw4b1x1`.
2. Add a richer `HMX_W4A16_DESC_DUMP` payload if needed: include QHPI
   activation/output block table lengths, first logical pointer deltas, and the
   final expanded mask words. Compare that against decoded native wrapper
   expectations.
3. Investigate whether the custom converter path can expose a prepared
   `SFixed8 [1,1,128,256]` W4 weight tensor to QHPI, or whether the prepared
   native sidecar must be imported through another static-tensor route.
4. Do not repeat the pointer-offset, y-stride-only, `0x700` mask-family, or
   float-weight dtype probes unless new evidence changes the premise.
