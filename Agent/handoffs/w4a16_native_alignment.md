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

Native-path first rule: before adding new custom probes, read
[`w4a16_qnn_native_path.md`](w4a16_qnn_native_path.md).  The current blocker is
the native `ConvLayer_s1.opt` contract as a whole: signed W4 sidecar carrier,
native activation/output Crouton surfaces, descriptor builder state, mask words,
and control pointer semantics.

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
OUT_DIR="$PWD/example/qnn_matmul_profile/output_codex_w4a16_control_i32_256" \
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

The runner also creates a normalized W4A16 comparison report under
`<OUT_DIR>/analysis/`:

- `w4a16_native_compare.json`
- `w4a16_native_compare.txt`

The analysis report combines quantized output comparison, row4/N32 spatial
coverage, saturation distribution, custom optrace cycles, native optrace
cycles, and the custom-versus-native graph-boundary tensor contract.  The text
report includes `custom-boundary`, `native-boundary`, and `boundary-mismatch`
lines so a failed probe records whether it even reached the native HNH tensor
surface.  This is the standard quick-read artifact for failed probes; the raw
optrace files remain the source of truth for timeline inspection.

Descriptor dumps should also use the checked-in parser instead of one-off
Python snippets:

```bash
scripts/parse_w4a16_desc_dump.py \
  example/qnn_matmul_profile/output_codex_w4a16_descdump_enriched_256/device_out/out.raw \
  --cols 256
```

The parser decodes descriptor fields, mask words, table pointer samples, and
the first two little-endian u32 words from the effective weight and bias/control
buffers passed to the HNH kernel.

`HMX_W4A16_DESC_DUMP_TABLE_SELECT` selects which 16-entry pointer-table sample
is written into the 256-byte dump payload:

| Value | Sample |
|---:|---|
| `0` | expanded activation table used by the HNH descriptor |
| `1` | expanded output table used by the HNH descriptor |
| `2` | source QHPI activation block table before expansion |
| `3` | source QHPI output block table before expansion |

Prepared native W4 sidecars can be imported into the standard generator flow for
diagnostics. Extract the native 256^3 sidecar first, then pass it through
`GEN_EXTRA_ARGS`:

```bash
dd if=example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/ctx/conv_ctx.bin \
  of=/tmp/native_w4a16_sidecar_256.raw bs=1 skip=$((0xcc00)) count=32768 status=none

GEN_EXTRA_ARGS="--bias-layout native_a16 --a16-quant-contract native \
  --reference-contract native --final-output-rank 3d \
  --w4-native-sidecar-raw /tmp/native_w4a16_sidecar_256.raw" \
bash example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh
```

The generator can also synthesize that sidecar without importing raw bytes:

```bash
W4_PACK_ORDER=native_nmajor_k4_lohi \
W4_NIBBLE_ENCODING=twos \
GEN_EXTRA_ARGS="--bias-layout native_a16 --a16-quant-contract native \
  --reference-contract native --final-output-rank 3d" \
bash example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh
```

## Current Evidence

Durable native oracle:

| Artifact | Path |
|---|---|
| native output | `example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/device_out/Y.raw` |
| native optrace | `example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/optrace/` |
| refreshed custom probe | `example/qnn_matmul_profile/output_codex_w4a16_control_i32_256/` |

Latest refreshed custom result:

| Check | Result |
|---|---:|
| output vs native | `4229/65536`, maxdiff `65535` |
| custom main-op cycles | `94579` |
| custom timeline span | `134324` |
| custom sum pid0 event cycles | `134416` |

Post descriptor-dump-enrichment recheck:

| Probe artifact | Native exact | Main-op cycles | Timeline span |
|---|---:|---:|---:|
| `output_codex_w4a16_after_descdump_enrich_256/` | `4229/65536` | `94496` | `135816` |

The precompute-record metadata added for descriptor dumping does not change the
current default correctness class or main-op cycle class.

Builder-derived mask probe:

| Probe artifact | Variant | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|
| `output_codex_w4a16_mask_arg5_20_builder_256/` | `HMX_W4A16_MASK_ARG5=0x20` from the native `0x3d9c54` call-shape hypothesis | `4229/65536` after native-output transpose | `94819` | `140184` |

The high-bit `arg5=0x20` probe quantizes to byte-identical output versus the
default probe (`65536/65536` custom-output match).  The non-transposed verifier
prints `3863/65536`; using the same native-output transpose as the baseline
keeps the expected `4229/65536`.  This builder-derived lane is therefore a no-op
for correctness, not the missing contract.

Native-builder `arg2` probe:

| Probe artifact | Variant | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|
| `output_codex_w4a16_mask_arg2_128_256/` | `HMX_W4A16_MASK_ARG2=128`, matching the likely W4 `K/2` metadata lane | `4229/65536` | `93932` | `138472` |

The second argument to `set_hmx_params_convw4b1x1` is not the missing contract
for the current standard flow; forcing it from `256` to `128` leaves native
exactness unchanged.

Latest control-word probe:

| Probe artifact | `HMX_W4A16_EXTRA_PARAM0` | Native exact | Main-op cycles | Timeline span |
|---|---:|---:|---:|---:|
| `output_codex_w4a16_control0_0_256/` | `0` | `3176/65536` | `97808` | `140988` |
| `output_codex_w4a16_control0_2_256/` | `2` | `2789/65536` | `105855` | `148598` |
| `output_codex_w4a16_control0_4_256/` | `4` | `3084/65536` | `131926` | `185463` |

The default first control word `1` is still the best known setting.  Since the
deep HNH body reads only the first control word on this path, changing the
three-word local table does not look like the missing native contract.

Compact bias/control plus control-word probe:

| Probe artifact | Bias/control layout | `HMX_W4A16_EXTRA_PARAM0` | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|---:|
| `output_codex_w4a16_control_i32_biascompact_256/` | `native_a16_w4compact` | `1` | `3846/65536` | `93300` | `134261` |
| `output_codex_w4a16_biascompact_control0_0_256/` | `native_a16_w4compact` | `0` | `3288/65536` | `96883` | `135938` |
| `output_codex_w4a16_biascompact_control0_2_256/` | `native_a16_w4compact` | `2` | `3072/65536` | `106045` | `152037` |

The deep body uses the first control word as a `bias/control` pointer step
(`r3 += extra_param0 << 10`), but changing that step does not rescue the
native-sized 2048B W4 compact sidecar.  The default step `1` remains best for
both default and compact bias/control layouts.

Latest activation-layout probes:

| Probe artifact | Native exact | Main-op cycles | Timeline span |
|---|---:|---:|---:|
| `output_codex_w4a16_native_op_layout_probe_256/` | `4092/65536` | `29956` | `78824` |
| `output_codex_w4a16_native_op_layout_biascompact_256/` | `3775/65536` | `30708` | `79507` |
| `output_codex_w4a16_native_op_layout_native_sidecar_256/` | `3757/65536` | `29515` | `66969` |
| `output_codex_w4a16_native_op_layout_native_sidecars_nobias_256/` | `1379/65536` | `30144` | `78471` |
| `output_codex_w4a16_native_conv_input_u16_probe_256/` | `3784/65536` | `94236` | `139087` |
| `output_codex_w4a16_native_conv_surface_real_256/` | graph execution fails | no valid optrace | n/a |
| host native Conv tensor-dump layout sweep | every tested `Y`/`D` output-layout flag still dumps `UFixed16 [1,256,1,256]`; `ConvLayer_s1.opt` remains `UFixed16 [1,8,32,256]` | n/a | n/a |

Native/custom raw contract checks:

- The native Conv input artifact is float `[1,K,1,M]`.  Quantizing it with the
  native A16 encoding and transposing to `[1,1,M,K]` matches the custom
  `chain_qdq` native input bytes exactly (`65536/65536`) for
  `output_codex_w4a16_control_i32_256/runtime_inputs_u8/act_w4a16.raw`.
- Native Conv ONNX weight `[N,K,1,1]`, divided by the native W4 scale
  `1/7` and transposed to `[K,N]`, matches
  `output_codex_w4a16_control_i32_256/w4a16.onnx.wRaw_KN.npy` exactly
  (`65536/65536`).

This rules out raw activation values and raw logical W4 codes as the current
source of mismatch.  The remaining gap is after QNN lowering into prepared HMX
sidecars/descriptors or inside the HNH interpretation of those prepared bytes.

Prepared W4 sidecar decoding update:

- The full native prepared W4 sidecar for the canonical 256^3 oracle starts at
  `conv_ctx.bin+0xcc00`, not `+0xd000`.  Its 32768-byte SHA-256 is
  `b0dbe7545ae03e7c5f9a2a4da06ba27fee7164b58948709ea69795845c261297`.
- The physical byte order is `N32 tile -> K8 group -> n-in-tile -> k4`, where
  each byte pairs `(k+0,k+4)`, `(k+1,k+5)`, `(k+2,k+6)`, and `(k+3,k+7)` with
  two's-complement W4 nibbles.  `W4_PACK_ORDER=native_nmajor_k4_lohi` now
  reproduces the native `+0xcc00` sidecar byte-for-byte.
- Standard-flow contexts using `native_nmajor_k4_lohi` embed that exact 32KB
  stream at custom `w4a16_ctx.bin+0xa000`, so the current low correctness class
  is not caused by W4 sidecar raw bytes.

Latest native-K4 sidecar probes:

| Probe artifact | Variant | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|
| `output_codex_w4a16_native_op_k4pack_256/` | `OP_INPUT_LAYOUT=native`, generated native-K4 sidecar | `3507/65536` | `30243` | `71611` |
| `output_codex_w4a16_k4pack_tiled_256/` | tiled public-QHPI shape, generated native-K4 sidecar | `2872/65536` | `96246` | `146055` |
| `output_codex_w4a16_native_op_k4pack_biascompact_256/` | native layout, generated native-K4 sidecar, `native_a16_w4compact` bias/control | `3183/65536` | `30436` | `75009` |
| `output_codex_w4a16_k4pack_biascompact_256/` | tiled public-QHPI shape, generated native-K4 sidecar, `native_a16_w4compact` bias/control | `3142/65536` | `95838` | `146213` |
| `output_codex_w4a16_native_op_desc32_physical_k4pack_256/` | native layout, `DESC_M_TILES_OVERRIDE=32`, physical-only tables, generated native-K4 sidecar | `460/65536` | `6626` | `49714` |
| `output_codex_w4a16_native_op_k4pack_skel_256/` | native layout, generated native-K4 sidecar, skel `hmx_v73_convhnh1x1_stride1` entry | `3507/65536` | `59686` | `107891` |

The native-K4 sidecar is therefore a byte-order finding and a reusable
diagnostic, not a semantic fix.  The next useful path remains the native HNH
descriptor-builder field derivation and the custom QHPI tensor contract.

Native-surface sidecar checkpoint after the `0x3d9920` field decode:

| Probe artifact | Boundary/sidecar facts | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|
| `output_codex_w4a16_descdump_k4compact_native_surface_nativeout_256/` | `tiled`, generated native-K4 sidecar; descriptor scalars match the current custom HNH form; W4 first words are native; `native_a16_w4compact` bias first words are `0x80405524`, `0x40000092` | descriptor dump | n/a | n/a |
| `output_codex_w4a16_descdump_k4_nobias_native_surface_nativeout_256/` | same descriptor/weight; `native_a16_nobias` bias first words are `0x80008000`, `0x80008000`; full 2048B bias block at custom `w4a16_ctx.bin+0x9000` matches native `conv_ctx.bin+0xc400` SHA `e595cebf33d435d88cc1e2d0d7382a122ed389f76f97b41ec9e62d736662bdf3` | descriptor dump | n/a | n/a |
| `output_codex_w4a16_k4_nobias_native_surface_256/` | real HMX run with native HNH activation/output/bias shapes, native K4 W4 bytes at `w4a16_ctx.bin+0x9800`, and native no-bias/control bytes at `+0x9000` | `1014/65536` | `94610` | `139704` |

This rules out two more static-sidecar explanations: the raw native K4 W4
sidecar and the raw native no-bias/control sidecar can both be reproduced in the
custom context.  The remaining execution mismatch is deeper than sidecar bytes:
either the native builder's dynamic mask/table metadata still differs, or the
custom graph's public QHPI carrier/control boundary is not equivalent to the
native HNH metadata state even when the byte payloads match.

Latest all-native-sidecar probes:

| Probe artifact | Variant | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|
| `output_codex_w4a16_mask_arg6_4_native_sidecars_nobias_256/` | default public Crouton shape, `HMX_W4A16_MASK_ARG6=0x4` | `1379/65536` | `93542` | `134323` |
| `output_codex_w4a16_mask_arg6_c_native_sidecars_nobias_256/` | default public Crouton shape, `HMX_W4A16_MASK_ARG6=0xc` | `1379/65536` | `95735` | `141032` |
| `output_codex_w4a16_native_conv_input_native_sidecars_nobias_256/` | Conv-style input transpose plus native W4/no-bias sidecars | `1379/65536` | `93472` | `139667` |

These runs keep the prepared native W4 stream and native no-bias/control bytes
byte-identical to the native context.  In the custom context they land at
`w4a16_ctx.bin+0x9800` and `w4a16_ctx.bin+0x9000`, respectively.  The output
distribution still matches the saturated all-native-sidecar failure class
(`11264` zeros and `8704` `65535` values), so the blocker is not the static
sidecar bytes, the Conv-style input transpose, or the simple low-bit
`MASK_ARG6` candidates implied by the native builder.

Native performance reference from
`output_codex_native_w4a16_same_custom_256/optrace/summary.json`:

| Native event/view | Cycles |
|---|---:|
| `q::ConvLayer_s1.opt` | `7702` |
| `q::ConvLayer.opt.weights_to_vtcm` | `3139` |
| `q::ConvLayer.opt.bias_to_vtcm` | `1049` |
| native `conv1x1` QNN-op aggregate | `38466` |
| native graph timeline span | `178332` |

The default custom main op is still far slower than the native W4 kernel event.
The native-shaped activation/output probe narrows main-op cycles to the old
physical-table class (`~29k-31k`), but it is still not correct and remains
slower than native `q::ConvLayer_s1.opt`.

Latest descriptor-table dump artifacts:

| Probe artifact | `HMX_W4A16_DESC_DUMP_TABLE_SELECT` | Sample class |
|---|---:|---|
| `output_codex_w4a16_descdump_table_act_local_256/` | `0` | expanded activation table |
| `output_codex_w4a16_descdump_table_out_local_256/` | `1` | expanded output table |
| `output_codex_w4a16_descdump_table_act_source_256/` | `2` | source activation block table |
| `output_codex_w4a16_descdump_table_out_source_256/` | `3` | source output block table |

All four decode with `act_block_entries=64`, `out_block_entries=64`,
`act_entries=512`, and `out_entries=512`.  The source QHPI tables are contiguous
physical block pointers, e.g. activation starts at `0x04020000, 0x04020800,
... 0x04027800`.  The expanded HNH descriptor tables insert the row4 offset
after each 8-tile physical row, e.g. activation entries `[8..15]` become
`0x04020100, 0x04020900, ... 0x04023900`; output follows the same pattern from
`0x04000000`.  Use this as the custom-side table-shape baseline when comparing
against native builder expectations.

Latest native-shaped loop probes:

| Probe artifact | Variant | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|
| `output_codex_w4a16_native_layout_desc32_256/` | `OP_INPUT_LAYOUT=native`, `DESC_M_TILES_OVERRIDE=32` | `505/65536` | `6877` | `47663` |
| `output_codex_w4a16_tiled_desc32_256/` | tiled public-QHPI shape, `DESC_M_TILES_OVERRIDE=32` | `531/65536` | `13686` | `62459` |
| `output_codex_w4a16_desc32_physical_tables_256/` | tiled, `desc32`, physical-only source block tables | `4092/65536` | `6566` | `50845` |
| `output_codex_w4a16_tiled_desc32_biascompact_256/` | tiled, `desc32`, `native_a16_w4compact` | `475/65536` | `14241` | `57682` |
| `output_codex_w4a16_tiled_desc32_native_w4sidecar_biascompact_256/` | tiled, `desc32`, native W4 sidecar, compact bias/control | `342/65536` | `14302` | `58010` |
| `output_codex_w4a16_desc32_ystride256_table8_256/` | tiled, `desc32`, table storage stride `8`, descriptor y-stride `256` | `531/65536` | `13575` | `60255` |
| `output_codex_w4a16_desc32_physical_ystride256_table8_256/` | tiled, `desc32`, physical-only tables, table storage stride `8`, descriptor y-stride `256` | `4092/65536` | `6504` | `54866` |
| `output_codex_w4a16_native_field_ystride8_k4_nobias_native_surface_256/` | closest native-surface flow, native-K4 sidecar, native no-bias/control, descriptor `act/out +0x08 = 8` | `1419/65536` | `94156` | `138885` |
| `output_codex_w4a16_native_field_ystride8_splitn128_k4_nobias_native_surface_256/` | same `+0x08 = 8` native-field probe plus `HMX_W4A16_INTERNAL_SPLIT_N128` | graph execution failure | n/a | n/a |

`DESC_M_TILES_OVERRIDE=32` identifies the native fast loop class but is not a
semantic fix.  Combining `desc32` with physical-only tables proves the hot loop
can run in native-class cycles (`6566`, versus native `7702`), but the output
distribution falls back to the saturated failure class.  The
`OP_INPUT_LAYOUT=native` fast probe is not a valid native HMX surface comparison
because the custom HMX output tensor is `[1,1,256,256]`, while the real native
Conv HMX input/output tensors are `[1,8,32,256]`.
Keeping table storage stride at `8` while forcing descriptor y-stride to `256`
does not change either desc32 semantic class, so y-stride is not the missing
field for this false path.

The closest native-surface y-stride probe is also not a fix.  Forcing
descriptor `act_desc+0x08` and `out_desc+0x08` to `8`, matching the table-stride
interpretation of the static builder formula and the aligned W8A16 path, moves
exactness only from `1014/65536` to `1419/65536`.  The output signature remains
the same half-written class: `32767` appears 32768 times, all in N32 groups
0..3.  Adding the existing `HMX_W4A16_INTERNAL_SPLIT_N128` diagnostic to that
field hypothesis fails graph execution, so this does not reproduce the native
wrapper descriptor-advance loop.

## Findings

Native W4A16 Conv evidence:

- Native Conv ONNX stores float weights as `[N,K,1,1]` and the DLC keeps full
  W4 codes before ctxgen.
- Ctxgen lowers native Conv through `weights_to_vtcm` to prepared W4 sidecars:
  weight tensor `SFixed8` (`data_type=776`) with dims `[1,1,128,256]`.
- Native no-bias/control sidecar is `Int32` (`data_type=50`) with dims
  `[1,8,1,64]`; the observed control block bytes at `conv_ctx.bin+0xc400`
  repeat `00 80 00 80`.
- The generator's `--bias-layout native_a16_nobias` emits the same 2048 bytes
  as the native no-bias/control sidecar at `conv_ctx.bin+0xc400`.
- Native Conv's final HMX input/output tensors in the bottom mapping are
  `UFixed16` dims `[1,8,32,256]`.  The custom default public-QHPI shape uses
  the same dims, while `OP_INPUT_LAYOUT=native` (`[1,1,256,256]`) remains a
  dense-table diagnostic shape rather than the literal native Conv tensor
  shape.

Custom W4A16 evidence:

- The custom int8 carrier path reaches ctxgen as `UFixed8`
  (`data_type=1032`) weight dims `[1,1,128,256]`; it does not trigger native
  Conv's W4 packer.
- A `native_full_codes` custom probe kept a full `[1,1,256,256]` byte tensor
  and still did not trigger the native W4 packer.
- The best observed pack/layout is `native_nmajor_kpair_hilo` with
  `--bias-layout native_a16`; it remains only `4229/65536` exact.
- The graph now uses a native-shaped fourth control input, `Int32 [1]`, instead
  of the old unused `UFixed8 [1,1,1,2048]` scratch tensor. This does not change
  correctness, but the standard 256^3 probe's constant-move sidecar cycles drop
  from the old class (`6717` in the refreshed pre-control artifact) to `3285`.
  The custom wrapper still passes the local three-word control table by default;
  direct use of the fourth QHPI input is a guarded diagnostic and is worse.
- `OP_INPUT_LAYOUT=native` is now a standard diagnostic path for W4A16. It
  emits HMX activation/output tensors as `[1,1,M,K]` and `[1,1,M,N]`, and the
  converter lowers them through `InputSlice + ForceFormat_Crouton`. This cuts
  the profiled custom main op to `29956` cycles at 256^3, but exactness remains
  only `4092/65536`.
- `HMX_W4A16_DESC_DUMP` on `OP_INPUT_LAYOUT=native` confirms the descriptor
  fields and mask stay identical to the tiled probe (`n_act_pairs=8`,
  `act/out_y_stride=256`, `n_tiles_pow2=256`, mask word 1 `0x700`). The
  material difference is QHPI table shape: native layout exposes dense
  `act_block_entries=512` and `out_block_entries=512`, while the tiled runtime
  dump exposed `64` physical table entries that the custom adapter expanded to
  `512`. Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_descdump_native_op_layout_256/`.
- `OP_INPUT_LAYOUT=native_conv` is also available for a Conv-style input probe.
  Its ONNX input is `[1,K,1,M]` with `QuantizeLinear -> Transpose -> Reshape`
  before HMX, and runtime input bytes are written in NCHW `uint16` order. Ctxgen
  folds this to the same `InputSlice + ForceFormat_Crouton` class; it does not
  improve alignment.
- `OP_INPUT_LAYOUT=native_conv_surface` is a newer W4A16-only diagnostic that
  feeds the custom op with the post-native-Conv QHPI surface shape observed by
  `HmxW4A16TensorDump`: activation/output `UFixed16 [1,256,1,256]`. Host
  conversion and ctxgen pass, and the custom boundary also gets native-shaped
  no-bias/control sidecar `Int32 [1,8,1,64]`, but device execution fails before
  a valid output or optrace is emitted. Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_native_conv_surface_real_256/`.
  Treat this as evidence that the post-Conv layout-restored QHPI surface is not
  the HNH compute surface. The compute target remains the internal
  `ConvLayer_s1.opt` wrapper state with activation/output `[1,8,32,256]`.
- Runtime `HMX_W4A16_DESC_DUMP` without QHPI precompute shows the 256^3
  activation and output QHPI block-table lengths are both `64`, with dense
  native pointer tables expanded to `512` entries. Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_descdump_runtime_blocklen_256/`.
- The enriched dump now keeps the source block-table lengths in precompute mode
  too and can sample one of four table views with
  `HMX_W4A16_DESC_DUMP_TABLE_SELECT={0,1,2,3}`.  The 256^3 custom flow expands
  64 source physical blocks per tensor into 512 HNH table entries by applying
  the row4 `+0x100` offset within each physical block.  This confirms the
  current adapter's table expansion explicitly; it does not by itself explain
  the native mismatch.
- The custom output is heavily saturated: about `27977` zeros and `27614`
  `65535` values in the refreshed best probe, versus native's `3309` zeros
  and `5895` `65535` values.
- A direct output-layout comparison between
  `output_codex_w4a16_control_i32_256/device_out/out.raw` and the quantized
  native oracle found no simple table/order fix.  The best simple transform was
  native transpose plus one-axis flip at `4358/65536`; the normal native
  transpose is `4229/65536`, and row/tile permutations stayed in the same low
  class.  The blocker is therefore arithmetic/control/weight interpretation or
  deeper descriptor state, not a final-output-only transpose or block
  permutation.
- Native `conv_ctx.bin` contains the full high-entropy 32KB prepared-W4 region at
  `0xcc00`; `+0xd000` is an interior offset into that sidecar. Injecting the
  native prepared sidecar directly into the custom weight initializer, with the
  custom `weights_to_vtcm` XOR convention reversed so the custom ctx contains
  identical bytes, does not fix correctness: `native_a16_nobias` reaches only
  `1379/65536`, and generated `native_nmajor_k4_lohi` sidecars stay in the same
  low class. The blocker is therefore not just W4 weight byte order.

Dead ends already checked:

- Weight pointer offsets `64,128,256,512,1024` do not improve alignment;
  the best was `4235/65536` at offset `256`.
- Bias/control pointer offsets `64,128,256,512,1024` do not improve alignment.
- `HMX_W4A16_USE_CONTROL_INPUT`, which passes the fourth QHPI `control`
  initializer pointer to the HNH body instead of the local `[1,1025,524]`
  table, worsens the standard 256^3 native comparison to `903/65536` exact with
  a `96943`-cycle custom main op. Default local-control behavior was restored
  and rechecked at `4229/65536`; artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_control_input_diag_256/`.
- `HMX_W4A16_EXTRA_PARAM0={0,2,4}` all worsen both correctness and, in most
  cases, cycles relative to the default first control word `1`.  Artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_control0_{0,2,4}_256/`.
- Combining `--bias-layout native_a16_w4compact` with
  `HMX_W4A16_EXTRA_PARAM0={0,2}` also worsens exactness relative to compact
  default `1`, despite the deep-body bias/control pointer step semantics.
  Artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_biascompact_control0_{0,2}_256/`.
- Native no-bias control layout `[1,8,1,64]` with repeated `0x80008000`
  reduces saturation but worsens exactness (`1755/65536` in the refreshed
  comparable probe).
- Importing the native prepared W4 sidecar through
  `--w4-native-sidecar-raw`, while keeping current `native_a16` bias/control,
  also worsens exactness (`3041/65536`, artifact
  `example/qnn_matmul_profile/output_codex_w4a16_control_i32_native_sidecar_nativebias_256/`).
- Forcing the W4A16 QHPI weight signature to signed with
  `HMX_W4A16_QHPI_SIGNED_WEIGHT` still fails at ctxgen. The converter/ctxgen
  path reports input tensor[1] as `QUInt8`, produced by
  `ConvLayer.opt.weights_to_vtcm@FB.fB`, so it cannot match a `QHPI_QInt8`
  kernel signature. Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_qhpi_signed_weight_probe_256_host/`.
- ONNX `INT8` initializer / quant-override carrier probes also do not reach
  native `SFixed8`. All four host-only variants below lower the prepared custom
  weight tensor as `UFixed8` (`data_type=1032`) with dims `[1,1,128,256]`;
  native remains `SFixed8` (`data_type=776`) for the comparable Conv path.
  Converter logs for the encoded variants state that `weight` is not
  quantizable because its source datatype is `QNN_DATATYPE_INT_8`, so changing
  `bitwidth` / `offset` in `quant_overrides.json` does not affect the QHPI
  carrier:

  | Artifact | Source / override change | Ctxgen weight carrier |
  |---|---|---|
  | `output_codex_w4a16_dtype_int8_no_weight_encoding_256_host/` | ONNX `INT8` initializer, no weight encoding | `data_type=1032`, `[1,1,128,256]` |
  | `output_codex_w4a16_dtype_int8_bw8_offset0_256_host/` | ONNX `INT8`, weight override bitwidth 8 offset 0 | `data_type=1032`, `[1,1,128,256]` |
  | `output_codex_w4a16_dtype_int8_bw8_offsetm128_256_host/` | ONNX `INT8`, weight override bitwidth 8 offset -128 | `data_type=1032`, `[1,1,128,256]` |
  | `output_codex_w4a16_dtype_int8_bw4_offset0_256_host/` | ONNX `INT8`, weight override bitwidth 4 offset 0 | `data_type=1032`, `[1,1,128,256]` |

  Treat signed-carrier work as blocked below quant-overrides / initializer
  dtype. The next useful route is a different converter/custom-op contract or
  a deeper native descriptor/QHPI ABI decode, not another param-encoding sweep.
- `HMX_W4A16_USE_SKEL_KERNEL` calls the external skel wrapper but preserves the
  same low exactness class.
- `DESC_M_TILES_OVERRIDE=32` lowers custom cycles to about `12.8k`, but only
  computes a small fraction of the output (`~531/65536` exact). It is not a
  performance fix.
- `HMX_W4A16_DESC_M_TOTAL_MINUS_STEP_OVERRIDE=0` preserves the default
  correctness class (`4229/65536`, main op `94171` cycles). Raising it to `16`
  or `32` makes graph execution fail before a valid optrace is emitted.
  Artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_desc_mtotal_{0,16,32}_256/`.
- `HMX_W4A16_DESC_K_TOTAL_BYTES_OVERRIDE=128` cuts the main op to `48576`
  cycles but worsens native exactness to `1972/65536`, consistent with
  under-computing the K span. `512` fails graph execution. Artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_desc_ktotal_{128,512}_256/`.
- Independent `act/out y_stride=8` overrides do not change correctness or
  cycles. Combining `desc_m=32` with `y_stride=8` remains a partial-coverage
  result.
- `MASK_ARG1` sweep over the `0x700` family did not beat `0x70b` class
  correctness. Several bits raise cycles to `~214k`.
- `HMX_W4A16_MASK_ARG4=1..3` targets the helper lane that feeds mask word
  `+0x08` for the native `0x3d9c54` call shape, but all three standard 256^3
  runs stay at `4229/65536` native exactness. Custom main-op cycles are
  `93180..94997`; artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_mask_arg4_{1..3}_256/`.
- `HMX_W4A16_MASK_ARG5=1..7` targets the low-bit helper lane that contributes
  to mask word `+0x08` for `arg1=0x70b`, but all seven standard 256^3 runs stay
  at `4229/65536` native exactness. Custom main-op cycles remain in the
  `93633..94431` range and timeline spans in `134072..144387`; artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_mask_arg5_{1..7}_256/`.
- A builder-derived high-bit probe with `HMX_W4A16_MASK_ARG5=0x20` is a
  correctness no-op: quantized custom output matches the default probe
  `65536/65536`, and the native-output-transposed comparison remains
  `4229/65536`. It profiles at `94819` custom main-op cycles with a
  `140184`-cycle timeline span; artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_mask_arg5_20_builder_256/`.
- External skel wrapper with `HMX_W4A16_MASK_ARG6=0` reaches only
  `4386/65536` and slows to about `122k` cycles. Direct deep with
  `HMX_W4A16_MASK_ARG6=0` drops to `2810/65536`.
- External skel wrapper with `HMX_W4A16_MASK_ARG6={0x4,0xc}` on the standard
  native-contract flow also stays at `4386/65536` exact and slows to about
  `114k` cycles.  Artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_skel_arg6_{4,c}_standard_256/`.
  These low-bit non-deep pre-entry candidates are not the native fast path.
- Converting the prepacked weight initializer to float with an 8-bit symmetric
  override did not produce native `SFixed8`; ctxgen converted it to `UFixed16`.
- `HMX_W4A16_ACT_PHYSICAL_ONLY` plus `HMX_W4A16_OUT_PHYSICAL_ONLY` lowers the
  custom main op to `29385` cycles but does not improve correctness
  (`4092/65536`, maxdiff `65535`) and remains slower than native
  `q::ConvLayer_s1.opt` (`7702` cycles). Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_native_nmajor_kpair_hilo_row4_nativebias_physical_tables_256/`.
- `--bias-layout native_a16_w4compact` changes W4 bias/control to native-shaped
  `Int32 [1,8,1,64]` while preserving the current native-a16 control words. It
  lowers static HMX input read accounting from `169984` to `167936`, but
  worsens correctness to `3846/65536`; artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_control_i32_biascompact_256/`.
- A large-table `y_stride=512` probe with `n_act_pairs=8` and
  `out_table_stride=8` fails graph execution before a valid optrace is emitted.
  Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_control_i32_ystride512_tablegap_256/`.
- A direct `act/out_y_stride=64` hypothesis from a naive read of the
  `0x3d9920` tensor-field stores is not valid for the current public-QHPI
  custom tables. Both the constrained descriptor form (`ACT_N_PAIRS=8`,
  `OUT_TABLE_STRIDE=8`) and the consistent table-stride form fail graph
  execution before optrace decode. Artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_ystride64_desc8_256/` and
  `example/qnn_matmul_profile/output_codex_w4a16_ystride64_table64_256/`.
- `HMX_W4A16_ROW4_BLOCK_ORDER_MOD8`, which forces the W8-style compact
  Crouton16 block order (`block_index=(row4&7)*K_t+kt`,
  `offset=(row4>>3)*256`), does not improve correctness (`4229/65536`) and is
  slightly slower (`95278` cycles). Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_control_i32_row4_mod8_256/`.
  Rechecking the same table order on the current closest native-surface probe
  (`native_nmajor_k4_lohi` W4, `native_a16_nobias`, tiled
  activation/output) also leaves correctness unchanged at `1014/65536` and
  slows the main op from `94610` to `96438` cycles. Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_k4_nobias_native_surface_row4mod8_256/`.
- `OP_INPUT_LAYOUT=native` alone improves custom main-op cycles but not
  correctness (`4092/65536`, artifact
  `example/qnn_matmul_profile/output_codex_w4a16_native_op_layout_probe_256/`).
- Combining `OP_INPUT_LAYOUT=native` with `--bias-layout native_a16_w4compact`
  worsens exactness to `3775/65536`; artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_native_op_layout_biascompact_256/`.
- Combining `OP_INPUT_LAYOUT=native` with `--w4-native-sidecar-raw` worsens
  exactness to `3757/65536`, despite the faster `29515`-cycle main op;
  artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_native_op_layout_native_sidecar_256/`.
- Combining `OP_INPUT_LAYOUT=native`, `--w4-native-sidecar-raw`, and
  `--bias-layout native_a16_nobias` aligns the known native activation shape,
  prepared W4 sidecar bytes, and no-bias/control sidecar bytes, but still
  reaches only `1379/65536` exact with a `30144`-cycle main op. Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_native_op_layout_native_sidecars_nobias_256/`.
- `OP_INPUT_LAYOUT=native_conv` with NCHW `uint16` input worsens exactness to
  `3784/65536` and keeps the main op in the `94k` class; artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_native_conv_input_u16_probe_256/`.
- `HMX_W4A16_MASK_ARG6={0x4,0xc}` on the default public Crouton shape with
  native W4 sidecar and native no-bias/control bytes keeps the same
  `1379/65536` native exactness as default `0x20`.  These candidates therefore
  do not explain the `0x3d9920` builder's dynamic final mask argument.
- Combining `OP_INPUT_LAYOUT=native_conv`, the imported native W4 sidecar, and
  `--bias-layout native_a16_nobias` also stays at `1379/65536`.  The Conv-style
  graph input transpose is not the missing native contract when static
  sidecars are already native-identical.

## Code State

`HmxU16I4ToU16MatMulOp.cpp` now has guarded descriptor override hooks matching
the w8a16 diagnostic style:

- `HMX_W4A16_MAX_TABLE_ENTRIES`
- `HMX_W4A16_MAX_COPIED_TABLE_ENTRIES`
- `HMX_W4A16_ACT_PHYSICAL_ONLY`
- `HMX_W4A16_OUT_PHYSICAL_ONLY`
- `HMX_W4A16_ROW4_BLOCK_ORDER_MOD8`
- `HMX_W4A16_ACT_N_PAIRS_OVERRIDE`
- `HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE`
- `HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE`
- `HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE`
- `HMX_W4A16_DESC_M_TOTAL_MINUS_STEP_OVERRIDE`
- `HMX_W4A16_DESC_K_TOTAL_BYTES_OVERRIDE`
- `HMX_W4A16_USE_CONTROL_INPUT`
- `HMX_W4A16_DESC_DUMP_TABLE_SELECT`
- existing `HMX_W4A16_DESC_M_TILES_OVERRIDE`,
  `HMX_W4A16_WEIGHT_PTR_OFFSET`, and `HMX_W4A16_BIAS_PTR_OFFSET`

Defaults preserve the prior runtime behavior; these hooks are for focused ABI
probes only.

`gen_quant_chain.py` also has W4A16-only diagnostics:

- the fourth input is now native-shaped `control` (`Int32 [1]`) instead of the
  old unused `scratch` (`UFixed8 [1,1,1,2048]`);
- `--op-input-layout native` is enabled for W4A16 activation/output layout
  probes;
- `--op-input-layout native_conv` emits a Conv-style `[1,K,1,M]` graph input
  and native NCHW-order `uint16` runtime input for activation-contract probes;
- `--bias-layout native_a16_w4compact` emits a native-sized 2048B W4
  bias/control tensor for dead-end checking;
- `--w4-native-sidecar-raw <path>` imports a prepared native W4 byte stream and
  applies the custom-op `weights_to_vtcm` carrier XOR convention.
- `HMX_W4A16_QHPI_SIGNED_WEIGHT` is a build-time diagnostic that changes only
  the QHPI weight signature from wildcard to `QHPI_QInt8`. It is intentionally
  off by default because current ctxgen still produces `QUInt8` for the custom
  weight sidecar.

`run_w4a16_chain.sh` exposes the generator layout through `OP_INPUT_LAYOUT`.
Use this rather than hiding layout probes inside `GEN_EXTRA_ARGS`.

The enriched `HMX_W4A16_DESC_DUMP` payload is capped at 256 bytes, matching the
safe first-output-block payload budget.  It decodes through the same Crouton16
row-interleave pattern used by w8a16: for each 32-word group, low 16-bit halves
export first and high 16-bit halves start 256 `uint16` elements later.  The
latest 256^3 descriptor dump reports:

| Field | Value |
|---|---:|
| `M_t`, `N_t`, `K_t` | `8`, `8`, `8` |
| `mt_per_block`, `mt_groups` | `2`, `64` |
| `act_entries`, `out_entries` | `512`, `512` |
| `out_stride`, `out_y_stride` | `8`, `256` |
| `n_tiles`, `m_total_minus_step`, `k_total_bytes` | `256`, `8`, `256` |
| `act_n_pairs`, `act_y_stride` | `8`, `256` |
| mask words | `[0, 0x700, 0, 0x77c, 0, 0, 0x3ff, 0, 0, 0, 0, 0, 0x20, 0, 0, 0]` |

Payload sampling from
`example/qnn_matmul_profile/output_codex_w4a16_descdump_payload_256/`:

| Sample | Value |
|---|---:|
| effective weight first two u32 | `0xf0debc9a`, `0x79563412` |
| effective bias/control first two u32 | `0x80405524`, `0x40000092` |
| default control words `[0..1]` | `1`, `1025` |

For comparison, the native prepared W4 sidecar at
`output_codex_native_w4a16_same_custom_256/ctx/conv_ctx.bin+0xcc00` starts with
`0x0cfbead9`, `0xead9c7b6`, `0xc7b6a594`, `0xa5947362`.  The generated
`native_nmajor_k4_lohi` path now passes these native prepared-W4 bytes to the
HNH body, and the direct native-sidecar injection already proved that byte
identity alone is not sufficient.

The old apparent `0x70b` mask argument expands to a `0x700` word in the actual
mask buffer, so the native descriptor-builder `0x700` evidence is not by itself
a contradiction.

## Native HNH Wrapper ABI Notes

The W4A16 native HNH wrapper evidence is in
`Agent/qnn_re/skel_text_full.S` around these anchors:

- `0x3ddc60` is the V73 HNH wrapper that saves original `r5` in `r16`, uses
  stack base `r21 = r29 + 0x30`, and calls the descriptor builder at
  `0x3d9920`.
- After builder return, the wrapper derives `r19 = base + 0x10`,
  `r20 = base + 0x28`, and `r21 = base + 0x48`. It loads weight and
  bias/control pointers from `base + 0x8` and `base + 0xc`.
- The final `hmx_v73_convhnh1x1_stride1` call at `0x3dde78` passes:
  `r0 = base + 0x28` (`out_desc`), `r1 = base + 0x10` (`act_desc`),
  `r2 = weight`, `r3 = bias/control`, `r4 = base + 0x48` (`mask`), and
  `r5 = original wrapper arg5`.
- The deep HNH body at `0x2fdb80` reads `r5` only as `memw(r5++#0x4)` near
  `0x2fdc0c`, so the first control word is the relevant ABI for this path.
  The custom three-word `[1, 1025, 524]` table is therefore unlikely to be the
  primary W4A16 blocker.  A direct-QHPI-control diagnostic is worse
  (`903/65536`), so the native wrapper's original `r5` is not equivalent to the
  current fourth custom-op initializer pointer.
- Be careful decoding Hexagon packets: stores without `.new` use the old
  register value. Several apparent native builder contradictions come from
  reading same-packet stores as if they used the newly assigned register.

Confirmed deep-body descriptor reads:

| Descriptor | Native pointer | Fields read by `0x2fdb80` |
|---|---|---|
| activation | `base+0x10` (`r1`) | `+0x0` pointer table, `+0x4` pair count, `+0x8` table y stride |
| output | `base+0x28` (`r0`) | `+0x0` pointer table, `+0x4` table stride, `+0x8` y stride, `+0xc` tile/count selector, `+0x10` inner loop span, `+0x14` byte span |
| mask | `base+0x48` (`r4`) | `+0x0..0x18` mask words and `+0x30` through the pre-entry deep selector |
| control | original wrapper `r5` | first 32-bit word only |

Current decoding of the native builder call site at `0x3d9c54`: it calls
`set_hmx_params_convw4b1x1(base+0x48, 0x70b, r28, 0, r4, r21, r6)`, where
`r4`, `r21`, and `r6` are derived from tensor metadata and wrapper flags rather
than literal constants.  The focused `HMX_W4A16_MASK_ARG6={0x4,0xc}` probes
show that simply replacing the custom default final argument does not close the
gap.  The helper itself stores the final stack argument into mask word `+0x30`;
for `arg1=0x70b`, `arg5` low bits feed the mask `+0x08` lane, but
`HMX_W4A16_MASK_ARG4=1..3`, `HMX_W4A16_MASK_ARG5=1..7`, and the
builder-derived high-bit `HMX_W4A16_MASK_ARG5=0x20` are correctness no-ops.  The
remaining work is to decode the full field derivation and compare it against the
enriched descriptor dump, not to keep sweeping one mask lane.

## Next Work

1. Use the decoded `0x3d9920` field map in
   [`w4a16_qnn_native_path.md`](w4a16_qnn_native_path.md) as the source of truth
   for descriptor work.  The scalar fields now point back to QNN tensor metadata;
   the next unknown is whether custom can expose the same internal table/data
   pointers that native stores at `base+0x10` and `base+0x28`.
2. Use `scripts/analyze_w4a16_native_run.py` boundary mismatches to keep probes
   honest.  A fast custom run with logical `[1,1,256,256]` activation/output is
   not a native HNH-boundary match even if its optrace is shorter.
3. Use the enriched `HMX_W4A16_DESC_DUMP` payload to compare QHPI block-table
   shape, pointer deltas, descriptor fields, and final mask words against the
   decoded native wrapper expectations.
4. Investigate whether the custom converter path can expose a prepared
   `SFixed8 [1,1,128,256]` W4 weight tensor to QHPI, or whether the prepared
   native sidecar must be imported through another static-tensor route.
5. Do not repeat the activation-layout, output-permutation, pointer-offset,
   y-stride-only, native-K4 sidecar, `0x700` mask-family, first-control-word, or
   float-weight dtype probes unless new evidence changes the premise.
