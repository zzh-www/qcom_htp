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

Prepared native W4 sidecars can be imported into the standard generator flow for
diagnostics. Extract the native 256^3 sidecar first, then pass it through
`GEN_EXTRA_ARGS`:

```bash
dd if=example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/ctx/conv_ctx.bin \
  of=/tmp/native_w4a16_sidecar_256.raw bs=1 skip=$((0xd000)) count=32768 status=none

GEN_EXTRA_ARGS="--bias-layout native_a16 --a16-quant-contract native \
  --reference-contract native --final-output-rank 3d \
  --w4-native-sidecar-raw /tmp/native_w4a16_sidecar_256.raw" \
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

Latest activation-layout probes:

| Probe artifact | Native exact | Main-op cycles | Timeline span |
|---|---:|---:|---:|
| `output_codex_w4a16_native_op_layout_probe_256/` | `4092/65536` | `29956` | `78824` |
| `output_codex_w4a16_native_op_layout_biascompact_256/` | `3775/65536` | `30708` | `79507` |
| `output_codex_w4a16_native_op_layout_native_sidecar_256/` | `3757/65536` | `29515` | `66969` |
| `output_codex_w4a16_native_op_layout_native_sidecars_nobias_256/` | `1379/65536` | `30144` | `78471` |
| `output_codex_w4a16_native_conv_input_u16_probe_256/` | `3784/65536` | `94236` | `139087` |

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
- Runtime `HMX_W4A16_DESC_DUMP` without QHPI precompute shows the 256^3
  activation and output QHPI block-table lengths are both `64`, with dense
  native pointer tables expanded to `512` entries. Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_descdump_runtime_blocklen_256/`.
- The custom output is heavily saturated: about `27977` zeros and `27614`
  `65535` values in the refreshed best probe, versus native's `3309` zeros
  and `5895` `65535` values.
- Native `conv_ctx.bin` contains a high-entropy 32KB prepared-W4 region at
  `0xd000`. Injecting that native prepared sidecar directly into the custom
  weight initializer, with the custom `weights_to_vtcm` XOR convention reversed
  so the custom ctx contains identical bytes, does not fix correctness:
  `native_a16_nobias` reaches only `1379/65536`. The blocker is therefore not
  just W4 weight byte order.

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
- `HMX_W4A16_MASK_ARG5=1..7` targets the low-bit helper lane that contributes
  to mask word `+0x08` for `arg1=0x70b`, but all seven standard 256^3 runs stay
  at `4229/65536` native exactness. Custom main-op cycles remain in the
  `93633..94431` range and timeline spans in `134072..144387`; artifacts:
  `example/qnn_matmul_profile/output_codex_w4a16_mask_arg5_{1..7}_256/`.
- External skel wrapper with `HMX_W4A16_MASK_ARG6=0` reaches only
  `4386/65536` and slows to about `122k` cycles. Direct deep with
  `HMX_W4A16_MASK_ARG6=0` drops to `2810/65536`.
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
- `HMX_W4A16_ROW4_BLOCK_ORDER_MOD8`, which forces the W8-style compact
  Crouton16 block order (`block_index=(row4&7)*K_t+kt`,
  `offset=(row4>>3)*256`), does not improve correctness (`4229/65536`) and is
  slightly slower (`95278` cycles). Artifact:
  `example/qnn_matmul_profile/output_codex_w4a16_control_i32_row4_mod8_256/`.
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

The enriched `HMX_W4A16_DESC_DUMP` payload decodes through the same Crouton16
row-interleave pattern used by w8a16: for each 32-word group, low 16-bit halves
export first and high 16-bit halves start 256 `uint16` elements later. The
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
`output_codex_native_w4a16_same_custom_256/ctx/conv_ctx.bin+0xd000` starts with
`0x403f2e1d`, `0x2e1d0cfb`, `0x0cfbead9`, `0xead9c7b6`. The current best custom
flow therefore still does not pass native prepared-W4 bytes to the HNH body,
although direct native-sidecar injection already proved that byte identity alone
is not sufficient.

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
`HMX_W4A16_MASK_ARG5=1..7` is also a no-op for correctness.  The remaining work
is to decode the full field derivation and compare it against the enriched
descriptor dump, not to keep sweeping one mask lane.

## Next Work

1. Continue decoding the `0x3d9920` native HNH descriptor builder field
   calculations, especially the values that land in `base+0x10..0x24`
   (`act_desc`/early out fields) and `base+0x28..0x3c` (`out_desc`).
2. Use the enriched `HMX_W4A16_DESC_DUMP` payload to compare QHPI block-table
   shape, pointer deltas, descriptor fields, and final mask words against the
   decoded native wrapper expectations.
3. Investigate whether the custom converter path can expose a prepared
   `SFixed8 [1,1,128,256]` W4 weight tensor to QHPI, or whether the prepared
   native sidecar must be imported through another static-tensor route.
4. Do not repeat the activation-layout, pointer-offset, y-stride-only, `0x700`
   mask-family, or float-weight dtype probes unless new evidence changes the
   premise.
