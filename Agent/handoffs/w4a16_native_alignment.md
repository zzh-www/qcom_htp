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
- The graph now uses a native-shaped fourth control input, `Int32 [1]`, instead
  of the old unused `UFixed8 [1,1,1,2048]` scratch tensor. This does not change
  correctness, but the standard 256^3 probe's constant-move sidecar cycles drop
  from the old class (`6717` in the refreshed pre-control artifact) to `3285`.
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

## Code State

`HmxU16I4ToU16MatMulOp.cpp` now has guarded descriptor override hooks matching
the w8a16 diagnostic style:

- `HMX_W4A16_MAX_TABLE_ENTRIES`
- `HMX_W4A16_MAX_COPIED_TABLE_ENTRIES`
- `HMX_W4A16_ACT_PHYSICAL_ONLY`
- `HMX_W4A16_OUT_PHYSICAL_ONLY`
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
| `extra_param[0..1]` | `1`, `1025` |

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
  primary W4A16 blocker.
- Be careful decoding Hexagon packets: stores without `.new` use the old
  register value. Several apparent native builder contradictions come from
  reading same-packet stores as if they used the newly assigned register.

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
4. Do not repeat the pointer-offset, y-stride-only, `0x700` mask-family, or
   float-weight dtype probes unless new evidence changes the premise.
