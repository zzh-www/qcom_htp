# w4a16 Native Alignment Handoff

Current status: `example/qnn_hmx_matmul_w4a16`
(`HmxU16I4ToU16MatMul`, i4 weight x u16 activation -> u16 output) now has a
256^3 chain8 native-contract path that is bit-exact against the clean QNN
native oracle when built with the real HMX body.  The shape/chain acceptance
gate is closed: all eight custom and native kernel nodes enter with activation
shape `UFixed16 [1,8,32,256]`, custom main cycles are `31419`, native
`q::ConvLayer_s1.opt` aggregate is `29815`, and custom/native outputs are
`65536/65536` exact.

Acceptance rule for this family:

1. Correctness must compare against a real QNN native output artifact for the
   same graph/input. Analytic formulas are diagnostic references only.
2. Performance must use the standard optrace artifact set under
   `<out_dir>/optrace/`, not ad-hoc `/tmp/_optrace*` files.
3. Report both custom main-op cycles and native comparable kernel cycles.
   For native W4A16 Conv, the closest kernel-only event is
   `q::ConvLayer_s1.opt`; full native graph timeline also includes
   transpose/quantize/dequantize work.
4. Do not accept single-op W4A16 performance numbers as final.  The final
   performance gate is chain-form custom/native comparison with the same
   kernel-entry activation shape `(1,8,32,256)`.  The current canonical
   artifact satisfies this for 256^3 chain8.

2026-05-08 standardization update: the old
`example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/`
artifact and the earlier `output_codex_native_w4a16_conv1x1_*` artifacts are
historical because they used float-sized runtime output and/or did not record
the required converter layout-preservation flags.  Refresh the native oracle with
`example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh` before taking new
correctness or performance numbers from QNN native.
2026-05-09 cleanup note: most `output_codex_*`, `output_w4a16_import_*`, and
other exploratory directories referenced later in this handoff were deleted as
temporary artifacts.  Treat those names as historical evidence labels; the live
canonical artifacts are `output_w4a16_aligned_e2e_256/` and
`output_w4a16_native_ref_e2e_256/`.

Current clean native oracle:

```text
example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/
```

This artifact uses native u16 runtime input/output, converter NONTRIVIAL layout
flags on A/Y, a context binary run via `--retrieve_context`, and the standard
`optrace/` directory.  The native output oracle is
`device_out/Y.raw`; the native performance reference is
`optrace/summary.json`.  The ONNX Conv graph still has a float public surface,
matching QNN's native quantized-model entry convention; do not treat that as the
runtime comparison contract.  Runtime acceptance is the `native_io.json` u16 raw
contract plus qnn-net-run native I/O flags.

Native-path first rule: before adding new custom probes, read
[`w4a16_qnn_native_path.md`](w4a16_qnn_native_path.md).  The canonical 256^3
chain8 blocker is closed by matching two named native boundaries: compact
source tables and K32-block-major/N32-inner W4 sidecar order, then validating
the kernel-entry activation shape and chain-form performance.  Broader shape
coverage plus LPBQ/per-group extensions follow after that.

## Standard Flow

Build the real-kernel packages explicitly:

```bash
EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w4a16/build.sh

EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w4a16/build_x86.sh
```

Run the current aligned custom flow:

```bash
OUT_DIR="$PWD/example/qnn_matmul_profile/output_w4a16_aligned_e2e_256" \
M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq \
NATIVE_OUTPUT=1 STRICT_OPTRACE=1 \
VERIFY_NATIVE_RAW="$PWD/example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/device_out/Y.raw" \
VERIFY_NATIVE_TRANSPOSE=1 \
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

All standard custom W4A16 runs now call
`scripts/check_qnn_artifact_standard.py <OUT_DIR> --require-native-io --require-layout-flags --reject-float-io` after
optrace decode.  Set `STRICT_ARTIFACT_STANDARD=0` only for deliberately
historical/debug runs.

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

Native patched-skel entry/base-record/table probes should use the checked-in
parser as well:

```bash
scripts/parse_w4a16_native_entry_probe.py /tmp/qnn_loader_probe_w4a16/Y.raw
```

Use the default `--layout crouton512 --record-kind auto` mode for the current
v3 entry probe, base-record probe, and table probe.  A pure-assembly pattern
probe recovered the first-512-word internal-output to public-output map, so the
parser now reconstructs the mask/table/weight/bias samples instead of relying on
the older stride-only read.  `HMXP` records are HNH entry samples; `HMXB`
records dump the prebuilt record at `base = out_desc - 0x28`; `HMXT` records
dump the native table memory starting at the active output and activation table
pointers.

`HMX_W4A16_DESC_DUMP_TABLE_SELECT` selects which 16-entry pointer-table sample
is written into the 256-byte dump payload:

| Value | Sample |
|---:|---|
| `0` | expanded activation table used by the HNH descriptor |
| `1` | expanded output table used by the HNH descriptor |
| `2` | source QHPI activation block table before expansion |
| `3` | source QHPI output block table before expansion |

Prepared native W4 sidecars can be imported into the standard generator flow for
diagnostics.  The old float-I/O artifact had a byte-for-byte generated W4
sidecar at `conv_ctx.bin+0xcc00`; that offset is historical and must not be
used as the current oracle.  In the current clean native reference, the best
candidate 32KB region starts at `ctx/conv_ctx.bin+0xbd00`.  Importing it was the
first proof that the compact-table path is correct:

```bash
dd if=example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/ctx/conv_ctx.bin \
  of=/tmp/native_w4a16_ref256_sidecar_0xbd00.raw bs=1 skip=$((0xbd00)) count=32768 status=none

GEN_EXTRA_ARGS="--bias-layout native_a16 --a16-quant-contract native \
  --reference-contract native --final-output-rank 3d \
  --w4-native-sidecar-raw /tmp/native_w4a16_ref256_sidecar_0xbd00.raw" \
bash example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh
```

The generator now synthesizes the clean-native sidecar order without importing
raw bytes:

```bash
W4_PACK_ORDER=native_kblock32_nmajor_k4_lohi \
W4_NIBBLE_ENCODING=twos \
bash example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh
```

## Current Evidence

Durable native oracle:

| Artifact | Path |
|---|---|
| native output | `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/device_out/Y.raw` |
| native input | `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/runtime_inputs_native/A.raw` |
| native context | `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/ctx/conv_ctx.bin` |
| native optrace | `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/optrace/` |
| refreshed custom probe | `example/qnn_matmul_profile/output_w4a16_k4pack_vs_nativeio_256/` |
| imported clean-native sidecar probe | `example/qnn_matmul_profile/output_w4a16_import_native_sidecar_bd00_vs_nativeio_256/` |

Current clean native boundary:

| Tensor | Contract |
|---|---|
| activation | `UFixed16 [1,8,32,256]` |
| weight | `SFixed8 [1,1,128,256]` |
| bias | `Int32 [1,8,1,128]` |
| control | `Int32 [1]` |
| extra control | `Int32 [1,1,1,3]` |
| output | `UFixed16 [1,8,32,256]` |

Latest refreshed custom results:

| Probe | Result |
|---|---:|
| generated native-K4 pack vs clean native output | `2883/65536`, maxdiff `65535` |
| generated native-K4 custom main-op cycles | `94809` |
| generated native-K4 custom timeline span | `134987` |
| imported `0xbd00` sidecar vs clean native output | `3298/65536`, maxdiff `65535` |
| imported `0xbd00` sidecar custom main-op cycles | `94196` |
| imported `0xbd00` sidecar custom timeline span | `136463` |
| imported `0xbd00` sidecar permutation check | output multiset equals native; `np.roll(custom, 32, axis=0)` is `65536/65536` exact after native-output transpose |

Before the compact-table fix, the imported clean-native sidecar result was the
strongest signal: arithmetic values were present, but the custom output was
rotated by one 32-row block group.  Existing
`HMX_W4A16_ROW4_BLOCK_ORDER_MOD8` did not fix that rotation.

Latest native-compact-table resolution:

| Probe artifact | Variant | Native exact | Main-op cycles | Timeline span |
|---|---|---:|---:|---:|
| `output_w4a16_native_compact_source_tables_256/` | `HMX_W4A16_NATIVE_COMPACT_SOURCE_TABLES` plus imported clean `0xbd00` sidecar | `65536/65536` | `5179` | `44845` |
| `output_w4a16_native_compact_generated_k4_256/` | compact source tables plus old `native_nmajor_k4_lohi` pack | `3522/65536` | `5629` | `42549` |
| `output_w4a16_native_compact_kblock32_pack_256/` | compact source tables plus generated `native_kblock32_nmajor_k4_lohi` pack | `65536/65536` | `6808` | `64996` |
| `output_w4a16_aligned_e2e_256/` | current chain8 e2e runner, real HMX body, encoded QAIRT flow, native kernel-entry shape matched | `65536/65536` | `31419` | `77854` |

This closes the two missing named native payload boundaries for the canonical
256^3 path and the later chain8 shape/performance gate:

- table/descriptor boundary: use the native 64-entry compact source tables
  observed by HMXT/HMXR, not the custom 512-entry row4-expanded table;
- W4 sidecar boundary: the clean native sidecar is the same 512-byte K4 tile
  payload as `native_nmajor_k4_lohi`, but with the outer 8x8 chunk grid ordered
  K32-block-major then N32 tile.

The custom graph boundary still reports the W4 tensor carrier as `UFixed8`
instead of native `SFixed8`, and the control tensor shape as `[1,1,1,1]`
instead of `[1]`; these are boundary-reporting differences after the generated
payload matches native bytes.  They do not prevent canonical output or main-op
performance alignment.

Follow-up table-source probes with the same imported clean-native sidecar:

| Probe artifact | Variant | Native exact | Permutation | Main-op cycles |
|---|---|---:|---|---:|
| `output_w4a16_import_native_sidecar_bd00_act_physical_256/` | `HMX_W4A16_ACT_PHYSICAL_ONLY` | `4540/65536` | `sorted_equal=False`, best row32 roll `32:12048` | `30865` |
| `output_w4a16_import_native_sidecar_bd00_out_physical_256/` | `HMX_W4A16_OUT_PHYSICAL_ONLY` | `12048/65536` | `sorted_equal=False`, best row32 roll `0:12048` | `93964` |

These probes rule out a simple physical-table substitution.  The default
activation table reconstruction is necessary to keep the native value multiset,
and the output physical table does not preserve the imported-sidecar
`sorted_equal=True` property.  Continue at the native wrapper/descriptor-loop
state, especially the `r23/r24/r27` table-base advance tuple, before adding any
new custom table-order logic.

Native entry and descriptor follow-up:

| Probe artifact | Result |
|---|---|
| isolated invalid skel in `~/qnn_loader_probe_w4a16` | Device Creation fails, proving local `libQnnHtpV75Skel.so` override is active under the isolated run command. |
| `/tmp/libQnnHtpV75Skel_hmx_entry_probe.so` | Native run completes with `Y.raw` SHA `372fecf39290f38b9d345e1e3e3cbf2fb986ee78283946a4b87934787593a0ca` instead of canonical `147b7752a5f8c55f59c8539d65dcffe69214e01f27f157f7ccd540d9377822a8`; first output word is probe magic `0x484d5850`, so `0x2fcd80` is on the current native path. |
| `output_w4a16_import_native_sidecar_bd00_descdump_256/` | Custom descriptor dump with the same imported sidecar reports `out_y_stride_words=256`, `act_table_y_stride_words=256`, `n_tiles_pow2=256`, `m_total_minus_step=8`, `k_total_bytes=256`, mask `[0,0x700,0,0x77c,...,0x20]`. |
| `output_w4a16_import_native_sidecar_bd00_out_y64_256/` | Forcing only `HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE=64`, based on the native entry-probe stride sample, leaves the result unchanged: `3298/65536`, `sorted_equal=True`, best row32 roll `32:65536`. |
| `output_w4a16_import_native_sidecar_bd00_desc32_out_y64_256/` | Forcing both visible native output descriptor scalars, `HMX_W4A16_DESC_M_TILES_OVERRIDE=32` and `HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE=64`, is a semantic false path: `408/65536`, `sorted_equal=False`, best row32 roll `32:8193`, main-op `14046` cycles. |
| `/tmp/libQnnHtpV75Skel_hmx_r31_probe.so` | Patching `0x2fcd80` to write `r31` reports return address `0x03de46c`, so the current clean native artifact reaches HNH through the simple prebuilt-record wrapper call at `0x3de464`, not the earlier hypothesized `0x3dde78` call. |
| `/tmp/libQnnHtpV75Skel_hmx_direct_pattern_endloop_probe.so` | A pure-assembly pattern probe with `:endloop0` recovers the public-output map for the first 512 internal u32 words: `public = (i % 32) * 128 + ((i // 32) // 2) * 16 + ((i // 32) & 1)`. |
| `/tmp/libQnnHtpV75Skel_hmx_entry_probe_v3.so` | Corrected copy loops and the recovered map produce reliable native samples: mask `[0,0x700,0,0x77c,0,0,0x3ff,0,0,0,0,0,0xa0,0,control_ptr,0]`, out table `0x046a0000..0x046a7800`, act table `0x046c9000..0x046d0800`, weight words `0x0cfbead9,0xead9c7b6,0xc7b6a594,0xa5947362`, control `[1,0x401,0x20c,0]`. |
| `/tmp/libQnnHtpV75Skel_hmx_base_record_probe.so` | Base-record probe confirms the active record pointer contract: `r31=0x031de46c`, `base=0x02d99408`, `weight=0x046c0000`, `bias=0x046c8000`, `act_desc=[0x02d99288,8,64,32,8,256]`, `out_desc=[0x02d994a0,8,64,32,8,256]`, mask `[0,0x700,0,0x77c,0,0,0x3ff,0,0,0,0,0,0xa0,0,control_ptr,0]`, and `control=0xfdd01c00`. |
| `/tmp/libQnnHtpV75Skel_hmx_table_probe.so` | Table-memory probe confirms the active table pointers expose a compact first 64-entry view: output table `0x046a0000..0x046bf800` and activation table `0x046c9000..0x046e8800`, both contiguous by `0x800`.  Entries after 64 are adjacent wrapper/metadata memory, not another 448 entries of the custom public-QHPI row4-expanded table. |
| `/tmp/libQnnHtpV75Skel_hmx_act_tile_probe.so` | Activation table[0] data probe (`HMXA`) confirms the first block stores pairs of M rows with K contiguous: rows `0/1`, then `2/3`, then `32/33`, then `34/35`, etc. for K `0..31`. |
| `/tmp/libQnnHtpV75Skel_hmx_act_entry_probe.so` | Per-entry activation probe (`HMXV`) maps compact `act_table[i]` to logical activation coordinates: first samples are `m=(i//8)*4` and `m+1`, with `k=(i%8)*32 + {0,1,2}`. |
| `/tmp/libQnnHtpV75Skel_hmx_act_block_full_probe.so` | Full activation block0 dump covers all 512 u32 offsets with zero formula misses: for `j=group*32+k`, the two u16 halves are logical activation `(m,k)` and `(m+1,k)`, where `m=32*(group//2)+2*(group&1)`. |
| `/tmp/libQnnHtpV75Skel_hmx_out_marker_probe.so` | Output table marker probe writes a unique marker to each compact `out_table[i]`.  In exported native `Y.raw` viewed as `[8,32,256]`, marker `i` lands at `m32_group=i%8`, `row=0`, `n=(i//8)*4` and `n+1`. |
| `/tmp/libQnnHtpV75Skel_hmx_out_block_probe.so` | Output block0 marker probe maps the first 256 u32 offsets inside `out_table[0]`: for `j=group*32+row`, the marker lands at `m32_group=0`, `row`, and `n=32*(group//2)+2*(group&1)` / `n+1`. |
| `/tmp/libQnnHtpV75Skel_hmx_out_block_full_probe.so` | Full output block0 marker probe covers all 512 u32 offsets with zero misses and confirms the same formula through `n=226/227`; `out_table[0]` covers one M32 group and all N pairs in the `[0,2,32,34,...,224,226]` order. |
| `/tmp/libQnnHtpV75Skel_hmx_record_window_probe.so` | Record-window probe (`HMXR`) dumps 496 u32 words starting at `base-0x180 = 0x02d99288`: compact activation table at words `0..63`, pre-base metadata at `64..95`, base record at `96..131`, compact output table at `134..197`, post-output metadata at `198..223`, and an adjacent restore/public-table-looking sample starting at `0x02d99608`. |
| `/tmp/libQnnHtpV75Skel_hmx_adjacent_marker_probe.so` | Negative marker check for the neighboring table at `base+0x200`: writing paired markers through its first 64 pointers produces no paired marker hits in exported `Y.raw`.  Treat it as adjacent wrapper/layout state, not the active HNH output table or direct public export table. |
| `/tmp/libQnnHtpV75Skel_hmx_record_prewindow_probe.so` | Pre-window probe (`HMXW`) starts at `base-0x300 = 0x02d99108` and reveals another descriptor-like record plus a contiguous 64-entry table at `0x02d99168` before the compact activation table. |
| `/tmp/libQnnHtpV75Skel_hmx_pretable_marker_probe.so` | Negative marker check for the `0x02d99168` pre-table: writing paired markers through its first 64 pointers produces no paired marker hits in exported `Y.raw`, so it is also not a direct public export table at the HNH entry point. |
| `output_w4a16_import_native_sidecar_bd00_maskarg6_a0_256/` | Forcing custom `HMX_W4A16_MASK_ARG6=0xa0` to match native mask word `[12]` is a no-op for correctness: `3298/65536`, `sorted_equal=True`, best row32 roll `32:65536`, main-op `95152` cycles. |
| `output_w4a16_import_native_sidecar_bd00_base_record_fields_256/` | Applying the visible base-record scalar fields to the imported-sidecar custom flow (`act_y=64`, `out_y=64`, `out_n_tiles=32`, `mask[12]=0xa0`) fails graph execution before a valid optrace/output.  The analyzer still records the graph-boundary mismatch: custom weight carrier `UFixed8` vs native `SFixed8`, and custom control shape `[1,1,1,1]` vs native `[1]`. |

The native entry probe currently writes into the internal ConvLayer output tile;
downstream native output ops transform that tile before `Y.raw` is emitted.  Do
not treat older public dumps as a fully linear descriptor record.  The
`crouton512` map above now inverts the first 512 u32 words for the v3 probe.
The stride-sampled fields that are readable today are still useful: native
entry sees `r2(weight)=0x046c0000`, `r3(bias)=0x046c8000`,
`r5(control)=0xfdd01c00`, `out_table=0x02d994a0`, and output descriptor scalar
words `[8,64,32,8,256]` after the table pointer.  Since copying just the
candidate `64` y-stride into custom is a no-op, the missing state is not a
single output descriptor scalar.  Copying both visible native `64` y-stride and
`32` tile selector was also a false path, so this historical probe showed the
gap was in the full native wrapper state rather than an isolated output
descriptor field.

The current native wrapper is simpler than the earlier `0x3ddc60` hypothesis for
this artifact.  It enters at `0x3de3c0` with a prebuilt record pointer in `r0`;
the HNH call at `0x3de464` passes `r0+0x28` as output descriptor, `r0+0x10` as
activation descriptor, `memw(r0+0x8)` as weight, `memw(r0+0x80)` as control, and
`r0+0x48` as mask.  The base-record probe now makes those scalar fields
explicit, and directly applying them to the custom public-QHPI table adapter is
not executable.  Continue by decoding the native table pointer arrays and loop
state behind `act_table_ptr=0x02d99288` and `out_table_ptr=0x02d994a0`, not by
copying more isolated descriptor constants.

The first table-memory probe closes the obvious "copy the native scalars into
the custom adapter" route.  At the active native table pointers, only the first
64 entries are the compact contiguous HNH table view; reading beyond them falls
into nearby wrapper records and other metadata.  The custom adapter currently
expands 64 public QHPI physical blocks into 512 row4-offset HNH entries.  That
is a different structure, so the next custom change must reproduce a named
native compact-table/wrapper state, not splice the native `[8,64,32,8,256]`
descriptor words into the existing 512-entry table path.

The follow-up activation/output layout probes make that named state more
concrete without adding any C/C++ probe code.  Native compact activation table
entries use `i//8` as the 4-row residue and `i%8` as the K32 tile.  Native
compact output table entries use the opposite visible order in exported
`Y.raw`: `i%8` selects the M32 group and `i//8` selects the N4 residue.  The
current custom table copy is still built as a 512-entry row4-expanded table
indexed `row4 * stride + tile`, so its output-table order is not the compact
native order observed by the marker probe.
Within one compact activation block, the full-block dump gives
`j = group * 32 + k`, `k = j % 32`, and
`m_pair_base = 32 * (group // 2) + 2 * (group & 1)`.  The two u16 halves are
logical activation rows `m_pair_base` and `m_pair_base + 1`.
Within one compact output block, the full-block marker probe gives
`j = group * 32 + row`, `row = j % 32`, and
`n_pair_base = 32 * (group // 2) + 2 * (group & 1)`.  Combining that with the
table-entry marker means native output table entry `i` selects
`m32_group = i % 8` and an N4 residue `4 * (i // 8)`.

The record-window dump anchors those tables in one contiguous native record
window.  The active `act_table_ptr` is exactly `base-0x180`, and the active
`out_table_ptr` is `base+0x98`; there are two metadata regions around the base
record, plus a neighboring pointer table starting at `base+0x200`.  Treat that
neighbor as adjacent QNN wrapper/layout state for now; it is not the HNH compact
output table passed at `out_desc+0`.  The adjacent-marker negative probe also
shows it is not a direct public-export table at the HNH entry point.
The wider pre-window dump shows `base+0x90` points back to a descriptor-like
record at `base-0x300`, with its own table at `base-0x2a0`; that table also
fails the paired-marker public-export check.  The only table proven to feed HNH
output remains the compact `out_desc+0` table at `base+0x98`.

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

- The `conv_ctx.bin+0xcc00` finding belongs to the historical float-I/O
  artifact.  Keep it only as a byte-order diagnostic.
- The current clean native reference has a different context layout.  The best
  32KB candidate region tested so far starts at `conv_ctx.bin+0xbd00` and has
  SHA-256
  `6f1016e71e87b727032c17528eaae834dab48017afbe5feae93703cd01a25bf6`.
- Importing that current clean-native candidate sidecar makes the custom output
  multiset exactly match native and a row32 roll of `+32` become bit-exact, but
  it still reports only `3298/65536` without the roll.  This points to a
  row32-block output/table rotation plus remaining boundary differences, not a
  pure arithmetic formula issue.
- The historical physical byte order remains useful:
  `N32 tile -> K8 group -> n-in-tile -> k4`, with each byte pairing
  `(k+0,k+4)`, `(k+1,k+5)`, `(k+2,k+6)`, and `(k+3,k+7)` as two's-complement W4
  nibbles.  `W4_PACK_ORDER=native_nmajor_k4_lohi` reproduces that historical
  stream but does not reproduce the full clean native context.

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

Native performance reference from the current clean artifact,
`output_w4a16_native_ref_e2e_256/optrace/summary.json`:

| Native event/view | Cycles |
|---|---:|
| `q::ConvLayer_s1.opt` chain8 aggregate | `29815` |
| `q::ConvLayer.opt.weights_to_vtcm` | `3385` |
| `q::ConvLayer.opt.bias_to_vtcm` | `1117` |
| native `conv1x1_*` QNN-op aggregate | `70408` |
| native graph timeline span | `253245` |

The current custom chain8 main op is `31419` cycles and is bit-exact against
the native chain8 output.  Earlier single-op or native-shaped activation/output
probes below are historical diagnostics.

Latest descriptor-table dump artifacts:

| Probe artifact | `HMX_W4A16_DESC_DUMP_TABLE_SELECT` | Sample class |
|---|---:|---|
| `output_codex_w4a16_descdump_table_act_local_256/` | `0` | expanded activation table |
| `output_codex_w4a16_descdump_table_out_local_256/` | `1` | expanded output table |
| `output_codex_w4a16_descdump_table_act_source_256/` | `2` | source activation block table |
| `output_codex_w4a16_descdump_table_out_source_256/` | `3` | source output block table |
| `output_codex_w4a16_descdump_qhpi_words_k4_nobias_native_surface_nativeout_256/` | `0` | expanded activation table plus raw QHPI tensor-object words |

All four decode with `act_block_entries=64`, `out_block_entries=64`,
`act_entries=512`, and `out_entries=512`.  The source QHPI tables are contiguous
physical block pointers, e.g. activation starts at `0x04020000, 0x04020800,
... 0x04027800`.  The expanded HNH descriptor tables insert the row4 offset
after each 8-tile physical row, e.g. activation entries `[8..15]` become
`0x04020100, 0x04020900, ... 0x04023900`; output follows the same pattern from
`0x04000000`.  Use this as the custom-side table-shape baseline when comparing
against native builder expectations.

`HMX_W4A16_DESC_DUMP` now also records activation and output QHPI tensor-object
raw words at descriptor words `64..95`, and
`scripts/parse_w4a16_desc_dump.py` prints them as `act_tensor_words` and
`out_tensor_words`.  The first captured custom native-surface dump shows opaque
handle-like values rather than the readable pointer/metadata words seen in the
native post-Conv `HmxW4A16TensorDump` artifact.  Treat this as a boundary
finding: custom QHPI tensor handles are not a substitute for the native
`ConvLayer_s1.opt` wrapper's internal tensor-object metadata.

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
can run in native-class cycles (`6566`, versus the current clean native kernel
at `7502`), but the output distribution falls back to the saturated failure
class.  The
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
- The old best observed pre-compact-table pack/layout was
  `native_nmajor_kpair_hilo` with `--bias-layout native_a16`; it remained only
  `4229/65536` exact.  The current clean-native pack/layout is
  `native_kblock32_nmajor_k4_lohi` plus compact source tables, which is
  `65536/65536` exact for the canonical 256^3 oracle.
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
  (`4092/65536`, maxdiff `65535`) and remains slower than the current clean
  native `q::ConvLayer_s1.opt` (`7502` cycles). Artifact:
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

For historical comparison only, the prepared W4 sidecar in the old float-I/O
artifact at `output_codex_native_w4a16_same_custom_256/ctx/conv_ctx.bin+0xcc00`
starts with `0x0cfbead9`, `0xead9c7b6`, `0xc7b6a594`, `0xa5947362`.  The generated
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
