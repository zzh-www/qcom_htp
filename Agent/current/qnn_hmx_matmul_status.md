# QNN HMX MatMul Status

## Current Target

The active custom MatMul line is the family of QNN-native quantized HMX MatMul
packages under `example/qnn_hmx_matmul_*`.  The latest completed milestone is
the `w8a16` 256^3 native-rank path above.

Public names:

| Surface | Name |
|---|---|
| Package | `QnnHmxMatMulU8I8Package` |
| Provider | `QnnHmxMatMulU8I8InterfaceProvider` |
| Op | `HmxU8I8ToU8MatMul` |
| XML | `standard_flow/custom_u8i8/QnnHmxMatMulU8I8Package.xml` |
| HTP lib | `libQnnHmxMatMulU8I8_htp.so` |
| CPU lib | `libQnnHmxMatMulU8I8_cpu.so` |
| x86 lib | `libQnnHmxMatMulU8I8.so` |
| Optrace decoder | `scripts/decode_qnn_optrace.py` |
| Legacy u8i8 perf helper | `scripts/perf_hmx_u8i8_matmul.py` |

The custom op signature is intentionally native-aligned:

```text
in[0] bias     int32 Flat4 Direct   TCM_Only  [1, N/32, 1, 64]
in[1] weight   u8    Flat4 Direct   TCM_Only  [1, 1, K, N], bytes are K-major HMX tiles
in[2] act      u8    Crouton_8 Indirect TCM_Only [1, M/32, 32, K]
in[3] scratch  u8    Flat4 Direct   TCM_Only  [1, 1, 1, 2048]
out[0] output  u8    Crouton_8 Indirect TCM_Only [1, M/32, 32, N]
```

## Design Principle

The current QNN MatMul mental model is simple:

```text
HMX only computes.
Everything else is prepared before the hot HMX body.
```

Native QNN does not ask the final HMX kernel to understand high-level MatMul tensors.  It first converts the problem into HMX-native Conv1x1 state:

```text
ordinary MatMul tensors
  -> QNN graph/context/runtime preparation
       - pack weight into K-major HMX tiles
       - fold bias into native bias blocks
       - format activation/output as Crouton_8 TCM blocks
       - move static data through weights_to_vtcm / bias_to_vtcm sidecars
       - prepare pointer tables, mask state, and tile counts
  -> hot compute event
       - stitch tiny native descriptors
       - enter V73DEEP HMX body
```

So the custom rule is: do not put input/layout preparation in the profiled hot callback.  The hot callback should not query tensor metadata, recover shapes, repack weights, copy QNN block tables, patch mask state, or handle broad generic cases.  Those belong in the generator, converter, QNN sidecar ops, or QHPI precompute.

Kernel design summary: this MatMul is implemented as a native HMX Conv1x1 compute body.  The C++ hot callback only translates prepared QHPI state into native descriptors; scale, zero point, and bias are folded into the prepared native bias record before `cvt.ub = acc(...)`.  Keep the detailed explanation in [`docs/qnn_custom_op_matmul_e2e.md`](../../docs/qnn_custom_op_matmul_e2e.md).

## Current Conclusion

The replicated kernel body is not the main gap anymore.  The owned V73DEEP inline-asm body is byte-equivalent to the native `hmx_v73_convbbb1x1deep_stride1` body for the 1132-byte kernel slice, and the 256^3 chain result is bit-exact.

Latest verified hot-op numbers at 256^3 chain8 on device, 2026-05-05:

| Path | cycles | packets | cpp | correctness |
|---|---:|---:|---:|---|
| native QNN kernel-only `ConvLayer_s1.opt` | 1140 | 346 | 3.296 | under-counts QNN MatMul |
| native QNN-op aggregate `MatMul_*` | 1452 | 451 | 3.220 | includes sidecar HTP ops |
| custom `HmxU8I8ToU8MatMul` precompute default | 1165 | 337 | 3.458 | bit-exact |

Current gap statement: after QHPI precompute, graph-load pointer-table copies, mask pre-initialization, and 64B stack-descriptor alignment, the latest clean hot-op run is `1165 - 1140 = +25 cycles` and `337 - 346 = -9 packets` versus native kernel-only `ConvLayer_s1.opt`.  Versus native QNN-op aggregate, custom is `287 cycles / 114 packets` lower because the native aggregate includes sidecar HTP setup events.

Probe split: an intrusive `HMX_U8I8_PROBE_CYCLES` run reports `kernel=1074` and `desc=29`.  The owned inline-asm body is therefore already faster than the native kernel-only chrometrace event (`1074 < 1140`); the remaining `+25 cycles` is QHPI callback/profiling envelope plus tiny descriptor glue and issue/locality effects, not extra matmul work.

Cycle-gap experiments: explicit dcfetch of QNN tables reduced some cycles but cost too many packets; graph-load table copies are the retained version. Writing descriptor/mask/extra into the existing Direct-TCM scratch is bit-exact but much slower (`~1700 cycles`, high cpp); pre-writing those records in QHPI precompute fails at device context creation. Raising the owned kernel entry alignment above 64B, static extra/mask variants, and precomputed descriptor records all worsened cycles.

Historical gap statement: the old custom wrapper path measured `1794 cycles / 471 pkts` because it still did input/layout preparation inside the profiled callback.  Comparing that to native kernel-only produced an apparent `+125 pkts`, but native QNN accounted for much of that preparation in sidecar HTP ops (`bias_to_vtcm`, `weights_to_vtcm`, `DmaCheckpointSet`) and graph-load work.  The real lesson is architectural: HMX should only see prepared TCM/VTCM state and perform compute.

## Quantized MatMul Kernel Goal

Goal: implement the QNN-native quantized inference kernel families that lower `MatMul` and `FullyConnected` into the same Conv1d/1x1-Conv HMX contract. This is not a goal to own the whole generic MatMul OpDef surface. The owned path should keep the current rule: graph-load/QHPI precompute owns layout, packing, bias folding, sidecar-like setup, and descriptor preparation; the profiled callback only stitches the native ABI and enters an owned HMX body.

Confirmed current native target surface:

| Kernel family | Activation | Weight | Weight granularity | Native/QNN signal |
|---|---|---|---|---|
| w4a8 | 8-bit, per-tensor | 4-bit, signed symmetric | per-channel baseline; LPBQ/per-group where blockwise expansion is selected | QAIRT INT4 guide lists `MatMul` and `FullyConnected`; 4-bit weights must be static/const and benefit when output channels > 32. |
| w4a16 | 16-bit, per-tensor | 4-bit, signed symmetric | per-channel baseline; LPBQ/per-group where blockwise expansion is selected | HTP OpDef allows 4-bit low-bit weights on INT16 FC/MatMul-style weight inputs; 16-bit activation paths need the INT16 kernel contract. |
| w8a8 | 8-bit, per-tensor | 8-bit, signed symmetric | per-channel/per-row for MatMul/FC; per-tensor fallback only when a trace confirms it | Existing w8a8 trace lowers to `q::ConvLayer.opt.weights_to_vtcm`, `q::ConvLayer.opt.bias_to_vtcm`, `DmaCheckpointSet`, and `q::ConvLayer_s1.opt`. Current custom path is the first owned instance. |
| w8a16 | 16-bit, per-tensor | 8-bit, signed symmetric | per-channel/per-row; blockwise expansion where native selects it | HTP OpDef lists INT16 activation with 8-bit weights and blockwise-expansion support on the weight input. |
| w16a16 | 16-bit, per-tensor | 16-bit, signed symmetric | per-channel/per-row | A16W16 is documented for convolution-type ops including `FullyConnected` and `Matmul`; it requires 16-bit activation plus 16-bit symmetric weight and v73+ style handling. |

Quantization rules to keep explicit:

| Area | Rule |
|---|---|
| Activation | Owned kernels support activation/output per-tensor quantization only. Do not add activation PCQ/BQ unless a later native trace proves a separate supported path. |
| Weight axis | For `FullyConnected`, row/output-channel quantization is the natural axis. For `MatMul` weight shape `[..., m, n]`, QNN chooses axis `n` when `transpose_in1=false` and axis `m` when `transpose_in1=true`; custom code should normalize packed weights before the hot callback so runtime still sees output-channel-major tiles. |
| Weight encoding | Per-channel uses AXIS or BW_AXIS scale/offset. LPBQ/per-group maps to BLOCKWISE_EXPANSION-style encodings and stays a second-stage feature after the per-channel body is stable. |
| INT4 storage | Treat `w4` as a 4-bit weight kernel family, not an activation type. QNN may represent it as bitwidth=4 in an 8-bit container or as packed 4-bit tensors; custom support must define the exact packed weight ABI per family. |
| Static weights | All owned kernel families assume static weights. Static or precomputed weight/bias movement belongs outside the hot callback. |

Current custom coverage:

| Path | Status |
|---|---|
| w8a8: u8 activation x i8 weight -> u8 output, static weight, s32 folded native bias | Implemented as `HmxU8I8ToU8MatMul`; bit-exact for 256^3 chain8; kernel body aligned with native V73DEEP Conv1x1. |
| Other w8a8 shapes | Partial. The current wrapper still has square/size assumptions and lacks native-style large-shape spill/fill tiling. |
| w4a8 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w4a8` / `HmxU8I4ToU8MatMul`; default builds no longer define `HMX_W4A8_SKIP_KERNEL`. The package embeds the native `hmx_v73_convbnb1x1deep_stride1` body, packs W4 weights as `[1,1,K,N/2]` with native K-major 32x64 tiling plus QNN sidecar sign-bit inversion, uses the W4 mask helper, and expands Crouton8 row8 pointer tables. Device correctness is bit-exact for canonical 256^3 single-kernel and chain8 runs. |
| w8a16 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w8a16` / `HmxU16I8ToU16MatMul`; converter+ctxgen native-rank `chain_qdq` flow passes and embeds native `hmx_v75_convhbh1x1deep_stride1`. The 256^3 native-contract real-kernel path is byte-identical to the saved native QNN artifact; optrace shows `dur=29842`, `pkts=4938`, `cpp=6.04`, aligned with native split kernels totaling about `30839` cycles and `5086` packets. Default builds still keep `HMX_W8A16_SKIP_KERNEL` until broader shape/split coverage is validated. |
| w4a16 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w4a16` / `HmxU16I4ToU16MatMul`; converter+ctxgen smoke passes with `u16` Crouton activation/output and packed `[1,1,K,N/2]` W4 payload. It embeds native `hmx_v73_convhnh1x1deep_stride1`; real A16/W4 QHPI/descriptor, packed-weight ABI validation, and LPBQ/per-group extension remain the compute gate. |
| w16a16 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w16a16` / `HmxU16I16ToU16MatMul`; converter+ctxgen smoke passes with `u16` activation/output/weight carrier. It embeds the native `hmx_v73_convhhh1x1_stride1` aligned path plus unaligned branch target slice; real A16/W16 QHPI/descriptor and bias/scale contract remain the compute gate. |
| Batch/rank/transpose semantics | Not the first implementation goal. Add only as generator/precompute normalization around the quantized Conv1d-family kernels. |

QHPI quantization boundary discovered while implementing the separate families:
custom-op tensors cannot be represented to ctxgen as per-channel QNN quant
tensors.  This matches the current `u8i8` architecture: QNN sees direct carrier
tensors, while per-channel/per-group weight scales, folded bias, and native
packing are owned by the family-specific prepared payload.

Plan:

1. Refresh native evidence for exactly these five families: `w8a8`, `w4a8`, `w8a16`, `w4a16`, `w16a16`. For each, record converter/ctxgen success, lowered node names, weight dtype/bitwidth, quant encoding, sidecar events, tile dimensions, and whether LPBQ/blockwise expansion appears.
2. Stabilize the current w8a8 family first. Remove square-only assumptions for static no-transpose MatMul/FC-as-Conv1d where M/K/N are multiples of the HMX tile requirements, while preserving QHPI precompute and no hot-path preparation work.
3. Add `w4a8` next. Use the same ConvLayer_s1 contract but define the 4-bit packed weight ABI, weight dequant/scaling path, and per-channel scale handling before writing the HMX body.
4. Add A16 families in order: `w8a16`, then `w4a16`, then `w16a16`. These need the 16-bit activation/output descriptor and accumulator/convert policy; `w16a16` is lower priority because it is a different weight-width body.
5. Add LPBQ/per-group only after the matching per-channel family is correct. Treat blockwise expansion as an ABI extension with explicit block scale tables, not as a generic per-channel tweak.
6. Keep generic MatMul features out of the critical path. Rank-5, batch broadcasting, and transpose flags should be implemented by generator/layout normalization and repeated prepared Conv1d jobs, not by making the hot kernel parse high-level MatMul semantics.
7. Verification gate per family: converter+ctxgen pass, bottom-mapping/optrace confirmation, bit-exact or tolerance-based output check, native aggregate and kernel-only perf comparison, and a trace check proving no packing/shape recovery work happens inside the profiled custom op.

Near-term milestone: support static MatMul/FullyConnected-as-Conv1d `w8a8` with per-tensor activation, per-channel signed symmetric static weights, folded s32 native bias, no transpose in the hot callback, and arbitrary M/K/N within the current HMX tiling constraints. Then implement `w4a8` on the same prepared-state architecture.

Broader quantized-kernel roadmap status: not complete.  Do not conflate the
now-aligned `w8a16` 256^3 milestone with the full-family acceptance rule:

1. implementation: each target family has an independent quantized inference
   kernel path, default build is non-`SKIP_KERNEL`, and the hot callback only
   enters prepared native HMX compute;
2. correctness: each family passes device output validation against real QNN
   native output for the same graph/input.  Analytic formulas are diagnostic
   references only, not the final acceptance oracle;
3. perf alignment: each family has a native QNN comparison covering both
   aggregate MatMul/FC lowering cost and the final Conv1x1/HMX kernel event.

Do not mark the quantized-kernel Goal complete unless all three checks pass
for `w8a8`, `w4a8`, `w8a16`, `w4a16`, and `w16a16`.

Latest probe evidence:

2026-05-08 artifact-standard note: treat
`example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/` as a
historical comparator only.  Its device run used `A:=input_A.raw` and emitted
fp32-sized output, and its converter run did not record the required
NONTRIVIAL layout flags.  Regenerate W4A16 native Conv references with
`example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh` before using QNN
native output or performance as the current oracle.

Current standardized W4A16 native oracle:
`example/qnn_matmul_profile/output_native_w4a16_conv_ref_256/`.  It uses
native u16 runtime input/output, `--retrieve_context`, converter NONTRIVIAL
layout flags on A/Y, and the full `optrace/` artifact set.  Its native
`q::ConvLayer_s1.opt` event is `7893` cycles, `conv1x1` QNN-op aggregate is
`37287` cycles, and the graph timeline span is `313032` cycles.  The current
custom W4A16 flow is not aligned yet: generated native-K4 packing is
`2883/65536` exact against this oracle, and importing the clean native
`conv_ctx.bin+0xbd00` candidate sidecar gives `3298/65536` exact but
`sorted_equal=True` with a `+32` row roll becoming `65536/65536`.  Remaining
boundary differences are weight carrier `UFixed8` versus native `SFixed8` and
custom control `Int32 [1,1,1,1]` versus native `Int32 [1]`.

Follow-up native instrumentation shows the local skel override path works:
an invalid isolated `libQnnHtpV75Skel.so` fails at Device Creation, and the
`/tmp/libQnnHtpV75Skel_hmx_entry_probe.so` patch at `0x2fcd80` changes native
output and writes magic `0x484d5850`.  This confirms the current native path
enters `hmx_v73_convhnh1x1_stride1`; the remaining instrumentation problem is
making the internal ConvLayer output-tile descriptor dump survive the post-Conv
output transforms in a linear, parseable form.

| Probe | Result | Next gate |
|---|---|---|
| Native QNN W4 MatMul 128^3, param bitwidth 4 | Lowers to `q::ConvLayer_s1.opt`; `weights_to_vtcm` sees `[1,1,128,64]` `SFixed8`, confirming W4 uses a 4-bit signed carrier with two output channels per byte. |
| Native QNN W4 same-weight probe | DLC stores full N-major 4-bit codes, then ctxgen lowers through `weights_to_vtcm` to the packed `[1,1,128,64]` native payload. Custom full-float fallback becomes Float16 and fails QHPI kernel matching; custom full-int8/QLinear probes stay `UFixed8 [1,1,128,128]`, so they do not reach the native W4 packer. |
| Custom w4a16 256^3 native-contract probe | Current best diagnostic flow is `native_nmajor_kpair_hilo` weight packing plus `native_a16` bias/control and real HMX enabled. It reaches the HMX body but remains unaligned: latest refreshed probe is `4229/65536` exact against `output_codex_native_w4a16_same_custom_256/device_out/Y.raw`, custom main op `94579` cycles, while native `q::ConvLayer_s1.opt` is `7702` cycles. The graph now uses a native-shaped `Int32 [1]` control input instead of the old unused 2048B scratch tensor, but passing that fourth QHPI input directly to the HNH body is worse (`903/65536`), so default local-control behavior stays in place; sweeping the first local control word to `{0,2,4}` is also worse than default `1`. The deep body uses that first word as a bias/control pointer step, but compact native-shaped bias/control with first word `{0,2}` is also worse than compact default `1`. Raw activation values and logical W4 codes are confirmed exact against native after the native `[1,K,1,M] -> [1,1,M,K]` activation transpose and W scale decode, so the gap is after QNN lowering into prepared sidecars/descriptors. Directly injecting the native prepared W4 sidecar from `conv_ctx.bin+0xcc00` also fails, so the blocker is not just weight byte order. Physical-only act/out pointer tables reduce custom main op to `29385` cycles but still fail correctness (`4092/65536`). `OP_INPUT_LAYOUT=native` similarly lowers the main op to `29956` cycles but remains incorrect (`4092/65536`); combining native activation shape, native W4 sidecar, and native no-bias/control bytes still reaches only `1379/65536`. Follow-up all-native-sidecar probes with `MASK_ARG6={0x4,0xc}` and with `OP_INPUT_LAYOUT=native_conv` stay at `1379/65536`; `MASK_ARG4=1..3`, `MASK_ARG5=1..7`, and the builder-derived high-bit `MASK_ARG5=0x20` also leave the standard flow at `4229/65536`. External skel non-deep `MASK_ARG6={0x4,0xc}` standard probes stay at `4386/65536` and are very slow. Descriptor field probes show `k_total_bytes=128` is a partial-compute false path (`1972/65536`, `48576` cycles), larger `m_total/k_total` values fail execution, and direct `act/out_y_stride=64` hypotheses also fail execution. The enriched descriptor dump now records source block lengths and selectable table samples: the custom adapter expands 64 physical QHPI blocks into 512 HNH table entries with a row4 `+0x100` intra-block offset pattern. A simple output transpose/flip/block permutation is not enough either. Continue with native HNH descriptor-builder decoding rather than repeating input-layout, static-sidecar, output-layout, control-word, or single-lane mask probes. See `Agent/handoffs/w4a16_native_alignment.md` before continuing. |
| Custom w4a8 256^3 default precompute, native `convbnb` body, packed `[1,1,K,N/2]`, twos-complement lo/hi nibbles, W4 mask helper | Device correctness is bit-exact for both single-kernel and chain8 runs (`65536/65536`). Custom chain8 perf observed `HmxU8I4ToU8MatMul` hot avg `dur=19005`, `pkts=2229`, `cpp=8.526`; native W4 same-shape perf compare is still required before closing the strict Goal. |
| Custom w8a16 256^3 native-rank `chain_qdq` graph | Ctxgen passes and device execution reaches the HMX body. The fast default descriptor uses mask `arg1=0x70b`, `n_tiles_pow2=row4_groups*4` (`256` at 256^3), and `m_total_minus_step=8`. Output is byte-identical to `output_codex_native_w8a16_custom_full_256/device_out/out.raw`. The analytic native-contract reference is only diagnostic (`22057/65536` exact and `65536/65536` within `abs<=3`). Optrace reports `dur=29842`, `pkts=4938`, `cpp=6.04`, aligned with native split `ConvLayer_s1.opt` total `30839` cycles and about `5086` packets. |

Latest W4A16 continuation notes: `HMX_W4A16_MASK_ARG2=128` is also a no-op
(`4229/65536`, `93932` cycles).  `DESC_M_TILES_OVERRIDE=32` reaches native-class
kernel cycles (`6877` cycles with `OP_INPUT_LAYOUT=native`, `13686` cycles with
the tiled HMX surface), and `desc32 + physical-only tables` reaches `6566`
cycles, but all remain semantic false paths (`505/65536`, `531/65536`, and
`4092/65536` exact, respectively).  Standard W4A16 runs now write
`<out_dir>/analysis/w4a16_native_compare.{json,txt}` in addition to
`<out_dir>/optrace/`.

After the clean-native artifact refresh, copying the visible native output
descriptor scalars into the closest imported-sidecar flow is also closed:
`HMX_W4A16_DESC_M_TILES_OVERRIDE=32` plus
`HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE=64` gives only `408/65536` exact,
`sorted_equal=False`, best row32 roll `32:8193`, and `14046` main-op cycles.
This reinforces that the row32 rotation/mismatch is not an isolated output
descriptor scalar.

2026-05-08 native-field probes: on the closest native-surface flow
(`native_nmajor_k4_lohi` plus `native_a16_nobias`), forcing descriptor
`act/out +0x08` to `8` as a builder-formula hypothesis improves only to
`1419/65536` exact with `94156` main-op cycles and preserves the half-written
`32767` N32 signature.  Combining the same field hypothesis with
`HMX_W4A16_INTERNAL_SPLIT_N128` fails graph execution before a valid optrace.
Treat descriptor y-stride and the existing split diagnostic as insufficient
without the full native wrapper record.

2026-05-08 descriptor-dump enrichment: `HMX_W4A16_DESC_DUMP` now records raw
QHPI activation/output tensor-object words, and
`scripts/parse_w4a16_desc_dump.py` prints them.  Artifact
`output_codex_w4a16_descdump_qhpi_words_k4_nobias_native_surface_nativeout_256`
confirms the usual custom descriptor fields and table samples, but the QHPI
tensor words are opaque handle-like values, not the readable metadata exposed by
the native post-Conv tensor-dump diagnostic.  This reinforces that native
`ConvLayer_s1.opt` internal tensor metadata must still be obtained from the
native wrapper path.

2026-05-07 continuation: the canonical native prepared-W4 sidecar was corrected
from the old interior `conv_ctx.bin+0xd000` note to the full 32768-byte region
at `conv_ctx.bin+0xcc00`.  `W4_PACK_ORDER=native_nmajor_k4_lohi` now reproduces
that sidecar byte-for-byte (`N32 tile -> K8 group -> n-in-tile -> k/k+4`, twos
nibbles), and standard-flow custom contexts embed the exact stream at
`w4a16_ctx.bin+0xa000`.  This still does not fix semantics:
`output_codex_w4a16_native_op_k4pack_256` is `3507/65536` exact with `30243`
main-op cycles, `output_codex_w4a16_k4pack_tiled_256` is `2872/65536` with
`96246` cycles, and the native-entry skel wrapper preserves the same `3507`
exact class while slowing to `59686` cycles.  Treat native-K4 packing as a
closed byte-order diagnostic; continue with native HNH descriptor-builder and
QHPI tensor-contract decoding.

Follow-up descriptor probes: combining native-K4 with compact W4 bias/control
does not rescue the sidecar path (`3183/65536` native-layout, `3142/65536`
tiled).  For `desc32`, keeping table storage stride at `8` and forcing
descriptor y-stride to `256` preserves the same false classes (`531/65536`, or
`4092/65536` with physical-only tables).  This rules out y-stride as the missing
desc32 field.

Native W4A16 path analysis is now split out in
`Agent/handoffs/w4a16_qnn_native_path.md`.  Treat it as the required entrypoint
for future custom work: native starts from float Conv plus W4 quant overrides,
ctxgen lowers weight to `SFixed8 [1,1,128,256]`, bias/control to
`Int32 [1,8,1,128]`, activation/output to `UFixed16 [1,8,32,256]`, then
`ConvLayer_s1.opt` enters the HNH wrapper/descriptor-builder/deep-body path.
The 2026-05-07 helper decode adds that `set_hmx_params_convw4b1x1` with
`arg1=0x70b` and final flags `0x20` ignores `arg2` for the observed default
mask words, explaining the `HMX_W4A16_MASK_ARG2=128` no-op.  Do not continue
single-lane mask sweeps before the native metadata tuple feeding the helper is
understood.
The W8-style compact row4 table order has now also been checked on the current
closest native-surface W4A16 probe (`native_nmajor_k4_lohi` plus
`native_a16_nobias`): it keeps exactness at `1014/65536` and slows the main op
to `96438` cycles, so table block order is not the remaining native mismatch.
The analyzer now records top-value N32 distribution as part of the standard
W4A16 quick-read report.  The current closest native-surface run has exactly
32768 outputs stuck at `32767`, all in N32 groups 0..3, while native's top value
is distributed across all groups.  Treat the next step as native output
descriptor/table/mask-state analysis, not another formula or byte-packing
probe.
`HMX_W4A16_OUT_TABLE_N_ROTATE=4` flips that stuck half to N32 groups 4..7
(`1219/65536` exact), proving the single custom call currently consumes only
the upper four output-table entries per row4 group.  A guarded full-descriptor
`HMX_W4A16_INTERNAL_SPLIT_N128` diagnostic writes both halves but is still
wrong and much slower (`2346/65536`, `204675` main-op cycles); the W8-style
split with `k_total_bytes=128` is worse (`587/65536`), and keeping
`k_total_bytes=256` in that split fails execution.  Continue by decoding the
native wrapper's per-half metadata tuple rather than promoting split as a fix.
The native HNH wrapper also has a descriptor-advance loop after each body call:
`0x3de060` increments `out_desc+0x00` by `r24`, increments `act_desc+0x00` by
`r27 * 4`, and loops until `r26 == r23`.  Those `r23/r24/r27` values come from
native tensor metadata and are not present in the current custom descriptor
dump, so the next useful native decode is that loop tuple plus the mask helper
tuple, not another custom-side table rotation.
The static decode now maps that loop tuple to QNN internal tensor metadata:
`r23=activation.meta[0x04]`, `r24=4*((output.meta[0x20]>>5)*
(output.meta[0x1c]>>2)*(output.meta[0x18]>>3))`, and
`r27=(activation.meta[0x20]>>5)*(activation.meta[0x1c]>>2)*
(activation.meta[0x18]>>3)`.  Bottom mapping exposes only visible shapes, not
these metadata words or native internal table bases.

Direction change: pause custom-side alignment patches until the QNN native
runtime path is instrumented or otherwise decoded.  The next W4A16 evidence
should be a native dump of wrapper args, tensor-object metadata, post-builder
stack descriptors, mask words, and the `r23/r24/r27` loop tuple for the
canonical 256^3 artifact.  After that, compare the native stack record with the
custom descriptor dump and only change the custom path for a named native
boundary.

2026-05-08 native-path diagnostic added: `HmxW4A16TensorDump` is registered in
the W4A16 package and `run_native_conv_tensor_dump.sh` now generates a side
branch `Conv(Y) -> HmxW4A16TensorDump(Y -> D)` from the canonical native Conv
artifact.  The standard artifact directory is
`example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/out/native_conv_tensor_dump_256/`.
It keeps decoded optrace under `optrace/` and parses `device_out/D.raw` into
`device_out/tensor_dump.{json,txt}`.  Latest dump validates the public QHPI
surface exposed after native Conv/layout restore: UFixed16, quant zero
`32768`, scale `3.0518509447574615e-05`, shape `[1,256,1,256]`, block shape
`[1,8,2,32]`, block table length `256`, and first block pointers
`0x04550000`, `0x04550800`, `0x04551000`, ... .  Ctxgen still contains native
`q::ConvLayer_s1.opt` with activation/output `[1,8,32,256]` and W sidecar
`SFixed8 [1,1,128,256]`, but the diagnostic branch perturbs bias/control to
`Int32 [1,8,1,128]`; use this as QHPI tensor-surface evidence, not as the
final wrapper-state dump.  The remaining target is still the native `0x3ddc60`
wrapper metadata/stack/mask/loop tuple.

The new `OP_INPUT_LAYOUT=native_conv_surface` diagnostic tries to feed
`HmxU16I4ToU16MatMul` directly with that post-Conv public surface
(`UFixed16 [1,256,1,256]`).  Host conversion/ctxgen pass and the boundary gets
native no-bias/control `Int32 [1,8,1,64]`, but device execution fails before a
valid optrace or output.  Artifact:
`example/qnn_matmul_profile/output_codex_w4a16_native_conv_surface_real_256/`.
This rules out treating the post-Conv export/custom-op surface as the HNH
compute surface; continue targeting the internal `ConvLayer_s1.opt`
activation/output `[1,8,32,256]` wrapper state.

2026-05-08 host layout sweep follow-up: converter output-layout flags for
`D`, and for `Y` plus `D`, were tested with `NCHW`, `NHWC`, and
`NCHW -> NHWC` forms on the native Conv tensor-dump graph.  All ctxgen bottom
mappings kept `HmxW4A16TensorDump` at `UFixed16 [1,256,1,256]` while native
`q::ConvLayer_s1.opt` stayed at the internal `UFixed16 [1,8,32,256]` HNH
boundary.  Treat public layout flags as closed for W4A16 native alignment.

2026-05-08 native-first direction update: W4A16 alignment work is now gated on
the QNN native implementation path, not another custom descriptor sweep.  The
visible HTP contract is known (`UFixed16 [1,8,32,256]`, `SFixed8
[1,1,128,256]`, `Int32 [1,8,1,128]`, `Int32 [1]` plus
`Int32 [1,1,1,3]`), but the missing state is the
runtime `ConvLayer_s1.opt` wrapper record: QNN tensor-object table pointers,
metadata-derived descriptor fields, W4 mask-helper arguments, and the
`r23/r24/r27` descriptor-advance tuple.  A direct patched-skel descriptor-dump
attempt did not produce usable runtime evidence.  Even an isolated all-local
backend/stub/skel rerun produced byte-identical canonical native output, so the
patched entry was not active; do not repeat that route without first proving the
patched HTP library is loaded.  See
`Agent/handoffs/w4a16_qnn_native_path.md` before starting any new custom probe.

The signed W4 carrier route is still blocked below quant-overrides.  Four
host-only probes using an ONNX `INT8` initializer plus no weight encoding,
8-bit offset `0`, 8-bit offset `-128`, and 4-bit offset `0` all lower the
prepared custom weight tensor as `UFixed8` (`data_type=1032`) with dims
`[1,1,128,256]`; none produce native Conv's `SFixed8` (`data_type=776`) carrier.
Do not repeat param-encoding sweeps until the converter/custom-op contract can
actually expose a signed QHPI tensor.

Latest w8a16 probes on 2026-05-07:

- The native-rank `MODE=chain_qdq --op-input-layout native --final-output-rank 3d` path is now the reference custom flow. With real HMX enabled through `-UHMX_W8A16_SKIP_KERNEL -DHMX_W8A16_ALLOW_UNVALIDATED_KERNEL`, the source defaults produce output byte-identical to `example/qnn_matmul_profile/output_codex_native_w8a16_custom_full_256/device_out/out.raw` (`65536/65536`, maxdiff `0`). Treat that QNN native artifact as the oracle; the analytic native-contract reference is only a diagnostic cross-check (`22057/65536` exact and `65536/65536` within `abs<=3`, `max=3`).
- The production descriptor fix is the combination `HMX_W8A16_MASK_ARG1=0x70b`, `n_tiles_pow2=row4_groups*4`, and `m_total_minus_step=8`. The earlier `m_total_minus_step=456` workaround was correct but slow (`dur=185183`, about `35178` packets); the fast default profiles at `dur=29842`, `pkts=4938`, `cpp=6.04`.
- Native QNN for the comparable 256^3 artifact lowers to two 128-channel `q::ConvLayer_s1.opt` HMX kernels (`17675` and `13164` cycles, total `30839`; about `5086` packets). Use that split-kernel total as the closest kernel-only target for the single custom node.
- Full-range native w8a16 weight layout is now validated. For `output_codex_w8a16_256/w8a16/ctx/matmul_w8a16_ctx.bin`, the native `shflSWeights [1,1,256,256]` payload contains the same 32x32 K-major W8 stream as the validated u8i8 packer at `0x9100`; custom `--w8-pack-order kmajor` embeds the same stream at `0xa000`. The older K-major regression only applied to special low-pattern/compact probes, not ordinary full W8.
- Native A16 quantization is symmetric QUInt16 with zero offset `32768` and step `1/32767`. Direct UINT16 custom inputs ignore quant overrides, but `chain_qdq --a16-quant-contract native` preserves the native metadata through `ForceFormat_Crouton`.
- Native full-range output is not the old saturated integer reference. It is approximately `round(((A_u16 - 32768) @ W_i8) / 127 + 32768)` clipped to u16, but that formula is only a reference model. Future native-contract custom probes must compare primarily to a real QNN native output artifact such as `output_codex_native_w8a16_custom_full_256/device_out/out.raw`; formula/tolerance checks are secondary diagnostics.
- Native w8a16 bias/control bytes are decoded for the full-W8 path. Each 32-output-channel tile is a 512B record: two 256B halves for even and odd channels; every lane starts with u16 `[0x4440, 0x8040, 0x0008, 0x4000]`, and the variable i32 lane value is `-128 * sum(W[:, n])`. `--bias-layout native_a16` reproduces `native_bias_0x19100.bin` byte-for-byte.
- Matching W8 bytes, native A16 metadata, and decoded native bias is still not sufficient in the old tiled custom graph: `output_codex_w8a16_chain_qdq_native_contract_kmajor_nativebias_256` reaches the HMX body but only matches native `2774/65536`; its `ForceFormat_Crouton` writes only `131072` bytes, while native A16 writes `1048576`.
- A native-rank single custom graph (`--op-input-layout native --final-output-rank 3d`) now executes on device. Output-only mode writes the expected marker pattern through a `[1,1,256,256]` internal custom output reshaped to `[1,256,256]`, so the older failure was the graph-output/export rank boundary, not native-rank allocation itself.
- Historical native-rank failures are now explained by the descriptor/mask pair, not by W8 bytes, decoded native bias, or input quant metadata. Early direct-`UINT16` probes reached only `288/65536`; `chain_qdq` fixed A16 input metadata but stayed low until the mask changed to `0x70b` and `n_tiles_pow2` changed to `row4_groups*4`. Internal split-N128 is no longer needed for correctness or kernel-cycle alignment on the single 256-channel custom node.
- A split-N graph (`--op-input-layout native_split`) mirrors native lowering as two 128-channel custom HMX nodes, each with `32768` W bytes, `2048` bias bytes, and `524288` output bytes, but still fails device graph execution. A follow-up `--native-split-output-mode separate` probe removes `Concat` and exports the two halves independently; it still fails in output-only mode, so concat is not the sole failure boundary.
- A single non-square native-rank graph shaped like one native split node (`M=256,K=256,N=128`) was enabled for chain=1 diagnostics. The final-3D graph executes but QHPI exposes no output block table to the output-only callback, leaving an uninitialized exported tensor; the final-4D graph fails execution. This blocks direct one-node native split comparison through the current public QHPI output contract.
- A hybrid single-op graph (`--op-input-layout native_in_tiled_out`) proves native-rank activation is accepted when the custom output stays in the old tiled shape: output-only diagnostics execute, `ForceFormat_Crouton` writes `1048576` bytes for activation, and the custom op writes `131072` bytes. Real HMX execution is still not native-aligned (`2574/65536` exact against the native full-range artifact with the embedded deep body, `2563/65536` with the external skel wrapper), so the blocker has moved from activation acceptance to the remaining descriptor/output/bias ABI.
- Internal 128-channel split inside that single accepted custom op is not sufficient. `--w8-pack-order kmajor_split128` plus `HMX_W8A16_INTERNAL_SPLIT_N128` keeps the graph executable but only reaches `2582/65536` exact on the compact-output hybrid; the native-rank final-3D internal-split probes remain low even after corrected `chain_qdq` input metadata (`290/65536` deep after native A16 re-quantization, older direct-input probes `287/65536` deep and `412/65536` skel non-deep).
- A row4-shaped output probe (`--op-input-layout native_in_row4_out`, output `[1,64,4,N]`) executes in output-only mode, but ctxgen still reports only `131072` custom-op write bytes. It does not coerce QNN into the native 512-block output allocation; the working path is the single-node native-rank final-3D output, while split and row4 output contracts remain poor public-QHPI matches.
- QHPI `Layout_Any` is not a workaround for native A16 tensor contracts. With default `Indirect` storage, ctxgen rejects `Any` as an invalid indirect input/output storage type; with `Direct_OR_Indirect`, the native-split graph segfaults in ctxgen graph optimization. The working single-node path stays on public `Crouton_16` plus final-3D export.
- The compact-output skel branch is also not a semantic fix. Forcing `HMX_W8A16_USE_SKEL_KERNEL` with `HMX_W8A16_MASK_ARG5=0` makes the executable `native_in_tiled_out` graph take the non-deep branch, but it only reaches `2899/65536` exact against native full-range output and remains heavily saturated.
- The current evidence no longer says native-rank custom output is impossible. It says the public QHPI contract is inconsistent across native A16 shapes: `[1,1,256,256]` plus final 3D export works for one custom node, two `[1,1,256,128]` custom nodes fail execution, and one `[1,1,256,128]` custom node runs without exposing an indirect output block table. SDK tensor definitions expose `QUint16Crouton_AR4/AR8`, but QHPI's standard layout enum does not expose those layouts for custom-op signatures.
- `QHPI_Layout_Custom` is still not a usable public DeepAR escape hatch in this package: `HMX_W8A16_OUT_LAYOUT=QHPI_Layout_Custom` with `Direct_OR_Indirect` segfaults ctxgen during graph optimization for native-rank output. A direct Flat4 output signature runs only for the compact output shape; native-rank direct output still fails or exports unusable data before a semantic HMX comparison is possible.
- Added a guarded diagnostic `HMX_W8A16_DIRECT_OUTPUT_RAW` that synthesizes an HMX output pointer table from a direct Flat4 output buffer. It proves the compact direct-output graph can execute, but both deep and non-deep HMX stores are semantically wrong (`308/65536` and `344/65536` exact against native full-range), so row-major direct output is not a simple replacement for DeepAR.
- Host probing `qhpi_standard_layout()` for IDs `0..80` and `1000..1015` confirms the public QHPI layout table stops at ID 16 (`WideCrouton2x2`). There are no hidden public IDs for `Crouton_16_DeepAR4` or `Crouton_16_DeepAR8`; those layouts exist in SDK tensor definitions only, not in the QHPI standard layout API.
- The wrapper now derives w8a16 tile counts from QHPI tensor shapes and chooses direct logical row4 Crouton pointers when QNN exposes a native 512-entry table, while preserving the old compact 64-entry row4 slicing for the default tiled graph.
- Historical bias initializer probes showed that native-shaped `[1, N/32, 1, 128]` / one 512B record per N tile was necessary but not sufficient; descriptor/mask fixes were still required for native equivalence.
- Descriptor sweeps resolved the core native-rank 256^3 ABI. With old mask `0x700`, `DESC_M_TILES_OVERRIDE` and `m_total` changes only improved overlap accidentally. With mask `0x70b`, holding `m_total=8` and raising `DESC_M_TILES_OVERRIDE` expands valid row coverage; `DESC_M_TILES_OVERRIDE=256` gives full native equivalence, while `320+` fails graph execution. This is now encoded as the default formula `n_tiles_pow2=row4_groups*4`.
- The exact V75 hbh wrapper at `Agent/qnn_re/skel_text_full.S:547134` calls the hbh descriptor builder `0x3d9920`, not the older bbb-oriented `0x3d7920` helper. That builder passes the hbh kernel descriptors as `r1=base+0x10` activation desc, `r0=base+0x28` output desc, `r4=base+0x48` mask. It calls `set_hmx_params_conv1x1` through the native hbh path with `arg1=0x70b` and nontrivial later args; however focused mask-lane sweeps around `arg4=32`, `arg5={0x20,0x2c,0x400,0x420,0x800,0x820}` only reached `2498/65536` best and kept `abs_le3=3172/65536`, so the mask tuple is not a standalone fix.
- `HMX_W8A16_DESC_DUMP` on the executable native-rank final-3D path can be decoded by reversing the Crouton16 export row interleave. The current fast descriptor for 256^3 is `S=256, M_t=N_t=K_t=8, mt_per_block=1, out_stride=8, out_y_stride=8, desc_m=256, m_total=8, k_total=256, act_n_pairs=8, act_y_stride=8`; default mask expansion comes from `set_hmx_params_conv1x1(..., arg1=0x70b, arg5=0x20)`.
- The w8a16 run-flow verifier now re-quantizes float output using `quant_overrides.json` before comparing. This matters for native A16 output because qnn-net-run emits dequantized floats unless native output files are requested; direct float rounding had inflated corrected-metadata scores such as `2307/65536`.
- Native split remains blocked at graph execution before callback semantics: with `HMX_W8A16_EARLY_RETURN`, both split-concat and split-separate `MODE=chain_qdq --op-input-layout native_split --final-output-rank 3d` graphs still fail execution, so the two-node `[1,1,256,128]` custom-output contract is rejected even when the custom callback returns before reading precomputed state.
- Historical bias-layout probes with the native-shaped 512B record showed that `swapped` and `a16_eff_all` could improve low-saturation diagnostics but did not solve the old compact/native-hybrid paths. Keep them as evidence that bias bytes alone were not the missing native-rank contract.
- Native bottom mapping uses signed weight VTCM type (`data_type=776`) while the custom integer-carrier path still reaches QHPI as `QUInt8`. Forcing the guarded `HMX_W4A16_QHPI_SIGNED_WEIGHT` signature fails ctxgen because input tensor[1] remains `QUInt8` from `ConvLayer.opt.weights_to_vtcm@FB.fB`; using float weights with symmetric overrides makes the custom op fall back to float and also fails ctxgen. Do not flip the default weight signature until the converter path can produce a signed QHPI tensor.
- The newer ONNX signed-carrier probe (`--w8-carrier-dtype int8`) is accepted by ctxgen on the native-input hybrid but worsens correctness (`2439/65536`), so signed carrier alone is not the missing runtime contract.
- Native ConvLayer_s1 carries a three-word Int32 control tensor `[1, 1025, 524]`. The custom wrapper passes all three words by default instead of the old two-word `{1, 0}` table; this fixed an ABI hazard before the final descriptor/mask correction.
- W8 payload packing for ordinary full-W8 native lowering is the validated u8i8 32x32 K-major tile order. Use `--w8-pack-order kmajor` for native-contract probes; the raw default remains only for legacy/tiled diagnostics.
- Signed carrier probing remains blocked below the custom OpDef layer. Even with an ONNX `INT8` initializer, signed-only XML, and a signed QHPI weight signature, ctxgen still reports input tensor[1] as `QUInt8` from `ConvLayer.opt.weights_to_vtcm@FB.fB` and fails kernel matching. The XML now permits signed and unsigned W8 so this can be reprobed without losing the default unsigned path.
- Bias scale was a real but incomplete signal in legacy compact-output probes. Tiny nonzero scales improved low-saturation scores, while `bias_scale=0` was a false positive caused by all-zero output. The native-rank aligned path uses decoded native A16 bias records instead of these scale sweeps.
- The comparable native artifact for the exact symmetric low pattern (`example/qnn_matmul_profile/output_codex_native_w8a16_custom_low_256`) does not lower like the ordinary random/native w8a16 path. Native `weights_to_vtcm` outputs `QInt8 [1,1,64,256]` with only `16384` bytes of DRAM/VTCM traffic, while ordinary native w8a16 uses `shflSWeights [1,1,256,256]`. DLC `model.params.bin` still stores the original 65536 signed bytes, and the context binary contains a repeating 2-bit-looking `4d d3 34` region but no simple full 16 KiB candidate match. Treat the low-saturation score as a diagnostic, not as a clean native-full-W8 comparator.
- Follow-up native marker artifacts are pattern-sensitive: `output_codex_native_w8a16_marker_ternary_256` and `output_codex_native_w8a16_marker_2bit_256` both lower through split `shflSWeights [1,1,256,128]` prepares, not the compact `[1,1,64,256]` path. The next useful w8a16 direction is therefore to recover/validate the native `shflSWeights` layout for full W8, not to patch the wrapper toward the special compact low-pattern path.

Do not mark the quantized-kernel goal complete from scaffold/build/ctxgen alone.

## How QNN Appears To Call This

QNN does not call a custom op like an internal `q::*` primitive.  The observed design is:

```text
ONNX custom node
  -> converter XML + converter op package shape/type hooks
  -> ctxgen resolves OpPackage provider and QHPI precompute registration
  -> runtime inserts surrounding built-ins for tensor movement/layout
  -> QHPI precompute records bias/weight pointers, copies small Crouton block tables, pre-initializes mask state
  -> hot callback stitches small native descriptors on stack
  -> owned V73DEEP inline-asm kernel runs
```

The important architectural boundary is the QHPI callback and accounting model.  Native QNN splits MatMul setup across sidecar HTP ops and graph-load preparation; the custom path must mirror that by using QHPI precompute so `HmxU8I8ToU8MatMul` is effectively a compute event.

## Perf Reading

The standard performance artifact flow is shared by all OPs:

```bash
scripts/decode_qnn_optrace.py <out_dir>
```

Every device run must keep `device_out/qnn-profiling-data_0.log` and a
matching `*schematic.bin`, then decode with `scripts/decode_qnn_optrace.py`.
Custom chain scripts call this automatically unless `DECODE_OPTRACE=0` is set.
Use `STRICT_OPTRACE=1` when a missing decode should fail the run.

For W4A16 probes, the standard runner additionally writes
`<out_dir>/analysis/w4a16_native_compare.{json,txt}` through
`scripts/analyze_w4a16_native_run.py`.  Treat that report as the quick-read
artifact for output exactness, row4/N32 coverage, saturation, custom main-op
cycles, and native kernel-cycle comparison.  The raw optrace directory remains
the timeline source of truth.

The standard decoded artifact directory is `<out_dir>/optrace/`:

| Artifact | Use |
|---|---|
| `chrometrace.json` | Primary per-event HTP timeline. |
| `chrometrace_htp.json` | Compiled HTP graph and per-node memory counters. |
| `chrometrace_htp_graph_before.json` | Pre-optimization HTP graph view. |
| `chrometrace_runtrace.json` | Runtime trace sidecar. |
| `chrometrace_qnn_htp_analysis_summary.json/html` | QHAS graph/resource summary. |
| `profile.txt` | Plain HTP profiling-reader text output for legacy wall/cycle lines. |
| `summary.json` | Repo-local stable summary: timeline span, pid0 event sum, per-HTP-type cycles, per-QNN-op cycles. |
| `manifest.json` / `_viewer.log` / `_optrace_config.json` | Repro metadata and viewer diagnostics. |

Meaning:

| Metric | Use |
|---|---|
| `dur` / cycles | user-visible hot-op time on HTP timeline |
| `pkts` | best proxy for extra executed Hexagon packet work |
| `cpp` | sanity check for stalls/issue quality; important after packet accounting is fixed |

Use `dur` as the final latency answer.  Use `pkts` to decide whether custom is
doing extra work.  Current precompute default is packet-better than native
kernel-only (`-9 pkts`), so the remaining `+25 cycles` is not caused by extra
matmul instructions.

Use QNN-op aggregate view for native comparisons.  Kernel-only `ConvLayer_s1.opt`
is useful for studying the HMX body, but it excludes native setup events that
the custom op pays inside its own callback.

`scripts/perf_hmx_u8i8_matmul.py` remains useful for old u8i8 packet-gap
reports, but new OP performance evidence should cite the standard `optrace/`
artifact set above.

Optional diagnostic build:

```bash
EXTRA_DEFS=-DHMX_U8I8_PROBE_CYCLES bash example/qnn_hmx_matmul_u8i8/build.sh
python scripts/perf_hmx_u8i8_matmul.py <out_dir> --probe-cycles
```

## Do Not Retry

Do not spend more time on these unless new evidence appears:

- Reintroducing old V2-V8, row-major, spike, pack/copy/untile kernels.
- Swapping back to dlsym/native-kernel call paths.
- VTCM pointer-table scratch as a gap closer; it did not remove the packet gap.
- PMU-heavy probes as a normal measurement path; they perturb packet counts.
- Parameter sweeps around old descriptor flags as production candidates.
- Compatibility aliases for old package/op names; regenerate DLC/context binaries instead.

## Evidence Kept

Non-Markdown RE evidence remains in `Agent/qnn_re/`:

```text
skel_text_full.S
hmx_v73_convbbb1x1deep_stride1_2ebe40.S
hmx_v73_convbbb1x1_stride1_2eadc0.S
set_hmx_params_conv1x1.S
wrapper_3dc2a8.S
descriptor_builder_3d7920.S
descriptor_builder_3d7920_full.S
descriptor_builder_full.S
descriptor_builder_pt2.S
hmx_convbbb1x1_stride1_2ea740.S
hmx_convbbb1x1_stride1_full.S
hmx_convbbb1x1_stride1_FULL_decoded.S
```

Use those only when re-checking ABI, descriptor fields, or wrapper control flow.

## Common Commands

```bash
bash example/qnn_hmx_matmul_u8i8/build.sh
bash example/qnn_hmx_matmul_u8i8/build_x86.sh

cd example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8
SKIP_DEVICE=1 bash run_u8i8_chain.sh

python scripts/perf_hmx_u8i8_matmul.py \
  example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/out/u8i8_chain_256
```

## Documentation Rule

Keep `Agent/README.md` as the short entrypoint.  Put longer current-status
notes in `Agent/current/`, handoffs in `Agent/handoffs/`, and non-Markdown RE
evidence in `Agent/qnn_re/`.
