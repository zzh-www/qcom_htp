# QNN HMX MatMul Status

## Current Target

The active custom MatMul line is the family of QNN-native quantized HMX MatMul
packages under `example/qnn_hmx_matmul_*`.  The latest completed canonical
milestone is W16A16 256^3 through the scoped `W16A16_KERNEL_PROFILE=accepted`
native-record path: custom/native raw output is bit-exact, generated sidecars
match the native prepared streams, and the W16A16 analyzer gate reports
`alignment_gate.accepted=True`.  W4A16 remains the latest strict chain8
same-surface milestone; W16A16 is accepted with an explicit boundary policy for
one tiled custom op internally issuing two N128 native-record calls.

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
| w4a8 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w4a8` / `HmxU8I4ToU8MatMul`; default builds no longer define `HMX_W4A8_SKIP_KERNEL`. The package embeds the native `hmx_v73_convbnb1x1deep_stride1` body, packs W4 weights as `[1,1,K,N/2]` with native K-major 32x64 tiling plus QNN sidecar sign-bit inversion, uses the W4 mask helper, and expands Crouton8 row8 pointer tables. Device correctness is bit-exact for canonical 256^3 single-kernel and chain8 runs. Performance is aligned: custom main `10025` cycles and timeline `38644` versus matched native kernel `11546` and timeline `48831`. LPBQ is also aligned through the explicit `LPBQ_ONLY=1` profile: `/tmp/qcom_htp_w4a8_lpbq_hmx_regress_after_scalar_undef_chain8` validates byte-exact against the matched native oracle with custom main `9629` cycles. |
| w8a16 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w8a16` / `HmxU16I8ToU16MatMul`; converter+ctxgen `chain_qdq` flow passes and embeds native `hmx_v75_convhbh1x1deep_stride1`. The current 256^3 chain8 artifact uses the native tiled custom-op surface: custom and native activation/output are `UFixed16 [1,8,32,256]`, output is byte-identical to matched QNN native (`65536/65536`, same SHA256), and chain=8 is preserved. Performance is aligned: custom main is `30871` cycles and timeline `80217`, versus native `q::ConvLayer_s1.opt=30182` and timeline `79095`. Default builds now run the real W8A16 kernel; `HMX_W8A16_SKIP_KERNEL` is only an explicit diagnostic override. |
| w4a16 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w4a16` / `HmxU16I4ToU16MatMul`; the current 256^3 chain8 native-contract path is bit-exact against QNN native (`65536/65536`) with `HMX_W4A16_NATIVE_COMPACT_SOURCE_TABLES` and `native_kblock32_nmajor_k4_lohi` packing. Bottom mapping shows all eight custom and native kernel-entry activation tensors are `UFixed16 [1,8,32,256]`. Chain8 optrace reports custom main `31419` cycles versus retained native `q::ConvLayer_s1.opt` aggregate `29815`, with custom timeline `77854` versus retained native timeline `253245`. Residual boundary reporting differences are weight carrier `UFixed8` vs native `SFixed8` and control `[1,1,1,1]` vs native `[1]`. LPBQ is aligned through the explicit `LPBQ_ONLY=1` profile: current live repeats validate native-exact `65536/65536` with `VERIFY_NATIVE_TRANSPOSE=1`, use `6400` HMX packets versus native `6464`, and report custom main `30820`/`30991` cycles after one cold-side `33744` run versus fresh native W4A16 Conv repeats `32563`/`33196`. |
| w16a16 | Independent package/op/flow exists as `example/qnn_hmx_matmul_w16a16` / `HmxU16I16ToU16MatMul`; the canonical 256^3 native-record path is accepted through `W16A16_KERNEL_PROFILE=accepted`. It embeds the byte-verified native `hmx_v73_convhhh1x1_stride1` body, auto-generates native-equivalent prepared weight and bias/control sidecars from native ONNX `W`, matches QNN native raw output exactly (`65536/65536`), and is native-class on packets/cycles. Default builds still stay skip-guarded because this accepted contract is scoped to 256^3 and uses a documented custom/native graph-boundary difference. |
| Batch/rank/transpose semantics | Not the first implementation goal. Add only as generator/precompute normalization around the quantized Conv1d-family kernels. |

QHPI quantization boundary discovered while implementing the separate families:
custom-op tensors cannot be represented to ctxgen as per-channel QNN quant
tensors.  This matches the current `u8i8` architecture: QNN sees direct carrier
tensors, while per-channel/per-group weight scales, folded bias, and native
packing are owned by the family-specific prepared payload.

Plan:

1. Turn the five canonical 256^3 artifacts into cheap regression gates,
   including the scoped W16A16 accepted profile.
2. Extend shape coverage family by family. Start with static no-transpose
   MatMul/FC-as-Conv1d where M/K/N are multiples of the validated HMX tiling
   constraints, while preserving QHPI precompute and no hot-path preparation.
3. Parameterize W16A16 native-record fields and sidecar generation beyond the
   canonical 256^3 case before calling that family generally aligned.
4. Add LPBQ/per-group only after the matching per-channel family is correct.
   Treat blockwise expansion as an ABI extension with explicit block scale
   tables, not as a generic per-channel tweak.
5. Keep generic MatMul features out of the critical path. Rank-5, batch
   broadcasting, and transpose flags should be implemented by generator/layout
   normalization and repeated prepared Conv1d jobs, not by making the hot kernel
   parse high-level MatMul semantics.
6. Verification gate per family/shape: converter+ctxgen pass,
   bottom-mapping/optrace confirmation, bit-exact or tolerance-based output
   check against QNN native raw, native aggregate and kernel-only perf
   comparison, and a trace check proving no packing/shape recovery work happens
   inside the profiled custom op.

Near-term milestone: add regression scripts for the current canonical artifacts,
then expand W16A16 and W8/W4 A16 shape coverage under the same native-oracle
rule.

Broader quantized-kernel roadmap status: not complete.  Do not conflate a
single 256^3 precision milestone, such as the current `w8a16` chain8
artifact, with the full-family acceptance rule:

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
`example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256/` and the
earlier `output_codex_native_w4a16_conv1x1_*` directories as historical
comparators only.  Those runs used float-sized runtime output and/or did not
record the required NONTRIVIAL layout flags.  Regenerate W4A16 native Conv
references with `example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh`
before using QNN native output or performance as the current oracle.
After the 2026-05-09 cleanup, `output_codex_*`, `output_w4a16_import_*`, and
other probe directories named below are historical labels only.  The live
artifact set for aligned families is the paired custom/native directory set
under `example/qnn_matmul_profile/`: `output_u8i8_{aligned,native_ref}_e2e_256`,
`output_w4a8_{aligned,native_ref}_e2e_256`,
`output_w8a16_{aligned,native_ref}_e2e_256`,
`output_w4a16_{aligned,native_ref}_e2e_256`, and
`output_w16a16_accepted_256` paired with
`output_w16a16_native_ref_e2e_256`.
The quantized custom runners now follow the u8i8 standard gate too: generators
emit `runtime_input_list.txt`, `quant_overrides.json`, and `native_io.json`.
Converter consumes those encodings and quantizer runs in fallback/save mode,
without `--input_list` or custom `--op_package_lib`, so QAIRT does not execute
custom ops through the CPU backend during quantization.  Runner artifact checks
require native I/O, NONTRIVIAL layout flags, and non-float runtime storage.
For native references, same-shape random tensors are no longer accepted as a
comparison oracle.  The native graph must reuse the custom artifact's exact
runtime input, logical weights, folded/effective bias, and chain topology.
`example/qnn_matmul_profile/run_matched_native_a8_ref.sh` is the current A8
implementation of this rule.

Current standardized W4A16 native oracle:
`example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/`.  It uses
native u16 runtime input/output, `--retrieve_context`, converter NONTRIVIAL
layout flags on A/Y, and the full `optrace/` artifact set.  New W4 native DLC
exports should use `example/qnn_matmul_profile/export_native_w4a16_quantized_dlc.sh`,
which runs QAIRT quantizer with W4 weights, A16 activations, and
`--pack_4_bit_weights` so DLC inspection reports `W: sFxp_4`.  The executable
native Conv oracle uses the encoding-driven converter plus
`qairt-quantizer --enable_float_fallback`, avoiding CPU quantization simulation
while keeping the `sFxp_8` W4 carrier that HTP ctxgen accepts.  The retained
native `q::ConvLayer_s1.opt` aggregate is `29815` cycles across eight Conv
kernels, the `conv1x1_*` QNN-op aggregate is `70408` cycles, and the graph
timeline span is `253245` cycles.  Fresh native reruns on 2026-05-10 produced
`32563` and `33196` cycles with stable `808` packets per Conv.  The retained
`29815` native sample is a low-side profile record: its final Conv events have
unusually low cpp (`3.11757`, `2.52104`, `2.69431`) compared with the repeat
native cpp band around `4.32` to `4.65`.  The current custom W4A16
native-contract path is aligned for 256^3 chain8 when built with
`HMX_W4A16_NATIVE_COMPACT_SOURCE_TABLES` and the real HMX body:
`W4_PACK_ORDER=native_kblock32_nmajor_k4_lohi`, native A16 bias/control,
`MODE=chain_qdq`, and `CHAIN=8` produce `65536/65536` exact against this
oracle.  The latest standard artifact is
`example/qnn_matmul_profile/output_w4a16_aligned_e2e_256/`:
custom main op `31419` cycles and timeline `77854` cycles.  All eight custom
and native HTP kernel-entry activation tensors are `UFixed16 [1,8,32,256]`.
Remaining graph-boundary differences are
weight carrier
`UFixed8` versus native `SFixed8` and custom control `Int32 [1,1,1,1]` versus
native `Int32 [1]`; generated payload bytes and output semantics now match.

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
| Custom w4a16 256^3 chain8 native-contract probe | Current flow is `HMX_W4A16_NATIVE_COMPACT_SOURCE_TABLES` plus `native_kblock32_nmajor_k4_lohi` W4 packing, `native_a16` bias/control, `MODE=chain_qdq`, `CHAIN=8`, and the real HMX body. It is bit-exact against `output_w4a16_native_ref_e2e_256/device_out/Y.raw` (`65536/65536`, maxdiff `0`). Custom main is `31419` cycles and timeline is `77854`; retained native chain8 reports `q::ConvLayer_s1.opt=29815`, `conv1x1_*` aggregate `70408`, and timeline `253245`. Both custom and native have eight kernel nodes with activation `UFixed16 [1,8,32,256]`. | Canonical 256^3 shape/chain gate is closed. Residual non-blocking boundary differences are custom weight carrier `UFixed8` vs native `SFixed8`, and custom control `[1,1,1,1]` vs native `[1]`. |
| Custom W4A16 LPBQ 256^3 chain8 fast profile | Build with `LPBQ_ONLY=1`; run with `W4_ENCODING=lpbq`, `MODE=chain_qdq`, `CHAIN=8`, `OP_INPUT_LAYOUT=tiled`, `VERIFY_NATIVE_TRANSPOSE=1`. The latest validation artifact `/tmp/qcom_htp_w4a16_lpbq_current_rerun3_20260510` passes `scripts/validate_qnn_kernel_e2e.py` against `output_w4a16_native_ref_e2e_256` (`65536` elements, sha256 `e752e44462209c7b0074512bacf622459e0748e614945c91d0bc29d576dbe790`). Custom LPBQ uses `6400` packets across eight HMX events and reports `30820` cycles; adjacent repeats are `30991` and a cold-side `33744`. Fresh native W4A16 Conv repeats are `32563` and `33196` cycles with `6464` packets. | LPBQ correctness and native-class kernel performance are closed for the canonical 256^3 chain8 gate. Keep scalar LPBQ as correctness/debug fallback only. |
| Native w8a8 / u8i8 256^3 matched reference | Current artifact is `output_u8i8_native_ref_e2e_256/`, generated by `run_matched_native_a8_ref.sh` from `output_u8i8_aligned_e2e_256/` with the same input, logical W, effective bias, and chain8 topology. It is bit-exact against custom output (`65536/65536`, maxdiff `0`). Optrace reports native `q::ConvLayer_s1.opt=12435`, MatMul aggregate `36922`, and timeline `53946`; custom reports main `10891` and timeline `36342`. |
| Custom w4a8 256^3 native-compact precompute, native `convbnb` body, packed `[1,1,K,N/2]`, twos-complement lo/hi nibbles, W4 mask helper | Current custom artifact is `output_w4a8_aligned_e2e_256/`, generated by the default W4A8 runner with native activation/output surface `UFixed8 [1,8,32,256]` and the native 32-entry physical compact source tables.  It is byte-identical to matched native `output_w4a8_native_ref_e2e_256/device_out/Y.raw` (`65536/65536`, maxdiff `0`, same SHA256).  Custom main is `10025` cycles and timeline is `38644`; matched native reports `q::ConvLayer_s1.opt=11546`, MatMul aggregate `29765`, timeline `48831`. | Canonical 256^3 shape/chain gate is closed.  Harmless boundary difference: custom weight carrier is still reported as `UFixed8 [1,1,256,128]` while native reports packed W4 as `SFixed8 [1,1,256,128]`. |
| Custom W16A16 256^3 accepted native-record path | Current custom artifact is `output_w16a16_accepted_256/`, run with `W16A16_KERNEL_PROFILE=accepted` and native oracle `output_w16a16_native_ref_e2e_256/`. It auto-generates prepared weight and bias/control sidecars from native ONNX `W`, uses the byte-verified `hmx_v73_convhhh1x1_stride1` body, and is byte-identical to native raw output (`65536/65536`, maxdiff `0`). Custom main is `71283` cycles and `17594` packets; native reports `q::ConvLayer_s1.opt=75433` cycles and `8836+8836` packets. `alignment_gate.accepted=True` with boundary policy `single_custom_op_internal_split_n128`. |
| Custom w8a16 256^3 chain8 native-tiled graph | Ctxgen passes and device execution reaches the HMX body. Output is byte-identical to matched native `output_w8a16_native_ref_e2e_256/device_out/Y.raw` (`65536/65536`, maxdiff `0`, same SHA256 `90444514529ccfb6a6fc7bad872b29479ec692a611401d826ea17be5b077a089`). The analytic native-contract reference is only diagnostic (`52003/65536`, maxdiff `65535`). Custom and native both have eight kernel nodes with activation/output `UFixed16 [1,8,32,256]`. Performance is aligned: custom main is `30871`, timeline `80217`; matched native reports `q::ConvLayer_s1.opt=30182`, timeline `79095`. |

Latest W4A8 continuation notes: patched-skel probes now confirm the matched
native W4A8 path reaches `hmx_v73_convbnb1x1_stride1` at `0x2f0780`.  The
public-output mapping for the first 512 internal probe words is
`public=(i//64)*2048+((i%64)//8)*64+(i%8)`, available through
`scripts/parse_w4a16_native_entry_probe.py --layout a8crouton512 --dtype u32`.
The parsed native BNB entry sees `out_desc=[table,8,32,32,8,256]`,
`act_desc=[table,8,...]`, mask
`[0,0x700,0,0x71f,0,0,0x7ff,0,0,0,0,0,0x20,0,control_ptr,0]`, and compact
32-entry output/activation tables with contiguous `0x800` spacing.

Historical W4A8 native-layout probes showed that copying only the native tile
selector is a performance-positive but semantic false path: it can lower kernel
cycles while leaving only about half the rows correct.  Those probe artifacts
were deleted after the conclusion was folded into the default path.  The retained
current result is the native surface plus 32 physical compact source tables,
which closes both correctness and performance in
`output_w4a8_aligned_e2e_256/`.

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

Native entry instrumentation is now more precise.  A direct `r31` probe at
`0x2fcd80` reports return address `0x03de46c`, identifying the active clean
native call as `0x3de464` in the simple `0x3de3c0` prebuilt-record wrapper.  A
corrected pure-assembly pattern probe recovered the first-512-word public-output
map used by
`scripts/parse_w4a16_native_entry_probe.py --layout crouton512 --record-kind auto`.
The reliable v3 native mask is
`[0,0x700,0,0x77c,0,0,0x3ff,0,0,0,0,0,0xa0,0,control_ptr,0]`, with native
control `[1,0x401,0x20c,0]`.  Matching only custom `HMX_W4A16_MASK_ARG6=0xa0`
does not change the imported-sidecar result (`3298/65536`, `sorted_equal=True`,
best row32 roll `32:65536`), so the remaining issue is still the full prebuilt
record/table/control contract.

The base-record probe now decodes that visible prebuilt record:
`act_desc=[table,8,64,32,8,256]`, `out_desc=[table,8,64,32,8,256]`,
`weight=0x046c0000`, `bias=0x046c8000`, and `control=0xfdd01c00`.
Applying those scalar fields to the imported-sidecar custom flow
(`output_w4a16_import_native_sidecar_bd00_base_record_fields_256/`) fails graph
execution before a valid optrace/output, while the graph boundary still differs
on `UFixed8` vs `SFixed8` weight carrier and control tensor shape.  Continue at
the native table memory / wrapper-loop contract, not another scalar descriptor
copy.

The follow-up HMXT table probe keeps the same pure-assembly patched-skel
approach and records the memory behind the active native table pointers.  The
first 64 output entries are contiguous `0x046a0000..0x046bf800`, the first 64
activation entries are contiguous `0x046c9000..0x046e8800`, and entries after
64 are adjacent wrapper/metadata words rather than a 512-entry row4-expanded
table.  This closes the direct scalar-copy route: the custom public-QHPI table
adapter is structurally different from the compact native table view.

Pure-assembly activation/output layout probes now decode the first compact-table
coordinate mappings.  Native compact `act_table[i]` starts at logical activation
`m=(i//8)*4`, `k=(i%8)*32`, with u16 pairs for rows `m/m+1` and K increasing;
the first block continues with row pairs `m+2/m+3`, then `m+32/m+33`, etc.
The full `act_table[0]` dump has zero formula misses for
`j=group*32+k -> rows 32*(group//2)+2*(group&1)` and `+1`.
Native compact `out_table[i]` is ordered differently in exported `Y.raw`:
viewed as `[8,32,256]`, marker `i` lands at `m32_group=i%8`, `row=0`,
`n=(i//8)*4`.  This is a real output-table order mismatch versus the current
custom 512-entry `row4 * stride + tile` table construction.
The full `out_table[0]` block marker has zero missing markers and confirms
block offset `j=group*32+row` maps to `row=j%32` and
`n=32*(group//2)+2*(group&1)` / `n+1`.

The HMXR record-window probe now anchors the compact tables around the active
prebuilt record: `act_table_ptr = base-0x180`, `out_table_ptr = base+0x98`,
pre-base metadata sits at `base-0x80..base-0x04`, and post-output metadata plus
an adjacent restore/public-table-looking pointer table starts after the compact
output table.  The adjacent table is not the HNH `out_desc+0` compact table;
paired-marker probing through that table also produces no paired marker hits in
exported `Y.raw`, so it is not a direct public-export table at this point.

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

Latest w8a16 status after artifact cleanup:

- The current acceptance artifact is
  `example/qnn_matmul_profile/output_w8a16_aligned_e2e_256/`, generated with
  `MODE=chain_qdq OP_INPUT_LAYOUT=tiled CHAIN=8` and matched native generation.
  It is bit-exact against
  `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/device_out/Y.raw`
  and custom/native both enter HTP as `UFixed16 [1,8,32,256]`.
- Performance is aligned: custom main is `30871` cycles, native
  `q::ConvLayer_s1.opt` is `30182`, custom timeline is `80217`, and native
  timeline is `79095`.  Packet counts are native-class: `682` custom versus
  `695` native per kernel event.
- The promoted contract is the tiled physical Crouton pointer-table path with
  `n_tiles_pow2=M_t*4`, native A16 quantization, native A16 bias records, the
  three-word control tensor `[1, 1025, 524]`, and full-W8 K-major tile packing.
- Obsolete native-rank, wrapper-bundle, descriptor-dump, table-window, and
  intermediate `M_t` probe artifact directories were deleted after their
  conclusions were folded into the handoff.  Regenerate such probes only with a
  new hypothesis and a fresh artifact name.
- The only retained W8A16 runtime artifacts are the current custom aligned
  artifact and the matched native oracle.  Older shape/optrace numbers should
  not be used as current evidence.

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

`scripts/perf_hmx_u8i8_matmul.py` is now a reader for the standard artifact
set: it reuses or creates `<out_dir>/optrace/chrometrace.json`, does not write
ad-hoc `/tmp/_optrace*` files, and treats native raw output as the default
bit-exact/probe surface while retaining legacy float-output decoding.

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
