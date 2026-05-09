# QNN Native Artifact And Performance Standard

This is the baseline flow for QNN-native references and custom-op performance
comparisons.  It replaces ad-hoc live-DLC runs and temporary `/tmp/_optrace*`
files as evidence.

## Required Flow

The u8i8 custom and native-baseline runners are the template for the standard:
generate ONNX plus raw runtime inputs, convert to DLC with layout flags, export
a context binary, run qnn-net-run from that context binary with native I/O, then
decode optrace into the same run directory.

1. Generate both input surfaces:
   - `input_A.raw` plus `input_list.txt` for converter/calibration or legacy
     non-native runs.
   - `runtime_inputs_native/*.raw` plus `runtime_input_list.txt` for native
     references, or the family runner's quantized runtime directory such as
     `runtime_inputs_u8/*.raw` for custom OPs, for device runs with
     `--use_native_input_files`.
   - Keep `native_io.json` beside them so the raw storage and encoding are
     explicit.
2. Convert with layout-preservation flags on every public graph input and
   output.  Quantized runs use the QAIRT two-stage flow by default.  Calibration
   flows use `qairt-converter` to emit a float DLC and `qairt-quantizer
   --input_list` to calibrate it.  Generated baselines and custom HMX runners
   should prefer the encoding-driven variant when complete encodings are known:
   the generator emits `quant_overrides.json`, `qairt-converter
   --quantization_overrides` writes those encodings into the DLC, then
   `qairt-quantizer --enable_float_fallback` saves the final DLC without
   `--input_list` or `--op_package_lib`.  This avoids CPU-backend quantization
   simulation of custom ops.
   - `--source_model_input_layout <name> NONTRIVIAL`
   - `--desired_input_layout <name> NONTRIVIAL`
   - `--source_model_output_layout <name> NONTRIVIAL`
   - `--desired_output_layout <name> NONTRIVIAL`
   - For W4 DLC-export quantizer invocations, pass `--pack_4_bit_weights` so
     the inspection artifact reports `sFxp_4`.  Executable encoded-oracle runs
     may still carry W4 values through QNN's `sFxp_8` Conv carrier when HTP
     ctxgen rejects packed Conv W tensors; record that explicitly in the run
     directory instead of mixing it with the packed DLC export.
3. Export a context binary before device execution:
   - `qnn-context-binary-generator --profiling_level detailed`
   - `--profiling_option optrace`
   - `--save_backend_op_mapping`
   - `--binary_file <name> --output_dir ctx`
4. Run on device from the context binary, not from `--dlc_path`:
   - `qnn-net-run --retrieve_context ctx.bin`
   - `--profiling_level detailed --profiling_option optrace`
   - `--use_native_input_files --use_native_output_files`
   - `--perf_profile burst`
5. Pull and decode into the run directory:
   - `device_out/qnn-profiling-data_0.log`
   - native output raw, for example `device_out/Y.raw`
   - `scripts/decode_qnn_optrace.py <out_dir>`

## Artifact Contract

Each run directory should contain:

| Path | Meaning |
|---|---|
| `matmul.dlc` or family-specific `.dlc` | Converted model used only as ctxgen input. |
| `ctx/*.bin` | Context binary used by device execution. |
| `ctx/*bottom_mapping.json` | Lowered backend-op mapping. |
| `ctx/*schematic.bin` | Schematic passed to the optrace reader. |
| `device_out/qnn-profiling-data_0.log` | Raw device profiling log. |
| `device_out/*.raw` | Native output used for output alignment. |
| `optrace/chrometrace.json` | Timeline used for event inspection. |
| `optrace/profile.txt` | Text profile-reader output. |
| `optrace/summary.json` | Compact decoded event summary. |
| `optrace/manifest.json` | Decode provenance and command record. |

Canonical 256^3 aligned-output naming:

| Directory rule | Meaning |
|---|---|
| `example/qnn_matmul_profile/output_<family>_aligned_e2e_256/` | Final custom-op e2e artifact for an aligned family. |
| `example/qnn_matmul_profile/output_<family>_native_ref_e2e_256/` | Final QNN-native reference artifact when the family needs a native oracle. |

Current live directories are:

| Family | Custom-op artifact | QNN-native artifact |
|---|---|---|
| u8i8 / native w8a8 | `output_u8i8_aligned_e2e_256/` | `output_u8i8_native_ref_e2e_256/` |
| w4a8 | `output_w4a8_aligned_e2e_256/` | `output_w4a8_native_ref_e2e_256/` |
| w8a16 | `output_w8a16_aligned_e2e_256/` | `output_w8a16_native_ref_e2e_256/` |
| w4a16 | `output_w4a16_aligned_e2e_256/` | `output_w4a16_native_ref_e2e_256/` |
| w16a16 | `output_w16a16_accepted_256/` | `output_w16a16_native_ref_e2e_256/` |

Every aligned family must keep both sides.  Native-only directories are allowed
only when the custom kernel is not aligned yet, and must be labeled that way in
handoffs/status notes.  Probe outputs and `/tmp/_optrace*` products are
temporary by definition and should not be cited as current evidence.

Native references for custom alignment must be matched to the custom artifact:
same runtime input bytes, same logical weights, same folded/effective bias when
the custom op has a bias input, and the same chain/topology being compared.
Same-shape random native runs are only exploratory performance evidence and are
not valid precision or final performance oracles.

## Performance Rule

Use QNN native output as the output oracle when comparing custom OP behavior.
Analytic formulas are diagnostic references only.  For performance, cite the
standard `optrace/` products in the run directory and prefer QHAS-derived
end-to-end metrics from `parse_qhas.py`; use per-op timeline breakdowns only
when explaining which HTP kernels dominate the run.

Reject native-reference artifacts that were executed with float runtime I/O,
that ran directly from a DLC instead of a context binary, or that lack the
layout-preservation converter flags above.  Such artifacts can stay as
historical debugging evidence, but they are not current correctness or
performance oracles.

QNN native graphs may still have float ONNX public input/output tensors when
quantization overrides are used.  That is not sufficient for comparison.  The
accepted runtime contract is the generated native raw input/output recorded in
`native_io.json` and exercised with qnn-net-run native I/O flags.

For W4A16 native DLC export, use the explicit QAIRT two-stage flow:

```bash
bash example/qnn_matmul_profile/export_native_w4a16_quantized_dlc.sh \
  --shape 256,256,256
```

That command runs `qairt-converter` to make a float DLC, then
`qairt-quantizer` with A16 activations, W4 weights, 32-bit bias, and
`--pack_4_bit_weights` to make the quantized DLC.  The expected DLC inspection
surface is `A/Y: uFxp_16` and `W: sFxp_4`.

Use the checker at the end of every quantized standard run:

```bash
scripts/check_qnn_artifact_standard.py <out_dir> \
  --require-layout-flags --require-native-io --reject-float-io
```

Custom-op and quantized native-reference runners use this same gate.  `native_io.json`
is not optional: it records the exact raw input/output storage accepted for the
run.  Use `NATIVE_OUTPUT=0` or float runtime inputs only for legacy diagnostics
with `STRICT_ARTIFACT_STANDARD=0`; those artifacts are not current oracles.

## Current Implementations

- `example/qnn_matmul_profile/gen_onnx.py` emits fp32 calibration input and
  native runtime input side by side.
- `example/qnn_matmul_profile/profile_all.sh` converts with NONTRIVIAL layout
  flags, uses `qairt-converter -> qairt-quantizer` for every non-fp16 config,
  exports `ctx/*.bin`, runs device execution with `--retrieve_context`, pulls
  native output, and decodes `optrace/`.
- `example/qnn_matmul_profile/export_native_w4a16_quantized_dlc.sh` is the
  canonical W4A16 native DLC export path when the requested artifact is the DLC
  itself.  It uses QAIRT's standalone quantizer and defaults to
  `--pack_4_bit_weights`.
- `example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh` is the canonical
  W4A16 native Conv reference generator; it replaces the old float-input,
  float-output `output_codex_native_w4a16_same_custom_256` style of artifact.
- `example/qnn_matmul_profile/run_matched_native_a8_ref.sh` is the canonical
  u8i8/w4a8 native reference generator.  It reads the custom output directory's
  `runtime_inputs_u8/act_*.raw`, `*.wRaw_KN.npy`, and
  `*.effective_int32.npy`, then emits a matched QNN-native MatMul+Add chain.
- `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_native_chain.sh`
  and `run_native_sweep.sh` now emit `native_io.json`, pull native output raw,
  and use the same checker/layout/context/optrace rule as the custom u8i8 flow.
- `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_device_optrace.sh`
  is kept as a Phase-A manual runner, but now uses native I/O and decodes into
  the standard local `optrace/` directory.
- `scripts/perf_hmx_u8i8_matmul.py` is a standard-artifact reader: it consumes
  or generates `<out_dir>/optrace/chrometrace.json` and supports native raw
  output before falling back to legacy float output.
- Quantized custom generators now emit `runtime_input_list.txt`,
  `quant_overrides.json`, and `native_io.json`; their runners default to native
  output (`NATIVE_OUTPUT=1`), use the encoding-driven quantizer path, and
  enforce `--require-native-io --reject-float-io`.

Current runner coverage:

| Family | Standard entrypoint | Current standard behavior |
|---|---|---|
| native profile sweep | `example/qnn_matmul_profile/profile_all.sh` | Emits native runtime input, preserves A/Y layout, applies `quant_overrides.json` in `qairt-converter`, runs encoding-driven `qairt-quantizer --enable_float_fallback` without calibration simulation for quantized configs, exports `ctx/*.bin`, runs `--retrieve_context`, pulls native output and `optrace/`. |
| u8i8/w4a8 matched native reference | `example/qnn_matmul_profile/run_matched_native_a8_ref.sh` | Reuses the exact custom A/W/effective-bias/chain, emits a native MatMul+Add chain, runs context-binary device execution with native I/O, decodes `optrace/`, and writes `analysis/matched_native_compare.*`. |
| W8A16 matched native reference | `example/qnn_matmul_profile/run_matched_native_w8a16_ref.sh` | Reuses the exact custom A/W/chain, emits a native MatMul chain, runs context-binary device execution with native I/O, decodes `optrace/`, and writes `analysis/matched_native_compare.*`. |
| W4A16 packed DLC export | `example/qnn_matmul_profile/export_native_w4a16_quantized_dlc.sh` | Exports a QAIRT-quantized W4A16 DLC with `W: sFxp_4` by default. |
| W4A16 native reference | `example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh` | Emits u16 native input/output for the Conv reference and records the full standard artifact set; supports `CHAIN=8` / `--chain 8` for chain-style acceptance; conversion is encoding-driven `qairt-converter -> qairt-quantizer --enable_float_fallback`, avoiding CPU quantization simulation while preserving the executable W4 carrier accepted by ctxgen. |
| u8i8 native baseline | `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_native_chain.sh` | Uses generated `runtime_input_list.txt` and `quant_overrides.json`, preserves graph I/O layout, applies encoding-driven `qairt-converter -> qairt-quantizer`, and runs from context binary with native I/O. |
| u8i8 native sweep | `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_native_sweep.sh` | Same native-I/O/context-binary and encoding-driven quantizer rule for size sweeps. |
| custom u8i8/w4a8/w8a16/w4a16/w16a16 | `example/qnn_hmx_matmul_*/standard_flow/custom_*/run_*_chain.sh` | Preserves public custom-op I/O layout, applies generated encodings in `qairt-converter`, runs `qairt-quantizer --enable_float_fallback` without calibration execution of the custom op, runs from context binary, defaults to native output, pulls profile log, decodes `optrace/`, and runs the standard artifact checker. |

## Current A8 And W16 Native References

The current clean 256^3 A8 native references are:

| Family | Artifact | Native config | Key optrace facts |
|---|---|---|---|
| u8i8 / native w8a8 | `example/qnn_matmul_profile/output_u8i8_native_ref_e2e_256/` | matched native MatMul+Add chain, same A/W/effective bias as custom | exact `65536/65536`, `q::ConvLayer_s1.opt = 12435` cycles, MatMul aggregate `36922`, timeline span `53946` |
| w4a8 | `example/qnn_matmul_profile/output_w4a8_native_ref_e2e_256/` | matched native MatMul+Add chain, same A/W/effective bias as custom, packed W4 DLC | exact `65536/65536`, `q::ConvLayer_s1.opt = 11546` cycles, MatMul aggregate `29765`, timeline span `48831` |
| w16a16 | `example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/` | `w16a16` MatMul | custom accepted raw exact `65536/65536`; custom `71283` cycles / `17594` packets; native `q::ConvLayer_s1.opt = 75433` cycles / `8836+8836` packets, MatMul aggregate `82644`, timeline span `124593` |

The A8 references are matched precision/performance oracles generated by
`run_matched_native_a8_ref.sh`, not random same-shape runs.  W16A16 uses a
separate scoped accepted artifact, `output_w16a16_accepted_256/`, because the
accepted native-record fields and generated sidecars are tied to the canonical
256^3 native oracle.

Current `w4a8` status is native-output and performance aligned for canonical
256^3 chain8.  The refreshed custom artifact is
`output_w4a8_aligned_e2e_256/`; its raw output is byte-identical to
`output_w4a8_native_ref_e2e_256/device_out/Y.raw`, custom main cycles are
`10025`, and custom timeline is `38644`.  The matched native reference reports
`q::ConvLayer_s1.opt = 11546`, MatMul aggregate `29765`, and timeline `48831`.

Current W4A8 follow-up uses the blackbox native-alignment loop from
`Agent/guides/qnn_native_alignment_blackbox_handbook.md`: artifact-standard
runs first, then lowered bottom mapping, then one-hypothesis probes.  The
validated negative probes show that wrapper overhead and descriptor-build cost
are not the dominant gap.  A native-layout custom surface can match native
activation/output shape as `UFixed8 [1,8,32,256]`, but it does not materially
move kernel cycles.  Reducing the descriptor M count to `mt_groups` moves cycles
toward native class while breaking about half the output, including when paired
with the native-layout surface.  Continue at native prepared sidecar bytes,
compact table memory, and wrapper/prebuilt-record state rather than more scalar
descriptor sweeps.

The W4A8 native-entry probe now has a concrete BNB oracle.  Patching
`0x2f0780` confirms the clean native path enters `hmx_v73_convbnb1x1_stride1`;
the recovered W4A8 probe-output map is
`public=(i//64)*2048+((i%64)//8)*64+(i%8)`.  Parsed with
`scripts/parse_w4a16_native_entry_probe.py --layout a8crouton512 --dtype u32`,
native BNB uses `out_desc=[table,8,32,32,8,256]`,
`act_desc=[table,8,...]`, and 32-entry compact physical act/out tables.  A
custom `desc_m_t=8` probe reached native-class per-node kernel cycles but only
covered about half the output; the record-window probe identified the missing
state as the physical compact table contract.

## Current W4A16 Native Reference

The current clean 256^3 W4A16 native reference is:

`example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/`

Key facts:

| Artifact | Value |
|---|---|
| native input | `runtime_inputs_native/A.raw`, u16, 131072 bytes |
| native output oracle | `device_out/Y.raw`, u16, 131072 bytes |
| chain | 8 native Conv nodes |
| context binary | `ctx/conv_ctx.bin`, 90112 bytes, SHA-256 `4cfbe7f2d99d754af2e59acc26181e089467990e0752704e0966ff800d09c69f` |
| DLC W storage policy | executable encoded oracle uses QNN's `sFxp_8` W4 carrier with bitwidth-4 encoding; packed `sFxp_4` is produced by the DLC-only export |
| kernel event | `q::ConvLayer_s1.opt = 29815` cycles aggregate across chain8 |
| native Conv group | `conv1x1_* = 70408` cycles aggregate |
| HTP timeline span | `253245` cycles |

Do not accept an unpacked DLC as the default W4 export artifact.  The standard
W4 DLC-export surface is `sFxp_4`, produced by `--pack_4_bit_weights`.  For
executable native Conv references, the accepted oracle is the context-binary run
with native raw I/O; on this SDK/HTP path that runnable DLC keeps the `sFxp_8`
carrier plus W4 encoding metadata.

Its final HNH boundary is:

```text
activation  UFixed16 [1,8,32,256]
weight      SFixed8  [1,1,128,256]
bias        Int32    [1,8,1,128]
control     Int32    [1]
extra ctrl  Int32    [1,1,1,3]
output      UFixed16 [1,8,32,256]
```

Current W4A16 256^3 canonical acceptance is closed for the shape/chain gate.
The paired custom artifact is
`example/qnn_matmul_profile/output_w4a16_aligned_e2e_256/`; it is
`65536/65536` exact against this chain8 native oracle.  Bottom mapping shows all
eight custom and native HTP kernel-entry activation tensors are
`UFixed16 [1,8,32,256]`.  Custom main cycles are `31419` versus native
`q::ConvLayer_s1.opt` aggregate `29815`; custom timeline is `77854` versus
native timeline `253245`.  Residual boundary reporting differences are custom
weight carrier `UFixed8` vs native `SFixed8`, and custom control `[1,1,1,1]`
vs native `[1]`.
