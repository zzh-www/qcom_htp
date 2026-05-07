# QNN Native Artifact And Performance Standard

This is the baseline flow for QNN-native references and custom-op performance
comparisons.  It replaces ad-hoc live-DLC runs and temporary `/tmp/_optrace*`
files as evidence.

## Required Flow

1. Generate both input surfaces:
   - `input_A.raw` plus `input_list.txt` for converter/calibration or legacy
     non-native runs.
   - `runtime_inputs_native/*.raw` plus `runtime_input_list.txt` for device
     runs with `--use_native_input_files`.
   - Keep `native_io.json` beside them so the raw storage and encoding are
     explicit.
2. Convert with layout-preservation flags on every public graph input and
   output:
   - `--source_model_input_layout <name> NONTRIVIAL`
   - `--desired_input_layout <name> NONTRIVIAL`
   - `--source_model_output_layout <name> NONTRIVIAL`
   - `--desired_output_layout <name> NONTRIVIAL`
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

## Current Implementations

- `example/qnn_matmul_profile/gen_onnx.py` emits fp32 calibration input and
  native runtime input side by side.
- `example/qnn_matmul_profile/profile_all.sh` converts with NONTRIVIAL layout
  flags, exports `ctx/*.bin`, runs device execution with `--retrieve_context`,
  pulls native output, and decodes `optrace/`.
- `example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh` is the canonical
  W4A16 native Conv reference generator; it replaces the old float-input,
  float-output `output_codex_native_w4a16_same_custom_256` style of artifact.
- `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_native_chain.sh`
  and `run_native_sweep.sh` now use the same converter layout rule as the
  custom u8i8 flow.
- `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_device_optrace.sh`
  is kept as a Phase-A manual runner, but now uses native I/O and decodes into
  the standard local `optrace/` directory.
- `scripts/perf_hmx_u8i8_matmul.py` is a standard-artifact reader: it consumes
  or generates `<out_dir>/optrace/chrometrace.json` and supports native raw
  output before falling back to legacy float output.
- Quantized custom runners default to native output (`NATIVE_OUTPUT=1`) while
  retaining `NATIVE_OUTPUT=0` for legacy float-output checks.

Current runner coverage:

| Family | Standard entrypoint | Current standard behavior |
|---|---|---|
| native profile sweep | `example/qnn_matmul_profile/profile_all.sh` | Emits native runtime input, preserves A/Y layout, exports `ctx/*.bin`, runs `--retrieve_context`, pulls native output and `optrace/`. |
| W4A16 native reference | `example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh` | Emits u16 native input/output for the Conv reference and records the full standard artifact set. |
| u8i8 native baseline | `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_native_chain.sh` | Uses generated `runtime_input_list.txt`, preserves graph I/O layout, runs from context binary with native I/O. |
| u8i8 native sweep | `example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_native_sweep.sh` | Same native-I/O/context-binary rule for size sweeps. |
| custom u8i8/w4a8/w8a16/w4a16/w16a16 | `example/qnn_hmx_matmul_*/standard_flow/custom_*/run_*_chain.sh` | Preserves public custom-op I/O layout, runs from context binary, defaults to native output, pulls profile log, and decodes `optrace/`. |

## Current W4A16 Native Reference

The current clean 256^3 W4A16 native reference is:

`example/qnn_matmul_profile/output_native_w4a16_conv_ref_256/`

Key facts:

| Artifact | Value |
|---|---|
| native input | `runtime_inputs_native/A.raw`, u16, 131072 bytes |
| native output oracle | `device_out/Y.raw`, u16, 131072 bytes |
| context binary | `ctx/conv_ctx.bin`, 90112 bytes, SHA-256 `b48db57c34c02741ded507eda349a4ca7e094c92302d28e573eddbaeef177e91` |
| kernel event | `q::ConvLayer_s1.opt = 7893` cycles |
| native Conv group | `conv1x1 = 37287` cycles |
| HTP timeline span | `313032` cycles |

Its final HNH boundary is:

```text
activation  UFixed16 [1,8,32,256]
weight      SFixed8  [1,1,128,256]
bias        Int32    [1,8,1,128]
control     Int32    [1]
extra ctrl  Int32    [1,1,1,3]
output      UFixed16 [1,8,32,256]
```
