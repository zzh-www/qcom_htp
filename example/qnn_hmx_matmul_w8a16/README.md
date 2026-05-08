# qnn_hmx_matmul_w8a16

Independent QNN custom-op package for `HmxU16I8ToU16MatMul`, the `w8a16`
MatMul/FullyConnected-as-Conv1d family.

Current status:

- Package/provider/op XML, converter hook, build scripts, and custom graph flow
  are separate from `u8i8`.
- QHPI ABI uses per-tensor `u16` activation/output and a direct `u8` carrier for
  logical signed W8 weights; per-channel scale/bias data belongs in the native
  prepared payload, matching the existing `u8i8` pattern.
- Default device builds still define `HMX_W8A16_SKIP_KERNEL`; real-HMX probes
  require explicit `EXTRA_DEFS` until broader shape coverage is validated.
- Native-contract diagnostics now use `--a16-quant-contract native`,
  `MODE=chain_qdq`, `--w8-pack-order kmajor`, and `--bias-layout native_a16`.
  Direct `MODE=chain` `UINT16` input ignores the activation quant override;
  `chain_qdq` preserves the native `ForceFormat_Crouton` metadata.
- The current canonical 256^3 chain8 artifact restores the native-rank custom
  op surface: custom activation/output are `UFixed16 [1,1,256,256]`, output is
  byte-identical to `output_w8a16_native_ref_e2e_256/device_out/Y.raw`, and the
  graph keeps `CHAIN=8`.
- Performance is not aligned.  Optrace reports custom main `184539` cycles and
  timeline `285461`; matched native reports `q::ConvLayer_s1.opt=30182`, MatMul
  aggregate `35747`, and timeline `79095`.  Native still enters HTP as
  `UFixed16 [1,8,32,256]`; the strict tiled shape probe matched that shape but
  regressed to `507368` custom main cycles, so it is diagnostic only.
- Native split remains a graph-execution blocker independent of the callback:
  split-concat and split-separate graphs still fail even with
  `HMX_W8A16_EARLY_RETURN`.
- The standard conversion path is encoding-driven
  `qairt-converter -> qairt-quantizer`: converter applies generated encodings,
  then quantizer runs without calibration input or a custom op package so CPU
  backend never executes the custom op during quantization.

Build:

```bash
bash example/qnn_hmx_matmul_w8a16/build.sh
bash example/qnn_hmx_matmul_w8a16/build_x86.sh
```

Smoke flow:

```bash
cd example/qnn_hmx_matmul_w8a16/standard_flow/custom_w8a16
SKIP_DEVICE=1 M=32 K=32 N=32 CHAIN=1 bash run_w8a16_chain.sh
```

Native-rank diagnostic example:

```bash
cd example/qnn_hmx_matmul_w8a16/standard_flow/custom_w8a16
EXTRA_DEFS="-UHMX_W8A16_SKIP_KERNEL -DHMX_W8A16_ALLOW_UNVALIDATED_KERNEL" \
bash ../../build.sh
EXTRA_DEFS="-UHMX_W8A16_SKIP_KERNEL -DHMX_W8A16_ALLOW_UNVALIDATED_KERNEL" \
bash ../../build_x86.sh
VERIFY_NATIVE_RAW="$PWD/../../../qnn_matmul_profile/output_w8a16_native_ref_e2e_256/device_out/Y.raw" \
M=256 K=256 N=256 CHAIN=8 MODE=chain_qdq OP_INPUT_LAYOUT=native \
bash run_w8a16_chain.sh
```

Verification rule: use real QNN native output as the primary oracle.  The
analytic native-contract reference only helps diagnose quantization/rounding
drift and is not the final acceptance target.

Performance artifact rule: use the shared QNN optrace decoder for custom and
native runs:

```bash
scripts/decode_qnn_optrace.py <out_dir>
```

Device runs keep raw profiling at `device_out/qnn-profiling-data_0.log` and
write the standard decoded set under `optrace/`: `chrometrace.json`,
`chrometrace_htp.json`, `chrometrace_runtrace.json`,
`chrometrace_qnn_htp_analysis_summary.json/html`, `profile.txt`,
`summary.json`, and `manifest.json`.  The custom chain scripts run this
automatically by default; set `DECODE_OPTRACE=0` only for smoke/debug runs
where performance artifacts are intentionally skipped.
