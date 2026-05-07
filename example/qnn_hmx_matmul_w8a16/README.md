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
  `chain_qdq` preserves the native `ForceFormat_Crouton` metadata.  A single
  native-rank custom output executes when the internal `[1,1,M,N]` tensor is
  reshaped to a final 3D graph output with `--final-output-rank 3d`.
- The 256^3 native-contract single-op path is aligned with native QNN when the
  real kernel gate is opened: custom output is byte-identical to
  `output_codex_native_w8a16_custom_full_256/device_out/out.raw`, and the
  analytic native-contract reference is only a diagnostic sanity check.
  The fast descriptor uses mask `arg1=0x70b`, `n_tiles_pow2=row4_groups*4`
  (`256` for 256^3), and `m_total_minus_step=8`.
- Optrace for the default real-kernel 256^3 path reports
  `HmxU16I8ToU16MatMul dur=29842, pkts=4938, cpp=6.04`; the comparable native
  split `ConvLayer_s1.opt` kernels total about `30839` cycles and `5086`
  packets.
- Native split remains a graph-execution blocker independent of the callback:
  split-concat and split-separate graphs still fail even with
  `HMX_W8A16_EARLY_RETURN`.

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
VERIFY_NATIVE_RAW="$PWD/../../../qnn_matmul_profile/output_codex_native_w8a16_custom_full_256/device_out/out.raw" \
M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq \
GEN_EXTRA_ARGS="--op-input-layout native --final-output-rank 3d --a16-quant-contract native --w8-pack-order kmajor --bias-layout native_a16 --reference-contract native" \
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
