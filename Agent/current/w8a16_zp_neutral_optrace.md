# W8A16 ZP Neutral Optrace

Date: 2026-05-23

Scope: historical `w8a16_per_channel_native_match` `zp_neutral` blocker at
`M=256 K=256 N=256 CHAIN=1`, now closed by the generated custom sidecar rule.

## Artifacts

- Custom failing trace:
  `/tmp/qcom_htp_w8a16_per_channel_native_match_ci/custom_zp_neutral/optrace/`
- Native failing-case trace decoded from the existing native profile log:
  `/tmp/qcom_htp_w8a16_per_channel_native_match_ci/native_zp_neutral/optrace/`
- Native retained chain8 reference trace:
  `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/optrace/`

Decode command used for the native `zp_neutral` trace:

```bash
source scripts/env.sh
uv run python scripts/decode_qnn_optrace.py \
  /tmp/qcom_htp_w8a16_per_channel_native_match_ci/native_zp_neutral \
  --profile-log /tmp/qcom_htp_w8a16_per_channel_native_match_ci/native_zp_neutral/device_out/qnn-profiling-data_0.log \
  --schematic /tmp/qcom_htp_w8a16_per_channel_native_match_ci/native_zp_neutral/case_encoded_schematic.bin
```

## Graph Difference

The custom failing graph is not structurally equivalent to the QNN Native HTP
graph even though the final A16 sidecar bytes are imported from native.

Custom `zp_neutral` HTP graph:

```text
nodes: 6
types:
  1 QnnHmxMatMulW8A16Package::HmxU16I8ToU16MatMul
  1 q::*InputSlice
  1 q::ForceFormat_Crouton
  3 q::ConvLayer.opt.weights_to_vtcm

compute node:
  QnnHmxMatMulW8A16Package::HmxU16I8ToU16MatMul
  in0 [1,8,1,128]   dtype=50    from weights_to_vtcm
  in1 [1,1,256,256] dtype=1032   from weights_to_vtcm
  in2 [1,8,32,256]  dtype=1046   from ForceFormat_Crouton
  in3 [1,1,1,2048]  dtype=1032   from weights_to_vtcm
  out [1,8,32,256]  dtype=1046
```

Native `zp_neutral` HTP graph:

```text
nodes: 82
types:
  32 q::*InputSlice
  16 q::Transpose.2D
  16 q::SlicePad_shape_inplace
   8 q::Transpose_impl
   2 q::Reshape
   2 q::Concat
   2 q::ForceFormat_Crouton
   1 q::ConvLayer.opt.bias_to_vtcm
   1 q::ConvLayer.opt.weights_to_vtcm
   1 q::ConvLayer.opt.activations_to_vtcm
   1 q::ConvLayer_s1.opt

compute node:
  q::ConvLayer_s1.opt
  in0 [1,8,32,256]  dtype=1046   from ForceFormat_Crouton
  in1 [1,1,256,256] dtype=776    from weights_to_vtcm
  in2 [1,8,1,128]   dtype=50     from bias_to_vtcm
  in3 [1]           dtype=50     const
  in4 [1,1,1,3]     dtype=50     const
  out [1,8,32,256]  dtype=1046
```

The retained native chain8 reference has the same native compute contract shape:
`q::ConvLayer_s1.opt` consumes activation, weight, bias sidecar, and two const
control inputs.  The chain8 reference repeats the compute node 8 times.

## Current Interpretation

The known `zp_neutral` failure is:

```text
native/custom sidecar bytes: 4096/4096
same-hardware output exact: 65280/65536
diff: custom is +256 for every row of channel 243
```

Entry-side dump evidence narrows this further.  Building the package with
`-DHMX_W8A16_ENTRY_DUMP` and parsing
`/tmp/qcom_htp_w8a16_entry_dump4/analysis/entry_dump_zp_neutral.json` proves
that the custom wrapper sees native-exact bytes for the failing lane before
entering the owned HMX body:

```text
channel: 243
tile/lane/parity: 7 / 9 / 1
control offset: 3912
effective offset: 4040
control dump/native: 1246400001000040 == 1246400001000040
effective dump/native: 802e000000000000 == 802e000000000000
effective value: 11904
```

A normal rebuild after the diagnostic confirms the original failure still
reproduces at
`/tmp/qcom_htp_w8a16_normal_after_entry_dump/analysis/custom_native_compare_zp_neutral.json`:
`65280/65536`, maxabs `256`, exactly column `243`.

The first fix was to stop using QNN Native's final A16 sidecar as the
byte-level custom-wrapper input.  That produced a hybrid diagnostic sidecar
with native control/drain bytes and generated effective-bias fields.

The final promoted fix is fully generated.  QNN's W8A16 drain scale is not the
direct per-channel expression `act_scale * weight_scale / output_scale`.
Ctxgen first emits a normalized weight-scale vector and a scalar max-channel
scale:

```python
max_w = max(float32(weight_scale))
normalized = float32(weight_scale / max_w)
max_channel_scale = float32(float32(max_w * act_scale) / output_scale)
drain_scale = float32(normalized * max_channel_scale)
```

With that scale path, `scripts/build_w8a16_custom_a16_sidecar.py --source
generated` is the default in `scripts/run_w8a16_python_case_custom_native_match.sh`.
Evidence:

```text
/tmp/qcom_htp_w8a16_ci_default_generated/output_w8a16_per_channel_native_match_ci/analysis/custom_native_compare_summary.json
cases: normal_random, zp_neutral, positive_boundary, negative_boundary,
       single_k_impulse, bias_only, scale_only
result: 65536/65536 for every case, maxabs 0
```

One code-level hypothesis was also rejected: enabling
`-DHMX_W8A16_INTERNAL_SPLIT_N128` to split the 8 N tiles into two 128-column
HMX calls does not fix the edge and instead corrupts the full output
(`/tmp/qcom_htp_w8a16_split_n128_zp/analysis/custom_native_compare_zp_neutral.json`,
`0/65536`).  Keep that route closed unless the split descriptor contract is
rederived from native code.

Conclusion: the old blocker had two layers.  First, native final sidecar bytes
are not a valid custom-wrapper input for `zp_neutral` effective bias.  Second,
fully generated control/drain requires QNN's normalized two-stage W8A16 scale
path.  With both rules implemented, the generated sidecar is the promoted path.
