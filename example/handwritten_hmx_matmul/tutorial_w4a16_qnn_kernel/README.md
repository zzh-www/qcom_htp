# W4A16 QNN Kernel Tutorial Wrapper

This directory keeps the tutorial-style CDSP wrapper for the recovered W4A16
HMX body.  It is a QNN-free runtime path: QNN artifacts provide retained bytes
and raw-output oracles offline, but the shared object runs through
`run_main_on_hexagon`, acquires HAP/HVX/HMX/VTCM resources directly, and calls
`hm_w4a16_v73deep_kernel`.

QNN is not used at runtime.

## Build

```bash
bash example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh
```

Useful environment variables:

- `OUT_DIR`: output directory for the generated source, manifest, and shared
  object.
- `CHAIN_STEPS`: number of HMX body calls for the tutorial route-gate run.
- `KERNEL_ENTRY`: `deep`, `wrapper`, `wrapper_nondeep`, or `split_n128`.
  Gate marker: `KERNEL_ENTRY=deep|wrapper|split_n128`.
- `PRE_CLEAR_ACC`, `NATIVE_WRAPPER_PREFETCH`,
  `PRELOAD_HMX_IDENTITY_BIAS`: diagnostic toggles retained for route-gate
  provenance.

## Run

```bash
DEVICE=oneplus bash example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/run_device.sh
```

The device runner records:

- prepared-state checksum visibility;
- call-ABI scalar visibility;
- VTCM offset visibility;
- step trace;
- HNH path fields;
- output checksum and byte-diff status.

The tutorial chain1 run is a route-gate/provenance check.  It is not the final
W4A16 acceptance oracle.

## Final W4A16 Acceptance

Final W4A16 acceptance is checked by the shared direct-body harness:

```bash
uv run python scripts/prepare_w4a16_small_shape_direct_hmx_artifact.py \
  --custom-artifact example/qnn_matmul_profile/output_w4a16_aligned_e2e_256 \
  --native-artifact example/qnn_matmul_profile/output_w4a16_aligned_e2e_256 \
  --out-dir /tmp/handwritten_hmx_matmul_custom_baseline/w4a16_256_chain8_custombaseline
uv run python scripts/run_handwritten_artifact_body_device.py \
  --family w4a16 \
  --artifact /tmp/handwritten_hmx_matmul_custom_baseline/w4a16_256_chain8_custombaseline \
  --kernel-entry deep \
  --public-output-layout default \
  --reference-raw-override example/qnn_matmul_profile/output_w4a16_aligned_e2e_256/device_out/out.raw \
  --measure-repeats 20 \
  --json-out /tmp/w4a16_256_chain8_custombaseline_deep_probe.json \
  --remote-dir handwritten_w4a16_256_chain8_custombaseline_deep_probe
uv run python scripts/summarize_w4a16_custom_baseline_native_bridge.py \
  --direct /tmp/w4a16_256_chain8_custombaseline_deep_probe.json \
  --custom-dir example/qnn_matmul_profile/output_w4a16_aligned_e2e_256 \
  --native-dir example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256 \
  --json-out /tmp/w4a16_chain8_custom_baseline_native_bridge.json
```

Accepted evidence:

- direct-HMX output matches custom chain8 raw output byte-for-byte;
- checksum is `0xfcdb7a52`;
- custom-to-native public layout is exactly `native_transpose_2d`;
- `native_transpose_2d` is `65536/65536` exact with byte diffs `0`.

## Gate

The canonical route gate is:

```bash
OUT_ROOT=/tmp/handwritten_hmx_matmul_gate \
DEVICE=oneplus \
tests/handwritten_hmx_matmul/run_all.sh
```

The last accepted device run used
`/tmp/handwritten_hmx_matmul_gate_device_refresh2` and promoted all active
families.
