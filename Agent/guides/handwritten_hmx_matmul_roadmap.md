# Handwritten HMX MatMul Roadmap

## Objective

Build owned, QNN-free handwritten implementations for the quantized MatMul
families that QNN lowers into HMX Conv1x1-style kernels.

Runtime execution must not use QNN graph execution, QNN ctxgen,
`qnn-net-run`, QNN custom-op callbacks, context binaries, or sidecar ops.  QNN
Native remains only an offline oracle source for prepared bytes, native raw
outputs, packet/body slices, and performance references.

W16A16 is retained as reference material only.  It is not part of the current
active gate.

## Current Acceptance State

The current route is complete for the active families:

```bash
OUT_ROOT=/tmp/handwritten_hmx_matmul_gate_device_refresh2 \
  DEVICE=oneplus tests/handwritten_hmx_matmul/run_all.sh
```

The gate reports:

- `handwritten HMX MatMul gate: ok`
- promoted families: `u8i8`, `w4a8`, `w8a16`, `w4a16`
- completion checklist: `pass=4`, `open=0`, `fail=0`
- roadmap audit: `pass=5`, `open=0`, `fail=0`, `w4a16_complete=true`

The machine-readable evidence lives under
`/tmp/handwritten_hmx_matmul_gate_device_refresh2/`:

| Evidence | Required result |
|---|---|
| `promotion_evidence.json` | promoted families are `u8i8`, `w4a8`, `w8a16`, `w4a16`; `unpromoted_families=[]` |
| `completion_checklist.json` | `roadmap_complete=true`, `pass=4`, `open=0`, `fail=0` |
| `roadmap_audit.json` | `w4a16_complete=true`, `pass=5`, `open=0`, `fail=0` |
| `device_body_w4a16_chain8_custom_baseline.json` | `byte_exact_device_diff`, byte diffs `0`, checksum `0xfcdb7a52` |
| `w4a16_chain8_custom_baseline_native_bridge.json` | `accepted_bridge=true`; `native_transpose_2d` is `65536/65536` exact with byte diffs `0` |

## Active Route

The active runtime and gate live under:

- `example/handwritten_hmx_matmul/`
- `scripts/run_handwritten_artifact_body_device.py`
- `tests/handwritten_hmx_matmul/run_all.sh`

The gate performs:

1. Oracle manifest, shape matrix, and profile matrix validation.
2. Byte-identity checks for the owned HMX body slices.
3. H2/hexagon-sim body-entry smoke.
4. Host and Android owned-runtime smoke.
5. Direct device-body exactness checks for `u8i8`, `w4a8`, `w8a16`.
6. W4A16 chain8 custom-baseline direct-HMX exactness.
7. W4A16 custom/native public-layout bridge through the existing
   `native_transpose_2d` Python rule.
8. Promotion evidence, roadmap audit, completion checklist, and final validator.

The W4A16 route is intentionally custom-baseline first:

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

Do not restart the old W4A16 QNN blackbox route as implementation work.  QNN
custom/native artifacts are retained only as offline oracle and provenance.

## Completion Criteria

The goal is complete only when all of these are true:

- the full device gate `tests/handwritten_hmx_matmul/run_all.sh` passes;
- `promotion_evidence.json` promotes all active families;
- `completion_checklist.json` has no blockers;
- `roadmap_audit.json` reports W4A16 complete;
- W4A16 direct-HMX chain8 custom-baseline output is byte-exact;
- the W4A16 native bridge uses `native_transpose_2d` and is exact after the
  transform.

These criteria are currently satisfied by
`/tmp/handwritten_hmx_matmul_gate_device_refresh2/`.
