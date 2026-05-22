# Handwritten HMX MatMul Runtime Boundary

This page records the current QNN-free runtime boundary for handwritten HMX
MatMul.

## Runtime Tree

Primary files:

- `example/handwritten_hmx_matmul/include/`
- `example/handwritten_hmx_matmul/kernels/`
- `example/handwritten_hmx_matmul/src/handwritten_hmx_matmul.cpp`
- `example/handwritten_hmx_matmul/prepare_owned_inputs.py`
- `example/handwritten_hmx_matmul/run_owned_smoke.py`
- `scripts/run_handwritten_artifact_body_device.py`
- `scripts/run_handwritten_artifact_body_sim.py`
- `tests/qnn_kernel_e2e/correctness/test_handwritten_hmx_matmul_e2e.sh`
- `tests/qnn_kernel_e2e/handwritten_hmx_matmul/run_all.sh`

The owned runtime does not execute QNN.  It prepares owned buffers, calls the
replicated HMX body directly, and compares against retained offline oracle raw
outputs.

## Active Families

The active gate covers:

| Family | Body |
|---|---|
| `u8i8` | `hm_u8i8_v73deep_kernel` |
| `w4a8` | `hm_w4a8_v73deep_kernel` |
| `w8a16` | `hm_w8a16_v75deep_kernel` |
| `w4a16` | `hm_w4a16_v73deep_kernel` |

W16A16 body material is retained as reference only and is not in the default
gate.

## W4A16 Runtime Contract

W4A16 acceptance uses a retained QNN custom-op artifact as the direct-HMX
baseline:

- custom artifact:
  `example/qnn_matmul_profile/output_w4a16_aligned_e2e_256`
- native bridge artifact:
  `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256`
- direct-HMX comparison raw:
  `example/qnn_matmul_profile/output_w4a16_aligned_e2e_256/device_out/out.raw`
- custom/native transform:
  `native_transpose_2d`

`scripts/run_handwritten_artifact_body_device.py` accepts
`--reference-raw-override` so the W4A16 chain8 direct body compares against the
custom-op public raw output.  `scripts/summarize_w4a16_custom_baseline_native_bridge.py`
then verifies that the same public raw output matches native raw after
`native_transpose_2d`.

## Gate

Run the formal CI device gate:

```bash
HANDWRITTEN_HMX_MATMUL_OUT_ROOT=/tmp/handwritten_hmx_matmul_gate \
DEVICE=oneplus \
tests/qnn_kernel_e2e/correctness/test_handwritten_hmx_matmul_e2e.sh
```

Artifact-only smoke:

```bash
ARTIFACT_ONLY=1 HANDWRITTEN_HMX_MATMUL_OUT_ROOT=/tmp/handwritten_hmx_matmul_gate \
tests/qnn_kernel_e2e/correctness/test_handwritten_hmx_matmul_e2e.sh
```

The wrapper delegates to
`tests/qnn_kernel_e2e/handwritten_hmx_matmul/run_all.sh`; that script is the
implementation sub-gate, not the formal CI entrypoint.

The last accepted device run used:

```bash
OUT_ROOT=/tmp/handwritten_hmx_matmul_gate_device_refresh2 \
DEVICE=oneplus \
tests/qnn_kernel_e2e/handwritten_hmx_matmul/run_all.sh
```

It produced a complete checklist with `pass=3`, `open=0`, `fail=0`.
