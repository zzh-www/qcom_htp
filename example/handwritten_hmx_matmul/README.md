# Handwritten HMX MatMul

This directory is the QNN-free runtime/spec tree for the handwritten HMX
MatMul work.

The active families are U8I8, W4A8, W8A16, and W4A16.  W16A16 material is
retained as inactive oracle and body-slice reference only.

- `oracles.json`: machine-readable QNN Native oracle manifest generated
  from retained standard artifacts.
- `include/handwritten_hmx_matmul.h`: prepared-state C ABI for all target
  families.
- `include/handwritten_hmx_u8i8_kernel.h` and
  `kernels/u8i8/v73deep_conv1x1_kernel.inc`: U8I8 Hexagon body ABI and
  byte-identical inline-asm body.
- `include/handwritten_hmx_w4a8_kernel.h` and
  `kernels/w4a8/v73deep_conv1x1_kernel.inc`: W4A8 Hexagon body ABI and
  byte-identical inline-asm body.
- `include/handwritten_hmx_w8a16_kernel.h` and
  `kernels/w8a16/v73deep_conv1x1_kernel.inc`: W8A16 Hexagon body ABI and
  byte-identical inline-asm body.
- `include/handwritten_hmx_w4a16_kernel.h` and
  `kernels/w4a16/v73deep_conv1x1_kernel.inc`: W4A16 Hexagon body ABI and
  byte-identical inline-asm body.
- `include/handwritten_hmx_w16a16_kernel.h` and
  `kernels/w16a16/v73deep_conv1x1_kernel.inc`: retained W16A16 reference
  material.  It is not part of the current active gate.
- `build_host.sh` and `run_owned_smoke.py`: host-only smoke path for the
  owned runtime boundary.
- `build_android.sh`: AArch64 Android build path for direct device smoke.
- `prepare_owned_inputs.py`: owned preparation scaffold that writes
  `prepared_state/`, `analysis/prep_compare.json`, and
  `analysis/prep_profile.json`.
- `tutorial_w4a16_qnn_kernel/`: tutorial-style direct CDSP wrapper that builds
  the recovered W4A16 QNN HMX body as a `run_main_on_hexagon` shared object.
  This is the active W4A16 route: QNN supplies retained prepared bytes and the
  native raw oracle offline, but runtime execution enters HAP/HVX/HMX/VTCM setup
  directly and calls `hm_w4a16_v73deep_kernel` without a QNN context.

Regenerate and validate the oracle freeze from repo root:

```bash
uv run python scripts/build_handwritten_oracle_manifest.py
uv run python scripts/check_handwritten_oracle_manifest.py
```

Verify current HMX body byte identity and ABI header compilation:

```bash
uv run python scripts/check_handwritten_hmx_body.py \
  --json-out /tmp/handwritten_hmx_body_check.json
```

Build and run the active W4A16 tutorial/direct-HMX wrapper:

```bash
bash example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh
DEVICE=oneplus bash example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/run_device.sh
```

Run the current owned smoke gate:

```bash
OUT_ROOT=/tmp/handwritten_hmx_matmul_gate \
DEVICE=oneplus \
tests/handwritten_hmx_matmul/run_all.sh
```

For host-only smoke without device execution:

```bash
ARTIFACT_ONLY=1 OUT_ROOT=/tmp/handwritten_hmx_matmul_gate \
tests/handwritten_hmx_matmul/run_all.sh
```

The artifact-only path regenerates each family artifact under `OUT_ROOT`
before running body-simulator and validator checks, so it must not depend
on stale `/tmp` artifacts from a previous run.

Each canonical family also has a named smoke entry under
`tests/handwritten_hmx_matmul/test_<family>_smoke.sh`.

Build and validate the current host smoke boundary:

```bash
example/handwritten_hmx_matmul/build_host.sh
uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \
  --family u8i8 --out-dir /tmp/handwritten_hmx_matmul_owned_smoke_u8i8
uv run python scripts/check_handwritten_runtime_artifact.py \
  /tmp/handwritten_hmx_matmul_owned_smoke_u8i8
```

Build and validate a direct device smoke artifact:

```bash
uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \
  --family u8i8 --device oneplus \
  --out-dir /tmp/handwritten_hmx_matmul_owned_device_u8i8
uv run python scripts/check_handwritten_runtime_artifact.py \
  /tmp/handwritten_hmx_matmul_owned_device_u8i8 --require-device
```

The generic owned smoke path remains a preparation/runtime boundary check, not
the final W4A16 acceptance path.  The active W4A16 path is the tutorial
`run_main_on_hexagon` wrapper above: it enters the recovered HMX body on CDSP
with retained native-prepared state and compares against matched QNN Native raw
oracles without executing QNN at runtime.

The owned gate validates current route evidence.  It includes per-family
artifact-body simulator smoke, direct device-body evidence, W4A16 route gates,
W4A16 chain8 custom-baseline exactness, the W4A16 `native_transpose_2d`
custom/native bridge, promotion evidence, roadmap audit, and completion
checklist.  The current device gate promotes all active families.
