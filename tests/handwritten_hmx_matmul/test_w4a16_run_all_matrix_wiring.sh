#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python - <<'PY'
from pathlib import Path

run_all = Path("tests/handwritten_hmx_matmul/run_all.sh")
text = run_all.read_text(encoding="utf-8")
tutorial_build = Path("example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh")
build_text = tutorial_build.read_text(encoding="utf-8")

required = [
    "test_w4a16_qnn_kernel_tutorial_wrapper.sh",
    "CHAIN_STEPS=1",
    "example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh",
    "example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/run_device.sh",
    "w4a16_qnn_kernel_tutorial/device_result.json",
    "scripts/run_handwritten_artifact_body_device.py",
    "--family w4a16",
    "prepare_w4a16_small_shape_direct_hmx_artifact.py",
    "device_body_w4a16_chain8_custom_baseline.json",
    "w4a16_chain8_custom_baseline_native_bridge.json",
    "--reference-raw-override",
    "--native-transpose-2d",
    "scripts/validate_handwritten_hmx_matmul.py",
]
missing = [item for item in required if item not in text]
assert not missing, f"run_all missing direct-HMX route marker(s): {missing}"

for old_marker in (
    "run_w4a16_descdump_payload_matrix.py",
    "plan_w4a16_native_private_dump_targets.py",
    "host_ctxgen_compile_exec",
    "context_record_state_before_bias_control_load",
    "wrapper_payload_window_probe",
    "selector_mutation_plan",
):
    assert old_marker not in text, f"old W4A16 route is still wired into run_all: {old_marker}"

for old_marker in (
    "DESCRIPTOR_CARRIER",
    "--descriptor-carrier",
):
    assert old_marker not in build_text, f"old descriptor-carrier route is still exposed by tutorial build: {old_marker}"
PY
