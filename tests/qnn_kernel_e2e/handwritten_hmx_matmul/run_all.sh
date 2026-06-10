#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/handwritten_hmx_matmul_gate}"
DEVICE="${DEVICE:-oneplus}"
ARTIFACT_ONLY="${ARTIFACT_ONLY:-0}"
DEVICE_BODY_MEASURE_REPEATS="${DEVICE_BODY_MEASURE_REPEATS:-20}"
W4A16_DEVICE_BODY_MEASURE_REPEATS="${W4A16_DEVICE_BODY_MEASURE_REPEATS:-20}"
W4A16_CHAIN8_CUSTOM_ARTIFACT="example/qnn_matmul_profile/output_w4a16_aligned_e2e_256"
W4A16_CHAIN8_NATIVE_ARTIFACT="example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256"
W4A16_CHAIN8_CUSTOM_RAW="$W4A16_CHAIN8_CUSTOM_ARTIFACT/device_out/out.raw"

ACTIVE_FAMILIES=(u8i8 w4a8 w8a16 w4a16)
SMOKE_TESTS=(
  test_u8i8_smoke.sh
  test_w4a8_smoke.sh
  test_w8a16_smoke.sh
  test_w4a16_smoke.sh
)
ACCEPTED_PROFILE_TESTS=(
  test_u8i8_profile.sh
  test_w4a8_profile.sh
  test_w8a16_profile.sh
)

cd "$ROOT_DIR"
mkdir -p "$OUT_ROOT"

uv run python scripts/check_handwritten_oracle_manifest.py
uv run python scripts/check_handwritten_shape_matrix.py
uv run python scripts/check_handwritten_profile_matrix.py
uv run python scripts/check_handwritten_hmx_body.py \
  --json-out "$OUT_ROOT/body_check.json"
uv run python scripts/check_handwritten_hmx_body_entry_sim.py \
  --json-out "$OUT_ROOT/body_entry_sim.json"

for family in "${ACTIVE_FAMILIES[@]}"; do
  uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \
    --family "$family" \
    --out-dir "$OUT_ROOT/$family"
  uv run python scripts/run_handwritten_artifact_body_sim.py \
    --family "$family" \
    --artifact "$OUT_ROOT/$family" \
    --json-out "$OUT_ROOT/artifact_body_${family}.json"
done

# QNN-free standalone W16A16 (handwriting route): byte-exact vs native, sim-only.
OUT_ROOT="$OUT_ROOT" \
  "tests/qnn_kernel_e2e/handwritten_hmx_matmul/test_w16a16_standalone.sh"

uv run python scripts/prepare_w4a16_small_shape_direct_hmx_artifact.py \
  --custom-artifact "$W4A16_CHAIN8_CUSTOM_ARTIFACT" \
  --native-artifact "$W4A16_CHAIN8_CUSTOM_ARTIFACT" \
  --out-dir "$OUT_ROOT/w4a16_chain8_custom_baseline"
uv run python scripts/run_handwritten_artifact_body_sim.py \
  --family w4a16 \
  --artifact "$OUT_ROOT/w4a16_chain8_custom_baseline" \
  --native-raw-override "$W4A16_CHAIN8_CUSTOM_RAW" \
  --json-out "$OUT_ROOT/artifact_body_w4a16_chain8_custom_baseline.json"

uv run python scripts/audit_handwritten_hmx_matmul_roadmap.py \
  --artifact-root "$OUT_ROOT" \
  --json-out "$OUT_ROOT/roadmap_audit.json"
uv run python scripts/summarize_handwritten_completion_checklist.py \
  --artifact-root "$OUT_ROOT" \
  --json-out "$OUT_ROOT/completion_checklist.json"

if [[ "$ARTIFACT_ONLY" == "1" ]]; then
  uv run python scripts/validate_handwritten_hmx_matmul.py \
    --artifact-root "$OUT_ROOT"
  exit 0
fi

example/handwritten_hmx_matmul/build_host.sh
example/handwritten_hmx_matmul/build_android.sh

for test_script in "${SMOKE_TESTS[@]}"; do
  OUT_ROOT="$OUT_ROOT" DEVICE="$DEVICE" SKIP_BUILD=1 \
    "tests/qnn_kernel_e2e/handwritten_hmx_matmul/$test_script"
done

for family in u8i8 w4a8 w8a16; do
  uv run python scripts/run_handwritten_artifact_body_device.py \
    --family "$family" \
    --artifact "$OUT_ROOT/$family" \
    --device "$DEVICE" \
    --remote-dir "handwritten_hmx_matmul_device_body_${family}" \
    --measure-repeats "$DEVICE_BODY_MEASURE_REPEATS" \
    --json-out "$OUT_ROOT/device_body_${family}.json"
done

# QNN-free standalone W16A16 on real CDSP (device-backed byte-exact gate).
OUT_ROOT="$OUT_ROOT" DEVICE="$DEVICE" \
  "tests/qnn_kernel_e2e/handwritten_hmx_matmul/test_w16a16_standalone_device.sh"

# QNN-free standalone W16A16 value-distribution gate (uniform + the extreme/impulse
# edge cases that catch the int16 weight high-byte clip bug). Always on.
DEVICE="$DEVICE" \
  "tests/qnn_kernel_e2e/handwritten_hmx_matmul/test_w16a16_standalone_dist.sh"

# Exhaustive multi-shape + full value-distribution sweeps are heavy (one QNN
# device run per shape/dist). Opt-in via W16A16_FULL_SWEEPS=1 to keep the
# every-push preflight bounded.
if [[ "${W16A16_FULL_SWEEPS:-0}" == "1" ]]; then
  DEVICE="$DEVICE" \
    "tests/qnn_kernel_e2e/handwritten_hmx_matmul/test_w16a16_standalone_sweep.sh"
fi

uv run python scripts/run_handwritten_artifact_body_device.py \
  --family w4a16 \
  --artifact "$OUT_ROOT/w4a16_chain8_custom_baseline" \
  --device "$DEVICE" \
  --remote-dir "handwritten_hmx_matmul_device_body_w4a16_chain8_custom_baseline" \
  --measure-repeats "$W4A16_DEVICE_BODY_MEASURE_REPEATS" \
  --reference-raw-override "$W4A16_CHAIN8_CUSTOM_RAW" \
  --json-out "$OUT_ROOT/device_body_w4a16_chain8_custom_baseline.json"
uv run python scripts/summarize_w4a16_custom_baseline_native_bridge.py \
  --direct "$OUT_ROOT/device_body_w4a16_chain8_custom_baseline.json" \
  --custom-dir "$W4A16_CHAIN8_CUSTOM_ARTIFACT" \
  --native-dir "$W4A16_CHAIN8_NATIVE_ARTIFACT" \
  --json-out "$OUT_ROOT/w4a16_chain8_custom_baseline_native_bridge.json"
uv run python scripts/validate_qnn_kernel_e2e.py \
  --kernel w4a16 \
  --custom-dir "$W4A16_CHAIN8_CUSTOM_ARTIFACT" \
  --native-dir "$W4A16_CHAIN8_NATIVE_ARTIFACT" \
  --expected-htp-type QnnHmxMatMulW4A16Package::HmxU16I4ToU16MatMul \
  --qnn-prefix hmx_w4a16_chain \
  --expected-qnn-ops 8 \
  --dtype uint16 \
  --native-transpose-2d

uv run python scripts/summarize_handwritten_promotion_evidence.py \
  --artifact-root "$OUT_ROOT" \
  --json-out "$OUT_ROOT/promotion_evidence.json"

for test_script in "${ACCEPTED_PROFILE_TESTS[@]}"; do
  OUT_ROOT="$OUT_ROOT" DEVICE="$DEVICE" SKIP_BUILD=1 \
    "tests/qnn_kernel_e2e/handwritten_hmx_matmul/$test_script"
done

uv run python scripts/audit_handwritten_hmx_matmul_roadmap.py \
  --artifact-root "$OUT_ROOT" --require-device \
  --json-out "$OUT_ROOT/roadmap_audit.json"
uv run python scripts/summarize_handwritten_completion_checklist.py \
  --artifact-root "$OUT_ROOT" \
  --json-out "$OUT_ROOT/completion_checklist.json"
uv run python scripts/validate_handwritten_hmx_matmul.py \
  --artifact-root "$OUT_ROOT" --require-device
