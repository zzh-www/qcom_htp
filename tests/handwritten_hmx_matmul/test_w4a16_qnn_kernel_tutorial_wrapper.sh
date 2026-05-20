#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python -m py_compile \
  scripts/build_w4a16_qnn_kernel_tutorial.py \
  scripts/run_w4a16_qnn_kernel_tutorial_device.py \
  scripts/check_w4a16_tutorial_chain1_sources.py \
  scripts/emulate_hmx_conv1x1_params.py
uv run python scripts/build_w4a16_qnn_kernel_tutorial.py --help >/dev/null
uv run python scripts/run_w4a16_qnn_kernel_tutorial_device.py --help >/dev/null
uv run python scripts/check_w4a16_tutorial_chain1_sources.py --help >/dev/null

test -x example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh
test -x example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/run_device.sh

uv run python - <<'PY'
from pathlib import Path

readme = Path("example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/README.md")
text = readme.read_text(encoding="utf-8")
required = [
    "run_main_on_hexagon",
    "HAP",
    "VTCM",
    "hm_w4a16_v73deep_kernel",
    "KERNEL_ENTRY=deep|wrapper|split_n128",
    "QNN is not used at runtime",
]
missing = [item for item in required if item not in text]
assert not missing, f"tutorial wrapper README missing required markers: {missing}"

builder = Path("scripts/build_w4a16_qnn_kernel_tutorial.py").read_text(encoding="utf-8")
for marker in (
    'source_stem="w4a16_qnn_kernel_tutorial"',
    '"qnn_runtime_used": False',
    '"run_main_on_hexagon_hap_vtcm_hmx_lock"',
    "expected_prepared_state_checksums",
    "expected_call_abi_scalars",
    "expected_vtcm_offsets",
    "expected_hnh_path",
    "expected_step0_native_raw",
    "step_trace_enabled",
    "--chain-steps",
    '"chain_steps": args.chain_steps',
    '"final_native_oracle": "chain1" if args.chain_steps == 1 else "chain8"',
    "--native-wrapper-prefetch",
    '"native_wrapper_prefetch": args.native_wrapper_prefetch',
    "--preload-hmx-identity-bias",
    '"preload_hmx_identity_bias": args.preload_hmx_identity_bias',
    "--output-seed-mode",
    '"output_seed_mode": args.output_seed_mode',
    "--activation-raw-override",
    '"activation_raw_override"',
    "--folded-bias-byte-offset",
    '"folded_bias_byte_offset": args.folded_bias_byte_offset',
    "--packed-weight-byte-offset",
    '"packed_weight_byte_offset": args.packed_weight_byte_offset',
    "--packed-weight-raw-override",
    '"packed_weight_raw_override"',
    "--folded-bias-raw-override",
    '"folded_bias_raw_override"',
    "--extra-word-override",
    '"extra_word_overrides"',
    "--mask-word-override",
    '"mask_word_overrides"',
    '"dynamic_mask14_pointer_patch": True',
    "--kernel-entry",
    '"kernel_entry": args.kernel_entry',
    "split_n128",
    "expected_hnh_path(",
    '"hm_w4a16_v73wrapper_entry_kernel"',
):
    assert marker in builder, f"tutorial builder missing marker: {marker}"

runner = Path("scripts/run_w4a16_qnn_kernel_tutorial_device.py").read_text(encoding="utf-8")
for marker in (
    "w4a16_qnn_kernel_tutorial_device_run.v1",
    "parse_device_log",
    '"qnn_runtime_used": False',
    "entered_and_returned",
    "prepared_state_compare",
    "call_abi_compare",
    "vtcm_offset_compare",
    "step_trace_compare",
    "hnh_path_compare",
    "final_native_oracle",
    "kernel_entry",
    "chain_steps",
    "native_wrapper_prefetch",
    "preload_hmx_identity_bias",
    "output_seed_mode",
    "activation_raw_override",
    "folded_bias_byte_offset",
    "packed_weight_byte_offset",
    "extra_word_overrides",
    "mask_word_overrides",
    "compare_prepared_checksums",
    "compare_call_abi_scalars",
    "compare_vtcm_offsets",
    "compare_step_trace",
    "compare_hnh_path",
):
    assert marker in runner, f"tutorial device runner missing marker: {marker}"

source_check = Path("scripts/check_w4a16_tutorial_chain1_sources.py").read_text(encoding="utf-8")
for marker in (
    "w4a16_tutorial_chain1_source_check.v1",
    "chain1_vs_chain8_activation_raw_exact",
    "chain1_vs_chain8_logical_weight_exact",
    "chain1_vs_custom_logical_weight_exact",
    "visible_abi_comparisons",
    "native_context_matches",
    "native_context_offset_hex",
    "pack_a16_crouton16_row4_surface",
    "pack_w4_kblock32_nmajor_k4_lohi",
    "pack_native_a16_bias",
    "convw4b1x1_words(0x70B, 256, 0, 0, 0, 0xA0)",
):
    assert marker in source_check, f"chain1 source checker missing marker: {marker}"

from scripts.emulate_hmx_conv1x1_params import convw4b1x1_words

assert convw4b1x1_words(0x70B, 256, 0, 0, 0, 0xA0) == [
    0,
    0x700,
    0,
    0x77C,
    0,
    0,
    0x3FF,
    0,
    0,
    0,
    0,
    0,
    0xA0,
    0,
    0,
    0,
]

parser = Path("scripts/run_handwritten_artifact_body_device.py").read_text(encoding="utf-8")
for marker in (
    'HM_REMOTE_TAG " prepared family=',
    "prepared_state_checksums",
    "prepared_state_device_visible",
    "call_abi_scalars",
    "call_abi_device_visible",
    "pointer_layout",
    "pointer_layout_device_visible",
    "data_callptr",
    "data_callptr_device_visible",
    "data_callptr_pattern",
    'HM_REMOTE_TAG " data_callptr family=',
    "hmx_identity_bias_preload",
    "hmx_identity_bias_preload_device_visible",
    "hmx_identity_bias_preload_pattern",
    'HM_REMOTE_TAG " hmx_identity_bias_preload family=',
    "preload_hmx_bias_state",
    "bias = mxmem2(%0)",
    "dynamic_mask",
    "dynamic_mask_device_visible",
    "dynamic_mask_pattern",
    "hnh_path",
    "hnh_path_device_visible",
    "hnh_path_pattern",
    'HM_REMOTE_TAG " hnh_path family=',
    "step_trace",
    "step_trace_present",
    "chain_steps",
    "reference_raw_override",
    "--reference-raw-override",
    "u16_diff_summary",
    "u16_diff_samples",
    "u16_diff_summary_present",
    "u16_value_bin_summaries",
    "u16_value_bin_summaries_present",
    "u16_value_bin_pattern",
    'HM_REMOTE_TAG " u16_value_bin family=',
    "byte_lane_summaries",
    "byte_lane_summaries_present",
    "byte_lane_pattern",
    'HM_REMOTE_TAG " byte_lane family=',
    "u16_region_summaries",
    "u16_region_summaries_present",
    "diff_u16_region_pattern",
    "internal_block_summaries",
    "internal_block_summaries_present",
    "internal_block_pattern",
    "written_block_summaries",
    "written_block_summaries_present",
    "written_block_pattern",
    'HM_REMOTE_TAG " written_block family=',
    "alt_layout_summaries",
    "alt_layout_summaries_present",
    "alt_layout_pattern",
    'HM_REMOTE_TAG " alt_layout family=',
    "crouton16_pair_major",
    "public_raw_to_internal",
    'HM_REMOTE_TAG " dynamic_mask family=',
    "mask_words[14] = (uint32_t)(uintptr_t)extra;",
    "diff_u16_pattern",
    "diff_u16_endpoint_pattern",
    "actual_zero_elements",
    "mismatch_actual_ffff_elements",
    "init_output_seed",
    "hm_w4a16_v73wrapper_entry_kernel",
    "--kernel-entry",
    "kernel_entry",
):
    assert marker in parser, f"device parser missing prepared-state marker: {marker}"

shell = Path("example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/run_device.sh").read_text(
    encoding="utf-8"
)
assert "scripts/run_w4a16_qnn_kernel_tutorial_device.py" in shell
assert "device_result.json" in shell

build_shell = Path("example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh").read_text(
    encoding="utf-8"
)
assert "CHAIN_STEPS" in build_shell
assert "--chain-steps" in build_shell
assert "PRE_CLEAR_ACC" in build_shell
assert "NATIVE_WRAPPER_PREFETCH" in build_shell
assert "--native-wrapper-prefetch" in build_shell
assert "PRELOAD_HMX_IDENTITY_BIAS" in build_shell
assert "--preload-hmx-identity-bias" in build_shell
assert "OUTPUT_SEED_MODE" in build_shell
assert "--output-seed-mode" in build_shell
assert "ACTIVATION_RAW_OVERRIDE" in build_shell
assert "--activation-raw-override" in build_shell
assert "FOLDED_BIAS_BYTE_OFFSET" in build_shell
assert "--folded-bias-byte-offset" in build_shell
assert "PACKED_WEIGHT_BYTE_OFFSET" in build_shell
assert "--packed-weight-byte-offset" in build_shell
assert "PACKED_WEIGHT_RAW_OVERRIDE" in build_shell
assert "--packed-weight-raw-override" in build_shell
assert "FOLDED_BIAS_RAW_OVERRIDE" in build_shell
assert "--folded-bias-raw-override" in build_shell
assert "EXTRA_WORD_OVERRIDES" in build_shell
assert "--extra-word-override" in build_shell
assert "MASK_WORD_OVERRIDES" in build_shell
assert "--mask-word-override" in build_shell
assert "KERNEL_ENTRY" in build_shell
assert "--kernel-entry" in build_shell
PY
