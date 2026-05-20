#!/usr/bin/env python3
"""Validate a handwritten HMX MatMul owned-runtime artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FORBIDDEN = (
    "qnn-context-binary-generator",
    "qnn-net-run",
    "qairt-converter",
    "qairt-quantizer",
    "QnnHmxMatMul",
    "libQnn",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_artifact(path: Path, require_device: bool) -> list[str]:
    errors: list[str] = []
    run_path = path / "owned_run.json"
    prep_path = path / "prepared_state" / "manifest.json"
    abi_manifest_path = path / "analysis" / "abi_manifest.json"
    prep_compare_path = path / "analysis" / "prep_compare.json"
    owned_output_compare_path = path / "analysis" / "owned_output_compare.json"
    prep_profile_path = path / "analysis" / "prep_profile.json"
    output_path = path / "device_out" / "out.raw"
    if not run_path.is_file():
        return [f"missing owned_run.json under {path}"]
    run = load_json(run_path)
    if run.get("schema") != "handwritten_hmx_matmul_owned_run.v1":
        errors.append(f"unexpected owned_run schema: {run.get('schema')!r}")
    if run.get("qnn_used") is not False:
        errors.append("owned_run qnn_used is not false")
    if run.get("compute_backend") != "copy_smoke":
        errors.append(f"owned_run compute_backend is unexpected for smoke gate: {run.get('compute_backend')!r}")
    if run.get("hmx_body_entered") is not False:
        errors.append("owned_run smoke artifact must not claim HMX body entry")
    if run.get("output_exactness_status") != "not_checked_copy_smoke":
        errors.append(
            "owned_run smoke artifact must record output_exactness_status=not_checked_copy_smoke"
        )
    if run.get("accepted_for_milestone4_compute_gate") is not False:
        errors.append("owned_run smoke artifact must not be accepted for milestone4 compute gate")
    if run.get("forbidden_qnn_tools_observed") not in ([], None):
        errors.append("owned_run reports forbidden QNN tools")
    if not isinstance(run.get("runtime_us"), int) or run.get("runtime_us") < 0:
        errors.append("owned_run missing non-negative runtime_us")
    if run.get("preparation_included") is not False:
        errors.append("owned_run must keep preparation_included=false")
    if require_device and run.get("device_execution") is not True:
        errors.append("device execution is required but owned_run is not device-backed")
    if require_device and run.get("accepted_for_milestone1_device_gate") is not True:
        errors.append("device execution is present but milestone1 device gate is not accepted")

    command = " ".join(str(part) for part in run.get("command", []))
    hits = [token for token in FORBIDDEN if token in command]
    if hits:
        errors.append(f"owned command contains forbidden QNN tokens: {hits}")

    if not prep_path.is_file():
        errors.append("missing prepared_state/manifest.json")
    else:
        prep = load_json(prep_path)
        if prep.get("schema") != "handwritten_hmx_matmul_prepared_state.v1":
            errors.append(f"unexpected prepared-state schema: {prep.get('schema')!r}")
        if prep.get("qnn_used") is not False:
            errors.append("prepared-state qnn_used is not false")
        files = prep.get("files", {})
        for key in (
            "activation_source",
            "activation",
            "packed_weight",
            "folded_bias",
            "output_surface",
            "control",
            "extra_control",
            "activation_table",
            "output_table",
            "descriptor",
            "mask_control",
        ):
            rel = files.get(key)
            if not rel:
                errors.append(f"prepared-state missing file entry {key}")
                continue
            file_path = prep_path.parent / rel
            if not file_path.is_file() or file_path.stat().st_size == 0:
                errors.append(f"prepared-state file missing or empty: {key} -> {rel}")
        contract = prep.get("native_compute_contract", {})
        if not contract.get("inputs") or not contract.get("outputs"):
            errors.append("prepared-state missing native compute contract")
    if not prep_compare_path.is_file():
        errors.append("missing analysis/prep_compare.json")
    else:
        prep_compare = load_json(prep_compare_path)
        if prep_compare.get("schema") != "handwritten_hmx_matmul_prep_compare.v1":
            errors.append(f"unexpected prep-compare schema: {prep_compare.get('schema')!r}")
        if prep_compare.get("qnn_used") is not False:
            errors.append("prep-compare qnn_used is not false")
        acceptance = prep_compare.get("acceptance", {})
        if acceptance.get("activation_exact_source_copy") is not True:
            errors.append("prep-compare does not prove activation exact source copy")
        blockers = prep_compare.get("milestone2_blockers")
        if not isinstance(blockers, list):
            errors.append("prep-compare missing milestone2_blockers list")
        elif acceptance.get("milestone2_complete") is True and blockers:
            errors.append("prep-compare milestone2_complete cannot keep open blockers")
        elif acceptance.get("milestone2_complete") is not True and not blockers:
            errors.append("prep-compare must list blockers while milestone2 is incomplete")
        elif isinstance(blockers, list):
            for idx, blocker in enumerate(blockers):
                if not isinstance(blocker, dict):
                    errors.append(f"prep-compare milestone2_blockers[{idx}] is not an object")
                    continue
                if not blocker.get("requirement"):
                    errors.append(f"prep-compare milestone2_blockers[{idx}] missing requirement")
                if blocker.get("status") != "open":
                    errors.append(f"prep-compare milestone2_blockers[{idx}] status must be open")
                details = blocker.get("details")
                if not isinstance(details, list) or not details:
                    errors.append(f"prep-compare milestone2_blockers[{idx}] missing details")
                    continue
                for detail_idx, detail in enumerate(details):
                    if not isinstance(detail, dict):
                        errors.append(
                            f"prep-compare milestone2_blockers[{idx}].details[{detail_idx}] is not an object"
                        )
                        continue
                    if not detail.get("buffer"):
                        errors.append(
                            f"prep-compare milestone2_blockers[{idx}].details[{detail_idx}] missing buffer"
                        )
                    if detail.get("status") not in (
                        "missing_comparison_record",
                        "missing_native_target",
                        "byte_mismatch",
                        "not_promoted_to_requirement",
                    ):
                        errors.append(
                            f"prep-compare milestone2_blockers[{idx}].details[{detail_idx}] has unknown status"
                        )
        outputs = prep_compare.get("outputs", {})
        comparisons = prep_compare.get("buffer_comparisons", {})
        for key in (
            "activation_source",
            "activation",
            "packed_weight",
            "folded_bias",
            "output_surface",
            "control",
            "extra_control",
            "activation_table",
            "output_table",
            "descriptor",
            "mask_control",
        ):
            if key not in outputs:
                errors.append(f"prep-compare missing output record {key}")
            record = comparisons.get(key)
            if not isinstance(record, dict):
                errors.append(f"prep-compare missing buffer comparison {key}")
                continue
            for field in (
                "target_kind",
                "target_available",
                "target_path",
                "target_bytes",
                "target_sha256",
                "exact",
                "first_mismatch_offset",
                "native_prepared_exact_claim",
            ):
                if field not in record:
                    errors.append(f"prep-compare comparison {key} missing field {field}")
        packing = prep_compare.get("packing", {})
        if "activation" not in packing:
            errors.append("prep-compare missing activation formatting record")
        if "output_surface" not in packing:
            errors.append("prep-compare missing output surface formatting record")
    if not owned_output_compare_path.is_file():
        errors.append("missing analysis/owned_output_compare.json")
    else:
        owned_output_compare = load_json(owned_output_compare_path)
        if owned_output_compare.get("schema") != "handwritten_hmx_matmul_owned_output_compare.v1":
            errors.append(
                f"unexpected owned-output compare schema: {owned_output_compare.get('schema')!r}"
            )
        if owned_output_compare.get("qnn_used") is not False:
            errors.append("owned-output compare qnn_used is not false")
        if (
            owned_output_compare.get("acceptance_role")
            != "diagnostic_copy_smoke_not_milestone4_acceptance"
        ):
            errors.append("owned-output compare must stay diagnostic copy-smoke evidence")
        if owned_output_compare.get("compute_backend") != "copy_smoke":
            errors.append("owned-output compare must record compute_backend=copy_smoke")
        if owned_output_compare.get("hmx_body_entered") is not False:
            errors.append("owned-output compare must not claim HMX body entry")
        if owned_output_compare.get("accepted_for_milestone4_compute_gate") is not False:
            errors.append("owned-output compare must not be accepted for milestone4 compute gate")
        for section in ("output", "native_raw"):
            record = owned_output_compare.get(section, {})
            if not isinstance(record.get("bytes"), int) or record.get("bytes", 0) <= 0:
                errors.append(f"owned-output compare missing positive {section} byte count")
            if not isinstance(record.get("sha256"), str):
                errors.append(f"owned-output compare missing {section} sha256")
        if not isinstance(owned_output_compare.get("exact"), bool):
            errors.append("owned-output compare missing boolean exact")
        if not isinstance(owned_output_compare.get("byte_differences"), int):
            errors.append("owned-output compare missing byte_differences")
    if not abi_manifest_path.is_file():
        errors.append("missing analysis/abi_manifest.json")
    else:
        abi_manifest = load_json(abi_manifest_path)
        if abi_manifest.get("schema") != "handwritten_hmx_matmul_abi_manifest.v1":
            errors.append(f"unexpected ABI manifest schema: {abi_manifest.get('schema')!r}")
        if abi_manifest.get("qnn_used") is not False:
            errors.append("ABI manifest qnn_used is not false")
        call_abi = abi_manifest.get("hexagon_call_abi", {})
        registers = call_abi.get("registers", {})
        for reg in ("r0", "r1", "r2", "r3", "r4", "r5"):
            if reg not in registers:
                errors.append(f"ABI manifest missing register {reg}")
        for section in ("out_desc_fields", "act_desc_fields", "mask_desc", "extra_param_words"):
            if section not in call_abi:
                errors.append(f"ABI manifest missing call ABI section {section}")
        for section in ("activation_table", "output_table"):
            record = abi_manifest.get(section, {})
            if not isinstance(record.get("entry_count"), int) or record.get("entry_count") <= 0:
                errors.append(f"ABI manifest {section} missing positive entry_count")
    if not prep_profile_path.is_file():
        errors.append("missing analysis/prep_profile.json")
    else:
        prep_profile = load_json(prep_profile_path)
        if prep_profile.get("schema") != "handwritten_hmx_matmul_prep_profile.v1":
            errors.append(f"unexpected prep-profile schema: {prep_profile.get('schema')!r}")
        if prep_profile.get("qnn_used") is not False:
            errors.append("prep-profile qnn_used is not false")
        if prep_profile.get("compute_included") is not False:
            errors.append("prep-profile must keep compute_included=false")
        if not isinstance(prep_profile.get("total_us"), int) or prep_profile.get("total_us") < 0:
            errors.append("prep-profile missing non-negative total_us")
        stages = prep_profile.get("stages_us", {})
        for key in (
            "read_activation",
            "format_activation_surface",
            "load_logical_weight",
            "generate_packed_weight",
            "generate_folded_bias",
            "format_output_surface",
            "generate_descriptor_tables",
            "write_prepared_files",
        ):
            if key not in stages:
                errors.append(f"prep-profile missing stage {key}")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        errors.append("missing or empty device_out/out.raw")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--require-device", action="store_true")
    args = parser.parse_args()
    errors = check_artifact(Path(args.artifact).resolve(), args.require_device)
    if errors:
        print("handwritten runtime artifact: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"handwritten runtime artifact: ok ({args.artifact})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
