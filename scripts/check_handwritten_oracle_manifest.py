#!/usr/bin/env python3
"""Validate the handwritten HMX MatMul Milestone-0 oracle manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FAMILIES = {
    "u8i8",
    "w4a8",
    "w8a16",
    "w4a16",
    "w16a16",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def exists_rel(value: str | None) -> bool:
    return bool(value) and (ROOT / value).exists()


def check_file_record(errors: list[str], family: str, label: str, record: dict[str, Any]) -> None:
    if not exists_rel(record.get("path")):
        errors.append(f"{family}: missing {label}: {record.get('path')}")
        return
    if not record.get("bytes"):
        errors.append(f"{family}: {label} has no byte count")
    if not record.get("sha256"):
        errors.append(f"{family}: {label} has no sha256")


def check_payload_sources(errors: list[str], family: str, sources: dict[str, Any]) -> None:
    required = ["logical_weight_kn", "bias_q_int32", "effective_bias_int32", "quant_overrides"]
    if family == "w16a16":
        required.extend(["generated_sidecars", "run_profile"])
    for key in required:
        record = sources.get(key)
        if not isinstance(record, dict):
            errors.append(f"{family}: missing prepared payload source {key}")
            continue
        path = record.get("path")
        if not exists_rel(path):
            errors.append(f"{family}: prepared payload source {key} does not exist: {path}")


def check_compare(errors: list[str], family: str, compare: dict[str, Any]) -> None:
    if family == "w16a16":
        if compare.get("native_raw_exact_pass") is not True:
            errors.append(f"{family}: native_raw_exact_pass is not true")
        return
    if compare.get("exact") != compare.get("total"):
        errors.append(f"{family}: exact count does not equal total in compare record")
    if compare.get("maxdiff") != 0:
        errors.append(f"{family}: maxdiff is not zero in compare record")


def check_manifest(path: Path) -> list[str]:
    manifest = load_json(path)
    errors: list[str] = []
    if manifest.get("schema") != "handwritten_hmx_matmul_oracles.v1":
        errors.append(f"unexpected schema: {manifest.get('schema')!r}")
    families = manifest.get("families")
    if not isinstance(families, dict):
        return ["manifest families is not a dict"]
    missing = EXPECTED_FAMILIES - set(families)
    extra = set(families) - EXPECTED_FAMILIES
    if missing:
        errors.append(f"missing families: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected families: {sorted(extra)}")

    for family in sorted(EXPECTED_FAMILIES & set(families)):
        item = families[family]
        for key in (
            "native_artifact",
            "matched_custom_artifact",
            "dtype",
            "shape_mkn",
            "chain",
            "quantization_mode",
            "native_hmx_body",
        ):
            if item.get(key) in (None, "", []):
                errors.append(f"{family}: missing {key}")
        if not exists_rel(item.get("native_artifact")):
            errors.append(f"{family}: native artifact path does not exist")
        if not exists_rel(item.get("matched_custom_artifact")):
            errors.append(f"{family}: matched custom artifact path does not exist")

        checks = item.get("acceptance_checks", {})
        if checks.get("native_artifact_standard_passed") is not True:
            errors.append(f"{family}: native artifact standard did not pass")
        if checks.get("native_artifact_standard_errors"):
            errors.append(f"{family}: native artifact standard errors are present")
        if checks.get("stale_probe_used") is not False:
            errors.append(f"{family}: stale_probe_used is not false")
        if checks.get("native_kernel_node_count_matches") is not True:
            errors.append(f"{family}: native kernel node count mismatch")

        check_file_record(errors, family, "raw input", item.get("raw_input", {}))
        check_file_record(errors, family, "raw output", item.get("raw_output", {}))
        check_payload_sources(errors, family, item.get("prepared_payload_sources", {}))

        bottom = item.get("bottom_mapping", {})
        if not exists_rel(bottom.get("path")):
            errors.append(f"{family}: bottom mapping path missing")
        if not bottom.get("native_kernel_nodes"):
            errors.append(f"{family}: bottom mapping has no native kernel nodes")
        contract = item.get("native_compute_contract", {})
        contract_inputs = contract.get("inputs", {})
        for key in ("activation", "packed_weight", "folded_bias", "control"):
            tensor = contract_inputs.get(key)
            if not isinstance(tensor, dict) or not tensor.get("dims"):
                errors.append(f"{family}: native compute contract missing {key}")
        if not contract.get("outputs"):
            errors.append(f"{family}: native compute contract has no output tensor")

        scope = item.get("comparison_scope", {})
        raw_scope = scope.get("raw_output", {})
        kernel_scope = scope.get("native_kernel", {})
        aggregate_scope = scope.get("qnn_aggregate", {})
        timeline_scope = scope.get("timeline", {})
        if raw_scope.get("acceptance_role") != "owned_public_output_exactness_target":
            errors.append(f"{family}: comparison scope raw output role is missing")
        if raw_scope.get("bytes") != item.get("raw_output", {}).get("bytes"):
            errors.append(f"{family}: comparison scope raw output byte count mismatch")
        if kernel_scope.get("event_type") != "q::ConvLayer_s1.opt":
            errors.append(f"{family}: comparison scope kernel event type mismatch")
        if kernel_scope.get("event_count") != checks.get("expected_native_kernel_nodes"):
            errors.append(f"{family}: comparison scope kernel event count mismatch")
        if not kernel_scope.get("packets"):
            errors.append(f"{family}: comparison scope missing packet counts")
        if not aggregate_scope.get("prefix") or not aggregate_scope.get("cycles_sum"):
            errors.append(f"{family}: comparison scope missing QNN aggregate scope")
        if not timeline_scope.get("span_cycles"):
            errors.append(f"{family}: comparison scope missing timeline span")
        if family == "w16a16":
            outputs = kernel_scope.get("outputs", [])
            if scope.get("accepted_boundary_policy") != "single_custom_op_internal_split_n128":
                errors.append(f"{family}: comparison scope missing split boundary policy")
            if len(outputs) != 2 or any(record.get("bytes") != 65536 for record in outputs):
                errors.append(f"{family}: comparison scope must record two 65536-byte N128 kernel outputs")

        optrace = item.get("optrace", {})
        if not exists_rel(optrace.get("path")):
            errors.append(f"{family}: optrace summary path missing")
        if not optrace.get("native_kernel_cycles_sum"):
            errors.append(f"{family}: missing native kernel cycles")
        if not optrace.get("native_qnn_op_cycles_sum"):
            errors.append(f"{family}: missing native QNN aggregate cycles")
        if not optrace.get("native_kernel_packets"):
            errors.append(f"{family}: missing native packet counts")
        if not optrace.get("totals", {}).get("timeline_span_cycles"):
            errors.append(f"{family}: missing timeline span cycles")

        check_compare(errors, family, item.get("compare", {}))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        default="example/handwritten_hmx_matmul/oracles.json",
        help="Path to oracles.json",
    )
    args = parser.parse_args()
    path = (ROOT / args.manifest).resolve()
    errors = check_manifest(path)
    if errors:
        print("handwritten oracle manifest: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"handwritten oracle manifest: ok ({path.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
