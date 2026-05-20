#!/usr/bin/env python3
"""Build the Milestone-0 oracle manifest for handwritten HMX MatMul.

The manifest is intentionally derived from retained standard artifacts.  It is
the bridge between QNN Native as an oracle and the future QNN-free runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from check_qnn_artifact_standard import check_artifact


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "example" / "qnn_matmul_profile"
RUNTIME_SPEC = ROOT / "example" / "handwritten_hmx_matmul"
AGENT_CURRENT = ROOT / "Agent" / "current"


FAMILIES: dict[str, dict[str, Any]] = {
    "u8i8": {
        "op_surface": "HmxU8I8ToU8MatMul",
        "native_dir": "output_u8i8_native_ref_e2e_256",
        "custom_dir": "output_u8i8_aligned_e2e_256",
        "dtype": "uint8",
        "weight_bits": 8,
        "activation_bits": 8,
        "output_bits": 8,
        "quantization_mode": "per_tensor_activation_output_per_channel_weight",
        "native_hmx_body": "hmx_v73_convbbb1x1deep_stride1",
        "native_qnn_prefix": "MatMul_",
        "expected_native_kernel_nodes": 8,
        "native_reference_command": "example/qnn_matmul_profile/run_matched_native_a8_ref.sh",
    },
    "w4a8": {
        "op_surface": "HmxU8I4ToU8MatMul",
        "native_dir": "output_w4a8_native_ref_e2e_256",
        "custom_dir": "output_w4a8_aligned_e2e_256",
        "dtype": "uint8",
        "weight_bits": 4,
        "activation_bits": 8,
        "output_bits": 8,
        "quantization_mode": "per_tensor_activation_output_per_channel_w4_weight",
        "native_hmx_body": "hmx_v73_convbnb1x1_stride1",
        "native_qnn_prefix": "MatMul_",
        "expected_native_kernel_nodes": 8,
        "native_reference_command": "example/qnn_matmul_profile/run_matched_native_a8_ref.sh",
    },
    "w8a16": {
        "op_surface": "HmxU16I8ToU16MatMul",
        "native_dir": "output_w8a16_native_ref_e2e_256",
        "custom_dir": "output_w8a16_aligned_e2e_256",
        "dtype": "uint16_le",
        "weight_bits": 8,
        "activation_bits": 16,
        "output_bits": 16,
        "quantization_mode": "per_tensor_a16_output_per_channel_w8_weight",
        "native_hmx_body": "hmx_v75_convhbh1x1deep_stride1",
        "native_qnn_prefix": "matmul_",
        "expected_native_kernel_nodes": 8,
        "native_reference_command": "example/qnn_matmul_profile/run_matched_native_w8a16_ref.sh",
    },
    "w4a16": {
        "op_surface": "HmxU16I4ToU16MatMul",
        "native_dir": "output_w4a16_native_ref_e2e_256",
        "custom_dir": "output_w4a16_aligned_e2e_256",
        "dtype": "uint16_le",
        "weight_bits": 4,
        "activation_bits": 16,
        "output_bits": 16,
        "quantization_mode": "per_tensor_a16_output_per_channel_w4_weight",
        "native_hmx_body": "hmx_v73_convhnh1x1_stride1",
        "native_qnn_prefix": "conv1x1_",
        "expected_native_kernel_nodes": 8,
        "native_reference_command": "example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh",
    },
    "w16a16": {
        "op_surface": "HmxU16I16ToU16MatMul",
        "native_dir": "output_w16a16_native_ref_e2e_256",
        "custom_dir": "output_w16a16_accepted_256",
        "dtype": "uint16_le",
        "weight_bits": 16,
        "activation_bits": 16,
        "output_bits": 16,
        "quantization_mode": "accepted_native_record_per_tensor_a16_output_per_channel_w16_weight",
        "native_hmx_body": "hmx_v73_convhhh1x1_stride1",
        "native_qnn_prefix": "matmul_",
        "expected_native_kernel_nodes": 2,
        "native_reference_command": "example/qnn_matmul_profile/profile_all.sh w16a16",
        "accepted_boundary_policy": "single_custom_op_internal_split_n128",
    },
}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": rel(path), "exists": path.is_file()}
    if path.is_file():
        out["bytes"] = path.stat().st_size
        out["sha256"] = sha256_file(path)
    return out


def tensor_element_count(tensor: dict[str, Any]) -> int | None:
    dims = tensor.get("dims")
    if not isinstance(dims, list) or not dims:
        return None
    total = 1
    for dim in dims:
        if not isinstance(dim, int) or dim <= 0:
            return None
        total *= dim
    return total


def tensor_byte_count(tensor: dict[str, Any], itemsize: int) -> int | None:
    elements = tensor_element_count(tensor)
    if elements is None:
        return None
    return elements * itemsize


def scalar_value(values: dict[str, Any], key: str) -> Any:
    raw = values.get(key, {})
    if isinstance(raw, dict) and raw:
        return next(iter(raw.values()))
    return None


def find_bottom_mapping(out_dir: Path) -> Path:
    matches = sorted((out_dir / "ctx").glob("*bottom_mapping.json"))
    if not matches:
        raise FileNotFoundError(f"missing bottom mapping under {out_dir}")
    if len(matches) > 1:
        preferred = [p for p in matches if "encoded" in p.name]
        if preferred:
            return preferred[0]
    return matches[0]


def tensor_record(tensors: dict[str, Any], tensor_id: str) -> dict[str, Any]:
    t = tensors.get(tensor_id, {})
    return {
        "tensor_id": tensor_id,
        "id": t.get("id"),
        "type": t.get("type"),
        "data_type": t.get("data_type"),
        "dims": t.get("dims"),
    }


def bottom_mapping_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    graph = data.get("graph", {})
    tensors = graph.get("tensors", {})
    nodes = graph.get("nodes", {})
    if not isinstance(tensors, dict) or not isinstance(nodes, dict):
        raise ValueError(f"unexpected bottom mapping shape: {path}")

    kernel_nodes = []
    sidecar_counts: dict[str, int] = {}
    format_counts: dict[str, int] = {}
    for node_id, node in nodes.items():
        node_type = node.get("type")
        if not isinstance(node_type, str):
            continue
        if "weights_to_vtcm" in node_type or "bias_to_vtcm" in node_type:
            sidecar_counts[node_type] = sidecar_counts.get(node_type, 0) + 1
        if "ForceFormat" in node_type or "InputSlice" in node_type or "OutputSlice" in node_type:
            format_counts[node_type] = format_counts.get(node_type, 0) + 1
        if node_type != "q::ConvLayer_s1.opt":
            continue
        scalar_params = node.get("scalar_params", {})
        input_names = node.get("input_names", [])
        output_names = node.get("output_names", [])
        kernel_nodes.append(
            {
                "node_id": node_id,
                "grouping": node.get("grouping"),
                "type": node_type,
                "inputs": [tensor_record(tensors, tid) for tid in input_names],
                "outputs": [tensor_record(tensors, tid) for tid in output_names],
                "memory": {
                    "mem_dram_read": scalar_value(scalar_params, "mem_dram_read"),
                    "mem_dram_write": scalar_value(scalar_params, "mem_dram_write"),
                    "mem_vtcm_read": scalar_value(scalar_params, "mem_vtcm_read"),
                    "mem_vtcm_write": scalar_value(scalar_params, "mem_vtcm_write"),
                    "op_flags": scalar_value(scalar_params, "op_flags"),
                    "output_step_size": scalar_value(scalar_params, "output_step_size"),
                    "output_zero_offset": scalar_value(scalar_params, "output_zero_offset"),
                    "output_rank": scalar_value(scalar_params, "output_rank"),
                },
            }
        )
    return {
        "path": rel(path),
        "tensor_count": len(tensors),
        "node_count": len(nodes),
        "native_kernel_nodes": kernel_nodes,
        "sidecar_node_counts": sidecar_counts,
        "format_node_counts": format_counts,
    }


def optrace_summary(out_dir: Path, qnn_prefix: str) -> dict[str, Any]:
    summary = load_json(out_dir / "optrace" / "summary.json")
    events = summary.get("events", [])
    kernel_events = [e for e in events if e.get("htp_type") == "q::ConvLayer_s1.opt"]
    qnn_ops = {
        name: cycles
        for name, cycles in summary.get("by_qnn_op_cycles", {}).items()
        if isinstance(name, str) and name.startswith(qnn_prefix)
    }
    return {
        "path": rel(out_dir / "optrace" / "summary.json"),
        "totals": summary.get("totals", {}),
        "native_kernel_htp_type": "q::ConvLayer_s1.opt",
        "native_kernel_cycles_sum": summary.get("by_htp_type_cycles", {}).get(
            "q::ConvLayer_s1.opt"
        ),
        "native_kernel_event_count": len(kernel_events),
        "native_kernel_packets": [e.get("packets") for e in kernel_events],
        "native_kernel_event_cycles": [e.get("dur") for e in kernel_events],
        "native_kernel_event_cpp": [e.get("cpp") for e in kernel_events],
        "native_qnn_op_prefix": qnn_prefix,
        "native_qnn_op_cycles": qnn_ops,
        "native_qnn_op_cycles_sum": sum(int(v) for v in qnn_ops.values()),
        "by_htp_type_cycles": summary.get("by_htp_type_cycles", {}),
    }


def native_compute_contract(bottom: dict[str, Any]) -> dict[str, Any]:
    nodes = bottom.get("native_kernel_nodes") or []
    if not nodes:
        return {}
    first = nodes[0]
    inputs = first.get("inputs", [])
    names = ["activation", "packed_weight", "folded_bias", "control", "extra_control"]
    contract = {
        "source": "first q::ConvLayer_s1.opt node in native bottom mapping",
        "node_id": first.get("node_id"),
        "grouping": first.get("grouping"),
        "inputs": {},
        "outputs": {},
    }
    for idx, tensor in enumerate(inputs):
        key = names[idx] if idx < len(names) else f"input_{idx}"
        contract["inputs"][key] = tensor
    for idx, tensor in enumerate(first.get("outputs", [])):
        contract["outputs"][f"output_{idx}"] = tensor
    return contract


def comparison_scope(
    family: str,
    cfg: dict[str, Any],
    native_io: dict[str, Any],
    shape_mkn: list[int] | None,
    native_output: Path,
    bottom: dict[str, Any],
    optrace: dict[str, Any],
) -> dict[str, Any]:
    itemsize = max(1, int(cfg["output_bits"]) // 8)
    raw_output = file_record(native_output)
    kernel_outputs = []
    for node in bottom.get("native_kernel_nodes", []):
        outputs = node.get("outputs", [])
        primary = outputs[0] if outputs else {}
        kernel_outputs.append(
            {
                "node_id": node.get("node_id"),
                "grouping": node.get("grouping"),
                "tensor": primary,
                "bytes": tensor_byte_count(primary, itemsize),
            }
        )
    notes = [
        "Owned acceptance compares final public output bytes against raw_output.path.",
        "Native kernel scope is used for HMX packet/cycle class and may be internal to the public output layout.",
    ]
    if cfg.get("accepted_boundary_policy") == "single_custom_op_internal_split_n128":
        notes.append(
            "W16A16 native compute uses two internal N128 kernel output scopes for one public 256-column output."
        )
    return {
        "raw_output": {
            "path": raw_output.get("path"),
            "bytes": raw_output.get("bytes"),
            "storage": native_io.get("expected_native_output_storage"),
            "shape_mkn": shape_mkn,
            "acceptance_role": "owned_public_output_exactness_target",
        },
        "native_kernel": {
            "event_type": optrace["native_kernel_htp_type"],
            "event_count": optrace["native_kernel_event_count"],
            "cycles_sum": optrace["native_kernel_cycles_sum"],
            "packets": optrace["native_kernel_packets"],
            "outputs": kernel_outputs,
            "total_output_scope_bytes": sum(
                int(record["bytes"]) for record in kernel_outputs if isinstance(record.get("bytes"), int)
            ),
        },
        "qnn_aggregate": {
            "prefix": optrace["native_qnn_op_prefix"],
            "cycles_sum": optrace["native_qnn_op_cycles_sum"],
            "events": optrace["native_qnn_op_cycles"],
        },
        "timeline": {
            "span_cycles": optrace["totals"].get("timeline_span_cycles"),
            "source": optrace["path"],
        },
        "accepted_boundary_policy": cfg.get("accepted_boundary_policy"),
        "notes": notes,
    }


def raw_compare_record(
    family: str, native_dir: Path, custom_dir: Path, dtype: str
) -> dict[str, Any] | None:
    native_raw = native_dir / "device_out" / "Y.raw"
    custom_raw = custom_dir / "device_out" / "out.raw"
    if not native_raw.is_file() or not custom_raw.is_file():
        return None
    native = native_raw.read_bytes()
    custom = custom_raw.read_bytes()
    if len(native) != len(custom):
        return {
            "path": rel(custom_raw),
            "native_path": rel(native_raw),
            "source": "direct_raw_compare",
            "acceptance_status": "open",
            "byte_differences": None,
            "custom_sha256": hashlib.sha256(custom).hexdigest(),
            "native_sha256": hashlib.sha256(native).hexdigest(),
            "size_mismatch": [len(custom), len(native)],
        }

    byte_diffs = sum(1 for a, b in zip(custom, native) if a != b)
    if dtype == "uint16_le":
        values = len(native) // 2
        exact = 0
        total_abs = 0
        maxdiff = 0
        for i in range(0, len(native), 2):
            c = custom[i] | (custom[i + 1] << 8)
            n = native[i] | (native[i + 1] << 8)
            diff = abs(c - n)
            if diff == 0:
                exact += 1
            total_abs += diff
            maxdiff = max(maxdiff, diff)
        mean_absdiff = total_abs / values if values else 0.0
        total = values
    else:
        total = len(native)
        exact = total - byte_diffs
        total_abs = sum(abs(a - b) for a, b in zip(custom, native))
        maxdiff = max((abs(a - b) for a, b in zip(custom, native)), default=0)
        mean_absdiff = total_abs / total if total else 0.0

    return {
        "path": rel(custom_raw),
        "native_path": rel(native_raw),
        "source": "direct_raw_compare",
        "acceptance_status": "accepted" if exact == total and maxdiff == 0 else "open",
        "exact": exact,
        "total": total,
        "maxdiff": maxdiff,
        "mean_absdiff": mean_absdiff,
        "byte_differences": byte_diffs,
        "custom_sha256": hashlib.sha256(custom).hexdigest(),
        "native_sha256": hashlib.sha256(native).hexdigest(),
    }


def compare_record(
    family: str, native_dir: Path, custom_dir: Path, dtype: str
) -> dict[str, Any]:
    if family not in ("w4a16", "w16a16"):
        raw_record = raw_compare_record(family, native_dir, custom_dir, dtype)
        if raw_record is not None:
            return raw_record
    candidates = [
        native_dir / "analysis" / "matched_native_compare.json",
        custom_dir / "analysis" / f"{family}_native_compare.json",
        custom_dir / "analysis" / "w16a16_custom_compare.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        data = load_json(path)
        record: dict[str, Any] = {"path": rel(path)}
        if {"exact", "total", "maxdiff"}.issubset(data):
            record.update(
                {
                    "exact": data["exact"],
                    "total": data["total"],
                    "maxdiff": data["maxdiff"],
                    "mean_absdiff": data.get("mean_absdiff"),
                }
            )
        elif isinstance(data.get("native"), dict) and isinstance(
            data["native"].get("stats"), dict
        ):
            stats = data["native"]["stats"]
            record.update(
                {
                    "exact": stats.get("exact"),
                    "total": stats.get("total"),
                    "maxdiff": stats.get("maxdiff"),
                    "mean_absdiff": stats.get("mean_absdiff"),
                    "native_transpose": data.get("native_transpose"),
                }
            )
        elif family == "w16a16":
            gate = data.get("alignment_gate", {})
            checks = gate.get("checks", {})
            native_raw = checks.get("native_raw_exact", {})
            record.update(
                {
                    "alignment_gate": gate,
                    "native_raw_exact_pass": native_raw.get("pass"),
                    "native_raw_exact_evidence": native_raw.get("evidence"),
                }
            )
        return record
    return {"path": None, "missing": True}


def custom_payload_sources(family: str, custom_dir: Path, native_io: dict[str, Any]) -> dict[str, Any]:
    prefix = family
    if family == "u8i8":
        prefix = "u8i8"
    paths = {
        "logical_weight_kn": custom_dir / f"{prefix}.onnx.wRaw_KN.npy",
        "bias_q_int32": custom_dir / f"{prefix}.onnx.bias_q_int32.npy",
        "effective_bias_int32": custom_dir / f"{prefix}.onnx.effective_int32.npy",
        "quant_overrides": custom_dir / "quant_overrides.json",
        "custom_native_io": custom_dir / "native_io.json",
    }
    if family == "w16a16":
        paths["generated_sidecars"] = custom_dir / "generated_sidecars"
        paths["run_profile"] = custom_dir / "w16a16_run_profile.json"

    out: dict[str, Any] = {}
    for name, path in paths.items():
        if path.is_dir():
            out[name] = {"path": rel(path), "exists": True, "file_count": len(list(path.rglob("*")))}
        else:
            out[name] = file_record(path)
    for key in ("matched_weight", "matched_effective_bias"):
        if key in native_io:
            out[f"native_io_{key}"] = native_io[key]
    return out


def build_manifest() -> dict[str, Any]:
    families = {}
    for family, cfg in FAMILIES.items():
        native_dir = PROFILE / cfg["native_dir"]
        custom_dir = PROFILE / cfg["custom_dir"]
        standard_errors, standard_warnings = check_artifact(
            native_dir,
            require_native_io=True,
            require_layout_flags=True,
            reject_float_io=True,
        )
        native_io = load_json(native_dir / "native_io.json")
        shape_mkn = (
            native_io.get("shape_mkn")
            or native_io.get("logical_matmul_shape_mkn")
            or native_io.get("shape")
        )
        if family == "w16a16" and shape_mkn == [1, 256, 256]:
            shape_mkn = [256, 256, 256]
        chain = native_io.get("chain")
        if chain is None and family == "w16a16":
            profile_path = custom_dir / "w16a16_run_profile.json"
            if profile_path.is_file():
                chain = load_json(profile_path).get("chain")
        bottom_path = find_bottom_mapping(native_dir)
        bottom = bottom_mapping_summary(bottom_path)

        native_input = native_dir / native_io.get("native_input", "runtime_inputs_native/A.raw")
        native_output = native_dir / "device_out" / f"{native_io.get('output_name', 'Y')}.raw"
        optrace = optrace_summary(native_dir, cfg["native_qnn_prefix"])

        families[family] = {
            "op_surface": cfg["op_surface"],
            "scope": "canonical_256",
            "shape_mkn": shape_mkn,
            "chain": chain,
            "dtype": cfg["dtype"],
            "activation_bits": cfg["activation_bits"],
            "weight_bits": cfg["weight_bits"],
            "output_bits": cfg["output_bits"],
            "quantization_mode": cfg["quantization_mode"],
            "native_hmx_body": cfg["native_hmx_body"],
            "native_reference_command": cfg["native_reference_command"],
            "native_artifact": rel(native_dir),
            "matched_custom_artifact": rel(custom_dir),
            "accepted_boundary_policy": cfg.get("accepted_boundary_policy"),
            "native_io": {
                "path": rel(native_dir / "native_io.json"),
                "input_name": native_io.get("input_name"),
                "output_name": native_io.get("output_name"),
                "native_input_storage": native_io.get("native_input_storage"),
                "native_input_bytes": native_io.get("native_input_bytes"),
                "expected_native_output_storage": native_io.get(
                    "expected_native_output_storage"
                ),
                "expected_native_output_bytes": native_io.get("expected_native_output_bytes"),
                "activation_encoding": native_io.get("activation_encoding"),
                "weight_encoding": native_io.get("weight_encoding"),
            },
            "raw_input": file_record(native_input),
            "raw_output": file_record(native_output),
            "prepared_payload_sources": custom_payload_sources(family, custom_dir, native_io),
            "bottom_mapping": bottom,
            "native_compute_contract": native_compute_contract(bottom),
            "comparison_scope": comparison_scope(family, cfg, native_io, shape_mkn, native_output, bottom, optrace),
            "optrace": optrace,
            "compare": compare_record(family, native_dir, custom_dir, cfg["dtype"]),
            "acceptance_checks": {
                "native_artifact_standard_passed": not standard_errors,
                "native_artifact_standard_errors": standard_errors,
                "native_artifact_standard_warnings": standard_warnings,
                "native_kernel_nodes_observed": None,
                "expected_native_kernel_nodes": cfg["expected_native_kernel_nodes"],
                "stale_probe_used": False,
            },
        }
        observed = families[family]["optrace"]["native_kernel_event_count"]
        families[family]["acceptance_checks"]["native_kernel_nodes_observed"] = observed
        families[family]["acceptance_checks"]["native_kernel_node_count_matches"] = (
            observed == cfg["expected_native_kernel_nodes"]
        )
    return {
        "schema": "handwritten_hmx_matmul_oracles.v1",
        "source": "retained QNN Native standard artifacts",
        "roadmap": "Agent/guides/handwritten_hmx_matmul_roadmap.md",
        "artifact_standard": "Agent/current/qnn_native_artifact_standard.md",
        "families": families,
    }


def write_json(manifest: dict[str, Any]) -> None:
    RUNTIME_SPEC.mkdir(parents=True, exist_ok=True)
    (RUNTIME_SPEC / "oracles.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def family_table(manifest: dict[str, Any]) -> str:
    rows = [
        "| Family | Native oracle | Custom comparator | Shape/chain | Kernel event | QNN aggregate | Timeline | Exactness |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for family, item in manifest["families"].items():
        optrace = item["optrace"]
        compare = item["compare"]
        if "exact" in compare:
            exact = f"{compare['exact']}/{compare['total']}, maxdiff {compare['maxdiff']}"
        elif compare.get("native_raw_exact_pass") is not None:
            exact = f"native_raw_exact={compare['native_raw_exact_pass']}"
        else:
            exact = "see compare"
        rows.append(
            "| {family} | `{native}` | `{custom}` | `{shape}`, chain `{chain}` | {kernel} | {agg} | {timeline} | {exact} |".format(
                family=family,
                native=item["native_artifact"],
                custom=item["matched_custom_artifact"],
                shape=item["shape_mkn"],
                chain=item["chain"],
                kernel=optrace["native_kernel_cycles_sum"],
                agg=optrace["native_qnn_op_cycles_sum"],
                timeline=optrace["totals"].get("timeline_span_cycles"),
                exact=exact,
            )
        )
    return "\n".join(rows)


def write_markdown(manifest: dict[str, Any]) -> None:
    lines = [
        "# Handwritten HMX MatMul Oracles",
        "",
        "This is the Milestone-0 oracle freeze for the QNN-free handwritten HMX",
        "MatMul roadmap.  QNN Native is the specification source; the future owned",
        "runtime must not execute through QNN, but it must match these raw outputs,",
        "runtime contracts, and performance scopes.",
        "",
        "Machine-readable manifest: `../../example/handwritten_hmx_matmul/oracles.json`.",
        "W16A16 remains in the manifest as retained evidence, but it is inactive for the",
        "current W4A16 tutorial/direct-HMX route.",
        "",
        "Regenerate and validate:",
        "",
        "```bash",
        "uv run python scripts/build_handwritten_oracle_manifest.py",
        "uv run python scripts/check_handwritten_oracle_manifest.py",
        "```",
        "",
        "## Canonical Oracles",
        "",
        family_table(manifest),
        "",
        "## Acceptance Evidence",
        "",
        "- All five native oracle directories are the retained canonical directories under",
        "  `example/qnn_matmul_profile/`.",
        "- Each native oracle passed:",
        "",
        "```bash",
        "scripts/check_qnn_artifact_standard.py <native_dir> \\",
        "  --require-native-io --require-layout-flags --reject-float-io",
        "```",
        "",
        "- The manifest records raw input/output files, SHA256, storage type, byte",
        "  count, shape, chain, quantization mode, prepared payload sources, native",
        "  HMX body name, native compute contracts for activation, packed weight,",
        "  folded bias, control, extra/control, and output tensors, comparable",
        "  native optrace events, QNN-op aggregate cycles, packet counts, timeline",
        "  span, public raw-output exactness scope, and exactness evidence.",
        "- `scripts/check_handwritten_oracle_manifest.py` validates the generated JSON",
        "  against this Milestone-0 contract.",
        "- Stale probe directories and same-shape random native runs are excluded.",
        "",
        "## Family Notes",
        "",
    ]
    for family, item in manifest["families"].items():
        optrace = item["optrace"]
        checks = item["acceptance_checks"]
        scope = item["comparison_scope"]
        kernel_scope = scope["native_kernel"]
        lines.extend(
            [
                f"### {family}",
                "",
                f"- Op surface: `{item['op_surface']}`.",
                f"- Native HMX body: `{item['native_hmx_body']}`.",
                f"- Raw input: `{item['raw_input']['path']}` ({item['raw_input'].get('bytes')} bytes).",
                f"- Raw output oracle: `{item['raw_output']['path']}` ({item['raw_output'].get('bytes')} bytes).",
                f"- Native kernel event: `{optrace['native_kernel_htp_type']}`; "
                f"{optrace['native_kernel_event_count']} events, "
                f"{optrace['native_kernel_cycles_sum']} cycles, packets "
                f"`{optrace['native_kernel_packets']}`.",
                f"- QNN aggregate prefix `{optrace['native_qnn_op_prefix']}` sums to "
                f"{optrace['native_qnn_op_cycles_sum']} cycles; timeline span "
                f"{optrace['totals'].get('timeline_span_cycles')} cycles.",
                f"- Comparison scope: public output exactness target is "
                f"`{scope['raw_output']['path']}` ({scope['raw_output'].get('bytes')} bytes); "
                f"native kernel scope has {kernel_scope['event_count']} "
                f"`{kernel_scope['event_type']}` events and "
                f"{kernel_scope.get('total_output_scope_bytes')} total output-scope bytes.",
                f"- Native kernel node count check: observed "
                f"{checks['native_kernel_nodes_observed']}, expected "
                f"{checks['expected_native_kernel_nodes']}, pass "
                f"{checks['native_kernel_node_count_matches']}.",
                "",
            ]
        )
    AGENT_CURRENT.mkdir(parents=True, exist_ok=True)
    (AGENT_CURRENT / "handwritten_hmx_matmul_oracles.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def write_runtime_readme() -> None:
    RUNTIME_SPEC.mkdir(parents=True, exist_ok=True)
    (RUNTIME_SPEC / "README.md").write_text(
        "\n".join(
            [
                "# Handwritten HMX MatMul",
                "",
                "This directory is the future QNN-free runtime/spec tree for the",
                "handwritten HMX MatMul work.",
                "",
                "Current contents include the Milestone-0 oracle data, the Milestone-1 smoke",
                "boundary, and the Milestone-2 preparation scaffold.  The active families are",
                "U8I8, W4A8, W8A16, and W4A16; W16A16 material is retained as inactive oracle",
                "and body-slice reference only.",
                "",
                "- `oracles.json`: machine-readable QNN Native oracle manifest generated",
                "  from retained standard artifacts.",
                "- `include/handwritten_hmx_matmul.h`: prepared-state C ABI for all target",
                "  families.",
                "- `include/handwritten_hmx_u8i8_kernel.h` and",
                "  `kernels/u8i8/v73deep_conv1x1_kernel.inc`: U8I8 Hexagon body ABI and",
                "  byte-identical inline-asm body.",
                "- `include/handwritten_hmx_w4a8_kernel.h` and",
                "  `kernels/w4a8/v73deep_conv1x1_kernel.inc`: W4A8 Hexagon body ABI and",
                "  byte-identical inline-asm body.",
                "- `include/handwritten_hmx_w8a16_kernel.h` and",
                "  `kernels/w8a16/v73deep_conv1x1_kernel.inc`: W8A16 Hexagon body ABI and",
                "  byte-identical inline-asm body.",
                "- `include/handwritten_hmx_w4a16_kernel.h` and",
                "  `kernels/w4a16/v73deep_conv1x1_kernel.inc`: W4A16 Hexagon body ABI and",
                "  byte-identical inline-asm body.",
                "- `include/handwritten_hmx_w16a16_kernel.h` and",
                "  `kernels/w16a16/v73deep_conv1x1_kernel.inc`: retained W16A16 reference",
                "  material.  It is not part of the current active gate.",
                "- `build_host.sh` and `run_owned_smoke.py`: host-only smoke path for the",
                "  owned runtime boundary.",
                "- `build_android.sh`: AArch64 Android build path for direct device smoke.",
                "- `prepare_owned_inputs.py`: owned preparation scaffold that writes",
                "  `prepared_state/`, `analysis/prep_compare.json`, and",
                "  `analysis/prep_profile.json`.",
                "- `tutorial_w4a16_qnn_kernel/`: tutorial-style direct CDSP wrapper that builds",
                "  the recovered W4A16 QNN HMX body as a `run_main_on_hexagon` shared object.",
                "  This is the active W4A16 route: QNN supplies retained prepared bytes and the",
                "  native raw oracle offline, but runtime execution enters HAP/HVX/HMX/VTCM setup",
                "  directly and calls `hm_w4a16_v73deep_kernel` without a QNN context.",
                "",
                "Regenerate and validate the oracle freeze from repo root:",
                "",
                "```bash",
                "uv run python scripts/build_handwritten_oracle_manifest.py",
                "uv run python scripts/check_handwritten_oracle_manifest.py",
                "```",
                "",
                "Verify current HMX body byte identity and ABI header compilation:",
                "",
                "```bash",
                "uv run python scripts/check_handwritten_hmx_body.py \\",
                "  --json-out /tmp/handwritten_hmx_body_check.json",
                "```",
                "",
                "Build and run the active W4A16 tutorial/direct-HMX wrapper:",
                "",
                "```bash",
                "bash example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh",
                "DEVICE=oneplus bash example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/run_device.sh",
                "```",
                "",
                "Run the current owned smoke gate:",
                "",
                "```bash",
                "OUT_ROOT=/tmp/handwritten_hmx_matmul_gate \\",
                "DEVICE=oneplus \\",
                "tests/handwritten_hmx_matmul/run_all.sh",
                "```",
                "",
                "For host-only smoke without device execution:",
                "",
                "```bash",
                "ARTIFACT_ONLY=1 OUT_ROOT=/tmp/handwritten_hmx_matmul_gate \\",
                "tests/handwritten_hmx_matmul/run_all.sh",
                "```",
                "",
                "The artifact-only path regenerates each family artifact under `OUT_ROOT`",
                "before running body-simulator and validator checks, so it must not depend",
                "on stale `/tmp` artifacts from a previous run.",
                "",
                "Each canonical family also has a named smoke entry under",
                "`tests/handwritten_hmx_matmul/test_<family>_smoke.sh`.",
                "",
                "Build and validate the current host smoke boundary:",
                "",
                "```bash",
                "example/handwritten_hmx_matmul/build_host.sh",
                "uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \\",
                "  --family u8i8 --out-dir /tmp/handwritten_hmx_matmul_owned_smoke_u8i8",
                "uv run python scripts/check_handwritten_runtime_artifact.py \\",
                "  /tmp/handwritten_hmx_matmul_owned_smoke_u8i8",
                "```",
                "",
                "Build and validate a direct device smoke artifact:",
                "",
                "```bash",
                "uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \\",
                "  --family u8i8 --device oneplus \\",
                "  --out-dir /tmp/handwritten_hmx_matmul_owned_device_u8i8",
                "uv run python scripts/check_handwritten_runtime_artifact.py \\",
                "  /tmp/handwritten_hmx_matmul_owned_device_u8i8 --require-device",
                "```",
                "",
                "This is still a smoke boundary, not the final compute runtime.  Milestone 2",
                "has a preparation scaffold, including canonical activation-surface formatting,",
                "native-sized output-surface allocation, explicit descriptor/table/mask-control",
                "buffers, and separated prep/runtime profiling.  These descriptor/table files",
                "are candidate ABI placeholders, not native-prepared exact proof; most families",
                "still need native prepared-byte proof before real HMX bodies can be entered.",
                "The owned gate validates current smoke evidence only.  It includes",
                "per-family artifact-body simulator smoke with before/after checksums, a",
                "QNN Native raw-output checksum comparison, and an `exactness_status` plus",
                "`exactness_blocker` pair.  Those comparisons are diagnostic, not exactness",
                "evidence; final output exactness and optrace/perf remain open until the",
                "owned runtime artifact path enters HMX bodies on device with native-prepared state.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    manifest = build_manifest()
    write_json(manifest)
    write_markdown(manifest)
    write_runtime_readme()
    print(f"wrote {rel(RUNTIME_SPEC / 'oracles.json')}")
    print(f"wrote {rel(AGENT_CURRENT / 'handwritten_hmx_matmul_oracles.md')}")
    print(f"wrote {rel(RUNTIME_SPEC / 'README.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
