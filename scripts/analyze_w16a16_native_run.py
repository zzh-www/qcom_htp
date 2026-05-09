#!/usr/bin/env python3
"""Summarize a W16A16 QNN-native oracle artifact.

The report is intentionally file-based.  It freezes the native runtime raw
contract, lowered HTP tensor surface, sidecar events, and optrace scopes so
custom W16A16 probes can compare against the same baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_mapping_json(out_dir: Path) -> Path | None:
    candidates = sorted((out_dir / "ctx").glob("*bottom_mapping.json"))
    return candidates[0] if candidates else None


def _find_schematic(out_dir: Path) -> Path | None:
    candidates = sorted((out_dir / "ctx").glob("*_schematic.bin"))
    return candidates[0] if candidates else None


def _scalar(node: dict[str, Any], name: str) -> Any:
    value = node.get("scalar_params", {}).get(name)
    if not isinstance(value, dict) or not value:
        return None
    return next(iter(value.values()))


def _producer_map(nodes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    producers: dict[str, dict[str, Any]] = {}
    for node_id, node in nodes.items():
        for tensor_id in node.get("output_names", []):
            producers[tensor_id] = {
                "node_id": node_id,
                "type": node.get("type"),
                "grouping": node.get("grouping"),
            }
    return producers


def _tensor_desc(
    tensors: dict[str, Any],
    tensor_id: str,
    producers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tensor = tensors.get(tensor_id, {})
    return {
        "tensor_id": tensor_id,
        "id": tensor.get("id"),
        "type": tensor.get("type"),
        "data_type": tensor.get("data_type"),
        "dims": tensor.get("dims"),
        "producer": producers.get(tensor_id),
    }


def _node_memory_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "mem_dram_read": _scalar(node, "mem_dram_read"),
        "mem_dram_write": _scalar(node, "mem_dram_write"),
        "mem_vtcm_read": _scalar(node, "mem_vtcm_read"),
        "mem_vtcm_write": _scalar(node, "mem_vtcm_write"),
        "op_flags": _scalar(node, "op_flags"),
        "output_step_size": _scalar(node, "output_step_size"),
        "output_zero_offset": _scalar(node, "output_zero_offset"),
        "output_rank": _scalar(node, "output_rank"),
    }


def _kernel_nodes(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    graph = mapping.get("graph", {})
    tensors = graph.get("tensors", {})
    nodes = graph.get("nodes", {})
    producers = _producer_map(nodes)
    roles = ["activation", "weight", "bias", "control", "extra_control"]
    kernels = []
    for node_id, node in nodes.items():
        if node.get("type") != "q::ConvLayer_s1.opt":
            continue
        inputs = {
            role: _tensor_desc(tensors, tensor_id, producers)
            for role, tensor_id in zip(roles, node.get("input_names", []))
        }
        outputs = {
            f"output_{idx}": _tensor_desc(tensors, tensor_id, producers)
            for idx, tensor_id in enumerate(node.get("output_names", []))
        }
        kernels.append({
            "node_id": node_id,
            "type": node.get("type"),
            "grouping": node.get("grouping"),
            "inputs": inputs,
            "outputs": outputs,
            "memory": _node_memory_summary(node),
        })
    return kernels


def _lowered_nodes(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    interesting = {
        "q::*InputSlice",
        "q::ForceFormat_Crouton",
        "q::ConvLayer.opt.weights_to_vtcm",
        "q::ConvLayer.opt.bias_to_vtcm",
        "q::ConvLayer_s1.opt",
        "q::Concat",
        "q::*OutputSlice",
    }
    graph = mapping.get("graph", {})
    tensors = graph.get("tensors", {})
    nodes = graph.get("nodes", {})
    producers = _producer_map(nodes)
    lowered = []
    for node_id, node in nodes.items():
        if node.get("type") not in interesting:
            continue
        lowered.append({
            "node_id": node_id,
            "type": node.get("type"),
            "grouping": node.get("grouping"),
            "inputs": [
                _tensor_desc(tensors, tensor_id, producers)
                for tensor_id in node.get("input_names", [])
            ],
            "outputs": [
                _tensor_desc(tensors, tensor_id, producers)
                for tensor_id in node.get("output_names", [])
            ],
            "memory": _node_memory_summary(node),
        })
    return lowered


def _optrace_summary(out_dir: Path) -> dict[str, Any]:
    summary_path = out_dir / "optrace" / "summary.json"
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "found": False}
    summary = _load_json(summary_path)
    kernel_events = [
        event for event in summary.get("events", [])
        if event.get("htp_type") == "q::ConvLayer_s1.opt"
    ]
    sidecar_types = {
        "q::ConvLayer.opt.weights_to_vtcm",
        "q::ConvLayer.opt.bias_to_vtcm",
        "q::ForceFormat_Crouton",
        "q::*InputSlice",
        "q::*OutputSlice",
        "q::Concat",
        "DmaCheckpointSet",
        "SyncOp",
    }
    sidecar_events = [
        event for event in summary.get("events", [])
        if event.get("htp_type") in sidecar_types
    ]
    return {
        "summary_path": str(summary_path),
        "found": True,
        "totals": summary.get("totals", {}),
        "by_htp_type_cycles": summary.get("by_htp_type_cycles", {}),
        "by_qnn_op_cycles": summary.get("by_qnn_op_cycles", {}),
        "kernel_events": kernel_events,
        "kernel_cycles_sum": int(sum(int(event.get("dur", 0)) for event in kernel_events)),
        "kernel_packets": [event.get("packets") for event in kernel_events],
        "sidecar_events": sidecar_events,
    }


def _schematic_sidecar_consts(out_dir: Path) -> dict[str, Any]:
    schematic = _find_schematic(out_dir)
    if schematic is None:
        return {"schematic_path": None, "sidecar_consts": []}
    text = schematic.read_text(encoding="utf-8", errors="replace")
    sidecars = []
    for match in re.finditer(r"\n\t(0x[0-9a-fA-F]+): \{\n(.*?)(?=\n\t0x[0-9a-fA-F]+: \{|\n\})", text, re.S):
        node_id, body = match.groups()
        if "'operation':'$Const'" not in body or "'splithist'" not in body:
            continue
        datalen_m = re.search(r"'datalen': ([0-9]+)", body)
        datalen = int(datalen_m.group(1)) if datalen_m else None
        if datalen not in (65536, 2048):
            continue
        hash_m = re.search(r"# CONTENT HASH: ([0-9a-fA-F]+)", body)
        dtype_m = re.search(r"'dtype': '([^']+)'", body)
        dims_m = re.search(r"'dims': \[([^\]]*)\]", body)
        split_m = re.search(r"'splithist' : \[([^\]]+)\]", body)
        sidecars.append({
            "node_id": node_id,
            "content_hash": hash_m.group(1) if hash_m else None,
            "datalen": datalen,
            "dtype": dtype_m.group(1) if dtype_m else None,
            "dims": [int(x.strip()) for x in dims_m.group(1).split(",") if x.strip()] if dims_m else None,
            "splithist": split_m.group(1).strip() if split_m else None,
        })
    return {
        "schematic_path": str(schematic),
        "sidecar_consts": sidecars,
    }


def analyze(out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    native_io_path = out_dir / "native_io.json"
    native_io = _load_json(native_io_path) if native_io_path.exists() else {}
    native_input = out_dir / native_io.get("native_input", "")
    native_output = out_dir / "device_out" / f"{native_io.get('output_name', 'Y')}.raw"
    if not native_output.exists():
        raws = sorted((out_dir / "device_out").glob("*.raw"))
        native_output = raws[0] if raws else native_output

    mapping_path = _find_mapping_json(out_dir)
    mapping = _load_json(mapping_path) if mapping_path else {}
    return {
        "out_dir": str(out_dir),
        "native_io_path": str(native_io_path),
        "native_io": native_io,
        "native_input": {
            "path": str(native_input),
            "bytes": native_input.stat().st_size if native_input.exists() else None,
            "sha256": _sha256(native_input),
        },
        "native_output": {
            "path": str(native_output),
            "bytes": native_output.stat().st_size if native_output.exists() else None,
            "sha256": _sha256(native_output),
        },
        "mapping_path": str(mapping_path) if mapping_path else None,
        "kernel_nodes": _kernel_nodes(mapping) if mapping else [],
        "lowered_nodes": _lowered_nodes(mapping) if mapping else [],
        "schematic": _schematic_sidecar_consts(out_dir),
        "optrace": _optrace_summary(out_dir),
    }


def _format_tensor(tensor: dict[str, Any]) -> str:
    producer = tensor.get("producer") or {}
    producer_text = ""
    if producer:
        producer_text = f" producer={producer.get('type')}:{producer.get('grouping')}"
    return (
        f"{tensor.get('data_type')}:{tensor.get('dims')}"
        f" id={tensor.get('tensor_id')}{producer_text}"
    )


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "w16a16_native_summary.json"
    txt_path = analysis_dir / "w16a16_native_summary.txt"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [f"W16A16 native summary: {report['out_dir']}"]
    native_io = report.get("native_io", {})
    lines.append(
        "native-io: "
        f"input={native_io.get('native_input_storage')} "
        f"{native_io.get('shape')} "
        f"output={native_io.get('expected_native_output_storage')} "
        f"{native_io.get('output_shape')}"
    )
    lines.append(
        "native-input: "
        f"bytes={report['native_input']['bytes']} "
        f"sha256={report['native_input']['sha256']}"
    )
    lines.append(
        "native-output: "
        f"bytes={report['native_output']['bytes']} "
        f"sha256={report['native_output']['sha256']}"
    )
    for idx, kernel in enumerate(report.get("kernel_nodes", [])):
        lines.append(f"kernel-node-{idx}: {kernel.get('node_id')} {kernel.get('type')}")
        for role in ("activation", "weight", "bias", "control", "extra_control"):
            tensor = kernel.get("inputs", {}).get(role)
            if tensor:
                lines.append(f"  input-{role}: {_format_tensor(tensor)}")
        for role, tensor in kernel.get("outputs", {}).items():
            lines.append(f"  {role}: {_format_tensor(tensor)}")
        mem = kernel.get("memory", {})
        lines.append(
            "  memory: "
            f"vtcm_read={mem.get('mem_vtcm_read')} "
            f"vtcm_write={mem.get('mem_vtcm_write')} "
            f"flags={mem.get('op_flags')}"
        )
    for const in report.get("schematic", {}).get("sidecar_consts", []):
        lines.append(
            "sidecar-const: "
            f"id={const.get('node_id')} dtype={const.get('dtype')} "
            f"dims={const.get('dims')} bytes={const.get('datalen')} "
            f"hash={const.get('content_hash')} split={const.get('splithist')}"
        )
    optrace = report.get("optrace", {})
    if optrace.get("found"):
        totals = optrace.get("totals", {})
        lines.append(
            "optrace: "
            f"kernel_sum={optrace.get('kernel_cycles_sum')} "
            f"packets={optrace.get('kernel_packets')} "
            f"matmul_1={optrace.get('by_qnn_op_cycles', {}).get('matmul_1')} "
            f"timeline={totals.get('timeline_span_cycles')} "
            f"sum_pid0={totals.get('sum_pid0_event_cycles')}"
        )
        for idx, event in enumerate(optrace.get("kernel_events", [])):
            lines.append(
                f"kernel-event-{idx}: "
                f"dur={event.get('dur')} packets={event.get('packets')} "
                f"cpp={event.get('cpp')} qnn_op={event.get('qnn_op')}"
            )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    report = analyze(args.out_dir)
    json_path, txt_path = write_reports(report, args.out_dir)
    print(f"  analysis json: {json_path}")
    print(f"  analysis txt: {txt_path}")
    for line in txt_path.read_text(encoding="utf-8").splitlines()[1:]:
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
