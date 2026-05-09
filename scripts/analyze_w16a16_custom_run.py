#!/usr/bin/env python3
"""Analyze a W16A16 custom run against a QNN-native W16A16 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_mapping_json(out_dir: Path) -> Path | None:
    candidates = sorted((out_dir / "ctx").glob("*bottom_mapping.json"))
    return candidates[0] if candidates else None


def _find_output_raw(out_dir: Path) -> Path:
    for name in ("out.raw", "Y.raw"):
        path = out_dir / "device_out" / name
        if path.exists():
            return path
    raws = sorted((out_dir / "device_out").glob("*.raw"))
    return raws[0] if raws else out_dir / "device_out" / "out.raw"


def _find_reference(out_dir: Path) -> Path | None:
    refs = sorted(out_dir.glob("*.out_ref_u*.npy"))
    return refs[0] if refs else None


def _load_run_profile(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "w16a16_run_profile.json"
    if not path.exists():
        return {"profile_path": str(path), "found": False}
    data = _load_json(path)
    data["profile_path"] = str(path)
    data["found"] = True
    return data


def _load_output_encoding(out_dir: Path) -> tuple[float, float]:
    overrides = out_dir / "quant_overrides.json"
    if not overrides.exists():
        return 1.0, 0.0
    encodings = _load_json(overrides).get("activation_encodings", {})
    if "out" not in encodings or not encodings["out"]:
        return 1.0, 0.0
    enc = encodings["out"][0]
    return float(enc.get("scale", 1.0)), float(enc.get("offset", 0.0))


def _load_quantized(path: Path, dtype: np.dtype, shape: tuple[int, ...], scale: float, offset: float) -> np.ndarray:
    raw_size = path.stat().st_size
    q_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    f_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if raw_size == q_bytes:
        return np.fromfile(path, dtype=dtype).reshape(shape)
    if raw_size != f_bytes:
        raise ValueError(f"{path} has unexpected byte count {raw_size}")
    if scale == 0.0:
        raise ValueError("cannot quantize float output with zero scale")
    max_value = np.iinfo(dtype).max
    out = np.fromfile(path, dtype=np.float32).reshape(shape)
    return np.clip(np.rint(out / scale - offset), 0, max_value).astype(dtype)


def _pair_stats(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    diff = np.abs(lhs.astype(np.int64) - rhs.astype(np.int64))
    return {
        "exact": int((lhs == rhs).sum()),
        "total": int(lhs.size),
        "maxdiff": int(diff.max()) if diff.size else 0,
        "mean_absdiff": float(diff.mean()) if diff.size else 0.0,
        "sorted_equal": bool(np.array_equal(np.sort(lhs.reshape(-1)), np.sort(rhs.reshape(-1)))),
    }


def _producer_map(nodes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    producers = {}
    for node_id, node in nodes.items():
        for tensor_id in node.get("output_names", []):
            producers[tensor_id] = {
                "node_id": node_id,
                "type": node.get("type"),
                "grouping": node.get("grouping"),
            }
    return producers


def _tensor_desc(tensors: dict[str, Any], producers: dict[str, dict[str, Any]], tensor_id: str) -> dict[str, Any]:
    tensor = tensors.get(tensor_id, {})
    return {
        "tensor_id": tensor_id,
        "id": tensor.get("id"),
        "type": tensor.get("type"),
        "data_type": tensor.get("data_type"),
        "dims": tensor.get("dims"),
        "producer": producers.get(tensor_id),
    }


def _custom_boundaries(out_dir: Path) -> list[dict[str, Any]]:
    mapping = _find_mapping_json(out_dir)
    if mapping is None:
        return []
    data = _load_json(mapping)
    graph = data.get("graph", {})
    tensors = graph.get("tensors", {})
    nodes = graph.get("nodes", {})
    producers = _producer_map(nodes)
    boundaries = []
    for node_id, node in nodes.items():
        if "HmxU16I16ToU16MatMul" not in str(node.get("type", "")):
            continue
        roles = ["bias", "weight", "activation", "scratch"]
        boundaries.append({
            "mapping": str(mapping),
            "node_id": node_id,
            "node_type": node.get("type"),
            "grouping": node.get("grouping"),
            "inputs": {
                role: _tensor_desc(tensors, producers, tensor_id)
                for role, tensor_id in zip(roles, node.get("input_names", []))
            },
            "outputs": {
                f"output_{idx}": _tensor_desc(tensors, producers, tensor_id)
                for idx, tensor_id in enumerate(node.get("output_names", []))
            },
        })
    return boundaries


def _native_boundary(native_out_dir: Path) -> list[dict[str, Any]]:
    mapping = _find_mapping_json(native_out_dir)
    if mapping is None:
        return []
    data = _load_json(mapping)
    graph = data.get("graph", {})
    tensors = graph.get("tensors", {})
    nodes = graph.get("nodes", {})
    producers = _producer_map(nodes)
    roles = ["activation", "weight", "bias", "control", "extra_control"]
    kernels = []
    for node_id, node in nodes.items():
        if node.get("type") != "q::ConvLayer_s1.opt":
            continue
        kernels.append({
            "mapping": str(mapping),
            "node_id": node_id,
            "node_type": node.get("type"),
            "grouping": node.get("grouping"),
            "inputs": {
                role: _tensor_desc(tensors, producers, tensor_id)
                for role, tensor_id in zip(roles, node.get("input_names", []))
            },
            "outputs": {
                f"output_{idx}": _tensor_desc(tensors, producers, tensor_id)
                for idx, tensor_id in enumerate(node.get("output_names", []))
            },
        })
    return kernels


def _optrace(out_dir: Path, custom_substr: str) -> dict[str, Any] | None:
    path = out_dir / "optrace" / "summary.json"
    if not path.exists():
        return None
    summary = _load_json(path)
    events = summary.get("events", [])
    custom_events = [
        event for event in events
        if custom_substr in str(event.get("htp_type", "")) or custom_substr in str(event.get("qnn_op", ""))
    ]
    return {
        "totals": summary.get("totals", {}),
        "by_htp_type_cycles": summary.get("by_htp_type_cycles", {}),
        "by_qnn_op_cycles": summary.get("by_qnn_op_cycles", {}),
        "custom_events": custom_events,
        "custom_cycles_sum": int(sum(int(event.get("dur", 0)) for event in custom_events)),
        "custom_packets": [event.get("packets") for event in custom_events],
    }


def _same_tensor_surface(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None) -> bool:
    if not lhs or not rhs:
        return False
    return lhs.get("data_type") == rhs.get("data_type") and lhs.get("dims") == rhs.get("dims")


def _alignment_gate(report: dict[str, Any]) -> dict[str, Any]:
    native = report.get("native") or {}
    custom_perf = report.get("custom_optrace") or {}
    native_perf = report.get("native_optrace") or {}
    run_profile = report.get("run_profile") or {}
    custom_boundaries = report.get("custom_boundaries") or []
    native_boundaries = report.get("native_boundary") or []

    custom_cycles = custom_perf.get("custom_cycles_sum")
    native_cycles = native_perf.get("custom_cycles_sum")
    custom_packets = custom_perf.get("custom_packets") or []
    native_packets = native_perf.get("custom_packets") or []
    custom_packet_sum = sum(int(p) for p in custom_packets if p is not None)
    native_packet_sum = sum(int(p) for p in native_packets if p is not None)

    native_exact = bool(native and native.get("exact") == native.get("total") and native.get("maxdiff") == 0)
    profile_name = str(run_profile.get("kernel_profile") or "")
    diagnostic_profile = profile_name not in ("", "skip", "accepted")

    boundary_match = False
    if len(custom_boundaries) == len(native_boundaries) and custom_boundaries:
        boundary_match = True
        for custom, native_kernel in zip(custom_boundaries, native_boundaries):
            c_inputs = custom.get("inputs", {})
            n_inputs = native_kernel.get("inputs", {})
            c_outputs = custom.get("outputs", {})
            n_outputs = native_kernel.get("outputs", {})
            for role in ("activation", "weight", "bias"):
                boundary_match = boundary_match and _same_tensor_surface(
                    c_inputs.get(role), n_inputs.get(role))
            boundary_match = boundary_match and _same_tensor_surface(
                c_outputs.get("output_0"), n_outputs.get("output_0"))
    boundary_policy = str(run_profile.get("boundary_policy") or "")
    boundary_justified = (
        profile_name == "accepted"
        and boundary_policy == "single_custom_op_internal_split_n128"
        and len(custom_boundaries) == 1
        and len(native_boundaries) == 2
    )
    boundary_pass = boundary_match or boundary_justified

    packet_delta = custom_packet_sum - native_packet_sum if custom_packet_sum and native_packet_sum else None
    cycle_delta = (
        int(custom_cycles) - int(native_cycles)
        if custom_cycles is not None and native_cycles is not None else None
    )

    checks = {
        "native_raw_exact": {
            "pass": native_exact,
            "evidence": f"{native.get('exact')}/{native.get('total')} maxdiff={native.get('maxdiff')}" if native else "missing native compare",
        },
        "default_acceptance_profile": {
            "pass": profile_name == "accepted",
            "evidence": profile_name or "no profile metadata",
        },
        "same_or_justified_boundary": {
            "pass": boundary_pass,
            "evidence": (
                f"custom_boundaries={len(custom_boundaries)} native_boundaries={len(native_boundaries)} "
                f"policy={boundary_policy or 'none'}"
            ),
        },
        "native_packet_budget": {
            "pass": packet_delta is not None and packet_delta <= 0,
            "evidence": f"custom={custom_packet_sum} native={native_packet_sum} delta={packet_delta}",
        },
        "native_cycle_budget": {
            "pass": cycle_delta is not None and cycle_delta <= 0,
            "evidence": f"custom={custom_cycles} native={native_cycles} delta={cycle_delta}",
        },
        "diagnostic_profile_only": {
            "pass": not diagnostic_profile,
            "evidence": profile_name or "none",
        },
    }
    return {
        "accepted": all(item["pass"] for item in checks.values()),
        "checks": checks,
        "cycle_delta": cycle_delta,
        "packet_delta": packet_delta,
    }


def analyze(out_dir: Path, native_out_dir: Path | None, native_raw: Path | None) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    scale, offset = _load_output_encoding(out_dir)
    report: dict[str, Any] = {
        "out_dir": str(out_dir),
        "output_raw": str(_find_output_raw(out_dir)),
        "reference": str(_find_reference(out_dir)) if _find_reference(out_dir) else None,
        "run_profile": _load_run_profile(out_dir),
        "custom_boundaries": _custom_boundaries(out_dir),
        "custom_optrace": _optrace(out_dir, "HmxU16I16ToU16MatMul"),
    }
    if native_out_dir:
        report["native_out_dir"] = str(native_out_dir.resolve())
        report["native_boundary"] = _native_boundary(native_out_dir.resolve())
        native_optrace = _optrace(native_out_dir.resolve(), "ConvLayer_s1.opt")
        if native_optrace:
            report["native_optrace"] = native_optrace
    ref_path = _find_reference(out_dir)
    out_raw = _find_output_raw(out_dir)
    try:
        if ref_path and out_raw.exists():
            ref = np.load(ref_path)
            out_q = _load_quantized(out_raw, ref.dtype, ref.shape, scale, offset)
            report["analytic"] = _pair_stats(out_q, ref)
            if native_raw:
                native_q = _load_quantized(native_raw.resolve(), ref.dtype, ref.shape, scale, offset)
                report["native_raw"] = str(native_raw.resolve())
                report["native"] = _pair_stats(out_q, native_q)
    except Exception as exc:
        report["compare_error"] = str(exc)
    report["alignment_gate"] = _alignment_gate(report)
    return report


def _fmt_tensor(tensor: dict[str, Any] | None) -> str:
    if not tensor:
        return "missing"
    return f"{tensor.get('data_type')}:{tensor.get('dims')}"


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "w16a16_custom_compare.json"
    txt_path = analysis_dir / "w16a16_custom_compare.txt"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [f"W16A16 custom analysis: {report['out_dir']}"]
    native = report.get("native")
    if native:
        lines.append(f"native-exact: {native['exact']}/{native['total']} maxdiff={native['maxdiff']} sorted_equal={native['sorted_equal']}")
    analytic = report.get("analytic")
    if analytic:
        lines.append(f"analytic-exact: {analytic['exact']}/{analytic['total']} maxdiff={analytic['maxdiff']} sorted_equal={analytic['sorted_equal']}")
    custom_perf = report.get("custom_optrace") or {}
    if custom_perf:
        totals = custom_perf.get("totals", {})
        lines.append(
            "custom-optrace: "
            f"cycles={custom_perf.get('custom_cycles_sum')} "
            f"packets={custom_perf.get('custom_packets')} "
            f"timeline={totals.get('timeline_span_cycles')}"
        )
    native_perf = report.get("native_optrace") or {}
    if native_perf:
        totals = native_perf.get("totals", {})
        lines.append(
            "native-optrace: "
            f"kernel_cycles={native_perf.get('custom_cycles_sum')} "
            f"packets={native_perf.get('custom_packets')} "
            f"timeline={totals.get('timeline_span_cycles')}"
        )
    gate = report.get("alignment_gate") or {}
    if gate:
        lines.append(f"alignment-gate: accepted={gate.get('accepted')}")
        for name, check in gate.get("checks", {}).items():
            lines.append(
                f"alignment-gate.{name}: pass={check.get('pass')} "
                f"evidence={check.get('evidence')}"
            )
    for idx, custom in enumerate(report.get("custom_boundaries", [])):
        inputs = custom.get("inputs", {})
        outputs = custom.get("outputs", {})
        lines.append(
            f"custom-boundary-{idx}: "
            f"act={_fmt_tensor(inputs.get('activation'))} "
            f"weight={_fmt_tensor(inputs.get('weight'))} "
            f"bias={_fmt_tensor(inputs.get('bias'))} "
            f"scratch={_fmt_tensor(inputs.get('scratch'))} "
            f"out={_fmt_tensor(outputs.get('output_0'))}"
        )
    for idx, kernel in enumerate(report.get("native_boundary", [])):
        inputs = kernel.get("inputs", {})
        outputs = kernel.get("outputs", {})
        lines.append(
            f"native-boundary-{idx}: "
            f"act={_fmt_tensor(inputs.get('activation'))} "
            f"weight={_fmt_tensor(inputs.get('weight'))} "
            f"bias={_fmt_tensor(inputs.get('bias'))} "
            f"control={_fmt_tensor(inputs.get('control'))} "
            f"extra={_fmt_tensor(inputs.get('extra_control'))} "
            f"out={_fmt_tensor(outputs.get('output_0'))}"
        )
    if "compare_error" in report:
        lines.append(f"compare-error: {report['compare_error']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--native-out-dir", type=Path)
    parser.add_argument("--native-raw", type=Path)
    args = parser.parse_args()

    report = analyze(args.out_dir, args.native_out_dir, args.native_raw)
    json_path, txt_path = write_reports(report, args.out_dir)
    print(f"  analysis json: {json_path}")
    print(f"  analysis txt: {txt_path}")
    for line in txt_path.read_text(encoding="utf-8").splitlines()[1:]:
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
