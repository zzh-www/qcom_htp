#!/usr/bin/env python3
"""Summarize retained HMX MatMul optrace performance artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("example/qnn_matmul_profile")
FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")


def _load_summary(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "optrace" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _events(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in summary.get("events", []) if isinstance(event, dict)]


def _kernel_events(summary: dict[str, Any], family: str, native: bool) -> list[dict[str, Any]]:
    events = _events(summary)
    if native:
        return [e for e in events if "ConvLayer_s1.opt" in str(e.get("htp_type", ""))]
    family_tag = family.upper().replace("U8I8", "U8I8").replace("W4A8", "W4A8")
    if family == "u8i8":
        substr = "HmxU8I8ToU8MatMul"
    elif family == "w4a8":
        substr = "HmxU8I4ToU8MatMul"
    elif family == "w8a16":
        substr = "HmxU16I8ToU16MatMul"
    elif family == "w4a16":
        substr = "HmxU16I4ToU16MatMul"
    else:
        substr = family_tag
    return [e for e in events if substr in str(e.get("htp_type", ""))]


def _record(root: Path, family: str, native: bool) -> dict[str, Any]:
    suffix = "native_ref" if native else "aligned"
    out_dir = root / f"output_{family}_{suffix}_e2e_256"
    summary = _load_summary(out_dir)
    events = _kernel_events(summary, family, native)
    totals = summary.get("totals", {})
    cycles = [int(e.get("dur", 0)) for e in events]
    packets = [int(e.get("packets", 0)) for e in events if e.get("packets") is not None]
    return {
        "family": family,
        "kind": "native" if native else "custom",
        "out_dir": str(out_dir),
        "timeline": int(totals.get("timeline_span_cycles", 0)),
        "sum_pid0": int(totals.get("sum_pid0_event_cycles", 0)),
        "kernel_event_count": len(events),
        "kernel_cycles_sum": sum(cycles),
        "kernel_cycles_first": cycles[0] if cycles else None,
        "kernel_cycles_rest_sum": sum(cycles[1:]) if len(cycles) > 1 else 0,
        "kernel_packets_per_event": packets,
        "kernel_packets_sum": sum(packets),
    }


def build_report(root: Path) -> dict[str, Any]:
    records = []
    for family in FAMILIES:
        for native in (False, True):
            try:
                records.append(_record(root, family, native))
            except FileNotFoundError:
                continue
    by_key = {(r["family"], r["kind"]): r for r in records}
    comparisons = []
    for w8, w4, label in (("u8i8", "w4a8", "A8"), ("w8a16", "w4a16", "A16")):
        for kind in ("custom", "native"):
            lhs = by_key.get((w8, kind))
            rhs = by_key.get((w4, kind))
            if lhs and rhs and lhs["kernel_cycles_sum"]:
                comparisons.append(
                    {
                        "pair": label,
                        "kind": kind,
                        "w8_family": w8,
                        "w4_family": w4,
                        "w8_cycles": lhs["kernel_cycles_sum"],
                        "w4_cycles": rhs["kernel_cycles_sum"],
                        "w4_over_w8": rhs["kernel_cycles_sum"] / lhs["kernel_cycles_sum"],
                        "speedup_w8_over_w4": lhs["kernel_cycles_sum"] / rhs["kernel_cycles_sum"],
                        "w8_timeline": lhs["timeline"],
                        "w4_timeline": rhs["timeline"],
                    }
                )
    return {"root": str(root), "records": records, "comparisons": comparisons}


def print_report(report: dict[str, Any]) -> None:
    print("=== HMX retained-artifact performance summary ===")
    print(f"root: {report['root']}")
    print("\n[records]")
    for r in report["records"]:
        print(
            f"  {r['family']:6s} {r['kind']:6s} "
            f"kernel_sum={r['kernel_cycles_sum']:6d} first={str(r['kernel_cycles_first']):>6s} "
            f"events={r['kernel_event_count']:2d} packets={r['kernel_packets_sum']:6d} "
            f"timeline={r['timeline']:6d}"
        )
    print("\n[w4 vs w8 comparisons]")
    for c in report["comparisons"]:
        print(
            f"  {c['pair']} {c['kind']}: "
            f"{c['w4_family']}/{c['w8_family']} cycles={c['w4_over_w8']:.3f} "
            f"speedup={c['speedup_w8_over_w4']:.2f}x "
            f"({c['w4_cycles']} vs {c['w8_cycles']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.root.resolve())
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
