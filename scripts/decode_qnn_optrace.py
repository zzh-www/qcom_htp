#!/usr/bin/env python3
"""Decode a QNN HTP optrace profile into repo-local artifacts.

Usage:
  scripts/decode_qnn_optrace.py <out_dir>
  scripts/decode_qnn_optrace.py <out_dir> --profile-log <log> --schematic <bin>

The run directory is expected to contain a pulled device profile log such as
device_out/qnn-profiling-data_0.log and a ctx/*schematic.bin.  The script keeps
all generated viewer artifacts under <out_dir>/optrace/ instead of /tmp.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


OPTRACE_CONFIG = {
    "enable_input_output_flow_events": False,
    "enable_sequencer_flow_events": False,
    "htp_json": True,
    "runtrace": True,
    "memory_info": True,
}


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        try:
            if path.exists() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def find_profile_log(out_dir: Path) -> Path:
    candidates = (
        sorted((out_dir / "device_out").glob("qnn-profiling-data*.log"))
        + sorted((out_dir / "ctx").glob("qnn-profiling-data*.log"))
        + sorted(out_dir.glob("profile.log"))
    )
    valid = [p for p in candidates if p.exists() and p.stat().st_size > 1500]
    if not valid:
        raise SystemExit(f"ERROR: no valid qnn-profiling-data*.log under {out_dir}")
    return valid[0]


def find_schematic(out_dir: Path) -> Path:
    schematic = _first_existing(
        sorted((out_dir / "ctx").glob("*schematic.bin"))
        + sorted(out_dir.glob("*schematic.bin"))
    )
    if schematic is None:
        raise SystemExit(f"ERROR: no *schematic.bin under {out_dir}/ctx or {out_dir}")
    return schematic


def qnn_tools() -> tuple[Path, Path, Path, Path]:
    qnn = os.environ.get("QNN_SDK_ROOT")
    if not qnn:
        raise SystemExit("ERROR: QNN_SDK_ROOT not set; source scripts/env.sh first")
    root = Path(qnn)
    viewer = root / "bin/x86_64-linux-clang/qnn-profile-viewer"
    reader = root / "lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
    profile_reader = root / "lib/x86_64-linux-clang/libQnnHtpProfilingReader.so"
    libdir = root / "lib/x86_64-linux-clang"
    for path in (viewer, reader, profile_reader, libdir):
        if not path.exists():
            raise SystemExit(f"ERROR: missing QNN profiler dependency: {path}")
    return viewer, reader, profile_reader, libdir


def run_viewer(
    out_dir: Path,
    profile_log: Path,
    schematic: Path,
    optrace_dir: Path,
) -> Path:
    viewer, reader, profile_reader, libdir = qnn_tools()
    optrace_dir.mkdir(parents=True, exist_ok=True)
    config = optrace_dir / "_optrace_config.json"
    config.write_text(json.dumps(OPTRACE_CONFIG, indent=2) + "\n", encoding="utf-8")
    chrometrace = optrace_dir / "chrometrace.json"
    viewer_log = optrace_dir / "_viewer.log"

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        str(viewer),
        "--config",
        str(config),
        "--reader",
        str(reader),
        "--input_log",
        str(profile_log),
        "--schematic",
        str(schematic),
        "--output",
        str(chrometrace),
    ]
    res = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    viewer_log.write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode != 0 or not chrometrace.exists():
        raise SystemExit(f"ERROR: qnn-profile-viewer failed; see {viewer_log}")

    profile_txt = optrace_dir / "profile.txt"
    profile_res = subprocess.run(
        [
            str(viewer),
            "--reader",
            str(profile_reader),
            "--input_log",
            str(profile_log),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    profile_txt.write_text(
        profile_res.stdout + profile_res.stderr,
        encoding="utf-8",
    )

    manifest = {
        "out_dir": str(out_dir),
        "profile_log": str(profile_log),
        "schematic": str(schematic),
        "viewer_log": str(viewer_log),
        "profile_txt": str(profile_txt),
        "command": cmd,
        "artifacts": sorted(str(p) for p in optrace_dir.iterdir() if p.is_file()),
    }
    (optrace_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return chrometrace


def _event_duration(event: dict) -> int:
    args = event.get("args", {})
    return int(args.get("Duration (cycles)", event.get("dur", 0)) or 0)


def _event_cpp(event: dict) -> float | None:
    value = event.get("args", {}).get("Cycles per Packet")
    if value in (None, 0, "0"):
        return None
    return float(value)


def _is_pid0_x_event(event: dict) -> bool:
    return event.get("ph") == "X" and event.get("pid") == 0


def build_summary(chrometrace: Path, optrace_dir: Path) -> dict:
    data = json.loads(chrometrace.read_text(encoding="utf-8"))
    events = [event for event in data.get("traceEvents", []) if _is_pid0_x_event(event)]
    if not events:
        summary = {"events": [], "totals": {}}
        (optrace_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    rows = []
    by_htp_type: dict[str, int] = defaultdict(int)
    by_qnn_op: dict[str, int] = defaultdict(int)
    start = min(int(event.get("ts", 0)) for event in events)
    end = max(int(event.get("ts", 0)) + _event_duration(event) for event in events)
    for event in sorted(events, key=lambda item: int(item.get("ts", 0))):
        args = event.get("args", {})
        dur = _event_duration(event)
        cpp = _event_cpp(event)
        packets = int(round(dur / cpp)) if cpp else None
        htp_type = args.get("HTP Op Type", event.get("name", ""))
        qnn_op = args.get("QNN Op Name", "")
        by_htp_type[htp_type] += dur
        by_qnn_op[qnn_op] += dur
        rows.append(
            {
                "ts": int(event.get("ts", 0)),
                "dur": dur,
                "packets": packets,
                "cpp": cpp,
                "htp_type": htp_type,
                "qnn_op": qnn_op,
            }
        )

    summary = {
        "totals": {
            "timeline_span_cycles": end - start,
            "sum_pid0_event_cycles": sum(row["dur"] for row in rows),
            "event_count_pid0": len(rows),
        },
        "by_htp_type_cycles": dict(sorted(by_htp_type.items(), key=lambda kv: -kv[1])),
        "by_qnn_op_cycles": dict(sorted(by_qnn_op.items(), key=lambda kv: -kv[1])),
        "events": rows,
    }
    (optrace_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def print_summary(summary: dict, optrace_dir: Path) -> None:
    totals = summary.get("totals", {})
    print(f"  optrace dir: {optrace_dir}")
    if totals:
        print(f"  timeline span cycles: {totals.get('timeline_span_cycles', 0)}")
        print(f"  sum pid0 event cycles: {totals.get('sum_pid0_event_cycles', 0)}")
    for name, cycles in list(summary.get("by_htp_type_cycles", {}).items())[:8]:
        print(f"    {cycles:8d}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--schematic", type=Path)
    parser.add_argument("--optrace-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    profile_log = args.profile_log.resolve() if args.profile_log else find_profile_log(out_dir)
    schematic = args.schematic.resolve() if args.schematic else find_schematic(out_dir)
    optrace_dir = args.optrace_dir.resolve() if args.optrace_dir else out_dir / "optrace"

    chrometrace = run_viewer(out_dir, profile_log, schematic, optrace_dir)
    summary = build_summary(chrometrace, optrace_dir)
    print_summary(summary, optrace_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
