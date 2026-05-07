#!/usr/bin/env python3
"""Validate the repo-standard QNN run artifact layout.

The checker is intentionally file-based.  It does not prove numerical
correctness; it catches the common invalid-performance baselines: DLC-only
runs, missing context binaries, missing optrace decode products, float runtime
I/O where a native raw contract was recorded, and converter logs that do not
show the layout-preservation flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _first_nonempty(paths: list[Path]) -> Path | None:
    for path in paths:
        if _nonempty(path):
            return path
    return None


def _glob_nonempty(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if _nonempty(path))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _storage_is_float(storage) -> bool:
    return any("float" in str(item).lower() or str(item).lower().startswith("fp") for item in _as_list(storage))


def check_artifact(
    out_dir: Path,
    require_native_io: bool,
    require_layout_flags: bool,
    reject_float_io: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    dlcs = _glob_nonempty(out_dir, "*.dlc")
    if not dlcs:
        errors.append("missing converted DLC (*.dlc)")

    ctx_bins = _glob_nonempty(out_dir / "ctx", "*.bin")
    context_bins = [p for p in ctx_bins if "schematic" not in p.name and "mapping" not in p.name]
    if not context_bins:
        errors.append("missing context binary under ctx/*.bin")

    mappings = _glob_nonempty(out_dir / "ctx", "*bottom_mapping*.json")
    if not mappings:
        errors.append("missing ctx/*bottom_mapping*.json")

    schematic = _first_nonempty(
        _glob_nonempty(out_dir / "ctx", "*schematic.bin")
        + _glob_nonempty(out_dir, "*schematic.bin")
    )
    if schematic is None:
        errors.append("missing schematic bin beside the context artifacts")

    device_logs = _glob_nonempty(out_dir / "device_out", "qnn-profiling-data*.log")
    if not device_logs:
        errors.append("missing device_out/qnn-profiling-data*.log")

    device_raws = _glob_nonempty(out_dir / "device_out", "*.raw")
    if not device_raws:
        errors.append("missing native/device output raw under device_out/*.raw")

    optrace_dir = out_dir / "optrace"
    for rel in (
        "summary.json",
        "profile.txt",
        "manifest.json",
        "chrometrace.json",
    ):
        if not _nonempty(optrace_dir / rel):
            errors.append(f"missing optrace/{rel}")

    convert_logs = [
        path
        for name in ("convert.log", "_convert.log")
        for path in (out_dir / name, out_dir.parent / name)
        if _nonempty(path)
    ]
    convert_log_text = ""
    if not convert_logs:
        warnings.append("missing converter log; cannot verify layout-preservation flags")
    elif require_layout_flags:
        convert_log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in convert_logs)
        if "NONTRIVIAL" not in convert_log_text:
            errors.append("converter log does not show NONTRIVIAL layout-preservation flags")

    native_io_path = out_dir / "native_io.json"
    if native_io_path.exists():
        native_io = _load_json(native_io_path)
        native_input = native_io.get("native_input")
        if native_input:
            native_input_path = out_dir / native_input
            if not _nonempty(native_input_path):
                errors.append(f"native_io.json points to missing native input: {native_input}")
            expected = native_io.get("native_input_bytes")
            if expected is not None and _nonempty(native_input_path):
                got = native_input_path.stat().st_size
                if got != int(expected):
                    errors.append(f"native input size mismatch: got {got}, expected {expected}")
        expected_out = native_io.get("expected_native_output_bytes")
        if expected_out is not None and device_raws:
            if not any(raw.stat().st_size == int(expected_out) for raw in device_raws):
                sizes = ", ".join(f"{raw.name}:{raw.stat().st_size}" for raw in device_raws)
                errors.append(f"no device output raw matches expected native size {expected_out} ({sizes})")
        if reject_float_io:
            if _storage_is_float(native_io.get("native_input_storage")):
                errors.append("native_io.json records float native input storage")
            if _storage_is_float(native_io.get("expected_native_output_storage")):
                errors.append("native_io.json records float native output storage")
        if require_layout_flags and convert_log_text:
            io_names = [item for item in _as_list(native_io.get("input_name")) + _as_list(native_io.get("output_name")) if isinstance(item, str)]
            missing_names = [name for name in io_names if name not in convert_log_text]
            if missing_names:
                errors.append(
                    "converter log does not mention layout-preserved native_io tensors: "
                    + ", ".join(missing_names)
                )
    elif require_native_io:
        errors.append("missing native_io.json for a native-reference artifact")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--require-native-io", action="store_true")
    parser.add_argument("--require-layout-flags", action="store_true")
    parser.add_argument(
        "--reject-float-io",
        action="store_true",
        help="fail if native_io.json records float native input/output storage",
    )
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    errors, warnings = check_artifact(
        out_dir,
        require_native_io=args.require_native_io,
        require_layout_flags=args.require_layout_flags,
        reject_float_io=args.reject_float_io,
    )
    for warning in warnings:
        print(f"  [warn] {warning}")
    if errors:
        print(f"  qnn artifact standard: FAIL ({out_dir})")
        for error in errors:
            print(f"    - {error}")
        return 0 if args.warn_only else 1
    print(f"  qnn artifact standard: ok ({out_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
