#!/usr/bin/env python3
"""Build and run the handwritten HMX body-entry simulator smoke."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "example" / "handwritten_hmx_matmul" / "tools" / "body_entry_smoke.c"
FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_h2_root() -> tuple[Path, Path]:
    h2_install = ROOT / "tools" / "h2-install"
    if h2_install.is_symlink():
        h2_root = h2_install.resolve().parent
    else:
        h2_root = h2_install.parent
    return h2_root, h2_install


def tool(name: str) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found)
    candidate = first_existing(
        sorted((ROOT / "tools" / "hexagon-sdk" / "tools" / "HEXAGON_Tools").glob("*/Tools/bin/" + name))
    )
    if candidate is not None:
        return candidate
    raise FileNotFoundError(f"missing tool: {name}")


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_results(output: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for family in FAMILIES:
        passed = re.search(rf"^\[PASS\] {re.escape(family)} body entry returned$", output, re.M)
        failed = re.search(rf"^\[FAIL\] {re.escape(family)} body entry.*$", output, re.M)
        results[family] = {
            "entered_and_returned": bool(passed),
            "status": "pass" if passed else "fail",
            "failure_line": failed.group(0) if failed else None,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    if not SOURCE.is_file():
        errors.append(f"missing source: {SOURCE}")
    try:
        clang = tool("hexagon-clang")
        sim = tool("hexagon-sim")
    except FileNotFoundError as exc:
        errors.append(str(exc))
        clang = sim = Path("")

    h2_root, h2_install = resolve_h2_root()
    booter = h2_install / "bin" / "booter"
    h2_include = h2_install / "include"
    h2_kernel_include = h2_root / "kernel" / "include"
    h2_lib = h2_install / "lib"
    for path in (booter, h2_include, h2_kernel_include, h2_lib):
        if not path.exists():
            errors.append(f"missing H2 prerequisite: {path}")

    payload = {
        "schema": "handwritten_hmx_body_entry_sim.v1",
        "qnn_used": False,
        "source": str(SOURCE.relative_to(ROOT)),
        "results": {},
        "compile": {},
        "simulate": {},
        "errors": errors,
    }
    if errors:
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("handwritten HMX body-entry simulator smoke: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    work = Path(tempfile.mkdtemp(prefix="handwritten_hmx_body_entry_"))
    binary = work / "body_entry_smoke"
    compile_cmd = [
        str(clang),
        "-O2",
        "-mv75",
        "-mhvx",
        "-mhvx-length=128B",
        "-mhmx",
        "-DARCHV=75",
        "-I",
        str(h2_include),
        "-I",
        str(h2_kernel_include),
        "-I",
        str(ROOT / "example" / "handwritten_hmx_matmul" / "include"),
        "-moslib=h2",
        "-Wl,-L," + str(h2_lib),
        "-Wl,--section-start=.start=0x02000000",
        "-o",
        str(binary),
        str(SOURCE),
    ]
    compile_result = run(compile_cmd, ROOT)
    payload["compile"] = {
        "command": compile_cmd,
        "returncode": compile_result.returncode,
        "output": compile_result.stdout.strip().splitlines(),
    }
    if compile_result.returncode == 0:
        sim_cmd = [
            str(sim),
            "--mv75",
            "--mhmx",
            "1",
            "--simulated_returnval",
            "--",
            str(booter),
            "--ext_power",
            "1",
            "--use_ext",
            "1",
            "--fence_hi",
            "0xfe000000",
            str(binary),
        ]
        sim_result = run(sim_cmd, ROOT)
    else:
        sim_cmd = []
        sim_result = subprocess.CompletedProcess([], 1, "")

    results = parse_results(sim_result.stdout)
    payload["results"] = results
    payload["simulate"] = {
        "command": sim_cmd,
        "returncode": sim_result.returncode,
        "output": sim_result.stdout.strip().splitlines(),
    }
    payload["pass"] = (
        compile_result.returncode == 0
        and sim_result.returncode == 0
        and all(record["entered_and_returned"] for record in results.values())
    )
    if args.keep_work:
        payload["work_dir"] = str(work)
    else:
        shutil.rmtree(work)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for family in FAMILIES:
        status = "ok" if results.get(family, {}).get("entered_and_returned") else "FAILED"
        print(f"{family}: {status} body-entry simulator smoke")
    if not payload["pass"]:
        if compile_result.returncode != 0:
            print("compile failed", file=sys.stderr)
            for line in compile_result.stdout.strip().splitlines():
                print(f"  {line}", file=sys.stderr)
        if sim_result.returncode != 0:
            print("simulation failed", file=sys.stderr)
            for line in sim_result.stdout.strip().splitlines()[-80:]:
                print(f"  {line}", file=sys.stderr)
        return 1
    print("handwritten HMX body-entry simulator smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
