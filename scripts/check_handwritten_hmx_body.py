#!/usr/bin/env python3
"""Validate handwritten HMX body byte identity against native slices."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / ".codex" / "skills" / "hmx-inline-asm" / "scripts" / "verify_hexagon_inline_asm.py"
DEFAULT_SO = "tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"
DEFAULT_CLANG = "tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/clang-19"
DEFAULT_NM = "tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-nm"

# native_vma is resolved at runtime from native_symbol via hexagon-nm against the
# actual skel .so (see resolve_vma), so it survives SDK relocations. The literal
# values below are a fallback only and track the current SDK (QNN 2.46.0.260424).
BODY_MANIFEST = {
    "u8i8": {
        "native_symbol": "hmx_v73_convbbb1x1deep_stride1",
        "native_vma": "0x2ef3c0",
        "native_size": 1132,
        "inc": "example/handwritten_hmx_matmul/kernels/u8i8/v73deep_conv1x1_kernel.inc",
        "abi_header": "example/handwritten_hmx_matmul/include/handwritten_hmx_u8i8_kernel.h",
    },
    "w4a8": {
        "native_symbol": "hmx_v73_convbnb1x1_stride1",
        "native_vma": "0x2f3d00",
        "native_size": 2624,
        "inc": "example/handwritten_hmx_matmul/kernels/w4a8/v73deep_conv1x1_kernel.inc",
        "abi_header": "example/handwritten_hmx_matmul/include/handwritten_hmx_w4a8_kernel.h",
    },
    "w4a16": {
        "native_symbol": "hmx_v73_convhnh1x1deep_stride1",
        "native_vma": "0x301100",
        "native_size": 804,
        "inc": "example/handwritten_hmx_matmul/kernels/w4a16/v73deep_conv1x1_kernel.inc",
        "abi_header": "example/handwritten_hmx_matmul/include/handwritten_hmx_w4a16_kernel.h",
    },
    "w8a16": {
        "native_symbol": "hmx_v75_convhbh1x1deep_stride1",
        "native_vma": "0x2f8780",
        "native_size": 1348,
        "inc": "example/handwritten_hmx_matmul/kernels/w8a16/v73deep_conv1x1_kernel.inc",
        "abi_header": "example/handwritten_hmx_matmul/include/handwritten_hmx_w8a16_kernel.h",
    },
    "w16a16": {
        "native_symbol": "hmx_v73_convhhh1x1_stride1",
        "native_vma": "0x2fdcc0",
        "native_size": 1800,
        "inc": "example/handwritten_hmx_matmul/kernels/w16a16/v73deep_conv1x1_kernel.inc",
        "abi_header": "example/handwritten_hmx_matmul/include/handwritten_hmx_w16a16_kernel.h",
    },
}
DEFAULT_FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")


def tool_path(value: str) -> str:
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    repo_candidate = ROOT / value
    if repo_candidate.exists():
        return str(repo_candidate)
    found = shutil.which(value)
    if found:
        return found
    return value


def run_header_compile(family: str, clang: str) -> dict:
    record = BODY_MANIFEST[family]
    header = record["abi_header"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        source = tmp / f"compile_{family}_header.c"
        obj = tmp / f"compile_{family}_header.o"
        source.write_text(
            "\n".join(
                [
                    "#include <stdint.h>",
                    f'#include "{header}"',
                    f"int hm_{family}_header_compile_smoke(void) {{",
                    "  return 0;",
                    "}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        command = [
            clang,
            "-target",
            "hexagon",
            "-mcpu=hexagonv75",
            "-mhmx",
            "-I",
            str(ROOT),
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    return {
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout.strip().splitlines(),
        "header_compile_pass": result.returncode == 0,
    }


def resolve_vma(symbol: str, so: str, nm: str, fallback: str) -> str:
    """Resolve a symbol's VMA from the skel .so via hexagon-nm.

    Keeps the gate correct across SDK relocations (the skel grows and shifts
    every release, so a hardcoded offset rots). Falls back to the manifest
    value if nm is unavailable or the symbol is not found.
    """
    nm_bin = tool_path(nm)
    if not (Path(nm_bin).exists() or shutil.which(nm_bin)):
        return fallback
    try:
        out = subprocess.run(
            [nm_bin, "-D", "--defined-only", so],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout
    except OSError:
        return fallback
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == symbol:
            return "0x" + parts[0].lstrip("0").rjust(1, "0")
    return fallback


def run_verify(family: str, so: str, nm: str) -> dict:
    record = BODY_MANIFEST[family]
    vma = resolve_vma(record["native_symbol"], so, nm, record["native_vma"])
    command = [
        sys.executable,
        str(VERIFY),
        "--inc",
        record["inc"],
        "--so",
        so,
        "--vma",
        vma,
        "--size",
        str(record["native_size"]),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    byte_identity_pass = result.returncode == 0
    return {
        "family": family,
        "native_symbol": record["native_symbol"],
        "native_vma": vma,
        "native_size": record["native_size"],
        "inc": record["inc"],
        "abi_header": record["abi_header"],
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout.strip().splitlines(),
        "byte_identity_pass": byte_identity_pass,
        "packet_equivalence_pass": byte_identity_pass,
        "packet_equivalence_method": "byte_identity_preserves_native_packet_stream",
        "packet_equivalence_scope": "whole_embedded_hmx_body_slice",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(BODY_MANIFEST), action="append")
    parser.add_argument("--so", default=DEFAULT_SO)
    parser.add_argument("--clang", default=DEFAULT_CLANG)
    parser.add_argument("--nm", default=DEFAULT_NM)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    families = args.family or list(DEFAULT_FAMILIES)
    clang = tool_path(args.clang)
    if not VERIFY.is_file():
        print(f"missing verifier: {VERIFY}", file=sys.stderr)
        return 1

    results = []
    for family in families:
        result = run_verify(family, args.so, args.nm)
        result["header_compile"] = run_header_compile(family, clang)
        results.append(result)
    payload = {
        "schema": "handwritten_hmx_body_check.v1",
        "qnn_used": False,
        "native_skel_used_as_byte_oracle": args.so,
        "hexagon_clang": clang,
        "results": results,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = [
        result
        for result in results
        if not result["byte_identity_pass"]
        or not result.get("header_compile", {}).get("header_compile_pass")
    ]
    for result in results:
        header_ok = result.get("header_compile", {}).get("header_compile_pass")
        status = "ok" if result["byte_identity_pass"] and header_ok else "FAILED"
        print(
            f"{result['family']}: {status} "
            f"{result['native_symbol']}@{result['native_vma']} "
            f"({result['native_size']} bytes)"
        )
        for line in result["output"]:
            print(f"  {line}")
        header = result.get("header_compile", {})
        header_status = "ok" if header.get("header_compile_pass") else "FAILED"
        print(f"  header compile: {header_status} ({result['abi_header']})")
        for line in header.get("output", []):
            print(f"    {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
