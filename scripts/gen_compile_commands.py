#!/usr/bin/env python3
"""
gen_compile_commands.py — generate compile_commands.json for the tutorial
sources under docs/hexagon-tutorial/ so VSCode / clangd can index them.

Each .c/.cpp file is classified as either "dsp" (Hexagon, uses hexagon-clang)
or "host" (aarch64 Android, uses NDK clang) based on its path, and gets an
entry with the appropriate -I flags and target/arch flags.

Run:
    python3 scripts/gen_compile_commands.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "tools" / "hexagon-sdk"
QNN = ROOT / "tools" / "qnn-sdk"
NDK = ROOT / "tools" / "android-ndk"
H2 = ROOT / "tools" / "h2-install"
H2_REAL = ROOT / "tools" / "hexagon-hypervisor"
TUTORIAL = ROOT / "docs" / "hexagon-tutorial"
EXAMPLE  = ROOT / "example"

HEX_CLANG = next(
    iter((SDK / "tools" / "HEXAGON_Tools").glob("*/Tools/bin/hexagon-clang")),
    None,
)
NDK_CLANG = (
    NDK / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "bin"
    / "aarch64-linux-android31-clang"
)
LIBNATIVE_INC = next(
    iter((SDK / "tools" / "HEXAGON_Tools").glob("*/Tools/libnative/include")),
    None,
)

# -------- Include path recipes --------

def dsp_includes(is_qnn: bool, is_ch01_sim: bool) -> list[str]:
    inc = [
        f"-I{SDK}/incs",
        f"-I{SDK}/incs/stddef",
        f"-I{SDK}/rtos/qurt/computev75/include/qurt",
        f"-I{SDK}/rtos/qurt/computev75/include/posix",
        f"-I{SDK}/ipc/fastrpc/incs",
    ]
    if LIBNATIVE_INC:
        inc.append(f"-I{LIBNATIVE_INC}")
    if is_qnn:
        inc.append(f"-I{QNN}/include/QNN")
    if is_ch01_sim:
        # ch01 uses H2 hypervisor headers instead of QuRT
        inc.append(f"-I{H2}/include")
        inc.append(f"-I{H2_REAL}/kernel/include")
    return inc


def host_includes(is_qnn: bool) -> list[str]:
    sysroot = (
        NDK / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "sysroot"
    )
    inc = [f"--sysroot={sysroot}"]
    if is_qnn:
        inc.append(f"-I{QNN}/include/QNN")
    if LIBNATIVE_INC:
        inc.append(f"-I{LIBNATIVE_INC}")
    inc += [f"-I{SDK}/ipc/fastrpc/incs", f"-I{SDK}/incs"]
    return inc


def dsp_flags() -> list[str]:
    return [
        "-std=c++17", "-O2",
        "-mv75", "-mhvx", "-mhvx-length=128B", "-mhmx",
        "-DARCHV=75", "-DUSE_OS_QURT",
    ]


def host_flags() -> list[str]:
    return [
        "-std=c17", "-O2",
        "--target=aarch64-none-linux-android31",
        "-DANDROID", "-DUSE_OS_LINUX",
    ]


# -------- Classification --------

def classify(path: Path) -> str | None:
    rel = "/" + path.relative_to(TUTORIAL).as_posix()
    if "/build/" in rel:
        return None
    if "/src/host/" in rel or "/src/arm/" in rel:
        return "host"
    if "/src/dsp/" in rel:
        return "dsp"
    if rel.startswith("/ch01-simulator-setup/") or rel.startswith("/ch02-real-device/"):
        return "dsp"
    if rel.startswith("/hmx-tutorial/") and "/src/" in rel:
        return "dsp"
    return None


def is_qnn(path: Path) -> bool:
    return "qnn-tutorial" in path.relative_to(TUTORIAL).parts


def is_ch01(path: Path) -> bool:
    return path.relative_to(TUTORIAL).parts[0] == "ch01-simulator-setup"


# -------- Main --------

def main() -> int:
    if HEX_CLANG is None:
        print("ERROR: hexagon-clang not found under tools/hexagon-sdk/tools/HEXAGON_Tools/*/Tools/bin/", file=sys.stderr)
        return 1
    if not NDK_CLANG.exists():
        print(f"ERROR: NDK clang not found at {NDK_CLANG}", file=sys.stderr)
        return 1

    entries = []
    # Tutorial sources (auto-classified by path)
    sources = sorted(
        p for p in TUTORIAL.rglob("*")
        if p.is_file() and p.suffix in {".c", ".cpp", ".cc", ".cxx"}
    )
    for src in sources:
        kind = classify(src)
        if kind is None:
            continue
        qnn = is_qnn(src)
        if kind == "dsp":
            args = [str(HEX_CLANG)] + dsp_flags() + dsp_includes(qnn, is_ch01(src)) + ["-c", str(src)]
        else:
            args = [str(NDK_CLANG)] + host_flags() + host_includes(qnn) + ["-c", str(src)]
        entries.append({
            "directory": str(ROOT),
            "file": str(src),
            "arguments": args,
        })

    # example/ sources — treat as DSP (Hexagon) with H2 includes by default.
    # Each example dir is self-contained; add its own dir to -I.
    if EXAMPLE.is_dir():
        ex_sources = sorted(
            p for p in EXAMPLE.rglob("*")
            if p.is_file() and p.suffix in {".c", ".cpp", ".cc", ".cxx"}
        )
        for src in ex_sources:
            args = (
                [str(HEX_CLANG)]
                + dsp_flags()
                + dsp_includes(is_qnn=False, is_ch01_sim=True)  # H2 headers available
                + [f"-I{src.parent}"]
                + ["-c", str(src)]
            )
            entries.append({
                "directory": str(ROOT),
                "file": str(src),
                "arguments": args,
            })

    out_dir = ROOT / ".vscode"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "compile_commands.json"
    with out.open("w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {out} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
