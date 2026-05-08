#!/usr/bin/env python3
"""Compile a Hexagon inline-asm .inc and byte-compare it to a native slice."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_CLANG = (
    "tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/clang-19"
)
DEFAULT_OBJCOPY = (
    "tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objcopy"
)
DEFAULT_SO = "tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"


def parse_int(text: str) -> int:
    return int(text, 0)


def tool_path(repo: Path, value: str, fallback: str | None = None) -> str:
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    repo_candidate = repo / value
    if repo_candidate.exists():
        return str(repo_candidate)
    found = shutil.which(value)
    if found:
        return found
    if fallback:
        return fallback
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inc", required=True, help="inline asm .inc to verify")
    parser.add_argument("--so", default=DEFAULT_SO, help="native skel .so")
    parser.add_argument("--vma", required=True, type=parse_int, help="native slice VMA/file offset")
    parser.add_argument("--size", required=True, type=parse_int, help="native slice size in bytes")
    parser.add_argument("--repo", default=".", help="repo root for relative paths")
    parser.add_argument("--clang", default=DEFAULT_CLANG)
    parser.add_argument("--objcopy", default=DEFAULT_OBJCOPY)
    parser.add_argument("--mcpu", default="hexagonv75")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    inc = (repo / args.inc).resolve() if not Path(args.inc).is_absolute() else Path(args.inc)
    so = (repo / args.so).resolve() if not Path(args.so).is_absolute() else Path(args.so)
    clang = tool_path(repo, args.clang)
    objcopy = tool_path(repo, args.objcopy)

    if not inc.is_file():
        raise SystemExit(f"missing inc: {inc}")
    if not so.is_file():
        raise SystemExit(f"missing so: {so}")
    if args.size <= 0:
        raise SystemExit("--size must be positive")

    tmp_obj = None
    tmp_text = None
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        c_path = tmpdir / "verify_hmx_inline.c"
        obj_path = tmpdir / "verify_hmx_inline.o"
        text_path = tmpdir / "verify_hmx_inline.text"
        native_path = tmpdir / "native_slice.bin"

        c_path.write_text(
            "\n".join(
                [
                    "#include <stdint.h>",
                    "struct hmx_conv_out_desc_t;",
                    "struct hmx_conv_act_desc_t;",
                    "struct hmx_conv_mask_desc_t;",
                    "__attribute__((naked, aligned(64), noinline))",
                    "void our_v73deep_kernel(",
                    "    const struct hmx_conv_out_desc_t *od,",
                    "    const struct hmx_conv_act_desc_t *ad,",
                    "    const uint8_t *wt,",
                    "    const uint8_t *bias,",
                    "    const struct hmx_conv_mask_desc_t *mask,",
                    "    const uint32_t *extra_param) {",
                    "  __asm__ volatile (",
                    f'#include "{inc}"',
                    "  );",
                    "}",
                    "",
                ]
            )
        )

        subprocess.check_call(
            [
                clang,
                "-target",
                "hexagon",
                f"-mcpu={args.mcpu}",
                "-mhmx",
                "-I",
                str(repo),
                "-c",
                str(c_path),
                "-o",
                str(obj_path),
            ]
        )
        subprocess.check_call(
            [objcopy, "-O", "binary", "--only-section=.text", str(obj_path), str(text_path)]
        )

        data = so.read_bytes()[args.vma : args.vma + args.size]
        if len(data) != args.size:
            raise SystemExit(f"native short read: got {len(data)} expected {args.size}")
        native_path.write_bytes(data)
        text = text_path.read_bytes()

        if args.keep_temp:
            tmp_obj = Path("/tmp") / "verify_hmx_inline.o"
            tmp_text = Path("/tmp") / "verify_hmx_inline.text"
            tmp_obj.write_bytes(obj_path.read_bytes())
            tmp_text.write_bytes(text)

        if text != data:
            first = next(i for i, (a, b) in enumerate(zip(text, data)) if a != b)
            raise SystemExit(
                f"byte mismatch at offset 0x{first:x}: generated=0x{text[first]:02x} "
                f"native=0x{data[first]:02x}"
            )

    print(f"OK: {inc} matches {so}@0x{args.vma:x} ({args.size} bytes)")
    if tmp_obj and tmp_text:
        print(f"kept {tmp_obj} and {tmp_text}")


if __name__ == "__main__":
    main()
