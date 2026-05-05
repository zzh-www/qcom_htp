#!/usr/bin/env python3
"""extract_v73deep_bytes.py — regenerate the V73DEEP Conv1x1 .byte fallback.

Reads libQnnHtpV75Skel.so + the existing disassembly at
Agent/qnn_re/hmx_v73_convbbb1x1deep_stride1_2ebe40.S, emits an annotated
.byte block where every 4-byte word line is paired with its decoded Hexagon
asm. The production file is now hand-written inline asm, but this script is
kept as a sanity/regeneration tool for the owned V73DEEP Conv1x1 body.

Runtime no longer dlsym's QNN's binary. The custom op jumps into its embedded
copy of `hmx_v73_convbbb1x1deep_stride1` (VMA 0x2ebe40, 1132 bytes). Code is
position-independent (PC-relative branches only, no calls out).
"""
import argparse
import re
from pathlib import Path

KERNEL_VMA  = 0x2ebe40
KERNEL_SIZE = 1132


def parse_disasm(path: Path) -> dict[int, str]:
    """Map address → decoded instruction text."""
    info: dict[int, str] = {}
    line_re = re.compile(
        r"\s*([0-9a-fA-F]+):\s*"
        r"[0-9a-fA-F]{2}\s+[0-9a-fA-F]{2}\s+[0-9a-fA-F]{2}\s+[0-9a-fA-F]{2}\s+"
        r"[0-9a-fA-F]{8}\s+(.+)$"
    )
    for line in path.read_text().splitlines():
        m = line_re.match(line)
        if m:
            info[int(m.group(1), 16)] = m.group(2).rstrip()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default="tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so")
    ap.add_argument("--disasm", default="Agent/qnn_re/hmx_v73_convbbb1x1deep_stride1_2ebe40.S")
    ap.add_argument("--out", default="example/qnn_hmx_matmul_u8i8/src/v73deep_conv1x1_kernel.inc")
    args = ap.parse_args()

    blob = Path(args.so).read_bytes()[KERNEL_VMA:KERNEL_VMA + KERNEL_SIZE]
    if len(blob) != KERNEL_SIZE:
        raise SystemExit(f"short read: got {len(blob)} expected {KERNEL_SIZE}")
    asm = parse_disasm(Path(args.disasm))

    lines = [
        f"  /* {KERNEL_SIZE} bytes — annotated byte-replica of {args.so}",
        f"   * 0x{KERNEL_VMA:x} `hmx_v73_convbbb1x1deep_stride1`.",
        f"   * One word (= one .byte directive) per line + decoded asm.",
        f"   * See scripts/extract_v73deep_bytes.py to regenerate.        */",
    ]
    for off in range(0, KERNEL_SIZE, 4):
        word = blob[off:off + 4]
        bs = ", ".join(f"0x{b:02x}" for b in word)
        addr = KERNEL_VMA + off
        instr = asm.get(addr, "").replace("\\", "\\\\").replace('"', '\\"')
        comment = f" /* {addr:08x}: {instr} */" if instr else ""
        lines.append(f'  ".byte {bs}\\n"{comment}')
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({KERNEL_SIZE} bytes / {KERNEL_SIZE//4} word lines)")


if __name__ == "__main__":
    main()
