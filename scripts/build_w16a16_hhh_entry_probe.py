#!/usr/bin/env python3
"""Build a patched W16A16 native hhh-entry probe skel.

The probe replaces the first instructions at `hmx_v73_convhhh1x1_stride1`
(`libQnnHtpV75Skel.so@0x2fa740`) with a tiny marker writer:

    out_table = memw(r0 + 0)
    out_block = memw(out_table + 0)
    memw(out_block + 0) = 0x484d5850  # "HMXP"
    return

It is a negative/positive entry probe only.  It deliberately does not preserve
native output correctness; a changed output containing the marker proves the
patched entry is reached.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SO = ROOT / "tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"
DEFAULT_OUT = Path("/tmp/libQnnHtpV75Skel_w16a16_hhh_entry_probe.so")
DEFAULT_VMA = 0x2FA740
MAGIC = 0x484D5850  # HMXP


ENTRY_ASM = f"""
.text
.global probe
.type probe,@function
probe:
{{
    r2 = memw(r0+#0)
}}
{{
    r2 = memw(r2+#0)
}}
{{
    immext(#{MAGIC & 0xFFFFFFC0:#x})
    r3 = ##{MAGIC:#x}
}}
{{
    memw(r2+#0) = r3
}}
{{
    jumpr r31
}}
"""

PATTERN_ASM = """
.text
.global probe
.type probe,@function
probe:
{
    r2 = memw(r0+#0)
}
{
    r2 = memw(r2+#0)
    r3 = #0
}
{
    immext(#0x484d0000)
    r4 = ##0x484d0000
}
{
    loop0(.Lloop, #512)
}
.Lloop:
{
    r5 = or(r4, r3)
}
{
    memw(r2++#4) = r5
    r3 = add(r3,#1)
}:endloop0
{
    jumpr r31
}
"""

PATTERN16_ASM = """
.text
.global probe
.type probe,@function
probe:
{
    r2 = memw(r0+#0)
}
{
    r2 = memw(r2+#0)
    r3 = #0
}
{
    immext(#0x8000)
    r4 = ##0x8000
}
{
    loop0(.Lloop, #512)
}
.Lloop:
{
    r5 = add(r4, r3)
}
{
    memh(r2++#2) = r5
    r3 = add(r3,#1)
}:endloop0
{
    jumpr r31
}
"""

RECORD16_ASM = """
.text
.global probe
.type probe,@function
.macro STORE_U32 src
{
    r8 = extractu(\\src,#0x10,#0x0)
}
{
    memh(r6++#2) = r8
    r8 = lsr(\\src,#0x10)
}
{
    memh(r6++#2) = r8
}
.endm
.macro STORE_MEM base, off
{
    r7 = memw(\\base+\\off)
}
STORE_U32 r7
.endm
probe:
{
    r6 = memw(r0+#0)
}
{
    r6 = memw(r6+#0)
}
{
    immext(#0x484d5840)
    r7 = ##0x484d5852
}
STORE_U32 r7
STORE_U32 r31
STORE_U32 r0
STORE_U32 r1
STORE_U32 r2
STORE_U32 r3
STORE_U32 r4
STORE_U32 r5
STORE_MEM r0, #0
STORE_MEM r0, #4
STORE_MEM r0, #8
STORE_MEM r0, #12
STORE_MEM r0, #16
STORE_MEM r0, #20
STORE_MEM r1, #0
STORE_MEM r1, #4
STORE_MEM r1, #8
STORE_MEM r4, #0
STORE_MEM r4, #4
STORE_MEM r4, #8
STORE_MEM r4, #12
STORE_MEM r4, #16
STORE_MEM r4, #20
STORE_MEM r4, #24
STORE_MEM r4, #28
STORE_MEM r4, #32
STORE_MEM r4, #36
STORE_MEM r4, #40
STORE_MEM r4, #44
STORE_MEM r4, #48
STORE_MEM r4, #52
STORE_MEM r4, #56
STORE_MEM r4, #60
STORE_MEM r5, #0
STORE_MEM r5, #4
{
    jumpr r31
}
"""


def _tool(name: str) -> str:
    candidates = [
        ROOT / "tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin" / name,
        shutil.which(name),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(name)


def build_patch_bytes(cpu: str, mode: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="w16a16_hhh_probe_") as td:
        tmp = Path(td)
        asm = tmp / "probe.S"
        obj = tmp / "probe.o"
        raw = tmp / "probe.bin"
        if mode == "pattern":
            asm_text = PATTERN_ASM
        elif mode == "pattern16":
            asm_text = PATTERN16_ASM
        elif mode == "record16":
            asm_text = RECORD16_ASM
        else:
            asm_text = ENTRY_ASM
        asm.write_text(asm_text, encoding="utf-8")
        subprocess.run(
            [_tool("hexagon-clang"), "-target", "hexagon", f"-mcpu=hexagon{cpu}", "-mhmx", "-c", str(asm), "-o", str(obj)],
            check=True,
        )
        subprocess.run([_tool("hexagon-llvm-objcopy"), "-O", "binary", "-j", ".text", str(obj), str(raw)], check=True)
        return raw.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so", type=Path, default=DEFAULT_SO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vma", type=lambda s: int(s, 0), default=DEFAULT_VMA)
    parser.add_argument("--cpu", default="v75")
    parser.add_argument("--mode", choices=["entry", "pattern", "pattern16", "record16"], default="entry")
    args = parser.parse_args()

    patch = build_patch_bytes(args.cpu, args.mode)
    data = bytearray(args.so.read_bytes())
    if args.vma + len(patch) > len(data):
        raise ValueError(f"patch outside file: vma={args.vma:#x} len={len(patch)} file={len(data)}")
    data[args.vma:args.vma + len(patch)] = patch
    args.out.write_bytes(data)
    print(f"wrote {args.out}")
    print(f"patched offset/vma {args.vma:#x} with {len(patch)} bytes")
    print(f"mode: {args.mode}")
    print("patch:", patch.hex())


if __name__ == "__main__":
    main()
