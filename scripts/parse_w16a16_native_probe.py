#!/usr/bin/env python3
"""Parse W16A16 hhh native record16 probes from public Y.raw.

`scripts/build_w16a16_hhh_entry_probe.py --mode record16` writes a compact
u32 record as little-endian halfwords into the first internal output block.
The first 128 internal halfwords are visible in public native `Y.raw` through
the mapping recovered by the `pattern16` probe:

    public[m=(j//64)*2 + (j&1), n=(j%64)//2] = internal_halfword[j]

This parser decodes that compact record without depending on the older W4
u32 crouton map.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


MAGIC_RECORD16 = 0x484D5852  # HMXR

LABELS = [
    "magic",
    "r31_return",
    "out_desc_arg",
    "act_desc_arg",
    "weight_arg",
    "bias_arg",
    "mask_arg",
    "extra_arg",
    "out_table_ptr",
    "out_table_stride_dwords",
    "out_y_stride_words",
    "out_n_tiles_pow2",
    "out_m_total_minus_step",
    "out_k_total_bytes",
    "act_table_ptr",
    "act_n_pairs",
    "act_y_stride_words",
]


def _as_i32(value: int) -> int:
    return struct.unpack("<i", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _read_u16_matrix(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != 256 * 256 * 2:
        raise ValueError(f"expected 256x256 u16 raw, got {len(raw)} bytes")
    return list(struct.unpack("<65536H", raw))


def _public_index_for_internal_halfword(j: int) -> int:
    m = (j // 64) * 2 + (j & 1)
    n = (j % 64) // 2
    return m * 256 + n


def decode_record16(path: Path, words: int) -> dict[str, Any]:
    public = _read_u16_matrix(path)
    halves = [public[_public_index_for_internal_halfword(j)] for j in range(words * 2)]
    record = [halves[i * 2] | (halves[i * 2 + 1] << 16) for i in range(words)]
    fields = {label: record[i] for i, label in enumerate(LABELS) if i < len(record)}
    if "out_m_total_minus_step" in fields:
        fields["out_m_total_minus_step_signed"] = _as_i32(fields["out_m_total_minus_step"])
    if len(record) >= 33:
        fields["mask_words"] = record[17:33]
    if len(record) >= 35:
        fields["extra_words"] = record[33:35]
    return {
        "path": str(path),
        "record_kind": "w16a16_hhh_record16",
        "record_words": len(record),
        "magic_ok": bool(record and record[0] == MAGIC_RECORD16),
        "fields": fields,
        "record": record,
    }


def _fmt(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:08x}"


def print_human(parsed: dict[str, Any]) -> None:
    fields = parsed["fields"]
    print(f"=== W16A16 hhh record16 probe: {parsed['path']} ===")
    print(f"magic={_fmt(fields.get('magic', 0))} ({'ok' if parsed['magic_ok'] else 'unexpected'})")
    print()
    for name in LABELS[1:17]:
        if name not in fields:
            continue
        value = fields[name]
        if name == "out_m_total_minus_step":
            print(f"{name:<28} = {_as_i32(value)} ({_fmt(value)})")
        else:
            print(f"{name:<28} = {_fmt(value)}")
    print()
    if "mask_words" in fields:
        print("mask_words:")
        for i, value in enumerate(fields["mask_words"]):
            print(f"  [{i:02d}] = {_fmt(value)}")
    if "extra_words" in fields:
        print("extra_words:")
        for i, value in enumerate(fields["extra_words"]):
            print(f"  [{i:02d}] = {_fmt(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--words", type=int, default=36)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    parsed = decode_record16(args.raw_path, args.words)
    if args.json:
        print(json.dumps(parsed, indent=2))
    else:
        print_human(parsed)


if __name__ == "__main__":
    main()
