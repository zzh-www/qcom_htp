#!/usr/bin/env python3
"""Parse the native W4A16 HNH entry probe exported through public Y.raw.

The `/tmp/libQnnHtpV75Skel_hmx_entry_probe.so` diagnostic replaces
`hmx_v73_convhnh1x1_stride1` and writes a compact u32 record into the first
internal ConvLayer output tile.  The native graph then exports that tiled output
through its usual output ops, so consecutive probe words appear every 128 public
u32 words in `Y.raw` for the canonical 256^3 W4A16 reference.

This parser makes that stride explicit and avoids ad-hoc hexdump indexing.
Only the early entry arguments plus the output descriptor scalars have been
validated with this public-output stride so far.  Later samples are kept as
diagnostic evidence, but they are labelled unverified because the native
post-Conv output ops may have transformed them before export.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


MAGIC = 0x484D5850  # "HMXP"

FIELD_WORDS = [
    (0, "magic", "hex"),
    (1, "out_desc_ptr", "hex"),
    (2, "act_desc_ptr", "hex"),
    (3, "weight_ptr", "hex"),
    (4, "bias_ptr", "hex"),
    (5, "mask_ptr", "hex"),
    (6, "control_ptr", "hex"),
    (7, "out_table_ptr", "hex"),
    (8, "out_table_stride_dwords", "uint"),
    (9, "out_y_stride_words", "uint"),
    (10, "out_n_tiles_pow2", "uint"),
    (11, "out_m_total_minus_step", "int"),
    (12, "out_k_total_bytes", "uint"),
    (16, "act_table_ptr", "hex"),
    (17, "act_n_pairs", "uint"),
    (18, "act_table_y_stride_words", "uint"),
]

UNVERIFIED_FIELD_NAMES = {
    "act_table_y_stride_words",
}


def _as_i32(value: int) -> int:
    return struct.unpack("<i", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _read_words(path: Path, dtype: str) -> list[int]:
    raw = path.read_bytes()
    if dtype == "u16":
        if len(raw) % 2:
            raise ValueError(f"u16 raw length must be even, got {len(raw)} bytes")
        cells = struct.unpack(f"<{len(raw) // 2}H", raw)
        return [cells[i] | (cells[i + 1] << 16) for i in range(0, len(cells) - 1, 2)]
    if dtype == "u32":
        if len(raw) % 4:
            raise ValueError(f"u32 raw length must be a multiple of 4, got {len(raw)} bytes")
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))
    raise ValueError(f"unsupported dtype: {dtype}")


def _decode_record(public_words: list[int], stride_words: int, count: int) -> list[int]:
    if stride_words <= 0:
        raise ValueError("--stride-words must be positive")
    needed = (count - 1) * stride_words
    if needed >= len(public_words):
        raise ValueError(
            f"file too short for {count} strided words: need public word {needed}, "
            f"have {len(public_words)}"
        )
    return [public_words[i * stride_words] for i in range(count)]


def parse_probe(path: Path, stride_words: int, dtype: str) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    record = _decode_record(public_words, stride_words, 96)
    trusted_fields: dict[str, Any] = {}
    unverified_fields: dict[str, Any] = {}
    for word_index, name, kind in FIELD_WORDS:
        value = record[word_index]
        if kind == "int":
            value = _as_i32(value)
        if name in UNVERIFIED_FIELD_NAMES:
            unverified_fields[name] = value
        else:
            trusted_fields[name] = value
    return {
        "path": str(path),
        "dtype": dtype,
        "public_u32_words": len(public_words),
        "stride_words": stride_words,
        "record_u32_words": len(record),
        "trusted_fields": trusted_fields,
        "unverified_strided_fields": unverified_fields,
        "unverified_samples": {
            "mask_words": record[32:48],
            "out_table_sample": record[48:64],
            "act_table_sample": record[64:80],
            "weight_words": record[80:84],
            "bias_words": record[84:88],
            "control_words": record[88:92],
        },
    }


def _fmt(value: int, *, signed: bool = False) -> str:
    if signed:
        return f"{value}"
    return f"0x{value & 0xFFFFFFFF:08x}"


def print_human(parsed: dict[str, Any]) -> None:
    fields = parsed["trusted_fields"]
    magic = fields["magic"]
    ok = "ok" if magic == MAGIC else "unexpected"
    print(f"=== W4A16 native HNH entry probe: {parsed['path']} ===")
    print(
        f"dtype={parsed['dtype']} public_u32_words={parsed['public_u32_words']} "
        f"stride_words={parsed['stride_words']}"
    )
    print(f"magic=0x{magic:08x} ({ok})")
    print(
        "trusted_scope=entry args, output descriptor scalars, and the first two "
        "activation descriptor fields"
    )
    print()

    groups = [
        (
            "entry_args",
            [
                "out_desc_ptr",
                "act_desc_ptr",
                "weight_ptr",
                "bias_ptr",
                "mask_ptr",
                "control_ptr",
            ],
        ),
        (
            "out_desc",
            [
                "out_table_ptr",
                "out_table_stride_dwords",
                "out_y_stride_words",
                "out_n_tiles_pow2",
                "out_m_total_minus_step",
                "out_k_total_bytes",
            ],
        ),
        ("act_desc", ["act_table_ptr", "act_n_pairs"]),
    ]
    width = max(len(name) for _, names in groups for name in names)
    hex_names = {name for _, names in groups for name in names if name.endswith("_ptr")}
    for title, names in groups:
        print(f"{title}:")
        for name in names:
            value = fields[name]
            if name in hex_names:
                text = _fmt(value)
            else:
                text = str(value)
            print(f"  {name:<{width}} = {text}")
        print()

    unverified_fields = parsed["unverified_strided_fields"]
    if unverified_fields:
        print("unverified_strided_fields:")
        for name, value in unverified_fields.items():
            print(f"  {name} = {value}")
        print()

    print(
        "unverified_samples: public-output post-processing may have transformed "
        "these words; use them only after a pattern probe validates the mapping."
    )
    for title, words in parsed["unverified_samples"].items():
        print(f"{title}:")
        for i, word in enumerate(words):
            print(f"  [{i:02d}] = 0x{word:08x}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path, help="Y.raw from the hmx_entry_probe patched native run")
    parser.add_argument("--stride-words", type=int, default=128, help="public u32 stride between probe words")
    parser.add_argument("--dtype", choices=("u16", "u32"), default="u16", help="public raw storage dtype")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    parsed = parse_probe(args.raw_path, args.stride_words, args.dtype)
    if args.json:
        print(json.dumps(parsed, indent=2, sort_keys=True))
    else:
        print_human(parsed)


if __name__ == "__main__":
    main()
