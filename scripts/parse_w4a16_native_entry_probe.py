#!/usr/bin/env python3
"""Parse the native W4A16 HNH entry probe exported through public Y.raw.

The `/tmp/libQnnHtpV75Skel_hmx_entry_probe.so` diagnostic replaces
`hmx_v73_convhnh1x1_stride1` and writes a compact u32 record into the first
internal ConvLayer output tile.  The native graph then exports that tiled output
through its usual output ops, so consecutive probe words appear every 128 public
u32 words in `Y.raw` for the canonical 256^3 W4A16 reference.

This parser makes that mapping explicit and avoids ad-hoc hexdump indexing.
Use `--layout crouton512` for the current v3 probe.  That mapping was recovered
by a pure-assembly pattern probe and reconstructs the first 512 internal u32
words from public `Y.raw`.  The older `--layout stride` mode is kept for
historical v1 dumps and should not be used for table samples.
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
    (18, "act_desc_word2_unclassified", "hex"),
]


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


def _crouton512_public_index(record_word: int) -> int:
    group = record_word // 32
    lane = record_word % 32
    return lane * 128 + (group // 2) * 16 + (group & 1)


def _decode_record(public_words: list[int], layout: str, stride_words: int, count: int) -> list[int]:
    if stride_words <= 0:
        raise ValueError("--stride-words must be positive")
    if layout == "stride":
        indexes = [i * stride_words for i in range(count)]
    elif layout == "crouton512":
        indexes = [_crouton512_public_index(i) for i in range(count)]
    else:
        raise ValueError(f"unsupported layout: {layout}")

    needed = max(indexes)
    if needed >= len(public_words):
        raise ValueError(
            f"file too short for {count} {layout} words: need public word {needed}, "
            f"have {len(public_words)}"
        )
    return [public_words[i] for i in indexes]


def parse_probe(path: Path, layout: str, stride_words: int, dtype: str) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    record = _decode_record(public_words, layout, stride_words, 96)
    fields: dict[str, Any] = {}
    for word_index, name, kind in FIELD_WORDS:
        value = record[word_index]
        if kind == "int":
            value = _as_i32(value)
        fields[name] = value
    return {
        "path": str(path),
        "dtype": dtype,
        "layout": layout,
        "public_u32_words": len(public_words),
        "stride_words": stride_words,
        "record_u32_words": len(record),
        "fields": fields,
        "samples": {
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
    fields = parsed["fields"]
    magic = fields["magic"]
    ok = "ok" if magic == MAGIC else "unexpected"
    print(f"=== W4A16 native HNH entry probe: {parsed['path']} ===")
    print(
        f"dtype={parsed['dtype']} public_u32_words={parsed['public_u32_words']} "
        f"layout={parsed['layout']} stride_words={parsed['stride_words']}"
    )
    print(f"magic=0x{magic:08x} ({ok})")
    if parsed["layout"] == "crouton512":
        print("mapping=verified first-512-word internal-output to public-output map")
    else:
        print("mapping=legacy stride mode; table samples are not validated")
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
        ("act_desc", ["act_table_ptr", "act_n_pairs", "act_desc_word2_unclassified"]),
    ]
    width = max(len(name) for _, names in groups for name in names)
    hex_names = {
        name
        for _, names in groups
        for name in names
        if name.endswith("_ptr") or name.endswith("_unclassified")
    }
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

    print("samples:")
    for title, words in parsed["samples"].items():
        print(f"{title}:")
        for i, word in enumerate(words):
            print(f"  [{i:02d}] = 0x{word:08x}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path, help="Y.raw from the hmx_entry_probe patched native run")
    parser.add_argument(
        "--layout",
        choices=("crouton512", "stride"),
        default="crouton512",
        help="how probe words are mapped into public Y.raw",
    )
    parser.add_argument("--stride-words", type=int, default=128, help="public u32 stride between probe words")
    parser.add_argument("--dtype", choices=("u16", "u32"), default="u16", help="public raw storage dtype")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    parsed = parse_probe(args.raw_path, args.layout, args.stride_words, args.dtype)
    if args.json:
        print(json.dumps(parsed, indent=2, sort_keys=True))
    else:
        print_human(parsed)


if __name__ == "__main__":
    main()
