#!/usr/bin/env python3
"""Parse the HMX_W4A16_DESC_DUMP payload exported through Crouton16 output.

The W4A16 descriptor dump writes little-endian u32 words into the first output
Crouton16 block.  qnn-net-run exports that block as row-major u16 cells, where
each group of 32 logical u32 words is split into low and high u16 halves:

    low  half index = group * (2 * cols) + within
    high half index = group * (2 * cols) + cols + within

Use this parser instead of ad-hoc hexdumps when comparing custom descriptor
state with native wrapper evidence.

Words 48..63 contain the 16-entry table sample selected at build time by
HMX_W4A16_DESC_DUMP_TABLE_SELECT.
"""

import argparse
import json
import math
import struct
from pathlib import Path


FIELD_WORDS = [
    (0, "magic", "hex"),
    (1, "S", "uint"),
    (2, "M_t", "uint"),
    (3, "N_t", "uint"),
    (4, "K_t", "uint"),
    (5, "mt_per_block", "uint"),
    (6, "act_table_ptr", "hex"),
    (7, "out_table_ptr", "hex"),
    (8, "out_table_stride_dwords", "uint"),
    (9, "out_y_stride_words", "uint"),
    (10, "n_tiles_pow2", "uint"),
    (11, "m_total_minus_step", "int"),
    (12, "k_total_bytes", "uint"),
    (13, "n_act_pairs", "uint"),
    (14, "act_table_y_stride_words", "uint"),
    (15, "mt_groups", "uint"),
    (16, "act_table_storage_stride", "uint"),
    (17, "out_table_storage_stride", "uint"),
    (18, "act_entries", "uint"),
    (19, "out_entries", "uint"),
    (20, "act_block_entries", "uint"),
    (21, "out_block_entries", "uint"),
    (22, "act_ptr0", "hex"),
    (23, "act_ptr1", "hex"),
    (24, "out_ptr0", "hex"),
    (25, "out_ptr1", "hex"),
    (26, "weight_word0", "hex"),
    (27, "weight_word1", "hex"),
    (28, "bias_word0", "hex"),
    (29, "bias_word1", "hex"),
    (30, "extra_param0", "uint"),
    (31, "extra_param1", "uint"),
]


def _as_i32(value: int) -> int:
    return struct.unpack("<i", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _read_u16_cells(raw: bytes, dtype: str) -> list[int]:
    if dtype == "u16":
        if len(raw) % 2:
            raise ValueError(f"u16 raw length must be even, got {len(raw)} bytes")
        return list(struct.unpack(f"<{len(raw) // 2}H", raw))
    if dtype == "f32":
        if len(raw) % 4:
            raise ValueError(f"f32 raw length must be a multiple of 4, got {len(raw)} bytes")
        floats = struct.unpack(f"<{len(raw) // 4}f", raw)
        return [max(0, min(65535, int(round(v)))) for v in floats]
    raise ValueError(f"unsupported dtype: {dtype}")


def _decode_u32(cells: list[int], word_index: int, cols: int) -> int:
    group = word_index // 32
    within = word_index % 32
    low_idx = group * (2 * cols) + within
    high_idx = group * (2 * cols) + cols + within
    if high_idx >= len(cells):
        raise ValueError(
            f"file too short for word {word_index}: need u16 index {high_idx}, "
            f"have {len(cells)} cells"
        )
    return cells[low_idx] | (cells[high_idx] << 16)


def parse_dump(raw_path: Path, cols: int, dtype: str) -> dict:
    if cols < 32:
        raise ValueError("--cols must be at least 32")
    cells = _read_u16_cells(raw_path.read_bytes(), dtype)
    words = [_decode_u32(cells, i, cols) for i in range(64)]

    fields = {}
    for word_index, name, kind in FIELD_WORDS:
        value = words[word_index]
        if kind == "int":
            value = _as_i32(value)
        fields[name] = value

    mask_words = words[32:48]
    return {
        "path": str(raw_path),
        "dtype": dtype,
        "cols": cols,
        "u16_cells": len(cells),
        "logical_u32_words_decoded": len(words),
        "fields": fields,
        "mask_words": mask_words,
        "table_sample": words[48:64],
    }


def _fmt_value(name: str, value: int) -> str:
    if (
        name.endswith("_ptr")
        or name.endswith("_word0")
        or name.endswith("_word1")
        or name in {"magic", "act_ptr0", "act_ptr1", "out_ptr0", "out_ptr1"}
    ):
        return f"0x{value & 0xFFFFFFFF:08x}"
    return str(value)


def print_human(parsed: dict) -> None:
    fields = parsed["fields"]
    magic = fields["magic"]
    ok = "ok" if magic == 0x48385844 else "unexpected"
    print(f"=== W4A16 descriptor dump: {parsed['path']} ===")
    print(f"dtype={parsed['dtype']} cols={parsed['cols']} cells={parsed['u16_cells']}")
    print(f"magic=0x{magic:08x} ({ok})")
    print()

    groups = [
        ("shape", ["S", "M_t", "N_t", "K_t", "mt_per_block", "mt_groups"]),
        (
            "out_desc",
            [
                "out_table_ptr",
                "out_table_stride_dwords",
                "out_y_stride_words",
                "n_tiles_pow2",
                "m_total_minus_step",
                "k_total_bytes",
            ],
        ),
        ("act_desc", ["act_table_ptr", "n_act_pairs", "act_table_y_stride_words"]),
        (
            "tables",
            [
                "act_table_storage_stride",
                "out_table_storage_stride",
                "act_entries",
                "out_entries",
                "act_block_entries",
                "out_block_entries",
                "act_ptr0",
                "act_ptr1",
                "out_ptr0",
                "out_ptr1",
            ],
        ),
        (
            "payload",
            [
                "weight_word0",
                "weight_word1",
                "bias_word0",
                "bias_word1",
                "extra_param0",
                "extra_param1",
            ],
        ),
    ]
    width = max(len(name) for _, names in groups for name in names)
    for title, names in groups:
        print(f"{title}:")
        for name in names:
            print(f"  {name:<{width}} = {_fmt_value(name, fields[name])}")
        print()

    print("mask_words:")
    for i, word in enumerate(parsed["mask_words"]):
        print(f"  [{i:02d}] = 0x{word:08x} ({word})")

    print()
    print("table_sample:")
    print("  selected by HMX_W4A16_DESC_DUMP_TABLE_SELECT")
    for i, word in enumerate(parsed["table_sample"]):
        print(f"  [{i:02d}] = 0x{word:08x}")

    n_tiles = fields["n_tiles_pow2"]
    k_total = fields["k_total_bytes"]
    print()
    print("derived:")
    print(f"  deep_n_loop_hint = (n_tiles_pow2 + 3) >> 2 = {(n_tiles + 3) >> 2}")
    print(f"  k_outer_hint     = (k_total_bytes + 31) >> 5 = {(k_total + 31) >> 5}")
    if fields["act_table_storage_stride"] and fields["out_table_storage_stride"]:
        rows_from_act = math.ceil(fields["act_entries"] / fields["act_table_storage_stride"])
        rows_from_out = math.ceil(fields["out_entries"] / fields["out_table_storage_stride"])
        print(f"  table_rows       = act {rows_from_act}, out {rows_from_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path, help="device_out/out.raw from HMX_W4A16_DESC_DUMP")
    parser.add_argument("--cols", type=int, default=256, help="row-major output column count in u16 cells")
    parser.add_argument("--dtype", choices=("u16", "f32"), default="u16", help="qnn-net-run output raw dtype")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    parsed = parse_dump(args.raw_path, args.cols, args.dtype)
    if args.json:
        print(json.dumps(parsed, indent=2, sort_keys=True))
    else:
        print_human(parsed)


if __name__ == "__main__":
    main()
