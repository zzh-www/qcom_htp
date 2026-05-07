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

Supported record magics are `HMXP` for entry samples, `HMXB` for the active
prebuilt base record, `HMXT` for table-memory samples, `HMXA` for activation
table[0] data samples, and `HMXV` for per-activation-entry value samples.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


MAGIC_ENTRY = 0x484D5850  # "HMXP"
MAGIC_BASE_RECORD = 0x484D5842  # "HMXB"
MAGIC_TABLE = 0x484D5854  # "HMXT"
MAGIC_ACT_TILE = 0x484D5841  # "HMXA"
MAGIC_ACT_ENTRIES = 0x484D5856  # "HMXV"

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


def _parse_entry_record(path: Path, dtype: str, layout: str, stride_words: int, record: list[int]) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    fields: dict[str, Any] = {}
    for word_index, name, kind in FIELD_WORDS:
        value = record[word_index]
        if kind == "int":
            value = _as_i32(value)
        fields[name] = value
    return {
        "path": str(path),
        "record_kind": "entry",
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


def _parse_base_record(path: Path, dtype: str, layout: str, stride_words: int, record: list[int]) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    args = {
        "r31_return": record[1],
        "out_desc_arg": record[2],
        "act_desc_arg": record[3],
        "weight_arg": record[4],
        "bias_arg": record[5],
        "mask_arg": record[6],
        "control_arg": record[7],
    }
    base_words = record[16:52]
    fields = {
        "magic": record[0],
        **args,
        "base_ptr_from_out_desc": (record[2] - 0x28) & 0xFFFFFFFF,
        "weight_ptr": base_words[2],
        "bias_ptr": base_words[3],
        "act_table_ptr": base_words[4],
        "act_n_pairs": base_words[5],
        "act_y_stride_words": base_words[6],
        "out_table_ptr": base_words[10],
        "out_table_stride_dwords": base_words[11],
        "out_y_stride_words": base_words[12],
        "out_n_tiles_pow2": base_words[13],
        "out_m_total_minus_step": _as_i32(base_words[14]),
        "out_k_total_bytes": base_words[15],
        "control_ptr_from_base_0x80": base_words[32],
    }
    return {
        "path": str(path),
        "record_kind": "base",
        "dtype": dtype,
        "layout": layout,
        "public_u32_words": len(public_words),
        "stride_words": stride_words,
        "record_u32_words": len(record),
        "fields": fields,
        "samples": {
            "entry_args": [record[i] for i in range(2, 8)],
            "base_words_0x00_0x8c": base_words,
            "act_desc_words": base_words[4:10],
            "out_desc_words": base_words[10:16],
            "mask_words": base_words[18:34],
            "post_mask_words": base_words[34:36],
        },
    }


def _parse_table_record(path: Path, dtype: str, layout: str, stride_words: int, record: list[int]) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    fields = {
        "magic": record[0],
        "r31_return": record[1],
        "out_desc_arg": record[2],
        "act_desc_arg": record[3],
        "weight_arg": record[4],
        "bias_arg": record[5],
        "mask_arg": record[6],
        "control_arg": record[7],
        "out_table_ptr": record[8],
        "act_table_ptr": record[9],
        "out_table_stride_dwords": record[10],
        "out_y_stride_words": record[11],
        "out_n_tiles_pow2": record[12],
        "act_n_pairs": record[13],
        "act_y_stride_words": record[14],
        "control_word0_sample": record[15],
    }
    return {
        "path": str(path),
        "record_kind": "table",
        "dtype": dtype,
        "layout": layout,
        "public_u32_words": len(public_words),
        "stride_words": stride_words,
        "record_u32_words": len(record),
        "fields": fields,
        "samples": {
            "out_table_128": record[16:144],
            "act_table_128": record[144:272],
        },
    }


def _parse_compact_header(record: list[int]) -> dict[str, int]:
    return {
        "magic": record[0],
        "r31_return": record[1],
        "out_desc_arg": record[2],
        "act_desc_arg": record[3],
        "weight_arg": record[4],
        "bias_arg": record[5],
        "mask_arg": record[6],
        "control_arg": record[7],
        "out_table_ptr": record[8],
        "act_table_ptr": record[9],
        "act_n_pairs": record[10],
        "act_y_stride_words": record[11],
        "out_table_stride_dwords": record[12],
        "out_y_stride_words": record[13],
        "out_n_tiles_pow2": record[14],
        "out_k_total_bytes": record[15],
    }


def _u16_pairs(words: list[int]) -> list[list[int]]:
    return [[word & 0xFFFF, word >> 16] for word in words]


def _parse_act_tile_record(path: Path, dtype: str, layout: str, stride_words: int, record: list[int]) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    fields = _parse_compact_header(record)
    data_words = record[80:272]
    return {
        "path": str(path),
        "record_kind": "act_tile",
        "dtype": dtype,
        "layout": layout,
        "public_u32_words": len(public_words),
        "stride_words": stride_words,
        "record_u32_words": len(record),
        "fields": fields,
        "samples": {
            "act_table_64": record[16:80],
            "act_table0_data_192_u32": data_words,
            "act_table0_data_192_u16_pairs": _u16_pairs(data_words),
        },
    }


def _parse_act_entries_record(path: Path, dtype: str, layout: str, stride_words: int, record: list[int]) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    fields = _parse_compact_header(record)
    entries = []
    for i in range(64):
        base = 16 + i * 4
        sample_words = record[base + 1 : base + 4]
        entries.append(
            {
                "index": i,
                "ptr": record[base],
                "first3_u32": sample_words,
                "first6_u16": [half for pair in _u16_pairs(sample_words) for half in pair],
            }
        )
    return {
        "path": str(path),
        "record_kind": "act_entries",
        "dtype": dtype,
        "layout": layout,
        "public_u32_words": len(public_words),
        "stride_words": stride_words,
        "record_u32_words": len(record),
        "fields": fields,
        "samples": {
            "act_entries_64": entries,
        },
    }


def parse_probe(
    path: Path,
    layout: str,
    stride_words: int,
    dtype: str,
    record_kind: str,
) -> dict[str, Any]:
    public_words = _read_words(path, dtype)
    header = _decode_record(public_words, layout, stride_words, 1)
    if record_kind == "auto":
        if header[0] == MAGIC_ACT_TILE:
            record_kind = "act_tile"
        elif header[0] == MAGIC_ACT_ENTRIES:
            record_kind = "act_entries"
        elif header[0] == MAGIC_TABLE:
            record_kind = "table"
        elif header[0] == MAGIC_BASE_RECORD:
            record_kind = "base"
        else:
            record_kind = "entry"

    record_words = 272 if record_kind in ("table", "act_tile", "act_entries") else 96
    record = _decode_record(public_words, layout, stride_words, record_words)
    if record_kind == "entry":
        return _parse_entry_record(path, dtype, layout, stride_words, record)
    if record_kind == "base":
        return _parse_base_record(path, dtype, layout, stride_words, record)
    if record_kind == "table":
        return _parse_table_record(path, dtype, layout, stride_words, record)
    if record_kind == "act_tile":
        return _parse_act_tile_record(path, dtype, layout, stride_words, record)
    if record_kind == "act_entries":
        return _parse_act_entries_record(path, dtype, layout, stride_words, record)
    raise ValueError(f"unsupported record kind: {record_kind}")


def _fmt(value: int, *, signed: bool = False) -> str:
    if signed:
        return f"{value}"
    return f"0x{value & 0xFFFFFFFF:08x}"


def print_human(parsed: dict[str, Any]) -> None:
    fields = parsed["fields"]
    magic = fields["magic"]
    kind = parsed.get("record_kind", "entry")
    expected_magic = {
        "entry": MAGIC_ENTRY,
        "base": MAGIC_BASE_RECORD,
        "table": MAGIC_TABLE,
        "act_tile": MAGIC_ACT_TILE,
        "act_entries": MAGIC_ACT_ENTRIES,
    }[kind]
    ok = "ok" if magic == expected_magic else "unexpected"
    print(f"=== W4A16 native HNH {kind} probe: {parsed['path']} ===")
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

    if kind == "base":
        print("entry_args:")
        for name in (
            "r31_return",
            "out_desc_arg",
            "act_desc_arg",
            "weight_arg",
            "bias_arg",
            "mask_arg",
            "control_arg",
            "base_ptr_from_out_desc",
        ):
            print(f"  {name:<26} = {_fmt(fields[name])}")
        print()

        print("native_base_record:")
        for name in (
            "weight_ptr",
            "bias_ptr",
            "act_table_ptr",
            "act_n_pairs",
            "act_y_stride_words",
            "out_table_ptr",
            "out_table_stride_dwords",
            "out_y_stride_words",
            "out_n_tiles_pow2",
            "out_m_total_minus_step",
            "out_k_total_bytes",
            "control_ptr_from_base_0x80",
        ):
            value = fields[name]
            if name.endswith("_ptr") or name.endswith("_arg") or name.endswith("_0x80"):
                text = _fmt(value)
            else:
                text = str(value)
            print(f"  {name:<26} = {text}")
        print()

        print("samples:")
        for title, words in parsed["samples"].items():
            print(f"{title}:")
            for i, word in enumerate(words):
                print(f"  [{i:02d}] = 0x{word:08x}")
            print()
        return

    if kind == "table":
        print("entry_args:")
        for name in (
            "r31_return",
            "out_desc_arg",
            "act_desc_arg",
            "weight_arg",
            "bias_arg",
            "mask_arg",
            "control_arg",
        ):
            print(f"  {name:<26} = {_fmt(fields[name])}")
        print()

        print("native_table_header:")
        for name in (
            "out_table_ptr",
            "act_table_ptr",
            "out_table_stride_dwords",
            "out_y_stride_words",
            "out_n_tiles_pow2",
            "act_n_pairs",
            "act_y_stride_words",
            "control_word0_sample",
        ):
            value = fields[name]
            if name.endswith("_ptr") or name.endswith("_sample"):
                text = _fmt(value)
            else:
                text = str(value)
            print(f"  {name:<26} = {text}")
        print()

        print("samples:")
        for title, words in parsed["samples"].items():
            print(f"{title}:")
            for base in range(0, len(words), 8):
                chunk = " ".join(f"0x{word:08x}" for word in words[base : base + 8])
                print(f"  [{base:03d}] {chunk}")
            print()
        return

    if kind in ("act_tile", "act_entries"):
        print("entry_args:")
        for name in (
            "r31_return",
            "out_desc_arg",
            "act_desc_arg",
            "weight_arg",
            "bias_arg",
            "mask_arg",
            "control_arg",
        ):
            print(f"  {name:<26} = {_fmt(fields[name])}")
        print()

        print("native_compact_header:")
        for name in (
            "out_table_ptr",
            "act_table_ptr",
            "act_n_pairs",
            "act_y_stride_words",
            "out_table_stride_dwords",
            "out_y_stride_words",
            "out_n_tiles_pow2",
            "out_k_total_bytes",
        ):
            value = fields[name]
            if name.endswith("_ptr"):
                text = _fmt(value)
            else:
                text = str(value)
            print(f"  {name:<26} = {text}")
        print()

        if kind == "act_entries":
            print("act_entries_64:")
            for entry in parsed["samples"]["act_entries_64"]:
                vals = ", ".join(str(v) for v in entry["first6_u16"])
                print(f"  [{entry['index']:02d}] ptr={_fmt(entry['ptr'])} first6_u16=[{vals}]")
            print()
            return

        print("samples:")
        for title, words in parsed["samples"].items():
            print(f"{title}:")
            if title.endswith("_u16_pairs"):
                for base in range(0, len(words), 8):
                    chunk = " ".join(f"[{a},{b}]" for a, b in words[base : base + 8])
                    print(f"  [{base:03d}] {chunk}")
            else:
                for base in range(0, len(words), 8):
                    chunk = " ".join(f"0x{word:08x}" for word in words[base : base + 8])
                    print(f"  [{base:03d}] {chunk}")
            print()
        return

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
    parser.add_argument(
        "--record-kind",
        choices=("auto", "entry", "base", "table", "act_tile", "act_entries"),
        default="auto",
        help="probe record format; auto selects known HMX* probe records by magic",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    parsed = parse_probe(args.raw_path, args.layout, args.stride_words, args.dtype, args.record_kind)
    if args.json:
        print(json.dumps(parsed, indent=2, sort_keys=True))
    else:
        print_human(parsed)


if __name__ == "__main__":
    main()
