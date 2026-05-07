#!/usr/bin/env python3
"""Parse HmxW4A16TensorDump diagnostic output.

The dump op writes 64 u32 words into the first exported U16 block.  Each u32 is
encoded as two exported U16 cells: low halfword first, high halfword second.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


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


def _decode_u32(cells: list[int], word_index: int, cols: int, encoding: str) -> int:
    if encoding == "pairs":
        low_idx = word_index * 2
        high_idx = low_idx + 1
    elif encoding == "crouton32":
        group = word_index // 32
        within = word_index % 32
        low_idx = group * (2 * cols) + within
        high_idx = group * (2 * cols) + cols + within
    else:
        raise ValueError(f"unsupported encoding: {encoding}")
    if high_idx >= len(cells):
        raise ValueError(
            f"file too short for word {word_index}: need u16 index {high_idx}, "
            f"have {len(cells)} cells"
        )
    return cells[low_idx] | (cells[high_idx] << 16)


def _f32_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _shape(words: list[int], start: int) -> dict:
    rank = words[start]
    return {
        "rank": rank,
        "dims": words[start + 1 : start + 5],
    }


def parse_dump(raw_path: Path, cols: int, dtype: str, encoding: str) -> dict:
    cells = _read_u16_cells(raw_path.read_bytes(), dtype)
    words = [_decode_u32(cells, i, cols, encoding) for i in range(64)]
    return {
        "path": str(raw_path),
        "dtype": dtype,
        "cols": cols,
        "encoding": encoding,
        "magic": f"0x{words[0]:08x}",
        "version": words[1],
        "input_tensor_ptr": f"0x{words[2]:08x}",
        "output_tensor_ptr": f"0x{words[3]:08x}",
        "raw_data_ptr": f"0x{words[4]:08x}",
        "block_table_ptr": f"0x{words[5]:08x}",
        "block_table_len": words[6],
        "element_type": words[7],
        "layout": words[8],
        "placement": words[9],
        "quant_zero_offset": struct.unpack("<i", struct.pack("<I", words[10]))[0],
        "quant_step_bits": f"0x{words[11]:08x}",
        "quant_step": _f32_bits(words[11]),
        "shape": _shape(words, 12),
        "padded_shape": _shape(words, 17),
        "block_shape": _shape(words, 22),
        "padding": _shape(words, 27),
        "block_ptrs": [f"0x{word:08x}" for word in words[32:48]],
        "raw_tensor_words": [f"0x{word:08x}" for word in words[48:63]],
        "output_tensor_ptr_repeat": f"0x{words[63]:08x}",
        "words": [f"0x{word:08x}" for word in words],
    }


def print_human(parsed: dict) -> None:
    print(f"=== W4A16 tensor dump: {parsed['path']} ===")
    print(f"magic={parsed['magic']} version={parsed['version']}")
    for name in (
        "input_tensor_ptr",
        "output_tensor_ptr",
        "raw_data_ptr",
        "block_table_ptr",
        "block_table_len",
        "element_type",
        "layout",
        "placement",
        "quant_zero_offset",
        "quant_step",
    ):
        print(f"{name}: {parsed[name]}")
    for name in ("shape", "padded_shape", "block_shape", "padding"):
        shape = parsed[name]
        print(f"{name}: rank={shape['rank']} dims={shape['dims']}")
    print("block_ptrs:")
    for i, ptr in enumerate(parsed["block_ptrs"]):
        print(f"  [{i:02d}] {ptr}")
    print("raw_tensor_words:")
    for i, word in enumerate(parsed["raw_tensor_words"]):
        print(f"  [{i:02d}] {word}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--dtype", choices=("u16", "f32"), default="u16")
    parser.add_argument("--encoding", choices=("pairs", "crouton32"), default="pairs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    parsed = parse_dump(args.raw_path, args.cols, args.dtype, args.encoding)
    if args.json:
        print(json.dumps(parsed, indent=2, sort_keys=True))
    else:
        print_human(parsed)


if __name__ == "__main__":
    main()
