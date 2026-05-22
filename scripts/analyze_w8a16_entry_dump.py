#!/usr/bin/env python3
"""Decode a W8A16 custom-op entry dump.

The dump is emitted only when the W8A16 package is built with
``-DHMX_W8A16_ENTRY_DUMP``.  It records the wrapper-side descriptor state and
one 512-byte A16 bias/control tile before entering the HMX body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAGIC = 0x48385845  # H8XE


def u32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 4], "little", signed=False)


def i32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 4], "little", signed=True)


def find_dump(buf: bytes) -> int:
    magic = MAGIC.to_bytes(4, "little")
    off = buf.find(magic)
    if off < 0:
        raise ValueError("missing H8XE dump magic")
    return off


def find_compacted_dump(buf: bytes) -> int:
    """Find OutputSlice-compacted H8XE dumps.

    A Crouton_16 output slice exposes the low u16 of each 32-bit diagnostic
    field as a contiguous flat u16 stream.  In that mode the full 32-bit magic
    is not present, but the low halfword 0x5845 appears at the dump start.
    """

    needle = (MAGIC & 0xFFFF).to_bytes(2, "little")
    off = buf.find(needle)
    if off < 0:
        raise ValueError("missing compacted H8XE dump magic")
    return off


def channel_offsets(channel: int) -> dict[str, int]:
    tile = channel // 32
    c = channel % 32
    parity = c & 1
    lane = c // 2
    half_base = parity * 256
    control = tile * 512 + half_base + 8 * lane
    effective = tile * 512 + half_base + 128 + 8 * lane
    return {
        "tile": tile,
        "channel_in_tile": c,
        "parity": parity,
        "lane": lane,
        "control": control,
        "effective": effective,
    }


def parse_compacted_dump(out_buf: bytes, native: np.ndarray, channel: int) -> dict:
    off = find_compacted_dump(out_buf)
    fields = np.frombuffer(out_buf[off:], dtype="<u2")

    def field(byte_offset: int) -> int:
        return int(fields[byte_offset // 4])

    header_names = [
        "magic_low16",
        "S",
        "M_t",
        "N_t",
        "K_t",
        "mt_groups",
        "act_table_stride",
        "out_table_stride",
        "out_desc_table_stride_dwords",
        "out_desc_y_stride_words",
        "out_desc_n_tiles_pow2",
        "out_desc_m_total_minus_step",
        "out_desc_k_total_bytes",
        "act_desc_n_act_pairs",
        "act_desc_y_stride_words",
        "bias_ptr_low16",
        "weight_ptr_low16",
        "act_table_ptr_low16",
        "out_table_ptr_low16",
        "extra0",
        "extra1",
        "extra2",
    ]
    header = {name: int(fields[idx]) for idx, name in enumerate(header_names)}
    header["selected_tile"] = field(96)
    header["selected_tile_offset"] = field(100)
    header["dump_channel"] = field(104)
    header["dump_tile"] = field(108)
    header["dump_lane"] = field(112)
    header["dump_parity"] = field(116)
    header["dump_effective_slot_offset"] = field(120)

    effective_bytes = bytes(field(32 + i * 4) & 0xFF for i in range(8))
    control_bytes = bytes(field(64 + i * 4) & 0xFF for i in range(8))

    ch = channel_offsets(channel)
    native_control = native[ch["control"] : ch["control"] + 8].tobytes()
    native_effective = native[ch["effective"] : ch["effective"] + 8].tobytes()

    return {
        "dump_format": "output_slice_compacted_low16",
        "dump_offset": off,
        "header": header,
        "native_sidecar_bytes": int(native.size),
        "channel": channel,
        "channel_offsets": ch,
        "channel_control_dump_hex": control_bytes.hex(),
        "channel_control_native_hex": native_control.hex(),
        "channel_control_match": control_bytes == native_control,
        "channel_effective_dump_hex": effective_bytes.hex(),
        "channel_effective_native_hex": native_effective.hex(),
        "channel_effective_match": effective_bytes == native_effective,
        "channel_effective_dump": int.from_bytes(effective_bytes[:4], "little", signed=True),
        "channel_effective_native": int.from_bytes(native_effective[:4], "little", signed=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-raw", required=True)
    ap.add_argument("--native-sidecar-raw", required=True)
    ap.add_argument("--channel", type=int, default=243)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    out_buf = Path(args.output_raw).read_bytes()
    native = np.fromfile(args.native_sidecar_raw, dtype=np.uint8)
    if native.size % 512:
        raise ValueError(f"native sidecar size is not 512B aligned: {native.size}")

    try:
        off = find_dump(out_buf)
    except ValueError:
        summary = parse_compacted_dump(out_buf, native, args.channel)
        text = json.dumps(summary, indent=2, sort_keys=True)
        print(text)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
        return

    dump = out_buf[off : off + 2048]

    header_names = [
        "magic",
        "S",
        "M_t",
        "N_t",
        "K_t",
        "mt_groups",
        "act_table_stride",
        "out_table_stride",
        "out_desc_table_stride_dwords",
        "out_desc_y_stride_words",
        "out_desc_n_tiles_pow2",
        "out_desc_m_total_minus_step",
        "out_desc_k_total_bytes",
        "act_desc_n_act_pairs",
        "act_desc_y_stride_words",
        "bias_ptr_low32",
        "weight_ptr_low32",
        "act_table_ptr_low32",
        "out_table_ptr_low32",
        "extra0",
        "extra1",
        "extra2",
    ]
    header = {name: u32(dump, idx * 4) for idx, name in enumerate(header_names)}
    header["selected_tile"] = u32(dump, 96)
    header["selected_tile_offset"] = u32(dump, 100)
    mask_words = [u32(dump, 128 + i * 4) for i in range(16)]

    selected_tile = header["selected_tile"]
    tile_start = selected_tile * 512
    tile = np.frombuffer(dump[256 : 256 + 512], dtype=np.uint8).copy()
    native_tile = native[tile_start : tile_start + 512].copy()
    tile_match = int((tile == native_tile).sum())

    ch = channel_offsets(args.channel)
    if ch["tile"] != selected_tile:
        raise ValueError(
            f"channel {args.channel} is in tile {ch['tile']}, "
            f"but dump selected tile {selected_tile}"
        )
    rel_control = ch["control"] - tile_start
    rel_eff = ch["effective"] - tile_start
    dump_control = tile[rel_control : rel_control + 8].copy()
    native_control = native_tile[rel_control : rel_control + 8].copy()
    dump_eff_bytes = tile[rel_eff : rel_eff + 4].copy()
    native_eff_bytes = native_tile[rel_eff : rel_eff + 4].copy()
    dump_eff = int(dump_eff_bytes.view("<i4")[0])
    native_eff = int(native_eff_bytes.view("<i4")[0])

    summary = {
        "dump_offset": off,
        "header": header,
        "mask_words_hex": [f"0x{v:08x}" for v in mask_words],
        "native_sidecar_bytes": int(native.size),
        "selected_tile_match": f"{tile_match}/512",
        "selected_tile_mismatch_offsets": [
            int(i) for i in np.where(tile != native_tile)[0][:32]
        ],
        "channel": args.channel,
        "channel_offsets": ch,
        "channel_control_dump_hex": dump_control.tobytes().hex(),
        "channel_control_native_hex": native_control.tobytes().hex(),
        "channel_control_match": bool(np.array_equal(dump_control, native_control)),
        "channel_effective_dump": dump_eff,
        "channel_effective_native": native_eff,
        "channel_effective_match": dump_eff == native_eff,
    }

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
