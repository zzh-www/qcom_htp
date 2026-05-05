#!/usr/bin/env python3
"""Parse the V73DEEP descriptor dump from a HMX_U8I8_DESC_DUMP run.

The op-pkg HMX_U8I8_DESC_DUMP branch writes 5 logical "rows" of 128 bytes each
into out[0..4] (after UntileToRowMajor). Each row starts with a u32
magic 0xD0DE000r and contains specific descriptor / shape / pointer data.

Usage:
    python scripts/parse_v73deep_desc_dump.py <Result_0/out.raw>
"""

import argparse
import struct
import sys
from pathlib import Path


MAGIC_BASE = 0xD0DE0000


def u32_at(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def hexdump(buf, off, n, group=4):
    out = []
    for i in range(0, n, group):
        word = buf[off + i:off + i + group]
        out.append(word.hex())
    return " ".join(out)


def parse_dump(raw_path: Path, n_cols: int, fp32: bool):
    raw = raw_path.read_bytes()
    if fp32:
        # qnn-net-run writes fp32 by default. Output uses scale=1.0, zp=0
        # (per gen_u8i8_chain.py quant_overrides), so fp32 cell ≈ u8 value.
        import numpy as np
        f = np.frombuffer(raw, dtype=np.float32)
        data = bytes(np.round(np.clip(f, 0, 255)).astype(np.uint8).tolist())
    else:
        data = raw

    if len(data) < n_cols * 5:
        print(f"warn: file too short ({len(data)} bytes), need at least {n_cols * 5}", file=sys.stderr)

    rows = []
    for r in range(5):
        off = r * n_cols
        if off + 128 > len(data):
            break
        rows.append(data[off:off + 128])

    print(f"=== V73DEEP descriptor dump from {raw_path} ===\n")

    for r, row in enumerate(rows):
        magic = u32_at(row, 0)
        expect = MAGIC_BASE | r
        marker = "✓" if magic == expect else f"✗ (got 0x{magic:08x}, want 0x{expect:08x})"
        print(f"--- Row {r}  magic 0x{magic:08x}  {marker} ---")

        if r == 0:
            # shape header + mask
            M, N, K = u32_at(row, 4), u32_at(row, 8), u32_at(row, 12)
            M_t, N_t, K_t = u32_at(row, 16), u32_at(row, 20), u32_at(row, 24)
            wt_pack = u32_at(row, 28)
            bias = u32_at(row, 32)
            mt_per_block = u32_at(row, 36)
            block_rows = u32_at(row, 40)
            act_blocks = u32_at(row, 44)
            out_blocks = u32_at(row, 48)
            print(f"  shape: M={M} N={N} K={K} → M_t={M_t} N_t={N_t} K_t={K_t}")
            print(f"  mt_per_block={mt_per_block} block_rows={block_rows}")
            print(f"  wt_pack=0x{wt_pack:08x} bias=0x{bias:08x}")
            print(f"  act_blocks=0x{act_blocks:08x} out_blocks=0x{out_blocks:08x}")
            print("  mask_buf[0..63]:")
            for i in range(0, 64, 4):
                v = u32_at(row, 64 + i)
                print(f"    +0x{i:02x} = 0x{v:08x}  ({v})")

        elif r == 1:
            # od + ad + extra_param
            od_table = u32_at(row, 4)
            od_stride = u32_at(row, 8)
            od_y = u32_at(row, 12)
            od_n_tiles = u32_at(row, 16)
            od_m = struct.unpack_from("<i", row, 20)[0]
            od_k = u32_at(row, 24)
            print(f"  od:")
            print(f"    out_table              = 0x{od_table:08x}")
            print(f"    out_table_stride_dwords= {od_stride}")
            print(f"    out_y_stride_words     = {od_y}")
            print(f"    n_tiles_pow2 (+0x0c)   = {od_n_tiles}  (→ r20 = {(od_n_tiles + 7) >> 3})")
            print(f"    m_total_minus_step (+0x10)= {od_m}")
            print(f"    k_total_bytes (+0x14)  = {od_k}  (→ r13_outer = {(od_k + 0x1f) >> 5})")
            ad_pairs = u32_at(row, 28)
            ad_n = u32_at(row, 32)
            ad_y = u32_at(row, 36)
            print(f"  ad:")
            print(f"    act_pairs              = 0x{ad_pairs:08x}")
            print(f"    n_act_pairs (+0x04)    = {ad_n}  (→ loop0 trip = {ad_n // 2})")
            print(f"    act_table_y_stride     = {ad_y}")
            print(f"  extra_param[16]:")
            for i in range(16):
                v = u32_at(row, 40 + i * 4)
                print(f"    [{i:2d}] = 0x{v:08x}  ({v})")

        elif r == 2:
            print(f"  act_tbl_all[0..30] (M_t × K_t pointer table):")
            for i in range(31):
                v = u32_at(row, 4 + i * 4)
                if v == 0:
                    break
                print(f"    [{i:2d}] = 0x{v:08x}")

        elif r == 3:
            print(f"  out_tbl_all[0..30] (M_t × N_t pointer table):")
            for i in range(31):
                v = u32_at(row, 4 + i * 4)
                if v == 0:
                    break
                print(f"    [{i:2d}] = 0x{v:08x}")

        elif r == 4:
            mtkt = u32_at(row, 4)
            ntkt = u32_at(row, 8)
            mtnt = u32_at(row, 12)
            cand1 = u32_at(row, 16)
            cand2 = u32_at(row, 20)
            cand3 = u32_at(row, 24)
            cand4 = u32_at(row, 28)
            print(f"  derived:")
            print(f"    M_t × K_t = {mtkt}  (act tile count)")
            print(f"    N_t × K_t = {ntkt}  (wt tile count)")
            print(f"    M_t × N_t = {mtnt}  (out tile count)")
            print(f"  wrapper-formula sd_tile_count candidates:")
            print(f"    HWD=(M,K,K): (M>>3)(K>>3)(K>>5) = {cand1}")
            print(f"    HWD=(K,N,N): (K>>3)(N>>3)(N>>5) = {cand2}")
            print(f"    HWD=(M,1,K): (M>>3)(1)(K>>5)    = {cand3}")
            print(f"    HWD=(1,K,N): (1)(K>>3)(N>>5)    = {cand4}")

        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_path", type=Path, help="Path to output raw file (e.g. Result_0/out.raw)")
    ap.add_argument("--N", type=int, default=256, help="output column count (= N)")
    ap.add_argument("--raw_u8", action="store_true", help="treat file as raw u8 bytes (default: fp32 from qnn-net-run)")
    args = ap.parse_args()
    parse_dump(args.raw_path, args.N, fp32=not args.raw_u8)


if __name__ == "__main__":
    main()
