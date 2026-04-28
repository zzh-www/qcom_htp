#!/usr/bin/env python3
"""Parse the V73DEEP per-region PMU probe output and fit a linear model.

Each row contains:
    [0..3]  magic 0xC0DE000r
    [4..7]  variant id (0..3)
    [8..11] n_tiles_pow2     (od.+0x0c)
    [12..15] k_total_bytes   (od.+0x14)
    [16..19] n_act_pairs     (ad.+0x04)
    [20..23] pkt_count
    [24..27] cyc_count
    [28..31] inst_count
    [32..35] r20  (loop1 trip)
    [36..39] r13  (initial)
    [40..43] r13_eff (effective outer iters = (r13+1)/2)
    [44..47] r28  (loop0 trip)

Fits packets = A + r13_eff * (B + r20 * (C + r28 * D))
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_path", type=Path)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--raw_u8", action="store_true")
    args = ap.parse_args()

    raw = args.raw_path.read_bytes()
    if args.raw_u8:
        data = raw
    else:
        f = np.frombuffer(raw, dtype=np.float32)
        data = bytes(np.round(np.clip(f, 0, 255)).astype(np.uint8).tolist())

    rows = []
    for r in range(4):
        off = r * args.N
        if off + 64 > len(data):
            break
        row = data[off:off + 128]
        magic = struct.unpack_from("<I", row, 0)[0]
        if (magic & 0xFFFFFFF0) != 0xC0DE0000:
            print(f"row {r}: bad magic 0x{magic:08x}", file=sys.stderr)
            continue
        rows.append({
            "v":           struct.unpack_from("<I", row, 4)[0],
            "n_tiles":     struct.unpack_from("<I", row, 8)[0],
            "k_total":     struct.unpack_from("<I", row, 12)[0],
            "n_act":       struct.unpack_from("<I", row, 16)[0],
            "pkts":        struct.unpack_from("<I", row, 20)[0],
            "cyc":         struct.unpack_from("<I", row, 24)[0],
            "inst":        struct.unpack_from("<I", row, 28)[0],
            "r20":         struct.unpack_from("<I", row, 32)[0],
            "r13":         struct.unpack_from("<I", row, 36)[0],
            "r13_eff":     struct.unpack_from("<I", row, 40)[0],
            "r28":         struct.unpack_from("<I", row, 44)[0],
            "pkts_t0":     struct.unpack_from("<I", row, 48)[0],
        })

    if not rows:
        print("no rows decoded", file=sys.stderr)
        sys.exit(1)

    print(f"=== V73DEEP per-region PMU probe @ {args.raw_path} ===\n")
    print(f"{'V':>2}  {'n_tiles':>8} {'k_total':>8} {'n_act':>5}  "
          f"{'r20':>4} {'r13e':>5} {'r28':>4}  "
          f"{'pkts(any)':>9} {'pkts(T0)':>9} {'cyc':>7} {'inst':>8}  cyc/pkt  T0/any  pkt(T0)/MAC")
    for r in rows:
        macs = r['r13_eff'] * r['r20'] * r['r28'] * 2
        cyc_per_pkt = r['cyc'] / r['pkts'] if r['pkts'] else 0
        t0_ratio = r['pkts_t0'] / r['pkts'] if r['pkts'] else 0
        pkt_per_mac = r['pkts_t0'] / macs if macs else 0
        print(f"{r['v']:>2}  {r['n_tiles']:>8} {r['k_total']:>8} {r['n_act']:>5}  "
              f"{r['r20']:>4} {r['r13_eff']:>5} {r['r28']:>4}  "
              f"{r['pkts']:>9} {r['pkts_t0']:>9} {r['cyc']:>7} {r['inst']:>8}  "
              f"{cyc_per_pkt:>6.2f}  {t0_ratio:>5.2f}  {pkt_per_mac:>6.2f}")

    if len(rows) < 4:
        print(f"\nneed 4 variants for linear fit; got {len(rows)}", file=sys.stderr)
        return

    # Solve: pkts = A + r13_eff * (B + r20 * (C + r28 * D))
    # = A + r13_eff*B + r13_eff*r20*C + r13_eff*r20*r28*D
    # 4 equations, 4 unknowns:
    M = np.zeros((4, 4), dtype=float)
    y = np.zeros(4, dtype=float)
    for i, r in enumerate(rows):
        M[i, 0] = 1.0
        M[i, 1] = float(r['r13_eff'])
        M[i, 2] = float(r['r13_eff'] * r['r20'])
        M[i, 3] = float(r['r13_eff'] * r['r20'] * r['r28'])
        y[i] = float(r['pkts'])

    def fit(label, key):
        y_l = np.array([float(r[key]) for r in rows])
        try:
            coeffs = np.linalg.solve(M, y_l)
            A, B, C, D = coeffs
            print(f"\n=== linear fit ({label}) pkts = A + r13e*(B + r20*(C + r28*D)) ===")
            print(f"  A (prologue+epilogue, per-call fixed)         = {A:>8.2f}")
            print(f"  B (per-outer overhead, bias+setup)             = {B:>8.2f}")
            print(f"  C (per-loop1 overhead, drain prep+stores)      = {C:>8.2f}")
            print(f"  D (per-loop0 packet count, K-MAC body)         = {D:>8.2f}")
            pred = M @ coeffs
            print(f"  residual = {(y_l - pred).round(2).tolist()}")
            return A, B, C, D
        except np.linalg.LinAlgError as e:
            print(f"linear solve failed: {e}", file=sys.stderr)
            return None

    fit("ANY threads", "pkts")
    res = fit("T0 only", "pkts_t0")
    A, B, C, D = res if res is not None else (0, 0, 0, 0)

    if False:
        pass

    if rows and all(k in rows[0] for k in ('pkts_t0',)):
        # native chrometrace pkt count at 256³ ≈ 346 (assumed T0-thread units)
        baseline_v0 = rows[0]
        native_pkts = 346
        v0_t0 = baseline_v0['pkts_t0']
        v0_any = baseline_v0['pkts']
        gap_t0 = v0_t0 - native_pkts
        print(f"\n=== gap analysis (vs native chrometrace 346 pkts @ 256³) ===")
        print(f"  V0 (our setup) ANY = {v0_any} pkts, T0 = {v0_t0} pkts")
        print(f"  native @ 256³      = {native_pkts} pkts")
        print(f"  gap (T0)           = {gap_t0} pkts")
        if A is not None:
            print(f"  per-outer (T0):    B*4 = {B*4:.0f}")
            print(f"  per-loop1 (T0):    C*32 = {C*32:.0f}")
            print(f"  per-loop0 (T0):    D*128 = {D*128:.0f}")
            print(f"  prologue (T0):     A = {A:.0f}")


if __name__ == "__main__":
    main()
