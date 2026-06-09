#!/usr/bin/env python3
"""CI gate: w16a16 is byte-exactly two 8-bit-weight passes (2x w8a16).

The HMX array is an integer byte-MAC and int16xint16 overflows the int32 accumulator, so a
w16a16 MatMul (uint16 act x int16 weight) is lowered to two int8-weight passes: the int16
weight is split into a signed high byte and an unsigned low byte, each multiplied against the
int16 activation, drained separately, and combined with a x256 shift. This gate proves the
split + matmul identity are byte-exact and that the high pass overflows int32 (so the two
passes must drain separately). Principle + usage: docs/w16a16_is_two_w8a16.md

  python3 scripts/verify_w16a16_two_w8a16.py [--M 64 --K 64 --N 64 --seed ...]
"""
from __future__ import annotations
import argparse

import numpy as np

ACT_ZP = 32768


def decompose_int16_to_bytes(q16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2's-complement byte split, exact for all int16: q16 == hi*256 + lo.
      hi = q16 >> 8   (int8,  [-128,127])   arithmetic high byte
      lo = q16 & 0xff (uint8, [0,255])      unsigned low byte
    A symmetric (int8,int8) split cannot cover the full int16 range, so the low pass uses an
    unsigned 8-bit carrier -- still an 8-bit-weight x 16-bit-act op (w8a16 class)."""
    q = q16.astype(np.int64)
    return q >> 8, q & 0xFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0xB17E)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    act = rng.integers(0, 65536, size=(a.M, a.K), dtype=np.uint16).astype(np.int64)
    q16 = rng.integers(-32768, 32768, size=(a.K, a.N)).astype(np.int64)
    hi, lo = decompose_int16_to_bytes(q16)

    split_ok = bool(np.array_equal(q16, hi * 256 + lo))
    carrier_ok = bool(hi.min() >= -128 and hi.max() <= 127 and lo.min() >= 0 and lo.max() <= 255)
    acc_hi, acc_lo = (act - ACT_ZP) @ hi, (act - ACT_ZP) @ lo
    matmul_ok = bool(np.array_equal((act - ACT_ZP) @ q16, acc_hi * 256 + acc_lo))
    overflow = bool(int(np.abs(acc_hi).max()) * 256 > 2 ** 31)

    print(f"=== verify w16a16 == 2x w8a16  ({a.M}x{a.K}x{a.N}) ===")
    print("  q16 == hi*256+lo, hi int8 / lo uint8 :", split_ok and carrier_ok)
    print("  (act-zp)@q16 == @hi*256 + @lo         :", matmul_ok)
    print("  acc_hi*256 overflows int32            :", overflow,
          "(=> two int8 passes drain separately)")
    ok = split_ok and carrier_ok and matmul_ok and overflow
    print("=== %s ===" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
