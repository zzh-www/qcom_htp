#!/usr/bin/env python3
"""GDN triangular-inverse precision gate: oc = ||T_dev - fp64 inv(I-A)|| / ||fp64 inv(I-A)||.

Reads the device output T.raw (u16, zpT) and the input A.raw (u16, zpA), dequantizes with the
fixed static scales, builds the fp64 reference inv(I-A) per head, and asserts oc <= gate.

Usage: gdn_solve_oc_check.py <A.raw> <T.raw> [--H 32] [--C 256] [--zpA 32768] [--zpT 32768]
       [--sA 2.770166930875267e-05] [--sT 6.103701895199438e-05] [--gate 1.05e-2]
Exit 0 if oc <= gate, else 1.
"""
import argparse, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("T")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--zpA", type=int, default=32768)
    ap.add_argument("--zpT", type=int, default=32768)
    ap.add_argument("--sA", type=float, default=2.770166930875267e-05)
    ap.add_argument("--sT", type=float, default=6.103701895199438e-05)
    ap.add_argument("--gate", type=float, default=1.05e-2)
    a = ap.parse_args()

    H, C = a.H, a.C
    A = (np.fromfile(a.A, dtype=np.uint16).reshape(H, C, C).astype(np.int64) - a.zpA) * a.sA
    T = (np.fromfile(a.T, dtype=np.uint16).reshape(H, C, C).astype(np.int64) - a.zpT) * a.sT
    Tref = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])
    oc = float(np.linalg.norm(T - Tref) / (np.linalg.norm(Tref) + 1e-12))

    ok = oc <= a.gate
    print(f"GDN solve oc = {oc:.4e}  (gate <= {a.gate:.2e})  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
