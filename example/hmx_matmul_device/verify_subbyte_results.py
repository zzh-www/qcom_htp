#!/usr/bin/env python3
"""
verify_subbyte_results.py — compare on-silicon probe output against the
predictions made from the hexagon-sim ISS probes.

Usage:
    ./verify_subbyte_results.py build/probe_subbyte_result.txt

The script parses LOG lines produced by probe_subbyte_device.c and checks
them against the expected values table. Exit code 0 = all checks pass
(silicon matches sim); non-zero = divergences found.
"""
import re
import sys
from pathlib import Path

# Expected output [0,0] value given weight type + byte_val, when A=1 everywhere.
# Derived from probe_subbyte_full on hexagon-sim ISS (all 32-cell outputs are
# identical for the all-ones activation fill).
#
# Formula summary:
#   weight.b    : 32 * byte_as_i8
#   weight.n    : 16 * (hi_nibble_i4 + lo_nibble_i4)
#   weight.c    :  8 * sum_of_4_crumbs_i2
#   weight.ubit :  4 * popcount(byte)

# Sub-byte helper: split a byte into N fields of `width` bits, treating
# the MSB as sign if `signed` is True.
def _sub(byte, width, n, signed):
    vals = []
    for i in range(n):
        raw = (byte >> (i * width)) & ((1 << width) - 1)
        if signed and (raw >> (width - 1)) & 1:
            raw -= (1 << width)
        vals.append(raw)
    return vals

def expected_out(wt_type, byte_val):
    b = byte_val & 0xFF
    if wt_type == "b":
        val = b if b < 128 else b - 256
        return (32 * val) & 0xFFFF
    if wt_type == "n":
        lo, hi = _sub(b, 4, 2, signed=True)
        return (16 * (lo + hi)) & 0xFFFF
    if wt_type == "c":
        # Empirical: weight.c is SIGNED int2 per probe evidence.
        crumbs = _sub(b, 2, 4, signed=True)
        return (8 * sum(crumbs)) & 0xFFFF
    if wt_type == "ubit":
        return 4 * bin(b).count("1")
    return None  # unknown / n:2x not predicted yet

def parse(path):
    """Extract (wt_type, byte_val, out00) tuples from the log file."""
    pat = re.compile(r"(b|n|c|ubit|n:2x)\s+byte=0x([0-9a-fA-F]+)\s+tile=\d+B.*?out\[0,0\]=(\d+)")
    rows = []
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            rows.append((m.group(1), int(m.group(2), 16), int(m.group(3))))
    return rows

def parse_cycles(path):
    """Extract (wt_type, pcyc) for Part 2 throughput table."""
    pat = re.compile(r"weight\.(\S+)\s*:\s*(\d+)\s*pcyc")
    out = {}
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out

def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = argv[1]
    rows = parse(path)
    cycles = parse_cycles(path)
    if not rows:
        print(f"[FAIL] no matching log lines found in {path}")
        return 3

    pass_ct, fail_ct = 0, 0
    print(f"{'type':<8}{'byte':<8}{'observed':>10}  {'expected':>10}  result")
    print("-" * 56)
    for wt, b, obs in rows:
        exp = expected_out(wt, b)
        if exp is None:
            print(f"{wt:<8}0x{b:02x}    {obs:>10}  {'':>10}  (no prediction)")
            continue
        ok = (obs == exp)
        tag = "PASS" if ok else "FAIL"
        print(f"{wt:<8}0x{b:02x}    {obs:>10}  {exp:>10}  {tag}")
        if ok: pass_ct += 1
        else:  fail_ct += 1

    print("")
    if cycles:
        print("Throughput (pcycles for ITERS packets):")
        base = cycles.get("b")
        for wt, cyc in cycles.items():
            ratio = (base / cyc) if base and cyc else 0.0
            print(f"  weight.{wt:<8} {cyc:>10}  ratio_vs_b={ratio:.3f}")
        if base:
            # Key verification: all sub-byte cycles should be ≈ baseline (±10%)
            devs = []
            for wt, cyc in cycles.items():
                if wt == "b": continue
                pct = abs(cyc - base) / base * 100
                status = "ok" if pct <= 10.0 else "DIVERGE"
                devs.append(f"    {wt}: {pct:.1f}% vs int8 [{status}]")
            print("  Cycle uniformity check (expected within ±10%):")
            for d in devs:
                print(d)

    print("")
    print(f"Summary: {pass_ct} PASS / {fail_ct} FAIL")
    return 0 if fail_ct == 0 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
