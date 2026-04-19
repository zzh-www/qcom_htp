#!/usr/bin/env python3
"""Parse the raw optrace profile.log (all iterations) and extract per-op
cycle counts across every inference. Unlike chrometrace.json (which only
captures the FIRST inference), this gives warmup vs steady-state numbers.

Usage:
    parse_profile_log.py <profile_dir>
where profile_dir contains subdirectories (fp16/, w8a8/, ...) each with
a profile.log.

Reports per-config:
    iter1 (warmup) | median of iters[3..N] (steady) | min | max | iters
"""
import os, re, sys, subprocess, statistics as stats

ROOT = os.path.dirname(os.path.abspath(__file__))
# Default to repo's QNN SDK if QNN_SDK_ROOT not set.
_default_qnn = os.path.abspath(os.path.join(ROOT, "..", "..", "tools", "qnn-sdk"))
QNN = os.environ.get("QNN_SDK_ROOT", _default_qnn)
READER = os.path.join(QNN, "lib/x86_64-linux-clang/libQnnHtpProfilingReader.so")


def extract_iters(profile_log: str) -> list[int]:
    try:
        out = subprocess.check_output(
            [os.path.join(QNN, "bin/x86_64-linux-clang/qnn-profile-viewer"),
             "--input_log", profile_log, "--reader", READER],
            stderr=subprocess.STDOUT, text=True,
        )
    except subprocess.CalledProcessError as e:
        return []
    # Lines of the form "    matmul_1:OpId_17 (cycles) : 53080  cycles"
    return [int(m.group(1)) for m in re.finditer(
        r"matmul_1:[A-Za-z_0-9]+ \(cycles\)\s*:\s*(\d+)", out)]


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr); sys.exit(2)
    root = sys.argv[1]

    preferred = ["fp16", "w16a16", "w8a16", "w8a8", "w4a16", "w4a8", "w4a4"]
    configs = []
    for c in preferred:
        p = os.path.join(root, c, "profile.log")
        if os.path.isfile(p):
            configs.append((c, p))

    print()
    print(f"{'config':<8} {'warmup':>10} {'steady':>10} {'min':>10} {'max':>10} {'n':>3}  cold/steady ratio")
    print("-" * 80)
    rows = []
    for name, log in configs:
        iters = extract_iters(log)
        if not iters:
            print(f"{name:<8} (no iterations found)"); continue
        warmup = iters[0]
        # Skip first 2 iters as warmup; median of the rest.
        tail = iters[2:] if len(iters) > 2 else iters
        steady = stats.median(tail) if tail else warmup
        ratio = warmup / steady if steady else 0
        print(f"{name:<8} {warmup:>10} {int(steady):>10} {min(iters):>10} {max(iters):>10} {len(iters):>3}  {ratio:.2f}x")
        rows.append((name, warmup, int(steady), min(iters), max(iters), len(iters)))
    print()

    # JSON summary
    import json
    with open(os.path.join(root, "profile_log_summary.json"), "w") as f:
        json.dump({"rows": [dict(zip(["config","warmup","steady","min","max","n"], r))
                             for r in rows]}, f, indent=2)
    print(f"wrote {os.path.join(root, 'profile_log_summary.json')}")


if __name__ == "__main__":
    main()
