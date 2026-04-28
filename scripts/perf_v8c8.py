#!/usr/bin/env python3
"""perf_v8c8.py — unified perf reader for V8C8 BbbKMajor + native ConvLayer comparison.

Outputs a side-by-side table of chrometrace dur/pkts/cpp + (optionally) PMU
counter deltas captured by V9_PMU_PROBE in HmxMatMulV9SkelOp.cpp.

Confirmed (2026-04-28 PM): chrometrace's "Cycles per Packet" / "Duration"
pkts metric ≈ Hexagon PMU COMMITTED_PKT_ANY counter. Both count real
executed packets across all HW threads. So:
  pkts_chrometrace ≈ packets_actually_executed_by_HMX_pipeline

Usage:
  perf_v8c8.py <out_dir>                    # decode V73DEEP run (chrometrace)
  perf_v8c8.py <out_dir> --pmu              # also decode V9_PMU_PROBE markers
  perf_v8c8.py <out_dir> --compare <native_dir>  # side-by-side vs native

<out_dir> must contain:
  ctx/v8c8_schematic.bin
  device_out/qnn-profiling-data*.log  (or pulled-back log)
  device_out/out.raw                  (optional, for PMU marker decode)
  v8c8.onnx.out_ref_u8.npy            (optional, for bit-exact check)

<native_dir> (optional) must contain a chrometrace JSON or runtime profile log
+ schematic. The script auto-detects.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def find_qnn_profiler():
    qnn = os.environ.get("QNN_SDK_ROOT")
    if not qnn:
        sys.exit("ERROR: QNN_SDK_ROOT not set; source scripts/env.sh first")
    return (
        f"{qnn}/bin/x86_64-linux-clang/qnn-profile-viewer",
        f"{qnn}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so",
        f"{qnn}/lib/x86_64-linux-clang",
    )


def decode_chrometrace(out_dir: Path, prof_log: Path | None = None, schematic: Path | None = None):
    """Run qnn-profile-viewer to produce optrace.txt; return list of op events."""
    if prof_log is None:
        cands = list((out_dir / "device_out").glob("qnn-profiling-data*.log")) + \
                list((out_dir / "ctx").glob("qnn-profiling-data*.log"))
        # filter broken symlinks and small files
        valid = []
        for c in cands:
            try:
                if c.exists() and c.stat().st_size > 1500:
                    valid.append(c)
            except (OSError, FileNotFoundError):
                continue
        if not valid:
            return []
        prof_log = valid[0]
    if schematic is None:
        for p in [out_dir / "ctx", out_dir]:
            sch_candidates = list(p.glob("*schematic.bin"))
            if sch_candidates:
                schematic = sch_candidates[0]
                break
        if schematic is None:
            return []
    pv, reader, ld = find_qnn_profiler()
    optrace = Path("/tmp") / f"_optrace_{out_dir.name}.txt"
    if optrace.exists():
        optrace.unlink()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")
    pv_res = subprocess.run(
        [pv, "--reader", reader, "--input_log", str(prof_log),
         "--schematic", str(schematic), "--output", str(optrace)],
        env=env, check=False, capture_output=True,
    )
    if not optrace.exists():
        # try chrometrace.json fallback (for native runs that have it pre-decoded)
        cj = out_dir / "chrometrace.json"
        if cj.exists():
            d = json.loads(cj.read_text())
            ev = []
            seen = set()
            for e in d.get("traceEvents", []):
                n = e.get("name", "")
                if not n: continue
                qn = e.get("args", {}).get("QNN Op Name", "")
                if (n, qn) in seen: continue
                seen.add((n, qn))
                cpp = e.get("args", {}).get("Cycles per Packet", 1) or 1
                ev.append({
                    "name": n, "qnn_name": qn,
                    "dur": e.get("dur", 0), "cpp": cpp,
                    "pkts": int(round(e.get("dur", 0)/cpp)) if cpp else 0,
                })
            return ev
        sys.stderr.write(f"qnn-profile-viewer failed: {pv_res.stderr.decode()[:200]}\n")
        return []
    txt = optrace.read_text()
    pat = re.compile(
        r'"name":\s*"([^"]+)".*?"dur":\s*(\d+).*?"Cycles per Packet":\s*([\d.]+).*?"QNN Op Name":\s*"([^"]+)"'
    )
    events = []
    seen = set()
    for m in pat.finditer(txt):
        name, dur, cpp, qnn_name = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4)
        key = (name, qnn_name)
        if key in seen:
            continue
        seen.add(key)
        pkts = dur / cpp if cpp else 0
        events.append({
            "name": name, "qnn_name": qnn_name,
            "dur": dur, "cpp": cpp, "pkts": int(round(pkts)),
        })
    return events


def decode_pmu_marker(out_raw: Path):
    """Decode V9_PMU_PROBE markers from out.raw[0..31]. Returns dict or None."""
    if not out_raw.exists():
        return None
    b = np.fromfile(out_raw, dtype=np.float32)
    if b.size < 16:
        return None
    u8 = np.round(b[:32]).astype(np.uint8)

    def le(o):
        return int.from_bytes(u8[o:o+4].tobytes(), "little")

    return {
        "kernel_pkt_any": le(0),
        "kernel_pkt_t0":  le(4),
        "kernel_insts":   le(8),
        "kernel_disp":    le(12),
        "kernel_cyc":     le(16),
        "op_pkt_any":     le(20),
        "op_pkt_t0":      le(24),
        "op_cyc":         le(28),
    }


def bit_exact_check(out_dir: Path, ref_npy: Path | None = None, S: int = 256):
    out_raw = out_dir / "device_out" / "out.raw"
    if ref_npy is None:
        cands = list(out_dir.glob("*.out_ref_u8.npy"))
        if not cands:
            return None
        ref_npy = cands[0]
    if not out_raw.exists() or not ref_npy.exists():
        return None
    out = np.fromfile(out_raw, dtype=np.float32).reshape(S, S)
    ref = np.load(ref_npy)
    out_u8 = np.round(out).astype(np.uint8)
    return (out_u8 == ref).sum() / out_u8.size * 100


def summarise(events, target_op="BbbKMajor", drop_chain0=True):
    """Filter to events of interest and compute hot stats (excluding chain0)."""
    rel = [e for e in events if target_op in e["name"]]
    if not rel:
        return None
    cold = next((e for e in rel if e["qnn_name"].endswith("_chain0") or "_0" in e["qnn_name"]), None)
    hot = [e for e in rel if e is not cold] if drop_chain0 else rel
    if not hot:
        return None
    return {
        "n_events": len(rel),
        "cold": cold,
        "hot_count": len(hot),
        "hot_avg_dur": sum(e["dur"] for e in hot) / len(hot),
        "hot_avg_pkts": sum(e["pkts"] for e in hot) / len(hot),
        "hot_avg_cpp": sum(e["dur"] for e in hot) / max(1, sum(e["pkts"] for e in hot)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="V8C8 run directory (must contain ctx/ + device_out/)")
    p.add_argument("--pmu", action="store_true", help="Also decode V9_PMU_PROBE markers")
    p.add_argument("--compare", help="Native run directory for side-by-side")
    p.add_argument("--target", default="BbbKMajor", help="Op name substring to filter (default BbbKMajor)")
    p.add_argument("--target-native", default="ConvLayer_s1.opt", help="Native op name substring")
    p.add_argument("--shape", type=int, default=256)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    print(f"=== {out_dir.name} ===")
    events = decode_chrometrace(out_dir)
    if not events:
        print("  no chrometrace events decoded")
    else:
        s = summarise(events, target_op=args.target)
        if s:
            print(f"  chrometrace ({args.target}): n={s['n_events']}  hot avg "
                  f"dur={s['hot_avg_dur']:.0f}  pkts={s['hot_avg_pkts']:.0f}  cpp={s['hot_avg_cpp']:.3f}")
            if s['cold']:
                print(f"    cold  ({s['cold']['qnn_name']}): "
                      f"dur={s['cold']['dur']}  pkts={s['cold']['pkts']}  cpp={s['cold']['cpp']:.3f}")
        else:
            print(f"  no events matched target='{args.target}'")

    bx = bit_exact_check(out_dir, S=args.shape)
    if bx is not None:
        print(f"  bit-exact: {bx:.2f}%")

    if args.pmu:
        pmu = decode_pmu_marker(out_dir / "device_out" / "out.raw")
        if pmu:
            print(f"  PMU (last hot instance, kernel-only):")
            print(f"    pkt_ANY={pmu['kernel_pkt_any']:>6}  pkt_T0={pmu['kernel_pkt_t0']:>6}  cyc={pmu['kernel_cyc']:>6}")
            print(f"  PMU (whole op, including pre-kernel wrapping):")
            print(f"    pkt_ANY={pmu['op_pkt_any']:>6}  pkt_T0={pmu['op_pkt_t0']:>6}  cyc={pmu['op_cyc']:>6}")
        else:
            print("  PMU: no marker decoded (need V9_PMU_PROBE build + out.raw)")

    if args.compare:
        nat_dir = Path(args.compare)
        print(f"\n=== compare → native: {nat_dir.name} ===")
        nat_events = decode_chrometrace(nat_dir)
        if not nat_events:
            # try chrometrace.json directly
            cj = nat_dir / "chrometrace.json"
            if cj.exists():
                d = json.loads(cj.read_text())
                seen = set()
                for e in d.get("traceEvents", []):
                    if "ConvLayer" not in e.get("name", ""): continue
                    qn = e.get("args", {}).get("QNN Op Name", "")
                    if (e["name"], qn) in seen: continue
                    seen.add((e["name"], qn))
                    cpp = e.get("args", {}).get("Cycles per Packet", 1)
                    nat_events.append({
                        "name": e["name"], "qnn_name": qn,
                        "dur": e.get("dur", 0), "cpp": cpp,
                        "pkts": int(round(e.get("dur", 0)/cpp)) if cpp else 0,
                    })
        ns = summarise(nat_events, target_op=args.target_native)
        if ns:
            print(f"  chrometrace ({args.target_native}): n={ns['n_events']}  hot avg "
                  f"dur={ns['hot_avg_dur']:.0f}  pkts={ns['hot_avg_pkts']:.0f}  cpp={ns['hot_avg_cpp']:.3f}")
            if 'hot_avg_dur' in (s := summarise(events, target_op=args.target) or {}):
                cyc_ratio = s['hot_avg_dur'] / ns['hot_avg_dur']
                pkt_ratio = s['hot_avg_pkts'] / ns['hot_avg_pkts']
                print(f"\n  GAP: ours/native  cyc={cyc_ratio:.2f}×  pkts={pkt_ratio:.2f}×  "
                      f"cpp_ratio={(s['hot_avg_cpp']/ns['hot_avg_cpp']):.2f}×")


if __name__ == "__main__":
    main()
