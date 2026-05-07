#!/usr/bin/env python3
"""Perf reader for custom HmxU8I8ToU8MatMul and native ConvLayer comparison.

Outputs a side-by-side table of chrometrace dur/pkts/cpp + (optionally) PMU
cycle markers captured by HMX_U8I8_PROBE_CYCLES.

Confirmed (2026-04-28 PM): chrometrace's "Cycles per Packet" / "Duration"
pkts metric ≈ Hexagon PMU COMMITTED_PKT_ANY counter. Both count real
executed packets across all HW threads. So:
  pkts_chrometrace ≈ packets_actually_executed_by_HMX_pipeline

Usage:
  perf_hmx_u8i8_matmul.py <out_dir>
  perf_hmx_u8i8_matmul.py <out_dir> --probe-cycles
  perf_hmx_u8i8_matmul.py <out_dir> --compare <native_dir>

<out_dir> must contain:
  ctx/u8i8_schematic.bin
  device_out/qnn-profiling-data*.log  (or pulled-back log)
  device_out/out.raw                  (optional, for probe marker decode)
  u8i8.onnx.out_ref_u8.npy            (optional, for bit-exact check)

<native_dir> (optional) must contain standard optrace artifacts or a runtime
profile log + schematic. The script auto-detects.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]


def _first_existing(candidates):
    for path in candidates:
        try:
            if path.exists() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def _find_profile_log(out_dir: Path):
    cands = (
        sorted((out_dir / "device_out").glob("qnn-profiling-data*.log"))
        + sorted((out_dir / "ctx").glob("qnn-profiling-data*.log"))
        + sorted(out_dir.glob("profile.log"))
    )
    return _first_existing(cands)


def _find_schematic(out_dir: Path):
    return _first_existing(
        sorted((out_dir / "ctx").glob("*schematic.bin"))
        + sorted(out_dir.glob("*schematic.bin"))
    )


def _ensure_standard_chrometrace(
    out_dir: Path,
    prof_log: Path | None = None,
    schematic: Path | None = None,
):
    chrometrace = _first_existing(
        [
            out_dir / "optrace" / "chrometrace.json",
            out_dir / "device_out" / "chrometrace.json",
            out_dir / "chrometrace.json",
        ]
    )
    if chrometrace is not None:
        return chrometrace

    prof_log = prof_log or _find_profile_log(out_dir)
    schematic = schematic or _find_schematic(out_dir)
    if prof_log is None or schematic is None:
        return None

    decoder = ROOT_DIR / "scripts" / "decode_qnn_optrace.py"
    cmd = [sys.executable, str(decoder), str(out_dir)]
    if prof_log is not None:
        cmd += ["--profile-log", str(prof_log)]
    if schematic is not None:
        cmd += ["--schematic", str(schematic)]
    res = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        return None
    return _first_existing([out_dir / "optrace" / "chrometrace.json"])


def _event_duration(event: dict) -> int:
    args = event.get("args", {})
    return int(args.get("Duration (cycles)", event.get("dur", 0)) or 0)


def _event_cpp(event: dict):
    value = event.get("args", {}).get("Cycles per Packet")
    if value in (None, 0, "0"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _events_from_chrometrace(chrometrace: Path):
    data = json.loads(chrometrace.read_text(encoding="utf-8"))
    raw_events = data if isinstance(data, list) else data.get("traceEvents", [])
    events = []
    seen = set()
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        args = event.get("args", {})
        name = str(event.get("name") or args.get("HTP Op Type", ""))
        if not name:
            continue
        qnn_name = str(args.get("QNN Op Name", ""))
        key = (name, qnn_name)
        if key in seen:
            continue
        seen.add(key)
        dur = _event_duration(event)
        cpp = _event_cpp(event)
        events.append({
            "name": name,
            "qnn_name": qnn_name,
            "dur": dur,
            "cpp": cpp if cpp else 0.0,
            "pkts": int(round(dur / cpp)) if cpp else 0,
        })
    return events


def decode_chrometrace(out_dir: Path, prof_log: Path | None = None, schematic: Path | None = None):
    """Return op events from the standard <out_dir>/optrace artifact set."""
    chrometrace = _ensure_standard_chrometrace(out_dir, prof_log, schematic)
    if chrometrace is None:
        return []
    return _events_from_chrometrace(chrometrace)


def _le_words_from_u8(u8):
    def le(o):
        return int.from_bytes(u8[o:o + 4].tobytes(), "little")
    return {
        "kernel_cyc": le(0),
        "desc_cyc": le(4),
        "table_cyc": le(8),
        "qhpi_setup_cyc": le(12),
    }


def _plausible_probe(words):
    vals = list(words.values())
    return any(vals) and all(0 <= v < 100_000_000 for v in vals)


def decode_probe_cycles(out_raw: Path):
    """Decode HMX_U8I8_PROBE_CYCLES markers from native-u8 or legacy-f32 raw."""
    if not out_raw.exists():
        return None
    raw = out_raw.read_bytes()
    if len(raw) < 16:
        return None
    native_words = _le_words_from_u8(np.frombuffer(raw[:16], dtype=np.uint8))
    if _plausible_probe(native_words):
        return native_words
    if len(raw) < 64 or len(raw) % 4:
        return native_words
    f = np.frombuffer(raw, dtype="<f4", count=min(len(raw) // 4, 32))
    if f.size < 16:
        return native_words
    legacy_u8 = np.clip(np.rint(f[:32]), 0, 255).astype(np.uint8)
    legacy_words = _le_words_from_u8(legacy_u8)
    return legacy_words if _plausible_probe(legacy_words) else native_words


def bit_exact_check(out_dir: Path, ref_npy: Path | None = None, S: int = 256):
    out_raw = out_dir / "device_out" / "out.raw"
    if ref_npy is None:
        cands = list(out_dir.glob("*.out_ref_u8.npy"))
        if not cands:
            return None
        ref_npy = cands[0]
    if not out_raw.exists() or not ref_npy.exists():
        return None
    ref = np.load(ref_npy)
    raw_size = out_raw.stat().st_size
    if raw_size == ref.nbytes:
        out_u8 = np.fromfile(out_raw, dtype=ref.dtype).reshape(ref.shape)
    elif raw_size == ref.size * np.dtype(np.float32).itemsize:
        out = np.fromfile(out_raw, dtype=np.float32).reshape(ref.shape)
        out_u8 = np.round(out).astype(ref.dtype)
    else:
        return None
    return (out_u8 == ref).sum() / out_u8.size * 100


def summarise(events, target_op="HmxU8I8ToU8MatMul", drop_chain0=True):
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


def _qnn_suffix_index(name: str):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def aggregate_qnn_ops(events, qnn_prefix: str, drop_index0=True):
    """Aggregate all HTP events that profiler maps to the same QNN op name.

    This is the apples-to-apples view for native MatMul.  QNN lowers one MatMul
    into multiple HTP events such as bias_to_vtcm, weights_to_vtcm,
    DmaCheckpointSet, and ConvLayer_s1.opt.  Looking only at ConvLayer_s1.opt
    undercounts the native QNN-op cost.
    """
    grouped = {}
    for e in events:
        qnn_name = e["qnn_name"]
        if not qnn_name.startswith(qnn_prefix):
            continue
        g = grouped.setdefault(qnn_name, {"dur": 0, "pkts": 0, "parts": []})
        g["dur"] += e["dur"]
        g["pkts"] += e["pkts"]
        g["parts"].append(e)
    if not grouped:
        return None

    hot = []
    cold = None
    for qnn_name, g in grouped.items():
        idx = _qnn_suffix_index(qnn_name)
        if drop_index0 and idx == 0:
            cold = (qnn_name, g)
        else:
            hot.append((qnn_name, g))
    if not hot:
        return None

    total_dur = sum(g["dur"] for _, g in hot)
    total_pkts = sum(g["pkts"] for _, g in hot)
    return {
        "n_qnn_ops": len(grouped),
        "hot_count": len(hot),
        "hot_avg_dur": total_dur / len(hot),
        "hot_avg_pkts": total_pkts / len(hot),
        "hot_avg_cpp": total_dur / max(1, total_pkts),
        "cold": cold,
        "hot": hot,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="custom_u8i8 run directory (must contain ctx/ + device_out/)")
    p.add_argument("--probe-cycles", action="store_true", help="Decode HMX_U8I8_PROBE_CYCLES markers")
    p.add_argument("--compare", help="Native run directory for side-by-side")
    p.add_argument("--target", default="HmxU8I8ToU8MatMul", help="Op name substring to filter")
    p.add_argument("--target-native", default="ConvLayer_s1.opt", help="Native op name substring")
    p.add_argument("--custom-qnn-prefix", default="hmx_u8i8_chain",
                   help="QNN op-name prefix for custom aggregate view")
    p.add_argument("--native-qnn-prefix", default="MatMul_",
                   help="QNN op-name prefix for native aggregate view")
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
            custom_agg = aggregate_qnn_ops(events, args.custom_qnn_prefix)
            if custom_agg:
                print(f"  qnn-op aggregate ({args.custom_qnn_prefix}*): hot avg "
                      f"dur={custom_agg['hot_avg_dur']:.0f}  pkts={custom_agg['hot_avg_pkts']:.0f}  "
                      f"cpp={custom_agg['hot_avg_cpp']:.3f}")
        else:
            print(f"  no events matched target='{args.target}'")

    bx = bit_exact_check(out_dir, S=args.shape)
    if bx is not None:
        print(f"  bit-exact: {bx:.2f}%")

    if args.probe_cycles:
        cyc = decode_probe_cycles(out_dir / "device_out" / "out.raw")
        if cyc:
            print("  HMX_U8I8_PROBE_CYCLES:")
            print(f"    kernel={cyc['kernel_cyc']:>6}  desc={cyc['desc_cyc']:>6}  "
                  f"table={cyc['table_cyc']:>6}  qhpi_setup={cyc['qhpi_setup_cyc']:>6}")
        else:
            print("  probe cycles: no marker decoded (need HMX_U8I8_PROBE_CYCLES build + out.raw)")

    if args.compare:
        nat_dir = Path(args.compare)
        print(f"\n=== compare → native: {nat_dir.name} ===")
        nat_events = decode_chrometrace(nat_dir)
        if not nat_events:
            # try chrometrace.json directly
            cj = _first_existing([
                nat_dir / "optrace" / "chrometrace.json",
                nat_dir / "device_out" / "chrometrace.json",
                nat_dir / "chrometrace.json",
            ])
            if cj is not None:
                nat_events = [
                    e for e in _events_from_chrometrace(cj)
                    if "ConvLayer" in e["name"]
                ]
        ns = summarise(nat_events, target_op=args.target_native)
        if ns:
            print(f"  chrometrace ({args.target_native}): n={ns['n_events']}  hot avg "
                  f"dur={ns['hot_avg_dur']:.0f}  pkts={ns['hot_avg_pkts']:.0f}  cpp={ns['hot_avg_cpp']:.3f}")
            if 'hot_avg_dur' in (s := summarise(events, target_op=args.target) or {}):
                cyc_ratio = s['hot_avg_dur'] / ns['hot_avg_dur']
                pkt_ratio = s['hot_avg_pkts'] / ns['hot_avg_pkts']
                print(f"\n  GAP vs native kernel-only HTP op: "
                      f"cyc={cyc_ratio:.2f}×  pkts={pkt_ratio:.2f}×  "
                      f"cpp_ratio={(s['hot_avg_cpp']/ns['hot_avg_cpp']):.2f}×")

        native_agg = aggregate_qnn_ops(nat_events, args.native_qnn_prefix)
        custom_agg = aggregate_qnn_ops(events, args.custom_qnn_prefix)
        if native_agg:
            print(f"  qnn-op aggregate ({args.native_qnn_prefix}*): hot avg "
                  f"dur={native_agg['hot_avg_dur']:.0f}  pkts={native_agg['hot_avg_pkts']:.0f}  "
                  f"cpp={native_agg['hot_avg_cpp']:.3f}")
            if custom_agg:
                print(f"\n  GAP vs native QNN-op aggregate: "
                      f"cyc={custom_agg['hot_avg_dur']/native_agg['hot_avg_dur']:.2f}×  "
                      f"pkts={custom_agg['hot_avg_pkts']/native_agg['hot_avg_pkts']:.2f}×  "
                      f"delta_cyc={custom_agg['hot_avg_dur']-native_agg['hot_avg_dur']:.0f}  "
                      f"delta_pkts={custom_agg['hot_avg_pkts']-native_agg['hot_avg_pkts']:.0f}")


if __name__ == "__main__":
    main()
