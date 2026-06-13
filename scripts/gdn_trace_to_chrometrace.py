#!/usr/bin/env python3
"""Convert a GDN baremetal GP_TRACE dump (T.raw) into a QNN-optrace-IDENTICAL Perfetto
chrometrace.json, and print the summary in QNN's EXACT metric口径 / field names.

WHY: so a bare-metal GDN perf number is read with the SAME tool and the SAME口径 as a QNN
optrace — no ad-hoc format, no cross-口径 misreading. The emitted chrometrace.json loads in
the same `trace_processor` / Perfetto UI as QNN's, with the SAME tracks (HMX=tid256,
HVX=tid512+) and the SAME args ("Duration (cycles)", "Dominant Path Cycles", "Start Cycle"),
and `scripts/decode_qnn_optrace.py` / `perfetto_qnn_optrace.py` ingest it unchanged.

口径 mapping (identical to QNN QHAS htp_overall_summary, skill htp-cycle-metric):
  - graph wall  = timeline span max(end)-min(start)            (口径① = THE verdict)
  - HMX cycles_used / utilization (tid 256, consumer matmul)   (口径③ busy / occupancy)
  - HVX cycles_used (tid 512+, producer pack; SPIN = idle-wait, NOT busy)
  - per-op "Dominant Path Cycles" = latency (口径②), NEVER compared to a bare-metal wall.

Source T.raw format (GP_TRACE, gdn_pure_solve.cpp): [magic u32=0x47545203][n u32][wall u64]
[base u64=0] then n*{tid u32, stage u32, t0 u64, t1 u64} (t0/t1 RELATIVE to solve start).
stages: 3=MM(consumer HMX) 5=PREP(producer HVX pack) 11=SPIN(producer wait).

Usage: gdn_trace_to_chrometrace.py T.raw [out_chrometrace.json] [--nmm-per-head 24]
"""
import sys, json, struct, argparse, collections

# match QNN's tid convention so the Perfetto tracks line up 1:1 with a QNN optrace.
HMX_TID = 256
HVX_TID0 = 512
STAGE = {3: ("MM", "gdn::w16a16_matmul.stride1", "HMX"),
         5: ("PREP", "gdn::producer_pack", "HVX"),
         11: ("SPIN", "gdn::producer_spin_wait", "HVX")}


def load_trace(path):
    b = open(path, "rb").read()
    magic, n = struct.unpack_from("<II", b, 0)
    if magic != 0x47545203:
        sys.exit(f"bad magic {magic:#x} (want 0x47545203 — is this a GP_TRACE T.raw?)")
    wall, base = struct.unpack_from("<QQ", b, 8)
    evs, off = [], 24
    for _ in range(n):
        tid, stage, t0, t1 = struct.unpack_from("<IIQQ", b, off); off += 24
        evs.append((tid, stage, t0, t1))
    return wall, base, evs


def to_chrometrace(wall, base, evs):
    """Emit QNN-format Chrome Trace Events (ph X on pid 0 'Core 0 Overview', + M metadata)."""
    out = {"header": {"header_version": {"major": 1, "minor": 0, "patch": 0},
                      "version": {"major": 1, "minor": 1, "patch": 0},
                      "artifact_type": "OP_TRACE", "producer": "gdn_baremetal"},
           "traceEvents": []}
    ev = out["traceEvents"]
    used_tids = sorted({(HMX_TID if STAGE[s][2] == "HMX" else HVX_TID0 + t) for (t, s, _, _) in evs if s in STAGE})
    # metadata: process 'Core 0 Overview' (pid 0) + thread names matching QNN ("Type: HMX"/"Type: HVX").
    ev.append({"cat": "__metadata", "pid": 0, "ph": "M", "name": "process_name", "args": {"name": "Core 0 Overview"}})
    ev.append({"cat": "__metadata", "pid": 0, "ph": "M", "name": "process_sort_index", "args": {"sort_index": 1}})
    for tid in used_tids:
        typ = "HMX" if tid == HMX_TID else "HVX"
        ev.append({"cat": "__metadata", "pid": 0, "tid": tid, "ph": "M", "name": "thread_name",
                   "args": {"name": f"Core:0 Type: {typ} Tid: {tid}"}})
    for (t, s, t0, t1) in evs:
        if s not in STAGE:
            continue
        _stg, name, typ = STAGE[s]
        tid = HMX_TID if typ == "HMX" else HVX_TID0 + t
        dur = int(t1 - t0)
        ev.append({"name": name, "ph": "X", "tid": tid, "pid": 0, "ts": int(t0), "dur": dur,
                   "args": {"HTP Op Type": name, "QNN Op Type": "MatMul" if typ == "HMX" else "Producer",
                            "Start Cycle": int(base + t0), "Duration (cycles)": dur,
                            "Dominant Path Cycles": dur, "Unit": typ}})
    return out


def summarize(wall, evs, nmm_per_head, heads):
    """Print the summary using QNN's EXACT field names / 口径 (htp_overall_summary + htp_op_types)."""
    busy = collections.Counter()          # cycles_used per unit-tid
    by_op = collections.defaultdict(lambda: [0, 0])   # name -> [Σdur, count]
    span_lo, span_hi = 1 << 62, 0
    for (t, s, t0, t1) in evs:
        if s not in STAGE:
            continue
        _stg, name, typ = STAGE[s]
        tid = HMX_TID if typ == "HMX" else HVX_TID0 + t
        if _stg != "SPIN":                # SPIN = idle-wait, excluded from busy (QNN: not active cycles)
            busy[tid] += t1 - t0
        by_op[name][0] += t1 - t0; by_op[name][1] += 1
        span_lo = min(span_lo, t0); span_hi = max(span_hi, t1)
    timeline = span_hi - span_lo if span_hi else wall
    hmx_busy = busy.get(HMX_TID, 0)
    hvx_busy = sum(v for k, v in busy.items() if k >= HVX_TID0)
    spin = sum((t1 - t0) for (t, s, t0, t1) in evs if s == 11)
    n_mm = next((c for (nm, (d, c)) in by_op.items() if nm.endswith("matmul.stride1")), 0)
    print("=== GDN baremetal solve — QNN-optrace-ALIGNED report (口径 == QHAS htp_overall_summary) ===")
    print(f"  graph wall (timeline_cycles, 口径① THE verdict)      = {timeline}")
    print(f"  HMX  tid256  cycles_used (口径③ busy) = {hmx_busy}  utilization = {hmx_busy*100.0/timeline:.1f}%  <- consumer matmul")
    print(f"  HVX  tid512+ cycles_used (口径③ busy) = {hvx_busy}  utilization = {hvx_busy*100.0/timeline:.1f}%  (producer pack; SPIN-wait {spin} excluded)")
    if n_mm:
        print(f"  cross-impl (口径① ONLY): graph-wall / N_mm = {timeline}/{n_mm} = {timeline//n_mm} cyc/mm")
        print(f"     QNN native anchors (same 口径①): single 64^3 = 11034 wall ; 128-batch = 2020/mm  <- targets")
    print("  per HTP-op-type (QNN htp_op_types: cycles=active/busy, dp=Dominant Path=口径② latency):")
    for nm, (d, c) in sorted(by_op.items(), key=lambda kv: -kv[1][0]):
        print(f"     {nm:28s} cycles(busy Σ)={d:>10}  instances={c:>4}  ~per-inst={d//max(c,1):>7}")
    print("  !! 口径②(Dominant Path / per-op latency) NEVER compared to a bare-metal wall (口径④). "
          "Cross-impl = graph-wall/N only. (skill htp-cycle-metric)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="GP_TRACE T.raw")
    ap.add_argument("out", nargs="?", default="chrometrace.json", help="output Perfetto chrometrace.json")
    ap.add_argument("--nmm-per-head", type=int, default=24, help="matmuls/head (Newton=0 -> 24)")
    ap.add_argument("--heads", type=int, default=32)
    a = ap.parse_args()
    wall, base, evs = load_trace(a.trace)
    ct = to_chrometrace(wall, base, evs)
    json.dump(ct, open(a.out, "w"))
    print(f"wrote {a.out}  ({len(ct['traceEvents'])} Chrome trace events; load in Perfetto / trace_processor "
          f"exactly like a QNN optrace, or decode_qnn_optrace.py)\n")
    summarize(wall, evs, a.nmm_per_head, a.heads)


if __name__ == "__main__":
    main()
