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


# HMX-active reference (口径③) per 64^3 matmul — from native parity (INVARIANT 5/9, NOT measured here).
# We have NO HMX-unit hardware counter in this bare-metal trace: the slice dur is the consumer
# KERNEL-CALL WALL (口径④ = matmul + mxmem feed/stall), which is what the thread is occupied by, NOT
# what the HMX unit computes. Setting HMX_BUSY_REF lets us SHOW the gap honestly, never claim it as measured.
HMX_ACTIVE_REF_PER_MM = 1430     # native 128-batch htp_resources.cycles_used / matmul (口径③, reference)
MATMUL_LATENCY_FLOOR = 256       # 口径② dominant-path floor


def summarize(wall, evs, nmm_per_head, heads):
    """Print honestly-separated 口径. CRITICAL: the slice dur = consumer kernel-call WALL (口径④),
    NOT HMX-unit compute. We do NOT have an HMX-active counter here; never call 口径④ 'HMX cycles_used'."""
    occ = collections.Counter()           # thread-occupancy per tid (= Σ slice dur, 口径④ for HMX thread)
    by_op = collections.defaultdict(lambda: [0, 0])   # name -> [Σdur, count]
    span_lo, span_hi = 1 << 62, 0
    for (t, s, t0, t1) in evs:
        if s not in STAGE:
            continue
        _stg, name, typ = STAGE[s]
        tid = HMX_TID if typ == "HMX" else HVX_TID0 + t
        if _stg != "SPIN":                # SPIN = idle-wait, not occupancy
            occ[tid] += t1 - t0
        by_op[name][0] += t1 - t0; by_op[name][1] += 1
        span_lo = min(span_lo, t0); span_hi = max(span_hi, t1)
    timeline = span_hi - span_lo if span_hi else wall
    hmx_occ = occ.get(HMX_TID, 0)         # 口径④ consumer-thread slice-sum (matmul + mxmem feed/stall)
    hvx_occ = sum(v for k, v in occ.items() if k >= HVX_TID0)
    spin = sum((t1 - t0) for (t, s, t0, t1) in evs if s == 11)
    n_mm = next((c for (nm, (d, c)) in by_op.items() if nm.endswith("matmul.stride1")), 0)
    hmx_active_est = n_mm * HMX_ACTIVE_REF_PER_MM     # 口径③ ESTIMATE (native ref), NOT measured
    print("=== GDN baremetal solve — QNN-optrace-ALIGNED report (口径 separated; skill htp-cycle-metric) ===")
    print(f"  口径① graph wall (timeline_cycles, THE verdict)            = {timeline}")
    print(f"  口径④ consumer-thread occupancy (HMX tid256 slice Σ)       = {hmx_occ}  = {hmx_occ*100.0/timeline:.0f}% of wall")
    print(f"        ^ = matmul + mxmem-load STALL. THREAD-busy, NOT HMX-compute. (= QNN op 'Duration', not htp_resources.cycles_used)")
    print(f"  口径③ HMX-UNIT active (true compute, = QNN htp_resources.cycles_used) — our baremetal has NO HMX counter,")
    print(f"        but the kernel is byte-identical to native, so QNN optrace of it MEASURES our HMX-active = {HMX_ACTIVE_REF_PER_MM}/mm")
    print(f"        x {n_mm} = ~{hmx_active_est} = ~{hmx_active_est*100.0/timeline:.0f}% of wall")
    print(f"        => HMX truly computes only ~{hmx_active_est*100.0/max(hmx_occ,1):.0f}% of the consumer wall; the other ~{100-hmx_active_est*100.0/max(hmx_occ,1):.0f}% = mxmem feed/stall (HMX idle/stalled)")
    print(f"  口径② matmul latency floor                                  = ~{MATMUL_LATENCY_FLOOR}/mm (dominant-path)")
    print(f"  HVX producers tid512+: pack occupancy {hvx_occ} ; SPIN-wait {spin} (idle, waiting on consumer)")
    if n_mm:
        print(f"  cross-impl (口径① ONLY): graph-wall / N_mm = {timeline}/{n_mm} = {timeline//n_mm} cyc/mm  "
              f"<-> QNN native: single 64^3=11034, 128-batch=2020/mm")
    print("  per HTP-op-type (slice Σ = thread occupancy 口径④, NOT HMX-active):")
    for nm, (d, c) in sorted(by_op.items(), key=lambda kv: -kv[1][0]):
        print(f"     {nm:28s} occ Σ(口径④)={d:>10}  instances={c:>4}  per-call wall={d//max(c,1):>7}")
    print("  !! NEVER read 口径④ (slice dur / thread occupancy) as HMX compute. True HMX-active needs the")
    print("     HMX PMU counter (TODO) or a QNN optrace htp_resources.cycles_used. Cross-impl = graph-wall/N only.")


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
