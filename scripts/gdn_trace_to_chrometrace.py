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
# op names mirror QNN's per-op granularity (q::ConvLayer / ForceFormat / weights_to_vtcm / OutputSlice)
# so each op is a SEPARATE line, and THE matmul op is distinct from the feed ops — exactly like QNN.
MATMUL_OP = "q::HmxW16A16MatMul_s1.opt"   # <-> QNN q::ConvLayer_s1.opt / HmxU8I8ToU8MatMul (the matmul op)
STAGE = {3:  ("MATMUL",     MATMUL_OP,                          "HMX"),  # the matmul op (HMX)
         4:  ("ACT_FORMAT", "q::ForceFormat_Crouton(act)",      "HVX"),  # act crouton format
         5:  ("WT_PACK",    "q::ConvLayer.opt.weights_to_vtcm", "HVX"),  # weight+bias -> VTCM
         10: ("OUT_COPY",   "q::*OutputSlice",                  "HVX"),  # output read-back
         11: ("SPIN",       "spin_idle",                        "HVX")}  # idle-wait: NOT an op (skipped)


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
        if s not in STAGE or s == 11:     # SPIN = idle-wait: QNN never emits idle as a slice (it's a gap). skip.
            continue
        _stg, name, typ = STAGE[s]
        tid = HMX_TID if typ == "HMX" else HVX_TID0 + t
        dur = int(t1 - t0)
        ev.append({"name": name, "ph": "X", "tid": tid, "pid": 0, "ts": int(t0), "dur": dur,
                   "args": {"HTP Op Type": name, "QNN Op Type": "MatMul" if typ == "HMX" else "Producer",
                            "Start Cycle": int(base + t0), "Duration (cycles)": dur,
                            "Dominant Path Cycles": dur, "Unit": typ}})
    return out


# DECOMPILE/VERIFY finding (cron#31): QNN's htp_resources.cycles_used = Σ(per-op Duration on that
# thread) EXACTLY (proven: u8i8 optrace HMX cycles_used 11567 == Σ X-event dur on tid256). QNN times
# ops with PcyclePoint (C15:14 PCYCLE); the host sums per thread. There is NO hidden HMX-compute
# hardware counter. So "cycles_used / utilization" is op-duration OCCUPANCY (matmul + the op's own
# feed), NOT pure MAC. The compute-ish number QNN exposes is per-op Dominant Path (= Duration for a
# leaf op). Pure-MAC (~256) is BELOW QNN's op granularity — not in the optrace at all.
QNN_BATCH_CYCLES_USED_PER_MM = 1430   # native [1,128,64,64] HMX cycles_used/mm — SAME 口径 as ours (Σ op-dur)
MATMUL_LATENCY_FLOOR = 256            # 口径② dominant-path floor (below QNN op granularity; hw reference)


def summarize(wall, evs, nmm_per_head, heads):
    """Report EXACTLY as QNN does (verified mechanism): per-unit cycles_used = Σ(op Duration on that
    thread) = occupancy; utilization = cycles_used/timeline; per-op Dominant Path = leaf latency.
    This is QNN's real definition — NOT pure MAC. Our cycles_used and QNN's are the SAME 口径."""
    cu = collections.Counter()            # cycles_used per tid = Σ op Duration (QNN's exact metric)
    by_op = collections.defaultdict(lambda: [0, 0])   # name -> [Σdur, count]
    tspan = {}                            # per-tid (lo,hi) for per-unit timeline_cycles
    span_lo, span_hi = 1 << 62, 0
    for (t, s, t0, t1) in evs:
        if s not in STAGE:
            continue
        _stg, name, typ = STAGE[s]
        tid = HMX_TID if typ == "HMX" else HVX_TID0 + t
        span_lo = min(span_lo, t0); span_hi = max(span_hi, t1)
        if _stg == "SPIN":                # SPIN = thread idle-waiting; not an executed op (excluded, like QNN)
            continue
        cu[tid] += t1 - t0
        lo, hi = tspan.get(tid, (1 << 62, 0)); tspan[tid] = (min(lo, t0), max(hi, t1))
        by_op[name][0] += t1 - t0; by_op[name][1] += 1
    timeline = span_hi - span_lo if span_hi else wall
    md = by_op.get(MATMUL_OP, [0, 0])
    mm_per = md[0] // md[1] if md[1] else 0     # matmul op cycles/instance (Duration/op) = THE number
    print("=== GDN baremetal — per-op trace, reported EXACTLY like QNN optrace (each op separate) ===")
    print("  >>> THE MATMUL OP (its own op, separate from feed — like QNN q::ConvLayer_s1.opt):")
    print(f"        {MATMUL_OP}  =  {mm_per} cyc/instance   (instances={md[1]}, total cycles_used={md[0]})")
    print(f"        <-> QNN q::ConvLayer_s1.opt (native [1,128,64,64]) = {QNN_BATCH_CYCLES_USED_PER_MM} cyc/instance "
          f" => {mm_per/QNN_BATCH_CYCLES_USED_PER_MM:.1f}x  (SAME 口径: per-op Duration on HMX thread; gap = batch amortization, A-warm)")
    print(f"        (pure-MAC ~{MATMUL_LATENCY_FLOOR} is BELOW op granularity — absent from QNN's optrace too; this op = matmul + its mxmem-load)")
    print("  per HTP-op-type (= QNN htp_op_types; EACH op separate; Duration/op is the per-instance cycle):")
    print(f"     {'op (HMX=matmul, HVX=feed/IO)':34s} {'unit':4s} {'instances':>9} {'Duration/op':>12} {'total cyc':>11}")
    for nm, (d, c) in sorted(by_op.items(), key=lambda kv: -kv[1][0]):
        typ = "HMX" if nm == MATMUL_OP else "HVX"
        mark = "  <== MATMUL" if nm == MATMUL_OP else ""
        print(f"     {nm:34s} {typ:4s} {c:>9} {d//max(c,1):>12} {d:>11}{mark}")
    print("  per-unit htp_resources (QNN field defs; cycles_used = Σ that unit's op Durations; SPIN idle excluded, like QNN):")
    for tid in sorted(cu):
        typ = "HMX" if tid == HMX_TID else "HVX"
        lo, hi = tspan[tid]; tl = (hi - lo) or timeline
        print(f"     tid{tid:<4d} {typ}  cycles_used={cu[tid]:>9}  timeline_cycles={tl:>9}  utilization={cu[tid]*100.0/tl:5.1f}%")
    print(f"  graph timeline_cycles (口径① verdict) = {timeline}")


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
