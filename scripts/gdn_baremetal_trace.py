#!/usr/bin/env python3
"""Assemble a Chrome/Perfetto trace from the bare-metal GDN solve's event blob.

The bare-metal solve (example/gdn_native/baremetal, built with -DGDN_BR_TRACE) records per-worker,
per-stage {tid, stage, t0, t1} spans using the C15:14 PCYCLE counter (a GLOBAL core cycle counter, so
the 4 worker threads share one timebase) and serializes them into the output T buffer. This turns that
blob into a traceEvents JSON loadable at https://ui.perfetto.dev (or chrome://tracing).

Time unit: 1 ts/dur unit = 1 PCYCLE (~0.703 ns @ 1.42 GHz TURBO). Spans nest per worker track:
  HEAD > DIAG / MERGE(i,j) > MM / QUANT.

Usage:
  scripts/gdn_baremetal_trace.py <T.raw> [out.json]      # default out: <T.raw>.trace.json
Produce the blob first, e.g.:
  cd example/gdn_native/baremetal
  EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDN_BR_MM_I8 -DGDN_BR_TRACE" bash build.sh
  # deploy + run with 4 threads, fetch T.raw, then run this script on it.
"""
import sys, json, struct

STAGE = {0: "head", 1: "diag", 2: "merge", 3: "mm", 4: "quant"}
CAT   = {0: "head", 1: "diag", 2: "merge", 3: "mm", 4: "quant"}

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    path = sys.argv[1]
    out  = sys.argv[2] if len(sys.argv) > 2 else path + ".trace.json"
    blob = open(path, "rb").read()
    magic, n = struct.unpack_from("<II", blob, 0)
    if magic != 0x47545202:
        print(f"bad magic 0x{magic:08x} (rebuild with -DGDN_BR_TRACE and run 4-thread)"); sys.exit(1)
    (total_wall,) = struct.unpack_from("<Q", blob, 8)
    print(f"{n} events, total wall = {total_wall:,} PCYCLE")

    evs = []
    # process/thread metadata
    evs.append({"name": "process_name", "ph": "M", "pid": 0, "tid": 0,
                "args": {"name": "GDN bare-metal solve (C=256, H=32)"}})
    seen_tids = set()
    nstage = {}
    off = 16
    for e in range(n):
        tid, stage, t0, t1 = struct.unpack_from("<IIQQ", blob, off + e * 24)
        if tid not in seen_tids:
            seen_tids.add(tid)
            evs.append({"name": "thread_name", "ph": "M", "pid": 0, "tid": int(tid),
                        "args": {"name": f"HVX worker {tid}"}})
        nstage[stage] = nstage.get(stage, 0) + 1
        evs.append({"name": STAGE.get(stage, f"s{stage}"), "cat": CAT.get(stage, "other"),
                    "ph": "X", "pid": 0, "tid": int(tid),
                    "ts": int(t0), "dur": int(max(1, t1 - t0)),
                    "args": {"cyc": int(t1 - t0)}})
    trace = {"traceEvents": evs, "displayTimeUnit": "ns",
             "metadata": {"unit": "1 tick = 1 PCYCLE (~0.703 ns @1.42GHz)", "total_wall_pcycle": total_wall}}
    json.dump(trace, open(out, "w"))
    span = {STAGE.get(k, k): v for k, v in sorted(nstage.items())}
    print("spans per stage:", span)
    print(f"workers: {sorted(seen_tids)}")
    print(f"wrote {out}  -> load at https://ui.perfetto.dev")

if __name__ == "__main__":
    main()
