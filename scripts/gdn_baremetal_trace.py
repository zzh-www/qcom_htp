#!/usr/bin/env python3
"""Assemble a QNN-optrace-format Chrome/Perfetto trace from the bare-metal GDN solve's event blob.

The bare-metal solve (example/gdn_native/baremetal, built with -DGDN_BR_TRACE) records per-worker,
per-stage {tid, stage, t0, t1} spans using the C15:14 PCYCLE counter (a GLOBAL core cycle counter, so
the 4 worker threads share one timebase) and serializes them into the output T buffer.

This emits a `chrometrace.json` that is SCHEMA-IDENTICAL to QNN's optrace output:
  - header {header_version, version, artifact_type:"OP_TRACE"}
  - tid convention: HVX worker slot s -> tid 512+s (QNN HVX threads are 512..515; HMX is 256)
  - ts/dur in relative PCYCLE (== QNN's unit), args carry HTP Op Type / Start Cycle / Duration (cycles) /
    Dominant Path Cycles / QNN Op Type / QNN Op Name, so the same perfetto SQL + qnn-profile-viewer-style
    arg readers work unchanged.
Semantics differ from a real QNN optrace (these are OUR solve stages HEAD/DIAG/MERGE/MM/QUANT, not a QNN
op graph; no q::* glue, DMA, or DDR-bandwidth C-counters), but the FORMAT matches exactly.

Usage:  scripts/gdn_baremetal_trace.py <T.raw> [chrometrace.json]
"""
import sys, json, struct

STAGE = {0: "GdnBR::head", 1: "GdnBR::diag", 2: "GdnBR::merge", 3: "GdnBR::mm_vrmpy", 4: "GdnBR::quant"}

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    path = sys.argv[1]
    out  = sys.argv[2] if len(sys.argv) > 2 else path + ".chrometrace.json"
    blob = open(path, "rb").read()
    magic, n = struct.unpack_from("<II", blob, 0)
    if magic != 0x47545203:
        print(f"bad magic 0x{magic:08x} (rebuild with -DGDN_BR_TRACE, run 4-thread)"); sys.exit(1)
    total_wall, base = struct.unpack_from("<QQ", blob, 8)
    print(f"{n} events, total wall = {total_wall:,} PCYCLE, base = {base}")

    ev = []
    # --- QNN-style __metadata thread_name events (one per worker track) ---
    tids = sorted({struct.unpack_from("<I", blob, 24 + e * 24)[0] for e in range(n)})
    for slot in tids:
        tid = 512 + slot
        ev.append({"cat": "__metadata", "pid": 0, "tid": tid, "ph": "M", "name": "thread_name",
                   "args": {"name": f"Core:0 Type: HVX Tid: {tid}"}})
    # --- X (complete) events: one per stage span, QNN arg schema ---
    nstage = {}
    for e in range(n):
        slot, stage, t0, t1 = struct.unpack_from("<IIQQ", blob, 24 + e * 24)
        nstage[stage] = nstage.get(stage, 0) + 1
        dur = max(1, t1 - t0)
        name = STAGE.get(stage, f"GdnBR::s{stage}")
        ev.append({
            "name": name, "ph": "X", "tid": 512 + slot, "pid": 0,
            "ts": int(t0), "dur": int(dur),
            "args": {
                "HTP Op Type": name,
                "QNN Op Type": "GdnSolveBR", "QNN Op Name": name,
                "Start Cycle": int(base + t0),
                "Duration (cycles)": int(dur),
                "Dominant Path Cycles": int(dur),
            },
        })
    trace = {
        "header": {"header_version": {"major": 1, "minor": 0, "patch": 0},
                   "version": {"major": 1, "minor": 1, "patch": 0},
                   "artifact_type": "OP_TRACE"},
        "traceEvents": ev,
    }
    json.dump(trace, open(out, "w"))
    print("spans per stage:", {STAGE.get(k, k): v for k, v in sorted(nstage.items())})
    print(f"workers (tid): {[512 + s for s in tids]}   (QNN HVX-thread convention)")
    print(f"wrote {out}  -> QNN-optrace schema; load at https://ui.perfetto.dev or feed the perfetto SQL flow")

if __name__ == "__main__":
    main()
