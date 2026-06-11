#!/usr/bin/env python3
"""Deeper analysis of a QNN HTP optrace via Perfetto trace_processor.

QNN's optrace JSONs are Chrome-Trace-Event / Perfetto-spec compliant, so the official
`trace_processor` binary ingests them directly — giving SQL access to the per-op timeline,
per-unit (HMX/HVX) tracks, and the VTCM/DRAM bandwidth counters that the QHAS summary and
`decode_qnn_optrace.py` flatten away.

CYCLES are the primary unit — `chrometrace.json` `dur` IS cycles (verified: GdnSolve max dur ==
QHAS `cycles` max), and the per-phase HTP frequency VARIES (Power-On/VTCM/Graph-Execute each run at a
different GHz), so µs is derived/unstable; report cycles.

Two files in an `optrace/` dir carry complementary truth (see the qnn-htp-profiling skill):
  * chrometrace_runtrace.json  — REAL WALL phases (Resource Power On, VTCM Acquire, Graph Execute);
                                 real wall cycles = `stat_end_cycles - stat_start_cycles` per event
                                 (the event `dur` field is a tick unit, NOT cycles — don't use it).
  * chrometrace.json           — per-op timeline; `dur` IS cycles but thread-AGGREGATE (summed over
                                 the HVX/HMX threads) = work volume, NOT wall
                                 (see feedback_perf_wall_not_aggregate_cycles).

This prints: (1) wall phase cycles, (2) compute work-volume cycles per op, (3) memory bandwidth.

Setup (once):  curl -LO https://get.perfetto.dev/trace_processor && chmod +x trace_processor
Usage:  scripts/perfetto_qnn_optrace.py <optrace_dir> [--tp /path/to/trace_processor]
Skills: github.com/google/perfetto/tree/main/ai/skills  (installed under ~/.claude/skills/)
"""
import sys, os, subprocess, json, argparse, shutil

def tp_query(tp, trace, sql):
    """Run one SQL via the trace_processor binary, return list[dict] (CSV parse)."""
    out = subprocess.run([tp, "-q", "/dev/stdin", trace], input=sql, capture_output=True,
                         text=True).stdout.strip().splitlines()
    rows = [l for l in out if l and not l.startswith(("Loading", "[", "column "))]
    if not rows:
        return []
    hdr = [h.strip('"') for h in rows[0].split(",")]
    res = []
    for l in rows[1:]:
        vals = [v.strip('"') for v in l.split(",")]
        res.append(dict(zip(hdr, vals)))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("optrace_dir")
    ap.add_argument("--tp", default=shutil.which("trace_processor")
                    or os.path.expanduser("~/.local/share/perfetto/trace_processor"))
    a = ap.parse_args()
    d = a.optrace_dir
    run = os.path.join(d, "chrometrace_runtrace.json")
    chrome = os.path.join(d, "chrometrace.json")
    if not os.path.exists(a.tp):
        sys.exit(f"trace_processor not found at {a.tp} — curl -LO https://get.perfetto.dev/trace_processor")

    # 1) WALL phases from runtrace — REAL cycles via stat_*_cycles (event dur is a tick unit, ignore).
    #    Per-phase frequency varies, so us (QNN's own summary_time_us) is shown only as a secondary hint.
    print(f"== WALL phases (chrometrace_runtrace.json) — real stat cycles ==")
    if os.path.exists(run):
        for r in tp_query(a.tp, run,
            "SELECT name,"
            " CAST(EXTRACT_ARG(arg_set_id,'args.stat_end_cycles') AS INT)"
            " - CAST(EXTRACT_ARG(arg_set_id,'args.stat_start_cycles') AS INT) AS cyc,"
            " EXTRACT_ARG(arg_set_id,'args.summary_time_us') us,"
            " EXTRACT_ARG(arg_set_id,'args.summary_htp_frequency_ghz') ghz "
            "FROM slice WHERE cyc IS NOT NULL ORDER BY ts;"):
            cyc = int(r["cyc"]) if r.get("cyc") else 0
            us = float(r["us"]) if r.get("us") else 0.0
            ghz = float(r["ghz"]) if r.get("ghz") else 0.0
            print(f"  {r['name']:24} {cyc:>12,} cyc   (~{us:.1f} us @ {ghz:.3f} GHz)")
    else:
        print("  (no runtrace.json)")

    # 2) compute work-volume from chrometrace — dur IS cycles, thread-AGGREGATE (work volume, not wall).
    #    QNN stuffs CYCLES into the Chrome µs `dur` field, and trace_processor ×1000's µs->ns on ingest,
    #    so its `dur` column = JSON-cycles * 1000; divide by 1000 to recover cycles (verified vs QHAS).
    print(f"\n== compute work-volume (chrometrace.json depth-0; AGGREGATE cycles over threads, NOT wall) ==")
    rows = tp_query(a.tp, chrome,
        "SELECT name, count(*) n, sum(dur)/1000 s, max(dur)/1000 mx FROM slice WHERE depth=0 GROUP BY name ORDER BY s DESC;")
    tot = sum(float(r["s"]) for r in rows) or 1.0
    for r in rows[:15]:
        s = float(r["s"]); mx = float(r["mx"])
        print(f"  {r['name']:34} n={int(r['n']):>4} sum={s:>12,.0f} cyc ({100*s/tot:4.1f}%)  max_tile={mx:>9,.0f} cyc")

    # 3) memory bandwidth counters
    print(f"\n== memory bandwidth (counter tracks) ==")
    for r in tp_query(a.tp, chrome,
        "SELECT ct.name, max(c.value) mx, avg(c.value) av FROM counter c "
        "JOIN track ct ON ct.id=c.track_id GROUP BY ct.name ORDER BY ct.name;"):
        print(f"  {r['name']:14} max={float(r['mx']):>12,.0f}  avg={float(r['av']):>12,.0f}")

if __name__ == "__main__":
    main()
