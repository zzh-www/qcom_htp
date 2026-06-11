#!/usr/bin/env python3
"""Render a per-thread ASCII timeline + measure true HVX∥HMX overlap from a QNN HTP optrace.

The overlap is the actual time-intersection of HMX-unit ops (tid 256) with HVX-unit ops
(tids 512-515), read from the op TIMESTAMPS in chrometrace.json — NOT a derived
(solve+mm-combined)/max ratio, which a pathological reference can pin at ~1.0 (the "98%"
artifact; see docs/qnn_htp_scheduling_and_custom_op_limits.md).

Usage:
  gdn_overlap_from_trace.py <chrometrace.json | out_dir> [--label NAME] [--width 96] [--producer SUBSTR]

--producer isolates a specific HVX op (e.g. "GdnSolve") and reports ITS overlap with the HMX unit
separately — the question "does the CUSTOM op overlap HMX". Without it, reports the literal
HVX-unit ∩ HMX-unit overlap, which for a native matmul also counts the matmul's OWN layout glue
(ForceFormat/convert_weights) overlapping its own matmul — a different thing.

If given a directory, looks for <dir>/optrace/chrometrace.json then <dir>/chrometrace.json.
Exit code 0 always; prints "OVERLAP <pct>%%" as the last line for easy scraping.
"""
import sys, os, json, argparse

GLUE = ("InputSlice", "OutputSlice", "Dma", "SyncOp", "$", "Checkpoint", "Preload", "Reshape")

def unit(name):
    return "HMX" if "HMX" in name else ("HVX" if "HVX" in name else "")

def load(path):
    if os.path.isdir(path):
        for c in (os.path.join(path, "optrace", "chrometrace.json"),
                  os.path.join(path, "chrometrace.json")):
            if os.path.isfile(c):
                path = c; break
    return json.load(open(path)), path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--label", default="")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--producer", default="", help="HVX op-name substring to isolate vs HMX (e.g. GdnSolve)")
    a = ap.parse_args()
    c, path = load(a.trace)
    tn = {e.get("tid"): e.get("args", {}).get("name", "")
          for e in c["traceEvents"] if e.get("ph") == "M" and e.get("name") == "thread_name"}
    U = lambda tid: unit(tn.get(tid, ""))
    evs = [e for e in c["traceEvents"] if e.get("ph") == "X"]
    seen, u = set(), []
    for e in sorted(evs, key=lambda e: e.get("ts", 0)):
        k = (e["tid"], e["ts"], e["name"])
        if k in seen:
            continue
        seen.add(k); u.append(e)
    if not u:
        print(f"{a.label or path}: no events"); print("OVERLAP 0%"); return
    t0 = min(e["ts"] for e in u); t1 = max(e["ts"] + e["dur"] for e in u); span = max(1, t1 - t0)
    W = a.width
    print(f"{a.label or os.path.basename(path)}  span {span:,} cyc  (1 col ≈ {span//W:,} cyc)")
    for tid in sorted(t for t in set(e["tid"] for e in u) if U(t)):
        line = [" "] * W
        for e in u:
            if e["tid"] != tid:
                continue
            x0 = int((e["ts"] - t0) / span * W); x1 = max(x0, int((e["ts"] + e["dur"] - t0) / span * W) - 1)
            n = e["name"]
            ch = ("M" if ("MatMul" in n or "ConvLayer" in n) else
                  ("S" if ("Solve" in n) else ("." if any(g in n for g in GLUE) else "#")))
            for i in range(x0, min(x1 + 1, W)):
                if line[i] == " ":
                    line[i] = ch
        print(f"  {U(tid)} {tid} |{''.join(line)}|")

    real = lambda e: not any(g in e["name"] for g in GLUE)
    hmx = [e for e in u if U(e["tid"]) == "HMX" and real(e)]
    hvx = [e for e in u if U(e["tid"]) == "HVX" and real(e)]
    def merge(iv):
        iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in iv); o = []
        for s, t in iv:
            if o and s <= o[-1][1]:
                o[-1] = (o[-1][0], max(o[-1][1], t))
            else:
                o.append((s, t))
        return o
    tot = lambda m: sum(t - s for s, t in m)
    overlap = lambda A, B: sum(max(0, min(t, t2) - max(s, s2)) for s, t in A for s2, t2 in B)
    mh, mv = merge(hmx), merge(hvx)
    inter = overlap(mh, mv)
    pct = inter / max(1, tot(mh)) * 100 if mh else 0
    if mh and mv:
        print(f"  HMX busy {tot(mh):,}  span [{mh[0][0]:,}..{mh[-1][1]:,}]")
        print(f"  HVX busy {tot(mv):,}  span [{mv[0][0]:,}..{mv[-1][1]:,}]")
        print(f"  HMX∩(any HVX) {inter:,}  =>  {pct:.0f}% of HMX overlaps the HVX unit "
              f"(incl. the matmul's OWN layout glue)")
    else:
        print(f"  (HMX ops={len(hmx)} HVX ops={len(hvx)} — need both for overlap)")
    headline = pct
    if a.producer:
        prod = merge([e for e in u if U(e["tid"]) == "HVX" and a.producer in e["name"]])
        pinter = overlap(mh, prod)
        ppct = pinter / max(1, tot(mh)) * 100 if mh else 0
        if prod:
            print(f"  {a.producer} span [{prod[0][0]:,}..{prod[-1][1]:,}]  "
                  f"∩HMX {pinter:,}  =>  {ppct:.0f}% of HMX overlaps the CUSTOM op '{a.producer}'")
        else:
            print(f"  (no HVX op matching '{a.producer}')")
        headline = ppct
    print(f"OVERLAP {headline:.0f}%")

if __name__ == "__main__":
    main()
