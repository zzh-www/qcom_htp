#!/usr/bin/env python3
"""Analyze a QNN HTP optrace chrometrace.json for HVX || HMX overlap.

Reads thread_name metadata to classify tids as HMX (tid 256) vs HVX (tids 512-515), computes each
unit's busy span (union of its X-event [ts, ts+dur) intervals), the cross-unit overlap (intersection
of HVX-union and HMX-union), and the overlap fraction.  Also reports per-op-name busy cycles per unit.

Usage: gdn_split_overlap_analyze.py <chrometrace.json> <label> <H>
"""
import json, sys
from collections import defaultdict

path, label, H = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(path))

tname = {}
for e in c["traceEvents"]:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        tname[e.get("tid")] = e.get("args", {}).get("name", "")

def unit_of(tid):
    n = tname.get(tid, "")
    if "HMX" in n: return "HMX"
    if "HVX" in n: return "HVX"
    return None

ev = [e for e in c["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
intervals = defaultdict(list)   # unit -> [(ts, ts+dur)]
busyname = defaultdict(lambda: defaultdict(float))  # unit -> name -> sum dur
for e in ev:
    u = unit_of(e.get("tid"))
    if u is None: continue
    ts = e["ts"]; dur = e["dur"]
    intervals[u].append((ts, ts + dur))
    busyname[u][e["name"]] += dur

def union(iv):
    if not iv: return [], 0.0, None, None
    iv = sorted(iv)
    merged = [list(iv[0])]
    for s, en in iv[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], en)
        else: merged.append([s, en])
    total = sum(b - a for a, b in merged)
    return merged, total, merged[0][0], merged[-1][1]

def intersect(a, b):
    i = j = 0; tot = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
        if hi > lo: tot += hi - lo
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return tot

hvx_u, hvx_busy, hvx_s, hvx_e = union(intervals["HVX"])
hmx_u, hmx_busy, hmx_s, hmx_e = union(intervals["HMX"])
ov = intersect(hvx_u, hmx_u) if hvx_u and hmx_u else 0.0

print(f"  [{label}] per-unit busy spans (chrometrace units = cycles):")
print(f"    HVX busy={hvx_busy:>12,.0f}  span=[{hvx_s},{hvx_e}]  events={len(intervals['HVX'])}")
print(f"    HMX busy={hmx_busy:>12,.0f}  span=[{hmx_s},{hmx_e}]  events={len(intervals['HMX'])}")
if hvx_u and hmx_u:
    base = min(hvx_busy, hmx_busy) or 1.0
    print(f"    cross-unit OVERLAP = {ov:,.0f}  -> {ov/base*100:.1f}% of min(HVX,HMX) busy   "
          f"({ov/max(hvx_busy,hmx_busy,1)*100:.1f}% of max)")
    print(f"    HVX/head={hvx_busy/max(H,1):,.0f}  HMX/head={hmx_busy/max(H,1):,.0f}")
# top ops per unit
for u in ("HVX", "HMX"):
    if busyname[u]:
        top = sorted(busyname[u].items(), key=lambda x: -x[1])[:5]
        print(f"    {u} top: " + " | ".join(f"{n.split('::')[-1][:24]}={d:,.0f}" for n, d in top))

# memory counters: confirm intermediates stay VTCM-resident (DRAM traffic << VTCM traffic)
mem = {}
for e in c["traceEvents"]:
    if e.get("ph") == "C" and e.get("name") in ("VTCM read", "VTCM write", "DRAM read", "DRAM write"):
        mem[e["name"]] = mem.get(e["name"], 0) + (e.get("args", {}).get("", 0) or 0)
if mem:
    vt = mem.get("VTCM read", 0) + mem.get("VTCM write", 0)
    dr = mem.get("DRAM read", 0) + mem.get("DRAM write", 0)
    print(f"    mem: VTCM {vt:,.0f} B  DRAM {dr:,.0f} B  -> DRAM/VTCM = {dr/max(vt,1)*100:.1f}%  "
          f"({'VTCM-resident' if dr < 0.1*vt else 'DDR SPILL'})")
