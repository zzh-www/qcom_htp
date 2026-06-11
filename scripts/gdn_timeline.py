#!/usr/bin/env python3
"""Render the full per-thread ASCII timeline from a QNN optrace chrometrace.json + extract the perf facts
that aggregate cyc tables hide: HVX∥HMX overlap, the @Spill/@Fill/Concat boundary tax, per-unit busy, and
total wall span (→ per-head).  MANDATED for every perf step (skill qnn-htp-profiling, METHOD MANDATE).

Usage: gdn_timeline.py <chrometrace.json> [H] [T.raw] [C]   (T.raw+C optional -> also prints relerr vs T_full_ref.raw)
"""
import sys, json, collections, os

TR = sys.argv[1]
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TRAW = sys.argv[3] if len(sys.argv) > 3 else None
C = int(sys.argv[4]) if len(sys.argv) > 4 else 0
W = 92
GROUPS = ("GdnSolveDiag", "GdnMergeHmx", "Spill", "Fill", "Concat", "ForceFormat", "flat_from_vtcm",
          "convert_we", "ConvLayer", "Slice", "Reshape", "DmaCheckpoint")

ev = json.load(open(TR))["traceEvents"]
tn = {(e.get("pid"), e.get("tid")): e["args"]["name"]
      for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
def unit(e):
    n = tn.get((e.get("pid"), e.get("tid")), "")
    return "HMX" if "HMX" in n else ("HVX" if "HVX" in n else "sc")
X = [e for e in ev if e.get("ph") == "X" and e.get("dur", 0) > 0]
if not X:
    print("  (no X events)"); sys.exit()
t0 = min(e["ts"] for e in X); t1 = max(e["ts"] + e["dur"] for e in X); span = t1 - t0

# per-op-group: [start, end, busy_sum, units]
sp = collections.defaultdict(lambda: [1e18, -1e18, 0, set()])
for e in X:
    nm = str(e.get("name", ""))
    k = next((g for g in GROUPS if g in nm), nm[:14])
    s = sp[k]; s[0] = min(s[0], e["ts"]); s[1] = max(s[1], e["ts"] + e["dur"]); s[2] += e["dur"]; s[3].add(unit(e))

# union-busy + cross-unit overlap (HVX vs HMX)
def union(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out
def total(iv): return sum(b - a for a, b in iv)
def ov(x, y):
    i = j = p = 0
    while i < len(x) and j < len(y):
        a = max(x[i][0], y[j][0]); b = min(x[i][1], y[j][1])
        if a < b: p += b - a
        i, j = (i + 1, j) if x[i][1] < y[j][1] else (i, j + 1)
    return p
hv = union([(e["ts"], e["ts"] + e["dur"]) for e in X if unit(e) == "HVX"])
hm = union([(e["ts"], e["ts"] + e["dur"]) for e in X if unit(e) == "HMX"])
overlap = ov(hv, hm)
boundary = sum(s[2] for k, s in sp.items() if k in ("Spill", "Fill", "Concat", "flat_from_vtcm", "ForceFormat", "convert_we"))

print(f"  --- ASCII timeline (span={span:,} cyc = {span//max(1,H):,}/head, width={W}) ---")
for k, (a, b, busy, u) in sorted(sp.items(), key=lambda kv: kv[1][0]):
    i = int((a - t0) / span * W); j = max(i + 1, int((b - t0) / span * W))
    bar = " " * i + "#" * min(W - i, j - i)
    print(f"  {k:14}{'/'.join(sorted(u)):7}|{bar:<{W}}| busy={busy:>10,}")
print(f"  HVXbusy={total(hv):,}  HMXbusy={total(hm):,}  overlap={overlap:,} "
      f"({100*overlap/max(1,min(total(hv),total(hm))):.0f}% of min-unit)")
print(f"  boundary glue (Spill+Fill+Concat+flat_from_vtcm+ForceFormat) = {boundary:,} = {boundary//max(1,H):,}/head")
print(f"  >>> total {span:,}/{H} = {span//max(1,H):,} cyc/head   vs baseline 70,201  "
      f"({span/max(1,H)/70201:.2f}x)")

if TRAW and C and os.path.exists(TRAW) and os.path.exists("T_full_ref.raw"):
    import numpy as np
    t = np.fromfile(TRAW, dtype=np.float32)
    if t.size >= H*C*C:
        t = t[:H*C*C].reshape(H, C, C); r = np.fromfile("T_full_ref.raw", dtype=np.float32)[:H*C*C].reshape(H, C, C)
        hs = list(range(1, H)) if H > 1 else [0]
        rel = np.mean([np.linalg.norm(t[h]-r[h])/(np.linalg.norm(r[h])+1e-12) for h in hs])
        print(f"  relerr vs np.linalg.inv (heads {hs[0]}..{hs[-1]}) mean = {rel:.3e}")
    else:
        print(f"  (T truncated: {t.size} < {H*C*C})")
