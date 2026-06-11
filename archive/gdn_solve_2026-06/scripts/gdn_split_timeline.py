#!/usr/bin/env python3
"""Render the full per-thread ASCII timeline from a QNN optrace chrometrace.json
for the M8 per-chunk split graph, + the M8 measurement numbers:
  1. Spill/Fill+Concat+flat_from_vtcm sum (cyc + per-head)
  2. cross-tid HVX|HMX overlap %
  3. total span / H = cyc/head vs 70,201
Usage: gdn_split_timeline.py <out_s_dir> <H> [CK]
"""
import json, sys, collections

d = sys.argv[1]; H = int(sys.argv[2]); CK = sys.argv[3] if len(sys.argv) > 3 else "?"
ev = json.load(open(f"{d}/optrace/chrometrace.json"))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
tn = {(e.get("pid"), e.get("tid")): e["args"]["name"]
      for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name" and "args" in e}
X = [e for e in ev if e.get("ph") == "X" and e.get("dur", 0) > 0]
t0 = min(e["ts"] for e in X); t1 = max(e["ts"] + e["dur"] for e in X)
span = t1 - t0


def unit(e):
    nm = tn.get((e["pid"], e["tid"]), "")
    return "HMX" if "HMX" in nm else "HVX" if "HVX" in nm else "scal"


def tidlabel(e):
    return e.get("tid")


# ---- grouped op-span rows (collapse node instances of the same kind) ----
GROUPS = ("GdnSolveDiag", "GdnMergeHmx", "Spill", "Fill", "Concat",
          "ForceFormat", "flat_from_vtcm", "ConvLayer")
sp = collections.defaultdict(lambda: [10**18, -10**18, 0, set(), 0])
for e in X:
    k = next((g for g in GROUPS if g in str(e["name"])), str(e["name"])[:16])
    s = sp[k]
    s[0] = min(s[0], e["ts"]); s[1] = max(s[1], e["ts"] + e["dur"])
    s[2] += e["dur"]; s[3].add(unit(e)); s[4] += 1

W = 90; sc = span / W if span else 1
print(f"\n=== ASCII TIMELINE  CK={CK} H={H}  total span={span:,} cyc  (per-head={span/H:,.0f}) ===")
print(f"{'op-group':16}{'unit':5}|{'0':<{W}}| span(cyc)        busy(sum)  n")
for k, (a, b, busy, u, n) in sorted(sp.items(), key=lambda kv: kv[1][0]):
    i = int((a - t0) / sc); j = max(i + 1, int((b - t0) / sc))
    bar = " " * i + "#" * (j - i)
    print(f"{k:16}{'/'.join(sorted(u)):5}|{bar:<{W}}| {a-t0:>9,}-{b-t0:<9,} {busy:>11,} {n}")

# ---- per-tid rows (HVX 512-515, HMX 256) ----
print(f"\n--- per-tid rows (raw threads) ---")
bytid = collections.defaultdict(lambda: [10**18, -10**18, 0, set()])
for e in X:
    if e.get("dur", 0) < 200:
        continue
    t = tidlabel(e); s = bytid[t]
    s[0] = min(s[0], e["ts"]); s[1] = max(s[1], e["ts"] + e["dur"])
    s[2] += e["dur"]; s[3].add(unit(e))
for t, (a, b, busy, u) in sorted(bytid.items()):
    i = int((a - t0) / sc); j = max(i + 1, int((b - t0) / sc))
    bar = " " * i + "#" * (j - i)
    print(f"tid {str(t):11} {'/'.join(sorted(u)):5}|{bar:<{W}}| busy={busy:,}")

# ---- (1) boundary glue sum ----
glue_keys = ("Spill", "Fill", "Concat", "flat_from_vtcm", "ForceFormat")
glue = 0
glue_detail = {}
for k, (a, b, busy, u, n) in sp.items():
    if any(g in k for g in glue_keys):
        glue += busy; glue_detail[k] = busy
print(f"\n(1) BOUNDARY GLUE (Spill/Fill/Concat/flat_from_vtcm/ForceFormat) sum={glue:,} cyc  "
      f"per-head={glue/H:,.0f}")
for k, v in sorted(glue_detail.items()):
    print(f"      {k}: {v:,}")

# ---- (2) HVX|HMX cross-tid overlap % ----
# Build per-unit busy intervals (merged) on the timeline; overlap = intersection of HVX-union & HMX-union.
def intervals(pred):
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in X if e["dur"] > 0 and pred(e))
    merged = []
    for s, en in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([s, en])
    return merged


hvx = intervals(lambda e: unit(e) == "HVX")
hmx = intervals(lambda e: unit(e) == "HMX")


def union_len(iv):
    return sum(b - a for a, b in iv)


def overlap_len(a_iv, b_iv):
    ov = 0; i = j = 0
    while i < len(a_iv) and j < len(b_iv):
        lo = max(a_iv[i][0], b_iv[j][0]); hi = min(a_iv[i][1], b_iv[j][1])
        if hi > lo:
            ov += hi - lo
        if a_iv[i][1] < b_iv[j][1]:
            i += 1
        else:
            j += 1
    return ov


hvx_busy = union_len(hvx); hmx_busy = union_len(hmx)
ov = overlap_len(hvx, hmx)
hvx_span = (hvx[-1][1] - hvx[0][0]) if hvx else 0
hmx_span = (hmx[-1][1] - hmx[0][0]) if hmx else 0
denom = min(hvx_busy, hmx_busy) or 1
print(f"\n(2) HVX|HMX OVERLAP: HVX_busy={hvx_busy:,} (span {hvx_span:,})  "
      f"HMX_busy={hmx_busy:,} (span {hmx_span:,})  overlap={ov:,} cyc  "
      f"= {100*ov/denom:.1f}% of min-busy, {100*ov/span:.1f}% of total span")

# ---- (3) cyc/head ----
print(f"\n(3) TOTAL WALL SPAN / H = {span/H:,.0f} cyc/head   vs baseline 70,201  "
      f"({span/H/70201:.2f}x)")
