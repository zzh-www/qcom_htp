#!/usr/bin/env python3
"""htp_perf_report.py — THE canonical GDN-solve perf report generator (P2.2).

Turns gdnbm `stats[]` output (one or more P-points) into the §7 perf-report template from
docs/cycle_metric_alignment.md § "HTP Kernel Measurement Standard":
  * mandatory file header (§7),
  * canonical-field table in §4 order (with the §4 stats[n]->field disambiguation: stats[4] =
    feed_Σ on HVXMix but spin_Σ on pure-HMX; lmax = stats[8] on HVXMix but stats[11] on pure-HMX),
  * serial floor as the per-P honest-tail  tail(P) = wall(P) - 实测feed_Σ(P)/P  (§3), NOT the
    constant-K regression intercept,
  * utilization口径 label (§6): PMU真值 if a PMU pass is supplied, else "steady-derived (non-PMU)".

INPUT — a small JSON spec (file path or stdin), one entry per P-point:
  {
    "title":  "ARES W16_N8 — b_serial honest-tail",
    "device": "oneplus (termux sshd; v75 cDSP TURBO)",
    "reproduce": "bash scripts/ares_bserial_psweep.sh",
    "points": [
      { "path": "hvxmix", "P": 4, "stats": [2521897,4,32,288599,9070354,1582,1582,2299911,2402059,87062,2299911,85250] },
      { "path": "hvxmix", "P": 1, "stats": [7932240,1,32,254093,7570102,4977,4977,7849889,7849889,279787,7849889,279787] }
    ],
    "pure_feed": { "4": 3643000, "2": 0 },     # OPTIONAL: pure-HMX feed_Σ is a production decomposition
                                               #           value, NOT in stats[] — supply it for honest-tail.
    "pmu":   { "4": { "coproc_busy": 0.69, "thread_idle": 0.13 } }  # OPTIONAL §6 tier-1 PMU真值 per P.
  }
The "path" of each point selects the §4 stats[n] mapping ("hvxmix" SHIP/ARES or "pure" pure-HMX).

USAGE:
  htp_perf_report.py spec.json [out.txt]      # out.txt default stdout
  echo '<json>' | htp_perf_report.py - [out.txt]

The honest-tail and the report header are computed/emitted here so "统一可信" is a default product,
not a hand-typed table. Cross-check: feeding the perf_3impl / perf_ares_bserial_pin raw stats reproduces
their field tables (ARES honest-tail @P4 = 0.254M).
"""
import sys, json

# §4 canonical stats[n] -> field, PER PATH (the disambiguation authority). value None = not in stats[].
# HVXMix (SHIP/ARES): stats[4]=feed_Σ, stats[8]=lmax, spin_Σ=stats[9]*P.
# pure-HMX (GP):      stats[4]=spin_Σ, stats[11]=lmax, feed_Σ NOT in stats (production decomposition).
HVXMIX_MAP = {
    "wall": 0, "cbusy": 3, "feed_sigma": 4, "us": 6, "lmin": 7, "lmax": 8,
    "spin_avg": 9, "wtpack_sigma": 24,
}
PURE_MAP = {
    "wall": 0, "cbusy": 3, "spin_sigma": 4, "percall_occ": 5, "us": 6,
    "wtpack_sigma": 7, "scatter_sigma": 9, "other_sigma": 10, "lmax": 11,
}


def m(stats, i):
    return stats[i] if (i is not None and i < len(stats)) else None


def field_for(point):
    """Extract the canonical fields for one P-point, honoring the §4 path disambiguation."""
    path = point["path"]
    s = point["stats"]
    P = point.get("P") or (m(s, 1) if len(s) > 1 else None)
    f = {"P": P, "path": path}
    if path == "hvxmix":
        f["wall"] = m(s, HVXMIX_MAP["wall"])
        f["cbusy"] = m(s, HVXMIX_MAP["cbusy"])
        f["feed_sigma"] = m(s, HVXMIX_MAP["feed_sigma"])
        f["lmax"] = m(s, HVXMIX_MAP["lmax"])
        f["us"] = m(s, HVXMIX_MAP["us"])
        spin_avg = m(s, HVXMIX_MAP["spin_avg"])
        f["spin_sigma"] = (spin_avg * P) if (spin_avg is not None and P) else None
        f["percall_occ"] = None
    elif path == "pure":
        f["wall"] = m(s, PURE_MAP["wall"])
        f["cbusy"] = m(s, PURE_MAP["cbusy"])
        f["spin_sigma"] = m(s, PURE_MAP["spin_sigma"])
        f["lmax"] = m(s, PURE_MAP["lmax"])
        f["us"] = m(s, PURE_MAP["us"])
        f["percall_occ"] = m(s, PURE_MAP["percall_occ"])
        # pure feed_Σ is a production decomposition value, NOT in stats[]; supplied via spec["pure_feed"].
        f["feed_sigma"] = None
    else:
        sys.exit(f"unknown path '{path}' (use 'hvxmix' or 'pure')")
    return f


def fmtM(v):
    return "—" if v is None else f"{v/1e6:.3f}M"


def honest_tail(wall, feed_sigma, P):
    """§3: tail(P) = wall(P) - 实测feed_Σ(P)/P. Returns None if feed_Σ unknown."""
    if wall is None or feed_sigma is None or not P:
        return None
    return wall - feed_sigma / P


def util_str(point, f, pmu):
    """§6 utilization口径: PMU真值 (tier-1) if supplied, else steady-derived (tier-2), labelled."""
    P = f["P"]
    pm = pmu.get(str(P)) if pmu else None
    if pm:
        cb = pm.get("coproc_busy"); idle = pm.get("thread_idle")
        parts = []
        if cb is not None:
            parts.append(f"HMX COPROC_BUSY/CYCLES={cb*100:.0f}%")
        if idle is not None:
            parts.append(f"THREAD_IDLE={idle*100:.0f}%")
        return "PMU真值 (§6 tier-1): " + ", ".join(parts)
    # steady-derived: (feed/P)/lmax if feed known, else cbusy/wall.
    if f.get("feed_sigma") and f.get("lmax") and P:
        u = (f["feed_sigma"] / P) / f["lmax"]
        return f"steady-derived (non-PMU, §6 tier-2): HVX util/thread (feed/P)/lmax = {u*100:.1f}%"
    if f.get("cbusy") and f.get("wall"):
        u = f["cbusy"] / f["wall"]
        return f"steady-derived (non-PMU, §6 tier-2): HMX util cbusy/wall = {u*100:.1f}%"
    return "steady-derived (non-PMU, §6 tier-2): insufficient fields"


def render(spec):
    title = spec.get("title", "<untitled>")
    device = spec.get("device", "oneplus (termux sshd; v75 cDSP TURBO)")
    reproduce = spec.get("reproduce", "<reproduce command>")
    pure_feed = {str(k): v for k, v in (spec.get("pure_feed") or {}).items()}
    pmu = spec.get("pmu") or {}
    points = [field_for(p) for p in spec["points"]]
    # fill in pure-HMX feed_Σ from the supplied decomposition (it is NOT in stats[]).
    for f in points:
        if f["path"] == "pure" and f["feed_sigma"] is None:
            v = pure_feed.get(str(f["P"]))
            if v:
                f["feed_sigma"] = v
    points.sort(key=lambda f: -(f["P"] or 0))
    L = []
    bar = "=" * 80
    L.append(bar)
    L.append(title)
    L.append(bar)
    L.append("DATE     : (fill at write time)")
    L.append(f"DEVICE   : {device}")
    L.append("口径     : 32-head TOTAL wall (stats[0]=makespan), VTCM-only, reps2-N median (绝不取 min),")
    L.append("           同热窗 ACAC 配对取 delta; PCYCLE=C15:14=QHAS (no conversion).")
    L.append("clock self-check : wall/µs = PCYCLE/µs ≈ TURBO 1594 (>> => wrong counter, re-measure).")
    paths = sorted({f["path"] for f in points})
    L.append(f"字段映射 : §4 dictionary; paths present = {paths}.")
    L.append("           ⚠ stats[4] = feed_Σ on HVXMix but spin_Σ on pure-HMX; lmax = stats[8] (HVXMix)")
    L.append("           vs stats[11] (pure-HMX). pure-HMX feed_Σ = production decomposition (not in stats).")
    L.append(f"REPRODUCE: {reproduce}")
    L.append("-" * 80)
    # canonical-field table (§4 order): wall | cbusy | feed_Σ | spin_Σ | lmax | per-call occ | wt-pack Σ
    hdr = ("| P | path    | wall    | cbusy  | feed_Σ  | spin_Σ | lmax    | percall | "
           "honest-tail (§3) |")
    L.append(hdr)
    L.append("|---|---------|---------|--------|---------|--------|---------|---------|------------------|")
    for f in points:
        tail = honest_tail(f["wall"], f.get("feed_sigma"), f["P"])
        L.append("| {P} | {path:7s} | {wall:7s} | {cb:6s} | {feed:7s} | {spin:6s} | {lmax:7s} | "
                 "{occ:7s} | {tail:16s} |".format(
            P=f["P"], path=f["path"], wall=fmtM(f["wall"]), cb=fmtM(f["cbusy"]),
            feed=fmtM(f.get("feed_sigma")), spin=fmtM(f.get("spin_sigma")), lmax=fmtM(f.get("lmax")),
            occ=("—" if f.get("percall_occ") is None else str(f["percall_occ"])),
            tail=(fmtM(tail) if tail is not None else "— (no feed_Σ)")))
    L.append("")
    L.append("serial floor (§3 honest-tail, NOT constant-K regression intercept):")
    tails = [(f["P"], honest_tail(f["wall"], f.get("feed_sigma"), f["P"])) for f in points]
    tails = [(P, t) for (P, t) in tails if t is not None]
    if tails:
        for P, t in tails:
            L.append(f"  tail(P={P}) = {t/1e6:.3f}M")
        flat = max(t for _, t in tails) - min(t for _, t in tails)
        L.append(f"  spread across P = {flat/1e6:.3f}M  ({'≈flat (true serial floor)' if flat < 0.15e6 else 'check feed_Σ-P growth (contention folds into K, §3)'})")
    else:
        L.append("  (no honest-tail: feed_Σ unknown for all points; supply pure_feed or use a path with stats[4]=feed_Σ)")
    L.append("")
    L.append("utilization (§6):")
    for f in points:
        L.append(f"  P={f['P']}: {util_str(f, f, pmu)}")
    L.append(bar)
    return "\n".join(L)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    raw = sys.stdin.read() if src == "-" else open(src).read()
    spec = json.loads(raw)
    txt = render(spec)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(txt + "\n")
        print(f"wrote {sys.argv[2]}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
