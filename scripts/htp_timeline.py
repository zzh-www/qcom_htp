#!/usr/bin/env python3
"""htp_timeline.py — THE canonical per-thread GDN-solve timeline renderer (P2.1, replaces the 5 ad-hoc
scripts). The stage-id -> class table below is the SINGLE source of truth, copied verbatim from
docs/cycle_metric_alignment.md § "HTP Kernel Measurement Standard" §5 (the cross-impl stage spec).
Do NOT re-define a stage map anywhere else — point at this file.

Trace blob format (shared, magic 0x47545203), §5:
  [magic u32][n u32][wall u64][base u64] then n*{tid u32, stage u32, t0 u64, t1 u64}
  tid 0..P-1 = HVX producers, tid GP_NT(=4) = the main-thread HMX consumer (the ONLY MM owner).

Two modes:
  single   : one impl, one swimlane per thread (P0..P3 producers + CONS consumer).
  aggregate: 3 impls stacked per thread ("p{t}({impl})") so the per-route stage MIX is comparable.

§5 HARD RULES enforced here (so every render is on the SAME 口径):
  * MM (stage 3) is drawn ONLY on the consumer tid. A producer never runs mxmem.
  * Stage 5 is a PACK LEAF in BOTH taxonomies (HVXMix PREP and pure-HMX WT-PACK both -> PACK leaf) —
    never a skipped container. (The old HVXMix renderer掏空ed producer rows by putting 5 in container.)
  * single-rep trace => trace-perturbed: a prominent caveat band says read STRUCTURE only; utilization
    is taken from the steady-state perf 表, NEVER off this figure (§5 + §6).

Usage:
  htp_timeline.py single   <impl: ship|ares|hvxmix|purehmx> <trace.raw> <out.svg>
  htp_timeline.py aggregate <ship.raw> <ares.raw> <purehmx.raw> <out.svg>

Capture blobs: scripts/gdn_capture_3impl_traces.sh  (or the archived ones in
  Agent/current/trace_blobs_3impl/{ship,w16n8,purehmx}_trace.raw).
"""
import sys, struct

# ===========================================================================================
# §5 CANONICAL stage-id -> CLASS table (THE single source of truth — docs §5). Two taxonomies,
# one colour vocabulary. The ONLY difference between HVXMix and pure-HMX is which raw ids exist;
# stage 5 is a PACK LEAF in BOTH (symmetry fix 2026-06-17). Containers = {0,2} (skip, draw nested).
# ===========================================================================================
HVXMIX_CLASS = {1: "DIAG", 3: "MM", 4: "ACT", 5: "PACK", 6: "ACC", 7: "REQ", 8: "PACK", 9: "EFF",
                10: "DEPACK", 11: "SPIN", 12: "LOAD", 13: "STORE", 14: "POST", 15: "TABS"}
GP_CLASS = {3: "MM", 4: "ACT", 5: "PACK", 6: "ACC", 10: "DEPACK", 11: "SPIN", 12: "LOAD", 13: "STORE"}
CONTAINER = {0, 2}   # HEAD / MERGE / WT-PACK-wrapper -> skip and draw nested (§5)

# §5: pure-HMX (GP) uses GP_CLASS; everything else (SHIP / ARES / generic HVXMix) uses HVXMIX_CLASS.
IMPL_TAXONOMY = {"ship": False, "ares": False, "hvxmix": False, "purehmx": True}

# One colour per semantic class, shared across impls (blue family = HVX operand-prep; orange = HMX MM;
# purple = DIAG; violet = ACC/REQ renorm; rose = SPIN; neutral = bookkeeping).
COLOR = {
    "PACK":   "#4e79a7",   # kmajor/crouton pack (HVXMix PACK · pure-HMX WT-PACK) — saturated blue
    "EFF":    "#8fc3e8",   # effective (-128*Σwt), HVXMix-only — light blue
    "DEPACK": "#9cc3e0",   # depack / OUT-COPY 归一 — pale blue
    "DIAG":   "#7b6fb0",   # diagonal forward-subst, HVXMix-only — purple
    "ACT":    "#59a14f",   # act / quant format — green
    "ACC":    "#b3a2d4",   # int32 acc / renorm — light violet
    "REQ":    "#8c7bb4",   # widen + requant (merge-final), HVXMix-only — violet
    "MM":     "#e08a52",   # the 64³ matmul on HMX (consumer) — orange
    "SPIN":   "#e8a0a8",   # producer idle-waiting on the 1 HMX — muted rose
    "LOAD":   "#bab0ac", "STORE": "#cfc6c2", "POST": "#e6e0db", "TABS": "#ddd5d0",
}
LEGEND = [
    ("PACK",   "kmajor/crouton pack (all)"),
    ("MM",     "64³ matmul · HMX — CONS row ONLY (all impls)"),
    ("ACT",    "act/quant (all)"),
    ("ACC",    "int32 acc/renorm (SHIP·pure-HMX)"),
    ("DIAG",   "fwd-subst diag (SHIP·ARES only)"),
    ("REQ",    "widen+requant (SHIP·ARES only)"),
    ("EFF",    "effective -128Σwt (SHIP only)"),
    ("DEPACK", "depack/out-copy (SHIP·pure-HMX)"),
    ("SPIN",   "idle wait on HMX (all)"),
    ("LOAD",   "load/store/post/tabs (bookkeeping)"),
]


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")


def _union(intervals):
    """Total length of the union of [a,b) intervals (de-double-count nested PACK container + children)."""
    iv = sorted((a, b) for a, b in intervals if b > a)
    tot = 0; ce = None
    for a, b in iv:
        if ce is None or a > ce:
            tot += b - a; ce = b
        elif b > ce:
            tot += b - ce; ce = b
    return tot


def load(path, is_gp):
    raw = open(path, "rb").read()
    magic, n = struct.unpack_from("<II", raw, 0)
    assert magic == 0x47545203, f"{path}: bad magic {magic:#x}"
    wall, base = struct.unpack_from("<QQ", raw, 8)
    cls = GP_CLASS if is_gp else HVXMIX_CLASS
    evs = []
    off = 24
    mm_nonconsumer = 0
    # consumer = the tid that owns MM(3) — must be the SINGLE tid that ever emits stage 3 (§5).
    raw_evs = []
    for _ in range(n):
        tid, stage, t0, t1 = struct.unpack_from("<IIQQ", raw, off); off += 24
        raw_evs.append((tid, stage, t0, t1))
    mm_tids = sorted({tid for (tid, st, a, b) in raw_evs if st == 3 and b > a})
    cons = mm_tids[-1] if mm_tids else max(tid for (tid, st, a, b) in raw_evs)
    for (tid, stage, t0, t1) in raw_evs:
        if t1 <= t0 or stage in CONTAINER:
            continue
        # §5 hard rule: MM only on the consumer tid; a producer-row MM is a double-count bug -> drop+flag.
        if stage == 3 and tid != cons:
            mm_nonconsumer += 1
            continue
        evs.append((tid, cls.get(stage, "LOAD"), t0, t1))
    tmin = min(e[2] for e in evs)
    span = max(1, max(e[3] for e in evs) - tmin)
    return {"evs": evs, "tmin": tmin, "span": span, "wall": wall, "cons": cons,
            "mm_nonconsumer": mm_nonconsumer}


# ------------------------------------------------------------------------------------------------------
# SINGLE-impl render: one swimlane per thread.
# ------------------------------------------------------------------------------------------------------
SINGLE_TITLE = {
    "ship":    "GDNSolveHVXMixHMX SHIP u8i8 — pipeline trace (Perfetto style)",
    "ares":    "GDNSolveHVXMixHMX W16_N8 ARES — pipeline trace (Perfetto style)",
    "hvxmix":  "GDNSolveHVXMixHMX (HVX-feed + HMX-matmul) — pipeline trace (Perfetto style)",
    "purehmx": "GDNSolveHMX (pure-HMX) — pipeline trace (Perfetto style)",
}


def render_single(impl, path, out):
    is_gp = IMPL_TAXONOMY[impl]
    d = load(path, is_gp)
    evs = d["evs"]; wall = d["wall"]; cons = d["cons"]
    tids = sorted({e[0] for e in evs})
    order = [t for t in tids if t != cons] + ([cons] if cons in tids else [])
    tmin = d["tmin"]; SPAN = d["span"]

    L, R, TOP = 64, 168, 110
    W = 1180
    TH, GAP = 38, 10
    plotw = W - L - R
    rows = len(order)
    ploth = rows * (TH + GAP) - GAP
    H = TOP + ploth + 168

    def X(t): return L + (t - tmin) / SPAN * plotw

    def label(t): return "CONS" if t == cons else f"P{t}"

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
         '<rect width="100%" height="100%" fill="#fbfbfc"/>',
         f'<text x="24" y="34" fill="#1f2933" font-size="19" font-weight="600">{esc(SINGLE_TITLE[impl])}</text>',
         f'<text x="24" y="56" fill="#52606d" font-size="12.5">4 HVX producers (P0–P{max(tids)-1 if cons in tids else max(tids)}) '
         f'feed one PURE-HMX consumer (CONS) · MM on CONS only (§5) · {len(evs)} drawn events · '
         f'in-trace wall {wall/1e6:.2f}M cyc.</text>']
    # §5 caveat band
    s.append(f'<rect x="24" y="62" width="{W-48}" height="38" rx="4" fill="#fdecde" stroke="#e0a878" stroke-width="1"/>')
    s.append(f'<text x="34" y="78" fill="#a8551f" font-size="11" font-weight="600">'
             f'⚠ SINGLE-REP TRACE — read STAGE STRUCTURE ONLY (stage mix + relative length). Do NOT read '
             f'per-thread utilization/occupancy off this figure (trace-perturbed).</text>')
    s.append(f'<text x="34" y="93" fill="#a8551f" font-size="11">'
             f'Authoritative steady-state util/occupancy = Agent/current/perf_3impl_cron82kqie.txt '
             f'(§6 PMU真值 if present, else steady-derived). §5 stage map = docs/cycle_metric_alignment.md §5.</text>')
    if d["mm_nonconsumer"]:
        s.append(f'<text x="34" y="106" fill="#b00020" font-size="10.5" font-weight="600">'
                 f'NOTE: dropped {d["mm_nonconsumer"]} non-consumer MM events (§5 double-count guard).</text>')

    for i in range(0, 11):
        gx = L + plotw * i / 10
        cyc = SPAN * i / 10
        s.append(f'<line x1="{gx:.1f}" y1="{TOP-6}" x2="{gx:.1f}" y2="{TOP+ploth}" stroke="#eceef0" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{TOP-12}" text-anchor="middle" fill="#9aa5b1" font-size="11">{cyc/1e6:.2f}M</text>')
    s.append(f'<text x="{L+plotw/2:.0f}" y="{TOP+ploth+34}" text-anchor="middle" fill="#7b8794" font-size="12">'
             f'domain cycles (trace-perturbed single rep; read STRUCTURE, not absolute wall) →</text>')

    for ri, t in enumerate(order):
        y = TOP + ri * (TH + GAP)
        te = [e for e in evs if e[0] == t]
        busy = _union([(a, b) for (_, _, a, b) in te])
        is_cons = (t == cons)
        s.append(f'<rect x="{L}" y="{y}" width="{plotw}" height="{TH}" fill="{"#fff6f0" if is_cons else "#f4f7fa"}" '
                 f'stroke="#e1e5ea" stroke-width="1" rx="3"/>')
        s.append(f'<text x="{L-12}" y="{y+TH/2+5:.0f}" text-anchor="end" fill="#1f2933" font-size="14" '
                 f'font-weight="600">{label(t)}</text>')
        # busy% is in-trace-relative and explicitly flagged "(trace)" — NOT the authoritative util.
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2-2:.0f}" fill="#3e4c59" font-size="11.5">'
                 f'{busy*100//wall}% (trace)</text>')
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2+13:.0f}" fill="#9aa5b1" font-size="10">'
                 f'{"HMX mxmem" if is_cons else "HVX prep"}</text>')
        for (_, cname, t0, t1) in sorted(te, key=lambda e: -(e[3] - e[2])):
            x0 = X(t0); w = max(0.4, X(t1) - X(t0))
            s.append(f'<rect x="{x0:.2f}" y="{y+3}" width="{w:.2f}" height="{TH-6}" fill="{COLOR.get(cname, "#ccc")}"/>')

    _legend(s, TOP + ploth + 58, W)
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    flag = f" | DROPPED {d['mm_nonconsumer']} non-consumer MM" if d["mm_nonconsumer"] else ""
    print(f"wrote {out}  (single {impl}, {len(evs)} events, {rows} tracks, in-trace wall {wall} cyc{flag})")


# ------------------------------------------------------------------------------------------------------
# AGGREGATE render: 3 impls stacked per thread.
# ------------------------------------------------------------------------------------------------------
IMPL3 = [("SHIP u8i8", "1.892M", False), ("W16_N8 ARES", "2.506M", False), ("pure-HMX", "1.259M", True)]


def render_aggregate(ship_p, ares_p, pure_p, out):
    impls = [load(ship_p, False), load(ares_p, False), load(pure_p, True)]
    SPAN = max(d["span"] for d in impls)
    dropped = sum(d["mm_nonconsumer"] for d in impls)
    TIDS = [0, 1, 2, 3, 4]

    def tlabel(t): return "CONS" if t == 4 else f"p{t}"

    L, R, TOP = 124, 250, 116
    W = 1420
    RH, RGAP, GGAP = 17, 2, 12
    plotw = W - L - R
    grp_h = 3 * RH + 2 * RGAP
    ploth = len(TIDS) * grp_h + (len(TIDS) - 1) * GGAP
    H = TOP + ploth + 196

    def X(t0): return L + t0 / SPAN * plotw

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
         '<rect width="100%" height="100%" fill="#fbfbfc"/>',
         f'<text x="24" y="34" fill="#1f2933" font-size="19" font-weight="600">'
         f'GDN solve — 3 implementations stacked per thread (Perfetto style aggregate)</text>',
         f'<text x="24" y="56" fill="#52606d" font-size="12.5">Each thread (p0–p3 HVX producers + CONS '
         f'HMX consumer) shows 3 rows: (0) SHIP u8i8 · (1) W16_N8 ARES · (2) pure-HMX — same-class '
         f'stages share a colour, so the per-route stage MIX is comparable down each thread group.</text>']
    s.append(f'<rect x="24" y="62" width="{W-48}" height="26" rx="4" fill="#fdecde" stroke="#e0a878" stroke-width="1"/>')
    s.append(f'<text x="34" y="79" fill="#a8551f" font-size="11.5" font-weight="600">'
             f'⚠ SINGLE-REP TRACE — read STAGE STRUCTURE ONLY (MM unified to consumer-only). '
             f'Do NOT read per-thread utilization/occupancy off this figure — it is trace-perturbed. '
             f'Authoritative steady-state util/occupancy = Agent/current/perf_3impl_cron82kqie.txt '
             f'(SHIP 1.892M / ARES 2.506M / pure-HMX 1.259M; HVX-util 96%/94%/83%).</text>')

    for i in range(0, 11):
        gx = L + plotw * i / 10
        cyc = SPAN * i / 10
        s.append(f'<line x1="{gx:.1f}" y1="{TOP-6}" x2="{gx:.1f}" y2="{TOP+ploth}" stroke="#eceef0" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{TOP-12}" text-anchor="middle" fill="#9aa5b1" font-size="10.5">{cyc/1e6:.2f}M</text>')
    s.append(f'<text x="{L+plotw/2:.0f}" y="{TOP+ploth+30}" text-anchor="middle" fill="#7b8794" font-size="12">'
             f'shared absolute domain-cycle axis (each impl 0-based; trace-perturbed single rep) →</text>')

    for gi, t in enumerate(TIDS):
        gy = TOP + gi * (grp_h + GGAP)
        is_cons = (t == 4)
        s.append(f'<rect x="{L}" y="{gy-1}" width="{plotw}" height="{grp_h+2}" '
                 f'fill="{"#fff6f0" if is_cons else "#f4f7fa"}" stroke="#dfe4ea" stroke-width="1" rx="3"/>')
        s.append(f'<text x="{L-46}" y="{gy+grp_h/2+5:.0f}" text-anchor="end" fill="#1f2933" font-size="15" '
                 f'font-weight="700">{tlabel(t)}</text>')
        s.append(f'<text x="{L-46}" y="{gy+grp_h/2+20:.0f}" text-anchor="end" fill="#9aa5b1" font-size="9.5">'
                 f'{"HMX" if is_cons else "HVX"}</text>')
        for ii, d in enumerate(impls):
            ry = gy + ii * (RH + RGAP)
            te = [e for e in d["evs"] if e[0] == t]
            busy = _union([(e[2], e[3]) for e in te])
            wall = d["wall"]
            s.append(f'<text x="{L-6}" y="{ry+RH-4:.0f}" text-anchor="end" fill="#52606d" font-size="10" '
                     f'font-weight="600">{tlabel(t)}({ii})</text>')
            s.append(f'<text x="{L+plotw+10}" y="{ry+RH-4:.0f}" fill="#3e4c59" font-size="10">'
                     f'{IMPL3[ii][0]} · {busy*100//wall}%</text>')
            for (_, cname, t0, t1) in sorted(te, key=lambda e: -(e[3] - e[2])):
                x0 = X(t0 - d["tmin"]); w = max(0.4, X(t1 - d["tmin"]) - X(t0 - d["tmin"]))
                s.append(f'<rect x="{x0:.2f}" y="{ry+1}" width="{w:.2f}" height="{RH-2}" fill="{COLOR.get(cname, "#ccc")}"/>')
            ex = X(d["span"])
            s.append(f'<line x1="{ex:.1f}" y1="{ry+1}" x2="{ex:.1f}" y2="{ry+RH-1}" stroke="#52606d" '
                     f'stroke-width="0.8" stroke-dasharray="2,1" opacity="0.55"/>')

    ly = _legend(s, TOP + ploth + 64, W)
    s.append(f'<text x="24" y="{ly}" fill="#7b8794" font-size="11">Legend index: (0)=SHIP u8i8 1.892M · '
             f'(1)=W16_N8 ARES 2.506M · (2)=pure-HMX 1.259M (32-head total VTCM-only production walls).</text>')
    ly += 18
    s.append(f'<text x="24" y="{ly}" fill="#9aa5b1" font-size="10.5">HONESTY: each route uses its own '
             f'taxonomy (SHIP/ARES = HVX fwd-subst+off-diag merge with DIAG/REQ/EFF/DEPACK; pure-HMX = '
             f'Taylor(3), NO DIAG); only same-class colours are comparable. pure-HMX is a different '
             f'harness — the shared cyc axis is a visual aid, cross-harness per-thread occupancy is NOT '
             f'hard-comparable, read the stage MIX only.</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    flag = f", DROPPED {dropped} non-consumer MM (§5)" if dropped else ""
    print(f"wrote {out}  (SHIP {impls[0]['wall']} / ARES {impls[1]['wall']} / pure-HMX {impls[2]['wall']} "
          f"in-trace cyc, shared axis {SPAN/1e6:.2f}M{flag})")


def _legend(s, ly, W):
    s.append(f'<text x="24" y="{ly-16}" fill="#3e4c59" font-size="13" font-weight="600">'
             f'Stage classes (same colour across implementations)</text>')
    lx = 24; lw_max = W - 40
    for cname, note in LEGEND:
        label = f"{cname} — {note}"
        cell = 22 + 6.4 * len(label)
        if lx + cell > lw_max:
            lx = 24; ly += 22
        s.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" rx="3" fill="{COLOR[cname]}"/>')
        s.append(f'<text x="{lx+19}" y="{ly}" fill="#52606d" font-size="11">{esc(label)}</text>')
        lx += cell + 8
    return ly + 26


def render_ascii(impl, path, width=130):
    """ASCII per-thread timeline (replaces gdn_pipe_timeline.py), same §5 class table + MM-consumer-only."""
    is_gp = IMPL_TAXONOMY[impl]
    d = load(path, is_gp)
    evs = d["evs"]; wall = d["wall"]; cons = d["cons"]
    CH = {"DIAG": "D", "MM": "m", "ACT": "q", "PACK": "K", "ACC": "a", "REQ": "r", "EFF": "e",
          "DEPACK": "x", "SPIN": "s", "LOAD": "L", "STORE": "S", "POST": "P", "TABS": "T"}
    tids = sorted({e[0] for e in evs})
    print(f"events={len(evs)}  wall={wall} cyc  ({width} cols, 1 col = {wall/width:.0f} cyc)\n")
    for tid in tids:
        row = [" "] * width
        busy = 0
        for (t, cname, t0, t1) in evs:
            if t != tid:
                continue
            busy += t1 - t0
            c0 = int(t0 * width / wall); c1 = max(c0 + 1, int(t1 * width / wall))
            for c in range(c0, min(c1, width)):
                if row[c] == " ":
                    row[c] = CH.get(cname, "?")
        label = "CONS" if tid == cons else f"P{tid}"
        print(f"{label:5s}|{''.join(row)}| busy={busy*100//wall}%wall (trace)")
    print("\nlegend: L=LOAD q=ACT K=PACK s=SPIN(wait HMX) m=matmul(HMX,CONS only) x=DEPACK a=ACC/renorm "
          "D=DIAG r=REQ e=EFF S=STORE")
    print("⚠ single-rep trace: read STRUCTURE only; util = perf 表 (§5/§6), NOT this figure.")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "ascii":
        impl = sys.argv[2]; path = sys.argv[3]
        width = int(sys.argv[4]) if len(sys.argv) > 4 else 130
        if impl not in IMPL_TAXONOMY:
            sys.exit(f"unknown impl '{impl}'; choose from {sorted(IMPL_TAXONOMY)}")
        render_ascii(impl, path, width)
    elif mode == "single":
        impl, path, out = sys.argv[2], sys.argv[3], sys.argv[4]
        if impl not in IMPL_TAXONOMY:
            sys.exit(f"unknown impl '{impl}'; choose from {sorted(IMPL_TAXONOMY)}")
        render_single(impl, path, out)
    elif mode == "aggregate":
        render_aggregate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        sys.exit(f"unknown mode '{mode}'; use 'single' or 'aggregate'")


if __name__ == "__main__":
    main()
