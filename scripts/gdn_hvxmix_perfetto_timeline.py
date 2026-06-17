#!/usr/bin/env python3
"""Render the GDNSolveHVXMixHMX (HVX-feed + HMX-matmul) pipeline trace (-DGDN_BR_TRACE events) as a
Perfetto-style SVG, in the SAME swimlane layout / palette family / 口径 as the pure-HMX renderer
(scripts/gdn_pure_perfetto_timeline.py) so the two routes are directly comparable.

Difference from the pure-HMX renderer = the stage taxonomy: this route's 4 HVX producers run the FULL
per-head solve (DIAG forward-subst + off-diag MERGE prep/glue + QUANT + PACK + EFF + ACC + REQ +
DEPACK), each delegating its 64^3 matmul to the single main-thread PURE-HMX consumer (CONS = MM only).

Trace format (shared, magic 0x47545203):
  [magic u32][n u32][wall u64][base u64] then n*{tid u32, stage u32, t0 u64, t1 u64}
HVXMixHMX stage ids (GdnSolveBROp.cpp enum + GdnSolveBR16.cpp pushes):
  1=DIAG 2=MERGE(container) 3=MM 4=QUANT/ACT 5=PREP(merge-operand container) 6=ACC 7=REQ
  8=PACK 9=EFF 10=DEPACK 11=SPIN 12=BIAS-PACK 13=SIG 14=POST 15=TABS ; consumer tid=GDN_BR_NT(=4) pushes MM(3).

busy% is WALL-relative (busy ÷ global wall) to match the doc / ASCII renderer 口径, NOT span-relative.
Producer busy is computed as the UNION of its leaf intervals (no double-count of PREP container + its
nested ACT/PACK/EFF children).

Usage: gdn_hvxmix_perfetto_timeline.py <trace.raw> <out.svg>
Reproduce the trace: EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL \
  -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST -DGDN_BR_TRACE" build, run, pull T.raw.
"""
import sys, struct

# HVXMixHMX taxonomy. PREP(5) is a container (its nested ACT/PACK/EFF children are drawn); MERGE(2)/HEAD(0)
# are pure containers (skipped). We DRAW the children, and draw PREP only where it has NO child (rare).
STAGE = {1: "DIAG", 3: "MM", 4: "QUANT/ACT", 6: "ACC", 7: "REQ", 8: "PACK", 9: "EFF",
         10: "DEPACK", 11: "SPIN", 12: "BIAS-PACK", 13: "SIG", 14: "POST", 15: "TABS"}
CONTAINER = {0, 2, 5}   # HEAD / MERGE / PREP-merge-container -> skip (draw their leaves)

# HVX producer-prep = blue/green family; the long-pole DIAG (fwd-subst) gets the saturated blue; the
# merge-glue prep (QUANT/ACT/PACK/EFF) a lighter blue; renorm/requant (ACC/REQ) violet; HMX matmul orange.
COLOR = {
    "DIAG":      "#4e79a7",   # diagonal forward-subst (the #1 producer Σ pole) — saturated blue
    "QUANT/ACT": "#76b7e0",   # act/quant format (merge + diag) — light blue
    "PACK":      "#8fc3e8",   # kmajor / crouton pack — lighter blue
    "EFF":       "#aecde0",   # effective (-128*Σwt) — pale blue
    "REQ":       "#8c7bb4",   # widen + requant (merge-final) — violet (2nd pole)
    "ACC":       "#b3a2d4",   # int32 acc / renorm — light violet
    "DEPACK":    "#59a14f",   # surface -> int16 depack — green
    "MM":        "#e08a52",   # the 64^3 matmul on HMX (consumer) — orange (HMX convention)
    "SPIN":      "#e8a0a8",   # producer idle-waiting on the 1 HMX — muted rose
    "BIAS-PACK": "#cfc6c2", "SIG": "#ddd5d0", "POST": "#e6e0db", "TABS": "#bab0ac",
}
LEGEND = ["DIAG", "QUANT/ACT", "PACK", "EFF", "REQ", "ACC", "DEPACK", "MM", "SPIN"]


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")


def union_len(ivs):
    ivs = sorted(ivs); tot = 0; c0 = c1 = None
    for a, b in ivs:
        if c0 is None: c0, c1 = a, b
        elif a <= c1: c1 = max(c1, b)
        else: tot += c1 - c0; c0, c1 = a, b
    if c0 is not None: tot += c1 - c0
    return tot


def main():
    raw = open(sys.argv[1], "rb").read()
    out = sys.argv[2]
    magic, n = struct.unpack_from("<II", raw, 0)
    assert magic == 0x47545203, f"bad magic {magic:#x}"
    wall, base = struct.unpack_from("<QQ", raw, 8)
    evs = []
    off = 24
    for _ in range(n):
        tid, stage, t0, t1 = struct.unpack_from("<IIQQ", raw, off); off += 24
        if t1 > t0:
            evs.append((tid, stage, t0, t1))
    tids = sorted(set(e[0] for e in evs))
    tmin = min(e[2] for e in evs); tmax = max(e[3] for e in evs)
    SPAN = max(1, tmax - tmin)
    cons = max(t for t in tids if any(e[0] == t and e[1] == 3 for e in evs))
    def label(t): return "CONS" if t == cons else f"P{t}"
    order = [t for t in tids if t != cons] + [cons]

    L, R, TOP = 64, 168, 84
    W = 1180
    TH, GAP = 38, 10
    plotw = W - L - R
    rows = len(order)
    ploth = rows * (TH + GAP) - GAP
    H = TOP + ploth + 158

    def X(t): return L + (t - tmin) / SPAN * plotw
    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
    s.append('<rect width="100%" height="100%" fill="#fbfbfc"/>')
    s.append(f'<text x="24" y="34" fill="#1f2933" font-size="19" font-weight="600">'
             f'GDNSolveHVXMixHMX (HVX-feed + HMX-matmul) — pipeline trace (Perfetto style)</text>')
    s.append(f'<text x="24" y="56" fill="#52606d" font-size="13">4 HVX producers run the FULL solve '
             f'(DIAG fwd-subst + off-diag MERGE) and delegate each 64³ matmul to one PURE-HMX consumer · '
             f'one 32-head solve · FEED-BOUND (producers ~78-85% busy, HMX ~7%) · {n} events</text>')

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
        is_cons = (t == cons)
        # busy% = WALL-relative union of work intervals. PREP(5) is real HVX work (a container over its
        # nested ACT/PACK/EFF merge children) so it IS counted; HEAD(0)/MERGE(2) are pure structural
        # containers and excluded. The union de-double-counts PREP vs its nested children.
        work_iv = [(t0, t1) for (_, st_, t0, t1) in te if st_ not in (0, 2)]
        busy = union_len(work_iv)
        s.append(f'<rect x="{L}" y="{y}" width="{plotw}" height="{TH}" fill="{"#fff6f0" if is_cons else "#f4f7fa"}" '
                 f'stroke="#e1e5ea" stroke-width="1" rx="3"/>')
        s.append(f'<text x="{L-12}" y="{y+TH/2+5:.0f}" text-anchor="end" fill="#1f2933" font-size="14" '
                 f'font-weight="600">{label(t)}</text>')
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2-2:.0f}" fill="#3e4c59" font-size="12" font-weight="600">'
                 f'{busy*100//wall}% busy</text>')
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2+13:.0f}" fill="#9aa5b1" font-size="10.5">'
                 f'{"HMX mxmem (matmul)" if is_cons else "HVX full solve"}</text>')
        # draw leaves (skip containers so the nested children show through)
        for (_, stage, t0, t1) in te:
            if stage in CONTAINER:
                continue
            x0 = X(t0); w = max(0.4, X(t1) - X(t0))
            col = COLOR.get(STAGE.get(stage, ""), "#cccccc")
            s.append(f'<rect x="{x0:.2f}" y="{y+3}" width="{w:.2f}" height="{TH-6}" fill="{col}"/>')

    ly = TOP + ploth + 60
    s.append(f'<text x="24" y="{ly-14}" fill="#3e4c59" font-size="13" font-weight="600">Stages</text>')
    lx = 24
    for name in LEGEND:
        s.append(f'<rect x="{lx}" y="{ly-12}" width="16" height="16" rx="3" fill="{COLOR[name]}"/>')
        s.append(f'<text x="{lx+22}" y="{ly+1}" fill="#52606d" font-size="12.5">{esc(name)}</text>')
        lx += 40 + 7.2 * len(name)
    s.append(f'<text x="24" y="{ly+30}" fill="#7b8794" font-size="11.5">'
             f'FEED-BOUND: the 4 HVX producers run ~78-85% busy on the full HVX solve (DIAG fwd-subst Σ31% + '
             f'off-diag merge-glue Σ60%); the single HMX consumer is only ~7% busy on the 64³ matmuls '
             f'(idle ~93%) — opposite balance to pure-HMX, here the HVX producers ARE the work.</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    print(f"wrote {out}  ({n} events, {len(order)} tracks, wall {wall} cyc)")


if __name__ == "__main__":
    main()
