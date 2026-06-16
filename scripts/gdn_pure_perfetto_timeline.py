#!/usr/bin/env python3
"""Render the pure-HMX (GDNSolveHMX) pipeline trace (-DGP_TRACE events) as a Perfetto-style SVG.
Same swimlane layout / palette family as scripts/gdn_perfetto_timeline.py (the HVXMixHMX renderer),
but with the pure-HMX stage taxonomy: ACT / WT-PACK / ACC-renorm / SPIN / OUT-COPY (producers, HVX)
+ MM (consumer, HMX). HVX-prep stages use the blue family, the HMX matmul uses orange — matching the
gdn_pipeline.svg / gdn_blocksolve.svg convention (HVX blue / HMX orange).

Trace format (shared with the int16 driver):
  [magic u32=0x47545203][n u32][wall u64][base u64] then n*{tid u32, stage u32, t0 u64, t1 u64}
Pure-HMX stage ids: 3=MM(consumer) 4=ACT 5=WT-PACK(PREP) 6=ACC/renorm 10=OUT-COPY 11=SPIN 12=LOAD 13=STORE.

Usage: gdn_pure_perfetto_timeline.py <trace.raw> <out.svg>
Reproduce the trace: EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE -DGP_TRACE" build, run, pull w16p4_T.raw.
"""
import sys, struct

# pure-HMX taxonomy (matches scripts/gdn_pipe_timeline.py CH legend)
STAGE = {3: "MM", 4: "ACT", 5: "WT-PACK", 6: "ACC", 10: "OUT-COPY", 11: "SPIN", 12: "LOAD", 13: "STORE"}
# containers wrap leaves -> skip to avoid double-draw (HEAD=0, MERGE=2)
CONTAINER = {0, 2}
# HVX producer-prep = blue family; the long-pole WT-PACK gets the saturated blue; HMX matmul = orange.
COLOR = {
    "WT-PACK": "#4e79a7",   # the #1 long pole (HVX kmajor byte-pack) — saturated blue
    "ACC":     "#8c7bb4",   # solve algebra (renorm/acc), HVX — violet (2nd pole)
    "ACT":     "#59a14f",   # act-format, HVX — green, tiny
    "OUT-COPY":"#9cc3e0",   # out surface -> cv, HVX — light blue, tiny
    "SPIN":    "#e8a0a8",   # producer idle-waiting on the 1 HMX — muted rose
    "MM":      "#e08a52",   # the 64^3 matmul on HMX — orange (HMX convention)
    "LOAD":    "#bab0ac", "STORE": "#cfc6c2",
}
LEGEND = ["WT-PACK", "ACC", "ACT", "OUT-COPY", "SPIN", "MM"]

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")

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
        if stage not in CONTAINER and t1 > t0:
            evs.append((tid, stage, t0, t1))
    tids = sorted(set(e[0] for e in evs))
    tmin = min(e[2] for e in evs); tmax = max(e[3] for e in evs)
    SPAN = max(1, tmax - tmin)
    cons = max(t for t in tids if any(e[0] == t and e[1] == 3 for e in evs))
    def label(t): return "CONS" if t == cons else f"P{t}"
    order = [t for t in tids if t != cons] + [cons]

    L, R, TOP = 64, 150, 84
    W = 1180
    TH, GAP = 38, 10
    plotw = W - L - R
    rows = len(order)
    ploth = rows*(TH+GAP) - GAP
    H = TOP + ploth + 150

    def X(t): return L + (t - tmin)/SPAN*plotw
    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
    s.append('<rect width="100%" height="100%" fill="#fbfbfc"/>')
    s.append(f'<text x="24" y="34" fill="#1f2933" font-size="19" font-weight="600">'
             f'GDNSolveHMX (pure-HMX) — pipeline trace (Perfetto style)</text>')
    s.append(f'<text x="24" y="56" fill="#52606d" font-size="13">4 HVX producers feed one PURE-HMX consumer · '
             f'one 32-head solve · FEED-BOUND (producers ~81% busy, HMX ~29%) · {n} events</text>')

    for i in range(0, 11):
        gx = L + plotw*i/10
        cyc = SPAN*i/10
        s.append(f'<line x1="{gx:.1f}" y1="{TOP-6}" x2="{gx:.1f}" y2="{TOP+ploth}" stroke="#eceef0" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{TOP-12}" text-anchor="middle" fill="#9aa5b1" font-size="11">'
                 f'{cyc/1e6:.2f}M</text>')
    s.append(f'<text x="{L+plotw/2:.0f}" y="{TOP+ploth+34}" text-anchor="middle" fill="#7b8794" font-size="12">'
             f'domain cycles (trace-perturbed single rep; read STRUCTURE, not absolute wall) →</text>')

    for ri, t in enumerate(order):
        y = TOP + ri*(TH+GAP)
        te = [e for e in evs if e[0] == t]
        busy = sum(e[3]-e[2] for e in te)
        # busy% is WALL-relative (busy ÷ global wall) to match the doc / ASCII renderer 口径
        # (producers ~81%, HMX ~29%), NOT span-relative.
        is_cons = (t == cons)
        s.append(f'<rect x="{L}" y="{y}" width="{plotw}" height="{TH}" fill="{"#fff6f0" if is_cons else "#f4f7fa"}" '
                 f'stroke="#e1e5ea" stroke-width="1" rx="3"/>')
        s.append(f'<text x="{L-12}" y="{y+TH/2+5:.0f}" text-anchor="end" fill="#1f2933" font-size="14" '
                 f'font-weight="600">{label(t)}</text>')
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2-2:.0f}" fill="#3e4c59" font-size="12" font-weight="600">'
                 f'{busy*100//wall}% busy</text>')
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2+13:.0f}" fill="#9aa5b1" font-size="10.5">'
                 f'{"HMX mxmem (matmul)" if is_cons else "HVX prep + renorm"}</text>')
        for (_, stage, t0, t1) in te:
            x0 = X(t0); w = max(0.4, X(t1)-X(t0))
            col = COLOR.get(STAGE.get(stage, ""), "#cccccc")
            s.append(f'<rect x="{x0:.2f}" y="{y+3}" width="{w:.2f}" height="{TH-6}" fill="{col}"/>')

    ly = TOP + ploth + 60
    s.append(f'<text x="24" y="{ly-14}" fill="#3e4c59" font-size="13" font-weight="600">Stages</text>')
    lx = 24
    for name in LEGEND:
        s.append(f'<rect x="{lx}" y="{ly-12}" width="16" height="16" rx="3" fill="{COLOR[name]}"/>')
        s.append(f'<text x="{lx+22}" y="{ly+1}" fill="#52606d" font-size="12.5">{esc(name)}</text>')
        lx += 40 + 7.2*len(name)
    s.append(f'<text x="24" y="{ly+30}" fill="#7b8794" font-size="11.5">'
             f'FEED-BOUND: the 4 HVX producers run ~81% busy on WT-PACK + renorm/acc; the single HMX '
             f'consumer is only ~29% busy on pure 64³ matmuls (idle 65–70%) — ample matmul headroom, '
             f'the cost is operand preparation.</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    print(f"wrote {out}  ({n} events, {len(order)} tracks, wall {wall} cyc)")

if __name__ == "__main__":
    main()
