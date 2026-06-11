#!/usr/bin/env python3
"""Render the GDN pipeline trace (gdn_tr_push events) as a Perfetto-style SVG:
one swimlane per thread (P0..P3 producers + CONS consumer), colored slices per stage,
a time axis in domain cycles, and a legend.

Trace format (written by -DGDN_BR_TRACE into the T output buffer):
  [magic u32=0x47545203][n u32][wall u64][base u64] then n*{tid u32, stage u32, t0 u64, t1 u64}

Usage: gdn_perfetto_timeline.py <trace.raw> <out.svg>
Reproduce the trace: build with -DGDN_BR_TRACE, run, pull the T_tr.raw output.
"""
import sys, struct

STAGE = {1:"DIAG",2:"MERGE",3:"MM",4:"QUANT",5:"PREP",6:"ACC",7:"REQ",8:"PACK",
         9:"EFF",10:"DEPACK",11:"SPIN",12:"BIAS",13:"SIG",14:"POST",15:"TABS"}
CONTAINER = {0, 5}   # HEAD container; PREP wraps the fine QUANT/PACK leaves -> skip to avoid double-draw
# Perfetto-ish categorical palette: compute = saturated, glue/wait = muted.
COLOR = {
    "DIAG":"#4e79a7", "MM":"#e15759", "QUANT":"#59a14f", "PACK":"#f0a23b",
    "REQ":"#76b7b2", "EFF":"#8c7bb4", "ACC":"#b07aa1", "DEPACK":"#9cc3e0",
    "SPIN":"#e8a0a8", "BIAS":"#bab0ac", "SIG":"#cfc6c2", "POST":"#d9d3cf",
    "TABS":"#d9d3cf", "MERGE":"#bab0ac",
}
# stages shown in the legend, in a sensible order
LEGEND = ["DIAG","QUANT","PACK","EFF","REQ","ACC","DEPACK","MM","SPIN","SIG"]

def esc(s): return s.replace("&","&amp;").replace("<","&lt;")

def main():
    raw = open(sys.argv[1], "rb").read()
    out = sys.argv[2]
    magic, n = struct.unpack_from("<II", raw, 0)
    wall, base = struct.unpack_from("<QQ", raw, 8)
    evs = []
    off = 24   # header = magic(4)+n(4)+wall(8)+base(8)
    for _ in range(n):
        tid, stage, t0, t1 = struct.unpack_from("<IIQQ", raw, off); off += 24
        if stage not in CONTAINER and t1 > t0:
            evs.append((tid, stage, t0, t1))
    tids = sorted(set(e[0] for e in evs))
    tmin = min(e[2] for e in evs); tmax = max(e[3] for e in evs)
    SPAN = max(1, tmax - tmin)
    # consumer = the tid that emits MM(3)
    cons = max(t for t in tids if any(e[0]==t and e[1]==3 for e in evs))
    def label(t): return "CONS" if t == cons else f"P{t}"
    order = [t for t in tids if t != cons] + [cons]

    # layout
    L, R, TOP = 64, 150, 84
    W = 1180                       # plot width
    TH, GAP = 38, 10               # track height / gap
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
             f'GDNSolveHVXMixHMX — pipeline trace (Perfetto style)</text>')
    s.append(f'<text x="24" y="56" fill="#52606d" font-size="13">4 HVX producers + 1 PURE-HMX consumer · '
             f'one 32-head solve · wall {wall/1e6:.2f}M domain cyc · {n} events</text>')

    # time grid + axis (cycles)
    for i in range(0, 11):
        gx = L + plotw*i/10
        cyc = SPAN*i/10
        s.append(f'<line x1="{gx:.1f}" y1="{TOP-6}" x2="{gx:.1f}" y2="{TOP+ploth}" stroke="#eceef0" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{TOP-12}" text-anchor="middle" fill="#9aa5b1" font-size="11">'
                 f'{cyc/1e6:.2f}M</text>')
    s.append(f'<text x="{L+plotw/2:.0f}" y="{TOP+ploth+34}" text-anchor="middle" fill="#7b8794" font-size="12">'
             f'domain cycles →</text>')

    # tracks
    for ri, t in enumerate(order):
        y = TOP + ri*(TH+GAP)
        te = [e for e in evs if e[0]==t]
        busy = sum(e[3]-e[2] for e in te)
        tspan = max(1, max(e[3] for e in te) - min(e[2] for e in te))
        is_cons = (t == cons)
        # lane bg
        s.append(f'<rect x="{L}" y="{y}" width="{plotw}" height="{TH}" fill="{"#fff6f0" if is_cons else "#f4f7fa"}" '
                 f'stroke="#e1e5ea" stroke-width="1" rx="3"/>')
        s.append(f'<text x="{L-12}" y="{y+TH/2+5:.0f}" text-anchor="end" fill="#1f2933" font-size="14" '
                 f'font-weight="600">{label(t)}</text>')
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2-2:.0f}" fill="#3e4c59" font-size="12" font-weight="600">'
                 f'{busy*100//tspan}% busy</text>')
        s.append(f'<text x="{L+plotw+10}" y="{y+TH/2+13:.0f}" fill="#9aa5b1" font-size="10.5">'
                 f'{"HMX mxmem" if is_cons else "HVX prep+diag"}</text>')
        # slices
        for (_, stage, t0, t1) in te:
            x0 = X(t0); w = max(0.4, X(t1)-X(t0))
            col = COLOR.get(STAGE.get(stage,""), "#cccccc")
            s.append(f'<rect x="{x0:.2f}" y="{y+3}" width="{w:.2f}" height="{TH-6}" fill="{col}"/>')

    # legend
    ly = TOP + ploth + 60
    s.append(f'<text x="24" y="{ly-14}" fill="#3e4c59" font-size="13" font-weight="600">Stages</text>')
    lx = 24
    for name in LEGEND:
        s.append(f'<rect x="{lx}" y="{ly-12}" width="16" height="16" rx="3" fill="{COLOR[name]}"/>')
        s.append(f'<text x="{lx+22}" y="{ly+1}" fill="#52606d" font-size="12.5">{name}</text>')
        lx += 40 + 7.2*len(name)
    s.append(f'<text x="24" y="{ly+30}" fill="#7b8794" font-size="11.5">'
             f'Producer-bound (~75%); the HMX consumer is ~5% busy — matmul headroom, cost is in operand prep.</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    print(f"wrote {out}  ({n} events, {len(order)} tracks, wall {wall} cyc)")

if __name__ == "__main__":
    main()
