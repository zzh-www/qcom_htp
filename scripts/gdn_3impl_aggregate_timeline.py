#!/usr/bin/env python3
"""Aggregate per-thread STRUCTURE timeline of the THREE GDN-solve implementations into ONE Perfetto-style
SVG, laid out as "p{i}({impl})" — each hardware thread (p0..p3 producers + HMX consumer) gets 3 stacked
rows, one per implementation, so the SAME producer's stage construction is directly comparable across the
three routes by colour:

    p0(0)  SHIP u8i8       p0
    p0(1)  W16_N8 ARES     p0
    p0(2)  pure-HMX        p0
    p1(0) ... (p1/p2/p3/consumer likewise)

Same-CLASS stages get the SAME colour ACROSS implementations (PACK blue / MM-HMX orange / DIAG purple /
ACT green / ACC-renorm violet / REQ light-violet / EFF pale-blue / DEPACK·OUT-COPY light-blue / SPIN rose
/ LOAD·STORE·POST·TABS·BIAS neutral) — so the colour composition of the three rows under one thread shows
the per-route stage-mix difference at a glance.

Trace format (shared, magic 0x47545203):
  [magic u32][n u32][wall u64][base u64] then n*{tid u32, stage u32, t0 u64, t1 u64}
Per-impl thread identity is identical: tid 0..3 = the 4 HVX producers (p0..p3), tid 4 = the main-thread
HMX consumer (the only thread that owns the 64^3 matmul).

口径 / HONESTY (rendered into the SVG too):
  * Each impl uses its OWN stage taxonomy (SHIP/ARES = HVXMix forward-subst + off-diag merge route;
    pure-HMX = Taylor(3) no-DIAG route) but same-CLASS stages share a colour; the mapping is in the
    legend. Stages that exist in only one route (DIAG/REQ/EFF/DEPACK/TABS in HVXMix; OUT-COPY/ACC in
    pure-HMX) are marked "(route X only)" in the legend.
  * Single-rep trace => trace-perturbed. READ STRUCTURE (stage mix + relative length), NOT the absolute
    in-trace wall. The authoritative production walls are annotated per-impl: SHIP 1.925M / ARES 2.55M /
    pure-HMX 1.258M (32-head total VTCM-only).
  * pure-HMX is a DIFFERENT harness/producer structure from HVXMix; the shared absolute cycle axis is a
    VISUAL aid only — do NOT hard-compare per-thread occupancy ACROSS harnesses, only the stage mix.

Usage:
  gdn_3impl_aggregate_timeline.py <ship.raw> <ares.raw> <purehmx.raw> <out.svg>
Capture the blobs:  scripts/gdn_capture_3impl_traces.sh   (writes /tmp/{ship,w16n8,purehmx}_trace.raw fresh).
  NOTE: ARES (W16_N8) needs the EXTENDED A input (natural-A + appended cv-block tail, 5.76MB).  Feeding it
  the plain 4MB natural A makes the resident bulk-load read past the FastRPC buffer -> rc=0x8000040d (this
  was the real cause of the historical "ARES+TRACE won't run", NOT a VTCM-size or trace-buffer issue).  The
  fresh ARES blob is produced by: build+run the DUMP variant (-DGDN_BR_W16_N8_ACVRES_DUMP) to emit the
  cv-block A, assemble A_ares.raw, then run the ARES+TRACE build with A_ares.raw (see scripts/w16n8_ares_check.sh
  STEP 1 for the assembly).  Do NOT reuse a stale /tmp/aresT_TR.raw blob — it predates the unified MM口径.

口径 alignment (trace-only, production codegen unaffected — PRECISE scope, do NOT over-claim "every edit"):
  This trace-口径 task's edits are ALL either inside `#if defined(GDN_BR_TRACE)` guards or pure comments;
  the production-default solve = pure-HMX (gdn_pure_solve.cpp) was NOT touched (git diff empty).  Preprocessing
  the SHIP TU (gdnbm_imp.cpp, -DGDNBM_HMX_PIPE -DGDN_BR_STATIC_FULL, GDN_BR_TRACE off) shows ZERO residue from
  this task (0 gdn_tr_push / 0 GDN_TR_MM; 0 differing SHIP-TU lines reference GdnSolveBR16/BROp.cpp).  The
  working tree ALSO carries pre-session NON-trace-guarded W16_N8/ACVRES/ARES/BP2 scaffolding (out of this
  task's scope; production-unaffected proven in its own rounds).  Evidence:
  Agent/current/trace_kqie_production_unaffected.txt.
  the HVXMix producers (SHIP/ARES) used to label the WHOLE synchronous merge handshake (fill slot, spin for
  the consumer's mxmem, depack) as MM(3) on their OWN tid — double-counting the consumer's HMX window and
  hiding the real SPIN.  That is fixed: producers now emit SPIN(11)+DEPACK(10) (+ACT/WT-PACK pre-dispatch)
  like pure-HMX, and MM(3) is emitted ONLY by the main-thread consumer (tid 4).  => orange (MM) appears on
  the CONS row only, identically for all three routes.
"""
import sys, struct

# ---------------------------------------------------------------------------------------------------
# Per-impl raw stage-id -> common semantic CLASS name.  Three taxonomies, one colour vocabulary.
#   HVXMix (SHIP & ARES) ids: 1=DIAG 3=MM 4=QUANT/ACT 5=PREP(container) 6=ACC 7=REQ 8=PACK 9=EFF
#                             10=DEPACK 11=SPIN 12=LOAD 13=STORE 14=POST 15=TABS  (0/2/5 containers)
#   pure-HMX (GP) ids:        3=MM 4=ACT 5=WT-PACK 6=ACC 10=OUT-COPY 11=SPIN 12=LOAD 13=STORE  (0/2 cont.)
# ---------------------------------------------------------------------------------------------------
# 口径 SYMMETRY (2026-06-17): stage 5 is PREP/operand-prep in BOTH taxonomies and MUST be drawn
# as a LEAF in BOTH — previously HVXMix put 5 in CONTAINER (skip-and-draw-nested) while GP drew 5
# as a PACK leaf. That asymmetry掏空ed the HVXMix producer rows ~20-30pp (nested PREP leaves were
# sparse), visually FAKING a low producer occupancy on SHIP/ARES vs pure-HMX. Both now draw stage 5
# as a PACK leaf so the three routes' producer occupancy is rendered on the SAME口径. (Per-thread
# utilization is NOT to be read off this single-rep trace anyway — see SVG caveat + perf 表.)
HVXMIX_CLASS = {1: "DIAG", 3: "MM", 4: "ACT", 5: "PACK", 6: "ACC", 7: "REQ", 8: "PACK", 9: "EFF",
                10: "DEPACK", 11: "SPIN", 12: "LOAD", 13: "STORE", 14: "POST", 15: "TABS"}
HVXMIX_CONTAINER = {0, 2}        # HEAD / MERGE container -> skip; PREP(5) is now a PACK LEAF (symmetric w/ GP)
GP_CLASS = {3: "MM", 4: "ACT", 5: "PACK", 6: "ACC", 10: "DEPACK", 11: "SPIN", 12: "LOAD", 13: "STORE"}
GP_CONTAINER = {0, 2}            # WT-PACK(5) in GP is a LEAF here (mapped to PACK), not a container

# One colour per semantic class, shared across impls.  Blue family = HVX operand-prep; orange = HMX MM;
# purple = DIAG fwd-subst; violet = ACC/REQ renorm; rose = SPIN; neutral = bookkeeping.
# 口径 (unified across all 3 impls, post trace-alignment): the single HMX unit is owned by the MAIN-thread
# CONSUMER, so MM(3) is emitted ONLY on the consumer tid (CONS row).  A PRODUCER never runs mxmem — it
# packs operands (ACT/WT-PACK/PACK), hands the job to the consumer, idle-SPINs (rose) for the result, then
# DEPACKs.  So producer rows carry NO orange in any of the three routes; orange lives only on CONS.
COLOR = {
    "PACK":   "#4e79a7",   # kmajor / crouton byte-pack (HVXMix PACK · pure-HMX WT-PACK) — saturated blue
    "EFF":    "#8fc3e8",   # effective (-128*Σwt), HVXMix-only — light blue
    "DEPACK": "#9cc3e0",   # surface->int16 depack (HVXMix DEPACK · pure-HMX OUT-COPY) — pale blue
    "DIAG":   "#7b6fb0",   # diagonal forward-subst, HVXMix-only — purple
    "ACT":    "#59a14f",   # act / quant format — green
    "ACC":    "#b3a2d4",   # int32 acc / renorm (HVXMix ACC · pure-HMX ACC) — light violet
    "REQ":    "#8c7bb4",   # widen + requant (merge-final), HVXMix-only — violet
    "MM":     "#e08a52",   # the 64^3 matmul on HMX (consumer) — orange (HMX convention)
    "SPIN":   "#e8a0a8",   # producer idle-waiting on the 1 HMX — muted rose
    "LOAD":   "#bab0ac", "STORE": "#cfc6c2", "POST": "#e6e0db", "TABS": "#ddd5d0",
}
# legend order + per-class "which route" note
LEGEND = [
    ("PACK",   "kmajor/crouton pack (all)"),
    ("MM",     "64³ matmul · HMX — CONS row ONLY (all 3 impls)"),
    ("ACT",    "act/quant (all)"),
    ("ACC",    "int32 acc/renorm (SHIP·pure-HMX)"),
    ("DIAG",   "fwd-subst diag (SHIP·ARES only)"),
    ("REQ",    "widen+requant (SHIP·ARES only)"),
    ("EFF",    "effective -128Σwt (SHIP only)"),
    ("DEPACK", "depack/out-copy (SHIP·pure-HMX)"),
    ("SPIN",   "idle wait on HMX (all)"),
    ("LOAD",   "load/store/post/tabs (bookkeeping)"),
]

IMPL = [("SHIP u8i8", "1.925M"), ("W16_N8 ARES", "2.55M"), ("pure-HMX", "1.258M")]


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")


def _union(intervals):
    """Total length covered by the union of [a,b) intervals (de-duplicates overlap, e.g. a PREP
    container drawn as a leaf together with its nested child leaves)."""
    iv = sorted((a, b) for a, b in intervals if b > a)
    tot = 0
    ce = None
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
    cont = GP_CONTAINER if is_gp else HVXMIX_CONTAINER
    cls = GP_CLASS if is_gp else HVXMIX_CLASS
    evs = []
    off = 24
    for _ in range(n):
        tid, stage, t0, t1 = struct.unpack_from("<IIQQ", raw, off); off += 24
        if t1 > t0 and stage not in cont:
            evs.append((tid, cls.get(stage, "LOAD"), t0, t1))
    tmin = min(e[2] for e in evs)
    span = max(1, max(e[3] for e in evs) - tmin)
    return {"evs": evs, "tmin": tmin, "span": span, "wall": wall}


def main():
    ship = load(sys.argv[1], False)
    ares = load(sys.argv[2], False)
    pure = load(sys.argv[3], True)
    out = sys.argv[4]
    impls = [ship, ares, pure]
    # one shared absolute-cycle axis spanning the longest trace (each starts at 0)
    SPAN = max(d["span"] for d in impls)

    # threads: tid 0..3 = p0..p3 producers, tid 4 = HMX consumer (CONS)
    TIDS = [0, 1, 2, 3, 4]
    def tlabel(t): return "CONS" if t == 4 else f"p{t}"

    L, R, TOP = 124, 250, 116   # TOP raised to clear the prominent single-rep caveat band (y=62..88)
    W = 1420
    RH, RGAP, GGAP = 17, 2, 12     # row height, gap inside a thread-group, gap between thread-groups
    plotw = W - L - R
    # vertical layout: 5 thread-groups x 3 impl-rows
    grp_h = 3 * RH + 2 * RGAP
    ploth = len(TIDS) * grp_h + (len(TIDS) - 1) * GGAP
    H = TOP + ploth + 196

    def X(t0):  # absolute cyc -> x (each impl already 0-based via its own tmin)
        return L + t0 / SPAN * plotw

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
    s.append('<rect width="100%" height="100%" fill="#fbfbfc"/>')
    s.append(f'<text x="24" y="34" fill="#1f2933" font-size="19" font-weight="600">'
             f'GDN solve — 3 implementations stacked per thread (Perfetto style aggregate)</text>')
    s.append(f'<text x="24" y="56" fill="#52606d" font-size="12.5">Each thread (p0–p3 HVX producers + CONS '
             f'HMX consumer) shows 3 rows: (0) SHIP u8i8 · (1) W16_N8 ARES · (2) pure-HMX — same-class '
             f'stages share a colour, so the per-route stage MIX is comparable down each thread group.</text>')
    # PROMINENT caveat band — single-rep trace, read STRUCTURE only, utilization lives in the perf 表
    s.append(f'<rect x="24" y="62" width="{W-48}" height="26" rx="4" fill="#fdecde" stroke="#e0a878" stroke-width="1"/>')
    s.append(f'<text x="34" y="79" fill="#a8551f" font-size="11.5" font-weight="600">'
             f'⚠ SINGLE-REP TRACE — read STAGE STRUCTURE ONLY (MM unified to consumer-only). '
             f'Do NOT read per-thread utilization/occupancy off this figure — it is trace-perturbed. '
             f'Authoritative steady-state util/occupancy = Agent/current/perf_3impl_cron82kqie.txt '
             f'(SHIP 1.892M / ARES 2.506M / pure-HMX 1.259M; HVX-util 96%/94%/83%).</text>')

    # gridlines on the shared absolute-cycle axis
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
        # thread-group backing + label
        s.append(f'<rect x="{L}" y="{gy-1}" width="{plotw}" height="{grp_h+2}" '
                 f'fill="{"#fff6f0" if is_cons else "#f4f7fa"}" stroke="#dfe4ea" stroke-width="1" rx="3"/>')
        s.append(f'<text x="{L-46}" y="{gy+grp_h/2+5:.0f}" text-anchor="end" fill="#1f2933" font-size="15" '
                 f'font-weight="700">{tlabel(t)}</text>')
        s.append(f'<text x="{L-46}" y="{gy+grp_h/2+20:.0f}" text-anchor="end" fill="#9aa5b1" font-size="9.5">'
                 f'{"HMX" if is_cons else "HVX"}</text>')
        for ii, d in enumerate(impls):
            ry = gy + ii * (RH + RGAP)
            te = [e for e in d["evs"] if e[0] == t]
            # busy = UNION of intervals (NOT naive Σ): PREP(stage5, now a PACK leaf) wraps nested child
            # leaves on the SAME tid, so a naive sum would double-count parent+child and inflate the %.
            # Union gives the honest wall-fraction this thread is occupied — symmetric across all 3 routes.
            busy = _union([(e[2], e[3]) for e in te])
            wall = d["wall"]
            # per-row index "(ii)" sits in the LEFT GUTTER (clear of the timeline blocks)
            s.append(f'<text x="{L-6}" y="{ry+RH-4:.0f}" text-anchor="end" fill="#52606d" font-size="10" '
                     f'font-weight="600">{tlabel(t)}({ii})</text>')
            s.append(f'<text x="{L+plotw+10}" y="{ry+RH-4:.0f}" fill="#3e4c59" font-size="10">'
                     f'{IMPL[ii][0]} · {busy*100//wall}%</text>')
            # draw widest intervals FIRST so nested child leaves (DIAG/ACT/REQ...) paint ON TOP of the
            # PREP(stage5) container leaf that wraps them — otherwise PREP would hide its own children.
            for (_, cname, t0, t1) in sorted(te, key=lambda e: -(e[3] - e[2])):
                x0 = X(t0 - d["tmin"]); w = max(0.4, X(t1 - d["tmin"]) - X(t0 - d["tmin"]))
                col = COLOR.get(cname, "#cccccc")
                s.append(f'<rect x="{x0:.2f}" y="{ry+1}" width="{w:.2f}" height="{RH-2}" fill="{col}"/>')
            # impl end-of-trace tick (relative length of this route on the shared axis)
            ex = X(d["span"])
            s.append(f'<line x1="{ex:.1f}" y1="{ry+1}" x2="{ex:.1f}" y2="{ry+RH-1}" stroke="#52606d" '
                     f'stroke-width="0.8" stroke-dasharray="2,1" opacity="0.55"/>')

    # legend
    ly = TOP + ploth + 64
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
    ly += 26
    s.append(f'<text x="24" y="{ly}" fill="#7b8794" font-size="11">Legend index: (0)=SHIP u8i8 1.925M · '
             f'(1)=W16_N8 ARES 2.55M · (2)=pure-HMX 1.258M (32-head total VTCM-only production walls).</text>')
    ly += 18
    s.append(f'<text x="24" y="{ly}" fill="#9aa5b1" font-size="10.5">HONESTY: each route uses its own '
             f'taxonomy (SHIP/ARES = HVX fwd-subst+off-diag merge with DIAG/REQ/EFF/DEPACK; pure-HMX = '
             f'Taylor(3), NO DIAG); only same-class colours are comparable. pure-HMX is a different '
             f'harness — the shared cyc axis is a visual aid, cross-harness per-thread occupancy is NOT '
             f'hard-comparable, read the stage MIX only.</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    print(f"wrote {out}  (SHIP {ship['wall']} / ARES {ares['wall']} / pure-HMX {pure['wall']} in-trace cyc, "
          f"shared axis {SPAN/1e6:.2f}M)")


if __name__ == "__main__":
    main()
