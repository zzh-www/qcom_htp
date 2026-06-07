#!/usr/bin/env python3
"""Render an ASCII per-thread timeline from a GDN pipeline TRACE dump (T.raw overwritten in trace mode).

Format: [magic u32=0x47545203][n u32][wall u64][base u64] then n*{tid u32, stage u32, t0 u64, t1 u64}.
Stages: 0=HEAD(container,skipped) 1=DIAG 2=MERGE 3=MM(consumer) 4=QUANT. Consumer tid = GDN_BR_NT(=4).
Usage: gdn_pipe_timeline.py T.raw [width]
"""
import sys, struct, collections

STAGE = {0: "HEAD", 1: "DIAG", 2: "MERGE", 3: "MM", 4: "QUANT", 5: "PREP", 6: "ACC", 7: "REQ", 8: "PACK", 9: "EFF"}
CH = {1: "D", 3: "m", 4: "q", 5: "p", 6: "a", 7: "r", 8: "K", 9: "e"}   # diag/matmul/quant/prep/acc/requant/pack/effective(-128*Sumwt)
import os
# coarse view (GDN_TL_COARSE=1): PREP(5) is a leaf, skip the fine QUANT/PACK/EFF (4/8/9) — for the int16
# driver which only emits coarse DIAG/PREP/MM/ACC/REQ (its getters don't push fine leaves).
if os.environ.get("GDN_TL_COARSE"):
    CONTAINER = {0, 2, 4, 8, 9}
else:
    CONTAINER = {0, 2, 5}   # fine view: HEAD/MERGE/PREP containers (QUANT/PACK leaves below PREP)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "T.raw"
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 130
    b = open(path, "rb").read()
    magic, n = struct.unpack_from("<II", b, 0)
    assert magic == 0x47545203, f"bad magic {magic:#x}"
    wall, base = struct.unpack_from("<QQ", b, 8)
    evs = []
    off = 24
    for _ in range(n):
        tid, stage, t0, t1 = struct.unpack_from("<IIQQ", b, off); off += 24
        evs.append((tid, stage, t0, t1))
    tids = sorted(set(e[0] for e in evs))
    print(f"events={n}  wall={wall} cyc  ({W} cols, 1 col = {wall/W:.0f} cyc)\n")
    # per-tid busy (exclude HEAD container) and timeline row
    for tid in tids:
        row = [" "] * W
        busy = collections.Counter()
        span_lo, span_hi = wall, 0
        for (t, stage, t0, t1) in evs:
            if t != tid:
                continue
            span_lo = min(span_lo, t0); span_hi = max(span_hi, t1)
            if stage in CONTAINER:
                continue
            busy[stage] += t1 - t0
            c0 = int(t0 * W / wall); c1 = max(c0 + 1, int(t1 * W / wall))
            for c in range(c0, min(c1, W)):
                # priority: kernel > merge > diag > quant for visibility
                cur = row[c]
                nch = CH.get(stage, "?")
                row[c] = nch if cur == " " else cur
        label = "CONS" if tid == max(tids) and 3 in [e[1] for e in evs if e[0] == tid] else f"P{tid}"
        tot_busy = sum(busy.values())
        span = (span_hi - span_lo) or 1
        bd = " ".join(f"{STAGE[s]}={busy[s]*100//span}%" for s in sorted(busy))
        print(f"{label:5s}|{''.join(row)}| busy={tot_busy*100//wall}%wall  [{bd}]")
    print("\nlegend: D=diag  q=QUANT(fold+quant+eff)  K=PACK(crouton/kmajor)  m=matmul  a=acc  r=requant+widen")
    # aggregate
    agg = collections.Counter()
    for (t, stage, t0, t1) in evs:
        if stage in CONTAINER: continue
        agg[stage] += t1 - t0
    print("aggregate stage cyc (summed over threads):", {STAGE[s]: agg[s] for s in sorted(agg)})

if __name__ == "__main__":
    main()
