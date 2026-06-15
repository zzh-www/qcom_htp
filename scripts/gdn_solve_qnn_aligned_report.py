#!/usr/bin/env python3
"""Unify our pure-HMX solve's perf breakdown with QNN optrace — SAME 口径, SAME categories.

WHY (the bug this kills): QNN optrace reports work as named ops, each tagged with a UNIT
(HMX/HVX/DMA) and the fields {cycles, num_dominant_path_cycles, cycles_per_packet,
cycles_used}. Our solve's FARF/stats used an ad-hoc taxonomy (actcopy/kmajor/scatter/
renorm/spin/outcopy) that does NOT map to QNN's, so numbers got cross-compared
(e.g. native ConvLayer "263" MAC-only warm sub-op vs our "1577" full per-conv wall). This
script forces BOTH sides into the SAME three categories and prints them side-by-side,
so any comparison is automatically same-category + same-field.

THE THREE CATEGORIES (skill htp-cycle-metric):
  真算 (MAC, HMX unit)      = the only irreducible work: ConvLayer_s1.opt  <-> our consumer w16a16_mm_run
  装料 (prep, HVX unit)     = feed: weight pack / act format / bias / reshape  <-> our wt-pack/act-format/renorm
  卸料 + 输入 (DMA/IO)      = OutputSlice / InputSlice                       <-> our out-copy / head-load(store)
  (SPIN / idle = waste, NOT a category — reported separately as a gap)

THE FIELDS, per QNN op (read the SAME field on both sides at the SAME shape+scenario):
  cycles        = per-op-type total `cycles` (HVX/DMA) ; for HMX = `cycles_used` (occupancy of ONE op)
  num_dominant  = `num_dominant_path_cycles` (critical chain, ideal-overlap floor)
  cyc/pkt       = `cycles_per_packet` (stall indicator)
  packets       = cycles / cyc-per-pkt  (or PMU COMMITTED_PKT on our side, -DGP_PKTPROBE)
  THROUGHPUT    = start_cycle retire interval / graph-wall÷N — NEVER cycles_used/N (OVERSTATE, trap#6)

Usage:
  # native side only (decode the two reference optraces into the canonical category table):
  scripts/gdn_solve_qnn_aligned_report.py --native-only

  # full side-by-side: parse a saved device run log (host stdout) for our numbers:
  scripts/gdn_solve_qnn_aligned_report.py --our-log /tmp/our_run.txt

  # or run our solve on device first, then report (needs scripts/dssh.sh + a deployed gdnbm):
  scripts/gdn_solve_qnn_aligned_report.py --run --threads 4 --heads 32
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NATIVE = {
    "batch":  REPO / "example/gdn_native/solve_op/standalone/mm_64/out/optrace/chrometrace_qnn_htp_analysis_summary.json",
    "single": REPO / "example/gdn_native/solve_op/standalone/mm_1x1x64x64/out/optrace/chrometrace_qnn_htp_analysis_summary.json",
}

# QNN op-type -> (category, unit). The authoritative grouping (device-verified unit tags, cron#79).
#   真算 = the MAC; 装料 = HVX feed; 卸料/输入 = DMA I/O.
NATIVE_OP_CATEGORY = {
    "q::ConvLayer_s1.opt":                                   ("真算-MAC", "HMX"),
    "q::ConvLayer.opt.convert_weights_to_signed.packed.tcm": ("装料-wt",  "HVX"),
    "q::Cast":                                               ("装料-wt",  "HVX"),
    "q::ForceFormat_Crouton":                                ("装料-act", "HVX"),
    "q::Reshape":                                            ("装料-act", "HVX"),
    "q::ConvLayer.opt.bias_weight_update":                   ("装料-bias","HVX"),
    "q::ConvLayer.opt.bias_scale_shuff":                     ("装料-bias","HVX"),
    "q::Slice_contig.tcm":                                   ("装料-act", "HVX"),
    "q::*InputSlice":                                        ("输入-IO",  "DMA"),
    "q::*OutputSlice":                                       ("卸料-IO",  "DMA"),
    "DmaCheckpointSet":                                      ("其它-sync","DMA"),
    "SyncOp":                                                ("其它-sync","DMA"),
}
CAT_ORDER = ["真算-MAC", "装料-wt", "装料-act", "装料-bias", "输入-IO", "卸料-IO", "其它-sync"]


def native_table(path: Path) -> dict:
    """Return per-QNN-op rows + per-category rollup, all in QNN fields, for ONE optrace.

    Per-conv = total / instances (the batch has 32 convs; the single has 1). Reports both
    `cycles` (occupancy) and `num_dominant_path` (critical chain) so the cross-field trap
    (263 vs 1388 etc.) is impossible to fall into."""
    d = json.loads(path.read_text())["data"]
    ot = d["htp_op_types"]["data"]
    # graph wall + per-unit cycles_used from htp_overall_summary
    osum = d["htp_overall_summary"]["data"][0]
    res = {r["type"] + str(r["tid"]): r for r in osum["htp_resources"]["data"]}
    hmx = next((r for r in osum["htp_resources"]["data"] if r["type"] == "HMX"), None)
    n_conv = next((r["instances"] for r in ot if r["op"] == "q::ConvLayer_s1.opt"), 1) or 1
    # per-conv cyc/pkt for ConvLayer comes from the instance list (cycles_per_packet)
    oi = d["htp_op_instances"]["data"]
    conv_inst = [r for r in oi if r.get("hmx") and "ConvLayer_s1" in r.get("htp_op", "")]
    conv_cpp = sorted(r["cycles_per_packet"] for r in conv_inst)[len(conv_inst)//2] if conv_inst else None
    conv_cyc_med = sorted(r["cycles"] for r in conv_inst)[len(conv_inst)//2] if conv_inst else None

    rows, cat = [], {c: {"cyc": 0, "dp": 0} for c in CAT_ORDER}
    for r in ot:
        op = r["op"]
        category, unit = NATIVE_OP_CATEGORY.get(op, ("其它-sync", "?"))
        cyc, dp, inst = r["cycles"], r["num_dominant_path_cycles_htp_0"], r["instances"]
        cpp = (cyc / dp) if dp else None  # type-level proxy; instance cyc/pkt is exact for ConvLayer
        rows.append({"op": op, "cat": category, "unit": unit, "cyc": cyc, "dp": dp,
                     "inst": inst, "per": cyc // inst if inst else 0,
                     "per_dp": dp // inst if inst else 0})
        cat[category]["cyc"] += cyc
        cat[category]["dp"] += dp
    graph_wall = osum["timeline_cycles"]
    return {"path": str(path), "rows": rows, "cat": cat, "n_conv": n_conv,
            "graph_wall": graph_wall, "hmx_cycles_used": hmx["cycles_used"] if hmx else 0,
            "conv_per_conv_cyc": conv_cyc_med, "conv_cyc_per_pkt": conv_cpp,
            "dompath_total": sum(x["dp_cycles"] for x in d["dominant_path_htp_0"]["data"])}


def print_native(tag: str, t: dict) -> None:
    n = t["n_conv"]
    print(f"\n===== NATIVE {tag}  ({n} conv, graph-wall={t['graph_wall']})  {Path(t['path']).parts[-5]} =====")
    print(f"  {'QNN op':52s} {'cat':10s} {'unit':4s} {'Σcyc':>8} {'per-conv':>9} {'per-dp':>8} {'inst':>5}")
    for r in sorted(t["rows"], key=lambda x: (CAT_ORDER.index(x["cat"]), -x["cyc"])):
        print(f"  {r['op'][:52]:52s} {r['cat']:10s} {r['unit']:4s} {r['cyc']:8d} "
              f"{r['per']:9d} {r['per_dp']:8d} {r['inst']:5d}")
    print(f"  --- per-category Σcyc / per-conv (÷{n}) ---")
    for c in CAT_ORDER:
        cc = t["cat"][c]
        if cc["cyc"]:
            print(f"    {c:10s}  Σcyc={cc['cyc']:8d}  per-conv={cc['cyc']//n:7d}  Σdp={cc['dp']:8d}")
    print(f"  真算-MAC ConvLayer_s1.opt: per-conv cycles(occupancy)={t['conv_per_conv_cyc']} "
          f"cyc/pkt={t['conv_cyc_per_pkt']:.2f}  (THIS is the apples-to-apples MAC number)")


# ---- our side: parse the host stdout (gdnbm prints the QNN-aligned line, see gdnbm_test.c) ----
def parse_our(log: str) -> dict | None:
    """Pull our solve's per-category cycles from the host stdout. Prefers the explicit raw stats[]
    lines (most precise), and also reads the QNN-aligned labeled lines (cron#79) + the PROD-Σ FARF
    line (device-side, has actcopy/outcopy) so all categories are filled however the run was captured."""
    o = {}
    m = re.search(r"raw stats\[0\.\.11\]:\s*([\-\d ]+)", log)
    if m:
        s = [int(x) for x in m.group(1).split()]
        o["wall"], o["P"], o["H"], o["cons"], o["spin"] = s[0], s[1], s[2], s[3], s[4]
        o["mm64"], o["wt_kmajor"], o["packchk"], o["scatter"], o["other"], o["lmax"] = s[5], s[7], s[8], s[9], s[10], s[11]
    m2 = re.search(r"raw stats\[12\.\.19\]:\s*([\-\d ]+)", log)
    if m2:
        s2 = [int(x) for x in m2.group(1).split()]
        # [12]sc_mc [13]sc_pm [14]sc_ms [15]bulk_ld [16]bulk_st [17]wt_vec [18]wt_bia [19]actcopy
        o["bulk_ld"], o["bulk_st"], o["wt_vec"], o["wt_bia"] = s2[3], s2[4], s2[5], s2[6]
        if len(s2) > 7:
            o["actcopy"] = s2[7]
    # the QNN-aligned labeled host lines (cron#79) carry the categories directly:
    for pat, key in [(r"\[装料-act\][^Σ]*Σcyc=(\d+)", "actcopy"),
                     (r"\[卸料-IO\][^Σ]*Σcyc=(\d+)", "outcopy"),
                     (r"\[装料-wt\][^Σ]*Σcyc=(\d+)", "wt_vec"),
                     (r"\[装料-bias\][^Σ]*Σcyc=(\d+)", "wt_bia"),
                     (r"\[装料-alg\][^Σ]*Σcyc=(\d+)", "other")]:
        mm = re.search(pat, log)
        if mm:
            o[key] = int(mm.group(1))
    # PROD-Σ FARF line (device-side): the authoritative actcopy/outcopy if present
    m3 = re.search(r"actcopy=(\d+).*?outcopy=(\d+)", log)
    if m3:
        o["actcopy"], o["outcopy"] = int(m3.group(1)), int(m3.group(2))
    # consumer-MAC cyc/pkt (production NTSWEEP nt8 + optional PKTPROBE)
    m4 = re.search(r"NTSWEEP nt8=(\d+)", log)
    if m4:
        o["mac_cyc_per_call"] = int(m4.group(1))
    mp = re.search(r"PKTPROBE packets/call: nt8=(\d+)", log)
    if mp:
        o["mac_packets"] = int(mp.group(1))
    return o or None


def print_ours(o: dict, n_conv: int) -> None:
    print(f"\n===== OURS (pure-HMX solve, P={o.get('P','?')} H={o.get('H','?')}, "
          f"VTCM-only graph-wall={o.get('wall','?')}, {n_conv} conv) =====")
    print("  Our segment            ->  QNN op (same category)              | category   | unit | Σcyc      | per-conv")
    P = max(o.get("P", 1), 1)
    # producer Σ are summed over P threads => divide by nothing for Σ-work, but per-conv = Σ/n_conv
    def line(seg, qnn, cat, unit, cyc):
        pc = cyc // n_conv if n_conv else 0
        print(f"  {seg:22s} ->  {qnn:34s} | {cat:10s} | {unit:4s} | {cyc:9d} | {pc:8d}")
    # 真算 (HMX) — the consumer MAC; cycles_used = g_cbusy (occupancy Σ over all convs)
    line("w16a16_mm_run", "q::ConvLayer_s1.opt", "真算-MAC", "HMX", o.get("cons", 0))
    if "mac_cyc_per_call" in o:
        cpp = (o["mac_cyc_per_call"] / o["mac_packets"]) if o.get("mac_packets") else None
        extra = f" packets/call={o['mac_packets']} cyc/pkt={cpp:.1f}" if cpp else " (packets: build -DGP_PKTPROBE)"
        print(f"       MAC per-call(occupancy)={o['mac_cyc_per_call']} (vs native single 1970 / batch warm subop 263){extra}")
    # 装料 (HVX) feed
    line("gp_pack_wt_bias(vec)", "convert_weights_to_signed+Cast", "装料-wt", "HVX", o.get("wt_vec", 0))
    line("gp_pack_wt_bias(bias)", "bias_weight_update+bias_scale_shuff", "装料-bias", "HVX", o.get("wt_bia", 0))
    line("gp_cv_to_surf(act)", "q::ForceFormat_Crouton", "装料-act", "HVX", o.get("actcopy", 0))
    line("renorm/acc(solve)", "(solve-specific; no native op)", "装料-alg", "HVX", o.get("other", 0))
    # 卸料 / 输入 (DMA on native; HVX vxor on ours, VTCM-only so I/O is in bulk)
    line("gp_surf_to_cv(out)", "q::*OutputSlice", "卸料-IO", "HVX", o.get("outcopy", 0))
    line("bulk DDR<->VTCM(excl)", "q::*InputSlice + q::*OutputSlice", "输入/卸料", "DMA",
         o.get("bulk_ld", 0) + o.get("bulk_st", 0))
    # waste
    print(f"  {'SPIN (idle-wait)':22s} ->  {'(gap; not an op)':34s} | {'waste':10s} | {'-':4s} | "
          f"{o.get('spin',0):9d} | {o.get('spin',0)//n_conv if n_conv else 0:8d}")
    print(f"  --- totals: graph-wall={o.get('wall')}  slowest-prod-life(lmax)={o.get('lmax')}  "
          f"consumer-MAC-busy={o.get('cons')}  PACKCHK={o.get('packchk')}(0=ok) ---")
    print(f"  throughput 口径: graph-wall/{n_conv} = {o.get('wall',0)//n_conv if n_conv else 0} cyc/conv "
          f"(NOT cycles_used/N — that OVERSTATES, trap#6)")


def run_device(threads: int, heads: int, scale: float) -> str:
    cmd = (f"uv run python scripts/run_w16a16_head_phase4.py --deploy "
           f"--threads {threads} --heads {heads} --scale {scale}")
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    return r.stdout + r.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native-only", action="store_true", help="just decode the two native optraces")
    ap.add_argument("--our-log", type=Path, help="a saved device-run stdout to parse our numbers from")
    ap.add_argument("--run", action="store_true", help="run our solve on device first, then report")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--scale", type=float, default=0.05)
    a = ap.parse_args()

    print("=" * 100)
    print("QNN-ALIGNED SOLVE PERF — 真算(MAC,HMX) / 装料(prep,HVX) / 卸料+输入(IO,DMA), same fields both sides")
    print("=" * 100)
    nb = native_table(NATIVE["batch"]) if NATIVE["batch"].exists() else None
    ns = native_table(NATIVE["single"]) if NATIVE["single"].exists() else None
    if ns: print_native("SINGLE conv (mm_1x1x64x64) — APPLES-to-apples vs our single per-call", ns)
    if nb: print_native("BATCH 32-conv (mm_64) — note batch HMX sub-op cycles overlap (cron#78)", nb)

    if a.native_only:
        print("\n  [native-only] supply --our-log or --run to add our side-by-side breakdown.")
        return 0

    log = None
    if a.run:
        log = run_device(a.threads, a.heads, a.scale)
    elif a.our_log:
        log = a.our_log.read_text()
    if log:
        o = parse_our(log)
        if o:
            # per-conv basis: 32 heads * 24 mm/head (Newton=0) = 768 convs is our solve workload
            n_conv = max(o.get("H", a.heads), 1) * 24
            print_ours(o, n_conv)
            print("\n  >>> READ THIS WAY: compare our 真算-MAC per-conv to NATIVE SINGLE ConvLayer (1970), "
                  "NOT to the batch warm sub-op (263).")
            print("      compare our 装料-wt/act/bias per-conv to native HVX feed ops of the SAME category.")
            print("      NEVER compare across categories or against cycles_used/N.")
            print("  NOTE: our 装料-act (~150/conv) ≪ native ForceFormat (~3000) is BY DESIGN — crouton_pos makes")
            print("        our cv==surface so act-format is a flat XOR; native does a full layout transform. Not a bug.")
        else:
            print("\n  WARN: could not parse our numbers from the log (no 'raw stats' line).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
