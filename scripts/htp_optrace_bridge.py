#!/usr/bin/env python3
"""htp_optrace_bridge.py — P3.3: bridge optrace-path implementations (the GDNSolveHVX shipping
baseline + the handwritten/QNN matmul family) onto the SAME canonical §4 fields as the gdnbm
C15:14 stats[] table, so they sit in ONE PCYCLE table with the bare-metal routes.

WHY: GDNSolveHVX (solve_op/) and the matmul family (qnn_hmx_matmul_* / handwritten_hmx_matmul/)
are measured via optrace summary.json only — they were never in the C15:14 stats[] table. But
QNN cycles == C15:14 PCYCLE (skill htp-cycle-metric, ratio 0.9963, NO conversion). This reads the
optrace chrometrace_qnn_htp_analysis_summary.json and maps it to the §4 canonical fields:

  wall  (Q4, graph-wall)        = htp_overall_summary.timeline_cycles  (= max(end)-min(start) PCYCLE)
  cbusy (Q2, cycles_used)       = HMX unit cycles_used (the MAC occupancy)
  bottleneck-busy (Q2)          = max over units of cycles_used (the saturated unit = the real domain)
  num_dominant (Q1)             = Σ dp_cycles over dominant_path_htp_0  (critical path)
  per-op cycles                 = htp_op_types[].cycles  (装料/真算/IO breakdown)
  time_us                       = graph_execute_us  (clock self-check: wall/us ≈ 1422-1594 TURBO)

IRON LAW: GDNSolveHVX is "measure-only" (gdn_solve.md §5) — this bridge reads optrace, it NEVER
touches op code. It is a measurement-side adapter.

CROSS-IMPL RULE (§2 Q4 + feedback_cross_impl_compare_graphwall_only): the legal cross-impl number
is graph-wall ÷ N at the SAME shape+scenario. Supply --n (graph repeats folded into this trace, e.g.
H heads of 64^3 solve) so the table shows wall/N comparable to the gdnbm 32-head TOTAL ÷ N_conv.

Usage:
  # one optrace dir -> canonical fields:
  htp_optrace_bridge.py <optrace_dir_or_summary.json> [--label NAME] [--n N]
  # multi-impl same-PCYCLE table (mix optrace + gdnbm stats JSON), one --add per impl:
  htp_optrace_bridge.py --table \
     --add "GDNSolveHVX:example/gdn_native/solve_op/standalone/out_solve/optrace:768" \
     --add "u8i8-matmul:example/qnn_matmul_profile/output_u8i8_route_probe_rowid_default/optrace:1" \
     --add-gdnbm "pure-HMX:/tmp/htp_perf/phase4/spec.json"
"""
import sys, os, json, argparse


def _summary_path(p):
    if os.path.isdir(p):
        c = os.path.join(p, "chrometrace_qnn_htp_analysis_summary.json")
        return c if os.path.exists(c) else os.path.join(p, "summary.json")
    return p


def parse_optrace(path):
    """Read an optrace analysis summary -> canonical §4 field dict (PCYCLE, no conversion)."""
    sp = _summary_path(path)
    d = json.load(open(sp))
    data = d.get("data", d)
    ov = data["htp_overall_summary"]["data"]
    ov = ov[0] if isinstance(ov, list) else ov
    units = ov["htp_resources"]["data"]
    hmx = [u for u in units if u["type"] == "HMX"]
    cbusy = max((u["cycles_used"] for u in hmx), default=0)
    bottleneck = max((u["cycles_used"] for u in units), default=0)
    bunit = max(units, key=lambda u: u["cycles_used"])["type"] if units else "?"
    dp = data.get("dominant_path_htp_0", {}).get("data", [])
    num_dom = sum(x.get("dp_cycles", x.get("cycles", 0)) for x in dp)
    optypes = data.get("htp_op_types", {}).get("data", [])
    return {
        "wall": ov["timeline_cycles"],            # Q4 graph-wall = max(end)-min(start)
        "cbusy": cbusy,                            # Q2 HMX cycles_used (MAC occupancy)
        "bottleneck_busy": bottleneck,             # Q2 busiest unit cycles_used (the real domain)
        "bottleneck_unit": bunit,
        "num_dominant": num_dom,                   # Q1 critical path Σ dp_cycles
        "us": ov.get("graph_execute_us") or ov.get("time_us"),
        "optypes": optypes,
        "source": "optrace",
    }


def parse_gdnbm_spec(specpath):
    """Read a gdnbm htp_perf_report spec.json (P3.2) -> the same canonical fields, for one P-point."""
    spec = json.load(open(specpath))
    pt = spec["points"][0]
    s = pt["stats"]
    path = pt["path"]
    # §4 mapping: wall=stats[0], cbusy=stats[3]; lmax/feed differ by path (we only need wall+cbusy here).
    return {
        "wall": s[0], "cbusy": s[3] if len(s) > 3 else 0,
        "bottleneck_busy": s[3] if len(s) > 3 else 0, "bottleneck_unit": "HMX(cbusy)",
        "num_dominant": None, "us": s[6] if len(s) > 6 else None,
        "optypes": [], "source": f"gdnbm({path})",
    }


def fmtM(v):
    return "—" if v is None else (f"{v/1e6:.3f}M" if v >= 1e6 else f"{v/1e3:.1f}K")


def render_table(rows):
    """rows: list of (label, n, fields). Print the unified PCYCLE table (§4 fields, graph-wall÷N)."""
    out = []
    out.append("=" * 96)
    out.append("UNIFIED PCYCLE table — all handwriting impls on ONE 口径 (QNN cycles == C15:14 PCYCLE,")
    out.append("ratio 0.9963, no conversion). cross-impl number = graph-wall ÷ N (§2 Q4).")
    out.append("=" * 96)
    out.append(f"| {'impl':<16} | {'src':<14} | {'wall (Q4)':>10} | {'wall/N':>9} | "
               f"{'cbusy(Q2)':>9} | {'busy-unit':>14} | {'num_dom(Q1)':>11} |")
    out.append("|" + "-" * 18 + "|" + "-" * 16 + "|" + "-" * 12 + "|" + "-" * 11 + "|"
               + "-" * 11 + "|" + "-" * 16 + "|" + "-" * 13 + "|")
    for label, n, f in rows:
        wn = f["wall"] / n if (f["wall"] and n) else None
        out.append(f"| {label:<16} | {f['source']:<14} | {fmtM(f['wall']):>10} | {fmtM(wn):>9} | "
                   f"{fmtM(f['cbusy']):>9} | {f['bottleneck_unit'][:14]:>14} | {fmtM(f['num_dominant']):>11} |")
    out.append("=" * 96)
    out.append("Q4 graph-wall = the verdict; cbusy = HMX cycles_used (MAC occupancy, Q2); busy-unit = the")
    out.append("SATURATED unit (its cycles_used is the domain); num_dom = Σ dominant-path (Q1, critical chain).")
    out.append("N: optrace path = graph repeats folded in this trace (e.g. H*N_conv); gdnbm = its N_conv.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="optrace dir or summary.json (single-impl mode)")
    ap.add_argument("--label", default="impl")
    ap.add_argument("--n", type=float, default=1.0)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--add", action="append", default=[], help="optrace impl 'label:optrace_dir:N'")
    ap.add_argument("--add-gdnbm", action="append", default=[], help="gdnbm impl 'label:spec.json:N' (N optional)")
    a = ap.parse_args()
    if a.table:
        rows = []
        for spec in a.add:
            label, path, n = (spec.split(":") + ["1"])[:3]
            rows.append((label, float(n), parse_optrace(path)))
        for spec in a.add_gdnbm:
            parts = spec.split(":")
            label, path = parts[0], parts[1]
            n = float(parts[2]) if len(parts) > 2 else 1.0
            rows.append((label, n, parse_gdnbm_spec(path)))
        print(render_table(rows))
        return 0
    if not a.src:
        print(__doc__); return 2
    f = parse_optrace(a.src)
    print(render_table([(a.label, a.n, f)]))
    # per-op §4 breakdown (装料/真算/IO) for the single impl
    print("\nper-op (htp_op_types, field=cycles / num_dominant_path):")
    for r in sorted(f["optypes"], key=lambda x: -x.get("cycles", 0))[:8]:
        print(f"  {r['op']:<32} cycles={fmtM(r.get('cycles',0)):>8}  "
              f"num_dom={fmtM(r.get('num_dominant_path_cycles_htp_0',0)):>8}  inst={r.get('instances')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
