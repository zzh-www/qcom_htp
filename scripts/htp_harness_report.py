#!/usr/bin/env python3
"""htp_harness_report.py — P3.2 glue: turn a gdnbm run's stdout into the canonical perf-report
JSON (htp_perf_report.py input, docs §"HTP Kernel Measurement Standard" §4/§7) and auto-invoke
htp_perf_report.py (+ htp_timeline.py when a trace blob is present). Shared by the three gdnbm
harnesses (run_w16a16_head_phase4.py / gdn_hvxmix_acac.sh / gdn_capture_3impl_traces.sh) so the
standard report is a DEFAULT product, not a hand-typed table.

This is ADDITIVE — callers keep their existing stdout; they just additionally call
emit_report(...) which writes <outdir>/{spec.json, report.txt} (+ timeline.svg if a trace path
is given) and returns the report path. Nothing in the caller's original behavior changes.

stats[] reconstruction: the gdnbm host prints
   raw stats[0..11]: a b c ...
   raw stats[12..19]: ...
   raw PMU stats[20..23]: COPROC THREAD_IDLE CYC1T PKT   (only when stats[26]=="PMUU")
We parse those lines back into a stats list (0..23 sufficient for §4 + §6).
"""
import sys, os, re, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_stats(stdout: str):
    """Reconstruct stats[0..23] (or as far as printed) from the host's 'raw stats' echo lines.
    Returns (stats_list, pmu_dict_or_None)."""
    stats = {}
    pmu_line = False
    for ln in stdout.splitlines():
        m = re.search(r"raw stats\[(\d+)\.\.(\d+)\]:\s*(.+)$", ln)
        if m:
            lo = int(m.group(1))
            vals = re.findall(r"-?\d+", m.group(3))
            for i, v in enumerate(vals):
                stats[lo + i] = int(v)
        mp = re.search(r"raw PMU stats\[20\.\.23\]:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", ln)
        if mp:
            pmu_line = True
            for i in range(4):
                stats[20 + i] = int(mp.group(i + 1))
    if not stats:
        return None, None
    n = max(stats) + 1
    arr = [stats.get(i, 0) for i in range(n)]
    pmu = None
    # PMU真值 present iff the host printed the 'raw PMU stats[20..23]' line (only emitted under the
    # stats[26]=="PMUU" marker) AND CYCLES_1T(stats[22])>0.
    if pmu_line and stats.get(22):
        cyc1 = stats[22]
        pmu = {"coproc_busy": stats.get(20, 0) / cyc1, "thread_idle": stats.get(21, 0) / cyc1}
    return arr, pmu


def emit_report(stdout: str, path: str, P: int, title: str, reproduce: str,
                outdir: str, trace: str = None, impl: str = None,
                pure_feed: dict = None):
    """Write <outdir>/{spec.json, report.txt} from a gdnbm run's stdout; optionally render a
    timeline SVG from `trace` (a magic-0x47545203 blob path). Returns dict of artifact paths.
      path  = "hvxmix" (SHIP/ARES) or "pure" (pure-HMX) — selects §4 stats[n] mapping.
      P     = producer thread count.
    """
    os.makedirs(outdir, exist_ok=True)
    arr, pmu = parse_stats(stdout)
    art = {}
    if arr is None:
        print(f"[htp_harness_report] no 'raw stats' line in stdout — skip report ({title})")
    else:
        spec = {"title": title,
                "device": "oneplus (termux sshd; v75 cDSP TURBO)",
                "reproduce": reproduce,
                "points": [{"path": path, "P": P, "stats": arr}]}
        if pure_feed:
            spec["pure_feed"] = pure_feed
        if pmu:
            spec["pmu"] = {str(P): pmu}
        specp = os.path.join(outdir, "spec.json")
        with open(specp, "w") as f:
            json.dump(spec, f, indent=2)
        repp = os.path.join(outdir, "report.txt")
        r = subprocess.run([sys.executable, os.path.join(HERE, "htp_perf_report.py"), specp, repp],
                           capture_output=True, text=True)
        if r.returncode == 0:
            art["spec"] = specp
            art["report"] = repp
            print(f"[htp_harness_report] perf report -> {repp}  (spec {specp})")
        else:
            print(f"[htp_harness_report] htp_perf_report.py FAILED: {r.stderr.strip()}")
    if trace and os.path.exists(trace) and impl:
        svg = os.path.join(outdir, "timeline.svg")
        r = subprocess.run([sys.executable, os.path.join(HERE, "htp_timeline.py"),
                            "single", impl, trace, svg], capture_output=True, text=True)
        if r.returncode == 0:
            art["timeline"] = svg
            print(f"[htp_harness_report] timeline -> {svg}")
        else:
            print(f"[htp_harness_report] htp_timeline.py FAILED: {r.stderr.strip()}")
    return art


# CLI: feed a saved stdout log -> report. `htp_harness_report.py <log> <path> <P> <outdir> [trace impl]`
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(2)
    log = open(sys.argv[1]).read()
    path, P, outdir = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    trace = sys.argv[5] if len(sys.argv) > 5 else None
    impl = sys.argv[6] if len(sys.argv) > 6 else None
    emit_report(log, path, P, f"gdnbm {path} P={P}", " ".join(sys.argv), outdir, trace, impl)
