#!/usr/bin/env python3
"""Parse QNN-HTP-Analysis-Summary (QHAS) with post-hoc fixup for the
profiler reader's `type=UNK` bug.

Usage:
    parse_qhas.py <profile_dir>

The QNN HTP optrace profile-viewer sometimes fails to classify HTP
resources when many HMX tiles are active (observed on int16 matmul at
≥256³ matmul with 2+ HMX kernel invocations). In that case `htp_resources`
entries report:
    type: UNK
    cycles_used: <valid>
    dram_read / dram_write / vtcm_read / vtcm_write: 0
This hides the fact that HMX is actually being used and wipes out I/O
counters.

This tool fixes both on read:
  * **type fixup** — TID → type mapping is consistent across all known
    good traces on HTP v75: tid=256 is HMX, tid=512..515 are HVX. Apply
    that mapping to any `UNK` entries and note `_type_fixup: "tid"`.
  * **I/O counter fixup** — `chrometrace_htp.json` has per-compiled-node
    `mem_{dram,vtcm}_{read,write}` in scalar_params. When htp_overall_summary
    reports 0 totals while the compiled graph shows non-zero memory ops,
    reconstruct totals by summing across nodes.

The `has_unk_fixup` column in the output flags rows where either fixup
kicked in.
"""
import json, os, sys
from collections import defaultdict


# HTP v75 TID → resource type mapping. Inferred from all known-good QHAS
# entries across fp16/w8a8/w8a16 at 32/128/256/512³ — tid 256 is the
# primary HMX thread, 512..515 are HVX contexts. If this table ever
# misses, fall back to the original UNK.
TID_TYPE = {
    256: "HMX",
    512: "HVX", 513: "HVX", 514: "HVX", 515: "HVX",
}


def _scalar(sp: dict, key: str) -> int:
    # scalar_params entries look like {"100": 4096} — single-value dict.
    v = sp.get(key, {})
    if isinstance(v, dict) and v:
        return int(next(iter(v.values())))
    return 0


def _io_from_htp(htp_path: str) -> dict:
    """Sum per-compiled-node memory ops from chrometrace_htp.json.

    Only nodes with grouping=matmul_1 (or the main op) contribute; the
    framework-level ops (Input/Output) are handled separately by QHAS.
    """
    if not os.path.isfile(htp_path):
        return {}
    d = json.load(open(htp_path))
    nodes = d.get("graph", {}).get("nodes", {})
    tot = defaultdict(int)
    for n in nodes.values():
        sp = n.get("scalar_params", {})
        tot["dram_read"]  += _scalar(sp, "mem_dram_read")
        tot["dram_write"] += _scalar(sp, "mem_dram_write")
        tot["vtcm_read"]  += _scalar(sp, "mem_vtcm_read")
        tot["vtcm_write"] += _scalar(sp, "mem_vtcm_write")
    return dict(tot)


def _kernel_counts(htp_path: str) -> dict:
    """Count HTP kernel types in compiled graph. HMX-compute kernels
    match `*ConvLayer*s1*` excluding the staging variants."""
    if not os.path.isfile(htp_path):
        return {"compute_tiles": 0, "total_nodes": 0, "kernel_types": ""}
    d = json.load(open(htp_path))
    nodes = d.get("graph", {}).get("nodes", {})
    types = defaultdict(int)
    for n in nodes.values():
        types[n.get("type", "?")] += 1
    compute_tiles = sum(
        c for t, c in types.items()
        if "ConvLayer" in t and "to_vtcm" not in t and "Concat" not in t
    )
    return {
        "compute_tiles": compute_tiles,
        "total_nodes":   sum(types.values()),
        "kernel_types":  dict(types),
    }


def load_qhas_fixed(cfg_dir: str) -> dict | None:
    qhas = os.path.join(cfg_dir, "chrometrace_qnn_htp_analysis_summary.json")
    htp  = os.path.join(cfg_dir, "chrometrace_htp.json")
    if not os.path.isfile(qhas):
        return None
    d = json.load(open(qhas))
    o = d["data"]["htp_overall_summary"]["data"][0]

    # --- type fixup: UNK -> HMX/HVX via tid ---
    fixup_applied = False
    for r in o["htp_resources"]["data"]:
        if r["type"] == "UNK" and r["tid"] in TID_TYPE:
            r["_original_type"] = "UNK"
            r["type"] = TID_TYPE[r["tid"]]
            fixup_applied = True

    # --- I/O counter fixup: if QHAS total I/O is 0 but compiled nodes
    #     report non-zero, restore from _htp.json. ---
    io = _io_from_htp(htp)
    io_fixup_applied = False
    if (o.get("total_dram_read", 0) == 0 and o.get("total_dram_write", 0) == 0
            and o.get("total_vtcm_read", 0) == 0 and o.get("total_vtcm_write", 0) == 0
            and sum(io.values()) > 0):
        o["total_dram_read"]  = io.get("dram_read", 0)
        o["total_dram_write"] = io.get("dram_write", 0)
        o["total_vtcm_read"]  = io.get("vtcm_read", 0)
        o["total_vtcm_write"] = io.get("vtcm_write", 0)
        o["total_dram"] = o["total_dram_read"] + o["total_dram_write"]
        o["total_vtcm"] = o["total_vtcm_read"] + o["total_vtcm_write"]
        io_fixup_applied = True

    hmx = [r for r in o["htp_resources"]["data"] if r["type"] == "HMX"]
    hvx = [r for r in o["htp_resources"]["data"] if r["type"] == "HVX"]

    kernel = _kernel_counts(htp)

    return {
        "timeline_cycles":  o["timeline_cycles"],
        "graph_execute_us": o["graph_execute_us"],
        "time_us":          o["time_us"],
        "inf_per_s":        round(o["inf_per_s"]),
        "hmx_used":         sum(r["cycles_used"] for r in hmx),
        "hmx_util_pct":     round(max((r["utilization"] for r in hmx), default=0), 1),
        "hvx_used":         sum(r["cycles_used"] for r in hvx),
        "has_hmx":          "yes" if hmx else "NO",
        "compute_tiles":    kernel["compute_tiles"],
        "total_nodes":      kernel["total_nodes"],
        "total_dram":       o["total_dram"],
        "total_vtcm":       o["total_vtcm"],
        "fixup_type":       fixup_applied,
        "fixup_io":         io_fixup_applied,
    }


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr); sys.exit(2)
    root = sys.argv[1]

    preferred = ["fp16", "w16a16", "w8a16", "w8a8", "w4a16", "w4a8", "w4a4"]
    rows = []
    failed = []
    for c in preferred:
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            continue
        r = load_qhas_fixed(d)
        if r:
            r["config"] = c
            rows.append(r)
        else:
            # Pick a reason from whichever log exists.
            for log in ("_convert.log", "_ctxgen.log", "_run.log"):
                lp = os.path.join(d, log)
                if os.path.isfile(lp):
                    with open(lp) as f:
                        txt = f.read()
                    hits = [ln.strip() for ln in txt.splitlines()
                            if "ERROR" in ln or "not supported" in ln or "Failed" in ln]
                    if hits:
                        failed.append((c, log.replace("_","").replace(".log",""), hits[-1][:110]))
                        break
            else:
                failed.append((c, "?", "no QHAS json"))

    hdr = ("config", "timeline_cyc", "graph_us", "hmx_util%",
           "hmx_used", "hvx_used", "hmx_tiles", "inf/s", "fixup")
    w = (8, 13, 10, 10, 10, 10, 10, 7, 8)
    print()
    print(f"{hdr[0]:<{w[0]}} {hdr[1]:>{w[1]}} {hdr[2]:>{w[2]}} "
          f"{hdr[3]:>{w[3]}} {hdr[4]:>{w[4]}} {hdr[5]:>{w[5]}} "
          f"{hdr[6]:>{w[6]}} {hdr[7]:>{w[7]}} {hdr[8]:>{w[8]}}")
    print("-" * 110)
    for r in rows:
        fix = []
        if r["fixup_type"]: fix.append("T")
        if r["fixup_io"]:   fix.append("I")
        fix_s = "".join(fix) or "-"
        print(f"{r['config']:<{w[0]}} "
              f"{r['timeline_cycles']:>{w[1]}} "
              f"{r['graph_execute_us']:>{w[2]}} "
              f"{r['hmx_util_pct']:>{w[3]}.1f} "
              f"{r['hmx_used']:>{w[4]}} {r['hvx_used']:>{w[5]}} "
              f"{r['compute_tiles']:>{w[6]}} {r['inf_per_s']:>{w[7]}} "
              f"{fix_s:>{w[8]}}")
    print()
    print("fixup:  T = type UNK→HMX/HVX via TID,  I = I/O counters rebuilt from _htp.json")

    if failed:
        print()
        print("Unsupported / failed configs:")
        for n, stage, msg in failed:
            print(f"  {n:8s} [{stage}]  {msg}")

    with open(os.path.join(root, "qhas_summary.json"), "w") as f:
        json.dump({"rows": rows, "failed": failed}, f, indent=2)
    print(f"\nwrote {os.path.join(root, 'qhas_summary.json')}")


if __name__ == "__main__":
    main()
