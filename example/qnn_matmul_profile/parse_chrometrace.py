#!/usr/bin/env python3
"""Aggregate per-op cycle breakdown from QNN chrometrace.json files.

Usage:
    python parse_chrometrace.py <profile_dir>

<profile_dir> contains subdirectories like fp16/, w8a8/, w4a4/, each with
a chrometrace.json. Prints a summary that separates the matmul op's
cycles into:
    compute  -- the actual HMX MAC kernel (`q::ConvLayer.fp16.s1.tcm`,
                `q::ConvLayer_s1.opt`, etc.)
    staging  -- VTCM weight/bias staging kernels
    dma      -- DmaCheckpointSet / SyncOp / ChunkPreload
and also shows total op cycles, graph input/output op cycles, and the
name of the dominant compute kernel.
"""
import json, os, sys
from collections import defaultdict


def load_cycles(path: str) -> dict:
    """Return {qnn_op_name: {"total": int, "by_type": {type: cycles}}}.

    chrometrace emits each event twice (on 2 threads). We deduplicate by
    halving the summed durations.
    """
    doc = json.load(open(path))
    events = doc["traceEvents"] if isinstance(doc, dict) else doc
    by_op: dict = defaultdict(lambda: {"total": 0, "by_type": defaultdict(int)})

    for e in events:
        if e.get("ph") != "X":
            continue
        a = e.get("args", {})
        qop = a.get("QNN Op Name", "")
        htp = a.get("HTP Op Type", "")
        d = a.get("Duration (cycles)", e.get("dur", 0))
        by_op[qop]["total"] += d
        by_op[qop]["by_type"][htp] += d

    for op in by_op:
        by_op[op]["total"] //= 2
        by_op[op]["by_type"] = {t: d // 2 for t, d in by_op[op]["by_type"].items()}
    return dict(by_op)


def pick_matmul(cycles: dict) -> dict | None:
    for k in cycles:
        if "matmul" in k.lower() or k.startswith("matmul"):
            return cycles[k]
    return None


# Classify HTP op types into buckets. The actual MAC kernels start with
# `q::` and usually contain `s1` (stream-1, the HMX compute tile).
def _classify(htp_type: str) -> str:
    t = htp_type
    if "weights_to_vtcm" in t or "bias_to_vtcm" in t:
        return "staging"
    if t in ("DmaCheckpointSet", "SyncOp", "ChunkPreload") or "DmaCheckpoint" in t:
        return "dma"
    if "InputSlice" in t or "OutputSlice" in t or "Crouton" in t:
        return "io"
    if t.startswith("q::"):
        return "compute"
    return "other"


def split_matmul(matmul: dict) -> dict:
    out = {"compute": 0, "staging": 0, "dma": 0, "other": 0, "compute_kernel": "?"}
    best = 0
    for t, c in matmul["by_type"].items():
        cls = _classify(t)
        if cls == "compute" and c > best:
            best = c
            out["compute_kernel"] = t
        if cls in ("compute", "staging", "dma"):
            out[cls] += c
        else:
            out["other"] += c
    # When chrometrace collapses everything to SystemService (happens at
    # larger matmul sizes where the optrace reader can't demux sub-events),
    # `compute` is 0 and `other` holds the entire bucket. Surface those
    # as "compute" so the table isn't deceptive, and flag the kernel.
    if out["compute"] == 0 and out["staging"] == 0 and out["other"] > 0:
        out["compute"] = out["other"]
        out["other"] = 0
        out["compute_kernel"] = "SystemService(lumped)"
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr); sys.exit(2)
    root = sys.argv[1]

    preferred = ["fp16", "w16a16", "w8a16", "w8a8", "w4a16", "w4a8", "w4a4"]
    configs = []
    for c in preferred:
        p = os.path.join(root, c, "chrometrace.json")
        if os.path.isfile(p):
            configs.append((c, p))
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "chrometrace.json")
        if os.path.isfile(p) and all(c[0] != name for c in configs):
            configs.append((name, p))

    failed = []
    for c in preferred:
        if not os.path.isfile(os.path.join(root, c, "chrometrace.json")) \
                and os.path.isdir(os.path.join(root, c)):
            for log in ("_convert.log", "_ctxgen.log", "_run.log"):
                lp = os.path.join(root, c, log)
                if os.path.isfile(lp):
                    with open(lp) as f:
                        txt = f.read()
                    hit = [ln.strip() for ln in txt.splitlines()
                           if "ERROR" in ln or "not supported" in ln or "Failed" in ln]
                    if hit:
                        failed.append((c, log.replace("_", "").replace(".log", ""),
                                       hit[-1][:120])); break
            else:
                failed.append((c, "?", "no logs"))

    if not configs:
        print("No chrometrace.json files found under", root, file=sys.stderr)
        if failed:
            print("\nFailed configs:")
            for n, stage, msg in failed:
                print(f"  {n:8s} @ {stage}: {msg}")
        sys.exit(1)

    rows = []
    for name, path in configs:
        cyc = load_cycles(path)
        matmul = pick_matmul(cyc)
        inp    = cyc.get("Input", None)
        out    = cyc.get("Output", None)

        if matmul:
            split = split_matmul(matmul)
            row = {
                "config":  name,
                "compute": split["compute"],
                "staging": split["staging"],
                "dma":     split["dma"],
                "matmul_total": matmul["total"],
                "compute_kernel": split["compute_kernel"],
                "input":   inp["total"] if inp else None,
                "output":  out["total"] if out else None,
                "e2e":     sum(v["total"] for v in cyc.values()),
            }
        else:
            row = {"config": name, "compute": None, "staging": None, "dma": None,
                   "matmul_total": None, "compute_kernel": "?", "input": None,
                   "output": None, "e2e": None}
        rows.append(row)

    # --- print summary ---
    hdr = ("config", "compute", "staging", "dma", "matmul",
           "input", "output", "e2e", "compute kernel")
    widths = (8, 8, 8, 6, 8, 7, 7, 7)
    print()
    print(f"{hdr[0]:<{widths[0]}} {hdr[1]:>{widths[1]}} {hdr[2]:>{widths[2]}} "
          f"{hdr[3]:>{widths[3]}} {hdr[4]:>{widths[4]}} "
          f"{hdr[5]:>{widths[5]}} {hdr[6]:>{widths[6]}} {hdr[7]:>{widths[7]}}  {hdr[8]}")
    print("-" * 100)
    def _f(x, w=0):
        s = "—" if x is None else str(x)
        return s.rjust(w) if w else s
    for r in rows:
        print(f"{r['config']:<{widths[0]}} "
              f"{_f(r['compute'], widths[1])} {_f(r['staging'], widths[2])} "
              f"{_f(r['dma'], widths[3])} {_f(r['matmul_total'], widths[4])} "
              f"{_f(r['input'], widths[5])} {_f(r['output'], widths[6])} "
              f"{_f(r['e2e'], widths[7])}  {r['compute_kernel']}")
    print()

    if failed:
        print("Unsupported / failed configs:")
        for n, stage, msg in failed:
            print(f"  {n:8s} [{stage}]  {msg}")
        print()

    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump({"configs": rows, "failed": failed}, f, indent=2)
    print(f"wrote {os.path.join(root, 'summary.json')}")


if __name__ == "__main__":
    main()
