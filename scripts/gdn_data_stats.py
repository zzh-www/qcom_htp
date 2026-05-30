#!/usr/bin/env python3
"""Report shapes + activation ranges of the frozen GDN golden test data (torch).

Scans tests/gdn/golden, reconstructs the bf16-stored tensors, and reports per-tensor
global min/max/mean/std/abs-max over all captures, the sequence-length (T) coverage, and
the per-layer abs-max spread (how much the range varies across the 24 layers — relevant
because each layer gets its own static scales).

Usage: .venv/bin/python scripts/gdn_data_stats.py [--golden tests/gdn/golden]
"""
import argparse, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gdn_ref_kernel import bf16_to_f32  # noqa: E402

TENSORS = ["query", "key", "value", "g", "beta", "o"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(ROOT, "tests", "gdn", "golden"))
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.golden, "manifest.json")))
    recs = man["records"]

    g = {t: {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "sq": 0.0, "n": 0,
             "amax": 0.0} for t in TENSORS}
    per_layer_amax = {t: {} for t in TENSORS}
    Ts, shapes = set(), {}
    for r in recs:
        z = np.load(os.path.join(args.golden, r["file"]))
        Ts.add(r["T"]); L = r["layer"]
        for t in TENSORS:
            x = torch.from_numpy(np.ascontiguousarray(bf16_to_f32(z[t]))).float()
            shapes.setdefault(t, list(x.shape))
            mn, mx = x.min().item(), x.max().item()
            am = max(abs(mn), abs(mx))
            d = g[t]
            d["min"] = min(d["min"], mn); d["max"] = max(d["max"], mx)
            d["amax"] = max(d["amax"], am)
            d["sum"] += x.double().sum().item(); d["sq"] += (x.double() ** 2).sum().item()
            d["n"] += x.numel()
            per_layer_amax[t][L] = max(per_layer_amax[t].get(L, 0.0), am)

    print(f"golden: {args.golden}  captures={len(recs)}  layers={len(set(r['layer'] for r in recs))} "
          f"prompts={man['n_prompts']} (calib {man['n_calib']} / test {man['n_test']})")
    print(f"T (sequence length) values: {sorted(Ts)}  -> n_chunks {sorted(set((t+63)//64 for t in Ts))}")
    print(f"\nBoundary shapes (B=1, H=32, D=128; T varies): "
          f"query/key/value [1,T,32,128], g/beta [1,T,32], o [1,T,32,128]")
    print(f"Fixed kernel unit gdn_chunk: q/k/v [1,32,64,128], g/beta [1,32,64], "
          f"S_in/S_out [1,32,128,128], o [1,32,64,128]\n")

    hdr = f"{'tensor':7} | {'min':>10} {'max':>10} {'mean':>10} {'std':>9} {'abs-max':>9} | per-layer abs-max [lo, hi]"
    print(hdr); print("-" * len(hdr))
    for t in TENSORS:
        d = g[t]; mean = d["sum"] / d["n"]; std = (d["sq"] / d["n"] - mean ** 2) ** 0.5
        la = per_layer_amax[t]; lo, hi = min(la.values()), max(la.values())
        print(f"{t:7} | {d['min']:10.3f} {d['max']:10.3f} {mean:10.4f} {std:9.4f} {d['amax']:9.3f} | "
              f"[{lo:.3f}, {hi:.3f}]")
    print("\nNotes: q/k/v are POST-conv+silu, PRE-l2norm (kernel normalizes q,k internally, "
          "so per-element q,k -> ~unit-norm after). g = log-decay (<=0). beta in (0,1).")


if __name__ == "__main__":
    main()
