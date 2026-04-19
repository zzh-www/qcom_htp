#!/usr/bin/env bash
#
# bench_repeat.sh — run profile_all.sh N times and aggregate the per-run
# `summary.json` files into a single median + min/max table. Useful
# because chrometrace captures ONE inference per invocation and has
# ±30 % run-to-run noise from DCVS + cache state.
#
# Usage:
#   bench_repeat.sh [N] [-- profile_all.sh args...]
# Examples:
#   bench_repeat.sh 5
#   bench_repeat.sh 10 -- --arch v75 --device oneplus
set -euo pipefail

N="${1:-5}"; shift || true
[ "${1:-}" = "--" ] && shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$PWD/output}"
BASE="$(dirname "$OUT_DIR")/output_bench_$$"
mkdir -p "$BASE"

for i in $(seq 1 "$N"); do
    echo "=== run $i / $N ==="
    rm -rf "$OUT_DIR"
    bash "$SCRIPT_DIR/profile_all.sh" "$@" >/dev/null
    cp -r "$OUT_DIR" "$BASE/run_$i"
done

source "$SCRIPT_DIR/../../scripts/env.sh" >/dev/null
python - <<PY
import json, os, glob, statistics as s

base = "$BASE"
runs = sorted(glob.glob(os.path.join(base, "run_*")))
agg = {}   # config -> field -> [values]
kernels = {}  # config -> set(kernel)
fails = {}  # config -> list of reasons

for r in runs:
    sj = os.path.join(r, "summary.json")
    if not os.path.isfile(sj): continue
    d = json.load(open(sj))
    for row in d.get("configs", []):
        c = row["config"]
        for k in ("compute", "staging", "dma", "matmul_total", "input", "output", "e2e"):
            if row.get(k) is None: continue
            agg.setdefault(c, {}).setdefault(k, []).append(row[k])
        kernels.setdefault(c, set()).add(row.get("compute_kernel", "?"))
    for f in d.get("failed", []):
        name = f[0] if isinstance(f, (list, tuple)) else f["config"]
        reason = f[2] if isinstance(f, (list, tuple)) else f.get("reason", "?")
        fails.setdefault(name, set()).add(reason[:60])

print()
print(f"aggregated across {len(runs)} runs (median | min-max)")
print()
hdr = ("config", "compute", "staging", "dma", "matmul", "input", "output", "e2e", "kernel")
widths = (8, 14, 14, 10, 14, 12, 12, 14)
def h(s, w): return str(s).rjust(w)
print(f"{hdr[0]:<{widths[0]}} {h(hdr[1],widths[1])} {h(hdr[2],widths[2])} "
      f"{h(hdr[3],widths[3])} {h(hdr[4],widths[4])} "
      f"{h(hdr[5],widths[5])} {h(hdr[6],widths[6])} {h(hdr[7],widths[7])}  {hdr[8]}")
print("-" * 130)

def fmt(vals):
    if not vals: return "—".rjust(14)
    m = s.median(vals)
    return f"{int(m):>5} [{min(vals):>5}-{max(vals):>5}]"

preferred = ["fp16", "w16a16", "w8a16", "w8a8", "w4a16", "w4a8", "w4a4"]
for c in preferred:
    if c not in agg: continue
    row = agg[c]
    kk = sorted(kernels.get(c, {"?"}))
    print(f"{c:<{widths[0]}} "
          f"{fmt(row.get('compute',[])):>{widths[1]}} "
          f"{fmt(row.get('staging',[])):>{widths[2]}} "
          f"{fmt(row.get('dma',[])):>{widths[3]}} "
          f"{fmt(row.get('matmul_total',[])):>{widths[4]}} "
          f"{fmt(row.get('input',[])):>{widths[5]}} "
          f"{fmt(row.get('output',[])):>{widths[6]}} "
          f"{fmt(row.get('e2e',[])):>{widths[7]}}  {','.join(kk)}")

if fails:
    print()
    print("Failed configs (union of reasons across runs):")
    for c in preferred:
        if c in fails:
            for r in fails[c]:
                print(f"  {c:<8s}  {r}")

# Write aggregate JSON.
summary = {"runs": len(runs), "configs": {}}
for c, fields in agg.items():
    summary["configs"][c] = {k: {"median": s.median(v), "min": min(v), "max": max(v), "n": len(v)}
                              for k, v in fields.items()}
    summary["configs"][c]["kernels"] = sorted(kernels.get(c, []))
summary["failed"] = {c: sorted(r) for c, r in fails.items()}
with open(os.path.join(base, "agg_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {base}/agg_summary.json")
PY
echo
echo "per-run outputs kept under: $BASE"
