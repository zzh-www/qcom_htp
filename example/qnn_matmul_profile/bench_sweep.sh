#!/usr/bin/env bash
#
# bench_sweep.sh — run profile_all.sh across a sweep of matmul sizes.
#
# Produces a final cross-size table. For each SIZE S, runs the bench at
# shape=S,S,S (square matmul) and collects per-config cycles.
#
# Usage:
#   bench_sweep.sh [SIZES...] [-- profile_all args...]
# Examples:
#   bench_sweep.sh 32 128 512
#   bench_sweep.sh 32 128 512 -- --arch v75
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIZES=()
while [[ $# -gt 0 && "$1" != "--" ]]; do SIZES+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ "${#SIZES[@]}" -eq 0 ] && SIZES=(32 128 512)

BASE="$PWD/sweep_output_$$"
mkdir -p "$BASE"

for s in "${SIZES[@]}"; do
    echo ""
    echo "####################### size=${s}x${s}x${s} #######################"
    OUT_DIR="$BASE/s${s}" bash "$SCRIPT_DIR/profile_all.sh" --shape "${s},${s},${s}" "$@"
done

source "$SCRIPT_DIR/../../scripts/env.sh" >/dev/null
python - "$BASE" <<'PY'
import json, os, sys
base = sys.argv[1]
# Collect summaries by size.
sizes = sorted(int(d.lstrip('s')) for d in os.listdir(base) if d.startswith('s'))
data = {}
for s in sizes:
    sj = os.path.join(base, f"s{s}", "summary.json")
    if not os.path.isfile(sj): continue
    data[s] = json.load(open(sj))

# Build cross-size table: rows = config, columns = size.
configs = ["fp16", "w16a16", "w8a16", "w8a8", "w4a16", "w4a8", "w4a4"]
print()
print("="*110)
print("COMPUTE cycles (actual HMX MAC + overhead labeled as compute/systemservice)")
print("="*110)
print(f"{'config':<8} " + " ".join(f"{s:>12d}x{s}x{s}" for s in sizes) + "  kernels_seen")
for c in configs:
    rows_here = {}
    kernels = set()
    for s in sizes:
        d = data.get(s, {})
        for row in d.get("configs", []):
            if row["config"] == c:
                rows_here[s] = row.get("compute")
                kernels.add(row.get("compute_kernel","?"))
    if not rows_here: continue
    def fmt(v): return f"{v:>13}" if v is not None else f"{'—':>13}"
    print(f"{c:<8} " + " ".join(fmt(rows_here.get(s)) for s in sizes) + "  " + ",".join(sorted(kernels)))

print()
print("="*110)
print("MATMUL TOTAL cycles (compute + staging + dma for the matmul op)")
print("="*110)
print(f"{'config':<8} " + " ".join(f"{s:>12d}x{s}x{s}" for s in sizes))
for c in configs:
    rows = {}
    for s in sizes:
        d = data.get(s, {})
        for row in d.get("configs", []):
            if row["config"] == c:
                rows[s] = row.get("matmul_total")
    if not rows: continue
    def fmt(v): return f"{v:>13}" if v is not None else f"{'—':>13}"
    print(f"{c:<8} " + " ".join(fmt(rows.get(s)) for s in sizes))

print()
print("="*110)
print("Per-MAC cycle efficiency = matmul_total / (M*K*N)")
print("="*110)
print(f"{'config':<8} " + " ".join(f"{s:>12d}x{s}x{s}" for s in sizes))
for c in configs:
    rows = {}
    for s in sizes:
        d = data.get(s, {})
        for row in d.get("configs", []):
            if row["config"] == c:
                rows[s] = row.get("matmul_total")
    if not rows: continue
    def fmt(v, s):
        if v is None: return f"{'—':>13}"
        macs = s*s*s
        return f"{v/macs:>13.5f}"
    print(f"{c:<8} " + " ".join(fmt(rows.get(s), s) for s in sizes))

# Write aggregated JSON.
with open(os.path.join(base, "sweep_summary.json"), "w") as f:
    json.dump({"sizes": sizes, "per_size": data}, f, indent=2)
print(f"\nwrote {base}/sweep_summary.json")
PY

echo ""
echo "per-size outputs under: $BASE"
