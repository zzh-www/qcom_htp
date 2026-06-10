#!/usr/bin/env bash
# Multi-shape byte-exact sweep for the QNN-free W16A16 standalone (handwriting).
#
# For each 32-multiple shape (M>=64): generate the QNN-native reference, build a
# shape-general prepared_state, run the byte-verified kernel body bare in
# hexagon-sim (zero QNN), and assert the recovered output is byte-exact to the
# native Y.raw. This gives the handwriting route the same shape coverage the
# productized QNN op gets from scripts/w16a16_shape_sweep.sh.
#
#   bash scripts/w16a16_standalone_shape_sweep.sh
#   SHAPES="256,256,256 128,128,96" bash scripts/w16a16_standalone_shape_sweep.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh >/dev/null 2>&1 || true
PROF="$ROOT/example/qnn_matmul_profile"
DEVICE="${DEVICE:-oneplus}"

# Supported standalone envelope: M=256 x any K,N (multiples of 32). Covers the
# split path (N_t%4==0: N=256,128), the single-call path (N_t%4!=0 / N<128:
# N=96,160,32), and K in {32,64,128,256}. The productized QNN op is M-general
# down to M=64 (see scripts/w16a16_shape_sweep.sh); the *standalone* needs the
# per-shape Crouton output-block padding (which QHPI hands the op) for M<256,
# so M<256 is not yet covered by this self-contained reconstruction.
SHAPES="${SHAPES:-256,256,256 256,256,128 256,256,96 256,256,160 256,128,256 256,64,128 256,32,64}"

fails=0
for shape in $SHAPES; do
    M="${shape%%,*}"; rest="${shape#*,}"; K="${rest%%,*}"; N="${rest##*,}"
    od="/tmp/w16a16_sa_nat_${M}x${K}x${N}"
    ( cd "$PROF" && CONFIGS=w16a16 bash profile_all.sh --shape "$M,$K,$N" --out-dir "$od" ) \
        >/tmp/w16a16_sa_nat.log 2>&1 \
      || { echo "[$M x $K x $N] NATIVE-REF FAIL"; tail -3 /tmp/w16a16_sa_nat.log; fails=$((fails+1)); continue; }
    nat="$od/w16a16"
    prep="/tmp/w16a16_sa_prep_${M}x${K}x${N}"
    rm -rf "$prep"
    uv run python scripts/build_w16a16_standalone_prepared.py \
        --shape "$M,$K,$N" --act-raw "$nat/runtime_inputs_native/A.raw" \
        --onnx "$nat/matmul.onnx" --out-dir "$prep" >/tmp/w16a16_sa_build.log 2>&1 \
      || { echo "[$M x $K x $N] BUILD FAIL"; tail -3 /tmp/w16a16_sa_build.log; fails=$((fails+1)); continue; }
    line=$(uv run python scripts/run_w16a16_standalone_kernel.py \
        --artifact "$prep" --shape "$M,$K,$N" --native-raw "$nat/device_out/Y.raw" 2>&1 | grep -oE 'BYTE-EXACT|MISMATCH' | head -1)
    if [ "$line" = "BYTE-EXACT" ]; then
        echo "[$M x $K x $N] PASS  standalone byte-exact vs native"
    else
        echo "[$M x $K x $N] FAIL  (${line:-no-result})"
        fails=$((fails+1))
    fi
done

echo "==============================="
if [ "$fails" -eq 0 ]; then echo "w16a16 standalone shape sweep: ALL PASS (byte-exact vs native)"; exit 0
else echo "w16a16 standalone shape sweep: $fails FAILED"; exit 1; fi
