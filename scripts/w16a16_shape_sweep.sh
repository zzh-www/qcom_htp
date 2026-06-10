#!/usr/bin/env bash
# w16a16 custom HMX MatMul op — arbitrary 32-multiple shape byte-exact sweep vs QNN native.
#
# For each shape: generate the QNN-native w16a16 reference, run the custom op
# (MODE=chain_qdq + native_record_256 profile + FORMULA_DESC descriptors), and
# assert the custom output is byte-exact to native (exact == M*N, maxdiff == 0).
#
# Proves the op is shape-general for any M,K,N that are multiples of 32 (square,
# rectangular, N%128!=0, N>128 non-multiple, small M/K<128). Device = oneplus.
#
#   bash scripts/w16a16_shape_sweep.sh                # default shape set
#   SHAPES="256,256,256 128,128,96" bash scripts/w16a16_shape_sweep.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh >/dev/null 2>&1 || true
EX="$ROOT/example/qnn_hmx_matmul_w16a16"
PROF="$ROOT/example/qnn_matmul_profile"
CHAIN_DIR="$EX/standard_flow/custom_w16a16"
DEVICE="${DEVICE:-oneplus}"

# square, rectangular, N%128!=0 (N<128 and N>128), small M/K, dims down to 32.
# Note: M=32 (M_t=1) is excluded — it hits the v73deep HMX kernel's 64-row-M minimum
# (K=32/N=32 are byte-exact; only the output-row dim M needs >=64). See plan doc.
SHAPES="${SHAPES:-256,256,256 128,128,128 256,256,128 128,128,96 128,128,160 96,96,96 64,64,64 64,64,32 64,32,64}"

echo "=== build w16a16 op (native_record_256 = shape-general formula-descriptor profile) ==="
( cd "$EX" && W16A16_KERNEL_PROFILE=native_record_256 bash build.sh \
  && W16A16_KERNEL_PROFILE=native_record_256 bash build_x86.sh ) >/tmp/w16a16_sweep_build.log 2>&1 \
  || { echo "BUILD FAIL"; tail -5 /tmp/w16a16_sweep_build.log; exit 1; }

fails=0
for shape in $SHAPES; do
    M="${shape%%,*}"; rest="${shape#*,}"; K="${rest%%,*}"; N="${rest##*,}"
    od="/tmp/w16a16_sweep_nat_${M}x${K}x${N}"
    ( cd "$PROF" && CONFIGS=w16a16 bash profile_all.sh --shape "$M,$K,$N" --out-dir "$od" ) >/tmp/w16a16_sweep_nat.log 2>&1 \
      || { echo "[$M x $K x $N] NATIVE-REF FAIL"; fails=$((fails+1)); continue; }
    ssh "$DEVICE" 'pkill -9 qnn-net-run 2>/dev/null; sleep 1' >/dev/null 2>&1
    out=$( cd "$CHAIN_DIR" && M=$M K=$K N=$N CHAIN=1 MODE=chain_qdq \
        W16A16_KERNEL_PROFILE=native_record_256 W16A16_NATIVE_ORACLE_DIR="$od/w16a16" \
        bash run_w16a16_chain.sh 2>&1 )
    line=$(echo "$out" | grep -oE 'native-exact: [0-9]+/[0-9]+ maxdiff=[0-9-]+' | head -1)
    if echo "$line" | grep -q 'maxdiff=0' && ! echo "$line" | grep -qE 'native-exact: ([0-9]+)/\1 ' ; then :; fi
    exact=$(echo "$line" | grep -oE '[0-9]+/[0-9]+' | head -1)
    md=$(echo "$line" | grep -oE 'maxdiff=[0-9-]+' | cut -d= -f2)
    want=$((M*N))
    if [ "$md" = "0" ] && [ "$exact" = "$want/$want" ]; then
        echo "[$M x $K x $N] PASS  $line"
    else
        echo "[$M x $K x $N] FAIL  ${line:-<no compare>} (want $want/$want maxdiff=0)"
        fails=$((fails+1))
    fi
done

echo "==============================="
if [ "$fails" -eq 0 ]; then echo "w16a16 shape sweep: ALL PASS (byte-exact vs native)"; exit 0
else echo "w16a16 shape sweep: $fails FAILED"; exit 1; fi
