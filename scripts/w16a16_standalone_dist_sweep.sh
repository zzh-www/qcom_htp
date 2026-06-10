#!/usr/bin/env bash
# Value-distribution + seed sweep for the QNN-free W16A16 standalone.
#
# The native-a16 quant contract fixes all scales at 1/32767 (act/weight/output),
# so "scale" is not a free axis (and neither the op nor the other kernels sweep
# it). The meaningful correctness axis is the VALUE DISTRIBUTION of weights and
# activations. For each (dist, seed) this generates a QNN-native reference at
# that distribution, builds the standalone prepared state, and asserts the
# byte-verified kernel body is byte-exact to native.
#
# All distributions are byte-exact, including the extreme/impulse edge cases that
# originally exposed the int16 weight high-byte clip bug (q16 in [32640,32767]
# overflowed the signed int8 high byte; fixed by clipping q16 to 32639 in
# generate_w16a16_weight_sidecar.py). Covered: uniform (multi-seed), signs,
# sparse, zeros, extreme (+-max), impulse (mostly zero-point + spikes).
#
#   bash scripts/w16a16_standalone_dist_sweep.sh
#   CONFIGS="uniform:42 extreme:0" bash scripts/w16a16_standalone_dist_sweep.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/env.sh >/dev/null 2>&1 || true
PROF="$ROOT/example/qnn_matmul_profile"
SHAPE="${SHAPE:-256,256,256}"
CONFIGS="${CONFIGS:-uniform:42 uniform:7 signs:0 sparse:0 zeros:0 extreme:0 impulse:0}"

fails=0
for cfg in $CONFIGS; do
    dist="${cfg%%:*}"; seed="${cfg##*:}"
    od="/tmp/w16a16_dist_${dist}_${seed}"
    ( cd "$PROF" && GEN_DIST="$dist" GEN_SEED="$seed" CONFIGS=w16a16 \
        bash profile_all.sh --shape "$SHAPE" --out-dir "$od" ) >/tmp/w16dist_nat.log 2>&1 \
      || { echo "[$dist seed=$seed] NATIVE-REF FAIL"; tail -3 /tmp/w16dist_nat.log; fails=$((fails+1)); continue; }
    nat="$od/w16a16"
    prep="/tmp/w16a16_dist_prep_${dist}_${seed}"; rm -rf "$prep"
    uv run python scripts/build_w16a16_standalone_prepared.py --shape "$SHAPE" \
        --act-raw "$nat/runtime_inputs_native/A.raw" --onnx "$nat/matmul.onnx" \
        --out-dir "$prep" >/tmp/w16dist_build.log 2>&1 \
      || { echo "[$dist seed=$seed] BUILD FAIL"; tail -3 /tmp/w16dist_build.log; fails=$((fails+1)); continue; }
    res=$(uv run python scripts/run_w16a16_standalone_kernel.py --artifact "$prep" \
        --shape "$SHAPE" --native-raw "$nat/device_out/Y.raw" 2>&1 | grep -oE 'BYTE-EXACT|MISMATCH' | head -1)
    if [ "$res" = "BYTE-EXACT" ]; then
        echo "[$dist seed=$seed] PASS  standalone byte-exact vs native"
    else
        echo "[$dist seed=$seed] FAIL  (${res:-no-result})"
        fails=$((fails+1))
    fi
done

echo "==============================="
if [ "$fails" -eq 0 ]; then echo "w16a16 standalone dist sweep: ALL PASS (byte-exact vs native)"; exit 0
else echo "w16a16 standalone dist sweep: $fails FAILED"; exit 1; fi
