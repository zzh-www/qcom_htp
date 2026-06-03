#!/usr/bin/env bash
# bench.sh — perf regression gate for the GDN solve baselines (reads CYCLES).
#
# Reads aligned PCYCLE cycles per baseline and compares to recorded refs (PASS/FAIL):
#   bare-metal (bm_hvx_int8 / bm_hvx_int16): build -> deploy -> run K times @4-thread,
#       read C15:14 total `wall=` (gdnbm stats[0]), take the MIN (contention-free steady).
#   qnn (qnn_hvx_int16): CS=256 H=32 gdn_shape.sh -> optrace WARM-tile per-head cycles + graph WALL us.
#
# Aligned metric: QHAS `cycles` == C15:14 PCYCLE (no conversion). PCYCLE/us = 1422 @ TURBO.
# Usage:
#   bash bench.sh                 # bare-metal baselines only (fast, ~30s)
#   bash bench.sh --with-qnn      # also run the shipped QNN baseline (slow, +~4min QNN build)
#   ONLY=bm_hvx_int8 bash bench.sh  # bench just one baseline (1 build, fast)
#   K=12 TOL=1.20 bash bench.sh   # K samples / fail threshold (default K=8, TOL=1.25x)
# NOTE: each bare-metal baseline rebuilds (~90s hexagon-clang); full run ~3min. That's the CI cost.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
DEVICE="${DEVICE:-oneplus}"
K="${K:-8}"; TOL="${TOL:-1.25}"
WITH_QNN=0; [ "${1:-}" = "--with-qnn" ] && WITH_QNN=1
PCY_PER_US=1422
BM="$ROOT/example/gdn_native/baremetal"
SA="2.770166930875267e-05"; ST="6.103701895199438e-05"

# ---- reference cycles (4-thread total, 32 heads, C=256). Regenerate with: bash bench.sh --rebase ----
#  name             ref_cyc(4thr total)   note
REF_bm_hvx_int8=2844240        # ~2.00 ms
REF_bm_hvx_int16=4030000       # ~2.83 ms
REF_qnn_hvx_int16_us=4220      # graph wall us

W="$(ssh "$DEVICE" 'echo $HOME/gdnbm_run' 2>/dev/null)"
pass=1
printf "%-16s %14s %10s %12s %8s  %s\n" baseline "min_cyc(4thr)" "ms" "ref_cyc" "ratio" verdict

bm_run() {  # $1=name  $2=EXTRA_DEFS  $3=ref_cyc
    local name="$1" defs="$2" ref="$3"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "$name" ] && return   # ONLY=bm_hvx_int8 -> bench just one (1 build, fast)
    ( cd "$BM" && EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT $defs" bash build.sh ) >/dev/null 2>&1 || { echo "$name BUILDFAIL"; pass=0; return; }
    ssh "$DEVICE" "cat > $W/libgdnbm_skel.so" < "$BM/build/libgdnbm_skel.so"
    ssh "$DEVICE" "cat > $W/gdnbm" < "$BM/build/gdnbm"; ssh "$DEVICE" "chmod +x $W/gdnbm"
    ssh "$DEVICE" "pkill -9 gdnbm 2>/dev/null; true"   # clear any DSP-deadlocked straggler before sampling
    # Each sample = its OWN ssh with a per-run timeout. Rapid repeated FastRPC sessions OCCASIONALLY
    # deadlock the DSP; isolating each run (and bounding it) means a single hang is killed+skipped
    # instead of blocking the whole sweep (a loop-in-one-ssh would hang on the first stuck run).
    local mn=999999999 v ok=0
    for i in $(seq "$K"); do
        v=$(timeout 12 ssh "$DEVICE" "cd $W && LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 32768 32768 $SA $ST 2>/dev/null | grep -oE 'wall=[0-9]+' | grep -oE '[0-9]+'")
        if [ -n "$v" ]; then ok=$((ok+1)); [ "$v" -lt "$mn" ] && mn=$v
        else ssh "$DEVICE" "pkill -9 gdnbm 2>/dev/null; true" >/dev/null 2>&1; fi   # a run hung -> reap, continue
    done
    [ "$ok" = 0 ] && { echo "$name RUNFAIL (all $K samples hung)"; pass=0; return; }
    local ms ratio verdict
    ms=$(awk "BEGIN{printf \"%.3f\", $mn/$PCY_PER_US/1000}")
    ratio=$(awk "BEGIN{printf \"%.2f\", $mn/$ref}")
    verdict=$(awk "BEGIN{print ($mn <= $ref*$TOL)?\"PASS\":\"FAIL\"}")
    [ "$verdict" = FAIL ] && pass=0
    printf "%-16s %14s %10s %12s %8s  %s\n" "$name" "$mn" "$ms" "$ref" "$ratio" "$verdict"
}

bm_run bm_hvx_int8  "-DGDN_BR_MM_I8" "$REF_bm_hvx_int8"
bm_run bm_hvx_int16 ""               "$REF_bm_hvx_int16"

if [ "$WITH_QNN" = 1 ]; then
    out=$( cd "$ROOT/example/gdn_native/solve_op/standalone" && CS=256 H=32 bash gdn_shape.sh 2>/dev/null | grep -E '>>> C=256' )
    us=$(echo "$out" | grep -oE 'WALL=[0-9]+' | grep -oE '[0-9]+')
    cph=$(echo "$out" | grep -oE 'cyc/head= *[0-9,]+' | grep -oE '[0-9,]+' | tr -d ,)
    if [ -n "$us" ]; then
        ratio=$(awk "BEGIN{printf \"%.2f\", $us/$REF_qnn_hvx_int16_us}")
        verdict=$(awk "BEGIN{print ($us <= $REF_qnn_hvx_int16_us*$TOL)?\"PASS\":\"FAIL\"}")
        [ "$verdict" = FAIL ] && pass=0
        printf "%-16s %14s %10s %12s %8s  %s\n" "qnn_hvx_int16" "${cph:-?}cyc/h" "$(awk "BEGIN{printf \"%.3f\",$us/1000}")" "${REF_qnn_hvx_int16_us}us" "$ratio" "$verdict"
    else echo "qnn_hvx_int16 RUNFAIL"; pass=0; fi
fi

echo
[ "$pass" = 1 ] && { echo "BENCH PASS (all <= ${TOL}x ref)"; exit 0; } || { echo "BENCH FAIL (regression > ${TOL}x)"; exit 1; }
