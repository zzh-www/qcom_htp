#!/usr/bin/env bash
# bench.sh — perf regression gate for the GDN solve baselines (reads CYCLES).
#
# Reads aligned PCYCLE cycles per baseline and compares to recorded refs (PASS/FAIL):
#   bare-metal (bm_hvx_int8 / bm_hvx_int16): build -> deploy -> ONE run with GDNBM_REPS=K @4-thread,
#       read C15:14 `wall=` from each rep (gdnbm stats[0]), take the MIN (contention-free steady).
#   qnn (qnn_hvx_int16): CS=256 H=32 gdn_shape.sh -> optrace WARM-tile per-head cycles + graph WALL us.
#
# Aligned metric: QHAS `cycles` == C15:14 PCYCLE (no conversion). PCYCLE/us = 1422 @ TURBO.
#
# ROBUSTNESS: per-connection ssh to the device's termux sshd intermittently hangs (~5-15%, NOT a DSP/
# gdnbm bug — gdnbm always completes; it's the ssh channel). Fix = ONE persistent ControlMaster
# connection reused for every command (eliminates per-connection flakiness) + GDNBM_REPS so K samples
# are one remote call.
#
# Usage:
#   bash bench.sh                 # bare-metal baselines (fast); PASS if all <= 1.25x ref
#   ONLY=bm_hvx_int8 bash bench.sh
#   bash bench.sh --with-qnn      # + shipped QNN baseline (slow QNN build)
#   K=12 TOL=1.20 bash bench.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
DEVICE="${DEVICE:-oneplus}"
K="${K:-8}"; TOL="${TOL:-1.25}"
WITH_QNN=0; [ "${1:-}" = "--with-qnn" ] && WITH_QNN=1
PCY_PER_US=1422
BM="$ROOT/example/gdn_native/baremetal"
SA="2.770166930875267e-05"; ST="6.103701895199438e-05"

# ---- reference cycles (4-thread total, 32 heads, C=256). ----
REF_bm_hvx_int8=2844240        # ~2.00 ms
REF_bm_hvx_int16=4030000       # ~2.83 ms
REF_qnn_hvx_int16_us=4220      # graph wall us

# ---- ssh ControlMaster: one persistent connection, reused (kills the per-connection hang) ----
CM="/tmp/gdnbench-cm-$$"
ssh -o ControlMaster=auto -o ControlPath="$CM" -o ControlPersist=300 -o ServerAliveInterval=5 "$DEVICE" true 2>/dev/null
D() { ssh -o ControlPath="$CM" "$DEVICE" "$@"; }   # all device commands go through the mux
cleanup() { ssh -o ControlPath="$CM" -O exit "$DEVICE" 2>/dev/null; rm -f "$CM"; }
trap cleanup EXIT
W="$(D 'echo $HOME/gdnbm_run')"
pass=1
printf "%-16s %14s %10s %12s %8s  %s\n" baseline "min_cyc(4thr)" "ms" "ref_cyc" "ratio" verdict

bm_run() {  # $1=name  $2=EXTRA_DEFS  $3=ref_cyc
    local name="$1" defs="$2" ref="$3"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "$name" ] && return
    ( cd "$BM" && EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT $defs" bash build.sh ) >/dev/null 2>&1 || { echo "$name BUILDFAIL"; pass=0; return; }
    D "cat > $W/libgdnbm_skel.so" < "$BM/build/libgdnbm_skel.so"
    D "cat > $W/gdnbm" < "$BM/build/gdnbm"; D "chmod +x $W/gdnbm"
    # K reps in ONE muxed remote process (one FastRPC session) -> K wall samples -> min
    local mn
    mn=$(D "cd $W && GDNBM_REPS=$K LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 32768 32768 $SA $ST 2>/dev/null | grep -oE 'wall=[0-9]+' | grep -oE '[0-9]+' | sort -n | head -1")
    [ -z "$mn" ] && { echo "$name RUNFAIL"; pass=0; return; }
    local ms ratio verdict
    ms=$(awk "BEGIN{printf \"%.3f\", $mn/$PCY_PER_US/1000}")
    ratio=$(awk "BEGIN{printf \"%.2f\", $mn/$ref}")
    verdict=$(awk "BEGIN{print ($mn <= $ref*$TOL)?\"PASS\":\"FAIL\"}")
    [ "$verdict" = FAIL ] && pass=0
    printf "%-16s %14s %10s %12s %8s  %s\n" "$name" "$mn" "$ms" "$ref" "$ratio" "$verdict"
}

bm_run bm_hvx_int8  "-DGDN_BR_MM_I8" "$REF_bm_hvx_int8"
bm_run bm_hvx_int16 ""               "$REF_bm_hvx_int16"

if [ "$WITH_QNN" = 1 ] && [ -z "${ONLY:-}" ]; then
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
