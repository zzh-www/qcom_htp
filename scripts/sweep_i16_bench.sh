#!/usr/bin/env bash
# Sweep int16 64^3 bench descriptor candidates: rebuild + device-run each, report rc + Cmm.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
source scripts/dssh.sh 2>/dev/null; dssh_open oneplus 2>/dev/null
W=$(dssh 'echo $HOME/gdnbm_run' 2>/dev/null)
BM=example/gdn_native/baremetal

run_one() {
  local defs="$1" label="$2"
  ( cd "$BM" && EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDNBM_HMX_BENCH_I16 $defs" bash build.sh >/tmp/i16build.log 2>&1 )
  if [ $? -ne 0 ]; then echo "$label: BUILD FAIL"; tail -3 /tmp/i16build.log; return; fi
  dssh "cat > $W/libgdnbm_skel.so" < "$BM/build/libgdnbm_skel.so" 2>/dev/null
  local out
  out=$(dssh "cd $W && LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ./gdnbm 1 A_i16_pad.raw T_i16.raw 1 256 0 0 1.0 1.0 2>&1" 2>/dev/null)
  local rc cmm
  rc=$(echo "$out" | grep -oE 'rc=0x[0-9a-f]+' | head -1)
  cmm=$(echo "$out" | grep -oE 'stats: \[0\]=[0-9-]+' | head -1)
  echo "$label: $rc  $cmm"
}

for ext in 1536 0 2048; do
  for kb in 64 128; do
    run_one "-DGDN_I16_EXTRA1=${ext}u -DGDN_I16_K_TOTAL_BYTES=${kb}u" "EXT=$ext KB=$kb"
  done
done
