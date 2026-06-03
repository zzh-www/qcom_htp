#!/usr/bin/env bash
# Deploy + run the bare-metal FastRPC HMX-threading probe on `ssh oneplus`.
set -uo pipefail
cd "$(dirname "$0")"; bash build.sh >/dev/null || exit 1
W=$(ssh oneplus 'echo $HOME/gdnbm_run'); ssh oneplus "mkdir -p $W"
ssh oneplus "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
ssh oneplus "cat > $W/gdnbm" < build/gdnbm; ssh oneplus "chmod +x $W/gdnbm"
ssh oneplus "cd $W && LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ./gdnbm ${1:-4}"
