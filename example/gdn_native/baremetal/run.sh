#!/usr/bin/env bash
# Deploy + run the bare-metal FastRPC HMX-threading probe on `ssh oneplus`.
set -uo pipefail
cd "$(dirname "$0")"; bash build.sh >/dev/null || exit 1
DSSH_HOST="${DEVICE:-oneplus}"; source "$(cd ../../.. && pwd)/scripts/dssh.sh"; dssh_open  # robust device ssh (skill: device-ssh-exec)
W=$(dssh 'echo $HOME/gdnbm_run'); dssh "mkdir -p $W"
dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
dssh "cd $W && LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ./gdnbm ${1:-4}"
