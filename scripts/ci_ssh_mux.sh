#!/usr/bin/env bash
# CI device-ssh hardening — route every `ssh <host>` in the kernel CI through ONE persistent
# ControlMaster connection.
#
# WHY: the device's termux sshd intermittently fails to return on a *fresh per-connection* ssh
# (~5-15%; the remote command actually completes, it's the ssh channel that hangs — see scripts/dssh.sh).
# A full device gate opens dozens of ssh connections, so at ~10% each it is near-certain to hang
# somewhere. Multiplexing all connections over one master eliminates it.
#
# HOW: this defines an `ssh()` shell function that injects ControlMaster/ControlPath and exports it.
# Exported bash functions propagate to child `bash subscript.sh` invocations, so the many scattered
# `ssh "$DEVICE" ...` calls in the QNN run scripts transparently reuse the master — no per-call edits.
# Also adds ConnectTimeout + ServerAlive so a genuinely dead link fails fast instead of hanging forever.
#
# Source this once near the top of the CI entry (scripts/run_qnn_kernel_e2e_ci.sh).

if [ -z "${CI_SSH_MUX_ACTIVE:-}" ]; then
  export CI_SSH_MUX_ACTIVE=1
  export CI_SSH_CM="${TMPDIR:-/tmp}/qcom_htp_ci_cm_%C"   # %C = hash of (host,port,user) — one master per target

  ssh() {
    command ssh \
      -o ControlMaster=auto \
      -o ControlPath="$CI_SSH_CM" \
      -o ControlPersist=600 \
      -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=4 \
      "$@"
  }
  export -f ssh
fi
