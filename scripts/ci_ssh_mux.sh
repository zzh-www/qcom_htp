#!/usr/bin/env bash
# ci_ssh_mux.sh — compatibility shim. The device-ssh ControlMaster mux now lives in ONE place,
# scripts/dssh.sh, which exports an `ssh()`/`scp()` override (zero-edit muxing for existing call sites)
# plus the explicit dssh/dssh_put/dssh_get helpers. This file just sources it so the CI entry
# (scripts/run_qnn_kernel_e2e_ci.sh) and anything that already sourced ci_ssh_mux.sh keep working unchanged.
#
# WHY this exists: see scripts/dssh.sh — per-connection ssh to the device's termux sshd intermittently
# hangs (~5-15%); a long multi-ssh gate would otherwise hang. One master eliminates it.

# shellcheck source=scripts/dssh.sh
. "$(dirname "${BASH_SOURCE[0]:-$0}")/dssh.sh"
# back-compat: some scripts may still gate on this flag.
export CI_SSH_MUX_ACTIVE=1
