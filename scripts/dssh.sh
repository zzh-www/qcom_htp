#!/usr/bin/env bash
# dssh.sh — ONE unified device-ssh interface. SOURCE this near the top of any script that touches the
# device; after that EVERY `ssh`/`scp` (existing call sites included) and the explicit `dssh*` helpers
# all ride ONE persistent ControlMaster connection. Source it once and you are compliant — no per-call edits.
#
# THE RULE: any script that reaches the device must `source scripts/dssh.sh`. That is the whole contract.
# Two equivalent ways to use it afterward, both muxed over the same master:
#   - legacy / zero-edit:  `ssh "$DEVICE" "cmd"`, `ssh "$DEVICE" "cat > r" < l`   (transparently muxed)
#   - explicit (new code): `dssh "cmd"`, `dssh_put l r`, `dssh_get r l`           (muxed + timeout + retry)
# What is BANNED is reaching the device from a shell that has NOT sourced this file (a raw per-connection
# ssh re-enters the flaky path — see WHY).
#
# WHY: per-connection ssh to the device's termux sshd intermittently fails to return (~5-15%; verified
# with plain `ssh oneplus 'echo hi'` = 2/40 hung, NO app involved). The remote command COMPLETES — it's
# the ssh channel, NOT a DSP/FastRPC bug. Multiplexing every command over ONE master eliminates it
# (0/40 hung over the mux) and is faster (no re-handshake).
#
# HOW the zero-edit part works: we define an `ssh()` (and `scp()`) shell function that injects
# ControlMaster/ControlPath and `export -f` it. Exported bash functions + the exported scalar $DSSH_CM
# propagate into child `bash subscript.sh` calls, so scattered `ssh "$DEVICE" ...` reuse the master with
# no edits. ControlPath uses the `%C` token (hash of host/port/user) so each distinct target gets its own
# master — overriding `ssh` globally is safe even when a script also ssh-es somewhere else.
#
# API:
#   source scripts/dssh.sh [host]   # exports the ssh/scp override immediately; default host $DSSH_HOST or oneplus
#   dssh_open [host]                # pre-warm the master + install EXIT-close trap (optional; ssh() also self-bootstraps)
#   dssh "cmd ..."                  # explicit exec over the mux (stdin/stdout passthrough; DSSH_TIMEOUT-bounded)
#   dssh_put localfile  remote      # upload   (idempotent; retried up to DSSH_RETRIES)
#   dssh_get remotefile local       # download (idempotent; retried up to DSSH_RETRIES)
#   dssh_close                      # tear down the master (EXIT trap also does this if dssh_open ran)
# Knobs: DSSH_HOST (default oneplus), DSSH_TIMEOUT (per explicit-call seconds, default 60), DSSH_RETRIES (default 3).
# Notes:
#   - The exported ssh()/scp() override has NO hard timeout (a real device run can exceed 60s); it relies on
#     ServerAlive to drop a dead link. Only the explicit dssh/dssh_put/dssh_get carry DSSH_TIMEOUT.
#   - dssh (arbitrary command) is NOT retried — it may be non-idempotent (a benchmark/build). Only the
#     whole-file transfers dssh_put/dssh_get retry. Make a remote command idempotent before wrapping a retry.
#   - Localize a real hang: redirect the REMOTE cmd to a device file (`ssh "$DEVICE" 'cmd >run.log 2>&1'`)
#     then read it back — ssh stdout is lost on SIGKILL, a device file is not.

export DSSH_HOST="${DSSH_HOST:-oneplus}"
# %C = hash(host,port,user): one master per target, stable across PIDs and child scripts.
export DSSH_CM="${DSSH_CM:-${TMPDIR:-/tmp}/dssh-cm-%C}"

# ---- the zero-edit override: route every raw ssh/scp in this shell + children through the master ----
# Guard so re-sourcing (parent + child both source) doesn't redefine repeatedly.
if [ -z "${DSSH_OVERRIDE_ACTIVE:-}" ]; then
    export DSSH_OVERRIDE_ACTIVE=1
    ssh() {
        command ssh -o ControlMaster=auto -o ControlPath="$DSSH_CM" -o ControlPersist=600 \
            -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$@"
    }
    scp() {
        command scp -o ControlMaster=auto -o ControlPath="$DSSH_CM" -o ControlPersist=600 \
            -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$@"
    }
    # export -f is a bash builtin; guard so sourcing from zsh/sh doesn't error (those shells still
    # get the muxing ssh()/scp() in-process; only child-process propagation is bash-only, and the
    # repo's device scripts are all #!/usr/bin/env bash so they re-source + propagate fine).
    [ -n "${BASH_VERSION:-}" ] && export -f ssh scp
fi

# ---- explicit helpers (new code): same master, plus timeout + retry on idempotent transfers ----
dssh_open() {
    [ -n "${1:-}" ] && export DSSH_HOST="$1"
    local i
    for i in $(seq "${DSSH_RETRIES:-3}"); do
        command ssh -o ControlMaster=auto -o ControlPath="$DSSH_CM" -o ControlPersist=600 \
            -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
            "$DSSH_HOST" true 2>/dev/null && break
    done
    trap dssh_close EXIT
}
# NB: `timeout` execs the real ssh binary (it cannot run the ssh() function or the `command` builtin),
# so the -o ControlPath/-o ControlMaster options are passed explicitly to keep these muxed.
dssh()      { timeout "${DSSH_TIMEOUT:-60}" ssh -o ControlPath="$DSSH_CM" -o ControlMaster=auto -o ControlPersist=600 "$DSSH_HOST" "$@"; }
dssh_put()  {
    local i
    for i in $(seq "${DSSH_RETRIES:-3}"); do
        timeout "${DSSH_TIMEOUT:-60}" ssh -o ControlPath="$DSSH_CM" -o ControlMaster=auto -o ControlPersist=600 "$DSSH_HOST" "cat > $2" < "$1" && return 0
    done
    echo "dssh_put: failed after ${DSSH_RETRIES:-3} tries: $1 -> $2" >&2; return 1
}
dssh_get()  {
    local i
    for i in $(seq "${DSSH_RETRIES:-3}"); do
        timeout "${DSSH_TIMEOUT:-60}" ssh -o ControlPath="$DSSH_CM" -o ControlMaster=auto -o ControlPersist=600 "$DSSH_HOST" "cat $1" > "$2" && return 0
    done
    echo "dssh_get: failed after ${DSSH_RETRIES:-3} tries: $1 -> $2" >&2; return 1
}
dssh_close() { command ssh -o ControlPath="$DSSH_CM" -O exit "$DSSH_HOST" 2>/dev/null; }

# Convenience: `source scripts/dssh.sh oneplus` sets the host in one line (does not auto-open).
# NB: use an `if` (not `[ ] && ...`) so this last line returns 0 — otherwise sourcing with no arg
# returns non-zero and trips `set -e` in callers (e.g. the QNN run_*_chain.sh flows).
if [ -n "${1:-}" ]; then export DSSH_HOST="$1"; fi
