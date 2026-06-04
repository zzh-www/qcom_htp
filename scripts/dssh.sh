#!/usr/bin/env bash
# dssh.sh — robust device ssh via ONE persistent ControlMaster connection.  SOURCE this.
#
# WHY: per-connection ssh to the device's termux sshd intermittently fails to return (~5-15%; verified
# with plain `ssh oneplus 'echo hi'` = 2/40 hung, NO app involved).  That is NOT a DSP/FastRPC bug — the
# remote command completes; it's the ssh channel.  Multiplexing all commands over ONE master connection
# eliminates the per-connection flakiness (0/40 hung over the mux) and is much faster (no re-handshake).
#
# Usage:
#   source scripts/dssh.sh
#   dssh_open [host]              # open the master (default host: $DSSH_HOST or oneplus); auto-closed on EXIT
#   dssh "cmd ..."               # run a remote command over the mux (stdin/stdout passthrough)
#   dssh_put localfile  remote   # upload  (cat > remote)
#   dssh_get remotefile local    # download (cat < remote)
#   W=$(dssh 'echo $HOME/run')   # capture output normally
#   dssh_close                   # optional; EXIT trap also closes
#
# Notes:
#  - ControlPersist keeps the master alive briefly after the script ends (harmless).
#  - To localize a real "hang": redirect the REMOTE cmd to a device file (`dssh 'cmd >run.log 2>&1'`)
#    then `dssh 'cat run.log'` — ssh stdout is lost on SIGKILL, a device file is not.
#  - Isolate ssh-flakiness from an app bug: loop plain `dssh 'echo hi'`; if THAT hangs, it's the link.

DSSH_HOST="${DSSH_HOST:-oneplus}"
DSSH_CM="${DSSH_CM:-/tmp/dssh-${DSSH_HOST}-$$}"

dssh_open() {
    [ -n "${1:-}" ] && DSSH_HOST="$1"
    ssh -o ControlMaster=auto -o ControlPath="$DSSH_CM" -o ControlPersist=300 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
        "$DSSH_HOST" true 2>/dev/null
    trap dssh_close EXIT
}
# each call is bounded by DSSH_TIMEOUT (default 60s) so a rare residual mux hiccup can't hang forever.
dssh()      { timeout "${DSSH_TIMEOUT:-60}" ssh -o ControlPath="$DSSH_CM" "$DSSH_HOST" "$@"; }
dssh_put()  { timeout "${DSSH_TIMEOUT:-60}" ssh -o ControlPath="$DSSH_CM" "$DSSH_HOST" "cat > $2" < "$1"; }
dssh_get()  { timeout "${DSSH_TIMEOUT:-60}" ssh -o ControlPath="$DSSH_CM" "$DSSH_HOST" "cat $1" > "$2"; }
dssh_close() { ssh -o ControlPath="$DSSH_CM" -O exit "$DSSH_HOST" 2>/dev/null; rm -f "$DSSH_CM"; }
