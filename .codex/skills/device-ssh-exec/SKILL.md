---
name: device-ssh-exec
description: The MANDATORY way to run ANY command on the test device (`ssh oneplus`, a termux sshd on the phone) — exec, upload, download, or probe. Use whenever ANYTHING touches the device: a one-off `echo`, deploying/running a bare-metal or QNN binary, pushing libs/inputs, pulling outputs, or a benchmark loop; and whenever a device run "hangs"/times out. The rule is one line: `source scripts/dssh.sh` once near the top of the script — that exports an `ssh()`/`scp()` override so EVERY `ssh`/`scp` (existing call sites included) plus the explicit `dssh`/`dssh_put`/`dssh_get` helpers all ride ONE persistent ControlMaster mux. What is banned is reaching the device from a shell that has NOT sourced it. Pins the proven root cause (per-connection ssh to the termux sshd intermittently fails to return ~5-15% — NOT a DSP/FastRPC/app deadlock; the remote command actually completes), the fix, large-file retry, how to localize a real hang (device-side log, not ssh stdout), and how to tell link-flakiness from an app bug.
---

# Device ssh / exec — one unified interface

Device = `ssh oneplus` (a **termux sshd on a phone**, not a normal Linux box). The cdsp runs HTP ops;
the host binary (`gdnbm`, `qnn-net-run`, …) is launched over ssh.

## THE RULE (one line)

**Any script that reaches the device must `source scripts/dssh.sh`.** That is the whole contract.
Sourcing it exports an `ssh()`/`scp()` shell-function override (and the explicit `dssh*` helpers), so
from that point on **every** `ssh`/`scp` — your existing `ssh "$DEVICE" …` call sites included — rides
ONE persistent ControlMaster connection. No per-call-site edits needed.

What is **banned**: reaching the device from a shell that has **not** sourced `dssh.sh` (a raw
per-connection ssh re-enters the flaky path — see WHY). That's the only failure mode left.

Two equivalent ways to call the device after sourcing, both muxed over the same master:
- **legacy / zero-edit** — `ssh "$DEVICE" "cmd"`, `ssh "$DEVICE" "cat > r" < l`  (transparently muxed)
- **explicit / new code** — `dssh "cmd"`, `dssh_put l r`, `dssh_get r l`  (muxed **+ timeout + retry**)

Prefer the explicit `dssh*` helpers in new code (they add `DSSH_TIMEOUT` + transfer retry); the override
exists so the dozens of pre-existing `ssh "$DEVICE"` call sites become compliant by adding ONE source line.

## Canonical pattern (new code)

```bash
source "$ROOT/scripts/dssh.sh"          # exports ssh()/scp() override + dssh* helpers; $ROOT = repo root
dssh_open                               # optional: pre-warm the master + EXIT-close trap (ssh() also self-bootstraps)
W=$(dssh 'echo $HOME/gdnbm_run'); dssh "mkdir -p $W"
dssh_put build/gdnbm            "$W/gdnbm";  dssh "chmod +x $W/gdnbm"
dssh_put build/libgdnbm_skel.so "$W/libgdnbm_skel.so"
dssh "cd $W && LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
      ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
      ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 … >run.log 2>&1"   # redirect to a DEVICE file (Rule 3)
dssh "cat $W/run.log"
dssh_get "$W/out/result.raw" out/result.raw
```

## Making an existing script compliant (the zero-edit path)

Add ONE line after the script sets its repo-root var and before its first device `ssh`/`scp`:
```bash
source "$ROOT_DIR/scripts/dssh.sh"     # reuse whatever root var the script already uses for $ROOT/scripts/env.sh
```
Leave every `ssh "$DEVICE" …` / `cat >` call site as-is — the exported override muxes them. The override
also propagates into child `bash subscript.sh` calls (exported function + the exported scalar `$DSSH_CM`),
so a parent that sourced it covers its children too.

## API and knobs (`scripts/dssh.sh`)

- `dssh_open [host]` — pre-warm the master + install an EXIT-close trap. Optional: the `ssh()` override
  self-bootstraps via `ControlMaster=auto` on first use, but `dssh_open` warms it and guarantees clean close.
- `dssh "cmd"` / `dssh_put local remote` / `dssh_get remote local` / `dssh_close`.
- `DSSH_HOST` (default `oneplus`), `DSSH_TIMEOUT` (explicit-call seconds, default 60), `DSSH_RETRIES` (default 3).
- ControlPath uses the `%C` token (hash of host/port/user) → one master per target, so overriding `ssh`
  globally is safe even when a script also ssh-es to some other host.
- `scripts/ci_ssh_mux.sh` is now a thin back-compat shim that just sources `dssh.sh` (one implementation).

## Why — the one root-caused fact (device, 2026-06-04)

**Per-connection ssh to this device intermittently fails to RETURN (~5–15%).** The remote command
*completes* — it is NOT a DSP / FastRPC / app deadlock. Proven:
- A "hung" gdnbm run's **device-side `run.log` showed full completion** (`solve rc=0x0`, output written),
  **no stuck process** on the device.
- Plain **`ssh oneplus 'echo hi'` (no app at all) hung 2/40** — the flakiness is the ssh channel itself.

One master connection multiplexes every command (0/40 hung over the mux) and skips re-handshake (faster).
So a run that "hangs" is almost always a flaky ssh channel, not your code — do not go debugging the DSP/op.

## Large files / transient resets

Even over the mux a big `cat >` can occasionally draw a reset. The explicit `dssh_put`/`dssh_get` are
**idempotent (whole-file overwrite) and auto-retry up to `DSSH_RETRIES`** — prefer them for transfers in
new code. The `ssh()` override and `dssh` (arbitrary command) are **not** retried (a command may be
non-idempotent — a benchmark, a build); the override relies on `ServerAlive` to drop a dead link, `dssh`
is `DSSH_TIMEOUT`-bounded. After a transfer, sanity-check size: `dssh "stat -c%s $remote"` vs host `stat -c%s`.

## Two derived rules

1. **Batch samples into ONE remote call.** Don't open a new connection per sample (even muxed, it's churn).
   Loop device-side in one `dssh "for i in \$(seq K); do …; done"`, or give the binary an internal repeat
   (`gdnbm` has `GDNBM_REPS=K` → K solves in one FastRPC session). Fewer connections = faster.
2. **Never trust ssh stdout to localize a hang** — it's lost when `timeout` SIGKILLs the local ssh client.
   Redirect the REMOTE command to a **device file** (`dssh 'cmd >run.log 2>&1'`), then `dssh 'cat run.log'`.

## Diagnosing "is it my op or the link?"

- Loop plain `dssh 'echo hi'` ~40×. If THAT hangs → it's the link (you are already on the mux; stop debugging the op).
- Check the device `run.log` of a "hung" run + `dssh 'ps -A | grep <binary>'` — completion + no stuck process ⇒ link.
- Clean strays only if a process is genuinely stuck: `dssh 'pkill -9 <binary>'`. (Usually none — the old
  "批量跑挂很多进程" impression was orphaned *remote* shells from `timeout`-killed per-conn ssh, which the mux avoids.)

## Compliance check

Every `.sh`/`.py` that reaches the device must source `dssh.sh` (directly or via `ci_ssh_mux.sh`). Find any
that don't:
```bash
grep -rlE '\bssh +("?\$\{?(DEVICE|DEV)\}?"?|oneplus)' --include='*.sh' --include='*.py' example/ scripts/ \
  | while read f; do grep -Eq 'dssh\.sh|ci_ssh_mux' "$f" || echo "NOT SOURCED: $f"; done
```
As of 2026-06-13 every device-touching script in `example/` + `scripts/` sources it (33/33). Exemplars:
`example/gdn_native/baselines/bench.sh`, `gdn_solve_chain.sh`.
