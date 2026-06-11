---
name: device-ssh-exec
description: How to correctly ssh into the test device (`ssh oneplus`, a termux sshd on the phone) and run commands/benchmarks WITHOUT the intermittent "hangs". Use whenever you ssh to the device in a loop, deploy+run a bare-metal/QNN binary, or a device run "hangs"/times out. Pins the proven root cause (per-connection ssh to the termux sshd intermittently fails to return ~5-15% — NOT a DSP/FastRPC/app deadlock; the remote command actually completes) and the fix (one persistent ssh ControlMaster connection reused for every command, via the sourceable helper scripts/dssh.sh). Also: how to localize a real hang (device-side log file, not ssh stdout) and how to isolate link-flakiness from an app bug.
---

# Correct device ssh / exec (and why runs "hang")

Device = `ssh oneplus` (a **termux sshd on the phone**, not a normal Linux box). The cdsp runs HTP ops;
the host binary (e.g. `gdnbm`, `qnn-net-run`) is launched over ssh.

## The one fact (root-caused on device, 2026-06-04)

**Per-connection ssh to this device intermittently fails to RETURN (~5–15%).** The remote command
*completes* — it is NOT a DSP / FastRPC / gdnbm deadlock. Proven:
- A "hung" gdnbm run's **device-side `run.log` showed full completion** (`solve rc=0x0`, output written)
  and **no stuck process** on the device.
- Plain **`ssh oneplus 'echo hi'` (no app at all) hung 2/40** — the flakiness is the ssh channel itself.

So a run that "hangs" is almost always a flaky ssh connection, not your code. Do not go debugging the
DSP/op for it.

## The fix — one persistent connection (ControlMaster), reused

Multiplex every command over ONE master connection (0/40 hung over the mux). Use the helper:

```bash
source scripts/dssh.sh
dssh_open                 # opens the ControlMaster (host: $DSSH_HOST, default oneplus); auto-closed on EXIT
W=$(dssh 'echo $HOME/gdnbm_run')
dssh_put build/gdnbm        "$W/gdnbm";  dssh "chmod +x $W/gdnbm"
dssh_put build/libgdnbm_skel.so "$W/libgdnbm_skel.so"
dssh "cd $W && LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ./gdnbm 4 A_u16_h32.raw /dev/null 32 256 ..."
dssh_close                # optional; EXIT trap also closes
```

`dssh`/`dssh_put`/`dssh_get` each carry a `DSSH_TIMEOUT` (default 60s) so a rare residual hiccup is
bounded, not infinite. Inline form (no helper): open `ssh -o ControlMaster=auto -o ControlPath=$CM
-o ControlPersist=300 -o ServerAliveInterval=5 oneplus true`, then `ssh -o ControlPath=$CM oneplus …`
for every command, and `ssh -o ControlPath=$CM -O exit oneplus` at the end.

## Two more rules that follow from the root cause

1. **Batch samples into ONE remote call.** Don't open a new ssh (new connection) per sample. Either loop
   device-side in one `dssh "for i in \$(seq K); do …; done"`, or give the binary an internal repeat
   (`gdnbm` has `GDNBM_REPS=K` → K solves in one FastRPC session). Fewer connections = fewer hangs + faster.
2. **Never trust ssh stdout to localize a hang** — it's lost when `timeout` SIGKILLs the local ssh client.
   Redirect the REMOTE command to a **device file** (`dssh 'cmd >run.log 2>&1'`) then `dssh 'cat run.log'`.
   A device file survives; the unbuffered markers (or full output) tell you exactly where it stopped.

## Diagnosing "is it my op or the link?"

- Loop plain `dssh 'echo hi'` ~40×. If THAT hangs → it's the link (use the mux; stop debugging the op).
- Check the device run.log of a "hung" run + `ps -A | grep <binary>` — completion + no stuck process ⇒ link.
- Clean strays only if a process is genuinely stuck: `dssh 'pkill -9 <binary>'`. (Usually none — the
  earlier "批量跑挂很多进程" impression was orphaned *remote* shells from `timeout`-killed per-conn ssh,
  which the mux avoids.)

## Where this is used
- `example/gdn_native/baselines/bench.sh` (regression gate) and `gdn_solve_chain.sh` both use the mux +
  `GDNBM_REPS`. Helper: `scripts/dssh.sh`.
