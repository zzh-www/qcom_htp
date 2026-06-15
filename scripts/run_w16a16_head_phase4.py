#!/usr/bin/env python3
"""Phase-4: H heads of C=256 solve in cv (crouton) domain, zero per-mm depack. Single FastRPC call.
Checks per-head oc vs fp64 and reports wall/mm cycles (32-head TOTAL wall is the metric).

  uv run python scripts/run_w16a16_head_phase4.py [--heads 32] [--scale 0.05] [--deploy]
"""
from __future__ import annotations
import argparse, re, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BM = ROOT / "example/gdn_native/baremetal"
SSH = ("ssh -o ControlMaster=auto -o ControlPath=/tmp/dssh-oneplus-p1 -o ControlPersist=300 "
       "-o ServerAliveInterval=5 -o ConnectTimeout=10 oneplus")


def sh(cmd):
    return subprocess.run(f"timeout 600 {cmd}", shell=True, check=True, capture_output=True)


def _crouton_posf():
    """crouton_pos(r,c) bit-permutation table, flattened [4096] (must match the C CROUTON_POS macro).
    cv[posf[r*64+c]] = block[r,c]. Used by the GP_CVIO=1 contract (cron#73): host does the linear<->cv
    permute so the DSP skips the per-block vgather (was scatter's #1 feed cost)."""
    r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
    pos = ((r & 1) | ((c & 1) << 1) | (((c >> 1) & 1) << 2) | (((c >> 2) & 1) << 3)
           | (((c >> 3) & 1) << 4) | (((c >> 4) & 1) << 5) | (((r >> 1) & 1) << 6)
           | (((r >> 3) & 1) << 7) | (((r >> 4) & 1) << 8) | (((r >> 5) & 1) << 9)
           | (((c >> 5) & 1) << 10) | (((r >> 2) & 1) << 11))
    return pos.ravel().astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--threads", type=int, default=1, help=">=2 = producer pool + HMX consumer")
    ap.add_argument("--reps", type=int, default=1,
                    help="GDNBM_REPS: run the solve N times in ONE FastRPC session; N>1 prints reps2-N "
                         "median wall (steady-state metric, loop doc 口径). default 1.")
    ap.add_argument("--linear", action="store_true",
                    help="legacy linear I/O (GP_CVIO=0 build); default = cv-block contract (GP_CVIO=1)")
    a = ap.parse_args()
    H = a.heads
    cvio = not a.linear
    posf = _crouton_posf()
    rng = np.random.default_rng(a.seed)
    A = np.stack([np.tril(rng.uniform(-a.scale, a.scale, (256, 256)), -1) for _ in range(H)])
    q16 = np.round(A * 32767).astype(np.int16)
    if cvio:
        # A in cv-blocks: block (bi,bj) at int16 offset (bi*4+bj)*4096, crouton_pos order. 16 blocks/head.
        Abuf = np.zeros((H, 16, 4096), dtype=np.int16)
        for h in range(H):
            for bi in range(4):
                for bj in range(bi + 1):
                    blk = q16[h, bi * 64:(bi + 1) * 64, bj * 64:(bj + 1) * 64].ravel()
                    Abuf[h, bi * 4 + bj, posf] = blk
        Path("/tmp/w16p4_A.raw").write_bytes(Abuf.tobytes())
    else:
        Path("/tmp/w16p4_A.raw").write_bytes(q16.tobytes())

    if a.deploy:
        for f in ["libgdnbm_skel.so", "gdnbm"]:
            sh(f"{SSH} 'cat > $HOME/gdnbm_run/{f}' < {BM}/build/{f}")
        sh(f"{SSH} 'chmod +x $HOME/gdnbm_run/gdnbm'")
    sh(f"{SSH} 'cat > $HOME/gdnbm_run/w16p4_A.raw' < /tmp/w16p4_A.raw")
    reps = max(1, a.reps)
    repenv = f"GDNBM_REPS={reps} " if reps > 1 else ""   # cron#80: --reps N passes GDNBM_REPS env
    out = sh(f"{SSH} 'cd $HOME/gdnbm_run && {repenv}LD_LIBRARY_PATH=$PWD:/vendor/lib64:/system/lib64 "
             f"ADSP_LIBRARY_PATH=\"$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" "
             f"./gdnbm {a.threads} w16p4_A.raw w16p4_T.raw {H} 256 32768 32768 3.05e-5 3.05e-5 2>&1'").stdout.decode()
    # surface the QNN-aligned cycle report (host prints 真算/装料/卸料 by QNN op + field; cron#79).
    _keys = ("rc=", "QNN-ALIGNED", "graph-wall", "basis:", "真算", "装料", "卸料", "waste",
             "clock self-check", "THROUGHPUT", "apples-to-apples", "raw stats", "NTSWEEP")
    print("\n".join(l for l in out.splitlines() if any(k in l for k in _keys)))
    # cron#80: reps2-N median wall (steady-state metric, loop doc 口径 — drop rep1 warm-up, never min).
    if reps > 1:
        walls = []
        for ln in out.splitlines():
            m = re.search(r"rep=(\d+)/\d+\s+wall=(\d+)", ln)
            if m:
                walls.append((int(m.group(1)), int(m.group(2))))
        steady = sorted(w for r, w in walls if r >= 2)
        if steady:
            med = steady[len(steady) // 2] if len(steady) % 2 else \
                (steady[len(steady) // 2 - 1] + steady[len(steady) // 2]) // 2
            spread = (steady[-1] - steady[0]) / med * 100 if med else 0.0
            print(f"WALL reps2-{reps} MEDIAN={med} cyc  (steady; n={len(steady)} "
                  f"min={steady[0]} max={steady[-1]} spread={spread:.1f}%; metric=VTCM-only, never min)")
        else:
            print(f"WALL reps2-{reps}: no per-rep wall lines parsed (check stdout above)")
    sh(f"{SSH} 'cat $HOME/gdnbm_run/w16p4_T.raw' > /tmp/w16p4_T.raw")
    raw = Path("/tmp/w16p4_T.raw").read_bytes()

    ocs = []
    for h in range(H):
        base = h * 131072
        Aq = q16[h].astype(np.float64) / 32767.0
        Tref = np.linalg.inv(np.eye(256) - Aq)
        Tdev = np.zeros((256, 256))
        if cvio:
            # T in cv-blocks: block (bi,bj) at int16 offset (bi*4+bj)*4096; exps in block (0,1) @ offset 4096.
            head = np.frombuffer(raw[base:base + 131072], np.int16)
            eb = np.frombuffer(raw[base + 1 * 4096 * 2:base + 1 * 4096 * 2 + 64], np.int32).reshape(4, 4)
            for bi in range(4):
                for bj in range(bi + 1):
                    cvT = head[(bi * 4 + bj) * 4096:(bi * 4 + bj) * 4096 + 4096].astype(np.float64)
                    Tdev[bi*64:(bi+1)*64, bj*64:(bj+1)*64] = (
                        cvT[posf].reshape(64, 64) * (2.0 ** int(eb[bi, bj])) / 32767.0)
        else:
            Tc = np.frombuffer(raw[base:base + 131072], np.int16).reshape(256, 256).astype(np.float64)
            eb = np.frombuffer(raw[base + 128:base + 192], np.int32).reshape(4, 4)
            for bi in range(4):
                for bj in range(bi + 1):
                    Tdev[bi*64:(bi+1)*64, bj*64:(bj+1)*64] = (
                        Tc[bi*64:(bi+1)*64, bj*64:(bj+1)*64] * (2.0 ** int(eb[bi, bj])) / 32767.0)
        ocs.append(np.abs(Tdev - Tref).mean() / max(np.abs(Tref).mean(), 1e-12))
    ocs = np.array(ocs)
    print(f"H={H} oc mean={ocs.mean():.3e} max={ocs.max():.3e} min={ocs.min():.3e}")
    print("PHASE4 RAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
