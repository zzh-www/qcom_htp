#!/usr/bin/env python3
"""Phase-4: H heads of C=256 solve in cv (crouton) domain, zero per-mm depack. Single FastRPC call.
Checks per-head oc vs fp64 and reports wall/mm cycles (32-head TOTAL wall is the metric).

  uv run python scripts/run_w16a16_head_phase4.py [--heads 32] [--scale 0.05] [--deploy]
"""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BM = ROOT / "example/gdn_native/baremetal"
SSH = ("ssh -o ControlMaster=auto -o ControlPath=/tmp/dssh-oneplus-p1 -o ControlPersist=300 "
       "-o ServerAliveInterval=5 -o ConnectTimeout=10 oneplus")


def sh(cmd):
    return subprocess.run(f"timeout 600 {cmd}", shell=True, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--threads", type=int, default=1, help=">=2 = producer pool + HMX consumer")
    a = ap.parse_args()
    H = a.heads
    rng = np.random.default_rng(a.seed)
    A = np.stack([np.tril(rng.uniform(-a.scale, a.scale, (256, 256)), -1) for _ in range(H)])
    q16 = np.round(A * 32767).astype(np.int16)
    Path("/tmp/w16p4_A.raw").write_bytes(q16.tobytes())

    if a.deploy:
        for f in ["libgdnbm_skel.so", "gdnbm"]:
            sh(f"{SSH} 'cat > $HOME/gdnbm_run/{f}' < {BM}/build/{f}")
        sh(f"{SSH} 'chmod +x $HOME/gdnbm_run/gdnbm'")
    sh(f"{SSH} 'cat > $HOME/gdnbm_run/w16p4_A.raw' < /tmp/w16p4_A.raw")
    out = sh(f"{SSH} 'cd $HOME/gdnbm_run && LD_LIBRARY_PATH=$PWD:/vendor/lib64:/system/lib64 "
             f"ADSP_LIBRARY_PATH=\"$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" "
             f"./gdnbm {a.threads} w16p4_A.raw w16p4_T.raw {H} 256 32768 32768 3.05e-5 3.05e-5 2>&1'").stdout.decode()
    print("\n".join(l for l in out.splitlines() if "wall" in l or "stats" in l))
    sh(f"{SSH} 'cat $HOME/gdnbm_run/w16p4_T.raw' > /tmp/w16p4_T.raw")
    raw = Path("/tmp/w16p4_T.raw").read_bytes()

    ocs = []
    for h in range(H):
        Tc = np.frombuffer(raw[h*131072:(h+1)*131072], np.int16).reshape(256, 256).astype(np.float64)
        eb = np.frombuffer(raw[h*131072+128:h*131072+192], np.int32).reshape(4, 4)
        Aq = q16[h].astype(np.float64) / 32767.0
        Tref = np.linalg.inv(np.eye(256) - Aq)
        Tdev = np.zeros((256, 256))
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
