#!/usr/bin/env python3
"""Phase-3 gate: full C=256 head T = (I-A)^-1 on device, 56 REAL w16a16 matmuls
(4 diag Taylor+Newton + 16 merge). Per-64-block exponents returned. Compares vs fp64.

  uv run python scripts/run_w16a16_head_phase3.py [--seed 1] [--scale 0.3] [--deploy]
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
    return subprocess.run(f"timeout 300 {cmd}", shell=True, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A = np.tril(rng.uniform(-a.scale, a.scale, (256, 256)), -1)
    q16 = np.round(A * 32767).astype(np.int16)
    Aq = q16.astype(np.float64) / 32767.0
    Tref = np.linalg.inv(np.eye(256) - Aq)

    payload = bytearray(3 * 256 * 256 * 2)
    payload[:131072] = q16.tobytes()
    Path("/tmp/w16h_A.raw").write_bytes(bytes(payload))
    if a.deploy:
        for f in ["libgdnbm_skel.so", "gdnbm"]:
            sh(f"{SSH} 'cat > $HOME/gdnbm_run/{f}' < {BM}/build/{f}")
        sh(f"{SSH} 'chmod +x $HOME/gdnbm_run/gdnbm'")
    sh(f"{SSH} 'cat > $HOME/gdnbm_run/w16h_A.raw' < /tmp/w16h_A.raw")
    out = sh(f"{SSH} 'cd $HOME/gdnbm_run && LD_LIBRARY_PATH=$PWD:/vendor/lib64:/system/lib64 "
             f"ADSP_LIBRARY_PATH=\"$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" "
             f"./gdnbm 1 w16h_A.raw w16h_T.raw 3 256 32768 32768 3.05e-5 3.05e-5 2>&1'").stdout.decode()
    print("\n".join(l for l in out.splitlines() if "wall" in l or "stats" in l))
    sh(f"{SSH} 'cat $HOME/gdnbm_run/w16h_T.raw' > /tmp/w16h_T.raw")
    raw = Path("/tmp/w16h_T.raw").read_bytes()
    Tc = np.frombuffer(raw[:131072], np.int16).reshape(256, 256).astype(np.float64)
    eb = np.frombuffer(raw[131072:131136], np.int32).reshape(4, 4)

    Tdev = np.zeros((256, 256))
    for bi in range(4):
        for bj in range(bi + 1):
            blk = Tc[bi*64:(bi+1)*64, bj*64:(bj+1)*64] * (2.0 ** int(eb[bi, bj])) / 32767.0
            Tdev[bi*64:(bi+1)*64, bj*64:(bj+1)*64] = blk

    print("block exps:", eb[np.tril_indices(4)])
    oc = np.abs(Tdev - Tref).mean() / max(np.abs(Tref).mean(), 1e-12)
    print(f"||A||2={np.linalg.norm(Aq,2):.3f}  head oc={oc:.3e}  max|err|={np.abs(Tdev-Tref).max():.4g}")
    for bi in range(4):
        row = []
        for bj in range(bi + 1):
            s = (slice(bi*64,(bi+1)*64), slice(bj*64,(bj+1)*64))
            o = np.abs(Tdev[s]-Tref[s]).mean()/max(np.abs(Tref[s]).mean(),1e-12)
            row.append(f"{o:.1e}")
        print(f"  blk row {bi}: " + " ".join(row))
    print("PHASE3 RAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
