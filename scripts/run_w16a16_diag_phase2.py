#!/usr/bin/env python3
"""Phase-2 gate: X = (I-A)^-1 of a strictly-lower 64-block via 10 REAL w16a16 matmuls on device
(Taylor p=3 seed + 4 Newton, int16 codes + software 2-pow exponents). Compares vs fp64 inverse.
Reports oc + ||A||_2 + saturation. Per-doc: high-||A|| saturation is the expected truth.

  uv run python scripts/run_w16a16_diag_phase2.py [--seed 1] [--scale 1.0] [--deploy]
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
    return subprocess.run(f"timeout 120 {cmd}", shell=True, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="A entries ~ U(-scale,scale), strictly lower")
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A = np.tril(rng.uniform(-a.scale, a.scale, (64, 64)), -1)
    q16 = np.round(A * 32767).astype(np.int16)
    Aq = q16.astype(np.float64) / 32767.0
    Xref = np.linalg.inv(np.eye(64) - Aq)
    norm = np.linalg.norm(Aq, 2)

    payload = bytearray(2 * 256 * 256 * 2)
    payload[:8192] = q16.tobytes()
    Path("/tmp/w16d_A.raw").write_bytes(bytes(payload))
    if a.deploy:
        for f in ["libgdnbm_skel.so", "gdnbm"]:
            sh(f"{SSH} 'cat > $HOME/gdnbm_run/{f}' < {BM}/build/{f}")
        sh(f"{SSH} 'chmod +x $HOME/gdnbm_run/gdnbm'")
    sh(f"{SSH} 'cat > $HOME/gdnbm_run/w16d_A.raw' < /tmp/w16d_A.raw")
    out = sh(f"{SSH} 'cd $HOME/gdnbm_run && LD_LIBRARY_PATH=$PWD:/vendor/lib64:/system/lib64 "
             f"ADSP_LIBRARY_PATH=\"$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" "
             f"./gdnbm 1 w16d_A.raw w16d_T.raw 2 256 32768 32768 3.05e-5 3.05e-5 2>&1'").stdout.decode()
    m = re.search(r"\[7\]=(-?\d+) \[8\]=(-?\d+)", out)
    eX, sat = (int(m.group(1)), int(m.group(2))) if m else (0, -1)
    sh(f"{SSH} 'cat $HOME/gdnbm_run/w16d_T.raw' > /tmp/w16d_T.raw")
    X = np.frombuffer(Path("/tmp/w16d_T.raw").read_bytes()[:8192], np.int16).reshape(64, 64).astype(np.float64)
    Xdev = X * (2.0 ** eX) / 32767.0

    oc = np.abs(Xdev - Xref).mean() / max(np.abs(Xref).mean(), 1e-12)
    print(f"||A||2={norm:.3f}  eX={eX}  sat={sat}  max|Xref|={np.abs(Xref).max():.1f}  "
          f"max|Xdev-Xref|={np.abs(Xdev-Xref).max():.4g}  oc={oc:.3e}")
    print("PHASE2 RAN (gate: real X out; oc is the recorded truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
