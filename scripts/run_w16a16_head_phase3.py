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


def _crouton_posf():
    """crouton_pos(r,c) flattened [4096] (matches C CROUTON_POS). GP_CVIO=1 cv-block I/O contract (cron#73)."""
    r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
    pos = ((r & 1) | ((c & 1) << 1) | (((c >> 1) & 1) << 2) | (((c >> 2) & 1) << 3)
           | (((c >> 3) & 1) << 4) | (((c >> 4) & 1) << 5) | (((r >> 1) & 1) << 6)
           | (((r >> 3) & 1) << 7) | (((r >> 4) & 1) << 8) | (((r >> 5) & 1) << 9)
           | (((c >> 5) & 1) << 10) | (((r >> 2) & 1) << 11))
    return pos.ravel().astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--linear", action="store_true", help="legacy linear I/O (GP_CVIO=0 build)")
    a = ap.parse_args()
    cvio = not a.linear
    posf = _crouton_posf()

    rng = np.random.default_rng(a.seed)
    A = np.tril(rng.uniform(-a.scale, a.scale, (256, 256)), -1)
    q16 = np.round(A * 32767).astype(np.int16)
    Aq = q16.astype(np.float64) / 32767.0
    Tref = np.linalg.inv(np.eye(256) - Aq)

    payload = bytearray(3 * 256 * 256 * 2)
    if cvio:
        Abuf = np.zeros((16, 4096), dtype=np.int16)   # head 0 in cv-blocks; heads 1,2 stay zero
        for bi in range(4):
            for bj in range(bi + 1):
                Abuf[bi * 4 + bj, posf] = q16[bi * 64:(bi + 1) * 64, bj * 64:(bj + 1) * 64].ravel()
        payload[:131072] = Abuf.tobytes()
    else:
        payload[:131072] = q16.tobytes()
    Path("/tmp/w16h_A.raw").write_bytes(bytes(payload))
    if a.deploy:
        for f in ["libgdnbm_skel.so", "gdnbm"]:
            sh(f"{SSH} 'cat > $HOME/gdnbm_run/{f}' < {BM}/build/{f}")
        sh(f"{SSH} 'chmod +x $HOME/gdnbm_run/gdnbm'")
    sh(f"{SSH} 'cat > $HOME/gdnbm_run/w16h_A.raw' < /tmp/w16h_A.raw")
    # P=2 (not 1): single-thread P=1 is broken under crouton8/cv-block (main-stack overflow); P>=2 works.
    out = sh(f"{SSH} 'cd $HOME/gdnbm_run && LD_LIBRARY_PATH=$PWD:/vendor/lib64:/system/lib64 "
             f"ADSP_LIBRARY_PATH=\"$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" "
             f"./gdnbm 2 w16h_A.raw w16h_T.raw 3 256 32768 32768 3.05e-5 3.05e-5 2>&1'").stdout.decode()
    print("\n".join(l for l in out.splitlines() if "wall" in l or "stats" in l))
    sh(f"{SSH} 'cat $HOME/gdnbm_run/w16h_T.raw' > /tmp/w16h_T.raw")
    raw = Path("/tmp/w16h_T.raw").read_bytes()
    Tdev = np.zeros((256, 256))
    if cvio:
        head = np.frombuffer(raw[:131072], np.int16)
        eb = np.frombuffer(raw[1 * 4096 * 2:1 * 4096 * 2 + 64], np.int32).reshape(4, 4)
        for bi in range(4):
            for bj in range(bi + 1):
                cvT = head[(bi * 4 + bj) * 4096:(bi * 4 + bj) * 4096 + 4096].astype(np.float64)
                Tdev[bi*64:(bi+1)*64, bj*64:(bj+1)*64] = (
                    cvT[posf].reshape(64, 64) * (2.0 ** int(eb[bi, bj])) / 32767.0)
    else:
        Tc = np.frombuffer(raw[:131072], np.int16).reshape(256, 256).astype(np.float64)
        eb = np.frombuffer(raw[131072:131136], np.int32).reshape(4, 4)
        for bi in range(4):
            for bj in range(bi + 1):
                Tdev[bi*64:(bi+1)*64, bj*64:(bj+1)*64] = (
                    Tc[bi*64:(bi+1)*64, bj*64:(bj+1)*64] * (2.0 ** int(eb[bi, bj])) / 32767.0)

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
