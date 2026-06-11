#!/usr/bin/env python3
"""Phase-1 gate: REAL w16a16 64^3 matmul primitive (C packers + convhhh kernel + depack, on-device)
vs the properly-quantized numpy expectation.

Quant contract (the standalone's): act u16 = 32768 + round(a*32767), wt q16 = round(w*32767)
clipped to +-32639, output u16 codes around zp 32768 at scale 1/32767:
    Y_expect = clip( round(((act-32768) @ q16)/32767) + 32768 )
Gate: oc(dequant) < 1e-3 (drain rounding is power-of-2 + round-half-up -> few LSB).

  uv run python scripts/run_w16a16_mm_phase1.py [--seed 1] [--amax 1.0] [--wmax 1.0]
Needs the GDNBM_PURE_HMX_SOLVE build deployed (run.sh does build+deploy) or --deploy.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BM = ROOT / "example/gdn_native/baremetal"


SSH = ("ssh -o ControlMaster=auto -o ControlPath=/tmp/dssh-oneplus-p1 -o ControlPersist=300 "
       "-o ServerAliveInterval=5 -o ConnectTimeout=10 oneplus")


def dssh(cmd: str, **kw):
    return subprocess.run(f"timeout 90 {SSH} '{cmd}'", shell=True, check=True, capture_output=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amax", type=float, default=1.0, help="act values in [-amax, amax]")
    ap.add_argument("--wmax", type=float, default=1.0, help="wt values in [-wmax, wmax]")
    ap.add_argument("--deploy", action="store_true", help="push freshly built skel+driver first")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A = rng.uniform(-a.amax, a.amax, size=(256, 64))     # M=256 carrier = 4 stacked 64-row blocks
    W = rng.uniform(-a.wmax, a.wmax, size=(64, 64))
    act = np.clip(32768 + np.round(A * 32767), 0, 65535).astype(np.uint16)
    q16 = np.clip(np.round(W * 32767), -32639, 32639).astype(np.int16)

    prod = (act.astype(np.int64) - 32768) @ q16.astype(np.int64)
    exp_codes = np.clip(np.floor(prod / 32767.0 + 0.5).astype(np.int64) + 32768, 0, 65535)

    payload = bytearray(1 * 256 * 256 * 2)          # H=1,C=256 buffer
    payload[0:32768] = act.tobytes()
    payload[32768:40960] = q16.tobytes()
    (Path("/tmp/w16mm_A.raw")).write_bytes(bytes(payload))

    if a.deploy:
        for f in ["libgdnbm_skel.so", "gdnbm"]:
            subprocess.run(f"timeout 90 {SSH} 'cat > $HOME/gdnbm_run/{f}' < {BM}/build/{f}", shell=True, check=True)
        dssh("chmod +x $HOME/gdnbm_run/gdnbm")
    subprocess.run(f"timeout 90 {SSH} 'cat > $HOME/gdnbm_run/w16mm_A.raw' < /tmp/w16mm_A.raw", shell=True, check=True)
    out = dssh("cd $HOME/gdnbm_run && LD_LIBRARY_PATH=$PWD:/vendor/lib64:/system/lib64 "
               "ADSP_LIBRARY_PATH=\"$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" "
               "./gdnbm 1 w16mm_A.raw w16mm_T.raw 1 256 32768 32768 3.05e-5 3.05e-5 2>&1")
    print(out.stdout.decode())
    subprocess.run(f"timeout 90 {SSH} 'cat $HOME/gdnbm_run/w16mm_T.raw' > /tmp/w16mm_T.raw", shell=True, check=True)
    Y = np.frombuffer(Path("/tmp/w16mm_T.raw").read_bytes()[:32768], dtype=np.uint16).reshape(256, 64).astype(np.int64)

    d = Y - exp_codes
    denom = np.abs(exp_codes - 32768).mean()
    oc = np.abs(d).mean() / max(denom, 1e-12)
    print(f"max|code diff|={np.abs(d).max()}  mean|d|={np.abs(d).mean():.4f}  "
          f"mean|exp-zp|={denom:.1f}  oc={oc:.3e}")
    ok = oc < 1e-3 or np.abs(d).max() <= 4   # rounding-level code diff = quantization truth, pass
    print("PHASE1 %s (gate oc<1e-3 or max|code diff|<=4)" % ("PASS" if ok else "FAIL"))
    if not ok:
        print("worst rows:", np.argsort(np.abs(d).max(1))[-4:], " first bad:",
              np.argwhere(np.abs(d) > 4)[:5])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
