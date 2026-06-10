#!/usr/bin/env python3
"""Build a shape-general prepared_state dir for the QNN-free W16A16 standalone.

Given an arbitrary 32-multiple shape (M>=64) plus the matched native artifact
(activation A.raw + matmul.onnx), writes the prepared_state/*.raw the standalone
driver consumes -- using the SAME proven packers as the 256^3 owned path:

  activation.raw       = pack_a16_crouton16_row4_surface(A)            (Crouton16 row4)
  packed_weight.raw    = generate_w16a16_weight_sidecar (dilated int16)
  folded_bias.raw      = generate_w16a16_weight_sidecar --bias-out
  mask_control.raw     = conv1x1_words(0x70b,0,0,0,0x80)               (dilate; word6=0x3ff)
  activation_table.raw = analytic Crouton16 row4 act offsets

This lets the standalone (and its CI sweep) cover many shapes, not just 256^3.

  uv run python scripts/build_w16a16_standalone_prepared.py \
      --shape 128,128,128 --act-raw .../A.raw --onnx .../matmul.onnx --out-dir /tmp/prep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "handwritten_hmx_matmul"
for p in (str(ROOT / "scripts"), str(EXAMPLE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_owned_inputs import pack_a16_crouton16_row4_surface  # noqa: E402
from emulate_hmx_conv1x1_params import conv1x1_words  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, help="M,K,N (all %32, M>=64)")
    ap.add_argument("--act-raw", required=True, type=Path, help="native A.raw (M*K uint16)")
    ap.add_argument("--onnx", required=True, type=Path, help="matmul.onnx (carries the int16 weight)")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    m, k, n = (int(v) for v in args.shape.split(","))
    if m % 32 or k % 32 or n % 32 or m < 64:
        ap.error("shape must be 32-multiples with M>=64")
    m_t, k_t = m // 32, k // 32
    mt_groups = m_t * 8

    prepared = args.out_dir / "prepared_state"
    prepared.mkdir(parents=True, exist_ok=True)

    # activation Crouton16 row4 surface
    a = np.fromfile(args.act_raw, dtype="<u2")
    if a.size != m * k:
        ap.error(f"act-raw size {a.size} != M*K {m*k}")
    packed = pack_a16_crouton16_row4_surface(a.reshape(m, k)).astype("<u2", copy=False)
    packed.tofile(prepared / "activation.raw")
    (prepared / "activation_source.raw").write_bytes(a.astype("<u2").tobytes())

    # weight + bias sidecars (dilated int16)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_w16a16_weight_sidecar.py"),
         "--onnx", str(args.onnx),
         "-o", str(prepared / "packed_weight.raw"),
         "--bias-out", str(prepared / "folded_bias.raw")],
        check=True, cwd=ROOT)

    # mask (dilate, arg5=0x80 -> word6=0x3ff)
    mask = np.array(conv1x1_words(0x70b, 0, 0, 0, 0x80), dtype="<u4")
    mask.tofile(prepared / "mask_control.raw")

    # analytic Crouton16 row4 activation offsets:
    #   block(row4_phase=mt&7, kt) at ((mt&7)*k_t + kt) * m_t * 256 bytes
    act_offsets = np.array(
        [(((mt & 7) * k_t) + kt) * m_t * 256 for mt in range(mt_groups) for kt in range(k_t)],
        dtype="<u4")
    act_offsets.tofile(prepared / "activation_table.raw")
    # output_table.raw is generated analytically by the driver; write a stub so
    # the prepared dir mirrors the owned layout.
    np.zeros(mt_groups * (n // 32), dtype="<u4").tofile(prepared / "output_table.raw")

    print(f"built prepared_state for {m}x{k}x{n} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
