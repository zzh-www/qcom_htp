#!/usr/bin/env python3
"""Build the A16 sidecar consumed by the custom W8A16 HMX wrapper.

The default path generates the full custom sidecar from the recovered QNN HTP
prepare rules.  ``hybrid`` and ``native_final`` remain diagnostic modes for
comparing against QNN Native's final ``bias_to_vtcm`` record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_a16_native_bias_record import expected_record, parse_record


def overwrite_effective(dst: np.ndarray, src: np.ndarray) -> None:
    tiles = dst.reshape(-1, 512)
    src_tiles = src.reshape(-1, 512)
    for nt in range(tiles.shape[0]):
        for c in range(32):
            parity = c & 1
            lane = c // 2
            off = parity * 256 + 128 + 8 * lane
            tiles[nt, off : off + 4] = src_tiles[nt, off : off + 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--native-sidecar-raw")
    ap.add_argument(
        "--source",
        choices=["hybrid", "generated", "native_final"],
        default="generated",
        help="sidecar source; generated is the promoted custom/native exact path",
    )
    ap.add_argument("--json-out")
    args = ap.parse_args()

    _, generated, _ = expected_record(Path(args.case_dir))
    native = None
    if args.native_sidecar_raw:
        native = np.fromfile(args.native_sidecar_raw, dtype=np.uint8).reshape(generated.shape)

    if args.source == "generated":
        record = generated.copy()
    elif args.source == "native_final":
        if native is None:
            raise SystemExit("--source native_final requires --native-sidecar-raw")
        record = native.copy()
    else:
        if native is None:
            raise SystemExit("--source hybrid requires --native-sidecar-raw")
        record = native.copy()
        overwrite_effective(record, generated)

    out = Path(args.out_raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    record.astype(np.uint8).tofile(out)

    summary = {
        "case_dir": str(Path(args.case_dir)),
        "source": args.source,
        "out_raw": str(out),
        "bytes": int(record.size),
    }
    if native is not None:
        nat_control, nat_eff, _ = parse_record(native)
        rec_control, rec_eff, _ = parse_record(record)
        gen_control, gen_eff, _ = parse_record(generated)
        summary.update(
            {
                "native_vs_record_bytes": f"{int((native == record).sum())}/{record.size}",
                "generated_vs_record_bytes": f"{int((generated == record).sum())}/{record.size}",
                "native_vs_record_control_bytes": f"{int((nat_control == rec_control).sum())}/{nat_control.size}",
                "native_vs_record_effective_fields": f"{int((nat_eff == rec_eff).sum())}/{nat_eff.size}",
                "generated_vs_record_control_bytes": f"{int((gen_control == rec_control).sum())}/{gen_control.size}",
                "generated_vs_record_effective_fields": f"{int((gen_eff == rec_eff).sum())}/{gen_eff.size}",
            }
        )

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        dst = Path(args.json_out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
