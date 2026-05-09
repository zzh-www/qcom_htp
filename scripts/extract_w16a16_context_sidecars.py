#!/usr/bin/env python3
"""Extract candidate W16A16 native sidecars from an HTP context binary.

This is intentionally a narrow diagnostic for the current 256^3 native
W16A16 MatMul oracle.  `qnn-context-binary-utility` reports a 143360-byte
HTP const area for that artifact.  In the observed binary the const area starts
at file offset 0x9000 and contains:

  0x9000   8 bytes      extra/control scalar [1, 1536]
  0x9100   65536 bytes  QInt8 weight sidecar for N[0:128]
  0x19100  65536 bytes  QInt8 weight sidecar for N[128:256]
  0x29100  2048 bytes   Int32 bias/control sidecar for N[0:128]
  0x29900  2048 bytes   Int32 bias/control sidecar for N[128:256]

The script does not claim this layout is universal.  It records exact offsets
and hashes so each probe can be tied back to a concrete context binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SIDEcars = [
    ("extra_0.bin", 0x9000, 8, "extra_control"),
    ("weight_0_qint8.bin", 0x9100, 65536, "weight_qint8_split0"),
    ("weight_1_qint8.bin", 0x19100, 65536, "weight_qint8_split1"),
    ("bias_0_i32.bin", 0x29100, 2048, "bias_i32_split0"),
    ("bias_1_i32.bin", 0x29900, 2048, "bias_i32_split1"),
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("context_binary", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--const-offset", type=lambda x: int(x, 0), default=0x9000)
    args = parser.parse_args()

    blob = args.context_binary.read_bytes()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name, observed_off, size, role in SIDEcars:
        offset = args.const_offset + (observed_off - 0x9000)
        end = offset + size
        if end > len(blob):
            raise SystemExit(
                f"{name}: requested [{offset:#x}, {end:#x}) beyond file size {len(blob):#x}"
            )
        data = blob[offset:end]
        out_path = args.out_dir / name
        out_path.write_bytes(data)
        records.append({
            "name": name,
            "role": role,
            "path": str(out_path),
            "offset": offset,
            "offset_hex": f"{offset:#x}",
            "bytes": size,
            "sha256": _sha256(data),
            "nonzero_bytes": sum(byte != 0 for byte in data),
            "first32_hex": data[:32].hex(),
        })

    weights = (args.out_dir / "weight_0_qint8.bin").read_bytes() + (
        args.out_dir / "weight_1_qint8.bin"
    ).read_bytes()
    weight_path = args.out_dir / "weights_qint8_2x65536.bin"
    weight_path.write_bytes(weights)
    records.append({
        "name": weight_path.name,
        "role": "weight_qint8_combined",
        "path": str(weight_path),
        "offset": None,
        "offset_hex": None,
        "bytes": len(weights),
        "sha256": _sha256(weights),
        "nonzero_bytes": sum(byte != 0 for byte in weights),
        "first32_hex": weights[:32].hex(),
    })

    biases = (args.out_dir / "bias_0_i32.bin").read_bytes() + (
        args.out_dir / "bias_1_i32.bin"
    ).read_bytes()
    bias_path = args.out_dir / "bias_i32_2x2048.bin"
    bias_path.write_bytes(biases)
    records.append({
        "name": bias_path.name,
        "role": "bias_i32_combined",
        "path": str(bias_path),
        "offset": None,
        "offset_hex": None,
        "bytes": len(biases),
        "sha256": _sha256(biases),
        "nonzero_bytes": sum(byte != 0 for byte in biases),
        "first32_hex": biases[:32].hex(),
    })

    manifest = {
        "context_binary": str(args.context_binary),
        "context_binary_bytes": len(blob),
        "const_offset": args.const_offset,
        "const_offset_hex": f"{args.const_offset:#x}",
        "records": records,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    for record in records:
        print(
            f"{record['name']}: off={record['offset_hex']} bytes={record['bytes']} "
            f"sha256={record['sha256']} nonzero={record['nonzero_bytes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
