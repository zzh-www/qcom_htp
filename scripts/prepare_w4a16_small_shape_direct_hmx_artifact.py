#!/usr/bin/env python3
"""Prepare a reduced W4A16 artifact for the direct-HMX tutorial path.

This keeps the experiment on the recovered HMX body/wrapper path.  It does not
introduce a CPU value model; it only builds a smaller prepared_state from a
device-backed native Conv oracle plus the matching custom-op weight artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE_OWNED_INPUTS = ROOT / "example" / "handwritten_hmx_matmul" / "prepare_owned_inputs.py"
DEFAULT_CUSTOM = Path("/tmp/qcom_htp_w4a16_small_m32k32n32_chain1")
DEFAULT_NATIVE = Path("/tmp/qcom_htp_w4a16_small_native_m32k32n32_chain1")
DEFAULT_OUT = Path("/tmp/handwritten_hmx_matmul_small/w4a16_m32k32n32_chain1")


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_owned_inputs", PREPARE_OWNED_INPUTS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PREPARE_OWNED_INPUTS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_or_abs(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_oracle(custom_artifact: Path, native_artifact: Path) -> dict:
    custom_io = read_json(custom_artifact / "native_io.json")
    native_io = read_json(native_artifact / "native_io.json")
    shape = custom_io.get("shape_mkn") or custom_io.get("logical_matmul_shape_mkn")
    native_shape = native_io.get("logical_matmul_shape_mkn") or native_io.get("shape_mkn")
    if shape != native_shape:
        raise ValueError(f"custom/native shape mismatch: {shape!r} vs {native_shape!r}")
    if custom_io.get("chain") != native_io.get("chain"):
        raise ValueError(f"custom/native chain mismatch: {custom_io.get('chain')} vs {native_io.get('chain')}")
    m, k, n = [int(v) for v in shape]
    if m % 32 or k % 32 or n % 32:
        raise ValueError(f"W4A16 reduced shape must stay on 32-wide HMX tiles, got {shape!r}")

    native_input = native_artifact / native_io["native_input"]
    output_name = native_io.get("output_name", "Y")
    if isinstance(output_name, list):
        output_name = output_name[0]
    native_output = native_artifact / "device_out" / f"{output_name}.raw"
    custom_ctx = custom_artifact / "ctx" / "w4a16_ctx.bin"
    for required in (
        native_input,
        native_output,
        custom_ctx,
        custom_artifact / "w4a16.onnx.wRaw_KN.npy",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    return {
        "shape_mkn": [m, k, n],
        "chain": int(custom_io["chain"]),
        "dtype": "uint16_le",
        "native_artifact": rel_or_abs(native_artifact),
        "matched_custom_artifact": rel_or_abs(custom_artifact),
        "raw_input": {
            "path": rel_or_abs(native_input),
            "bytes": native_input.stat().st_size,
            "exists": True,
            "sha256": "",
        },
        "raw_output": {
            "path": rel_or_abs(native_output),
            "bytes": native_output.stat().st_size,
            "exists": True,
            "sha256": "",
        },
        "native_compute_contract": {
            "source": "reduced native W4A16 Conv oracle plus matching custom W4A16 ctx",
            "grouping": "conv1x1_0",
            "inputs": {
                "activation": {
                    "type": 3,
                    "data_type": 1046,
                    "dims": [1, m // 32, 32, k],
                },
                "packed_weight": {
                    "type": 3,
                    "data_type": 776,
                    "dims": [1, 1, k // 2, n],
                },
                "folded_bias": {
                    "type": 3,
                    "data_type": 50,
                    "dims": [1, n // 32, 1, 128],
                },
                "control": {"type": 4, "data_type": 50, "dims": [1]},
                "extra_control": {"type": 4, "data_type": 50, "dims": [1, 1, 1, 3]},
            },
            "outputs": {
                "output_0": {
                    "type": 3,
                    "data_type": 1046,
                    "dims": [1, m // 32, 32, n],
                },
            },
        },
        "reduced_shape_policy": {
            "purpose": "small-shape direct-HMX bringup",
            "cpu_value_model_used": False,
            "native_oracle": "device-backed QNN native Conv Y.raw",
            "minimum_valid_w4a16_tile": True,
            "expansion_order": [
                "32x64x32 chain1",
                "32x256x32 chain1",
                "256x256x256 chain1",
                "256x256x256 chain8",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-artifact", type=Path, default=DEFAULT_CUSTOM)
    parser.add_argument("--native-artifact", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    custom_artifact = args.custom_artifact.resolve()
    native_artifact = args.native_artifact.resolve()
    out_dir = args.out_dir.resolve()
    oracle = build_oracle(custom_artifact, native_artifact)

    module = load_prepare_module()
    module.load_manifest = lambda: {"families": {"w4a16": oracle}}
    summary = module.write_prepared_state("w4a16", out_dir)

    small_manifest = {
        "schema": "w4a16_small_shape_direct_hmx_artifact.v1",
        "custom_artifact": rel_or_abs(custom_artifact),
        "native_artifact": rel_or_abs(native_artifact),
        "out_dir": rel_or_abs(out_dir),
        "oracle": oracle,
        "prepared_summary": summary,
    }
    (out_dir / "small_shape_manifest.json").write_text(
        json.dumps(small_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(small_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
