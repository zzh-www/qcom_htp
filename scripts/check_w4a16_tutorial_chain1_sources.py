#!/usr/bin/env python3
"""Check that the W4A16 tutorial wrapper uses the chain1 native source state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "handwritten_hmx_matmul"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))

from prepare_owned_inputs import (  # noqa: E402
    convw4b1x1_words,
    pack_a16_crouton16_row4_surface,
    pack_native_a16_bias,
    pack_w4_kblock32_nmajor_k4_lohi,
)


DEFAULT_ARTIFACT = Path("/tmp/handwritten_hmx_matmul_gate_w4a16_device_diag/w4a16")
DEFAULT_CHAIN8_NATIVE = ROOT / "example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256"
DEFAULT_CHAIN1_NATIVE = (
    ROOT / "example/qnn_matmul_profile/output_w4a16_native_chain1_default_ci_e2e_256"
)
DEFAULT_CUSTOM = ROOT / "example/qnn_matmul_profile/output_w4a16_aligned_e2e_256"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def first_mismatch(left: bytes, right: bytes) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def compare_bytes(name: str, actual_path: Path, expected: bytes, expected_source: str) -> dict:
    actual = actual_path.read_bytes()
    mismatch = first_mismatch(actual, expected)
    return {
        "name": name,
        "actual_path": rel(actual_path),
        "actual_bytes": len(actual),
        "actual_sha256": sha256_bytes(actual),
        "expected_source": expected_source,
        "expected_bytes": len(expected),
        "expected_sha256": sha256_bytes(expected),
        "exact": mismatch is None,
        "first_mismatch_offset": mismatch,
    }


def native_context_match(name: str, actual_path: Path, ctx_path: Path) -> dict:
    actual = actual_path.read_bytes()
    ctx = ctx_path.read_bytes()
    offset = ctx.find(actual)
    return {
        "name": name,
        "actual_path": rel(actual_path),
        "actual_bytes": len(actual),
        "actual_sha256": sha256_bytes(actual),
        "native_context": rel(ctx_path),
        "native_context_offset": offset,
        "native_context_offset_hex": f"0x{offset:x}" if offset >= 0 else None,
        "exact": offset >= 0,
    }


def load_w(path: Path) -> np.ndarray:
    return np.load(path).astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--chain8-native", type=Path, default=DEFAULT_CHAIN8_NATIVE)
    parser.add_argument("--chain1-native", type=Path, default=DEFAULT_CHAIN1_NATIVE)
    parser.add_argument("--custom", type=Path, default=DEFAULT_CUSTOM)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    prepared = args.artifact.resolve() / "prepared_state"
    if not prepared.is_dir():
        raise FileNotFoundError(f"missing prepared_state directory: {prepared}")

    chain1 = args.chain1_native.resolve()
    chain8 = args.chain8_native.resolve()
    custom = args.custom.resolve()
    activation_source = (chain1 / "runtime_inputs_native" / "A.raw").read_bytes()
    activation_u16 = np.frombuffer(activation_source, dtype="<u2").reshape(256, 256)
    activation = pack_a16_crouton16_row4_surface(activation_u16).astype("<u2", copy=False).tobytes()
    chain1_w = load_w(chain1 / "wRaw_KN.npy")
    packed_weight = pack_w4_kblock32_nmajor_k4_lohi(chain1_w).tobytes()
    folded_bias, _ = pack_native_a16_bias(4, chain1_w)
    folded_bias_bytes = folded_bias.tobytes()
    table_offsets = np.arange(64, dtype="<u4") * np.uint32(0x800)
    mask_control = np.array(
        convw4b1x1_words(0x70B, 256, 0, 0, 0, 0xA0),
        dtype="<u4",
    ).tobytes()
    control = np.array([1, 0x401, 0x20C, 0], dtype="<u4").tobytes()
    descriptor = np.array(
        [
            0x48345844,
            2,
            8,
            64,
            32,
            8,
            256,
            8,
            64,
            1,
            0x401,
            0x20C,
            0,
            1,
            14,
            0,
        ],
        dtype="<u4",
    ).tobytes()
    output_surface = bytes(131072)
    extra_control = bytes(8)
    chain1_ctx = chain1 / "ctx" / "conv_ctx.bin"

    source_equivalence = {
        "chain1_vs_chain8_activation_raw_exact": (
            sha256_file(chain1 / "runtime_inputs_native" / "A.raw")
            == sha256_file(chain8 / "runtime_inputs_native" / "A.raw")
        ),
        "chain1_vs_chain8_logical_weight_exact": (
            load_w(chain1 / "wRaw_KN.npy").shape == load_w(chain8 / "wRaw_KN.npy").shape
            and np.array_equal(load_w(chain1 / "wRaw_KN.npy"), load_w(chain8 / "wRaw_KN.npy"))
        ),
        "chain1_vs_custom_logical_weight_exact": (
            chain1_w.shape == load_w(custom / "w4a16.onnx.wRaw_KN.npy").shape
            and np.array_equal(chain1_w, load_w(custom / "w4a16.onnx.wRaw_KN.npy"))
        ),
    }
    prepared_comparisons = [
        compare_bytes(
            "activation_source",
            prepared / "activation_source.raw",
            activation_source,
            rel(chain1 / "runtime_inputs_native" / "A.raw"),
        ),
        compare_bytes(
            "activation",
            prepared / "activation.raw",
            activation,
            "chain1 runtime input formatted by pack_a16_crouton16_row4_surface",
        ),
        compare_bytes(
            "packed_weight",
            prepared / "packed_weight.raw",
            packed_weight,
            "chain1 wRaw_KN.npy packed by pack_w4_kblock32_nmajor_k4_lohi",
        ),
        compare_bytes(
            "folded_bias",
            prepared / "folded_bias.raw",
            folded_bias_bytes,
            "chain1 wRaw_KN.npy packed by pack_native_a16_bias(weight_bits=4)",
        ),
    ]
    visible_abi_comparisons = [
        compare_bytes(
            "activation_table",
            prepared / "activation_table.raw",
            table_offsets.tobytes(),
            "64 normalized native table offsets with 0x800 byte stride",
        ),
        compare_bytes(
            "output_table",
            prepared / "output_table.raw",
            table_offsets.tobytes(),
            "64 normalized native table offsets with 0x800 byte stride",
        ),
        compare_bytes(
            "mask_control",
            prepared / "mask_control.raw",
            mask_control,
            "convw4b1x1_words(0x70b, 256, 0, 0, 0, 0xa0)",
        ),
        compare_bytes(
            "control",
            prepared / "control.raw",
            control,
            "native W4A16 extra/control tuple [1, 0x401, 0x20c, 0]",
        ),
        compare_bytes(
            "descriptor",
            prepared / "descriptor.raw",
            descriptor,
            "W4A16 native-style static descriptor words",
        ),
        compare_bytes(
            "extra_control",
            prepared / "extra_control.raw",
            extra_control,
            "unused placeholder extra_control bytes",
        ),
        compare_bytes(
            "output_surface",
            prepared / "output_surface.raw",
            output_surface,
            "zero initialized write-only output surface",
        ),
    ]
    native_context_matches = [
        native_context_match("packed_weight", prepared / "packed_weight.raw", chain1_ctx),
        native_context_match("folded_bias", prepared / "folded_bias.raw", chain1_ctx),
        native_context_match("control", prepared / "control.raw", chain1_ctx),
    ]
    all_exact = (
        all(source_equivalence.values())
        and all(item["exact"] for item in prepared_comparisons)
        and all(item["exact"] for item in visible_abi_comparisons)
        and all(item["exact"] for item in native_context_matches)
    )
    payload = {
        "schema": "w4a16_tutorial_chain1_source_check.v1",
        "qnn_runtime_used": False,
        "artifact": rel(args.artifact.resolve()),
        "chain1_native": rel(chain1),
        "chain8_native": rel(chain8),
        "custom_source": rel(custom),
        "source_equivalence": source_equivalence,
        "prepared_comparisons": prepared_comparisons,
        "visible_abi_comparisons": visible_abi_comparisons,
        "native_context_matches": native_context_matches,
        "all_exact": all_exact,
        "conclusion": (
            "tutorial chain1 prepared sources and visible ABI state match the expected chain1 native contract"
            if all_exact
            else "tutorial chain1 prepared sources or visible ABI state do not match the expected chain1 native contract"
        ),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "w4a16 tutorial chain1 source check: "
        f"{'ok' if all_exact else 'FAILED'} "
        f"(prepared_exact={sum(1 for item in prepared_comparisons if item['exact'])}/"
        f"{len(prepared_comparisons)}, "
        f"visible_abi_exact={sum(1 for item in visible_abi_comparisons if item['exact'])}/"
        f"{len(visible_abi_comparisons)})"
    )
    for key, value in source_equivalence.items():
        print(f"  {key}={value}")
    for item in prepared_comparisons:
        print(f"  {item['name']}: exact={item['exact']} sha256={item['actual_sha256']}")
    for item in visible_abi_comparisons:
        print(f"  {item['name']}: exact={item['exact']} sha256={item['actual_sha256']}")
    for item in native_context_matches:
        print(
            f"  native_context {item['name']}: exact={item['exact']} "
            f"offset={item['native_context_offset_hex']} sha256={item['actual_sha256']}"
        )
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
