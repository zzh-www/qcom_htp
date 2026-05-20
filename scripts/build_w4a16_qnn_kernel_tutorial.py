#!/usr/bin/env python3
"""Build a tutorial-style W4A16 wrapper around the recovered QNN HMX body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_handwritten_artifact_body_device import (
    ROOT,
    build_device_so,
    parse_internal_u16_sample,
    rel,
    shifted_pointer_storage,
)
from run_handwritten_artifact_body_sim import fnv1a32


DEFAULT_ARTIFACT = Path("/tmp/handwritten_hmx_matmul_gate_w4a16_device_diag/w4a16")
DEFAULT_OUT_DIR = (
    ROOT
    / "example"
    / "handwritten_hmx_matmul"
    / "tutorial_w4a16_qnn_kernel"
    / "build"
)
DEFAULT_CHAIN1_NATIVE_RAW = (
    ROOT
    / "example"
    / "qnn_matmul_profile"
    / "output_w4a16_native_chain1_default_ci_e2e_256"
    / "device_out"
    / "Y.raw"
)


def prepared_checksum_manifest(
    artifact: Path,
    output_seed_mode: str,
    activation_raw_override: Path | None = None,
    packed_weight_byte_offset: int = 0,
    packed_weight_raw_override: Path | None = None,
    folded_bias_raw_override: Path | None = None,
    folded_bias_byte_offset: int = 0,
    mask_word_overrides: dict[int, int] | None = None,
) -> dict[str, str]:
    prepared = artifact / "prepared_state"
    file_map = {
        "activation_offsets": "activation_table.raw",
        "output_offsets": "output_table.raw",
    }
    checksums: dict[str, str] = {}
    activation_source = activation_raw_override if activation_raw_override is not None else prepared / "activation.raw"
    if not activation_source.is_file():
        raise FileNotFoundError(f"missing activation raw payload: {activation_source}")
    checksums["activation"] = fnv1a32(activation_source.read_bytes())
    for name, filename in file_map.items():
        path = prepared / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared-state file: {path}")
        checksums[name] = fnv1a32(path.read_bytes())
    packed_weight_source = (
        packed_weight_raw_override if packed_weight_raw_override is not None else prepared / "packed_weight.raw"
    )
    packed_weight, _ = shifted_pointer_storage(packed_weight_source.read_bytes(), packed_weight_byte_offset)
    checksums["packed_weight"] = fnv1a32(packed_weight)
    folded_bias_source = (
        folded_bias_raw_override if folded_bias_raw_override is not None else prepared / "folded_bias.raw"
    )
    folded_bias, _ = shifted_pointer_storage(folded_bias_source.read_bytes(), folded_bias_byte_offset)
    checksums["folded_bias"] = fnv1a32(folded_bias)
    output_seed = (prepared / "output_surface.raw").read_bytes()
    if output_seed_mode == "prepared":
        checksums["output_seed"] = fnv1a32(output_seed)
    elif output_seed_mode == "zero":
        checksums["output_seed"] = fnv1a32(bytes(len(output_seed)))
    elif output_seed_mode == "ff":
        checksums["output_seed"] = fnv1a32(bytes([0xFF]) * len(output_seed))
    else:
        raise ValueError(f"unsupported output seed mode: {output_seed_mode}")
    mask_control = (prepared / "mask_control.raw").read_bytes()
    mask_words = [
        int.from_bytes(mask_control[offset : offset + 4], "little")
        for offset in range(0, len(mask_control), 4)
    ]
    for index, value in (mask_word_overrides or {}).items():
        if index < 0 or index >= len(mask_words):
            raise ValueError(f"mask override index out of range: {index}")
        mask_words[index] = int(value)
    checksums["mask_control"] = fnv1a32(
        b"".join(int(value).to_bytes(4, "little") for value in mask_words)
    )
    return checksums


def parse_u32_override(item: str) -> tuple[int, int]:
    raw_index, raw_value = item.split("=", 1)
    index = int(raw_index, 0)
    value = int(raw_value, 0)
    if index < 0:
        raise ValueError("INDEX must be non-negative")
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError("VALUE must fit in u32")
    return index, value


def expected_call_abi_scalars(
    artifact: Path,
    extra_word_overrides: dict[int, int] | None = None,
    mask_word_overrides: dict[int, int] | None = None,
) -> dict[str, int]:
    abi = json.loads((artifact / "analysis" / "abi_manifest.json").read_text(encoding="utf-8"))
    out_desc = abi["hexagon_call_abi"]["out_desc_fields"]
    act_desc = abi["hexagon_call_abi"]["act_desc_fields"]
    extra = list(abi["hexagon_call_abi"]["extra_param_words"])
    for index, value in (extra_word_overrides or {}).items():
        if index < 0:
            raise ValueError(f"extra override index out of range: {index}")
        while index >= len(extra):
            extra.append(0)
        extra[index] = int(value)
    mask_words = json.loads((artifact / "analysis" / "abi_manifest.json").read_text(encoding="utf-8"))[
        "hexagon_call_abi"
    ]["mask_desc"]["native_generator"]["emulated_words"]
    for index, value in (mask_word_overrides or {}).items():
        if index < 0 or index >= len(mask_words):
            raise ValueError(f"mask override index out of range: {index}")
        mask_words[index] = int(value)
    return {
        "out_table_stride_dwords": int(out_desc["out_table_stride_dwords"]),
        "out_y_stride_words": int(out_desc["out_y_stride_words"]),
        "n_tiles_pow2": int(out_desc["n_tiles_pow2"]),
        "m_total_minus_step": int(out_desc["m_total_minus_step"]),
        "k_total_bytes": int(out_desc["k_total_bytes"]),
        "n_act_pairs": int(act_desc["n_act_pairs"]),
        "act_table_y_stride_words": int(act_desc["act_table_y_stride_words"]),
        "extra0": int(extra[0]),
        "extra1": int(extra[1]),
        "extra2": int(extra[2]),
        "extra3": int(extra[3]) if len(extra) > 3 else 0,
        "mask6": int(mask_words[6]),
    }


def expected_vtcm_offsets() -> dict[str, int]:
    return {
        "activation": 0x00000,
        "packed_weight": 0x20000,
        "folded_bias": 0x40000,
        "output_surface": 0x50000,
        "public_output": 0x80000,
        "activation_table": 0x70000,
        "output_table": 0x71000,
    }


def expected_hnh_path(
    artifact: Path,
    kernel_entry: str,
    extra_word_overrides: dict[int, int] | None = None,
    mask_word_overrides: dict[int, int] | None = None,
) -> dict[str, int | str]:
    abi = json.loads((artifact / "analysis" / "abi_manifest.json").read_text(encoding="utf-8"))
    out_desc = abi["hexagon_call_abi"]["out_desc_fields"]
    act_desc = abi["hexagon_call_abi"]["act_desc_fields"]
    extra = list(abi["hexagon_call_abi"]["extra_param_words"])
    for index, value in (extra_word_overrides or {}).items():
        if index < 0:
            raise ValueError(f"extra override index out of range: {index}")
        while index >= len(extra):
            extra.append(0)
        extra[index] = int(value)
    mask_words = list(abi["hexagon_call_abi"]["mask_desc"]["native_generator"]["emulated_words"])
    for index, value in (mask_word_overrides or {}).items():
        if index < 0 or index >= len(mask_words):
            raise ValueError(f"mask override index out of range: {index}")
        mask_words[index] = int(value)
    n_act_pairs = int(act_desc["n_act_pairs"])
    k_total_bytes = int(out_desc["k_total_bytes"])
    k_groups = (k_total_bytes + 31) // 32
    return {
        "kernel_entry": kernel_entry,
        "main_path": int(n_act_pairs > 1),
        "odd_act_pair": int(n_act_pairs & 1),
        "k_groups": k_groups,
        "k_pair_groups": k_groups // 2,
        "final_reduce": int((k_groups & 1) != 0),
        "drain_count": int(extra[0]),
        "cvt0": int(extra[1]),
        "cvt1": int(extra[2]),
        "mask12": int(mask_words[12]),
        "mask14_ptr_patched": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--measure-repeats", type=int, default=1)
    parser.add_argument(
        "--chain-steps",
        type=int,
        default=8,
        help="Number of direct HMX body calls to run; use 1 for single-call step0 diagnostics",
    )
    parser.add_argument(
        "--no-pre-clear-acc",
        action="store_true",
        help="Do not emit the tutorial entry accumulator clear before the HMX body call",
    )
    parser.add_argument(
        "--native-wrapper-prefetch",
        action="store_true",
        help="Prefetch activation/output pointer tables before each direct HMX body call",
    )
    parser.add_argument(
        "--preload-hmx-identity-bias",
        action="store_true",
        help="Preload tutorial-style HMX scale=1/bias=0 state with mxmem2.bias before the direct body loop",
    )
    parser.add_argument(
        "--output-seed-mode",
        choices=("prepared", "zero", "ff"),
        default="prepared",
        help="Initial internal output surface before each direct body call",
    )
    parser.add_argument(
        "--activation-raw-override",
        type=Path,
        help="Diagnostic raw activation payload override",
    )
    parser.add_argument(
        "--folded-bias-byte-offset",
        type=int,
        default=0,
        help="Diagnostic byte offset added to the folded-bias/control pointer; negative values add a guard prefix",
    )
    parser.add_argument(
        "--packed-weight-byte-offset",
        type=int,
        default=0,
        help="Diagnostic byte offset added to the packed-weight pointer; negative values add a guard prefix",
    )
    parser.add_argument(
        "--packed-weight-raw-override",
        type=Path,
        help="Diagnostic raw packed-weight/window payload override",
    )
    parser.add_argument(
        "--folded-bias-raw-override",
        type=Path,
        help="Diagnostic raw folded-bias/control payload override",
    )
    parser.add_argument(
        "--extra-word-override",
        action="append",
        default=[],
        metavar="INDEX=VALUE",
        help="Override one extra_param u32 word for single-call HNH/cvt diagnostics",
    )
    parser.add_argument(
        "--mask-word-override",
        action="append",
        default=[],
        metavar="INDEX=VALUE",
        help="Override one mask/control u32 word for direct HMX ABI diagnostics",
    )
    parser.add_argument(
        "--kernel-entry",
        choices=("deep", "wrapper", "wrapper_nondeep", "split_n128"),
        default="deep",
        help=(
            "Use the direct deep-body entry, a native wrapper-entry shim, "
            "or the two-call N128 split pattern from the old custom wrapper"
        ),
    )
    parser.add_argument(
        "--step0-native-raw",
        type=Path,
        default=DEFAULT_CHAIN1_NATIVE_RAW,
        help="Optional native chain1 raw output used to compare the first direct body step",
    )
    parser.add_argument(
        "--u16-sample-index",
        action="append",
        default=[],
        type=int,
        metavar="INDEX",
        help="Log one explicit public-output u16 sample against the native oracle",
    )
    parser.add_argument(
        "--internal-u16-sample",
        action="append",
        default=[],
        metavar="BLOCK:INDEX",
        help="Log one explicit internal-output u16 sample against the mapped native oracle",
    )
    parser.add_argument(
        "--focused-sample-log",
        action="store_true",
        help="Only emit requested u16 samples plus required result/perf lines after the body call",
    )
    args = parser.parse_args()
    if args.measure_repeats <= 0:
        parser.error("--measure-repeats must be positive")
    if args.chain_steps <= 0 or args.chain_steps > 8:
        parser.error("--chain-steps must be in [1, 8]")
    u16_sample_indices: list[int] = []
    for index in args.u16_sample_index:
        if index < 0:
            parser.error(f"invalid --u16-sample-index {index!r}: INDEX must be non-negative")
        u16_sample_indices.append(index)
    internal_u16_samples: list[tuple[int, int]] = []
    for item in args.internal_u16_sample:
        try:
            internal_u16_samples.append(parse_internal_u16_sample(item))
        except ValueError as exc:
            parser.error(f"invalid --internal-u16-sample {item!r}: {exc}")
    extra_word_overrides: dict[int, int] = {}
    for item in args.extra_word_override:
        try:
            index, value = parse_u32_override(item)
        except ValueError as exc:
            parser.error(f"invalid --extra-word-override {item!r}: {exc}")
        extra_word_overrides[index] = value
    mask_word_overrides: dict[int, int] = {}
    for item in args.mask_word_override:
        try:
            index, value = parse_u32_override(item)
        except ValueError as exc:
            parser.error(f"invalid --mask-word-override {item!r}: {exc}")
        mask_word_overrides[index] = value

    artifact = args.artifact.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    step0_native_raw_path = args.step0_native_raw.resolve() if args.step0_native_raw else None
    activation_raw_override = (
        args.activation_raw_override.resolve() if args.activation_raw_override else None
    )
    packed_weight_raw_override = (
        args.packed_weight_raw_override.resolve() if args.packed_weight_raw_override else None
    )
    folded_bias_raw_override = (
        args.folded_bias_raw_override.resolve() if args.folded_bias_raw_override else None
    )
    if activation_raw_override is not None and not activation_raw_override.is_file():
        raise FileNotFoundError(f"missing activation raw override: {activation_raw_override}")
    if packed_weight_raw_override is not None and not packed_weight_raw_override.is_file():
        raise FileNotFoundError(f"missing packed weight raw override: {packed_weight_raw_override}")
    if folded_bias_raw_override is not None and not folded_bias_raw_override.is_file():
        raise FileNotFoundError(f"missing folded bias raw override: {folded_bias_raw_override}")
    step0_native_raw = None
    if step0_native_raw_path is not None:
        if not step0_native_raw_path.is_file():
            raise FileNotFoundError(f"missing step0 native raw: {step0_native_raw_path}")
        step0_native_raw = step0_native_raw_path.read_bytes()
    source, binary = build_device_so(
        "w4a16",
        artifact,
        out_dir,
        measure_repeats=args.measure_repeats,
        native_wrapper_prefetch=args.native_wrapper_prefetch,
        pre_clear_acc=not args.no_pre_clear_acc,
        activation_raw_override=activation_raw_override,
        packed_weight_byte_offset=args.packed_weight_byte_offset,
        packed_weight_raw_override=packed_weight_raw_override,
        folded_bias_raw_override=folded_bias_raw_override,
        folded_bias_byte_offset=args.folded_bias_byte_offset,
        source_stem="w4a16_qnn_kernel_tutorial",
        step0_native_raw=step0_native_raw,
        chain_steps=args.chain_steps,
        output_seed_mode=args.output_seed_mode,
        extra_word_overrides=extra_word_overrides,
        mask_word_overrides=mask_word_overrides,
        kernel_entry=args.kernel_entry,
        preload_hmx_identity_bias=args.preload_hmx_identity_bias,
        u16_sample_indices=u16_sample_indices,
        internal_u16_samples=internal_u16_samples,
        focused_sample_log=args.focused_sample_log,
    )
    manifest = {
        "schema": "w4a16_qnn_kernel_tutorial_build.v1",
        "artifact": rel(artifact),
        "source": rel(source),
        "binary": rel(binary),
        "qnn_runtime_used": False,
        "kernel_body": (
            "hm_w4a16_v73wrapper_entry_kernel"
            if args.kernel_entry == "wrapper"
            else "hm_w4a16_v73wrapper_nondeep_kernel"
            if args.kernel_entry == "wrapper_nondeep"
            else "hm_w4a16_v73deep_kernel"
        ),
        "kernel_entry": args.kernel_entry,
        "entry_style": "run_main_on_hexagon_hap_vtcm_hmx_lock",
        "pre_clear_acc": not args.no_pre_clear_acc,
        "native_wrapper_prefetch": args.native_wrapper_prefetch,
        "preload_hmx_identity_bias": args.preload_hmx_identity_bias,
        "output_seed_mode": args.output_seed_mode,
        "activation_raw_override": (
            rel(activation_raw_override) if activation_raw_override is not None else None
        ),
        "packed_weight_raw_override": (
            rel(packed_weight_raw_override) if packed_weight_raw_override is not None else None
        ),
        "folded_bias_raw_override": (
            rel(folded_bias_raw_override) if folded_bias_raw_override is not None else None
        ),
        "packed_weight_byte_offset": args.packed_weight_byte_offset,
        "folded_bias_byte_offset": args.folded_bias_byte_offset,
        "extra_word_overrides": {str(k): v for k, v in sorted(extra_word_overrides.items())},
        "mask_word_overrides": {str(k): v for k, v in sorted(mask_word_overrides.items())},
        "u16_sample_indices": u16_sample_indices,
        "internal_u16_samples": [
            {"block": block, "index": index} for block, index in internal_u16_samples
        ],
        "focused_sample_log": args.focused_sample_log,
        "prepared_state_carrier": "owned_vtcm_separate_buffers",
        "measure_repeats": args.measure_repeats,
        "chain_steps": args.chain_steps,
        "final_native_oracle": "chain1" if args.chain_steps == 1 else "chain8",
        "expected_prepared_state_checksums": prepared_checksum_manifest(
            artifact,
            args.output_seed_mode,
            activation_raw_override,
            args.packed_weight_byte_offset,
            packed_weight_raw_override,
            folded_bias_raw_override,
            args.folded_bias_byte_offset,
            mask_word_overrides,
        ),
        "expected_call_abi_scalars": expected_call_abi_scalars(
            artifact, extra_word_overrides, mask_word_overrides
        ),
        "dynamic_mask14_pointer_patch": True,
        "expected_hnh_path": expected_hnh_path(
            artifact,
            args.kernel_entry,
            extra_word_overrides,
            mask_word_overrides,
        ),
        "expected_vtcm_offsets": expected_vtcm_offsets(),
        "step_trace_enabled": True,
        "expected_chain_steps": args.chain_steps,
    }
    if step0_native_raw_path is not None:
        manifest["expected_step0_native_raw"] = {
            "path": rel(step0_native_raw_path),
            "bytes": len(step0_native_raw or b""),
            "checksum": fnv1a32(step0_native_raw or b""),
        }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"w4a16 tutorial wrapper source: {source}")
    print(f"w4a16 tutorial wrapper binary: {binary}")
    print(f"w4a16 tutorial wrapper manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
