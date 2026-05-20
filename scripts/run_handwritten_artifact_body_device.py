#!/usr/bin/env python3
"""Run a prepared handwritten artifact through an owned HMX body on device.

This is the device-backed companion to run_handwritten_artifact_body_sim.py.
It builds a temporary CDSP shared object, embeds the prepared artifact bytes and
the QNN Native raw output, runs the body via run_main_on_hexagon, and parses the
DSP FARF evidence into JSON.  Exact families return success only when device
output is byte-identical to the matched native raw output.  Inexact diagnostic
families can be run with ``--allow-inexact`` to require device entry evidence
without promoting the result to accepted output exactness.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

try:
    from run_handwritten_artifact_body_sim import c_array, c_u32_array, fnv1a32, pad, read_u32_le
except ModuleNotFoundError:
    from scripts.run_handwritten_artifact_body_sim import c_array, c_u32_array, fnv1a32, pad, read_u32_le


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "handwritten_hmx_matmul"
SDK = ROOT / "tools" / "hexagon-sdk"
RUN_MAIN = SDK / "libs" / "run_main_on_hexagon" / "ship" / "android_aarch64" / "run_main_on_hexagon"
RUN_MAIN_SKEL = (
    SDK
    / "libs"
    / "run_main_on_hexagon"
    / "ship"
    / "hexagon_toolv87_v75"
    / "librun_main_on_hexagon_skel.so"
)


FAMILY_CONFIG = {
    "u8i8": {
        "header": "handwritten_hmx_u8i8_kernel.h",
        "out_desc": "HmU8I8OutDesc",
        "act_desc": "HmU8I8ActDesc",
        "mask_desc": "HmU8I8MaskDesc",
        "function": "hm_u8i8_v73deep_kernel",
        "output_mapping": "a8_crouton8",
    },
    "w4a8": {
        "header": "handwritten_hmx_w4a8_kernel.h",
        "out_desc": "HmW4A8OutDesc",
        "act_desc": "HmW4A8ActDesc",
        "mask_desc": "HmW4A8MaskDesc",
        "function": "hm_w4a8_v73deep_kernel",
        "output_mapping": "a8_crouton8",
    },
    "w8a16": {
        "header": "handwritten_hmx_w8a16_kernel.h",
        "out_desc": "HmW8A16OutDesc",
        "act_desc": "HmW8A16ActDesc",
        "mask_desc": "HmW8A16MaskDesc",
        "function": "hm_w8a16_v75deep_kernel",
        "output_mapping": "a16_crouton16_row4",
    },
    "w4a16": {
        "header": "handwritten_hmx_w4a16_kernel.h",
        "out_desc": "HmW4A16OutDesc",
        "act_desc": "HmW4A16ActDesc",
        "mask_desc": "HmW4A16MaskDesc",
        "function": "hm_w4a16_v73deep_kernel",
        "output_mapping": "a16_crouton16_row4",
    },
}


def tool(name: str) -> Path:
    candidates = sorted((SDK / "tools" / "HEXAGON_Tools").glob(f"*/Tools/bin/{name}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing tool: {name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ssh_cat_to(device: str, remote_path: str, data: bytes) -> None:
    subprocess.run(["ssh", device, f"cat > {remote_path}"], input=data, check=True)


def ssh_text(device: str, command: str, *, timeout: int | None = None) -> str:
    return subprocess.run(
        ["ssh", device, command],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    ).stdout


def public_raw_to_internal(raw: bytes, mapping: str, m: int, n: int) -> bytes:
    if mapping == "a8_crouton8":
        out = bytearray(len(raw))
        phase_block_bytes = (m // 32) * 256
        for row8_group in range(4):
            for nt in range(n // 32):
                block_base = (row8_group * (n // 32) + nt) * phase_block_bytes
                for m32_group in range(m // 32):
                    for row_sub in range(8):
                        row = m32_group * 32 + row8_group * 8 + row_sub
                        row_base = row * n + nt * 32
                        for col_word in range(8):
                            src_base = row_base + col_word * 4
                            dst_base = block_base + (m32_group * 64 + row_sub * 8 + col_word) * 4
                            out[dst_base : dst_base + 4] = raw[src_base : src_base + 4]
        return bytes(out)
    if mapping == "a16_crouton16_row4":
        out = bytearray(len(raw))
        phase_block_bytes = (m // 32) * 256
        for row4_phase in range(8):
            for nt in range(n // 32):
                block_base = (row4_phase * (n // 32) + nt) * phase_block_bytes
                for m32_group in range(m // 32):
                    for row_pair in range(2):
                        row0 = m32_group * 32 + row4_phase * 4 + row_pair * 2
                        row1 = row0 + 1
                        row0_base = row0 * n * 2 + nt * 64
                        row1_base = row1 * n * 2 + nt * 64
                        pair_base = block_base + (m32_group * 2 + row_pair) * 128
                        for col in range(32):
                            dst_base = pair_base + col * 4
                            out[dst_base : dst_base + 2] = raw[row0_base + col * 2 : row0_base + col * 2 + 2]
                            out[dst_base + 2 : dst_base + 4] = raw[row1_base + col * 2 : row1_base + col * 2 + 2]
        return bytes(out)
    raise ValueError(f"unsupported internal output mapping: {mapping}")


def shifted_pointer_storage(source: bytes, pointer_offset: int) -> tuple[bytes, int]:
    """Return storage bytes and a non-negative call offset for pointer probes."""

    prefix = max(0, -pointer_offset)
    suffix = max(0, pointer_offset)
    call_offset = prefix + pointer_offset
    return bytes(prefix) + source + bytes(suffix), call_offset


def generate_source(
    family: str,
    artifact: Path,
    abi: dict,
    oracle: dict,
    native_raw: bytes,
    measure_repeats: int,
    native_wrapper_prefetch: bool = False,
    pre_clear_acc: bool = False,
    activation_raw_override: Path | None = None,
    packed_weight_byte_offset: int = 0,
    packed_weight_raw_override: Path | None = None,
    folded_bias_raw_override: Path | None = None,
    folded_bias_byte_offset: int = 0,
    descriptor_carrier: str = "separate",
    step0_native_raw: bytes | None = None,
    chain_steps: int | None = None,
    output_seed_mode: str = "prepared",
    extra_word_overrides: dict[int, int] | None = None,
    mask_word_overrides: dict[int, int] | None = None,
    kernel_entry: str = "deep",
    preload_hmx_identity_bias: bool = False,
    u16_sample_indices: list[int] | None = None,
    internal_u16_samples: list[tuple[int, int]] | None = None,
    focused_sample_log: bool = False,
    native_internal_raw_override: bytes | None = None,
    public_output_layout: str = "default",
) -> str:
    config = FAMILY_CONFIG[family]
    if kernel_entry not in ("deep", "wrapper", "wrapper_nondeep", "split_n128"):
        raise ValueError(f"{family}: unsupported kernel_entry {kernel_entry!r}")
    if kernel_entry != "deep" and family != "w4a16":
        raise ValueError("--kernel-entry variants are only valid for w4a16")
    call_function = config["function"]
    if family == "w4a16" and kernel_entry == "wrapper":
        call_function = "hm_w4a16_v73wrapper_entry_kernel"
    if family == "w4a16" and kernel_entry == "wrapper_nondeep":
        call_function = "hm_w4a16_v73wrapper_nondeep_kernel"
    prepared = artifact / "prepared_state"
    activation_source = activation_raw_override if activation_raw_override is not None else prepared / "activation.raw"
    activation = activation_source.read_bytes()
    packed_weight_source_path = (
        packed_weight_raw_override if packed_weight_raw_override is not None else prepared / "packed_weight.raw"
    )
    packed_weight_source = packed_weight_source_path.read_bytes()
    packed_weight_storage, packed_weight_call_offset = shifted_pointer_storage(
        packed_weight_source, packed_weight_byte_offset
    )
    packed_weight = pad(packed_weight_storage, len(packed_weight_storage))
    folded_bias_source = (
        folded_bias_raw_override if folded_bias_raw_override is not None else prepared / "folded_bias.raw"
    )
    folded_bias_storage, folded_bias_call_offset = shifted_pointer_storage(
        folded_bias_source.read_bytes(), folded_bias_byte_offset
    )
    folded_bias = pad(folded_bias_storage, len(folded_bias_storage))
    output_surface = pad((prepared / "output_surface.raw").read_bytes(), 2048)
    mask_control_words = read_u32_le(prepared / "mask_control.raw")
    for index, value in (mask_word_overrides or {}).items():
        if index < 0 or index >= len(mask_control_words):
            raise ValueError(f"{family}: mask word override index out of range: {index}")
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"{family}: mask word override value must fit in u32: {value}")
        mask_control_words[index] = int(value)
    mask_control = pad(b"".join(int(v).to_bytes(4, "little") for v in mask_control_words), 64)
    activation_offsets = read_u32_le(prepared / "activation_table.raw")
    output_offsets = read_u32_le(prepared / "output_table.raw")
    out_desc = abi["hexagon_call_abi"]["out_desc_fields"]
    act_desc = abi["hexagon_call_abi"]["act_desc_fields"]
    extra = list(abi["hexagon_call_abi"]["extra_param_words"])
    for index, value in (extra_word_overrides or {}).items():
        if index < 0:
            raise ValueError(f"{family}: extra word override index out of range: {index}")
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"{family}: extra word override value must fit in u32: {value}")
        while index >= len(extra):
            extra.append(0)
        extra[index] = int(value)
    act_entries = int(abi["activation_table"]["entry_count"])
    out_entries = int(abi["output_table"]["entry_count"])
    m, _, n = [int(v) for v in oracle["shape_mkn"]]
    phase_block_bytes = (m // 32) * 256
    chain = int(oracle["chain"])
    run_chain_steps = chain if chain_steps is None else int(chain_steps)
    if run_chain_steps <= 0 or run_chain_steps > chain:
        raise ValueError(f"{family}: chain_steps must be in [1, {chain}], got {run_chain_steps}")
    extra_init = ", ".join(f"{int(v)}u" for v in extra)
    extra_arg_exprs = [f"extra[{index}]" if index < len(extra) else "0u" for index in range(4)]
    if descriptor_carrier not in ("separate", "w4a16_hmxi_private_payload"):
        raise ValueError(f"{family}: unsupported descriptor carrier {descriptor_carrier!r}")
    if descriptor_carrier != "separate" and family != "w4a16":
        raise ValueError("--descriptor-carrier is only valid with --family w4a16")
    if output_seed_mode not in ("prepared", "zero", "ff"):
        raise ValueError(f"{family}: unsupported output seed mode {output_seed_mode!r}")
    if public_output_layout not in ("default", "native_out_block"):
        raise ValueError(f"{family}: unsupported public output layout {public_output_layout!r}")
    if public_output_layout == "native_out_block" and config["output_mapping"] != "a16_crouton16_row4":
        raise ValueError("--public-output-layout native_out_block is only valid for A16 crouton16 outputs")
    public_deblock_function = (
        "deblock_public_native_out_block"
        if public_output_layout == "native_out_block"
        else "deblock_public"
    )
    hmxi_private_payload_carrier = descriptor_carrier == "w4a16_hmxi_private_payload"
    final_native_raw = step0_native_raw if run_chain_steps == 1 and step0_native_raw is not None else native_raw
    native_internal_raw = (
        native_internal_raw_override
        if native_internal_raw_override is not None
        else public_raw_to_internal(final_native_raw, config["output_mapping"], m, n)
    )
    if len(native_internal_raw) != len(final_native_raw):
        raise ValueError(
            f"{family}: native internal raw size {len(native_internal_raw)} "
            f"does not match output size {len(final_native_raw)}"
        )
    hmx_identity_bias = bytes([0x00, 0x3C]) * 64 + bytes(128)
    selected_u16_indices = list(u16_sample_indices or [])
    selected_internal_u16_samples = list(internal_u16_samples or [])
    for index in selected_u16_indices:
        if index < 0:
            raise ValueError(f"{family}: selected u16 sample index must be non-negative: {index}")
    for block, index in selected_internal_u16_samples:
        if block < 0 or index < 0:
            raise ValueError(
                f"{family}: selected internal u16 sample block/index must be non-negative: {block}:{index}"
            )
    selected_u16_init = ", ".join(f"{int(index)}u" for index in selected_u16_indices)
    selected_internal_init = ", ".join(
        f"{{{int(block)}u, {int(index)}u}}" for block, index in selected_internal_u16_samples
    )
    native_checksum = fnv1a32(final_native_raw)
    step0_native_checksum = fnv1a32(step0_native_raw) if step0_native_raw is not None else None
    if native_wrapper_prefetch and family != "w4a16":
        raise ValueError("--native-wrapper-prefetch is only valid for w4a16")
    split_n128_entry = family == "w4a16" and kernel_entry == "split_n128"
    if split_n128_entry:
        if n != 256 or int(oracle["shape_mkn"][1]) != 256 or m != 256:
            raise ValueError("--kernel-entry=split_n128 is currently only wired for canonical 256^3 W4A16")
    if config["output_mapping"] == "a8_crouton8":
        deblock_helper = [
            "static void deblock_public(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row8_group = 0; row8_group < 4u; ++row8_group) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row8_group * "
            + f"{n // 32}u"
            + f" + nt) * {phase_block_bytes}u);",
            f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "        for (uint32_t row_sub = 0; row_sub < 8u; ++row_sub) {",
            "          uint32_t row = m32_group * 32u + row8_group * 8u + row_sub;",
            f"          uint8_t *dst_row = dst + row * {n}u + nt * 32u;",
            "          for (uint32_t col_word = 0; col_word < 8u; ++col_word) {",
            "            const uint8_t *src_word = block + (m32_group * 64u + row_sub * 8u + col_word) * 4u;",
            "            for (uint32_t b = 0; b < 4u; ++b) dst_row[col_word * 4u + b] = src_word[b];",
            "          }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
        ]
    elif config["output_mapping"] == "a16_crouton16_row4":
        deblock_helper = [
            "static void deblock_public(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * "
            + f"{n // 32}u"
            + f" + nt) * {phase_block_bytes}u);",
            f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "        for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
            "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
            "          uint32_t row1 = row0 + 1u;",
            f"          uint8_t *dst0 = dst + row0 * {n * 2}u + nt * 64u;",
            f"          uint8_t *dst1 = dst + row1 * {n * 2}u + nt * 64u;",
            "          const uint8_t *src_pair = block + (m32_group * 2u + row_pair) * 128u;",
            "          for (uint32_t col = 0; col < 32u; ++col) {",
            "            const uint8_t *word = src_pair + col * 4u;",
            "            dst0[col * 2u + 0u] = word[0];",
            "            dst0[col * 2u + 1u] = word[1];",
            "            dst1[col * 2u + 0u] = word[2];",
            "            dst1[col * 2u + 1u] = word[3];",
            "          }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "static void deblock_public_col16_swap(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * "
            + f"{n // 32}u"
            + f" + nt) * {phase_block_bytes}u);",
            f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "        for (uint32_t row_pair = 0u; row_pair < 2u; ++row_pair) {",
            "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
            "          uint32_t row1 = row0 + 1u;",
            f"          uint8_t *dst0 = dst + row0 * {n * 2}u + nt * 64u;",
            f"          uint8_t *dst1 = dst + row1 * {n * 2}u + nt * 64u;",
            "          const uint8_t *src_pair = block + (m32_group * 2u + row_pair) * 128u;",
            "          for (uint32_t col = 0u; col < 32u; ++col) {",
            "            uint32_t dst_col = col ^ 16u;",
            "            const uint8_t *word = src_pair + col * 4u;",
            "            dst0[dst_col * 2u + 0u] = word[0];",
            "            dst0[dst_col * 2u + 1u] = word[1];",
            "            dst1[dst_col * 2u + 0u] = word[2];",
            "            dst1[dst_col * 2u + 1u] = word[3];",
            "          }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "static void deblock_public_pair_swap(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * "
            + f"{n // 32}u"
            + f" + nt) * {phase_block_bytes}u);",
            f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "        for (uint32_t row_pair = 0u; row_pair < 2u; ++row_pair) {",
            "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
            "          uint32_t row1 = row0 + 1u;",
            f"          uint8_t *dst0 = dst + row0 * {n * 2}u + nt * 64u;",
            f"          uint8_t *dst1 = dst + row1 * {n * 2}u + nt * 64u;",
            "          const uint8_t *src_pair = block + (m32_group * 2u + row_pair) * 128u;",
            "          for (uint32_t col = 0u; col < 32u; ++col) {",
            "            const uint8_t *word = src_pair + col * 4u;",
            "            dst0[col * 2u + 0u] = word[2];",
            "            dst0[col * 2u + 1u] = word[3];",
            "            dst1[col * 2u + 0u] = word[0];",
            "            dst1[col * 2u + 1u] = word[1];",
            "          }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "static void deblock_public_pair_major(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * "
            + f"{n // 32}u"
            + f" + nt) * {phase_block_bytes}u);",
            "      for (uint32_t row_pair = 0u; row_pair < 2u; ++row_pair) {",
            f"        for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "          uint32_t row0 = m32_group * 32u + row_pair * 16u + row4_phase * 2u;",
            "          uint32_t row1 = row0 + 1u;",
            f"          uint8_t *dst0 = dst + row0 * {n * 2}u + nt * 64u;",
            f"          uint8_t *dst1 = dst + row1 * {n * 2}u + nt * 64u;",
            "          const uint8_t *src_pair = block + (row_pair * "
            + f"{m // 32}u"
            + " + m32_group) * 128u;",
            "          for (uint32_t col = 0u; col < 32u; ++col) {",
            "            const uint8_t *word = src_pair + col * 4u;",
            "            dst0[col * 2u + 0u] = word[0];",
            "            dst0[col * 2u + 1u] = word[1];",
            "            dst1[col * 2u + 0u] = word[2];",
            "            dst1[col * 2u + 1u] = word[3];",
            "          }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "static void deblock_public_native_out_block(uint8_t *dst, const uint8_t *src) {",
            f"  for (uint32_t m32_group = 0u; m32_group < {m // 32}u; ++m32_group) {{",
            f"    for (uint32_t nt = 0u; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + (m32_group * "
            + f"{n // 32}u"
            + " + nt) * 2048u;",
            "      for (uint32_t col_pair = 0u; col_pair < 16u; ++col_pair) {",
            "        for (uint32_t row_sub = 0u; row_sub < 32u; ++row_sub) {",
            "          const uint8_t *word = block + (col_pair * 32u + row_sub) * 4u;",
            "          uint32_t row = m32_group * 32u + row_sub;",
            "          uint32_t col = nt * 32u + col_pair * 2u;",
            f"          uint8_t *dst0 = dst + (row * {n}u + col) * 2u;",
            "          dst0[0] = word[0];",
            "          dst0[1] = word[1];",
            "          dst0[2] = word[2];",
            "          dst0[3] = word[3];",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "static uint32_t exact_u16_count(const uint8_t *actual, const uint8_t *expected, uint32_t n) {",
            "  uint32_t total = n / 2u;",
            "  uint32_t exact = 0u;",
            "  for (uint32_t i = 0u; i < total; ++i) {",
            "    if (load_u16_le(actual + i * 2u) == load_u16_le(expected + i * 2u)) ++exact;",
            "  }",
            "  return exact;",
            "}",
            "static void log_alt_layout_summary(const char *name, uint8_t *scratch, const uint8_t *internal, const uint8_t *expected, void (*deblock)(uint8_t *, const uint8_t *)) {",
            "  fill_bytes(scratch, 0u, sizeof(k_output_seed));",
            "  deblock(scratch, internal);",
            "  uint32_t first = 0u;",
            "  uint32_t diffs = byte_diff(scratch, expected, sizeof(k_output_seed), &first);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " alt_layout family={family} name=%s checksum=0x%08x diffs=%u first_diff=%d exact_u16=%u",',
            "       name, checksum(scratch, sizeof(k_output_seed)), diffs,",
            "       first == 0xffffffffu ? -1 : (int)first,",
            "       exact_u16_count(scratch, expected, sizeof(k_output_seed)));",
            "}",
            "static void log_alt_layout_summaries(uint8_t *scratch, const uint8_t *internal, const uint8_t *expected) {",
            '  log_alt_layout_summary("crouton16_row4", scratch, internal, expected, deblock_public);',
            '  log_alt_layout_summary("crouton16_col16_swap", scratch, internal, expected, deblock_public_col16_swap);',
            '  log_alt_layout_summary("crouton16_pair_swap", scratch, internal, expected, deblock_public_pair_swap);',
            '  log_alt_layout_summary("crouton16_pair_major", scratch, internal, expected, deblock_public_pair_major);',
            '  log_alt_layout_summary("native_out_block", scratch, internal, expected, deblock_public_native_out_block);',
            "}",
        ]
    else:
        raise ValueError(f"{family}: unsupported device output mapping {config['output_mapping']}")
    wrapper_prefetch_helper: list[str] = []
    wrapper_prefetch_call: list[str] = []
    if native_wrapper_prefetch:
        wrapper_prefetch_helper = [
            "static inline void native_dcfetch(const void *ptr) {",
            '  __asm__ volatile("dcfetch(%0+#0)" :: "r"(ptr));',
            "}",
            "static void native_wrapper_prefetch_tables(",
            "    const void *act_table, uint32_t act_entries,",
            "    const void *out_table, uint32_t out_entries) {",
            "  const uint8_t *act = (const uint8_t *)act_table;",
            "  for (uint32_t off = 0u; off < act_entries * 4u; off += 64u) native_dcfetch(act + off);",
            "  const uint8_t *outp = (const uint8_t *)out_table;",
            "  for (uint32_t off = 0u; off < out_entries * 4u; off += 64u) native_dcfetch(outp + off);",
            "}",
        ]
        wrapper_prefetch_call = [
            f"    native_wrapper_prefetch_tables(act_table, {act_entries}u, out_table, {out_entries}u);"
        ]
    k_tiles = int(oracle["shape_mkn"][1]) // 32
    n_tiles = n // 32
    out_table_stride_words = int(out_desc["out_table_stride_dwords"])
    row4_groups = out_entries // out_table_stride_words
    kernel_call_lines = [
        f"{call_function}(out_desc_ptr, act_desc_ptr, weight + {packed_weight_call_offset}u, bias + {folded_bias_call_offset}u, (const {config['mask_desc']} *)mask_words, extra);"
    ]
    if split_n128_entry:
        split_tiles = 4
        split_count = n_tiles // split_tiles
        split_weight_bytes = k_tiles * split_tiles * 512
        split_bias_bytes = split_tiles * 256
        split_entries = row4_groups * out_table_stride_words
        kernel_call_lines = [
            f"int32_t out_tbl_split[{split_entries}] __attribute__((aligned(64)));",
            f"for (uint32_t split = 0u; split < {split_count}u; ++split) {{",
            f"  uint32_t rotate = ((split + 1u) * {split_tiles}u) % {n_tiles}u;",
            f"  for (uint32_t row4 = 0u; row4 < {row4_groups}u; ++row4) {{",
            f"    for (uint32_t nt = 0u; nt < {out_table_stride_words}u; ++nt) {{",
            "      uint32_t src_nt = nt;",
            f"      if (nt < {n_tiles}u) src_nt = (nt + rotate) % {n_tiles}u;",
            f"      out_tbl_split[row4 * {out_table_stride_words}u + nt] = out_table[row4 * {out_table_stride_words}u + src_nt];",
            "    }",
            "  }",
            f"  {config['out_desc']} split_out_desc = *out_desc_ptr;",
            "  split_out_desc.out_tile_ptr_table = out_tbl_split;",
            f"  {config['function']}(&split_out_desc, act_desc_ptr, weight + {packed_weight_call_offset}u + split * {split_weight_bytes}u, bias + {folded_bias_call_offset}u + split * {split_bias_bytes}u, (const {config['mask_desc']} *)mask_words, extra);",
            "}",
        ]

    return "\n".join(
        [
            "#include <stdint.h>",
            "#include <string.h>",
            '#include "HAP_compute_res.h"',
            '#include "HAP_farf.h"',
            '#include "HAP_perf.h"',
            '#include "HAP_power.h"',
            f'#include "{config["header"]}"',
            "",
            "#define HM_REMOTE_TAG \"[HM_DEVICE]\"",
            "",
            c_array("k_activation", activation),
            c_array("k_packed_weight", packed_weight),
            c_array("k_folded_bias", folded_bias),
            c_array("k_output_seed", output_surface),
            c_array("k_native_raw", final_native_raw),
            c_array("k_native_internal_raw", native_internal_raw),
            *( [c_array("k_step0_native_raw", step0_native_raw)] if step0_native_raw is not None else [] ),
            c_array("k_hmx_identity_bias", hmx_identity_bias),
            c_array("k_mask_control", mask_control),
            c_u32_array("k_activation_offsets", activation_offsets[:act_entries]),
            c_u32_array("k_output_offsets", output_offsets[:out_entries]),
            "static int g_power_ctx;",
            "",
            "static void copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t n) {",
            "  for (uint32_t i = 0; i < n; ++i) dst[i] = src[i];",
            "}",
            "static void fill_bytes(uint8_t *dst, uint8_t value, uint32_t n) {",
            "  for (uint32_t i = 0; i < n; ++i) dst[i] = value;",
            "}",
            "static void init_output_seed(uint8_t *dst) {",
            *(
                ["  copy_bytes(dst, k_output_seed, sizeof(k_output_seed));"]
                if output_seed_mode == "prepared"
                else [
                    f"  fill_bytes(dst, {('0xff' if output_seed_mode == 'ff' else '0x00')}, sizeof(k_output_seed));"
                ]
            ),
            "}",
            "",
            "static uint32_t checksum(const uint8_t *data, uint32_t n) {",
            "  uint32_t h = 2166136261u;",
            "  for (uint32_t i = 0; i < n; ++i) { h ^= data[i]; h *= 16777619u; }",
            "  return h;",
            "}",
            "static inline void clear_hmx_acc(void) { __asm__ volatile(\"{ mxclracc }\"); }",
            "static inline void preload_hmx_bias_state(const void *ptr) {",
            '  __asm__ volatile("bias = mxmem2(%0)" :: "r"(ptr) : "memory");',
            "}",
            "static void init_extra_control(uint32_t *extra) {",
            f"  const uint32_t seed[{len(extra)}] = {{{extra_init}}};",
            f"  for (uint32_t i = 0; i < {len(extra)}u; ++i) extra[i] = seed[i];",
            "}",
            "static void init_w4a16_hmxi_private_payload(",
            "    uint32_t *payload,",
            "    uint8_t *weight,",
            "    uint8_t *bias,",
            "    int32_t *act_table,",
            "    int32_t *out_table,",
            "    uint8_t *out,",
            "    uint32_t *extra) {",
            "  payload[0] = 0u;",
            "  payload[1] = 0u;",
            "  payload[2] = (uint32_t)(uintptr_t)weight;",
            "  payload[3] = (uint32_t)(uintptr_t)bias;",
            "  payload[4] = (uint32_t)(uintptr_t)act_table;",
            "  payload[7] = 32u;",
            "  payload[8] = 8u;",
            "  payload[9] = 256u;",
            "  payload[10] = (uint32_t)(uintptr_t)out_table;",
            "  payload[16] = 64u;",
            "  payload[17] = 64u;",
            "  payload[32] = (uint32_t)(uintptr_t)extra;",
            "  payload[34] = (uint32_t)(uintptr_t)(payload + 34u);",
            "  payload[35] = (uint32_t)(uintptr_t)payload;",
            "  payload[36] = (uint32_t)(uintptr_t)(payload + 4u);",
            "  payload[37] = (uint32_t)(uintptr_t)out_table;",
            f"  for (uint32_t i = 0u; i < {out_entries}u; ++i) {{",
            "    payload[38u + i] = (uint32_t)(uintptr_t)(out + k_output_offsets[i]);",
            "  }",
            "}",
            "",
            "static uint32_t byte_diff(const uint8_t *a, const uint8_t *b, uint32_t n, uint32_t *first) {",
            "  uint32_t diff = 0u;",
            "  *first = 0xffffffffu;",
            "  for (uint32_t i = 0; i < n; ++i) {",
            "    if (a[i] != b[i]) {",
            "      if (*first == 0xffffffffu) *first = i;",
            "      ++diff;",
            "    }",
            "  }",
            "  return diff;",
            "}",
            "static uint32_t load_u16_le(const uint8_t *p) {",
            "  return ((uint32_t)p[0]) | (((uint32_t)p[1]) << 8);",
            "}",
            "static void log_u16_diff_summary(const uint8_t *actual, const uint8_t *expected, uint32_t n) {",
            "  uint32_t total = n / 2u;",
            "  uint32_t exact = 0u;",
            "  uint32_t max_abs = 0u;",
            "  uint32_t first = 0xffffffffu;",
            "  uint32_t samples = 0u;",
            "  uint32_t actual_zero = 0u;",
            "  uint32_t actual_ffff = 0u;",
            "  uint32_t native_zero = 0u;",
            "  uint32_t native_ffff = 0u;",
            "  uint32_t mismatch_actual_zero = 0u;",
            "  uint32_t mismatch_actual_ffff = 0u;",
            "  uint64_t sum_abs = 0u;",
            "  for (uint32_t i = 0u; i < total; ++i) {",
            "    uint32_t a = load_u16_le(actual + i * 2u);",
            "    uint32_t e = load_u16_le(expected + i * 2u);",
            "    if (a == 0u) ++actual_zero;",
            "    if (a == 65535u) ++actual_ffff;",
            "    if (e == 0u) ++native_zero;",
            "    if (e == 65535u) ++native_ffff;",
            "    if (a == e) {",
            "      ++exact;",
            "      continue;",
            "    }",
            "    if (a == 0u) ++mismatch_actual_zero;",
            "    if (a == 65535u) ++mismatch_actual_ffff;",
            "    uint32_t abs = a > e ? a - e : e - a;",
            "    if (first == 0xffffffffu) first = i;",
            "    if (abs > max_abs) max_abs = abs;",
            "    sum_abs += abs;",
            "    if (samples < 8u) {",
            f'      FARF(ALWAYS, HM_REMOTE_TAG " diff_u16_sample family={family} index=%u actual=%u native=%u abs=%u",',
            "           i, a, e, abs);",
            "      ++samples;",
            "    }",
            "  }",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " diff_u16 family={family} total=%u exact=%u mismatches=%u max_abs=%u sum_abs=%llu first=%d samples=%u",',
            "       total, exact, total - exact, max_abs, (unsigned long long)sum_abs,",
            "       first == 0xffffffffu ? -1 : (int)first, samples);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " diff_u16_endpoint family={family} actual_zero=%u actual_ffff=%u native_zero=%u native_ffff=%u mismatch_actual_zero=%u mismatch_actual_ffff=%u",',
            "       actual_zero, actual_ffff, native_zero, native_ffff, mismatch_actual_zero,",
            "       mismatch_actual_ffff);",
            "}",
            "static uint32_t u16_bin(uint32_t value) {",
            "  if (value == 0u) return 0u;",
            "  if (value < 1024u) return 1u;",
            "  if (value < 32768u) return 2u;",
            "  if (value < 64512u) return 3u;",
            "  if (value < 65535u) return 4u;",
            "  return 5u;",
            "}",
            "static void log_u16_value_bin_summaries(const uint8_t *actual, const uint8_t *expected, uint32_t n) {",
            "  uint32_t total[6] = {0u, 0u, 0u, 0u, 0u, 0u};",
            "  uint32_t exact[6] = {0u, 0u, 0u, 0u, 0u, 0u};",
            "  uint32_t actual_zero[6] = {0u, 0u, 0u, 0u, 0u, 0u};",
            "  uint32_t actual_ffff[6] = {0u, 0u, 0u, 0u, 0u, 0u};",
            "  uint32_t actual_mid[6] = {0u, 0u, 0u, 0u, 0u, 0u};",
            "  uint32_t total_elems = n / 2u;",
            "  for (uint32_t i = 0u; i < total_elems; ++i) {",
            "    uint32_t a = load_u16_le(actual + i * 2u);",
            "    uint32_t e = load_u16_le(expected + i * 2u);",
            "    uint32_t bin = u16_bin(e);",
            "    ++total[bin];",
            "    if (a == e) ++exact[bin];",
            "    if (a == 0u) ++actual_zero[bin];",
            "    else if (a == 65535u) ++actual_ffff[bin];",
            "    else ++actual_mid[bin];",
            "  }",
            "  for (uint32_t bin = 0u; bin < 6u; ++bin) {",
            f'    FARF(ALWAYS, HM_REMOTE_TAG " u16_value_bin family={family} bin=%u total=%u exact=%u actual_zero=%u actual_ffff=%u actual_mid=%u",',
            "         bin, total[bin], exact[bin], actual_zero[bin], actual_ffff[bin], actual_mid[bin]);",
            "  }",
            "}",
            f"static const uint32_t k_selected_u16_indices[{max(len(selected_u16_indices), 1)}] = {{{selected_u16_init or '0u'}}};",
            "static void log_selected_u16_samples(const uint8_t *actual, const uint8_t *expected, uint32_t n) {",
            f"  for (uint32_t sample = 0u; sample < {len(selected_u16_indices)}u; ++sample) {{",
            "    uint32_t index = k_selected_u16_indices[sample];",
            "    if (index >= n / 2u) continue;",
            "    uint32_t a = load_u16_le(actual + index * 2u);",
            "    uint32_t e = load_u16_le(expected + index * 2u);",
            "    uint32_t abs = a > e ? a - e : e - a;",
            f'    FARF(ALWAYS, HM_REMOTE_TAG " selected_u16_sample family={family} scope=public index=%u actual=%u native=%u abs=%u",',
            "         index, a, e, abs);",
            "  }",
            "}",
            (
                f"static const uint32_t k_selected_internal_u16_samples[{max(len(selected_internal_u16_samples), 1)}][2] = {{{selected_internal_init or '{0u, 0u}'}}};"
            ),
            "static void log_selected_internal_u16_samples(const uint8_t *actual, const uint8_t *expected) {",
            f"  const uint32_t block_bytes = {phase_block_bytes}u;",
            "  const uint32_t blocks = sizeof(k_native_internal_raw) / block_bytes;",
            "  const uint32_t block_u16 = block_bytes / 2u;",
            f"  for (uint32_t sample = 0u; sample < {len(selected_internal_u16_samples)}u; ++sample) {{",
            "    uint32_t block = k_selected_internal_u16_samples[sample][0];",
            "    uint32_t index = k_selected_internal_u16_samples[sample][1];",
            "    if (block >= blocks) continue;",
            "    if (index >= block_u16) continue;",
            "    uint32_t offset = block * block_bytes + index * 2u;",
            "    uint32_t a = load_u16_le(actual + offset);",
            "    uint32_t e = load_u16_le(expected + offset);",
            "    uint32_t abs = a > e ? a - e : e - a;",
            f'    FARF(ALWAYS, HM_REMOTE_TAG " selected_u16_sample family={family} scope=internal block=%u index=%u actual=%u native=%u abs=%u",',
            "         block, index, a, e, abs);",
            "  }",
            "}",
            "static void log_byte_lane_summaries(const uint8_t *actual, const uint8_t *expected, uint32_t n) {",
            "  for (uint32_t lane = 0u; lane < 4u; ++lane) {",
            "    uint32_t total = 0u;",
            "    uint32_t exact = 0u;",
            "    uint32_t actual_zero = 0u;",
            "    uint32_t actual_ff = 0u;",
            "    uint32_t native_zero = 0u;",
            "    uint32_t native_ff = 0u;",
            "    uint32_t actual_hash = 2166136261u;",
            "    uint32_t native_hash = 2166136261u;",
            "    for (uint32_t i = lane; i < n; i += 4u) {",
            "      uint8_t a = actual[i];",
            "      uint8_t e = expected[i];",
            "      ++total;",
            "      if (a == e) ++exact;",
            "      if (a == 0u) ++actual_zero;",
            "      if (a == 255u) ++actual_ff;",
            "      if (e == 0u) ++native_zero;",
            "      if (e == 255u) ++native_ff;",
            "      actual_hash ^= a; actual_hash *= 16777619u;",
            "      native_hash ^= e; native_hash *= 16777619u;",
            "    }",
            f'    FARF(ALWAYS, HM_REMOTE_TAG " byte_lane family={family} lane=%u total=%u exact=%u actual_zero=%u actual_ff=%u native_zero=%u native_ff=%u actual_hash=0x%08x native_hash=0x%08x",',
            "         lane, total, exact, actual_zero, actual_ff, native_zero, native_ff,",
            "         actual_hash, native_hash);",
            "  }",
            "}",
            "static void log_u16_region_summaries(const uint8_t *actual, const uint8_t *expected) {",
            f"  const uint32_t rows = {m}u;",
            f"  const uint32_t cols = {n}u;",
            "  const uint32_t axes = 3u;",
            "  for (uint32_t axis = 0u; axis < axes; ++axis) {",
            "    for (uint32_t bucket = 0u; bucket < 8u; ++bucket) {",
            "      uint32_t total = 0u;",
            "      uint32_t exact = 0u;",
            "      uint32_t actual_zero = 0u;",
            "      uint32_t actual_ffff = 0u;",
            "      uint32_t native_zero = 0u;",
            "      uint32_t native_ffff = 0u;",
            "      uint32_t actual_hash = 2166136261u;",
            "      uint32_t native_hash = 2166136261u;",
            "      for (uint32_t row = 0u; row < rows; ++row) {",
            "        for (uint32_t col = 0u; col < cols; ++col) {",
            "          uint32_t in_bucket = 0u;",
            "          if (axis == 0u) in_bucket = ((row & 31u) >> 2u) == bucket;",
            "          else if (axis == 1u) in_bucket = (row >> 5u) == bucket;",
            "          else in_bucket = (col >> 5u) == bucket;",
            "          if (!in_bucket) continue;",
            "          const uint8_t *ap = actual + (row * cols + col) * 2u;",
            "          const uint8_t *ep = expected + (row * cols + col) * 2u;",
            "          uint32_t a = load_u16_le(ap);",
            "          uint32_t e = load_u16_le(ep);",
            "          ++total;",
            "          if (a == e) ++exact;",
            "          if (a == 0u) ++actual_zero;",
            "          if (a == 65535u) ++actual_ffff;",
            "          if (e == 0u) ++native_zero;",
            "          if (e == 65535u) ++native_ffff;",
            "          actual_hash ^= ap[0]; actual_hash *= 16777619u;",
            "          actual_hash ^= ap[1]; actual_hash *= 16777619u;",
            "          native_hash ^= ep[0]; native_hash *= 16777619u;",
            "          native_hash ^= ep[1]; native_hash *= 16777619u;",
            "        }",
            "      }",
            f'      FARF(ALWAYS, HM_REMOTE_TAG " diff_u16_region family={family} axis=%u bucket=%u total=%u exact=%u actual_zero=%u actual_ffff=%u native_zero=%u native_ffff=%u actual_hash=0x%08x native_hash=0x%08x",',
            "           axis, bucket, total, exact, actual_zero, actual_ffff, native_zero, native_ffff,",
            "           actual_hash, native_hash);",
            "    }",
            "  }",
            "}",
            "static void log_internal_block_summaries(const uint8_t *actual, const uint8_t *expected) {",
            f"  const uint32_t block_bytes = {phase_block_bytes}u;",
            "  const uint32_t blocks = sizeof(k_native_internal_raw) / block_bytes;",
            "  for (uint32_t block = 0u; block < blocks; ++block) {",
            "    const uint8_t *ap = actual + block * block_bytes;",
            "    const uint8_t *ep = expected + block * block_bytes;",
            "    uint32_t exact_bytes = 0u;",
            "    uint32_t first = 0xffffffffu;",
            "    uint32_t actual_hash = 2166136261u;",
            "    uint32_t native_hash = 2166136261u;",
            "    for (uint32_t i = 0u; i < block_bytes; ++i) {",
            "      if (ap[i] == ep[i]) ++exact_bytes;",
            "      else if (first == 0xffffffffu) first = i;",
            "      actual_hash ^= ap[i]; actual_hash *= 16777619u;",
            "      native_hash ^= ep[i]; native_hash *= 16777619u;",
            "    }",
            f'    FARF(ALWAYS, HM_REMOTE_TAG " internal_block family={family} block=%u exact_bytes=%u byte_diffs=%u first_diff=%d actual_hash=0x%08x native_hash=0x%08x",',
            "         block, exact_bytes, block_bytes - exact_bytes, first == 0xffffffffu ? -1 : (int)first,",
            "         actual_hash, native_hash);",
            "  }",
            "}",
            "static void log_internal_block_u16_value_summaries(const uint8_t *actual, const uint8_t *expected) {",
            f"  const uint32_t block_bytes = {phase_block_bytes}u;",
            "  const uint32_t blocks = sizeof(k_native_internal_raw) / block_bytes;",
            "  for (uint32_t block = 0u; block < blocks; ++block) {",
            "    if ((block & 7u) != 0u) continue;",
            "    const uint8_t *ap = actual + block * block_bytes;",
            "    const uint8_t *ep = expected + block * block_bytes;",
            "    uint32_t exact = 0u;",
            "    uint32_t actual_zero = 0u;",
            "    uint32_t actual_ffff = 0u;",
            "    uint32_t actual_mid = 0u;",
            "    uint32_t native_zero = 0u;",
            "    uint32_t native_ffff = 0u;",
            "    uint32_t native_mid = 0u;",
            "    for (uint32_t i = 0u; i < block_bytes / 2u; ++i) {",
            "      uint32_t a = load_u16_le(ap + i * 2u);",
            "      uint32_t e = load_u16_le(ep + i * 2u);",
            "      if (a == e) ++exact;",
            "      if (a == 0u) ++actual_zero;",
            "      else if (a == 65535u) ++actual_ffff;",
            "      else ++actual_mid;",
            "      if (e == 0u) ++native_zero;",
            "      else if (e == 65535u) ++native_ffff;",
            "      else ++native_mid;",
            "    }",
            f'    FARF(ALWAYS, HM_REMOTE_TAG " internal_block_u16_value family={family} block=%u total=%u exact=%u actual_zero=%u actual_ffff=%u actual_mid=%u native_zero=%u native_ffff=%u native_mid=%u",',
            "         block, block_bytes / 2u, exact, actual_zero, actual_ffff, actual_mid,",
            "         native_zero, native_ffff, native_mid);",
            "  }",
            "}",
            "static void log_internal_endpoint_mid_samples(const uint8_t *actual, const uint8_t *expected) {",
            f"  const uint32_t block_bytes = {phase_block_bytes}u;",
            "  const uint32_t blocks = sizeof(k_native_internal_raw) / block_bytes;",
            "  for (uint32_t block = 0u; block < blocks; ++block) {",
            "    if ((block & 7u) != 0u) continue;",
            "    const uint8_t *ap = actual + block * block_bytes;",
            "    const uint8_t *ep = expected + block * block_bytes;",
            "    uint32_t samples = 0u;",
            "    for (uint32_t i = 0u; i < block_bytes / 2u && samples < 2u; ++i) {",
            "      uint32_t a = load_u16_le(ap + i * 2u);",
            "      uint32_t e = load_u16_le(ep + i * 2u);",
            "      if ((a == 0u || a == 65535u) && e > 0u && e < 65535u) {",
            f'        FARF(ALWAYS, HM_REMOTE_TAG " internal_endpoint_mid_sample family={family} block=%u index=%u actual=%u native=%u",',
            "             block, i, a, e);",
            "        ++samples;",
            "      }",
            "    }",
            "  }",
            "}",
            "static void log_written_block_summaries(const uint8_t *actual, const uint8_t *seed) {",
            f"  const uint32_t block_bytes = {phase_block_bytes}u;",
            "  const uint32_t blocks = sizeof(k_native_internal_raw) / block_bytes;",
            "  for (uint32_t block = 0u; block < blocks; ++block) {",
            "    const uint8_t *ap = actual + block * block_bytes;",
            "    const uint8_t *sp = seed + block * block_bytes;",
            "    uint32_t changed_bytes = 0u;",
            "    uint32_t first = 0xffffffffu;",
            "    uint32_t actual_hash = 2166136261u;",
            "    uint32_t seed_hash = 2166136261u;",
            "    for (uint32_t i = 0u; i < block_bytes; ++i) {",
            "      if (ap[i] != sp[i]) {",
            "        if (first == 0xffffffffu) first = i;",
            "        ++changed_bytes;",
            "      }",
            "      actual_hash ^= ap[i]; actual_hash *= 16777619u;",
            "      seed_hash ^= sp[i]; seed_hash *= 16777619u;",
            "    }",
            f'    FARF(ALWAYS, HM_REMOTE_TAG " written_block family={family} block=%u changed_bytes=%u first_change=%d actual_hash=0x%08x seed_hash=0x%08x",',
            "         block, changed_bytes, first == 0xffffffffu ? -1 : (int)first,",
            "         actual_hash, seed_hash);",
            "  }",
            "}",
            "",
            "static int power_on_hvx_hmx(void) {",
            "  HAP_power_request_t req;",
            "  memset(&req, 0, sizeof(req));",
            "  req.type = HAP_power_set_apptype;",
            "  req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;",
            "  if (HAP_power_set((void *)&g_power_ctx, &req) != 0) return -1;",
            "  memset(&req, 0, sizeof(req));",
            "  req.type = HAP_power_set_DCVS_v3;",
            "  req.dcvs_v3.set_dcvs_enable = 1;",
            "  req.dcvs_v3.dcvs_enable = 1;",
            "  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;",
            "  req.dcvs_v3.set_bus_params = 1;",
            "  req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;",
            "  req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;",
            "  req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;",
            "  req.dcvs_v3.set_core_params = 1;",
            "  req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;",
            "  req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;",
            "  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;",
            "  req.dcvs_v3.set_sleep_disable = 1;",
            "  req.dcvs_v3.sleep_disable = 1;",
            "  if (HAP_power_set((void *)&g_power_ctx, &req) != 0) return -2;",
            "  memset(&req, 0, sizeof(req));",
            "  req.type = HAP_power_set_HVX;",
            "  req.hvx.power_up = 1;",
            "  if (HAP_power_set((void *)&g_power_ctx, &req) != 0) return -3;",
            "  memset(&req, 0, sizeof(req));",
            "  req.type = HAP_power_set_HMX;",
            "  req.hmx.power_up = 1;",
            "  if (HAP_power_set((void *)&g_power_ctx, &req) != 0) return -4;",
            "  return 0;",
            "}",
            "",
            *deblock_helper,
            *wrapper_prefetch_helper,
            "",
            "int main(int argc, char **argv) {",
            "  (void)argc; (void)argv;",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " start family={family} native_checksum={native_checksum}");',
            "  int ret = power_on_hvx_hmx();",
            '  if (ret != 0) { FARF(ALWAYS, HM_REMOTE_TAG " power_failed ret=%d", ret); return 10; }',
            "  unsigned int vtcm_size = 8u * 1024u * 1024u;",
            "  HAP_compute_res_query_VTCM(0, &vtcm_size, 0, 0, 0);",
            "  compute_res_attr_t attr;",
            "  HAP_compute_res_attr_init(&attr);",
            "  HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);",
            "  HAP_compute_res_attr_set_hmx_param(&attr, 1);",
            "  unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);",
            '  if (ctx_id == 0) { FARF(ALWAYS, HM_REMOTE_TAG " vtcm_acquire_failed"); return 11; }',
            "  uint8_t *base = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);",
            '  if (!base) { FARF(ALWAYS, HM_REMOTE_TAG " vtcm_null"); HAP_compute_res_release(ctx_id); return 12; }',
            "  ret = HAP_compute_res_hmx_lock(ctx_id);",
            '  if (ret != 0) { FARF(ALWAYS, HM_REMOTE_TAG " hmx_lock_failed ret=%d", ret); HAP_compute_res_release(ctx_id); return 13; }',
            '  FARF(ALWAYS, HM_REMOTE_TAG " vtcm=%p vtcm_size=%u", base, vtcm_size);',
            "  uint8_t *act = base + 0x00000u;",
            "  uint8_t *weight = base + 0x20000u;",
            "  uint8_t *bias = base + 0x40000u;",
            "  uint8_t *out = base + 0x50000u;",
            "  uint8_t *hmx_identity_bias = base + 0x73000u;",
            "  uint8_t *public_out = base + 0x80000u;",
            *(
                [
                    "  uint8_t *record_window = base + 0x70000u;",
                    "  uint8_t *record_base = record_window + 0x180u;",
                    "  int32_t *act_table = (int32_t *)(record_window + 0x000u);",
                    "  int32_t *out_table = (int32_t *)(record_base + 0x098u);",
                ]
                if hmxi_private_payload_carrier
                else [
                    "  int32_t *act_table = (int32_t *)(base + 0x70000u);",
                    "  int32_t *out_table = (int32_t *)(base + 0x71000u);",
                ]
            ),
            "  copy_bytes(act, k_activation, sizeof(k_activation));",
            "  copy_bytes(weight, k_packed_weight, sizeof(k_packed_weight));",
            "  copy_bytes(bias, k_folded_bias, sizeof(k_folded_bias));",
            "  copy_bytes(hmx_identity_bias, k_hmx_identity_bias, sizeof(k_hmx_identity_bias));",
            "  init_output_seed(out);",
            *(
                [
                    "  for (uint32_t i = 0u; i < 1024u; ++i) ((uint32_t *)record_window)[i] = 0u;",
                ]
                if hmxi_private_payload_carrier
                else []
            ),
            f"  for (uint32_t i = 0; i < {act_entries}u; ++i) act_table[i] = (int32_t)(uintptr_t)(act + k_activation_offsets[i]);",
            f"  for (uint32_t i = 0; i < {out_entries}u; ++i) out_table[i] = (int32_t)(uintptr_t)(out + k_output_offsets[i]);",
            (
                "  uint32_t *mask_words = (uint32_t *)(record_base + 0x048u);"
                if hmxi_private_payload_carrier
                else "  uint32_t mask_words[16] __attribute__((aligned(16)));"
            ),
            "  copy_bytes((uint8_t *)mask_words, k_mask_control, 64u);",
            (
                "  uint32_t *extra = (uint32_t *)(base + 0x72000u); init_extra_control(extra);"
                if hmxi_private_payload_carrier
                else f"  uint32_t extra[{len(extra)}] __attribute__((aligned(16))) = {{{extra_init}}};"
            ),
            *(
                [
                    f"  {config['act_desc']} *act_desc_ptr = ({config['act_desc']} *)(record_base + 0x010u);",
                    f"  {config['out_desc']} *out_desc_ptr = ({config['out_desc']} *)(record_base + 0x028u);",
                    "  act_desc_ptr->act_ptr_pairs = act_table;",
                    f"  act_desc_ptr->n_act_pairs = {int(act_desc['n_act_pairs'])}u;",
                    f"  act_desc_ptr->act_table_y_stride_words = {int(act_desc['act_table_y_stride_words'])}u;",
                    "  out_desc_ptr->out_tile_ptr_table = out_table;",
                    f"  out_desc_ptr->out_table_stride_dwords = {int(out_desc['out_table_stride_dwords'])}u;",
                    f"  out_desc_ptr->out_y_stride_words = {int(out_desc['out_y_stride_words'])}u;",
                    f"  out_desc_ptr->n_tiles_pow2 = {int(out_desc['n_tiles_pow2'])}u;",
                    f"  out_desc_ptr->m_total_minus_step = {int(out_desc['m_total_minus_step'])};",
                    f"  out_desc_ptr->k_total_bytes = {int(out_desc['k_total_bytes'])}u;",
                    f"  init_w4a16_hmxi_private_payload((uint32_t *)record_base, weight + {packed_weight_call_offset}u, bias + {folded_bias_call_offset}u, act_table, out_table, out, extra);",
                    '  FARF(ALWAYS, HM_REMOTE_TAG " carrier=w4a16_hmxi_private_payload words=64 known_words=61 provisional_context_words=3");',
                ]
                if hmxi_private_payload_carrier
                else [
                    f"  {config['out_desc']} out_desc = {{",
                    "      out_table,",
                    f"      {int(out_desc['out_table_stride_dwords'])}u,",
                    f"      {int(out_desc['out_y_stride_words'])}u,",
                    f"      {int(out_desc['n_tiles_pow2'])}u,",
                    f"      {int(out_desc['m_total_minus_step'])},",
                    f"      {int(out_desc['k_total_bytes'])}u,",
                    "  };",
                    f"  {config['act_desc']} act_desc = {{",
                    "      act_table,",
                    f"      {int(act_desc['n_act_pairs'])}u,",
                    f"      {int(act_desc['act_table_y_stride_words'])}u,",
                    "  };",
                    f"  {config['out_desc']} *out_desc_ptr = &out_desc;",
                    f"  {config['act_desc']} *act_desc_ptr = &act_desc;",
                ]
            ),
            f'  FARF(ALWAYS, HM_REMOTE_TAG " prepared family={family} act=0x%08x weight=0x%08x bias=0x%08x seed=0x%08x mask=0x%08x act_offsets=0x%08x out_offsets=0x%08x",',
            "       checksum(act, sizeof(k_activation)),",
            "       checksum(weight, sizeof(k_packed_weight)),",
            "       checksum(bias, sizeof(k_folded_bias)),",
            "       checksum(out, sizeof(k_output_seed)),",
            "       checksum((const uint8_t *)mask_words, 64u),",
            "       checksum((const uint8_t *)k_activation_offsets, sizeof(k_activation_offsets)),",
            "       checksum((const uint8_t *)k_output_offsets, sizeof(k_output_offsets)));",
            "  mask_words[14] = (uint32_t)(uintptr_t)extra;",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " dynamic_mask family={family} mask14=0x%08x extra=0x%08x patched=1",',
            "       ((const uint32_t *)mask_words)[14],",
            "       (uint32_t)(uintptr_t)extra);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " callabi family={family} out_stride=%u out_y=%u out_tiles=%u out_m_step=%d out_k=%u act_pairs=%u act_y=%u extra0=%u extra1=%u extra2=%u extra3=%u mask6=%u",',
            "       out_desc_ptr->out_table_stride_dwords,",
            "       out_desc_ptr->out_y_stride_words,",
            "       out_desc_ptr->n_tiles_pow2,",
            "       out_desc_ptr->m_total_minus_step,",
            "       out_desc_ptr->k_total_bytes,",
            "       act_desc_ptr->n_act_pairs,",
            "       act_desc_ptr->act_table_y_stride_words,",
            f"       {extra_arg_exprs[0]}, {extra_arg_exprs[1]}, {extra_arg_exprs[2]}, {extra_arg_exprs[3]},",
            "       ((const uint32_t *)mask_words)[6]);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " hnh_path family={family} kernel_entry={kernel_entry} main_path=%u odd_act_pair=%u k_groups=%u k_pair_groups=%u final_reduce=%u drain_count=%u cvt0=%u cvt1=%u mask12=%u mask14_ptr_patched=1",',
            "       act_desc_ptr->n_act_pairs > 1u,",
            "       act_desc_ptr->n_act_pairs & 1u,",
            "       (out_desc_ptr->k_total_bytes + 31u) >> 5,",
            "       ((out_desc_ptr->k_total_bytes + 31u) >> 5) >> 1,",
            "       (((out_desc_ptr->k_total_bytes + 31u) >> 5) & 1u) != 0u,",
            f"       {extra_arg_exprs[0]}, {extra_arg_exprs[1]}, {extra_arg_exprs[2]},",
            "       ((const uint32_t *)mask_words)[12]);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " layout_vtcm family={family} base=0x%08x act=0x%08x weight=0x%08x bias=0x%08x out=0x%08x public=0x%08x act_table=0x%08x out_table=0x%08x",',
            "       (uint32_t)(uintptr_t)base,",
            "       (uint32_t)(uintptr_t)act,",
            "       (uint32_t)(uintptr_t)weight,",
            "       (uint32_t)(uintptr_t)bias,",
            "       (uint32_t)(uintptr_t)out,",
            "       (uint32_t)(uintptr_t)public_out,",
            "       (uint32_t)(uintptr_t)act_table,",
            "       (uint32_t)(uintptr_t)out_table);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " layout_callptr family={family} mask=0x%08x extra=0x%08x out_desc=0x%08x act_desc=0x%08x",',
            "       (uint32_t)(uintptr_t)mask_words,",
            "       (uint32_t)(uintptr_t)extra,",
            "       (uint32_t)(uintptr_t)out_desc_ptr,",
            "       (uint32_t)(uintptr_t)act_desc_ptr);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " data_callptr family={family} weight=0x%08x bias=0x%08x weight_offset=%d bias_offset=%d",',
            f"       (uint32_t)(uintptr_t)(weight + {packed_weight_call_offset}u),",
            f"       (uint32_t)(uintptr_t)(bias + {folded_bias_call_offset}u),",
            f"       {packed_weight_byte_offset}, {folded_bias_byte_offset});",
            *(
                [
                    "  preload_hmx_bias_state(hmx_identity_bias);",
                    f'  FARF(ALWAYS, HM_REMOTE_TAG " hmx_identity_bias_preload family={family} ptr=0x%08x bytes=%u scale_half=0x3c00 bias_half=0x0000",',
                    "       (uint32_t)(uintptr_t)hmx_identity_bias, (uint32_t)sizeof(k_hmx_identity_bias));",
                ]
                if preload_hmx_identity_bias
                else []
            ),
            f"  {public_deblock_function}(public_out, out);",
            "  uint32_t before_hash = checksum(public_out, sizeof(k_output_seed));",
            "  uint64_t t0 = HAP_perf_get_time_us();",
            "  uint64_t q0 = HAP_perf_get_qtimer_count();",
            "  uint64_t p0 = HAP_perf_get_pcycles();",
            f"  for (uint32_t step = 0; step < {run_chain_steps}u; ++step) {{",
            "    init_output_seed(out);",
            *wrapper_prefetch_call,
            *(['    clear_hmx_acc();'] if pre_clear_acc else []),
            *["    " + line for line in kernel_call_lines],
            f"    {public_deblock_function}(public_out, out);",
            "    uint32_t step_hash = checksum(public_out, sizeof(k_output_seed));",
            *(
                [
                    "    if (step == 0u) {",
                    "      uint32_t step_first_diff = 0u;",
                    "      uint32_t step_diffs = byte_diff(public_out, k_step0_native_raw, sizeof(k_step0_native_raw), &step_first_diff);",
                    f'      FARF(ALWAYS, HM_REMOTE_TAG " step family={family} step=%u checksum=0x%08x native={step0_native_checksum} diffs=%u first_diff=%d",',
                    "           step, step_hash, step_diffs, step_first_diff == 0xffffffffu ? -1 : (int)step_first_diff);",
                    "    } else {",
                    f'      FARF(ALWAYS, HM_REMOTE_TAG " step family={family} step=%u checksum=0x%08x", step, step_hash);',
                    "    }",
                ]
                if step0_native_raw is not None
                else [
                    f'    FARF(ALWAYS, HM_REMOTE_TAG " step family={family} step=%u checksum=0x%08x", step, step_hash);'
                ]
            ),
            f"    if (step + 1u < {run_chain_steps}u) copy_bytes(act, out, sizeof(k_activation));",
            "  }",
            "  uint64_t p1 = HAP_perf_get_pcycles();",
            "  uint64_t q1 = HAP_perf_get_qtimer_count();",
            "  uint64_t t1 = HAP_perf_get_time_us();",
            f"  {public_deblock_function}(public_out, out);",
            "  uint32_t out_hash = checksum(public_out, sizeof(k_output_seed));",
            "  uint32_t first_diff = 0u;",
            "  uint32_t diffs = byte_diff(public_out, k_native_raw, sizeof(k_native_raw), &first_diff);",
            *(["  log_u16_diff_summary(public_out, k_native_raw, sizeof(k_native_raw));"] if not focused_sample_log else []),
            "  log_selected_u16_samples(public_out, k_native_raw, sizeof(k_native_raw));",
            *(
                [
                    "  log_u16_value_bin_summaries(public_out, k_native_raw, sizeof(k_native_raw));",
                    "  log_byte_lane_summaries(public_out, k_native_raw, sizeof(k_native_raw));",
                    "  log_u16_region_summaries(public_out, k_native_raw);",
                    "  log_written_block_summaries(out, k_output_seed);",
                    "  log_internal_block_summaries(out, k_native_internal_raw);",
                    "  log_internal_block_u16_value_summaries(out, k_native_internal_raw);",
                    "  log_internal_endpoint_mid_samples(out, k_native_internal_raw);",
                ]
                if not focused_sample_log
                else []
            ),
            "  log_selected_internal_u16_samples(out, k_native_internal_raw);",
            *(
                ["  log_alt_layout_summaries(public_out, out, k_native_raw);"]
                if config["output_mapping"] == "a16_crouton16_row4" and not focused_sample_log
                else []
            ),
            "  uint64_t empty_q0 = HAP_perf_get_qtimer_count();",
            "  uint64_t empty_p0 = HAP_perf_get_pcycles();",
            f"  for (uint32_t repeat = 0; repeat < {measure_repeats}u; ++repeat) {{",
            "    copy_bytes(act, k_activation, sizeof(k_activation));",
            f"    for (uint32_t step = 0; step < {run_chain_steps}u; ++step) {{",
            "      init_output_seed(out);",
            f"      if (step + 1u < {run_chain_steps}u) copy_bytes(act, out, sizeof(k_activation));",
            "    }",
            "  }",
            "  uint64_t empty_p1 = HAP_perf_get_pcycles();",
            "  uint64_t empty_q1 = HAP_perf_get_qtimer_count();",
            "  uint64_t body_q0 = HAP_perf_get_qtimer_count();",
            "  uint64_t body_p0 = HAP_perf_get_pcycles();",
            f"  for (uint32_t repeat = 0; repeat < {measure_repeats}u; ++repeat) {{",
            "    copy_bytes(act, k_activation, sizeof(k_activation));",
            f"    for (uint32_t step = 0; step < {run_chain_steps}u; ++step) {{",
            "      init_output_seed(out);",
            *["      " + line.strip() for line in wrapper_prefetch_call],
            *(['      clear_hmx_acc();'] if pre_clear_acc else []),
            *["      " + line for line in kernel_call_lines],
            f"      if (step + 1u < {run_chain_steps}u) copy_bytes(act, out, sizeof(k_activation));",
            "    }",
            "  }",
            "  uint64_t body_p1 = HAP_perf_get_pcycles();",
            "  uint64_t body_q1 = HAP_perf_get_qtimer_count();",
            "  uint64_t empty_pcycles = empty_p1 - empty_p0;",
            "  uint64_t body_pcycles = body_p1 - body_p0;",
            "  uint64_t net_pcycles = body_pcycles > empty_pcycles ? body_pcycles - empty_pcycles : 0u;",
            f"  uint64_t net_per_repeat = net_pcycles / {measure_repeats}u;",
            f"  uint64_t net_per_step = net_pcycles / ({measure_repeats}u * {run_chain_steps}u);",
            "  uint64_t run_qticks = q1 - q0;",
            "  uint64_t empty_qticks = empty_q1 - empty_q0;",
            "  uint64_t body_qticks = body_q1 - body_q0;",
            "  uint64_t net_qticks = body_qticks > empty_qticks ? body_qticks - empty_qticks : 0u;",
            f"  uint64_t net_qticks_per_repeat = net_qticks / {measure_repeats}u;",
            f"  uint64_t net_qticks_per_step = net_qticks / ({measure_repeats}u * {run_chain_steps}u);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " result family={family} entered=1 before=0x%08x checksum=0x%08x native=0x%08x bytes=%u diffs=%u first_diff=%d runtime_us=%llu pcycles=%llu",',
            "       before_hash, out_hash, checksum(k_native_raw, sizeof(k_native_raw)),",
            "       (uint32_t)sizeof(k_output_seed), diffs, first_diff == 0xffffffffu ? -1 : (int)first_diff,",
            "       (unsigned long long)(t1 - t0), (unsigned long long)(p1 - p0));",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " perf family={family} repeats=%u empty=%llu body=%llu net=%llu per_repeat=%llu per_step=%llu",',
            f"       {measure_repeats}u, (unsigned long long)empty_pcycles, (unsigned long long)body_pcycles,",
            "       (unsigned long long)net_pcycles, (unsigned long long)net_per_repeat,",
            "       (unsigned long long)net_per_step);",
            f'  FARF(ALWAYS, HM_REMOTE_TAG " timeline family={family} run_qticks=%llu empty_qticks=%llu body_qticks=%llu net_qticks=%llu net_per_repeat=%llu net_per_step=%llu",',
            "       (unsigned long long)run_qticks, (unsigned long long)empty_qticks,",
            "       (unsigned long long)body_qticks, (unsigned long long)net_qticks,",
            "       (unsigned long long)net_qticks_per_repeat, (unsigned long long)net_qticks_per_step);",
            "  HAP_compute_res_hmx_unlock(ctx_id);",
            "  HAP_compute_res_release(ctx_id);",
            "  return diffs == 0u ? 0 : 20;",
            "}",
            "",
        ]
    )


def build_device_so(
    family: str,
    artifact: Path,
    out_dir: Path,
    measure_repeats: int = 1,
    native_wrapper_prefetch: bool = False,
    pre_clear_acc: bool = False,
    activation_raw_override: Path | None = None,
    packed_weight_byte_offset: int = 0,
    packed_weight_raw_override: Path | None = None,
    folded_bias_raw_override: Path | None = None,
    folded_bias_byte_offset: int = 0,
    descriptor_carrier: str = "separate",
    source_stem: str | None = None,
    step0_native_raw: bytes | None = None,
    reference_raw_override: bytes | None = None,
    chain_steps: int | None = None,
    output_seed_mode: str = "prepared",
    extra_word_overrides: dict[int, int] | None = None,
    mask_word_overrides: dict[int, int] | None = None,
    kernel_entry: str = "deep",
    preload_hmx_identity_bias: bool = False,
    u16_sample_indices: list[int] | None = None,
    internal_u16_samples: list[tuple[int, int]] | None = None,
    focused_sample_log: bool = False,
    native_internal_raw_override: bytes | None = None,
    public_output_layout: str = "default",
) -> tuple[Path, Path]:
    abi = load_json(artifact / "analysis" / "abi_manifest.json")
    oracle = dict(load_json(EXAMPLE / "oracles.json")["families"][family])
    abi_shape = [int(v) for v in abi["shape_mkn"]]
    abi_chain = int(abi["chain"])
    canonical_shape = [int(v) for v in oracle["shape_mkn"]]
    reduced_or_custom_shape = abi_shape != canonical_shape or abi_chain != int(oracle["chain"])
    if reduced_or_custom_shape:
        oracle["shape_mkn"] = abi_shape
        oracle["chain"] = abi_chain
        requested_chain_steps = abi_chain if chain_steps is None else int(chain_steps)
        if reference_raw_override is not None:
            native_raw = reference_raw_override
        elif step0_native_raw is None or requested_chain_steps != 1:
            raise ValueError(
                f"{family}: reduced/custom artifact {abi_shape} chain={abi_chain} "
                "requires --reference-raw-override, or --step0-native-raw "
                "with --chain-steps 1"
            )
        else:
            native_raw = step0_native_raw
    else:
        native_raw = (ROOT / oracle["raw_output"]["path"]).read_bytes()
        if reference_raw_override is not None:
            native_raw = reference_raw_override
    stem = source_stem or f"artifact_body_device_{family}"
    source = out_dir / f"{stem}.c"
    binary = out_dir / f"lib{stem}.so"
    source.write_text(
        generate_source(
            family,
            artifact,
            abi,
            oracle,
            native_raw,
            measure_repeats,
            native_wrapper_prefetch,
            pre_clear_acc,
            activation_raw_override,
            packed_weight_byte_offset,
            packed_weight_raw_override,
            folded_bias_raw_override,
            folded_bias_byte_offset,
            descriptor_carrier,
            step0_native_raw,
            chain_steps,
            output_seed_mode,
            extra_word_overrides,
            mask_word_overrides,
            kernel_entry,
            preload_hmx_identity_bias,
            u16_sample_indices,
            internal_u16_samples,
            focused_sample_log,
            native_internal_raw_override,
            public_output_layout,
        ),
        encoding="utf-8",
    )
    clang = tool("hexagon-clang")
    cmd = [
        str(clang),
        "-mv75",
        "-O2",
        "-G0",
        "-fPIC",
        "-shared",
        "-mhvx",
        "-mhvx-length=128B",
        "-mhmx",
        "-I",
        str(SDK / "incs"),
        "-I",
        str(SDK / "incs" / "stddef"),
        "-I",
        str(SDK / "rtos" / "qurt" / "computev75" / "include"),
        "-I",
        str(SDK / "rtos" / "qurt" / "computev75" / "include" / "qurt"),
        "-I",
        str(EXAMPLE / "include"),
        str(source),
        "-o",
        str(binary),
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError("hexagon-clang failed:\n" + result.stdout)
    return source, binary


def parse_device_log(log: str, family: str) -> dict:
    pattern = re.compile(
        rf"\[HM_DEVICE\] result family={re.escape(family)} entered=(?P<entered>[0-9]+) "
        rf"before=(?P<before>0x[0-9a-fA-F]+) checksum=(?P<checksum>0x[0-9a-fA-F]+) "
        rf"native=(?P<native>0x[0-9a-fA-F]+) bytes=(?P<bytes>[0-9]+) "
        rf"diffs=(?P<diffs>[0-9]+) first_diff=(?P<first>-?[0-9]+) "
        rf"runtime_us=(?P<runtime>[0-9]+)(?: pcycles=(?P<pcycles>[0-9]+))?"
    )
    perf_pattern = re.compile(
        rf"\[HM_DEVICE\] perf family={re.escape(family)} repeats=(?P<repeats>[0-9]+) "
        rf"empty=(?P<empty_pcycles>[0-9]+) body=(?P<body_pcycles>[0-9]+) "
        rf"net=(?P<net_pcycles>[0-9]+) per_repeat=(?P<net_per_repeat>[0-9]+) "
        rf"per_step=(?P<net_per_step>[0-9]+)"
    )
    timeline_pattern = re.compile(
        rf"\[HM_DEVICE\] timeline family={re.escape(family)} run_qticks=(?P<run_qticks>[0-9]+) "
        rf"empty_qticks=(?P<empty_qticks>[0-9]+) body_qticks=(?P<body_qticks>[0-9]+) "
        rf"net_qticks=(?P<net_qticks>[0-9]+) net_per_repeat=(?P<net_qticks_per_repeat>[0-9]+) "
        rf"net_per_step=(?P<net_qticks_per_step>[0-9]+)"
    )
    prepared_pattern = re.compile(
        rf"\[HM_DEVICE\] prepared family={re.escape(family)} "
        rf"act=(?P<activation>0x[0-9a-fA-F]+) "
        rf"weight=(?P<packed_weight>0x[0-9a-fA-F]+) "
        rf"bias=(?P<folded_bias>0x[0-9a-fA-F]+) "
        rf"seed=(?P<output_seed>0x[0-9a-fA-F]+) "
        rf"mask=(?P<mask_control>0x[0-9a-fA-F]+) "
        rf"act_offsets=(?P<activation_offsets>0x[0-9a-fA-F]+) "
        rf"out_offsets=(?P<output_offsets>0x[0-9a-fA-F]+)"
    )
    callabi_pattern = re.compile(
        rf"\[HM_DEVICE\] callabi family={re.escape(family)} "
        rf"out_stride=(?P<out_table_stride_dwords>[0-9]+) "
        rf"out_y=(?P<out_y_stride_words>[0-9]+) "
        rf"out_tiles=(?P<n_tiles_pow2>[0-9]+) "
        rf"out_m_step=(?P<m_total_minus_step>-?[0-9]+) "
        rf"out_k=(?P<k_total_bytes>[0-9]+) "
        rf"act_pairs=(?P<n_act_pairs>[0-9]+) "
        rf"act_y=(?P<act_table_y_stride_words>[0-9]+) "
        rf"extra0=(?P<extra0>[0-9]+) "
        rf"extra1=(?P<extra1>[0-9]+) "
        rf"extra2=(?P<extra2>[0-9]+) "
        rf"extra3=(?P<extra3>[0-9]+) "
        rf"mask6=(?P<mask6>[0-9]+)"
    )
    layout_vtcm_pattern = re.compile(
        rf"\[HM_DEVICE\] layout_vtcm family={re.escape(family)} "
        rf"base=(?P<base>0x[0-9a-fA-F]+) "
        rf"act=(?P<activation>0x[0-9a-fA-F]+) "
        rf"weight=(?P<packed_weight>0x[0-9a-fA-F]+) "
        rf"bias=(?P<folded_bias>0x[0-9a-fA-F]+) "
        rf"out=(?P<output_surface>0x[0-9a-fA-F]+) "
        rf"public=(?P<public_output>0x[0-9a-fA-F]+) "
        rf"act_table=(?P<activation_table>0x[0-9a-fA-F]+) "
        rf"out_table=(?P<output_table>0x[0-9a-fA-F]+)"
    )
    layout_callptr_pattern = re.compile(
        rf"\[HM_DEVICE\] layout_callptr family={re.escape(family)} "
        rf"mask=(?P<mask_control>0x[0-9a-fA-F]+) "
        rf"extra=(?P<extra_param>0x[0-9a-fA-F]+) "
        rf"out_desc=(?P<out_desc>0x[0-9a-fA-F]+) "
        rf"act_desc=(?P<act_desc>0x[0-9a-fA-F]+)"
    )
    data_callptr_pattern = re.compile(
        rf"\[HM_DEVICE\] data_callptr family={re.escape(family)} "
        rf"weight=(?P<packed_weight_call>0x[0-9a-fA-F]+) "
        rf"bias=(?P<folded_bias_call>0x[0-9a-fA-F]+) "
        rf"weight_offset=(?P<packed_weight_requested_offset>-?[0-9]+) "
        rf"bias_offset=(?P<folded_bias_requested_offset>-?[0-9]+)"
    )
    hmx_identity_bias_preload_pattern = re.compile(
        rf"\[HM_DEVICE\] hmx_identity_bias_preload family={re.escape(family)} "
        rf"ptr=(?P<ptr>0x[0-9a-fA-F]+) bytes=(?P<bytes>[0-9]+) "
        rf"scale_half=(?P<scale_half>0x[0-9a-fA-F]+) "
        rf"bias_half=(?P<bias_half>0x[0-9a-fA-F]+)"
    )
    dynamic_mask_pattern = re.compile(
        rf"\[HM_DEVICE\] dynamic_mask family={re.escape(family)} "
        rf"mask14=(?P<mask14>0x[0-9a-fA-F]+) "
        rf"extra=(?P<extra>0x[0-9a-fA-F]+) "
        rf"patched=(?P<patched>[0-9]+)"
    )
    hnh_path_pattern = re.compile(
        rf"\[HM_DEVICE\] hnh_path family={re.escape(family)} "
        rf"kernel_entry=(?P<kernel_entry>[a-z0-9_]+) "
        rf"main_path=(?P<main_path>[0-9]+) "
        rf"odd_act_pair=(?P<odd_act_pair>[0-9]+) "
        rf"k_groups=(?P<k_groups>[0-9]+) "
        rf"k_pair_groups=(?P<k_pair_groups>[0-9]+) "
        rf"final_reduce=(?P<final_reduce>[0-9]+) "
        rf"drain_count=(?P<drain_count>[0-9]+) "
        rf"cvt0=(?P<cvt0>[0-9]+) "
        rf"cvt1=(?P<cvt1>[0-9]+) "
        rf"mask12=(?P<mask12>[0-9]+) "
        rf"mask14_ptr_patched=(?P<mask14_ptr_patched>[0-9]+)"
    )
    step_pattern = re.compile(
        rf"\[HM_DEVICE\] step family={re.escape(family)} step=(?P<step>[0-9]+) "
        rf"checksum=(?P<checksum>0x[0-9a-fA-F]+)"
        rf"(?: native=(?P<native>0x[0-9a-fA-F]+) diffs=(?P<diffs>[0-9]+) "
        rf"first_diff=(?P<first>-?[0-9]+))?"
    )
    diff_u16_pattern = re.compile(
        rf"\[HM_DEVICE\] diff_u16 family={re.escape(family)} total=(?P<total>[0-9]+) "
        rf"exact=(?P<exact>[0-9]+) mismatches=(?P<mismatches>[0-9]+) "
        rf"max_abs=(?P<max_abs>[0-9]+) sum_abs=(?P<sum_abs>[0-9]+) "
        rf"first=(?P<first>-?[0-9]+) samples=(?P<samples>[0-9]+)"
    )
    diff_u16_endpoint_pattern = re.compile(
        rf"\[HM_DEVICE\] diff_u16_endpoint family={re.escape(family)} "
        rf"actual_zero=(?P<actual_zero>[0-9]+) actual_ffff=(?P<actual_ffff>[0-9]+) "
        rf"native_zero=(?P<native_zero>[0-9]+) native_ffff=(?P<native_ffff>[0-9]+) "
        rf"mismatch_actual_zero=(?P<mismatch_actual_zero>[0-9]+) "
        rf"mismatch_actual_ffff=(?P<mismatch_actual_ffff>[0-9]+)"
    )
    u16_value_bin_pattern = re.compile(
        rf"\[HM_DEVICE\] u16_value_bin family={re.escape(family)} "
        rf"bin=(?P<bin>[0-9]+) total=(?P<total>[0-9]+) exact=(?P<exact>[0-9]+) "
        rf"actual_zero=(?P<actual_zero>[0-9]+) actual_ffff=(?P<actual_ffff>[0-9]+) "
        rf"actual_mid=(?P<actual_mid>[0-9]+)"
    )
    diff_u16_sample_pattern = re.compile(
        rf"\[HM_DEVICE\] diff_u16_sample family={re.escape(family)} "
        rf"index=(?P<index>[0-9]+) actual=(?P<actual>[0-9]+) "
        rf"native=(?P<native>[0-9]+) abs=(?P<abs>[0-9]+)"
    )
    selected_public_u16_sample_pattern = re.compile(
        rf"\[HM_DEVICE\] selected_u16_sample family={re.escape(family)} scope=public "
        rf"index=(?P<index>[0-9]+) actual=(?P<actual>[0-9]+) "
        rf"native=(?P<native>[0-9]+) abs=(?P<abs>[0-9]+)"
    )
    selected_internal_u16_sample_pattern = re.compile(
        rf"\[HM_DEVICE\] selected_u16_sample family={re.escape(family)} scope=internal "
        rf"block=(?P<block>[0-9]+) index=(?P<index>[0-9]+) "
        rf"actual=(?P<actual>[0-9]+) native=(?P<native>[0-9]+) abs=(?P<abs>[0-9]+)"
    )
    byte_lane_pattern = re.compile(
        rf"\[HM_DEVICE\] byte_lane family={re.escape(family)} "
        rf"lane=(?P<lane>[0-9]+) total=(?P<total>[0-9]+) exact=(?P<exact>[0-9]+) "
        rf"actual_zero=(?P<actual_zero>[0-9]+) actual_ff=(?P<actual_ff>[0-9]+) "
        rf"native_zero=(?P<native_zero>[0-9]+) native_ff=(?P<native_ff>[0-9]+) "
        rf"actual_hash=(?P<actual_hash>0x[0-9a-fA-F]+) native_hash=(?P<native_hash>0x[0-9a-fA-F]+)"
    )
    diff_u16_region_pattern = re.compile(
        rf"\[HM_DEVICE\] diff_u16_region family={re.escape(family)} "
        rf"axis=(?P<axis>[0-9]+) bucket=(?P<bucket>[0-9]+) "
        rf"total=(?P<total>[0-9]+) exact=(?P<exact>[0-9]+) "
        rf"actual_zero=(?P<actual_zero>[0-9]+) actual_ffff=(?P<actual_ffff>[0-9]+) "
        rf"native_zero=(?P<native_zero>[0-9]+) native_ffff=(?P<native_ffff>[0-9]+) "
        rf"actual_hash=(?P<actual_hash>0x[0-9a-fA-F]+) native_hash=(?P<native_hash>0x[0-9a-fA-F]+)"
    )
    internal_block_pattern = re.compile(
        rf"\[HM_DEVICE\] internal_block family={re.escape(family)} "
        rf"block=(?P<block>[0-9]+) exact_bytes=(?P<exact_bytes>[0-9]+) "
        rf"byte_diffs=(?P<byte_diffs>[0-9]+) first_diff=(?P<first_diff>-?[0-9]+) "
        rf"actual_hash=(?P<actual_hash>0x[0-9a-fA-F]+) native_hash=(?P<native_hash>0x[0-9a-fA-F]+)"
    )
    internal_block_u16_value_pattern = re.compile(
        rf"\[HM_DEVICE\] internal_block_u16_value family={re.escape(family)} "
        rf"block=(?P<block>[0-9]+) total=(?P<total>[0-9]+) exact=(?P<exact>[0-9]+) "
        rf"actual_zero=(?P<actual_zero>[0-9]+) actual_ffff=(?P<actual_ffff>[0-9]+) "
        rf"actual_mid=(?P<actual_mid>[0-9]+) native_zero=(?P<native_zero>[0-9]+) "
        rf"native_ffff=(?P<native_ffff>[0-9]+) native_mid=(?P<native_mid>[0-9]+)"
    )
    internal_endpoint_mid_sample_pattern = re.compile(
        rf"\[HM_DEVICE\] internal_endpoint_mid_sample family={re.escape(family)} "
        rf"block=(?P<block>[0-9]+) index=(?P<index>[0-9]+) "
        rf"actual=(?P<actual>[0-9]+) native=(?P<native>[0-9]+)"
    )
    written_block_pattern = re.compile(
        rf"\[HM_DEVICE\] written_block family={re.escape(family)} "
        rf"block=(?P<block>[0-9]+) changed_bytes=(?P<changed_bytes>[0-9]+) "
        rf"first_change=(?P<first_change>-?[0-9]+) "
        rf"actual_hash=(?P<actual_hash>0x[0-9a-fA-F]+) seed_hash=(?P<seed_hash>0x[0-9a-fA-F]+)"
    )
    alt_layout_pattern = re.compile(
        rf"\[HM_DEVICE\] alt_layout family={re.escape(family)} "
        rf"name=(?P<name>[A-Za-z0-9_]+) checksum=(?P<checksum>0x[0-9a-fA-F]+) "
        rf"diffs=(?P<diffs>[0-9]+) first_diff=(?P<first_diff>-?[0-9]+) "
        rf"exact_u16=(?P<exact_u16>[0-9]+)"
    )
    matches = list(pattern.finditer(log))
    if not matches:
        return {
            "entered_and_returned": False,
            "exactness_status": "body_not_entered",
            "raw_log_tail": log.strip().splitlines()[-80:],
        }
    match = matches[-1]
    diffs = int(match.group("diffs"))
    checksum = match.group("checksum").lower()
    native = match.group("native").lower()
    perf_matches = list(perf_pattern.finditer(log))
    perf_match = perf_matches[-1] if perf_matches else None
    timeline_matches = list(timeline_pattern.finditer(log))
    timeline_match = timeline_matches[-1] if timeline_matches else None
    prepared_matches = list(prepared_pattern.finditer(log))
    prepared_match = prepared_matches[-1] if prepared_matches else None
    prepared_checksums = (
        {key: prepared_match.group(key).lower() for key in prepared_match.groupdict()}
        if prepared_match
        else None
    )
    callabi_matches = list(callabi_pattern.finditer(log))
    callabi_match = callabi_matches[-1] if callabi_matches else None
    callabi_scalars = (
        {key: int(value) for key, value in callabi_match.groupdict().items()}
        if callabi_match
        else None
    )
    layout_matches = list(layout_vtcm_pattern.finditer(log))
    layout_match = layout_matches[-1] if layout_matches else None
    dynamic_mask_matches = list(dynamic_mask_pattern.finditer(log))
    dynamic_mask_match = dynamic_mask_matches[-1] if dynamic_mask_matches else None
    dynamic_mask = None
    if dynamic_mask_match:
        dynamic_mask = {
            "mask14": dynamic_mask_match.group("mask14").lower(),
            "extra": dynamic_mask_match.group("extra").lower(),
            "patched": dynamic_mask_match.group("patched") == "1",
            "matches_extra": dynamic_mask_match.group("mask14").lower()
            == dynamic_mask_match.group("extra").lower(),
        }
    hnh_path_matches = list(hnh_path_pattern.finditer(log))
    hnh_path_match = hnh_path_matches[-1] if hnh_path_matches else None
    hnh_path = None
    if hnh_path_match:
        hnh_path = {
            "kernel_entry": hnh_path_match.group("kernel_entry"),
            **{
                key: int(value)
                for key, value in hnh_path_match.groupdict().items()
                if key != "kernel_entry"
            },
        }
    pointer_layout = None
    if layout_match:
        absolute = {key: int(value, 16) for key, value in layout_match.groupdict().items()}
        callptr_matches = list(layout_callptr_pattern.finditer(log))
        callptr_match = callptr_matches[-1] if callptr_matches else None
        if callptr_match:
            absolute.update(
                {key: int(value, 16) for key, value in callptr_match.groupdict().items()}
            )
        base_ptr = absolute["base"]
        deltas = {
            key: (value - base_ptr) & 0xFFFFFFFF
            for key, value in absolute.items()
            if key != "base"
        }
        pointer_layout = {
            "absolute_hex": {key: f"0x{value:08x}" for key, value in absolute.items()},
            "vtcm_delta_bytes": deltas,
        }
    data_callptr_matches = list(data_callptr_pattern.finditer(log))
    data_callptr_match = data_callptr_matches[-1] if data_callptr_matches else None
    data_callptr = None
    if data_callptr_match:
        data_callptr = {
            "absolute_hex": {
                "packed_weight_call": data_callptr_match.group("packed_weight_call").lower(),
                "folded_bias_call": data_callptr_match.group("folded_bias_call").lower(),
            },
            "requested_offsets": {
                "packed_weight": int(data_callptr_match.group("packed_weight_requested_offset")),
                "folded_bias": int(data_callptr_match.group("folded_bias_requested_offset")),
            },
        }
    hmx_identity_bias_preload_matches = list(hmx_identity_bias_preload_pattern.finditer(log))
    hmx_identity_bias_preload_match = (
        hmx_identity_bias_preload_matches[-1] if hmx_identity_bias_preload_matches else None
    )
    hmx_identity_bias_preload = None
    if hmx_identity_bias_preload_match:
        hmx_identity_bias_preload = {
            "absolute_hex": hmx_identity_bias_preload_match.group("ptr").lower(),
            "bytes": int(hmx_identity_bias_preload_match.group("bytes")),
            "scale_half": hmx_identity_bias_preload_match.group("scale_half").lower(),
            "bias_half": hmx_identity_bias_preload_match.group("bias_half").lower(),
        }
    step_trace = []
    for step_match in step_pattern.finditer(log):
        item = {
            "step": int(step_match.group("step")),
            "checksum": step_match.group("checksum").lower(),
        }
        if step_match.group("native") is not None:
            item.update(
                {
                    "native_checksum": step_match.group("native").lower(),
                    "byte_differences": int(step_match.group("diffs")),
                    "first_mismatch_offset": int(step_match.group("first")),
                    "native_exact": int(step_match.group("diffs")) == 0
                    and step_match.group("checksum").lower()
                    == step_match.group("native").lower(),
                }
            )
        step_trace.append(item)
    diff_u16_matches = list(diff_u16_pattern.finditer(log))
    diff_u16_match = diff_u16_matches[-1] if diff_u16_matches else None
    diff_u16_endpoint_matches = list(diff_u16_endpoint_pattern.finditer(log))
    diff_u16_endpoint_match = (
        diff_u16_endpoint_matches[-1] if diff_u16_endpoint_matches else None
    )
    u16_diff_summary = (
        {
            "total_elements": int(diff_u16_match.group("total")),
            "exact_elements": int(diff_u16_match.group("exact")),
            "mismatched_elements": int(diff_u16_match.group("mismatches")),
            "max_absdiff": int(diff_u16_match.group("max_abs")),
            "sum_absdiff": int(diff_u16_match.group("sum_abs")),
            "first_mismatch_index": int(diff_u16_match.group("first")),
            "sample_count": int(diff_u16_match.group("samples")),
        }
        if diff_u16_match
        else None
    )
    if u16_diff_summary is not None and diff_u16_endpoint_match:
        u16_diff_summary.update(
            {
                "actual_zero_elements": int(diff_u16_endpoint_match.group("actual_zero")),
                "actual_ffff_elements": int(diff_u16_endpoint_match.group("actual_ffff")),
                "native_zero_elements": int(diff_u16_endpoint_match.group("native_zero")),
                "native_ffff_elements": int(diff_u16_endpoint_match.group("native_ffff")),
                "mismatch_actual_zero_elements": int(
                    diff_u16_endpoint_match.group("mismatch_actual_zero")
                ),
                "mismatch_actual_ffff_elements": int(
                    diff_u16_endpoint_match.group("mismatch_actual_ffff")
                ),
            }
        )
    u16_diff_samples = [
        {key: int(value) for key, value in sample_match.groupdict().items()}
        for sample_match in diff_u16_sample_pattern.finditer(log)
    ]
    selected_u16_samples = [
        {key: int(value) for key, value in sample_match.groupdict().items()}
        for sample_match in selected_public_u16_sample_pattern.finditer(log)
    ]
    selected_internal_u16_samples = [
        {key: int(value) for key, value in sample_match.groupdict().items()}
        for sample_match in selected_internal_u16_sample_pattern.finditer(log)
    ]
    u16_value_bin_summaries = [
        {
            "bin": int(bin_match.group("bin")),
            "total_elements": int(bin_match.group("total")),
            "exact_elements": int(bin_match.group("exact")),
            "actual_zero_elements": int(bin_match.group("actual_zero")),
            "actual_ffff_elements": int(bin_match.group("actual_ffff")),
            "actual_mid_elements": int(bin_match.group("actual_mid")),
        }
        for bin_match in u16_value_bin_pattern.finditer(log)
    ]
    byte_lane_summaries = [
        {
            "lane": int(lane_match.group("lane")),
            "total_bytes": int(lane_match.group("total")),
            "exact_bytes": int(lane_match.group("exact")),
            "byte_differences": int(lane_match.group("total")) - int(lane_match.group("exact")),
            "actual_zero_bytes": int(lane_match.group("actual_zero")),
            "actual_ff_bytes": int(lane_match.group("actual_ff")),
            "native_zero_bytes": int(lane_match.group("native_zero")),
            "native_ff_bytes": int(lane_match.group("native_ff")),
            "actual_hash": lane_match.group("actual_hash").lower(),
            "native_hash": lane_match.group("native_hash").lower(),
        }
        for lane_match in byte_lane_pattern.finditer(log)
    ]
    axis_names = {
        0: "row4_phase",
        1: "m32_group",
        2: "n32_group",
    }
    u16_region_summaries: dict[str, list[dict]] = {
        name: [] for name in axis_names.values()
    }
    for region_match in diff_u16_region_pattern.finditer(log):
        axis = int(region_match.group("axis"))
        item = {
            "bucket": int(region_match.group("bucket")),
            "total_elements": int(region_match.group("total")),
            "exact_elements": int(region_match.group("exact")),
            "actual_zero_elements": int(region_match.group("actual_zero")),
            "actual_ffff_elements": int(region_match.group("actual_ffff")),
            "native_zero_elements": int(region_match.group("native_zero")),
            "native_ffff_elements": int(region_match.group("native_ffff")),
            "actual_hash": region_match.group("actual_hash").lower(),
            "native_hash": region_match.group("native_hash").lower(),
        }
        u16_region_summaries.setdefault(axis_names.get(axis, f"axis_{axis}"), []).append(item)
    internal_block_summaries = [
        {
            "block": int(block_match.group("block")),
            "exact_bytes": int(block_match.group("exact_bytes")),
            "byte_differences": int(block_match.group("byte_diffs")),
            "first_mismatch_offset": int(block_match.group("first_diff")),
            "actual_hash": block_match.group("actual_hash").lower(),
            "native_hash": block_match.group("native_hash").lower(),
        }
        for block_match in internal_block_pattern.finditer(log)
    ]
    internal_block_u16_value_summaries = [
        {
            "block": int(block_match.group("block")),
            "total_elements": int(block_match.group("total")),
            "exact_elements": int(block_match.group("exact")),
            "actual_zero_elements": int(block_match.group("actual_zero")),
            "actual_ffff_elements": int(block_match.group("actual_ffff")),
            "actual_mid_elements": int(block_match.group("actual_mid")),
            "native_zero_elements": int(block_match.group("native_zero")),
            "native_ffff_elements": int(block_match.group("native_ffff")),
            "native_mid_elements": int(block_match.group("native_mid")),
        }
        for block_match in internal_block_u16_value_pattern.finditer(log)
    ]
    internal_endpoint_mid_samples = [
        {
            "block": int(sample_match.group("block")),
            "index": int(sample_match.group("index")),
            "actual": int(sample_match.group("actual")),
            "native": int(sample_match.group("native")),
        }
        for sample_match in internal_endpoint_mid_sample_pattern.finditer(log)
    ]
    written_block_summaries = [
        {
            "block": int(block_match.group("block")),
            "changed_bytes": int(block_match.group("changed_bytes")),
            "first_change_offset": int(block_match.group("first_change")),
            "actual_hash": block_match.group("actual_hash").lower(),
            "seed_hash": block_match.group("seed_hash").lower(),
        }
        for block_match in written_block_pattern.finditer(log)
    ]
    alt_layout_summaries = [
        {
            "name": alt_match.group("name"),
            "checksum": alt_match.group("checksum").lower(),
            "byte_differences": int(alt_match.group("diffs")),
            "first_mismatch_offset": int(alt_match.group("first_diff")),
            "exact_u16_elements": int(alt_match.group("exact_u16")),
        }
        for alt_match in alt_layout_pattern.finditer(log)
    ]
    return {
        "entered_and_returned": match.group("entered") == "1",
        "output_before_checksum": match.group("before").lower(),
        "output_checksum": checksum,
        "native_raw_checksum": native,
        "output_checksum_scope_bytes": int(match.group("bytes")),
        "byte_differences": diffs,
        "first_mismatch_offset": int(match.group("first")),
        "runtime_us": int(match.group("runtime")),
        "pcycles": int(match.group("pcycles")) if match.group("pcycles") else None,
        "measure_repeats": int(perf_match.group("repeats")) if perf_match else None,
        "empty_pcycles": int(perf_match.group("empty_pcycles")) if perf_match else None,
        "body_pcycles": int(perf_match.group("body_pcycles")) if perf_match else None,
        "net_pcycles": int(perf_match.group("net_pcycles")) if perf_match else None,
        "net_pcycles_per_repeat": int(perf_match.group("net_per_repeat")) if perf_match else None,
        "net_pcycles_per_step": int(perf_match.group("net_per_step")) if perf_match else None,
        "run_qticks": int(timeline_match.group("run_qticks")) if timeline_match else None,
        "empty_qticks": int(timeline_match.group("empty_qticks")) if timeline_match else None,
        "body_qticks": int(timeline_match.group("body_qticks")) if timeline_match else None,
        "net_qticks": int(timeline_match.group("net_qticks")) if timeline_match else None,
        "net_qticks_per_repeat": int(timeline_match.group("net_qticks_per_repeat")) if timeline_match else None,
        "net_qticks_per_step": int(timeline_match.group("net_qticks_per_step")) if timeline_match else None,
        "owned_timeline_evidence": "qtimer_direct_body_span" if timeline_match else None,
        "prepared_state_checksums": prepared_checksums,
        "prepared_state_device_visible": prepared_checksums is not None,
        "call_abi_scalars": callabi_scalars,
        "call_abi_device_visible": callabi_scalars is not None,
        "pointer_layout": pointer_layout,
        "pointer_layout_device_visible": pointer_layout is not None,
        "data_callptr": data_callptr,
        "data_callptr_device_visible": data_callptr is not None,
        "hmx_identity_bias_preload": hmx_identity_bias_preload,
        "hmx_identity_bias_preload_device_visible": hmx_identity_bias_preload is not None,
        "dynamic_mask": dynamic_mask,
        "dynamic_mask_device_visible": dynamic_mask is not None,
        "hnh_path": hnh_path,
        "hnh_path_device_visible": hnh_path is not None,
        "step_trace": step_trace,
        "step_trace_present": bool(step_trace),
        "u16_diff_summary": u16_diff_summary,
        "u16_diff_samples": u16_diff_samples,
        "selected_u16_samples": selected_u16_samples,
        "selected_u16_samples_present": bool(selected_u16_samples),
        "u16_value_bin_summaries": u16_value_bin_summaries,
        "u16_value_bin_summaries_present": bool(u16_value_bin_summaries),
        "byte_lane_summaries": byte_lane_summaries,
        "byte_lane_summaries_present": bool(byte_lane_summaries),
        "u16_region_summaries": u16_region_summaries,
        "u16_region_summaries_present": any(u16_region_summaries.values()),
        "internal_block_summaries": internal_block_summaries,
        "internal_block_summaries_present": bool(internal_block_summaries),
        "internal_block_u16_value_summaries": internal_block_u16_value_summaries,
        "internal_block_u16_value_summaries_present": bool(internal_block_u16_value_summaries),
        "internal_endpoint_mid_samples": internal_endpoint_mid_samples,
        "internal_endpoint_mid_samples_present": bool(internal_endpoint_mid_samples),
        "selected_internal_u16_samples": selected_internal_u16_samples,
        "selected_internal_u16_samples_present": bool(selected_internal_u16_samples),
        "written_block_summaries": written_block_summaries,
        "written_block_summaries_present": bool(written_block_summaries),
        "alt_layout_summaries": alt_layout_summaries,
        "alt_layout_summaries_present": bool(alt_layout_summaries),
        "u16_diff_summary_present": u16_diff_summary is not None,
        "hmx_body_entered": True,
        "device_execution": True,
        "compute_backend": "run_main_on_hexagon_hmx_body",
        "exactness_status": "byte_exact_device_diff" if diffs == 0 and checksum == native else "checksum_mismatch",
        "raw_log_tail": log.strip().splitlines()[-80:],
    }


def run_on_device(device: str, remote_dir: str, binary: Path, timeout_s: int) -> tuple[int, str]:
    remote_abs = ssh_text(
        device,
        f"rm -rf {remote_dir} && mkdir -p {remote_dir} && cd {remote_dir} && pwd",
        timeout=20,
    ).strip().splitlines()[-1]
    ssh_cat_to(device, f"{remote_abs}/run_main_on_hexagon", RUN_MAIN.read_bytes())
    ssh_cat_to(device, f"{remote_abs}/librun_main_on_hexagon_skel.so", RUN_MAIN_SKEL.read_bytes())
    ssh_cat_to(device, f"{remote_abs}/{binary.name}", binary.read_bytes())
    ssh_text(device, f"chmod +x {remote_abs}/run_main_on_hexagon", timeout=20)
    ssh_text(device, "logcat -c >/dev/null 2>&1 || true", timeout=20)
    ssh_text(device, f"echo '0x1f' > {remote_abs}/run_main_on_hexagon.farf", timeout=20)
    library_path = f"{remote_abs}:/vendor/lib64:/system/vendor/lib64:/odm/lib64"
    command = (
        f"cd {remote_abs} && "
        f"DSP_LIBRARY_PATH={remote_abs} LD_LIBRARY_PATH={library_path} ADSP_LIBRARY_PATH={remote_abs} "
        f"./run_main_on_hexagon 3 {binary.name}"
    )
    run = subprocess.run(
        ["ssh", device, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    log = ssh_text(device, "logcat -d -v brief 2>/dev/null | grep '\\[HM_DEVICE\\]' || true", timeout=30)
    return run.returncode, run.stdout + "\n" + log


def parse_internal_u16_sample(text: str) -> tuple[int, int]:
    raw_block, raw_index = text.split(":", 1)
    block = int(raw_block, 0)
    index = int(raw_index, 0)
    if block < 0 or index < 0:
        raise ValueError("BLOCK and INDEX must be non-negative")
    return block, index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="u8i8", choices=sorted(FAMILY_CONFIG))
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--device", default="oneplus")
    parser.add_argument("--remote-dir", default="handwritten_hmx_matmul_device_body")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--measure-repeats", type=int, default=1)
    parser.add_argument(
        "--chain-steps",
        type=int,
        help="Number of direct HMX body calls to run; required for reduced chain1 artifacts",
    )
    parser.add_argument(
        "--step0-native-raw",
        type=Path,
        help="Native raw output used to compare a reduced single-step direct body run",
    )
    parser.add_argument(
        "--reference-raw-override",
        type=Path,
        help="Final public raw output oracle for custom/reduced or chain coverage runs",
    )
    parser.add_argument(
        "--native-internal-raw-override",
        type=Path,
        help="Native internal output oracle used only for internal direct-body comparisons",
    )
    parser.add_argument(
        "--public-output-layout",
        choices=("default", "native_out_block"),
        default="default",
        help="Deblock layout used for the public output comparison",
    )
    parser.add_argument(
        "--native-wrapper-prefetch",
        action="store_true",
        help="Diagnostic W4A16-only path that emits the native wrapper's table dcfetch sequence before each body call",
    )
    parser.add_argument(
        "--pre-clear-acc",
        action="store_true",
        help="Diagnostic W4A16 path that clears HMX accumulators immediately before each body call",
    )
    parser.add_argument(
        "--preload-hmx-identity-bias",
        action="store_true",
        help="Diagnostic W4A16 path that emits tutorial-style mxmem2.bias scale=1/bias=0 setup before the body loop",
    )
    parser.add_argument(
        "--activation-raw-override",
        type=Path,
        help="Diagnostic raw activation payload override",
    )
    parser.add_argument(
        "--folded-bias-raw-override",
        type=Path,
        help="Diagnostic raw folded-bias/control payload override",
    )
    parser.add_argument(
        "--packed-weight-raw-override",
        type=Path,
        help="Diagnostic raw packed-weight/window payload override",
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
        "--descriptor-carrier",
        choices=("separate", "w4a16_hmxi_private_payload"),
        default="separate",
        help="Diagnostic W4A16 descriptor/private-payload carrier layout",
    )
    parser.add_argument(
        "--kernel-entry",
        choices=("deep", "wrapper", "wrapper_nondeep", "split_n128"),
        default="deep",
        help="W4A16 diagnostic entry: direct deep body, native wrapper-entry shim, or two-call N128 split",
    )
    parser.add_argument(
        "--allow-inexact",
        action="store_true",
        help="Return success when the device body enters and returns even if output remains inexact",
    )
    parser.add_argument(
        "--mask-word-override",
        action="append",
        default=[],
        metavar="INDEX=VALUE",
        help="Override one mask/control u32 word for direct HMX ABI diagnostics",
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
    if args.native_wrapper_prefetch and args.family != "w4a16":
        parser.error("--native-wrapper-prefetch is only valid with --family w4a16")
    if args.preload_hmx_identity_bias and args.family != "w4a16":
        parser.error("--preload-hmx-identity-bias is only valid with --family w4a16")
    if args.descriptor_carrier != "separate" and args.family != "w4a16":
        parser.error("--descriptor-carrier is only valid with --family w4a16")
    if args.kernel_entry != "deep" and args.family != "w4a16":
        parser.error("--kernel-entry variants are only valid with --family w4a16")
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
    mask_word_overrides: dict[int, int] = {}
    for item in args.mask_word_override:
        raw_index, raw_value = item.split("=", 1)
        index = int(raw_index, 0)
        value = int(raw_value, 0)
        if index < 0:
            parser.error(f"invalid --mask-word-override {item!r}: INDEX must be non-negative")
        if value < 0 or value > 0xFFFFFFFF:
            parser.error(f"invalid --mask-word-override {item!r}: VALUE must fit in u32")
        mask_word_overrides[index] = value
    step0_native_raw = None
    if args.step0_native_raw:
        step0_native_path = args.step0_native_raw.resolve()
        if not step0_native_path.is_file():
            parser.error(f"missing --step0-native-raw file: {step0_native_path}")
        step0_native_raw = step0_native_path.read_bytes()
    reference_raw_override = None
    if args.reference_raw_override:
        reference_raw_path = args.reference_raw_override.resolve()
        if not reference_raw_path.is_file():
            parser.error(f"missing --reference-raw-override file: {reference_raw_path}")
        reference_raw_override = reference_raw_path.read_bytes()
    native_internal_raw_override = None
    if args.native_internal_raw_override:
        native_internal_path = args.native_internal_raw_override.resolve()
        if not native_internal_path.is_file():
            parser.error(
                f"missing --native-internal-raw-override file: {native_internal_path}"
            )
        native_internal_raw_override = native_internal_path.read_bytes()

    artifact = args.artifact.resolve()
    work = Path(tempfile.mkdtemp(prefix="handwritten_hmx_device_body_"))
    payload = {
        "schema": "handwritten_hmx_matmul_device_body.v1",
        "family": args.family,
        "artifact": rel(artifact),
        "device": args.device,
        "remote_dir": args.remote_dir,
        "qnn_used": False,
        "chain_steps": args.chain_steps,
        "step0_native_raw": rel(args.step0_native_raw.resolve()) if args.step0_native_raw else None,
        "reference_raw_override": (
            rel(args.reference_raw_override.resolve())
            if args.reference_raw_override
            else None
        ),
        "native_internal_raw_override": (
            rel(args.native_internal_raw_override.resolve())
            if args.native_internal_raw_override
            else None
        ),
        "public_output_layout": args.public_output_layout,
        "native_wrapper_prefetch": args.native_wrapper_prefetch,
        "pre_clear_acc": args.pre_clear_acc,
        "preload_hmx_identity_bias": args.preload_hmx_identity_bias,
        "activation_raw_override": (
            rel(args.activation_raw_override.resolve()) if args.activation_raw_override else None
        ),
        "folded_bias_raw_override": (
            rel(args.folded_bias_raw_override.resolve()) if args.folded_bias_raw_override else None
        ),
        "packed_weight_raw_override": (
            rel(args.packed_weight_raw_override.resolve()) if args.packed_weight_raw_override else None
        ),
        "packed_weight_byte_offset": args.packed_weight_byte_offset,
        "folded_bias_byte_offset": args.folded_bias_byte_offset,
        "mask_word_overrides": {str(k): v for k, v in sorted(mask_word_overrides.items())},
        "u16_sample_indices": u16_sample_indices,
        "internal_u16_samples": [
            {"block": block, "index": index} for block, index in internal_u16_samples
        ],
        "focused_sample_log": args.focused_sample_log,
        "descriptor_carrier": args.descriptor_carrier,
        "kernel_entry": args.kernel_entry,
        "compile": {},
        "run": {},
        "result": {},
    }
    try:
        source, binary = build_device_so(
            args.family,
            artifact,
            work,
            args.measure_repeats,
            native_wrapper_prefetch=args.native_wrapper_prefetch,
            pre_clear_acc=args.pre_clear_acc,
            activation_raw_override=args.activation_raw_override.resolve()
            if args.activation_raw_override
            else None,
            packed_weight_byte_offset=args.packed_weight_byte_offset,
            packed_weight_raw_override=args.packed_weight_raw_override.resolve()
            if args.packed_weight_raw_override
            else None,
            folded_bias_raw_override=args.folded_bias_raw_override.resolve()
            if args.folded_bias_raw_override
            else None,
            folded_bias_byte_offset=args.folded_bias_byte_offset,
            descriptor_carrier=args.descriptor_carrier,
            step0_native_raw=step0_native_raw,
            reference_raw_override=reference_raw_override,
            chain_steps=args.chain_steps,
            mask_word_overrides=mask_word_overrides,
            kernel_entry=args.kernel_entry,
            preload_hmx_identity_bias=args.preload_hmx_identity_bias,
            u16_sample_indices=u16_sample_indices,
            internal_u16_samples=internal_u16_samples,
            focused_sample_log=args.focused_sample_log,
            native_internal_raw_override=native_internal_raw_override,
            public_output_layout=args.public_output_layout,
        )
        payload["compile"] = {"source": str(source), "binary": str(binary), "returncode": 0}
        returncode, log = run_on_device(args.device, args.remote_dir, binary, args.timeout_s)
        payload["run"] = {"returncode": returncode, "log_tail": log.strip().splitlines()[-80:]}
        payload["result"] = parse_device_log(log, args.family)
        payload["result"]["native_wrapper_prefetch"] = args.native_wrapper_prefetch
        payload["result"]["pre_clear_acc"] = args.pre_clear_acc
        payload["result"]["preload_hmx_identity_bias"] = args.preload_hmx_identity_bias
        payload["result"]["packed_weight_byte_offset"] = args.packed_weight_byte_offset
        payload["result"]["folded_bias_byte_offset"] = args.folded_bias_byte_offset
        payload["result"]["mask_word_overrides"] = {
            str(k): v for k, v in sorted(mask_word_overrides.items())
        }
        payload["result"]["u16_sample_indices"] = u16_sample_indices
        payload["result"]["internal_u16_samples_requested"] = [
            {"block": block, "index": index} for block, index in internal_u16_samples
        ]
        payload["result"]["focused_sample_log"] = args.focused_sample_log
        payload["result"]["public_output_layout"] = args.public_output_layout
        payload["result"]["descriptor_carrier"] = args.descriptor_carrier
        payload["result"]["kernel_entry"] = args.kernel_entry
    except Exception as exc:
        payload["compile"].setdefault("returncode", 1)
        payload["result"] = {
            "entered_and_returned": False,
            "exactness_status": "runner_error",
            "error": str(exc),
        }
    finally:
        if not args.keep_work:
            # The generated source and binary are large; preserve their paths in
            # JSON only when explicitly requested with --keep-work.
            import shutil

            shutil.rmtree(work, ignore_errors=True)

    result = payload.get("result", {})
    payload["pass"] = result.get("exactness_status") == "byte_exact_device_diff"
    if args.allow_inexact and result.get("entered_and_returned") is True:
        payload["diagnostic_pass"] = True
        payload["acceptance_role"] = "diagnostic_device_entry_not_output_acceptance"
    else:
        payload["diagnostic_pass"] = payload["pass"]
        payload["acceptance_role"] = "device_output_exactness_acceptance"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = payload.get("result", {}).get("exactness_status")
    checksum = payload.get("result", {}).get("output_checksum")
    print(f"{args.family}: device body {status} {checksum or ''}".rstrip())
    return 0 if payload["pass"] or payload.get("diagnostic_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
