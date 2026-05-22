#!/usr/bin/env python3
"""Run a prepared handwritten artifact through an owned HMX body in hexagon-sim.

This is the artifact-body simulator bridge.  It proves that a retained owned
prepared artifact can feed an owned body in the simulator.  It does not claim
byte-exactness or native performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILY_CONFIG = {
    "u8i8": {
        "header": "handwritten_hmx_u8i8_kernel.h",
        "out_desc": "HmU8I8OutDesc",
        "act_desc": "HmU8I8ActDesc",
        "mask_desc": "HmU8I8MaskDesc",
        "function": "hm_u8i8_v73deep_kernel",
    },
    "w4a8": {
        "header": "handwritten_hmx_w4a8_kernel.h",
        "out_desc": "HmW4A8OutDesc",
        "act_desc": "HmW4A8ActDesc",
        "mask_desc": "HmW4A8MaskDesc",
        "function": "hm_w4a8_v73deep_kernel",
    },
    "w8a16": {
        "header": "handwritten_hmx_w8a16_kernel.h",
        "out_desc": "HmW8A16OutDesc",
        "act_desc": "HmW8A16ActDesc",
        "mask_desc": "HmW8A16MaskDesc",
        "function": "hm_w8a16_v75deep_kernel",
    },
    "w4a16": {
        "header": "handwritten_hmx_w4a16_kernel.h",
        "out_desc": "HmW4A16OutDesc",
        "act_desc": "HmW4A16ActDesc",
        "mask_desc": "HmW4A16MaskDesc",
        "function": "hm_w4a16_v73deep_kernel",
    },
    "w16a16": {
        "header": "handwritten_hmx_w16a16_kernel.h",
        "out_desc": "HmW16A16OutDesc",
        "act_desc": "HmW16A16ActDesc",
        "mask_desc": "HmW16A16MaskDesc",
        "function": "hm_w16a16_v73_kernel",
    },
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def tool(name: str) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found)
    candidate = first_existing(
        sorted((ROOT / "tools" / "hexagon-sdk" / "tools" / "HEXAGON_Tools").glob("*/Tools/bin/" + name))
    )
    if candidate is not None:
        return candidate
    raise FileNotFoundError(f"missing tool: {name}")


def resolve_h2_root() -> tuple[Path, Path]:
    h2_install = ROOT / "tools" / "h2-install"
    if h2_install.is_symlink():
        h2_root = h2_install.resolve().parent
    else:
        h2_root = h2_install.parent
    return h2_root, h2_install


def c_array(name: str, data: bytes) -> str:
    chunks = []
    for i in range(0, len(data), 16):
        line = ", ".join(f"0x{b:02x}" for b in data[i : i + 16])
        chunks.append("  " + line)
    body = ",\n".join(chunks)
    return f"static const uint8_t {name}[{len(data)}] = {{\n{body}\n}};\n"


def c_u32_array(name: str, values: list[int]) -> str:
    chunks = []
    for i in range(0, len(values), 8):
        line = ", ".join(f"{int(v)}u" for v in values[i : i + 8])
        chunks.append("  " + line)
    body = ",\n".join(chunks)
    return f"static const uint32_t {name}[{len(values)}] = {{\n{body}\n}};\n"


def read_u32_le(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"u32 table has non-multiple-of-4 byte count: {path}")
    return [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]


def pad(data: bytes, min_size: int) -> bytes:
    if len(data) >= min_size:
        return data
    return data + bytes(min_size - len(data))


def parse_int(text: str) -> int:
    return int(text, 0)


def parse_table_word_overrides(items: list[str], option_name: str) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for item in items:
        try:
            raw_index, raw_value = item.split("=", 1)
            index = int(raw_index, 0)
            value = int(raw_value, 0)
        except ValueError as exc:
            raise ValueError(f"invalid {option_name} {item!r}; expected INDEX=VALUE: {exc}") from exc
        if index < 0:
            raise ValueError(f"invalid {option_name} {item!r}; INDEX must be non-negative")
        if value < 0:
            raise ValueError(f"invalid {option_name} {item!r}; VALUE must be non-negative")
        overrides[index] = value
    return overrides


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def fnv1a32(data: bytes) -> str:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return f"0x{value:08x}"


def classify_exactness(
    entered_and_returned: bool,
    output_checksum: str | None,
    before_checksum: str | None,
    output_bytes: int | None,
    native_raw_bytes: int,
    native_checksum: str,
) -> tuple[str, str]:
    if not entered_and_returned:
        return "body_not_entered", "owned body did not enter and return"
    if not output_checksum or output_bytes is None:
        return "missing_output_checksum", "simulator did not report an output checksum"
    if output_bytes != native_raw_bytes:
        return (
            "not_comparable_size",
            f"simulated output scope is {output_bytes} bytes, native raw output is {native_raw_bytes} bytes",
        )
    if output_checksum.lower() == native_checksum:
        return "byte_exact_checksum", "simulated output checksum matches matched QNN Native raw output"
    if before_checksum and before_checksum.lower() == output_checksum.lower():
        return (
            "unchanged_output",
            "owned body returned but the output checksum did not change from the seed"
        )
    return (
        "checksum_mismatch",
        f"simulated output checksum {output_checksum} does not match native raw {native_checksum}",
    )


def parse_sample_line(output: str) -> dict[str, str]:
    match = re.search(r"^\[SAMPLE\](.*)$", output, re.M)
    if not match:
        return {}
    samples: dict[str, str] = {}
    for item in match.group(1).strip().split():
        if ":" not in item:
            continue
        index, value = item.split(":", 1)
        samples[index] = value.lower()
    return samples


def parse_alt_checksums(output: str) -> dict[str, str]:
    match = re.search(r"^\[ALT\](.*)$", output, re.M)
    if not match:
        return {}
    checksums: dict[str, str] = {}
    for item in match.group(1).strip().split():
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        checksums[name] = value.lower()
    return checksums


def parse_alt_diffs(output: str) -> dict[str, dict[str, int]]:
    diffs: dict[str, dict[str, int]] = {}
    for match in re.finditer(r"^\[ALT_DIFF\]\s+(\S+)(.*)$", output, re.M):
        name = match.group(1)
        parsed: dict[str, int] = {}
        for item in match.group(2).strip().split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                parsed[key] = int(value)
            except ValueError:
                continue
        diffs[name] = parsed
    return diffs


def parse_chain_hashes(output: str) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for match in re.finditer(r"^\[CHAIN_HASH\](.*)$", output, re.M):
        parsed: dict[str, int | str] = {}
        for item in match.group(1).strip().split():
            if "=" not in item:
                continue
            name, value = item.split("=", 1)
            if value.startswith("0x"):
                parsed[name] = value.lower()
                continue
            try:
                parsed[name] = int(value)
            except ValueError:
                parsed[name] = value
        records.append(parsed)
    return records


def parse_split_diff_line(output: str) -> dict[str, int | str]:
    match = re.search(r"^\[SPLIT_DIFF\](.*)$", output, re.M)
    if not match:
        return {}
    parsed: dict[str, int | str] = {}
    for item in match.group(1).strip().split():
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        if value.startswith("0x"):
            parsed[name] = value.lower()
            continue
        try:
            parsed[name] = int(value)
        except ValueError:
            parsed[name] = value
    return parsed


def parse_diff_line(output: str) -> dict[str, int | str]:
    match = re.search(r"^\[DIFF\](.*)$", output, re.M)
    if not match:
        return {}
    parsed: dict[str, int | str] = {}
    for item in match.group(1).strip().split():
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        if value.startswith("0x"):
            parsed[name] = value.lower()
            continue
        try:
            parsed[name] = int(value)
        except ValueError:
            parsed[name] = value
    return parsed


def parse_diff_window(output: str) -> dict[str, int | str]:
    match = re.search(r"^\[DIFF_WINDOW\]\s+first_byte=([0-9]+)\s+got=([0-9a-fA-F]*)\s+ref=([0-9a-fA-F]*)$", output, re.M)
    if not match:
        return {}
    return {
        "first_byte": int(match.group(1)),
        "got_hex": match.group(2).lower(),
        "native_hex": match.group(3).lower(),
    }


def parse_tile_diff(output: str) -> dict[str, int | str]:
    match = re.search(
        r"^\[TILE_DIFF\]\s+(\S+)\s+tile_rows=([0-9]+)\s+tile_cols=([0-9]+)\s+"
        r"nonzero_tiles=([0-9]+)\s+first_tile_m=(-?[0-9]+)\s+first_tile_n=(-?[0-9]+)\s+"
        r"first_tile_bytes=([0-9]+)\s+worst_tile_m=(-?[0-9]+)\s+worst_tile_n=(-?[0-9]+)\s+"
        r"worst_tile_bytes=([0-9]+)$",
        output,
        re.M,
    )
    if not match:
        return {}
    return {
        "mapping": match.group(1),
        "tile_rows": int(match.group(2)),
        "tile_cols": int(match.group(3)),
        "nonzero_tiles": int(match.group(4)),
        "first_tile_m": int(match.group(5)),
        "first_tile_n": int(match.group(6)),
        "first_tile_bytes": int(match.group(7)),
        "worst_tile_m": int(match.group(8)),
        "worst_tile_n": int(match.group(9)),
        "worst_tile_bytes": int(match.group(10)),
    }


def parse_tile_hash_match(output: str) -> dict[str, int | str]:
    match = re.search(
        r"^\[TILE_HASH\]\s+(\S+)\s+tile_count=([0-9]+)\s+same_position=([0-9]+)\s+"
        r"cross_matches=([0-9]+)\s+first_cross_got_m=(-?[0-9]+)\s+"
        r"first_cross_got_n=(-?[0-9]+)\s+first_cross_ref_m=(-?[0-9]+)\s+"
        r"first_cross_ref_n=(-?[0-9]+)$",
        output,
        re.M,
    )
    if not match:
        return {}
    return {
        "mapping": match.group(1),
        "tile_count": int(match.group(2)),
        "same_position": int(match.group(3)),
        "cross_matches": int(match.group(4)),
        "first_cross_got_m": int(match.group(5)),
        "first_cross_got_n": int(match.group(6)),
        "first_cross_ref_m": int(match.group(7)),
        "first_cross_ref_n": int(match.group(8)),
    }


def parse_value_stats(output: str) -> dict[str, int | str]:
    match = re.search(
        r"^\[VALUE_STATS\]\s+(\S+)\s+itemsize=([0-9]+)\s+elements=([0-9]+)\s+"
        r"exact=([0-9]+)\s+got_zero=([0-9]+)\s+ref_zero=([0-9]+)\s+"
        r"got_max=([0-9]+)\s+ref_max=([0-9]+)\s+got_lt_ref=([0-9]+)\s+"
        r"got_gt_ref=([0-9]+)$",
        output,
        re.M,
    )
    if not match:
        return {}
    return {
        "mapping": match.group(1),
        "itemsize": int(match.group(2)),
        "elements": int(match.group(3)),
        "exact": int(match.group(4)),
        "got_zero": int(match.group(5)),
        "ref_zero": int(match.group(6)),
        "got_max": int(match.group(7)),
        "ref_max": int(match.group(8)),
        "got_lt_ref": int(match.group(9)),
        "got_gt_ref": int(match.group(10)),
    }


def parse_axis_diff(output: str) -> dict[str, int | str]:
    match = re.search(
        r"^\[AXIS_DIFF\]\s+(\S+)\s+rows=([0-9]+)\s+cols=([0-9]+)\s+"
        r"nonzero_rows=([0-9]+)\s+first_row=(-?[0-9]+)\s+first_row_elems=([0-9]+)\s+"
        r"worst_row=(-?[0-9]+)\s+worst_row_elems=([0-9]+)\s+"
        r"nonzero_cols=([0-9]+)\s+first_col=(-?[0-9]+)\s+first_col_elems=([0-9]+)\s+"
        r"worst_col=(-?[0-9]+)\s+worst_col_elems=([0-9]+)$",
        output,
        re.M,
    )
    if not match:
        return {}
    return {
        "mapping": match.group(1),
        "rows": int(match.group(2)),
        "cols": int(match.group(3)),
        "nonzero_rows": int(match.group(4)),
        "first_row": int(match.group(5)),
        "first_row_elems": int(match.group(6)),
        "worst_row": int(match.group(7)),
        "worst_row_elems": int(match.group(8)),
        "nonzero_cols": int(match.group(9)),
        "first_col": int(match.group(10)),
        "first_col_elems": int(match.group(11)),
        "worst_col": int(match.group(12)),
        "worst_col_elems": int(match.group(13)),
    }


def parse_segment_diff(output: str) -> dict[str, int | str]:
    match = re.search(
        r"^\[SEGMENT_DIFF\]\s+(\S+)\s+segment_cols=([0-9]+)\s+segments=([0-9]+)\s+"
        r"nonzero_segments=([0-9]+)\s+first_segment=(-?[0-9]+)\s+"
        r"first_segment_elems=([0-9]+)\s+worst_segment=(-?[0-9]+)\s+"
        r"worst_segment_elems=([0-9]+)$",
        output,
        re.M,
    )
    if not match:
        return {}
    return {
        "mapping": match.group(1),
        "segment_cols": int(match.group(2)),
        "segments": int(match.group(3)),
        "nonzero_segments": int(match.group(4)),
        "first_segment": int(match.group(5)),
        "first_segment_elems": int(match.group(6)),
        "worst_segment": int(match.group(7)),
        "worst_segment_elems": int(match.group(8)),
    }


def parse_segment_vector(output: str) -> dict[str, int | str | list[int]]:
    match = re.search(
        r"^\[SEGMENT_VECTOR\]\s+(\S+)\s+segment_cols=([0-9]+)\s+counts=([0-9,]*)$",
        output,
        re.M,
    )
    if not match:
        return {}
    counts = [int(value) for value in match.group(3).split(",") if value]
    return {
        "mapping": match.group(1),
        "segment_cols": int(match.group(2)),
        "counts": counts,
    }


def parse_split_segment_vectors(output: str) -> dict[str, int | str | list[int]]:
    match = re.search(
        r"^\[SPLIT_SEGMENT\]\s+(\S+)\s+segment_cols=([0-9]+)\s+"
        r"left=([0-9,]*)\s+right=([0-9,]*)$",
        output,
        re.M,
    )
    if not match:
        return {}
    return {
        "mapping": match.group(1),
        "segment_cols": int(match.group(2)),
        "left": [int(value) for value in match.group(3).split(",") if value],
        "right": [int(value) for value in match.group(4).split(",") if value],
    }


def parse_phase_segment_matrix(output: str) -> dict[str, int | str | list[list[int]]]:
    match = re.search(
        r"^\[PHASE_SEGMENT\]\s+(\S+)\s+phase_rows=([0-9]+)\s+phases=([0-9]+)\s+"
        r"segment_cols=([0-9]+)\s+segments=([0-9]+)\s+counts=([0-9,;]*)$",
        output,
        re.M,
    )
    if not match:
        return {}
    counts = [
        [int(value) for value in row.split(",") if value]
        for row in match.group(6).split(";")
        if row
    ]
    return {
        "mapping": match.group(1),
        "phase_rows": int(match.group(2)),
        "phases": int(match.group(3)),
        "segment_cols": int(match.group(4)),
        "segments": int(match.group(5)),
        "counts": counts,
    }


def parse_byte_lane_stats(output: str) -> dict[str, int | str]:
    match = re.search(
        r"^\[BYTE_LANE\]\s+(\S+)\s+itemsize=([0-9]+)\s+elements=([0-9]+)\s+"
        r"low_diff=([0-9]+)\s+high_diff=([0-9]+)\s+low_only=([0-9]+)\s+"
        r"high_only=([0-9]+)\s+both_diff=([0-9]+)$",
        output,
        re.M,
    )
    if not match:
        return {}
    return {
        "mapping": match.group(1),
        "itemsize": int(match.group(2)),
        "elements": int(match.group(3)),
        "low_diff": int(match.group(4)),
        "high_diff": int(match.group(5)),
        "low_only": int(match.group(6)),
        "high_only": int(match.group(7)),
        "both_diff": int(match.group(8)),
    }


def parse_simulator_perf(output: str) -> dict:
    threads: dict[str, dict[str, int]] = {}
    for match in re.finditer(r"^\s*T([0-9]+):\s+Insns=([0-9]+)\s+Packets=([0-9]+)$", output, re.M):
        threads[f"T{match.group(1)}"] = {
            "insns": int(match.group(2)),
            "packets": int(match.group(3)),
        }
    total: dict[str, int] = {}
    total_match = re.search(r"^\s*Total:\s+Insns=([0-9]+)\s+Pcycles=([0-9]+)$", output, re.M)
    if total_match:
        total = {
            "insns": int(total_match.group(1)),
            "pcycles": int(total_match.group(2)),
        }
    total_packets = sum(record["packets"] for record in threads.values())
    if total_packets:
        total["packets"] = total_packets
    return {
        "source": "hexagon-sim",
        "threads": threads,
        "total": total,
    }


def native_perf_reference(comparison_scope: dict) -> dict:
    native_kernel = comparison_scope.get("native_kernel", {})
    qnn_aggregate = comparison_scope.get("qnn_aggregate", {})
    timeline = comparison_scope.get("timeline", {})
    packets = [
        int(value)
        for value in native_kernel.get("packets", [])
        if isinstance(value, int)
    ]
    return {
        "source": timeline.get("source"),
        "kernel_event_type": native_kernel.get("event_type"),
        "kernel_event_count": native_kernel.get("event_count"),
        "kernel_cycles_sum": native_kernel.get("cycles_sum"),
        "kernel_packets": native_kernel.get("packets", []),
        "kernel_packets_sum": sum(packets),
        "qnn_aggregate_cycles_sum": qnn_aggregate.get("cycles_sum"),
        "qnn_aggregate_prefix": qnn_aggregate.get("prefix"),
        "timeline_span_cycles": timeline.get("span_cycles"),
    }


def diagnostic_perf_comparison(simulator_perf: dict, native_reference: dict) -> dict:
    sim_total = simulator_perf.get("total", {})
    sim_packets = sim_total.get("packets")
    native_packets = native_reference.get("kernel_packets_sum")
    packet_ratio = None
    if isinstance(sim_packets, int) and isinstance(native_packets, int) and native_packets > 0:
        packet_ratio = sim_packets / native_packets
    return {
        "status": "diagnostic_simulator_not_native_perf",
        "simulator_total_packets": sim_packets,
        "native_kernel_packets_sum": native_packets,
        "simulator_to_native_packet_ratio": packet_ratio,
        "note": "hexagon-sim totals include harness and simulator overhead; native-class acceptance still requires device optrace/perf",
    }


def sample_printer_source(buffer_name: str, byte_count: int, stride: int) -> list[str]:
    max_index = max(0, byte_count - 1)
    sample_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        15,
        16,
        31,
        stride,
        stride + 1,
        stride + 31,
        2 * stride,
        63 * stride,
        64 * stride,
        127 * stride,
        128 * stride,
        max_index,
    ]
    sample_indices = sorted({idx for idx in sample_indices if 0 <= idx <= max_index})
    entries = ", ".join(f"{idx}u" for idx in sample_indices)
    return [
        f"  static const uint32_t sample_idx[] = {{{entries}}};",
        "  printf(\"[SAMPLE]\");",
        "  for (uint32_t si = 0; si < sizeof(sample_idx) / sizeof(sample_idx[0]); ++si) {",
        f"    printf(\" %lu:%02x\", (unsigned long)sample_idx[si], {buffer_name}[sample_idx[si]]);",
        "  }",
        "  printf(\"\\n\");",
    ]


def diff_printer_source(buffer_name: str, byte_count: int, dtype: str) -> list[str]:
    lines = [
        "  uint32_t diff_count = 0u;",
        "  uint32_t first_diff = 0xffffffffu;",
        f"  for (uint32_t i = 0; i < {byte_count}u; ++i) {{",
        f"    if ({buffer_name}[i] != k_native_raw[i]) {{",
        "      if (first_diff == 0xffffffffu) first_diff = i;",
        "      ++diff_count;",
        "    }",
        "  }",
    ]
    if dtype == "uint16_le" and byte_count % 2 == 0:
        lines.extend(
            [
                "  uint32_t elem_diff_count = 0u;",
                "  uint32_t first_elem_diff = 0xffffffffu;",
                "  uint32_t max_abs_diff = 0u;",
                f"  for (uint32_t i = 0; i < {byte_count // 2}u; ++i) {{",
                f"    uint32_t got = (uint32_t){buffer_name}[i * 2u] | ((uint32_t){buffer_name}[i * 2u + 1u] << 8);",
                "    uint32_t ref = (uint32_t)k_native_raw[i * 2u] | ((uint32_t)k_native_raw[i * 2u + 1u] << 8);",
                "    if (got != ref) {",
                "      uint32_t abs_diff = got > ref ? got - ref : ref - got;",
                "      if (first_elem_diff == 0xffffffffu) first_elem_diff = i;",
                "      if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;",
                "      ++elem_diff_count;",
                "    }",
                "  }",
                '  printf("[DIFF] bytes=%lu first_byte=%ld elems=%lu first_elem=%ld max_abs=%lu\\n",',
                "         (unsigned long)diff_count,",
                "         first_diff == 0xffffffffu ? -1L : (long)first_diff,",
                "         (unsigned long)elem_diff_count,",
                "         first_elem_diff == 0xffffffffu ? -1L : (long)first_elem_diff,",
                "         (unsigned long)max_abs_diff);",
            ]
        )
    else:
        lines.extend(
            [
                '  printf("[DIFF] bytes=%lu first_byte=%ld\\n",',
                "         (unsigned long)diff_count,",
                "         first_diff == 0xffffffffu ? -1L : (long)first_diff);",
            ]
        )
    lines.extend(
        [
            "  if (first_diff != 0xffffffffu) {",
            f"    uint32_t window_end = first_diff + 16u > {byte_count}u ? {byte_count}u : first_diff + 16u;",
            '    printf("[DIFF_WINDOW] first_byte=%lu got=", (unsigned long)first_diff);',
            f"    for (uint32_t i = first_diff; i < window_end; ++i) printf(\"%02x\", {buffer_name}[i]);",
            '    printf(" ref=");',
            "    for (uint32_t i = first_diff; i < window_end; ++i) printf(\"%02x\", k_native_raw[i]);",
            '    printf("\\n");',
            "  }",
        ]
    )
    return lines


def alt_diff_printer_function(dtype: str) -> list[str]:
    lines = [
        "static void print_alt_diff(const char *name, const uint8_t *got, const uint8_t *ref, uint32_t n) {",
        "  uint32_t diff_count = 0u;",
        "  uint32_t first_diff = 0xffffffffu;",
        "  for (uint32_t i = 0; i < n; ++i) {",
        "    if (got[i] != ref[i]) {",
        "      if (first_diff == 0xffffffffu) first_diff = i;",
        "      ++diff_count;",
        "    }",
        "  }",
    ]
    if dtype == "uint16_le":
        lines.extend(
            [
                "  uint32_t elem_diff_count = 0u;",
                "  uint32_t first_elem_diff = 0xffffffffu;",
                "  uint32_t max_abs_diff = 0u;",
                "  for (uint32_t i = 0; i < n / 2u; ++i) {",
                "    uint32_t lhs = (uint32_t)got[i * 2u] | ((uint32_t)got[i * 2u + 1u] << 8);",
                "    uint32_t rhs = (uint32_t)ref[i * 2u] | ((uint32_t)ref[i * 2u + 1u] << 8);",
                "    if (lhs != rhs) {",
                "      uint32_t abs_diff = lhs > rhs ? lhs - rhs : rhs - lhs;",
                "      if (first_elem_diff == 0xffffffffu) first_elem_diff = i;",
                "      if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;",
                "      ++elem_diff_count;",
                "    }",
                "  }",
                '  printf("[ALT_DIFF] %s bytes=%lu first_byte=%ld elems=%lu first_elem=%ld max_abs=%lu\\n",',
                "         name,",
                "         (unsigned long)diff_count,",
                "         first_diff == 0xffffffffu ? -1L : (long)first_diff,",
                "         (unsigned long)elem_diff_count,",
                "         first_elem_diff == 0xffffffffu ? -1L : (long)first_elem_diff,",
                "         (unsigned long)max_abs_diff);",
            ]
        )
    else:
        lines.extend(
            [
                '  printf("[ALT_DIFF] %s bytes=%lu first_byte=%ld\\n",',
                "         name,",
                "         (unsigned long)diff_count,",
                "         first_diff == 0xffffffffu ? -1L : (long)first_diff);",
            ]
        )
    lines.append("}")
    return lines


def tile_diff_printer_function(m: int, n: int, itemsize: int) -> list[str]:
    return [
        "static uint32_t load_value(const uint8_t *data, uint32_t idx) {",
        "  if (" + str(itemsize) + "u == 2u) return (uint32_t)data[idx * 2u] | ((uint32_t)data[idx * 2u + 1u] << 8);",
        "  return (uint32_t)data[idx];",
        "}",
        "static void print_value_stats(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t itemsize = {itemsize}u;",
        f"  const uint32_t elements = {m * n}u;",
        "  const uint32_t max_value = itemsize == 2u ? 65535u : 255u;",
        "  uint32_t exact = 0u;",
        "  uint32_t got_zero = 0u;",
        "  uint32_t ref_zero = 0u;",
        "  uint32_t got_max = 0u;",
        "  uint32_t ref_max = 0u;",
        "  uint32_t got_lt_ref = 0u;",
        "  uint32_t got_gt_ref = 0u;",
        "  for (uint32_t i = 0; i < elements; ++i) {",
        "    uint32_t gv = load_value(got, i);",
        "    uint32_t rv = load_value(ref, i);",
        "    if (gv == rv) ++exact;",
        "    if (gv == 0u) ++got_zero;",
        "    if (rv == 0u) ++ref_zero;",
        "    if (gv == max_value) ++got_max;",
        "    if (rv == max_value) ++ref_max;",
        "    if (gv < rv) ++got_lt_ref;",
        "    if (gv > rv) ++got_gt_ref;",
        "  }",
        '  printf("[VALUE_STATS] %s itemsize=%lu elements=%lu exact=%lu got_zero=%lu ref_zero=%lu got_max=%lu ref_max=%lu got_lt_ref=%lu got_gt_ref=%lu\\n",',
        "         name,",
        "         (unsigned long)itemsize,",
        "         (unsigned long)elements,",
        "         (unsigned long)exact,",
        "         (unsigned long)got_zero,",
        "         (unsigned long)ref_zero,",
        "         (unsigned long)got_max,",
        "         (unsigned long)ref_max,",
        "         (unsigned long)got_lt_ref,",
        "         (unsigned long)got_gt_ref);",
        "}",
        "static void print_axis_diff(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t rows = {m}u;",
        f"  const uint32_t cols = {n}u;",
        "  uint32_t nonzero_rows = 0u;",
        "  uint32_t first_row = 0xffffffffu;",
        "  uint32_t first_row_elems = 0u;",
        "  uint32_t worst_row = 0xffffffffu;",
        "  uint32_t worst_row_elems = 0u;",
        "  for (uint32_t row = 0; row < rows; ++row) {",
        "    uint32_t row_elems = 0u;",
        "    for (uint32_t col = 0; col < cols; ++col) {",
        "      uint32_t idx = row * cols + col;",
        "      if (load_value(got, idx) != load_value(ref, idx)) ++row_elems;",
        "    }",
        "    if (row_elems != 0u) {",
        "      if (first_row == 0xffffffffu) { first_row = row; first_row_elems = row_elems; }",
        "      if (row_elems > worst_row_elems) { worst_row = row; worst_row_elems = row_elems; }",
        "      ++nonzero_rows;",
        "    }",
        "  }",
        "  uint32_t nonzero_cols = 0u;",
        "  uint32_t first_col = 0xffffffffu;",
        "  uint32_t first_col_elems = 0u;",
        "  uint32_t worst_col = 0xffffffffu;",
        "  uint32_t worst_col_elems = 0u;",
        "  for (uint32_t col = 0; col < cols; ++col) {",
        "    uint32_t col_elems = 0u;",
        "    for (uint32_t row = 0; row < rows; ++row) {",
        "      uint32_t idx = row * cols + col;",
        "      if (load_value(got, idx) != load_value(ref, idx)) ++col_elems;",
        "    }",
        "    if (col_elems != 0u) {",
        "      if (first_col == 0xffffffffu) { first_col = col; first_col_elems = col_elems; }",
        "      if (col_elems > worst_col_elems) { worst_col = col; worst_col_elems = col_elems; }",
        "      ++nonzero_cols;",
        "    }",
        "  }",
        '  printf("[AXIS_DIFF] %s rows=%lu cols=%lu nonzero_rows=%lu first_row=%ld first_row_elems=%lu worst_row=%ld worst_row_elems=%lu nonzero_cols=%lu first_col=%ld first_col_elems=%lu worst_col=%ld worst_col_elems=%lu\\n",',
        "         name,",
        "         (unsigned long)rows,",
        "         (unsigned long)cols,",
        "         (unsigned long)nonzero_rows,",
        "         first_row == 0xffffffffu ? -1L : (long)first_row,",
        "         (unsigned long)first_row_elems,",
        "         worst_row == 0xffffffffu ? -1L : (long)worst_row,",
        "         (unsigned long)worst_row_elems,",
        "         (unsigned long)nonzero_cols,",
        "         first_col == 0xffffffffu ? -1L : (long)first_col,",
        "         (unsigned long)first_col_elems,",
        "         worst_col == 0xffffffffu ? -1L : (long)worst_col,",
        "         (unsigned long)worst_col_elems);",
        "}",
        "static void print_segment_diff(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t rows = {m}u;",
        f"  const uint32_t cols = {n}u;",
        "  const uint32_t segment_cols = 16u;",
        "  const uint32_t segments = (cols + segment_cols - 1u) / segment_cols;",
        f"  uint32_t segment_counts[{(n + 15) // 16}] = {{0u}};",
        "  uint32_t nonzero_segments = 0u;",
        "  uint32_t first_segment = 0xffffffffu;",
        "  uint32_t first_segment_elems = 0u;",
        "  uint32_t worst_segment = 0xffffffffu;",
        "  uint32_t worst_segment_elems = 0u;",
        "  for (uint32_t segment = 0; segment < segments; ++segment) {",
        "    uint32_t segment_elems = 0u;",
        "    uint32_t col_begin = segment * segment_cols;",
        "    uint32_t col_end = col_begin + segment_cols;",
        "    if (col_end > cols) col_end = cols;",
        "    for (uint32_t row = 0; row < rows; ++row) {",
        "      for (uint32_t col = col_begin; col < col_end; ++col) {",
        "        uint32_t idx = row * cols + col;",
        "        if (load_value(got, idx) != load_value(ref, idx)) ++segment_elems;",
        "      }",
        "    }",
        "    segment_counts[segment] = segment_elems;",
        "    if (segment_elems != 0u) {",
        "      if (first_segment == 0xffffffffu) { first_segment = segment; first_segment_elems = segment_elems; }",
        "      if (segment_elems > worst_segment_elems) { worst_segment = segment; worst_segment_elems = segment_elems; }",
        "      ++nonzero_segments;",
        "    }",
        "  }",
        '  printf("[SEGMENT_DIFF] %s segment_cols=%lu segments=%lu nonzero_segments=%lu first_segment=%ld first_segment_elems=%lu worst_segment=%ld worst_segment_elems=%lu\\n",',
        "         name,",
        "         (unsigned long)segment_cols,",
        "         (unsigned long)segments,",
        "         (unsigned long)nonzero_segments,",
        "         first_segment == 0xffffffffu ? -1L : (long)first_segment,",
        "         (unsigned long)first_segment_elems,",
        "         worst_segment == 0xffffffffu ? -1L : (long)worst_segment,",
        "         (unsigned long)worst_segment_elems);",
        '  printf("[SEGMENT_VECTOR] %s segment_cols=%lu counts=", name, (unsigned long)segment_cols);',
        "  for (uint32_t segment = 0; segment < segments; ++segment) {",
        '    if (segment != 0u) printf(",");',
        '    printf("%lu", (unsigned long)segment_counts[segment]);',
        "  }",
        '  printf("\\n");',
        "}",
        "static void print_split_segment_diff(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t rows = {m}u;",
        f"  const uint32_t cols = {n}u;",
        "  const uint32_t split_cols = cols / 2u;",
        "  const uint32_t segment_cols = 16u;",
        "  const uint32_t split_segments = (split_cols + segment_cols - 1u) / segment_cols;",
        f"  uint32_t left_counts[{((n // 2) + 15) // 16}] = {{0u}};",
        f"  uint32_t right_counts[{((n // 2) + 15) // 16}] = {{0u}};",
        "  for (uint32_t row = 0; row < rows; ++row) {",
        "    for (uint32_t col = 0; col < cols; ++col) {",
        "      uint32_t idx = row * cols + col;",
        "      if (load_value(got, idx) != load_value(ref, idx)) {",
        "        if (col < split_cols) ++left_counts[col / segment_cols];",
        "        else ++right_counts[(col - split_cols) / segment_cols];",
        "      }",
        "    }",
        "  }",
        '  printf("[SPLIT_SEGMENT] %s segment_cols=%lu left=", name, (unsigned long)segment_cols);',
        "  for (uint32_t segment = 0; segment < split_segments; ++segment) {",
        '    if (segment != 0u) printf(",");',
        '    printf("%lu", (unsigned long)left_counts[segment]);',
        "  }",
        '  printf(" right=");',
        "  for (uint32_t segment = 0; segment < split_segments; ++segment) {",
        '    if (segment != 0u) printf(",");',
        '    printf("%lu", (unsigned long)right_counts[segment]);',
        "  }",
        '  printf("\\n");',
        "}",
        "static void print_phase_segment_diff(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t rows = {m}u;",
        f"  const uint32_t cols = {n}u;",
        "  const uint32_t phase_rows = 4u;",
        "  const uint32_t phases = 8u;",
        "  const uint32_t segment_cols = 16u;",
        "  const uint32_t segments = (cols + segment_cols - 1u) / segment_cols;",
        f"  uint32_t counts[8][{(n + 15) // 16}] = {{{{0u}}}};",
        "  for (uint32_t row = 0; row < rows; ++row) {",
        "    uint32_t phase = (row % 32u) / phase_rows;",
        "    for (uint32_t col = 0; col < cols; ++col) {",
        "      uint32_t segment = col / segment_cols;",
        "      uint32_t idx = row * cols + col;",
        "      if (load_value(got, idx) != load_value(ref, idx)) ++counts[phase][segment];",
        "    }",
        "  }",
        '  printf("[PHASE_SEGMENT] %s phase_rows=%lu phases=%lu segment_cols=%lu segments=%lu counts=",',
        "         name,",
        "         (unsigned long)phase_rows,",
        "         (unsigned long)phases,",
        "         (unsigned long)segment_cols,",
        "         (unsigned long)segments);",
        "  for (uint32_t phase = 0; phase < phases; ++phase) {",
        '    if (phase != 0u) printf(";");',
        "    for (uint32_t segment = 0; segment < segments; ++segment) {",
        '      if (segment != 0u) printf(",");',
        '      printf("%lu", (unsigned long)counts[phase][segment]);',
        "    }",
        "  }",
        '  printf("\\n");',
        "}",
        "static void print_byte_lane_stats(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t itemsize = {itemsize}u;",
        f"  const uint32_t elements = {m * n}u;",
        "  uint32_t low_diff = 0u;",
        "  uint32_t high_diff = 0u;",
        "  uint32_t low_only = 0u;",
        "  uint32_t high_only = 0u;",
        "  uint32_t both_diff = 0u;",
        "  for (uint32_t i = 0; i < elements; ++i) {",
        "    uint32_t lo = got[i * itemsize] != ref[i * itemsize];",
        "    uint32_t hi = 0u;",
        "    if (itemsize == 2u) hi = got[i * 2u + 1u] != ref[i * 2u + 1u];",
        "    if (lo) ++low_diff;",
        "    if (hi) ++high_diff;",
        "    if (lo && hi) ++both_diff;",
        "    else if (lo) ++low_only;",
        "    else if (hi) ++high_only;",
        "  }",
        '  printf("[BYTE_LANE] %s itemsize=%lu elements=%lu low_diff=%lu high_diff=%lu low_only=%lu high_only=%lu both_diff=%lu\\n",',
        "         name,",
        "         (unsigned long)itemsize,",
        "         (unsigned long)elements,",
        "         (unsigned long)low_diff,",
        "         (unsigned long)high_diff,",
        "         (unsigned long)low_only,",
        "         (unsigned long)high_only,",
        "         (unsigned long)both_diff);",
        "}",
        "static uint32_t checksum_tile(const uint8_t *data, uint32_t tm, uint32_t tn) {",
        f"  const uint32_t row_bytes = {n * itemsize}u;",
        f"  const uint32_t tile_col_bytes = {32 * itemsize}u;",
        "  uint32_t h = 2166136261u;",
        "  for (uint32_t r = 0; r < 32u; ++r) {",
        "    uint32_t base = (tm * 32u + r) * row_bytes + tn * tile_col_bytes;",
        "    for (uint32_t b = 0; b < tile_col_bytes; ++b) { h ^= data[base + b]; h *= 16777619u; }",
        "  }",
        "  return h;",
        "}",
        "static void print_tile_diff(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t tile_rows = {m // 32}u;",
        f"  const uint32_t tile_cols = {n // 32}u;",
        f"  const uint32_t row_bytes = {n * itemsize}u;",
        f"  const uint32_t tile_col_bytes = {32 * itemsize}u;",
        "  uint32_t nonzero_tiles = 0u;",
        "  uint32_t first_m = 0xffffffffu;",
        "  uint32_t first_n = 0xffffffffu;",
        "  uint32_t first_bytes = 0u;",
        "  uint32_t worst_m = 0xffffffffu;",
        "  uint32_t worst_n = 0xffffffffu;",
        "  uint32_t worst_bytes = 0u;",
        "  for (uint32_t tm = 0; tm < tile_rows; ++tm) {",
        "    for (uint32_t tn = 0; tn < tile_cols; ++tn) {",
        "      uint32_t tile_bytes = 0u;",
        "      for (uint32_t r = 0; r < 32u; ++r) {",
        "        uint32_t base = (tm * 32u + r) * row_bytes + tn * tile_col_bytes;",
        "        for (uint32_t b = 0; b < tile_col_bytes; ++b) {",
        "          if (got[base + b] != ref[base + b]) ++tile_bytes;",
        "        }",
        "      }",
        "      if (tile_bytes != 0u) {",
        "        if (first_m == 0xffffffffu) { first_m = tm; first_n = tn; first_bytes = tile_bytes; }",
        "        if (tile_bytes > worst_bytes) { worst_bytes = tile_bytes; worst_m = tm; worst_n = tn; }",
        "        ++nonzero_tiles;",
        "      }",
        "    }",
        "  }",
        '  printf("[TILE_DIFF] %s tile_rows=%lu tile_cols=%lu nonzero_tiles=%lu first_tile_m=%ld first_tile_n=%ld first_tile_bytes=%lu worst_tile_m=%ld worst_tile_n=%ld worst_tile_bytes=%lu\\n",',
        "         name,",
        "         (unsigned long)tile_rows,",
        "         (unsigned long)tile_cols,",
        "         (unsigned long)nonzero_tiles,",
        "         first_m == 0xffffffffu ? -1L : (long)first_m,",
        "         first_n == 0xffffffffu ? -1L : (long)first_n,",
        "         (unsigned long)first_bytes,",
        "         worst_m == 0xffffffffu ? -1L : (long)worst_m,",
        "         worst_n == 0xffffffffu ? -1L : (long)worst_n,",
        "         (unsigned long)worst_bytes);",
        "}",
        "static void print_tile_hash_match(const char *name, const uint8_t *got, const uint8_t *ref) {",
        f"  const uint32_t tile_rows = {m // 32}u;",
        f"  const uint32_t tile_cols = {n // 32}u;",
        f"  const uint32_t tile_count = {(m // 32) * (n // 32)}u;",
        "  uint32_t same_position = 0u;",
        "  uint32_t cross_matches = 0u;",
        "  uint32_t first_gm = 0xffffffffu;",
        "  uint32_t first_gn = 0xffffffffu;",
        "  uint32_t first_rm = 0xffffffffu;",
        "  uint32_t first_rn = 0xffffffffu;",
        "  for (uint32_t gm = 0; gm < tile_rows; ++gm) {",
        "    for (uint32_t gn = 0; gn < tile_cols; ++gn) {",
        "      uint32_t got_hash = checksum_tile(got, gm, gn);",
        "      for (uint32_t rm = 0; rm < tile_rows; ++rm) {",
        "        for (uint32_t rn = 0; rn < tile_cols; ++rn) {",
        "          if (got_hash == checksum_tile(ref, rm, rn)) {",
        "            if (gm == rm && gn == rn) ++same_position;",
        "            else {",
        "              if (first_gm == 0xffffffffu) { first_gm = gm; first_gn = gn; first_rm = rm; first_rn = rn; }",
        "              ++cross_matches;",
        "            }",
        "          }",
        "        }",
        "      }",
        "    }",
        "  }",
        '  printf("[TILE_HASH] %s tile_count=%lu same_position=%lu cross_matches=%lu first_cross_got_m=%ld first_cross_got_n=%ld first_cross_ref_m=%ld first_cross_ref_n=%ld\\n",',
        "         name,",
        "         (unsigned long)tile_count,",
        "         (unsigned long)same_position,",
        "         (unsigned long)cross_matches,",
        "         first_gm == 0xffffffffu ? -1L : (long)first_gm,",
        "         first_gn == 0xffffffffu ? -1L : (long)first_gn,",
        "         first_rm == 0xffffffffu ? -1L : (long)first_rm,",
        "         first_rn == 0xffffffffu ? -1L : (long)first_rn);",
        "}",
    ]


def generate_source(
    family: str,
    artifact: Path,
    abi: dict,
    oracle: dict,
    native_raw: bytes,
    *,
    chain_override: int | None = None,
    descriptor_overrides: dict[str, int] | None = None,
    mask_word_overrides: dict[int, int] | None = None,
    extra_word_overrides: dict[int, int] | None = None,
    mask_extra_pointer_word: int | None = None,
    weight_byte_offset: int = 0,
    folded_bias_byte_offset: int = 0,
    buffer_layout: str = "default",
    activation_raw_override: Path | None = None,
    activation_table_raw_override: Path | None = None,
    output_table_raw_override: Path | None = None,
    activation_table_word_overrides: dict[int, int] | None = None,
    output_table_word_overrides: dict[int, int] | None = None,
    packed_weight_raw_override: Path | None = None,
    folded_bias_raw_override: Path | None = None,
    output_surface_raw_override: Path | None = None,
    output_raw_out: Path | None = None,
    internal_output_raw_out: Path | None = None,
    hmx_context_mask: int = 0x1,
    skel_mask_helper: bool = False,
    skel_mask_helper_arg6: int = 0xA0,
    native_wrapper_prefetch: bool = False,
    descriptor_carrier: str = "separate",
    pre_clear_acc: bool = False,
    extra_tail_mode: str = "stack",
    table_tail_mode: str = "untouched",
    record_tail_mode: str = "zero",
) -> str:
    config = FAMILY_CONFIG[family]
    prepared = artifact / "prepared_state"
    activation_source = activation_raw_override if activation_raw_override is not None else prepared / "activation.raw"
    activation = activation_source.read_bytes()
    packed_weight_source = packed_weight_raw_override if packed_weight_raw_override is not None else prepared / "packed_weight.raw"
    packed_weight = packed_weight_source.read_bytes()
    folded_bias_source = folded_bias_raw_override if folded_bias_raw_override is not None else prepared / "folded_bias.raw"
    folded_bias = pad(folded_bias_source.read_bytes(), 4096)
    output_surface_source = (
        output_surface_raw_override if output_surface_raw_override is not None else prepared / "output_surface.raw"
    )
    output_surface = pad(output_surface_source.read_bytes(), 4096)
    mask_control = bytearray(pad((prepared / "mask_control.raw").read_bytes(), 64))
    for index, value in (mask_word_overrides or {}).items():
        if index < 0 or index >= 16:
            raise ValueError(f"{family}: mask word override index out of range: {index}")
        mask_control[index * 4 : index * 4 + 4] = int(value).to_bytes(4, "little", signed=False)
    mask_control = bytes(mask_control)
    activation_table_source = (
        activation_table_raw_override if activation_table_raw_override is not None else prepared / "activation_table.raw"
    )
    output_table_source = output_table_raw_override if output_table_raw_override is not None else prepared / "output_table.raw"
    activation_offsets = read_u32_le(activation_table_source)
    output_offsets = read_u32_le(output_table_source)
    out_desc = abi["hexagon_call_abi"]["out_desc_fields"]
    act_desc = abi["hexagon_call_abi"]["act_desc_fields"]
    extra = list(abi["hexagon_call_abi"]["extra_param_words"])
    for index, value in (extra_word_overrides or {}).items():
        if index < 0:
            raise ValueError(f"{family}: extra word override index out of range: {index}")
        if index >= len(extra):
            extra.extend([0] * (index + 1 - len(extra)))
        extra[index] = int(value)
    if mask_extra_pointer_word is not None and (mask_extra_pointer_word < 0 or mask_extra_pointer_word >= 16):
        raise ValueError(f"{family}: mask extra pointer word index out of range: {mask_extra_pointer_word}")
    if weight_byte_offset < 0:
        raise ValueError(f"{family}: weight byte offset must be non-negative")
    if folded_bias_byte_offset < 0:
        raise ValueError(f"{family}: folded-bias byte offset must be non-negative")
    if buffer_layout not in ("default", "w4a16_native_vtcm", "w4a16_custom_qhpi_vtcm"):
        raise ValueError(f"{family}: unsupported buffer layout {buffer_layout!r}")
    if descriptor_carrier not in ("separate", "w4a16_native_record_window", "w4a16_hmxi_private_payload"):
        raise ValueError(f"{family}: unsupported descriptor carrier {descriptor_carrier!r}")
    if descriptor_carrier in ("w4a16_native_record_window", "w4a16_hmxi_private_payload") and family != "w4a16":
        raise ValueError(f"{descriptor_carrier} descriptor carrier is only valid for w4a16")
    if skel_mask_helper and family != "w4a16":
        raise ValueError("--skel-mask-helper is only valid for w4a16")
    if native_wrapper_prefetch and family != "w4a16":
        raise ValueError("--native-wrapper-prefetch is only valid for w4a16")
    if skel_mask_helper_arg6 < 0 or skel_mask_helper_arg6 > 0xFFFFFFFF:
        raise ValueError("--skel-mask-helper-arg6 must fit in u32")
    if extra_tail_mode not in ("stack", "zero", "repeat", "ffff"):
        raise ValueError(f"{family}: unsupported extra tail mode {extra_tail_mode!r}")
    if table_tail_mode not in ("untouched", "zero", "ffff", "repeat_tables", "linear_ptrs"):
        raise ValueError(f"{family}: unsupported table tail mode {table_tail_mode!r}")
    if table_tail_mode != "untouched" and family != "w4a16":
        raise ValueError("--table-tail-mode is only valid with --family w4a16")
    if table_tail_mode != "untouched" and descriptor_carrier != "separate":
        raise ValueError("--table-tail-mode currently requires --descriptor-carrier separate")
    if record_tail_mode not in ("zero", "ffff", "repeat_tables", "linear_ptrs"):
        raise ValueError(f"{family}: unsupported record tail mode {record_tail_mode!r}")
    if record_tail_mode != "zero" and not (family == "w4a16" and descriptor_carrier == "w4a16_native_record_window"):
        raise ValueError("--record-tail-mode is only meaningful with w4a16 --descriptor-carrier w4a16_native_record_window")
    descriptor_overrides = descriptor_overrides or {}
    out_desc_values = {
        "out_table_stride_dwords": int(out_desc["out_table_stride_dwords"]),
        "out_y_stride_words": int(out_desc["out_y_stride_words"]),
        "n_tiles_pow2": int(out_desc["n_tiles_pow2"]),
        "m_total_minus_step": int(out_desc["m_total_minus_step"]),
        "k_total_bytes": int(out_desc["k_total_bytes"]),
    }
    for key in tuple(out_desc_values):
        if key in descriptor_overrides:
            out_desc_values[key] = int(descriptor_overrides[key])
    act_desc_values = {
        "n_act_pairs": int(act_desc["n_act_pairs"]),
        "act_table_y_stride_words": int(act_desc["act_table_y_stride_words"]),
    }
    for key in tuple(act_desc_values):
        if key in descriptor_overrides:
            act_desc_values[key] = int(descriptor_overrides[key])
    extra_init = ", ".join(f"{int(v)}u" for v in extra)
    act_entries = int(abi["activation_table"]["entry_count"])
    out_entries = int(abi["output_table"]["entry_count"])
    if len(activation_offsets) < act_entries:
        raise ValueError(f"{family}: activation_table.raw has fewer entries than ABI manifest")
    if len(output_offsets) < out_entries:
        raise ValueError(f"{family}: output_table.raw has fewer entries than ABI manifest")
    activation_table_word_overrides = activation_table_word_overrides or {}
    output_table_word_overrides = output_table_word_overrides or {}
    for index, value in activation_table_word_overrides.items():
        if family != "w4a16":
            raise ValueError("--activation-table-word-override is only valid with --family w4a16")
        if index >= 1024:
            raise ValueError(f"{family}: activation table word override index out of range: {index}")
        if value >= len(activation):
            raise ValueError(f"{family}: activation table word override offset out of range: {value}")
    for index, value in output_table_word_overrides.items():
        if family != "w4a16":
            raise ValueError("--output-table-word-override is only valid with --family w4a16")
        if index >= 1024:
            raise ValueError(f"{family}: output table word override index out of range: {index}")
        if value >= len(output_surface):
            raise ValueError(f"{family}: output table word override offset out of range: {value}")

    comparison_scope = oracle.get("comparison_scope", {})
    split_policy = comparison_scope.get("accepted_boundary_policy") == "single_custom_op_internal_split_n128"
    native_raw_bytes = int(oracle["raw_output"]["bytes"])
    m, k_total, n = [int(v) for v in oracle["shape_mkn"]]
    chain = int(chain_override if chain_override is not None else oracle.get("chain", 1))
    itemsize_for_samples = 2 if oracle.get("dtype") == "uint16_le" else 1
    public_stride_bytes = n * itemsize_for_samples
    if family == "w16a16" and split_policy and native_raw_bytes == len(output_surface) * 2:
        itemsize = 2
        split_n = n // 2
        if m * split_n * itemsize != len(output_surface):
            raise ValueError(f"{family}: split output surface size does not match shape metadata")
        if buffer_layout != "default":
            raise ValueError(f"{family}: split W16A16 path does not support alternate buffer layout")
        if weight_byte_offset or folded_bias_byte_offset:
            raise ValueError(f"{family}: split W16A16 path does not support sidecar pointer offsets")
        if len(packed_weight) % 2 or len(folded_bias) % 2:
            raise ValueError(f"{family}: split sidecars must have even byte counts")
        weight_split_bytes = len(packed_weight) // 2
        bias_split_bytes = len(folded_bias) // 2
        public_seed = output_surface + output_surface
        row4_output = "row4" in str(abi.get("output_table", {}).get("table_contract", ""))
        row4_deblock_helper = []
        row4_merge = []
        if row4_output:
            row4_deblock_helper = [
                "static void deblock_w16_split_row4(uint8_t *dst, const uint8_t *src, uint32_t col_base) {",
                "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
                f"    for (uint32_t nt = 0; nt < {split_n // 32}u; ++nt) {{",
                "      const uint8_t *block = src + ((row4_phase * " + f"{split_n // 32}u" + " + nt) * 2048u);",
                f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
                "        for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
                "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
                "          uint32_t row1 = row0 + 1u;",
                f"          uint8_t *dst0 = dst + row0 * {n * itemsize}u + col_base + nt * 64u;",
                f"          uint8_t *dst1 = dst + row1 * {n * itemsize}u + col_base + nt * 64u;",
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
                "static void deblock_w16_split_row4_col16_swap(uint8_t *dst, const uint8_t *src, uint32_t col_base) {",
                "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
                f"    for (uint32_t nt = 0; nt < {split_n // 32}u; ++nt) {{",
                "      const uint8_t *block = src + ((row4_phase * " + f"{split_n // 32}u" + " + nt) * 2048u);",
                f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
                "        for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
                "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
                "          uint32_t row1 = row0 + 1u;",
                f"          uint8_t *dst0 = dst + row0 * {n * itemsize}u + col_base + nt * 64u;",
                f"          uint8_t *dst1 = dst + row1 * {n * itemsize}u + col_base + nt * 64u;",
                "          const uint8_t *src_pair = block + (m32_group * 2u + row_pair) * 128u;",
                "          for (uint32_t col = 0; col < 32u; ++col) {",
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
                "static void deblock_w16_split_row4_pair_swap(uint8_t *dst, const uint8_t *src, uint32_t col_base) {",
                "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
                f"    for (uint32_t nt = 0; nt < {split_n // 32}u; ++nt) {{",
                "      const uint8_t *block = src + ((row4_phase * " + f"{split_n // 32}u" + " + nt) * 2048u);",
                f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
                "        for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
                "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
                "          uint32_t row1 = row0 + 1u;",
                f"          uint8_t *dst0 = dst + row0 * {n * itemsize}u + col_base + nt * 64u;",
                f"          uint8_t *dst1 = dst + row1 * {n * itemsize}u + col_base + nt * 64u;",
                "          const uint8_t *src_pair = block + (m32_group * 2u + row_pair) * 128u;",
                "          for (uint32_t col = 0; col < 32u; ++col) {",
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
                "static void deblock_w16_split_row4_pair_major(uint8_t *dst, const uint8_t *src, uint32_t col_base) {",
                "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
                f"    for (uint32_t nt = 0; nt < {split_n // 32}u; ++nt) {{",
                "      const uint8_t *block = src + ((row4_phase * " + f"{split_n // 32}u" + " + nt) * 2048u);",
                "      for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
                f"        for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
                "          uint32_t row0 = m32_group * 32u + row_pair * 16u + row4_phase * 2u;",
                "          uint32_t row1 = row0 + 1u;",
                f"          uint8_t *dst0 = dst + row0 * {n * itemsize}u + col_base + nt * 64u;",
                f"          uint8_t *dst1 = dst + row1 * {n * itemsize}u + col_base + nt * 64u;",
                "          const uint8_t *src_pair = block + (row_pair * " + f"{m // 32}u" + " + m32_group) * 128u;",
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
            ]
            row4_merge = [
                f"  for (uint32_t row = 0; row < {m}u; ++row) {{",
                f"    copy_bytes(public_linear + row * {n * itemsize}u, out0 + row * {split_n * itemsize}u, {split_n * itemsize}u);",
                f"    copy_bytes(public_linear + row * {n * itemsize}u + {split_n * itemsize}u, out1 + row * {split_n * itemsize}u, {split_n * itemsize}u);",
                "  }",
                "  deblock_w16_split_row4(public_out, out0, 0u);",
                f"  deblock_w16_split_row4(public_out, out1, {split_n * itemsize}u);",
            ]
        else:
            row4_merge = [
                f"  for (uint32_t row = 0; row < {m}u; ++row) {{",
                f"    copy_bytes(public_out + row * {n * itemsize}u, out0 + row * {split_n * itemsize}u, {split_n * itemsize}u);",
                f"    copy_bytes(public_out + row * {n * itemsize}u + {split_n * itemsize}u, out1 + row * {split_n * itemsize}u, {split_n * itemsize}u);",
                "  }",
                "  copy_bytes(public_linear, public_out, sizeof(k_output_seed));",
            ]
        split_diagnostics = [
            "  uint32_t raw0_hash = checksum(out0, sizeof(k_output_half_seed));",
            "  uint32_t raw1_hash = checksum(out1, sizeof(k_output_half_seed));",
            "  uint32_t linear_hash = checksum(public_linear, sizeof(k_output_seed));",
            "  uint32_t col16_swap_hash = 0u;",
            "  uint32_t pair_swap_hash = 0u;",
            "  uint32_t pair_major_hash = 0u;",
            *(
                [
                    "  deblock_w16_split_row4_col16_swap(public_alt, out0, 0u);",
                    f"  deblock_w16_split_row4_col16_swap(public_alt, out1, {split_n * itemsize}u);",
                    "  col16_swap_hash = checksum(public_alt, sizeof(k_output_seed));",
                    "  print_alt_diff(\"row4_col16_swap\", public_alt, k_native_raw, sizeof(k_output_seed));",
                    "  copy_bytes(public_alt, k_output_seed, sizeof(k_output_seed));",
                    "  deblock_w16_split_row4_pair_swap(public_alt, out0, 0u);",
                    f"  deblock_w16_split_row4_pair_swap(public_alt, out1, {split_n * itemsize}u);",
                    "  pair_swap_hash = checksum(public_alt, sizeof(k_output_seed));",
                    "  print_alt_diff(\"row4_pair_swap\", public_alt, k_native_raw, sizeof(k_output_seed));",
                    "  copy_bytes(public_alt, k_output_seed, sizeof(k_output_seed));",
                    "  deblock_w16_split_row4_pair_major(public_alt, out0, 0u);",
                    f"  deblock_w16_split_row4_pair_major(public_alt, out1, {split_n * itemsize}u);",
                    "  pair_major_hash = checksum(public_alt, sizeof(k_output_seed));",
                    "  print_alt_diff(\"row4_pair_major\", public_alt, k_native_raw, sizeof(k_output_seed));",
                ]
                if row4_output
                else []
            ),
            "  print_alt_diff(\"linear_public\", public_linear, k_native_raw, sizeof(k_output_seed));",
            "  print_alt_diff(\"row4_public\", public_out, k_native_raw, sizeof(k_output_seed));",
            "  uint32_t left_diff = 0u;",
            "  uint32_t right_diff = 0u;",
            "  uint32_t left_first = 0xffffffffu;",
            "  uint32_t right_first = 0xffffffffu;",
            f"  for (uint32_t row = 0; row < {m}u; ++row) {{",
            f"    for (uint32_t i = 0; i < {split_n * itemsize}u; ++i) {{",
            f"      uint32_t left_idx = row * {n * itemsize}u + i;",
            f"      uint32_t right_idx = row * {n * itemsize}u + {split_n * itemsize}u + i;",
            "      if (public_out[left_idx] != k_native_raw[left_idx]) {",
            "        if (left_first == 0xffffffffu) left_first = left_idx;",
            "        ++left_diff;",
            "      }",
            "      if (public_out[right_idx] != k_native_raw[right_idx]) {",
            "        if (right_first == 0xffffffffu) right_first = right_idx;",
            "        ++right_diff;",
            "      }",
            "    }",
            "  }",
            '  printf("[ALT] raw_split0=0x%08lx raw_split1=0x%08lx linear_public=0x%08lx row4_public=0x%08lx row4_col16_swap=0x%08lx row4_pair_swap=0x%08lx row4_pair_major=0x%08lx\\n",',
            "         (unsigned long)raw0_hash,",
            "         (unsigned long)raw1_hash,",
            "         (unsigned long)linear_hash,",
            "         (unsigned long)out_hash,",
            "         (unsigned long)col16_swap_hash,",
            "         (unsigned long)pair_swap_hash,",
            "         (unsigned long)pair_major_hash);",
            '  printf("[SPLIT_DIFF] left_bytes=%lu left_first=%ld right_bytes=%lu right_first=%ld\\n",',
            "         (unsigned long)left_diff,",
            "         left_first == 0xffffffffu ? -1L : (long)left_first,",
            "         (unsigned long)right_diff,",
            "         right_first == 0xffffffffu ? -1L : (long)right_first);",
        ]
        return "\n".join(
            [
                "#include <stdint.h>",
                "#include <stdio.h>",
                "#include <h2.h>",
                "#include <h2_common_info.h>",
                "#include <h2_mxaccess.h>",
                f'#include "{config["header"]}"',
                "",
                c_array("k_activation", activation),
                c_array("k_packed_weight", packed_weight),
                c_array("k_folded_bias", folded_bias),
                c_array("k_output_half_seed", output_surface),
                c_array("k_output_seed", public_seed),
                c_array("k_native_raw", native_raw),
                c_array("k_mask_control", mask_control),
                c_u32_array("k_activation_offsets", activation_offsets[:act_entries]),
                c_u32_array("k_output_offsets", output_offsets[:out_entries]),
                "static void copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t n) {",
                "  for (uint32_t i = 0; i < n; ++i) dst[i] = src[i];",
                "}",
                "static uint32_t checksum(const uint8_t *data, uint32_t n) {",
                "  uint32_t h = 2166136261u;",
                "  for (uint32_t i = 0; i < n; ++i) { h ^= data[i]; h *= 16777619u; }",
                "  return h;",
                "}",
            "static inline void clear_hmx_acc(void) { __asm__ volatile(\"{ mxclracc }\"); }",
            "static void init_extra_control(uint32_t *extra) {",
            *(
                ["  for (uint32_t i = 0; i < 1024u; ++i) extra[i] = 0u;"]
                if extra_tail_mode in ("stack", "zero")
                else ["  for (uint32_t i = 0; i < 1024u; ++i) extra[i] = 0xffffffffu;"]
                if extra_tail_mode == "ffff"
                else [
                    f"  const uint32_t seed[{len(extra)}] = {{{extra_init}}};",
                    f"  for (uint32_t i = 0; i < 1024u; ++i) extra[i] = seed[i % {len(extra)}u];",
                ]
            ),
            *[f"  extra[{index}u] = {int(value)}u;" for index, value in enumerate(extra)],
            "}",
            "static void init_w4a16_record_tail(uint32_t *record_window, uint8_t *act, uint8_t *out) {",
            *(
                [
                    "  (void)record_window; (void)act; (void)out;",
                ]
                if record_tail_mode == "zero"
                else [
                    "  for (uint32_t i = 64u; i < 96u; ++i) record_window[i] = 0xffffffffu;",
                    "  for (uint32_t i = 198u; i < 496u; ++i) record_window[i] = 0xffffffffu;",
                ]
                if record_tail_mode == "ffff"
                else [
                    "  for (uint32_t i = 64u; i < 96u; ++i) record_window[i] = record_window[i % 64u];",
                    "  for (uint32_t i = 198u; i < 496u; ++i) record_window[i] = record_window[134u + ((i - 198u) % 64u)];",
                ]
                if record_tail_mode == "repeat_tables"
                else [
                    "  for (uint32_t i = 64u; i < 96u; ++i) record_window[i] = (uint32_t)(uintptr_t)(act + ((i - 64u) % 64u) * 0x800u);",
                    "  for (uint32_t i = 198u; i < 496u; ++i) record_window[i] = (uint32_t)(uintptr_t)(out + ((i - 198u) % 64u) * 0x800u);",
                ]
            ),
            "}",
            "static void init_w4a16_table_tail(int32_t *table, uint32_t active_entries, uint8_t *surface) {",
            *(
                [
                    "  (void)table; (void)active_entries; (void)surface;",
                ]
                if table_tail_mode == "untouched"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = 0;",
                ]
                if table_tail_mode == "zero"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = -1;",
                ]
                if table_tail_mode == "ffff"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = table[i % active_entries];",
                ]
                if table_tail_mode == "repeat_tables"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = (int32_t)(uintptr_t)(surface + (i % active_entries) * 0x800u);",
                ]
            ),
            "}",
            *alt_diff_printer_function(oracle.get("dtype", "")),
                *tile_diff_printer_function(m, n, itemsize_for_samples),
                *row4_deblock_helper,
                "int main(void) {",
                "  unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);",
                "  unsigned int vtcm_size = h2_info(INFO_VTCM_SIZE);",
                f'  printf("Handwritten artifact body sim: {family}\\n");',
                '  printf("[Init] VTCM base=0x%08x size=%u KB\\n", vtcm_base, vtcm_size);',
                "  if (vtcm_base == 0 || vtcm_size < 1024u) { h2_thread_stop(1); return 1; }",
                "  h2_mxaccess_state_t mxacc;",
                "  h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0, CFG_HMX_CONTEXTS, 0x1);",
                "  int mret = h2_mxaccess_acquire(&mxacc);",
                '  printf("[Init] HMX acquired (%d)\\n", mret);',
                "  uint8_t *base = (uint8_t *)(uintptr_t)(vtcm_base + 0x10000u);",
                "  uint8_t *act = base + 0x00000u;",
                "  uint8_t *weight = base + 0x20000u;",
                "  uint8_t *bias = base + 0x40000u;",
                "  uint8_t *out0 = base + 0x50000u;",
                "  uint8_t *out1 = base + 0x60000u;",
                "  uint8_t *public_out = base + 0x80000u;",
                "  uint8_t *public_linear = base + 0xc0000u;",
                "  uint8_t *public_alt = base + 0xe0000u;",
                "  int32_t *act_table = (int32_t *)(base + 0xa0000u);",
                "  int32_t *out_table0 = (int32_t *)(base + 0xa1000u);",
                "  int32_t *out_table1 = (int32_t *)(base + 0xa2000u);",
                "  copy_bytes(act, k_activation, sizeof(k_activation));",
                "  copy_bytes(weight, k_packed_weight, sizeof(k_packed_weight));",
                "  copy_bytes(bias, k_folded_bias, sizeof(k_folded_bias));",
                "  copy_bytes(out0, k_output_half_seed, sizeof(k_output_half_seed));",
                "  copy_bytes(out1, k_output_half_seed, sizeof(k_output_half_seed));",
                "  copy_bytes(public_out, k_output_seed, sizeof(k_output_seed));",
                "  copy_bytes(public_linear, k_output_seed, sizeof(k_output_seed));",
                "  copy_bytes(public_alt, k_output_seed, sizeof(k_output_seed));",
                f"  for (uint32_t i = 0; i < {act_entries}u; ++i) act_table[i] = (int32_t)(uintptr_t)(act + k_activation_offsets[i]);",
                f"  for (uint32_t i = 0; i < {out_entries}u; ++i) out_table0[i] = (int32_t)(uintptr_t)(out0 + k_output_offsets[i]);",
                f"  for (uint32_t i = 0; i < {out_entries}u; ++i) out_table1[i] = (int32_t)(uintptr_t)(out1 + k_output_offsets[i]);",
                *[
                    f"  act_table[{index}u] = (int32_t)(uintptr_t)(act + {value}u);"
                    for index, value in sorted(activation_table_word_overrides.items())
                ],
                *[
                    f"  out_table0[{index}u] = (int32_t)(uintptr_t)(out0 + {value}u);"
                    for index, value in sorted(output_table_word_overrides.items())
                ],
                *[
                    f"  out_table1[{index}u] = (int32_t)(uintptr_t)(out1 + {value}u);"
                    for index, value in sorted(output_table_word_overrides.items())
                ],
            "  uint32_t mask_words[16] __attribute__((aligned(16)));",
            "  copy_bytes((uint8_t *)mask_words, k_mask_control, 64u);",
                (
                    f"  uint32_t extra[{len(extra)}] __attribute__((aligned(16))) = {{{extra_init}}};"
                    if extra_tail_mode == "stack"
                    else "  uint32_t extra_storage[1024] __attribute__((aligned(16))); uint32_t *extra = extra_storage; init_extra_control(extra);"
                ),
            "  mask_words[14] = (uint32_t)(uintptr_t)extra;",
                f"  {config['out_desc']} out_desc0 = {{",
                "      out_table0,",
                f"      {out_desc_values['out_table_stride_dwords']}u,",
                f"      {out_desc_values['out_y_stride_words']}u,",
                f"      {out_desc_values['n_tiles_pow2']}u,",
                f"      {out_desc_values['m_total_minus_step']},",
                f"      {out_desc_values['k_total_bytes']}u,",
                "  };",
                f"  {config['out_desc']} out_desc1 = out_desc0;",
                "  out_desc1.out_tile_ptr_table = out_table1;",
                f"  {config['act_desc']} act_desc = {{",
                "      act_table,",
                f"      {act_desc_values['n_act_pairs']}u,",
                f"      {act_desc_values['act_table_y_stride_words']}u,",
                "  };",
                f'  printf("[RUN] {family} artifact body split0\\n");',
                "  uint32_t before_hash = checksum(public_out, sizeof(k_output_seed));",
                '  printf("[INFO] output before checksum=0x%08lx bytes=%lu\\n", (unsigned long)before_hash, (unsigned long)sizeof(k_output_seed));',
                *(['  clear_hmx_acc();'] if pre_clear_acc else []),
                f"  {config['function']}(&out_desc0, &act_desc, weight, bias, (const {config['mask_desc']} *)mask_words, extra);",
                f'  printf("[RUN] {family} artifact body split1\\n");',
                *(['  clear_hmx_acc();'] if pre_clear_acc else []),
                f"  {config['function']}(&out_desc1, &act_desc, weight + {weight_split_bytes}u, bias + {bias_split_bytes}u, (const {config['mask_desc']} *)mask_words, extra);",
                *row4_merge,
                "  uint32_t out_hash = checksum(public_out, sizeof(k_output_seed));",
                *split_diagnostics,
                *diff_printer_source("public_out", native_raw_bytes, oracle.get("dtype", "")),
                '  print_tile_diff("row4_public", public_out, k_native_raw);',
                '  print_tile_hash_match("row4_public", public_out, k_native_raw);',
                '  print_value_stats("row4_public", public_out, k_native_raw);',
                '  print_axis_diff("row4_public", public_out, k_native_raw);',
                '  print_segment_diff("row4_public", public_out, k_native_raw);',
                '  print_split_segment_diff("row4_public", public_out, k_native_raw);',
                '  print_phase_segment_diff("row4_public", public_out, k_native_raw);',
                '  print_byte_lane_stats("row4_public", public_out, k_native_raw);',
                *sample_printer_source("public_out", native_raw_bytes, public_stride_bytes),
                f'  printf("[PASS] {family} artifact body returned checksum=0x%08lx bytes=%lu\\n", (unsigned long)out_hash, (unsigned long)sizeof(k_output_seed));',
                "  h2_thread_stop(0);",
                "  return 0;",
                "}",
                "",
            ]
        )

    a8_public_output = family in ("u8i8", "w4a8")
    a16_crouton_public_output = (
        family in ("w8a16", "w4a16")
        and abi.get("output_table", {}).get("table_contract")
        in ("crouton16_row4_physical_offsets", "crouton16_row4_compact_offsets")
    )
    w4a16_native_compact_output = (
        family == "w4a16"
        and abi.get("output_table", {}).get("table_contract") == "crouton16_row4_compact_offsets"
    )
    a16_row32_public_output = (
        family in ("w8a16", "w4a16")
        and abi.get("output_table", {}).get("table_contract") == "a16_row32_tile_offsets"
    )
    a16_direct_output = (
        family in ("w8a16", "w4a16")
        and abi.get("output_table", {}).get("table_contract") == "a16_direct_row4_offsets"
    )
    internal_public_output = (
        a8_public_output or a16_crouton_public_output or a16_row32_public_output or a16_direct_output
    )
    can_chain_in_place = chain > 1 and len(activation) == len(output_surface)
    w4a16_chain_diagnostics = (
        family == "w4a16"
        and chain >= 1
        and len(activation) == len(output_surface)
        and a16_crouton_public_output
    )
    use_chain_loop = can_chain_in_place or w4a16_chain_diagnostics
    native_record_carrier = descriptor_carrier == "w4a16_native_record_window"
    hmxi_private_payload_carrier = descriptor_carrier == "w4a16_hmxi_private_payload"
    record_window_carrier = native_record_carrier or hmxi_private_payload_carrier
    skel_helper_decls: list[str] = []
    skel_helper_init: list[str] = []
    if skel_mask_helper:
        skel_helper_decls = [
            "extern void skel_set_hmx_params_convw4b1x1(",
            "    uint32_t *params,",
            "    uint32_t arg1,",
            "    uint32_t arg2,",
            "    uint32_t arg3,",
            "    uint32_t arg4,",
            "    uint32_t arg5,",
            "    uint32_t arg6)",
            '    __asm__("_Z25set_hmx_params_convw4b1x1P10hmx_paramsmmmmmm");',
            "static void init_mask_with_skel_helper(uint32_t *mask_words) {",
            (
                f"  skel_set_hmx_params_convw4b1x1(mask_words, 0x70bu, {k_total}u, 0u, 0u, 0u, "
                f"0x{skel_mask_helper_arg6:x}u);"
            ),
            "}",
        ]
        skel_helper_init = [
            "  init_mask_with_skel_helper(mask_words);",
            (
                '  printf("[SKEL_HELPER] set_hmx_params_convw4b1x1 arg1=0x70b '
                f'arg2={k_total} arg3=0 arg4=0 arg5=0 arg6=0x{skel_mask_helper_arg6:x}\\n");'
            ),
        ]
    wrapper_prefetch_decls: list[str] = []
    wrapper_prefetch_call: list[str] = []
    if native_wrapper_prefetch:
        wrapper_prefetch_decls = [
            "static inline void native_dcfetch(const void *ptr) {",
            '  __asm__ volatile("dcfetch(%0+#0)" :: "r"(ptr));',
            "}",
            "static void native_wrapper_prefetch_tables(",
            "    const void *record,",
            "    const void *act_table,",
            "    uint32_t act_entries,",
            "    const void *out_table,",
            "    uint32_t out_entries) {",
            "  if (record != 0) {",
            "    native_dcfetch((const uint8_t *)record + 0u);",
            "    native_dcfetch((const uint8_t *)record + 64u);",
            "    native_dcfetch((const uint8_t *)record + 128u);",
            "  }",
            "  const uint8_t *act = (const uint8_t *)act_table;",
            "  for (uint32_t off = 0u; off < act_entries * 4u; off += 64u) native_dcfetch(act + off);",
            "  const uint8_t *outp = (const uint8_t *)out_table;",
            "  for (uint32_t off = 0u; off < out_entries * 4u; off += 64u) native_dcfetch(outp + off);",
            "}",
        ]
        wrapper_prefetch_call = [
            (
                "  native_wrapper_prefetch_tables("
                + ("record_window" if record_window_carrier else "0")
                + f", act_table, {act_entries}u, out_table, {out_entries}u);"
            ),
            (
                '  printf("[WRAPPER_PREFETCH] record=%s act_entries=%u out_entries=%u\\n", '
                + ('"yes"' if record_window_carrier else '"no"')
                + f", {act_entries}u, {out_entries}u);"
            ),
        ]
    output_dump_lines: list[str] = []
    if output_raw_out is not None:
        output_dump_path = json.dumps(str(output_raw_out))
        output_dump_source = "public_out" if internal_public_output else "out"
        output_dump_lines = [
            f"  FILE *dump = fopen({output_dump_path}, \"wb\");",
            "  if (dump != 0) {",
            f"    fwrite({output_dump_source}, 1u, sizeof(k_output_seed), dump);",
            "    fclose(dump);",
            "  } else {",
            f"    printf(\"[DUMP_ERROR] path=%s\\n\", {output_dump_path});",
            "  }",
        ]
    internal_output_dump_lines: list[str] = []
    if internal_output_raw_out is not None:
        internal_output_dump_path = json.dumps(str(internal_output_raw_out))
        internal_output_dump_lines = [
            f"  FILE *internal_dump = fopen({internal_output_dump_path}, \"wb\");",
            "  if (internal_dump != 0) {",
            "    fwrite(out, 1u, sizeof(k_output_seed), internal_dump);",
            "    fclose(internal_dump);",
            "  } else {",
            f"    printf(\"[DUMP_ERROR] path=%s\\n\", {internal_output_dump_path});",
            "  }",
        ]
    a8_deblock_helper = []
    if a8_public_output:
        a8_deblock_helper = [
            "static void deblock_a8_row64_k32(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t mg = 0; mg < 4u; ++mg) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((mg * " + f"{n // 32}u" + " + nt) * 2048u);",
            "      for (uint32_t r = 0; r < 64u; ++r) {",
            f"        uint8_t *row = dst + ((mg * 64u + r) * {n}u + nt * 32u);",
            "        for (uint32_t c = 0; c < 32u; ++c) row[c] = block[r * 32u + c];",
            "      }",
            "    }",
            "  }",
            "}",
            "static void deblock_a8_crouton8(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row8_group = 0; row8_group < 4u; ++row8_group) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row8_group * " + f"{n // 32}u" + " + nt) * 2048u);",
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
    a16_deblock_helper = []
    if a16_crouton_public_output or a16_direct_output:
        a16_deblock_helper = [
            "static void deblock_a16_crouton16_row4(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * " + f"{n // 32}u" + " + nt) * 2048u);",
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
            *(
                [
                    "static void deblock_w4a16_native_compact(uint8_t *dst, const uint8_t *src) {",
                    f"  for (uint32_t entry = 0; entry < {out_entries}u; ++entry) {{",
                    f"    uint32_t m32_group = entry % {m // 32}u;",
                    f"    uint32_t n_word_base = (entry / {m // 32}u) * 2u;",
                    "    const uint8_t *block = src + k_output_offsets[entry];",
                    "    for (uint32_t src_word = 0; src_word < 512u; ++src_word) {",
                    "      uint32_t row = m32_group * 32u + (src_word % 32u);",
                    "      uint32_t group = src_word / 32u;",
                    f"      uint32_t dst_word = row * {n // 2}u + n_word_base + (group / 2u) * 16u + (group & 1u);",
                    "      copy_bytes(dst + dst_word * 4u, block + src_word * 4u, 4u);",
                    "    }",
                    "  }",
                    "}",
                ]
                if w4a16_native_compact_output
                else []
            ),
            "static void deblock_a16_crouton16_row4_col16_swap(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * " + f"{n // 32}u" + " + nt) * 2048u);",
            f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "        for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
            "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
            "          uint32_t row1 = row0 + 1u;",
            f"          uint8_t *dst0 = dst + row0 * {n * 2}u + nt * 64u;",
            f"          uint8_t *dst1 = dst + row1 * {n * 2}u + nt * 64u;",
            "          const uint8_t *src_pair = block + (m32_group * 2u + row_pair) * 128u;",
            "          for (uint32_t col = 0; col < 32u; ++col) {",
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
            "static void deblock_a16_crouton16_row4_pair_swap(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * " + f"{n // 32}u" + " + nt) * 2048u);",
            f"      for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "        for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
            "          uint32_t row0 = m32_group * 32u + row4_phase * 4u + row_pair * 2u;",
            "          uint32_t row1 = row0 + 1u;",
            f"          uint8_t *dst0 = dst + row0 * {n * 2}u + nt * 64u;",
            f"          uint8_t *dst1 = dst + row1 * {n * 2}u + nt * 64u;",
            "          const uint8_t *src_pair = block + (m32_group * 2u + row_pair) * 128u;",
            "          for (uint32_t col = 0; col < 32u; ++col) {",
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
            "static void deblock_a16_crouton16_row4_pair_major(uint8_t *dst, const uint8_t *src) {",
            "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((row4_phase * " + f"{n // 32}u" + " + nt) * 2048u);",
            "      for (uint32_t row_pair = 0; row_pair < 2u; ++row_pair) {",
            f"        for (uint32_t m32_group = 0; m32_group < {m // 32}u; ++m32_group) {{",
            "          uint32_t row0 = m32_group * 32u + row_pair * 16u + row4_phase * 2u;",
            "          uint32_t row1 = row0 + 1u;",
            f"          uint8_t *dst0 = dst + row0 * {n * 2}u + nt * 64u;",
            f"          uint8_t *dst1 = dst + row1 * {n * 2}u + nt * 64u;",
            "          const uint8_t *src_pair = block + (row_pair * " + f"{m // 32}u" + " + m32_group) * 128u;",
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
        ]
    a16_row32_deblock_helper = []
    if a16_row32_public_output:
        a16_row32_deblock_helper = [
            "static void deblock_a16_row32_tiles(uint8_t *dst, const uint8_t *src) {",
            f"  for (uint32_t mt = 0; mt < {m // 32}u; ++mt) {{",
            f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
            "      const uint8_t *block = src + ((mt * " + f"{n // 32}u" + " + nt) * 2048u);",
            "      for (uint32_t row_sub = 0; row_sub < 32u; ++row_sub) {",
            f"        uint8_t *dst_row = dst + (mt * 32u + row_sub) * {n * 2}u + nt * 64u;",
            "        copy_bytes(dst_row, block + row_sub * 64u, 64u);",
            "      }",
            "    }",
            "  }",
            "}",
        ]
    return "\n".join(
        [
            "#include <stdint.h>",
            "#include <stdio.h>",
            "#include <h2.h>",
            "#include <h2_common_info.h>",
            "#include <h2_mxaccess.h>",
            f'#include "{config["header"]}"',
            "",
            c_array("k_activation", activation),
            c_array("k_packed_weight", packed_weight),
            c_array("k_folded_bias", folded_bias),
            c_array("k_output_seed", output_surface),
            c_array("k_native_raw", native_raw),
            c_array("k_mask_control", mask_control),
            c_u32_array("k_activation_offsets", activation_offsets[:act_entries]),
            c_u32_array("k_output_offsets", output_offsets[:out_entries]),
            "static void copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t n) {",
            "  for (uint32_t i = 0; i < n; ++i) dst[i] = src[i];",
            "}",
            "static uint32_t checksum(const uint8_t *data, uint32_t n) {",
            "  uint32_t h = 2166136261u;",
            "  for (uint32_t i = 0; i < n; ++i) { h ^= data[i]; h *= 16777619u; }",
            "  return h;",
            "}",
            "static inline void clear_hmx_acc(void) { __asm__ volatile(\"{ mxclracc }\"); }",
            "static void init_extra_control(uint32_t *extra) {",
            *(
                ["  for (uint32_t i = 0; i < 1024u; ++i) extra[i] = 0u;"]
                if extra_tail_mode in ("stack", "zero")
                else ["  for (uint32_t i = 0; i < 1024u; ++i) extra[i] = 0xffffffffu;"]
                if extra_tail_mode == "ffff"
                else [
                    f"  const uint32_t seed[{len(extra)}] = {{{extra_init}}};",
                    f"  for (uint32_t i = 0; i < 1024u; ++i) extra[i] = seed[i % {len(extra)}u];",
                ]
            ),
            *[f"  extra[{index}u] = {int(value)}u;" for index, value in enumerate(extra)],
            "}",
            "static void init_w4a16_record_tail(uint32_t *record_window, uint8_t *act, uint8_t *out) {",
            *(
                [
                    "  (void)record_window; (void)act; (void)out;",
                ]
                if record_tail_mode == "zero"
                else [
                    "  for (uint32_t i = 64u; i < 96u; ++i) record_window[i] = 0xffffffffu;",
                    "  for (uint32_t i = 198u; i < 496u; ++i) record_window[i] = 0xffffffffu;",
                ]
                if record_tail_mode == "ffff"
                else [
                    "  for (uint32_t i = 64u; i < 96u; ++i) record_window[i] = record_window[i % 64u];",
                    "  for (uint32_t i = 198u; i < 496u; ++i) record_window[i] = record_window[134u + ((i - 198u) % 64u)];",
                ]
                if record_tail_mode == "repeat_tables"
                else [
                    "  for (uint32_t i = 64u; i < 96u; ++i) record_window[i] = (uint32_t)(uintptr_t)(act + ((i - 64u) % 64u) * 0x800u);",
                    "  for (uint32_t i = 198u; i < 496u; ++i) record_window[i] = (uint32_t)(uintptr_t)(out + ((i - 198u) % 64u) * 0x800u);",
                ]
            ),
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
            "  for (uint32_t i = 0u; i < 26u; ++i) {",
            "    payload[38u + i] = (uint32_t)(uintptr_t)(out + k_output_offsets[i]);",
            "  }",
            "}",
            "static void init_w4a16_table_tail(int32_t *table, uint32_t active_entries, uint8_t *surface) {",
            *(
                [
                    "  (void)table; (void)active_entries; (void)surface;",
                ]
                if table_tail_mode == "untouched"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = 0;",
                ]
                if table_tail_mode == "zero"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = -1;",
                ]
                if table_tail_mode == "ffff"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = table[i % active_entries];",
                ]
                if table_tail_mode == "repeat_tables"
                else [
                    "  for (uint32_t i = active_entries; i < 1024u; ++i) table[i] = (int32_t)(uintptr_t)(surface + (i % active_entries) * 0x800u);",
                ]
            ),
            "}",
            *alt_diff_printer_function(oracle.get("dtype", "")),
            *tile_diff_printer_function(m, n, itemsize_for_samples),
            *skel_helper_decls,
            *wrapper_prefetch_decls,
            *a8_deblock_helper,
            *a16_deblock_helper,
            *a16_row32_deblock_helper,
            "int main(void) {",
            "  unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);",
            "  unsigned int vtcm_size = h2_info(INFO_VTCM_SIZE);",
            f'  printf("Handwritten artifact body sim: {family}\\n");',
            '  printf("[Init] VTCM base=0x%08x size=%u KB\\n", vtcm_base, vtcm_size);',
            "  if (vtcm_base == 0 || vtcm_size < 1024u) { h2_thread_stop(1); return 1; }",
            "  h2_mxaccess_state_t mxacc;",
            f"  h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0, CFG_HMX_CONTEXTS, 0x{hmx_context_mask:x});",
            "  int mret = h2_mxaccess_acquire(&mxacc);",
            '  printf("[Init] HMX acquired (%d)\\n", mret);',
            "  uint8_t *base = (uint8_t *)(uintptr_t)(vtcm_base + 0x10000u);",
            (
                "  uint8_t *act = base + 0x29000u;"
                if buffer_layout == "w4a16_native_vtcm"
                else "  uint8_t *act = base + 0x29800u;"
                if buffer_layout == "w4a16_custom_qhpi_vtcm"
                else "  uint8_t *act = base + 0x00000u;"
            ),
            "  uint8_t *weight = base + 0x20000u;",
            (
                "  uint8_t *bias = base + 0x28000u;"
                if buffer_layout in ("w4a16_native_vtcm", "w4a16_custom_qhpi_vtcm")
                else "  uint8_t *bias = base + 0x40000u;"
            ),
            (
                "  uint8_t *out = base + 0x00000u;"
                if buffer_layout in ("w4a16_native_vtcm", "w4a16_custom_qhpi_vtcm")
                else "  uint8_t *out = base + 0x50000u;"
            ),
            *(['  uint8_t *public_out = base + 0x80000u;'] if internal_public_output else []),
            *(
                [
                    "  uint8_t *record_window = base + 0x70000u;",
                    "  uint8_t *record_base = record_window + 0x180u;",
                    "  int32_t *act_table = (int32_t *)(record_window + 0x000u);",
                    "  int32_t *out_table = (int32_t *)(record_base + 0x098u);",
                ]
                if record_window_carrier
                else [
                    "  int32_t *act_table = (int32_t *)(base + 0x70000u);",
                    "  int32_t *out_table = (int32_t *)(base + 0x71000u);",
                ]
            ),
            "  copy_bytes(act, k_activation, sizeof(k_activation));",
            "  copy_bytes(weight, k_packed_weight, sizeof(k_packed_weight));",
            "  copy_bytes(bias, k_folded_bias, sizeof(k_folded_bias));",
            "  copy_bytes(out, k_output_seed, sizeof(k_output_seed));",
            *(
                [
                    "  for (uint32_t i = 0; i < 1024u; ++i) record_window[i] = 0u;",
                ]
                if record_window_carrier
                else []
            ),
            f"  for (uint32_t i = 0; i < {act_entries}u; ++i) act_table[i] = (int32_t)(uintptr_t)(act + k_activation_offsets[i]);",
            f"  for (uint32_t i = 0; i < {out_entries}u; ++i) out_table[i] = (int32_t)(uintptr_t)(out + k_output_offsets[i]);",
            *(
                [
                    f"  init_w4a16_table_tail(act_table, {act_entries}u, act);",
                    f"  init_w4a16_table_tail(out_table, {out_entries}u, out);",
                    f'  printf("[TABLE_TAIL] mode={table_tail_mode} entries=64 tail_words=960\\n");',
                ]
                if table_tail_mode != "untouched"
                else []
            ),
            *[
                f"  act_table[{index}u] = (int32_t)(uintptr_t)(act + {value}u);"
                for index, value in sorted(activation_table_word_overrides.items())
            ],
            *[
                f"  out_table[{index}u] = (int32_t)(uintptr_t)(out + {value}u);"
                for index, value in sorted(output_table_word_overrides.items())
            ],
            *(
                ["  init_w4a16_record_tail((uint32_t *)record_window, act, out);"]
                if record_window_carrier
                else []
            ),
            (
                "  uint32_t *mask_words = (uint32_t *)(record_base + 0x048u);"
                if record_window_carrier
                else "  uint32_t mask_words[16] __attribute__((aligned(16)));"
            ),
            "  copy_bytes((uint8_t *)mask_words, k_mask_control, 64u);",
            *skel_helper_init,
            (
                "  uint32_t *extra_slot = (uint32_t *)(record_base + 0x080u);"
                " uint32_t *extra = (uint32_t *)(base + 0x72000u);"
                " extra_slot[0] = (uint32_t)(uintptr_t)extra;"
                " init_extra_control(extra);"
                if record_window_carrier
                else (
                    f"  uint32_t extra[{len(extra)}] __attribute__((aligned(16))) = {{{extra_init}}};"
                    if extra_tail_mode == "stack"
                    else "  uint32_t extra_storage[1024] __attribute__((aligned(16))); uint32_t *extra = extra_storage; init_extra_control(extra);"
                )
            ),
            (
                f"  mask_words[{mask_extra_pointer_word}] = (uint32_t)(uintptr_t)extra;"
                if mask_extra_pointer_word is not None
                else "  /* mask extra-pointer diagnostic disabled */"
            ),
            *(
                [
                    f"  {config['act_desc']} *act_desc_ptr = ({config['act_desc']} *)(record_base + 0x010u);",
                    f"  {config['out_desc']} *out_desc_ptr = ({config['out_desc']} *)(record_base + 0x028u);",
                    "  act_desc_ptr->act_ptr_pairs = act_table;",
                    f"  act_desc_ptr->n_act_pairs = {act_desc_values['n_act_pairs']}u;",
                    f"  act_desc_ptr->act_table_y_stride_words = {act_desc_values['act_table_y_stride_words']}u;",
                    "  out_desc_ptr->out_tile_ptr_table = out_table;",
                    f"  out_desc_ptr->out_table_stride_dwords = {out_desc_values['out_table_stride_dwords']}u;",
                    f"  out_desc_ptr->out_y_stride_words = {out_desc_values['out_y_stride_words']}u;",
                    f"  out_desc_ptr->n_tiles_pow2 = {out_desc_values['n_tiles_pow2']}u;",
                    f"  out_desc_ptr->m_total_minus_step = {out_desc_values['m_total_minus_step']};",
                    f"  out_desc_ptr->k_total_bytes = {out_desc_values['k_total_bytes']}u;",
                ]
                if record_window_carrier
                else [
                    f"  {config['out_desc']} out_desc = {{",
                    "      out_table,",
                    f"      {out_desc_values['out_table_stride_dwords']}u,",
                    f"      {out_desc_values['out_y_stride_words']}u,",
                    f"      {out_desc_values['n_tiles_pow2']}u,",
                    f"      {out_desc_values['m_total_minus_step']},",
                    f"      {out_desc_values['k_total_bytes']}u,",
                    "  };",
                    f"  {config['act_desc']} act_desc = {{",
                    "      act_table,",
                    f"      {act_desc_values['n_act_pairs']}u,",
                    f"      {act_desc_values['act_table_y_stride_words']}u,",
                    "  };",
                    f"  {config['act_desc']} *act_desc_ptr = &act_desc;",
                    f"  {config['out_desc']} *out_desc_ptr = &out_desc;",
                ]
            ),
            *(
                [
                    "  init_w4a16_hmxi_private_payload((uint32_t *)record_base, weight, bias, act_table, out_table, out, extra);",
                    '  printf("[HMXI_PAYLOAD] carrier=w4a16_hmxi_private_payload words=64 known_words=61 provisional_context_words=3\\n");',
                ]
                if hmxi_private_payload_carrier
                else []
            ),
            *wrapper_prefetch_call,
            f'  printf("[RUN] {family} artifact body\\n");',
            *(
                [
                    "  deblock_a8_crouton8(public_out, out);",
                    "  uint32_t before_hash = checksum(public_out, sizeof(k_output_seed));",
                ]
                if a8_public_output
                else [
                    "  deblock_a16_crouton16_row4(public_out, out);",
                    "  uint32_t before_hash = checksum(public_out, sizeof(k_output_seed));",
                ]
                if a16_crouton_public_output
                else [
                    "  deblock_a16_row32_tiles(public_out, out);",
                    "  uint32_t before_hash = checksum(public_out, sizeof(k_output_seed));",
                ]
                if a16_row32_public_output
                else [
                    "  uint32_t before_hash = checksum(out, sizeof(k_output_seed));",
                ]
                if a16_direct_output
                else ["  uint32_t before_hash = checksum(out, sizeof(k_output_seed));"]
            ),
            '  printf("[INFO] output before checksum=0x%08lx bytes=%lu\\n", (unsigned long)before_hash, (unsigned long)sizeof(k_output_seed));',
            *(
                [
                    f'  printf("[CHAIN] steps={chain}\\n");',
                    f"  for (uint32_t step = 0; step < {chain}u; ++step) {{",
                    "    copy_bytes(out, k_output_seed, sizeof(k_output_seed));",
                    *(['    clear_hmx_acc();'] if pre_clear_acc else []),
                    f"    {config['function']}(out_desc_ptr, act_desc_ptr, weight + {weight_byte_offset}u, bias + {folded_bias_byte_offset}u, (const {config['mask_desc']} *)mask_words, extra);",
                    *(
                        [
                            "    uint32_t step_raw_hash = checksum(out, sizeof(k_output_seed));",
                            "    deblock_a16_crouton16_row4(public_out, out);",
                            "    uint32_t step_row4_hash = checksum(public_out, sizeof(k_output_seed));",
                            "    deblock_a16_crouton16_row4_pair_swap(public_out, out);",
                            "    uint32_t step_pair_swap_hash = checksum(public_out, sizeof(k_output_seed));",
                            "    deblock_a16_crouton16_row4_pair_major(public_out, out);",
                            "    uint32_t step_pair_major_hash = checksum(public_out, sizeof(k_output_seed));",
                            "    printf(\"[CHAIN_HASH] step=%lu raw=0x%08lx crouton16_row4=0x%08lx crouton16_pair_swap=0x%08lx crouton16_pair_major=0x%08lx\\n\",",
                            "           (unsigned long)(step + 1u),",
                            "           (unsigned long)step_raw_hash,",
                            "           (unsigned long)step_row4_hash,",
                            "           (unsigned long)step_pair_swap_hash,",
                            "           (unsigned long)step_pair_major_hash);",
                        ]
                        if w4a16_chain_diagnostics
                        else []
                    ),
                    f"    if (step + 1u < {chain}u) copy_bytes(act, out, sizeof(k_activation));",
                    "  }",
                ]
                if use_chain_loop
                else [
                    *(['  clear_hmx_acc();'] if pre_clear_acc else []),
                    f"  {config['function']}(out_desc_ptr, act_desc_ptr, weight, bias, (const {config['mask_desc']} *)mask_words, extra);",
                ]
            ),
            *(
                [
                    "  deblock_a8_crouton8(public_out, out);",
                    "  uint32_t out_hash = checksum(public_out, sizeof(k_output_seed));",
                    "  uint32_t raw_hash = checksum(out, sizeof(k_output_seed));",
                    "  deblock_a8_row64_k32(public_out, out);",
                    "  uint32_t row64_hash = checksum(public_out, sizeof(k_output_seed));",
                    "  deblock_a8_crouton8(public_out, out);",
                    "  printf(\"[ALT] raw=0x%08lx row64=0x%08lx crouton8=0x%08lx\\n\", (unsigned long)raw_hash, (unsigned long)row64_hash, (unsigned long)out_hash);",
                    *diff_printer_source("public_out", native_raw_bytes, oracle.get("dtype", "")),
                    '  print_tile_diff("crouton8", public_out, k_native_raw);',
                    '  print_tile_hash_match("crouton8", public_out, k_native_raw);',
                    '  print_value_stats("crouton8", public_out, k_native_raw);',
                    '  print_axis_diff("crouton8", public_out, k_native_raw);',
                    '  print_segment_diff("crouton8", public_out, k_native_raw);',
                    '  print_split_segment_diff("crouton8", public_out, k_native_raw);',
                    '  print_phase_segment_diff("crouton8", public_out, k_native_raw);',
                    '  print_byte_lane_stats("crouton8", public_out, k_native_raw);',
                    *sample_printer_source("public_out", native_raw_bytes, public_stride_bytes),
                ]
                if a8_public_output
                else [
                    "  deblock_a16_crouton16_row4(public_out, out);",
                    "  uint32_t out_hash = checksum(public_out, sizeof(k_output_seed));",
                    "  uint32_t raw_hash = checksum(out, sizeof(k_output_seed));",
                    "  deblock_a16_crouton16_row4_col16_swap(public_out, out);",
                    "  uint32_t col16_swap_hash = checksum(public_out, sizeof(k_output_seed));",
                    "  deblock_a16_crouton16_row4_pair_swap(public_out, out);",
                    "  uint32_t pair_swap_hash = checksum(public_out, sizeof(k_output_seed));",
                    "  deblock_a16_crouton16_row4_pair_major(public_out, out);",
                    "  uint32_t pair_major_hash = checksum(public_out, sizeof(k_output_seed));",
                    *(
                        [
                            "  deblock_w4a16_native_compact(public_out, out);",
                            "  uint32_t native_compact_hash = checksum(public_out, sizeof(k_output_seed));",
                        ]
                        if w4a16_native_compact_output
                        else []
                    ),
                    "  deblock_a16_crouton16_row4(public_out, out);",
                    *(
                        [
                            "  printf(\"[ALT] raw=0x%08lx crouton16_row4=0x%08lx crouton16_col16_swap=0x%08lx crouton16_pair_swap=0x%08lx crouton16_pair_major=0x%08lx native_compact=0x%08lx\\n\", (unsigned long)raw_hash, (unsigned long)out_hash, (unsigned long)col16_swap_hash, (unsigned long)pair_swap_hash, (unsigned long)pair_major_hash, (unsigned long)native_compact_hash);",
                        ]
                        if w4a16_native_compact_output
                        else [
                            "  printf(\"[ALT] raw=0x%08lx crouton16_row4=0x%08lx crouton16_col16_swap=0x%08lx crouton16_pair_swap=0x%08lx crouton16_pair_major=0x%08lx\\n\", (unsigned long)raw_hash, (unsigned long)out_hash, (unsigned long)col16_swap_hash, (unsigned long)pair_swap_hash, (unsigned long)pair_major_hash);",
                        ]
                    ),
                    "  print_alt_diff(\"raw\", out, k_native_raw, sizeof(k_output_seed));",
                    "  print_alt_diff(\"crouton16_row4\", public_out, k_native_raw, sizeof(k_output_seed));",
                    "  deblock_a16_crouton16_row4_col16_swap(public_out, out);",
                    "  print_alt_diff(\"crouton16_col16_swap\", public_out, k_native_raw, sizeof(k_output_seed));",
                    "  deblock_a16_crouton16_row4_pair_swap(public_out, out);",
                    "  print_alt_diff(\"crouton16_pair_swap\", public_out, k_native_raw, sizeof(k_output_seed));",
                    "  deblock_a16_crouton16_row4_pair_major(public_out, out);",
                    "  print_alt_diff(\"crouton16_pair_major\", public_out, k_native_raw, sizeof(k_output_seed));",
                    *(
                        [
                            "  deblock_w4a16_native_compact(public_out, out);",
                            "  print_alt_diff(\"native_compact\", public_out, k_native_raw, sizeof(k_output_seed));",
                        ]
                        if w4a16_native_compact_output
                        else []
                    ),
                    "  deblock_a16_crouton16_row4(public_out, out);",
                    *diff_printer_source("public_out", native_raw_bytes, oracle.get("dtype", "")),
                    '  print_tile_diff("crouton16_row4", public_out, k_native_raw);',
                    '  print_tile_hash_match("crouton16_row4", public_out, k_native_raw);',
                    '  print_value_stats("crouton16_row4", public_out, k_native_raw);',
                    '  print_axis_diff("crouton16_row4", public_out, k_native_raw);',
                    '  print_segment_diff("crouton16_row4", public_out, k_native_raw);',
                    '  print_split_segment_diff("crouton16_row4", public_out, k_native_raw);',
                    '  print_phase_segment_diff("crouton16_row4", public_out, k_native_raw);',
                    '  print_byte_lane_stats("crouton16_row4", public_out, k_native_raw);',
                    *sample_printer_source("public_out", native_raw_bytes, public_stride_bytes),
                ]
                if a16_crouton_public_output
                else [
                    "  deblock_a16_row32_tiles(public_out, out);",
                    "  uint32_t out_hash = checksum(public_out, sizeof(k_output_seed));",
                    "  uint32_t raw_hash = checksum(out, sizeof(k_output_seed));",
                    "  printf(\"[ALT] raw=0x%08lx row32=0x%08lx\\n\", (unsigned long)raw_hash, (unsigned long)out_hash);",
                    *diff_printer_source("public_out", native_raw_bytes, oracle.get("dtype", "")),
                    '  print_tile_diff("row32", public_out, k_native_raw);',
                    '  print_tile_hash_match("row32", public_out, k_native_raw);',
                    '  print_value_stats("row32", public_out, k_native_raw);',
                    '  print_axis_diff("row32", public_out, k_native_raw);',
                    '  print_segment_diff("row32", public_out, k_native_raw);',
                    '  print_split_segment_diff("row32", public_out, k_native_raw);',
                    '  print_phase_segment_diff("row32", public_out, k_native_raw);',
                    '  print_byte_lane_stats("row32", public_out, k_native_raw);',
                    *sample_printer_source("public_out", native_raw_bytes, public_stride_bytes),
                ]
                if a16_row32_public_output
                else [
                    "  uint32_t out_hash = checksum(out, sizeof(k_output_seed));",
                    "  deblock_a16_crouton16_row4(public_out, out);",
                    "  uint32_t crouton_hash = checksum(public_out, sizeof(k_output_seed));",
                    "  printf(\"[ALT] raw=0x%08lx crouton16_row4=0x%08lx\\n\", (unsigned long)out_hash, (unsigned long)crouton_hash);",
                    *diff_printer_source("out", native_raw_bytes, oracle.get("dtype", "")),
                    '  print_tile_diff("raw", out, k_native_raw);',
                    '  print_tile_hash_match("raw", out, k_native_raw);',
                    '  print_value_stats("raw", out, k_native_raw);',
                    '  print_axis_diff("raw", out, k_native_raw);',
                    '  print_segment_diff("raw", out, k_native_raw);',
                    '  print_split_segment_diff("raw", out, k_native_raw);',
                    '  print_phase_segment_diff("raw", out, k_native_raw);',
                    '  print_byte_lane_stats("raw", out, k_native_raw);',
                    *sample_printer_source("out", native_raw_bytes, public_stride_bytes),
                ]
                if a16_direct_output
                else [
                    "  uint32_t out_hash = checksum(out, sizeof(k_output_seed));",
                    *diff_printer_source("out", native_raw_bytes, oracle.get("dtype", "")),
                    '  print_tile_diff("raw", out, k_native_raw);',
                    '  print_tile_hash_match("raw", out, k_native_raw);',
                    '  print_value_stats("raw", out, k_native_raw);',
                    '  print_axis_diff("raw", out, k_native_raw);',
                    '  print_segment_diff("raw", out, k_native_raw);',
                    '  print_split_segment_diff("raw", out, k_native_raw);',
                    '  print_phase_segment_diff("raw", out, k_native_raw);',
                    '  print_byte_lane_stats("raw", out, k_native_raw);',
                    *sample_printer_source("out", native_raw_bytes, public_stride_bytes),
                ]
            ),
            *output_dump_lines,
            *internal_output_dump_lines,
            f'  printf("[PASS] {family} artifact body returned checksum=0x%08lx bytes=%lu\\n", (unsigned long)out_hash, (unsigned long)sizeof(k_output_seed));',
            "  h2_thread_stop(0);",
            "  return 0;",
            "}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="u8i8", choices=sorted(FAMILY_CONFIG))
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument(
        "--output-raw-out",
        type=Path,
        help="Diagnostic path where the simulator program writes its final comparable output bytes",
    )
    parser.add_argument(
        "--internal-output-raw-out",
        type=Path,
        help="Diagnostic path where the simulator program writes the raw internal output surface bytes",
    )
    parser.add_argument("--chain-override", type=int, help="Diagnostic override for generated simulator chain replay")
    parser.add_argument("--native-raw-override", type=Path, help="Diagnostic native raw output override for checksum comparison")
    parser.add_argument("--hmx-context-mask", type=parse_int, default=0x1, help="Diagnostic H2 HMX context acquisition mask")
    parser.add_argument(
        "--skel-mask-helper",
        action="store_true",
        help="Diagnostic W4A16-only path that initializes mask words with QNN skel set_hmx_params_convw4b1x1",
    )
    parser.add_argument(
        "--skel-mask-helper-arg6",
        type=parse_int,
        default=0xA0,
        help="Diagnostic arg6 passed to QNN skel set_hmx_params_convw4b1x1 when --skel-mask-helper is enabled",
    )
    parser.add_argument(
        "--native-wrapper-prefetch",
        action="store_true",
        help="Diagnostic W4A16-only path that emits the native wrapper's record/table dcfetch sequence before body entry",
    )
    parser.add_argument("--out-table-stride-override", type=int, help="Diagnostic output descriptor out_table_stride_dwords override")
    parser.add_argument("--out-y-stride-override", type=int, help="Diagnostic output descriptor out_y_stride_words override")
    parser.add_argument("--n-tiles-pow2-override", type=int, help="Diagnostic output descriptor n_tiles_pow2 override")
    parser.add_argument("--m-total-minus-step-override", type=int, help="Diagnostic output descriptor m_total_minus_step override")
    parser.add_argument("--k-total-bytes-override", type=int, help="Diagnostic output descriptor k_total_bytes override")
    parser.add_argument("--act-n-pairs-override", type=int, help="Diagnostic activation descriptor n_act_pairs override")
    parser.add_argument("--act-table-y-stride-override", type=int, help="Diagnostic activation descriptor act_table_y_stride_words override")
    parser.add_argument(
        "--mask-word-override",
        action="append",
        default=[],
        metavar="INDEX=VALUE",
        help="Diagnostic mask/control u32 word override; VALUE accepts decimal or 0x-prefixed integers",
    )
    parser.add_argument(
        "--extra-word-override",
        action="append",
        default=[],
        metavar="INDEX=VALUE",
        help=(
            "Diagnostic extra_param u32 word override; VALUE accepts decimal or "
            "0x-prefixed integers. Overrides past the ABI seed extend the diagnostic "
            "extra buffer with zero-filled words."
        ),
    )
    parser.add_argument(
        "--mask-extra-pointer-word",
        type=int,
        help="Diagnostic mask/control word index to patch with the extra_param pointer",
    )
    parser.add_argument("--weight-byte-offset", type=int, default=0, help="Diagnostic byte offset added to packed_weight pointer")
    parser.add_argument(
        "--folded-bias-byte-offset",
        type=int,
        default=0,
        help="Diagnostic byte offset added to folded_bias/control pointer",
    )
    parser.add_argument(
        "--buffer-layout",
        choices=("default", "w4a16_native_vtcm", "w4a16_custom_qhpi_vtcm"),
        default="default",
        help="Diagnostic VTCM buffer placement profile",
    )
    parser.add_argument(
        "--descriptor-carrier",
        choices=("separate", "w4a16_native_record_window", "w4a16_hmxi_private_payload"),
        default="separate",
        help="Diagnostic descriptor/table/mask carrier layout profile",
    )
    parser.add_argument(
        "--pre-clear-acc",
        action="store_true",
        help="Diagnostic HMX accumulator clear immediately before each body call",
    )
    parser.add_argument(
        "--extra-tail-mode",
        choices=("stack", "zero", "repeat", "ffff"),
        default="stack",
        help="Diagnostic initialization for r5/extra control words beyond the explicit ABI seed",
    )
    parser.add_argument(
        "--table-tail-mode",
        choices=("untouched", "zero", "ffff", "repeat_tables", "linear_ptrs"),
        default="untouched",
        help="Diagnostic initialization for W4A16 separate activation/output table words after active entries",
    )
    parser.add_argument(
        "--record-tail-mode",
        choices=("zero", "ffff", "repeat_tables", "linear_ptrs"),
        default="zero",
        help="Diagnostic initialization for W4A16 HMXR record-window words outside active descriptors/tables",
    )
    parser.add_argument(
        "--folded-bias-raw-override",
        type=Path,
        help="Diagnostic raw folded-bias/control payload override",
    )
    parser.add_argument(
        "--packed-weight-raw-override",
        type=Path,
        help="Diagnostic raw packed-weight payload override",
    )
    parser.add_argument(
        "--activation-raw-override",
        type=Path,
        help="Diagnostic raw activation surface override",
    )
    parser.add_argument(
        "--activation-table-raw-override",
        type=Path,
        help="Diagnostic raw activation table u32-offset override",
    )
    parser.add_argument(
        "--output-table-raw-override",
        type=Path,
        help="Diagnostic raw output table u32-offset override",
    )
    parser.add_argument(
        "--activation-table-word-override",
        action="append",
        default=[],
        metavar="INDEX=BYTE_OFFSET",
        help="Diagnostic W4A16 table pointer override: act_table[INDEX] = activation + BYTE_OFFSET",
    )
    parser.add_argument(
        "--output-table-word-override",
        action="append",
        default=[],
        metavar="INDEX=BYTE_OFFSET",
        help="Diagnostic W4A16 table pointer override: out_table[INDEX] = output_surface + BYTE_OFFSET",
    )
    parser.add_argument(
        "--output-surface-raw-override",
        type=Path,
        help="Diagnostic raw initial output surface seed override",
    )
    args = parser.parse_args()
    if args.chain_override is not None and args.chain_override < 1:
        parser.error("--chain-override must be >= 1")
    if args.mask_extra_pointer_word is not None and not (0 <= args.mask_extra_pointer_word < 16):
        parser.error("--mask-extra-pointer-word must be in [0, 15]")
    if args.weight_byte_offset < 0:
        parser.error("--weight-byte-offset must be non-negative")
    if args.folded_bias_byte_offset < 0:
        parser.error("--folded-bias-byte-offset must be non-negative")
    if args.hmx_context_mask <= 0:
        parser.error("--hmx-context-mask must be positive")
    if args.skel_mask_helper and args.family != "w4a16":
        parser.error("--skel-mask-helper is only valid with --family w4a16")
    if args.native_wrapper_prefetch and args.family != "w4a16":
        parser.error("--native-wrapper-prefetch is only valid with --family w4a16")
    if args.skel_mask_helper_arg6 < 0 or args.skel_mask_helper_arg6 > 0xFFFFFFFF:
        parser.error("--skel-mask-helper-arg6 must fit in u32")
    descriptor_overrides = {
        key: value
        for key, value in {
            "out_table_stride_dwords": args.out_table_stride_override,
            "out_y_stride_words": args.out_y_stride_override,
            "n_tiles_pow2": args.n_tiles_pow2_override,
            "m_total_minus_step": args.m_total_minus_step_override,
            "k_total_bytes": args.k_total_bytes_override,
            "n_act_pairs": args.act_n_pairs_override,
            "act_table_y_stride_words": args.act_table_y_stride_override,
        }.items()
        if value is not None
    }
    mask_word_overrides: dict[int, int] = {}
    for item in args.mask_word_override:
        try:
            raw_index, raw_value = item.split("=", 1)
            index = int(raw_index, 0)
            value = int(raw_value, 0)
        except ValueError as exc:
            parser.error(f"invalid --mask-word-override {item!r}; expected INDEX=VALUE: {exc}")
        if index < 0 or index >= 16:
            parser.error(f"invalid --mask-word-override {item!r}; INDEX must be in [0, 15]")
        if value < 0 or value > 0xFFFFFFFF:
            parser.error(f"invalid --mask-word-override {item!r}; VALUE must fit in u32")
        mask_word_overrides[index] = value
    extra_word_overrides: dict[int, int] = {}
    for item in args.extra_word_override:
        try:
            raw_index, raw_value = item.split("=", 1)
            index = int(raw_index, 0)
            value = int(raw_value, 0)
        except ValueError as exc:
            parser.error(f"invalid --extra-word-override {item!r}; expected INDEX=VALUE: {exc}")
        if index < 0:
            parser.error(f"invalid --extra-word-override {item!r}; INDEX must be non-negative")
        if value < 0 or value > 0xFFFFFFFF:
            parser.error(f"invalid --extra-word-override {item!r}; VALUE must fit in u32")
        extra_word_overrides[index] = value
    try:
        activation_table_word_overrides = parse_table_word_overrides(
            args.activation_table_word_override,
            "--activation-table-word-override",
        )
        output_table_word_overrides = parse_table_word_overrides(
            args.output_table_word_override,
            "--output-table-word-override",
        )
    except ValueError as exc:
        parser.error(str(exc))

    artifact = args.artifact.resolve()
    abi_path = artifact / "analysis" / "abi_manifest.json"
    errors: list[str] = []
    if not abi_path.is_file():
        errors.append(f"missing ABI manifest: {abi_path}")
    try:
        clang = tool("hexagon-clang")
        sim = tool("hexagon-sim")
    except FileNotFoundError as exc:
        errors.append(str(exc))
        clang = sim = Path("")
    h2_root, h2_install = resolve_h2_root()
    booter = h2_install / "bin" / "booter"
    h2_include = h2_install / "include"
    h2_kernel_include = h2_root / "kernel" / "include"
    h2_lib = h2_install / "lib"
    qnn_skel_lib = ROOT / "tools" / "qnn-sdk" / "lib" / "hexagon-v75" / "unsigned"
    qnn_skel_so = qnn_skel_lib / "libQnnHtpV75Skel.so"
    for path in (booter, h2_include, h2_kernel_include, h2_lib):
        if not path.exists():
            errors.append(f"missing H2 prerequisite: {path}")
    if args.skel_mask_helper and not qnn_skel_so.is_file():
        errors.append(f"missing QNN skel helper library: {qnn_skel_so}")

    payload = {
        "schema": "handwritten_hmx_artifact_body_sim.v1",
        "family": args.family,
        "artifact": str(artifact),
        "qnn_used": bool(args.skel_mask_helper),
        "pass": False,
        "errors": errors,
        "compile": {},
        "simulate": {},
        "result": {},
    }
    if errors:
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    abi = load_json(abi_path)
    prep_compare_path = artifact / "analysis" / "prep_compare.json"
    prep_compare = load_json(prep_compare_path) if prep_compare_path.is_file() else {}
    oracle_manifest = load_json(ROOT / "example" / "handwritten_hmx_matmul" / "oracles.json")
    oracle = oracle_manifest["families"][args.family]
    comparison_scope = oracle.get("comparison_scope", {})
    split_policy = comparison_scope.get("accepted_boundary_policy") == "single_custom_op_internal_split_n128"
    native_kernel_scope = comparison_scope.get("native_kernel", {})
    native_raw_path = args.native_raw_override.resolve() if args.native_raw_override else ROOT / oracle["raw_output"]["path"]
    native_raw = native_raw_path.read_bytes()
    if args.output_raw_out:
        args.output_raw_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.internal_output_raw_out:
        args.internal_output_raw_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="handwritten_hmx_artifact_body_"))
    source = work / "artifact_body_sim.c"
    binary = work / "artifact_body_sim"
    source.write_text(
        generate_source(
            args.family,
            artifact,
            abi,
            oracle,
            native_raw,
            chain_override=args.chain_override,
            descriptor_overrides=descriptor_overrides,
            mask_word_overrides=mask_word_overrides,
            extra_word_overrides=extra_word_overrides,
            mask_extra_pointer_word=args.mask_extra_pointer_word,
            weight_byte_offset=args.weight_byte_offset,
            folded_bias_byte_offset=args.folded_bias_byte_offset,
            buffer_layout=args.buffer_layout,
            activation_raw_override=args.activation_raw_override.resolve() if args.activation_raw_override else None,
            activation_table_raw_override=(
                args.activation_table_raw_override.resolve() if args.activation_table_raw_override else None
            ),
            output_table_raw_override=args.output_table_raw_override.resolve() if args.output_table_raw_override else None,
            activation_table_word_overrides=activation_table_word_overrides,
            output_table_word_overrides=output_table_word_overrides,
            packed_weight_raw_override=args.packed_weight_raw_override.resolve() if args.packed_weight_raw_override else None,
            folded_bias_raw_override=args.folded_bias_raw_override.resolve() if args.folded_bias_raw_override else None,
            output_surface_raw_override=(
                args.output_surface_raw_override.resolve() if args.output_surface_raw_override else None
            ),
            output_raw_out=args.output_raw_out.resolve() if args.output_raw_out else None,
            internal_output_raw_out=(
                args.internal_output_raw_out.resolve() if args.internal_output_raw_out else None
            ),
            hmx_context_mask=args.hmx_context_mask,
            skel_mask_helper=args.skel_mask_helper,
            skel_mask_helper_arg6=args.skel_mask_helper_arg6,
            native_wrapper_prefetch=args.native_wrapper_prefetch,
            descriptor_carrier=args.descriptor_carrier,
            pre_clear_acc=args.pre_clear_acc,
            extra_tail_mode=args.extra_tail_mode,
            table_tail_mode=args.table_tail_mode,
            record_tail_mode=args.record_tail_mode,
        ),
        encoding="utf-8",
    )
    compile_cmd = [
        str(clang),
        "-O2",
        "-mv75",
        "-mhvx",
        "-mhvx-length=128B",
        "-mhmx",
        "-DARCHV=75",
        "-I",
        str(h2_include),
        "-I",
        str(h2_kernel_include),
        "-I",
        str(ROOT / "example" / "handwritten_hmx_matmul" / "include"),
        "-moslib=h2",
        "-Wl,-L," + str(h2_lib),
        "-Wl,--section-start=.start=0x02000000",
        "-o",
        str(binary),
        str(source),
    ]
    if args.skel_mask_helper:
        compile_cmd.extend(
            [
                "-Wl,-call_shared",
                "-Wl,--unresolved-symbols=ignore-all",
                str(qnn_skel_so),
            ]
        )
    compile_result = run(compile_cmd, ROOT)
    payload["compile"] = {
        "command": compile_cmd,
        "returncode": compile_result.returncode,
        "output": compile_result.stdout.strip().splitlines(),
    }
    if compile_result.returncode == 0:
        sim_cmd = [
            str(sim),
            "--mv75",
            "--mhmx",
            "1",
            "--simulated_returnval",
            "--",
            str(booter),
            "--ext_power",
            "1",
            "--use_ext",
            "1",
            "--fence_hi",
            "0xfe000000",
            str(binary),
        ]
        sim_result = run(sim_cmd, ROOT)
    else:
        sim_cmd = []
        sim_result = subprocess.CompletedProcess([], 1, "")
    output = sim_result.stdout
    before_match = re.search(r"^\[INFO\] output before checksum=(0x[0-9a-fA-F]+) bytes=([0-9]+)$", output, re.M)
    chain_match = re.search(r"^\[CHAIN\] steps=([0-9]+)$", output, re.M)
    match = re.search(rf"^\[PASS\] {re.escape(args.family)} artifact body returned checksum=(0x[0-9a-fA-F]+) bytes=([0-9]+)$", output, re.M)
    payload["simulate"] = {
        "command": sim_cmd,
        "returncode": sim_result.returncode,
        "output": output.strip().splitlines(),
    }
    output_bytes = int(match.group(2)) if match else None
    native_checksum = fnv1a32(native_raw)
    entered_and_returned = bool(match)
    output_checksum = match.group(1) if match else None
    before_checksum = before_match.group(1) if before_match else None
    exactness_status, exactness_blocker = classify_exactness(
        entered_and_returned,
        output_checksum,
        before_checksum,
        output_bytes,
        len(native_raw),
        native_checksum,
    )
    simulator_perf = parse_simulator_perf(output)
    native_reference = native_perf_reference(comparison_scope)
    payload["result"] = {
        "entered_and_returned": entered_and_returned,
        "output_before_checksum": before_checksum,
        "output_checksum": output_checksum,
        "output_changed": (
            before_checksum is not None and output_checksum is not None and before_checksum != output_checksum
        ),
        "output_checksum_scope_bytes": output_bytes,
        "native_raw_path": str(native_raw_path.relative_to(ROOT) if native_raw_path.is_relative_to(ROOT) else native_raw_path),
        "native_raw_override": args.native_raw_override is not None,
        "hmx_context_mask": args.hmx_context_mask,
        "skel_mask_helper": args.skel_mask_helper,
        "skel_mask_helper_arg6": args.skel_mask_helper_arg6 if args.skel_mask_helper else None,
        "skel_mask_helper_library": str(qnn_skel_so) if args.skel_mask_helper else None,
        "skel_mask_helper_link_mode": (
            "direct_shared_object_call_shared_ignore_unresolved" if args.skel_mask_helper else None
        ),
        "native_wrapper_prefetch": args.native_wrapper_prefetch,
        "chain_override": args.chain_override,
        "descriptor_overrides": descriptor_overrides,
        "mask_word_overrides": {str(k): v for k, v in sorted(mask_word_overrides.items())},
        "extra_word_overrides": {str(k): v for k, v in sorted(extra_word_overrides.items())},
        "mask_extra_pointer_word": args.mask_extra_pointer_word,
        "pre_clear_acc": args.pre_clear_acc,
        "extra_tail_mode": args.extra_tail_mode,
        "table_tail_mode": args.table_tail_mode,
        "record_tail_mode": args.record_tail_mode,
        "weight_byte_offset": args.weight_byte_offset,
        "folded_bias_byte_offset": args.folded_bias_byte_offset,
        "buffer_layout": args.buffer_layout,
        "descriptor_carrier": args.descriptor_carrier,
        "activation_raw_override": (
            str(args.activation_raw_override.resolve()) if args.activation_raw_override else None
        ),
        "activation_raw_override_sha256": (
            hashlib.sha256(args.activation_raw_override.read_bytes()).hexdigest()
            if args.activation_raw_override
            else None
        ),
        "activation_table_raw_override": (
            str(args.activation_table_raw_override.resolve()) if args.activation_table_raw_override else None
        ),
        "activation_table_raw_override_sha256": (
            hashlib.sha256(args.activation_table_raw_override.read_bytes()).hexdigest()
            if args.activation_table_raw_override
            else None
        ),
        "activation_table_word_overrides": {
            str(k): v for k, v in sorted(activation_table_word_overrides.items())
        },
        "output_table_raw_override": (
            str(args.output_table_raw_override.resolve()) if args.output_table_raw_override else None
        ),
        "output_table_raw_override_sha256": (
            hashlib.sha256(args.output_table_raw_override.read_bytes()).hexdigest()
            if args.output_table_raw_override
            else None
        ),
        "output_table_word_overrides": {
            str(k): v for k, v in sorted(output_table_word_overrides.items())
        },
        "packed_weight_raw_override": (
            str(args.packed_weight_raw_override.resolve()) if args.packed_weight_raw_override else None
        ),
        "packed_weight_raw_override_sha256": (
            hashlib.sha256(args.packed_weight_raw_override.read_bytes()).hexdigest()
            if args.packed_weight_raw_override
            else None
        ),
        "folded_bias_raw_override": (
            str(args.folded_bias_raw_override.resolve()) if args.folded_bias_raw_override else None
        ),
        "folded_bias_raw_override_sha256": (
            hashlib.sha256(args.folded_bias_raw_override.read_bytes()).hexdigest()
            if args.folded_bias_raw_override
            else None
        ),
        "output_surface_raw_override": (
            str(args.output_surface_raw_override.resolve()) if args.output_surface_raw_override else None
        ),
        "output_surface_raw_override_sha256": (
            hashlib.sha256(args.output_surface_raw_override.read_bytes()).hexdigest()
            if args.output_surface_raw_override
            else None
        ),
        "native_raw_bytes": len(native_raw),
        "native_raw_checksum": native_checksum,
        "native_kernel_output_scope_bytes": native_kernel_scope.get("total_output_scope_bytes"),
        "accepted_boundary_policy": comparison_scope.get("accepted_boundary_policy"),
        "activation_table_contract": abi.get("activation_table", {}).get("table_contract"),
        "activation_table_entries": abi.get("activation_table", {}).get("entry_count"),
        "output_table_contract": abi.get("output_table", {}).get("table_contract"),
        "output_table_entries": abi.get("output_table", {}).get("entry_count"),
        "split_output_row4_deblocked": (
            args.family == "w16a16"
            and split_policy
            and "row4" in str(abi.get("output_table", {}).get("table_contract", ""))
        ),
        "mask_control_method": prep_compare.get("packing", {}).get("mask_control", {}).get("method"),
        "mask_control_native_helper_emulated": prep_compare.get("packing", {}).get("mask_control", {}).get("native_helper_emulated"),
        "mask_extra_pointer_patched": (
            args.mask_extra_pointer_word is not None or (args.family == "w16a16" and split_policy)
        ),
        "checksum_comparable_to_native_raw": output_bytes == len(native_raw),
        "checksum_matches_native_raw": (
            output_checksum is not None and output_bytes == len(native_raw) and output_checksum.lower() == native_checksum
        ),
        "exactness_status": exactness_status,
        "exactness_blocker": None if exactness_status == "byte_exact_checksum" else exactness_blocker,
        "output_diff": parse_diff_line(output),
        "split_output_diff": parse_split_diff_line(output),
        "output_diff_window": parse_diff_window(output),
        "tile_output_diff": parse_tile_diff(output),
        "tile_hash_match": parse_tile_hash_match(output),
        "value_stats": parse_value_stats(output),
        "axis_diff": parse_axis_diff(output),
        "segment_diff": parse_segment_diff(output),
        "segment_vector": parse_segment_vector(output),
        "split_segment_vectors": parse_split_segment_vectors(output),
        "phase_segment_matrix": parse_phase_segment_matrix(output),
        "byte_lane_stats": parse_byte_lane_stats(output),
        "alternate_output_checksums": parse_alt_checksums(output),
        "alternate_output_diffs": parse_alt_diffs(output),
        "chain_step_checksums": parse_chain_hashes(output),
        "output_samples": parse_sample_line(output),
        "simulator_perf": simulator_perf,
        "native_perf_reference": native_reference,
        "diagnostic_perf_comparison": diagnostic_perf_comparison(simulator_perf, native_reference),
        "chain_steps_executed": int(chain_match.group(1)) if chain_match else 1,
        "artifact_activation_sha256": hashlib.sha256((artifact / "prepared_state" / "activation.raw").read_bytes()).hexdigest(),
        "effective_activation_sha256": hashlib.sha256(
            (
                args.activation_raw_override.resolve()
                if args.activation_raw_override
                else artifact / "prepared_state" / "activation.raw"
            ).read_bytes()
        ).hexdigest(),
        "artifact_weight_sha256": hashlib.sha256((artifact / "prepared_state" / "packed_weight.raw").read_bytes()).hexdigest(),
        "effective_weight_sha256": hashlib.sha256(
            (
                args.packed_weight_raw_override.resolve()
                if args.packed_weight_raw_override
                else artifact / "prepared_state" / "packed_weight.raw"
            ).read_bytes()
        ).hexdigest(),
    }
    if args.output_raw_out:
        payload["result"]["output_raw_out"] = str(args.output_raw_out.resolve())
        payload["result"]["output_raw_out_exists"] = args.output_raw_out.exists()
        payload["result"]["output_raw_out_bytes"] = args.output_raw_out.stat().st_size if args.output_raw_out.exists() else 0
    if args.internal_output_raw_out:
        payload["result"]["internal_output_raw_out"] = str(args.internal_output_raw_out.resolve())
        payload["result"]["internal_output_raw_out_exists"] = args.internal_output_raw_out.exists()
        payload["result"]["internal_output_raw_out_bytes"] = (
            args.internal_output_raw_out.stat().st_size if args.internal_output_raw_out.exists() else 0
        )
    payload["pass"] = compile_result.returncode == 0 and sim_result.returncode == 0 and bool(match)
    if args.keep_work:
        payload["work_dir"] = str(work)
    else:
        shutil.rmtree(work)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["pass"]:
        print(f"{args.family}: ok artifact-body simulator smoke {payload['result']['output_checksum']}")
        return 0
    print(f"{args.family}: FAILED artifact-body simulator smoke", file=sys.stderr)
    for line in output.strip().splitlines()[-80:]:
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
