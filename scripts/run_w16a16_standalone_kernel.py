#!/usr/bin/env python3
"""QNN-free standalone W16A16 HMX MatMul: run the real kernel body bare in
hexagon-sim and prove byte-exactness vs the QNN native output.

Unlike the generic ``run_handwritten_artifact_body_sim.py`` (which reconstructs
the W16A16 split output with a heuristic two-half-surface deblock), this driver
mirrors the *production custom op*
(``example/qnn_hmx_matmul_w16a16/src/HmxU16I16ToU16MatMulOp.cpp``) exactly:

  * ONE output Crouton16-row4 surface (not two halves);
  * the N is split into 128-column groups (kSplitNTiles=4), and each split
    writes its 4 N-tiles into that single surface through a per-split,
    stride-4 output sub-table -- this is ``maybe_split_n128``;
  * the public linear output is recovered with the SAME proven
    ``deblock_a16_crouton16_row4`` inverse that already makes W8A16 byte-exact.

Because every input (weight/bias sidecar, Crouton16 activation surface,
descriptors) is the proven-correct custom-op state and the kernel ``.inc`` is
byte-identical to native ``hmx_v73_convhhh1x1_stride1``, the standalone output
matches the QNN native ``Y.raw`` bit-for-bit -- the handwritten kernel detached
from QNN.

  uv run python scripts/run_w16a16_standalone_kernel.py \
      --artifact /tmp/hw_w16a16_check --json-out /tmp/w16a16_standalone.json

``--artifact`` is a prepared-state dir from
``example/handwritten_hmx_matmul/run_owned_smoke.py --family w16a16``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse the proven toolchain + helpers from the generic body runner.
from run_handwritten_artifact_body_sim import (  # noqa: E402
    c_array,
    fnv1a32,
    resolve_h2_root,
    run,
    tool,
)

KSPLIT_NTILES = 4
BLOCK_BYTES = 2048  # one Crouton16 32x32 u16 tile = 1024 elems * 2 bytes


def read_u32_le(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path}: not a u32 buffer ({len(data)} bytes)")
    return [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]


def c_u32_array(name: str, values: list[int]) -> str:
    body = ", ".join(f"{v & 0xffffffff}u" for v in values)
    return f"static const uint32_t {name}[{len(values)}] = {{{body}}};"


def _kernel_call_lines(m: int, n_t: int, k_t: int, mt_groups: int) -> list[str]:
    """Emit the kernel dispatch — split into 128-N groups when N_t>=4 and
    N_t%4==0 (each split writes 4 N-tiles via a stride-4 sub-table, weight/bias
    advance per split), else a single full-N call. Mirrors maybe_split_n128."""
    if n_t >= KSPLIT_NTILES and n_t % KSPLIT_NTILES == 0:
        return [
            f"  const uint32_t split_weight_bytes = {k_t * KSPLIT_NTILES * BLOCK_BYTES}u;",
            f"  const uint32_t split_bias_bytes = {KSPLIT_NTILES * 512}u;",
            f"  for (uint32_t split = 0; split < {n_t // KSPLIT_NTILES}u; ++split) {{",
            f"    for (uint32_t rg = 0; rg < {mt_groups}u; ++rg)",
            f"      for (uint32_t nt = 0; nt < {KSPLIT_NTILES}u; ++nt)",
            f"        sub_table[rg * {KSPLIT_NTILES}u + nt] = out_table[rg * {n_t}u + split * {KSPLIT_NTILES}u + nt];",
            "    HmW16A16OutDesc out_desc = {",
            f"        sub_table, {KSPLIT_NTILES}u, {m}u, {m}u, 1, {KSPLIT_NTILES * 32}u,",
            "    };",
            '    printf("[RUN] w16a16 split %u\\n", split);',
            "    hm_w16a16_v73_kernel(&out_desc, &act_desc, weight + split * split_weight_bytes,",
            "        bias + split * split_bias_bytes, (const HmW16A16MaskDesc *)mask_words, extra);",
            "  }",
        ]
    # single full-N call: full out_table (stride N_t), k_total = N, full weight/bias.
    return [
        "  HmW16A16OutDesc out_desc = {",
        f"      out_table, {n_t}u, {m}u, {m}u, 1, {n_t * 32}u,",
        "  };",
        '  printf("[RUN] w16a16 single full-N call\\n");',
        "  hm_w16a16_v73_kernel(&out_desc, &act_desc, weight, bias,",
        "      (const HmW16A16MaskDesc *)mask_words, extra);",
    ]


def generate_source(
    *,
    m: int,
    k: int,
    n: int,
    activation: bytes,
    packed_weight: bytes,
    folded_bias: bytes,
    mask_control: bytes,
    act_offsets: list[int],
    out_offsets: list[int],
    native_raw: bytes,
    dump_public: Path | None,
    dump_internal: Path | None,
) -> str:
    m_t, k_t, n_t = m // 32, k // 32, n // 32
    mt_groups = m_t * 8
    act_entries = mt_groups * k_t
    out_entries = mt_groups * n_t
    if len(act_offsets) < act_entries:
        raise ValueError("activation_table.raw has fewer entries than expected")
    if len(out_offsets) < out_entries:
        raise ValueError("output_table.raw has fewer entries than expected")
    out_bytes = m * n * 2

    # Proven Crouton16-row4 -> linear inverse (identical to the W8A16 path that
    # is already byte-exact through the generic runner).
    deblock = [
        "static void deblock_a16_crouton16_row4(uint8_t *dst, const uint8_t *src) {",
        "  for (uint32_t row4_phase = 0; row4_phase < 8u; ++row4_phase) {",
        f"    for (uint32_t nt = 0; nt < {n // 32}u; ++nt) {{",
        f"      const uint8_t *block = src + ((row4_phase * {n // 32}u + nt) * 2048u);",
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
    ]

    dump_public_lines: list[str] = []
    if dump_public is not None:
        p = json.dumps(str(dump_public))
        dump_public_lines = [
            f"  {{ FILE *f = fopen({p}, \"wb\");",
            f"    if (f) {{ fwrite(public_out, 1u, {out_bytes}u, f); fclose(f); }} }}",
        ]
    dump_internal_lines: list[str] = []
    if dump_internal is not None:
        p = json.dumps(str(dump_internal))
        dump_internal_lines = [
            f"  {{ FILE *f = fopen({p}, \"wb\");",
            f"    if (f) {{ fwrite(out, 1u, {out_bytes}u, f); fclose(f); }} }}",
        ]

    lines = [
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <h2.h>",
        "#include <h2_common_info.h>",
        "#include <h2_mxaccess.h>",
        '#include "handwritten_hmx_w16a16_kernel.h"',
        "",
        c_array("k_activation", activation),
        c_array("k_packed_weight", packed_weight),
        c_array("k_folded_bias", folded_bias),
        c_array("k_mask_control", mask_control),
        c_array("k_native_raw", native_raw),
        c_u32_array("k_act_offsets", act_offsets[:act_entries]),
        c_u32_array("k_out_offsets", out_offsets[:out_entries]),
        "",
        "static void copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t n) {",
        "  for (uint32_t i = 0; i < n; ++i) dst[i] = src[i];",
        "}",
        "static uint32_t checksum(const uint8_t *data, uint32_t n) {",
        "  uint32_t h = 0x811c9dc5u;",
        "  for (uint32_t i = 0; i < n; ++i) { h ^= data[i]; h *= 0x01000193u; }",
        "  return h;",
        "}",
        *deblock,
        "",
        "int main(void) {",
        "  unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);",
        "  unsigned int vtcm_size = h2_info(INFO_VTCM_SIZE);",
        '  printf("W16A16 standalone (QNN-free) sim\\n");',
        '  printf("[Init] VTCM base=0x%08x size=%u KB\\n", vtcm_base, vtcm_size);',
        "  if (vtcm_base == 0 || vtcm_size < 1024u) { h2_thread_stop(1); return 1; }",
        "  h2_mxaccess_state_t mxacc;",
        "  h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0, CFG_HMX_CONTEXTS, 0x1);",
        "  int mret = h2_mxaccess_acquire(&mxacc);",
        '  printf("[Init] HMX acquired (%d)\\n", mret);',
        "  uint8_t *base = (uint8_t *)(uintptr_t)(vtcm_base + 0x10000u);",
        "  uint8_t *act        = base + 0x000000u;",  # 0x20000 span
        "  uint8_t *weight     = base + 0x040000u;",  # 0x20000 span
        "  uint8_t *bias       = base + 0x080000u;",  # 0x01000 span
        "  uint8_t *out        = base + 0x0a0000u;",  # 0x20000 span (single surface)
        "  uint8_t *public_out = base + 0x100000u;",  # 0x20000 span
        "  int32_t *act_table  = (int32_t *)(base + 0x140000u);",
        "  int32_t *out_table  = (int32_t *)(base + 0x150000u);",
        "  int32_t *sub_table  = (int32_t *)(base + 0x160000u);",
        "  copy_bytes(act, k_activation, sizeof(k_activation));",
        "  copy_bytes(weight, k_packed_weight, sizeof(k_packed_weight));",
        "  copy_bytes(bias, k_folded_bias, sizeof(k_folded_bias));",
        f"  for (uint32_t i = 0; i < {out_bytes}u; ++i) out[i] = 0;",
        f"  for (uint32_t i = 0; i < {act_entries}u; ++i) act_table[i] = (int32_t)(uintptr_t)(act + k_act_offsets[i]);",
        f"  for (uint32_t i = 0; i < {out_entries}u; ++i) out_table[i] = (int32_t)(uintptr_t)(out + k_out_offsets[i]);",
        "  uint32_t mask_words[16] __attribute__((aligned(16)));",
        "  copy_bytes((uint8_t *)mask_words, k_mask_control, 64u);",
        "  uint32_t extra[2] __attribute__((aligned(16))) = {1u, 1536u};",
        "  mask_words[14] = (uint32_t)(uintptr_t)extra;",
        "  HmW16A16ActDesc act_desc = {",
        "      act_table,",
        f"      {k_t}u,",          # n_act_pairs = K_t
        f"      {k_t * 64}u,",     # act_table_y_stride = K_t*64
        "  };",
        # Mirror the op's maybe_split_n128: split into 128-N groups only when
        # N_t >= 4 and N_t % 4 == 0; otherwise a single full-N call. (op default
        # build defines HMX_W16A16_INTERNAL_SPLIT_N128.)
        *_kernel_call_lines(m, n_t, k_t, mt_groups),
        # Crouton16-row4 -> linear (same proven inverse as w8a16; the w16a16
        # :2x2 store lays out the full 32-col tile as col c -> pos 2c, so this
        # inverse is byte-exact once the mask is correct). Ground-truthed via a
        # QNN ForceFormat marker run on device.
        "  deblock_a16_crouton16_row4(public_out, out);",
        f"  uint32_t raw_hash = checksum(out, {out_bytes}u);",
        f"  uint32_t pub_hash = checksum(public_out, {out_bytes}u);",
        "  uint32_t diff_bytes = 0u; uint32_t first_diff = 0xffffffffu;",
        f"  for (uint32_t i = 0; i < {out_bytes}u; ++i) {{",
        "    if (public_out[i] != k_native_raw[i]) { diff_bytes++; if (first_diff == 0xffffffffu) first_diff = i; }",
        "  }",
        *dump_internal_lines,
        *dump_public_lines,
        '  printf("[RAW] internal_surface_checksum=0x%08lx\\n", (unsigned long)raw_hash);',
        '  printf("[DIFF] vs_native diff_bytes=%u first_diff=%d\\n", diff_bytes, (int)first_diff);',
        f'  printf("[PASS] w16a16 standalone checksum=0x%08lx bytes=%u\\n", (unsigned long)pub_hash, {out_bytes}u);',
        "  h2_thread_stop(0);",
        "  return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path,
                        help="prepared-state dir from run_owned_smoke.py --family w16a16")
    parser.add_argument("--native-raw", type=Path,
                        help="override native Y.raw (defaults to oracle raw_output)")
    parser.add_argument("--shape", type=str,
                        help="M,K,N override for the shape-general sweep (requires --native-raw); "
                             "default uses the 256^3 oracle")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--dump-public", type=Path, help="write recovered linear output")
    parser.add_argument("--dump-internal", type=Path, help="write raw crouton surface")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    prepared = artifact / "prepared_state"
    if args.shape:
        # Shape-general path: M,K,N + native-raw given directly (multi-shape
        # sweep). Prepared dir built by scripts/build_w16a16_standalone_prepared.py.
        m, k, n = (int(v) for v in args.shape.split(","))
        if args.native_raw is None:
            parser.error("--shape requires --native-raw")
        native_raw_path = args.native_raw.resolve()
    else:
        oracle = json.loads(
            (ROOT / "example" / "handwritten_hmx_matmul" / "oracles.json").read_text()
        )["families"]["w16a16"]
        m, k, n = (int(v) for v in oracle["shape_mkn"])
        native_raw_path = (
            args.native_raw.resolve() if args.native_raw else ROOT / oracle["raw_output"]["path"]
        )

    activation = (prepared / "activation.raw").read_bytes()
    packed_weight = (prepared / "packed_weight.raw").read_bytes()
    folded_bias = (prepared / "folded_bias.raw").read_bytes()
    mask_control = (prepared / "mask_control.raw").read_bytes()
    act_offsets = read_u32_le(prepared / "activation_table.raw")
    native_raw = native_raw_path.read_bytes()

    # The prepared output_table.raw covers only one 128-wide split surface. To
    # mirror the custom op's single full Crouton16-row4 output surface we build
    # the full output block-offset table analytically (the op's out_table_copy):
    #   out_table[rg*N_t + nt] = out_block[(rg & 7) * N_t + nt],
    # contiguous block j at byte offset j*2048. This is the exact inverse the
    # proven deblock_a16_crouton16_row4 consumes.
    m_t, k_t, n_t = m // 32, k // 32, n // 32
    mt_groups = m_t * 8
    out_offsets = [
        (((rg & 7) * n_t) + nt) * BLOCK_BYTES
        for rg in range(mt_groups)
        for nt in range(n_t)
    ]

    clang = tool("hexagon-clang")
    sim = tool("hexagon-sim")
    h2_root, h2_install = resolve_h2_root()
    booter = h2_install / "bin" / "booter"
    h2_include = h2_install / "include"
    h2_kernel_include = h2_root / "kernel" / "include"
    h2_lib = h2_install / "lib"

    source_text = generate_source(
        m=m, k=k, n=n,
        activation=activation,
        packed_weight=packed_weight,
        folded_bias=folded_bias,
        mask_control=mask_control,
        act_offsets=act_offsets,
        out_offsets=out_offsets,
        native_raw=native_raw,
        dump_public=args.dump_public.resolve() if args.dump_public else None,
        dump_internal=args.dump_internal.resolve() if args.dump_internal else None,
    )

    work = Path(tempfile.mkdtemp(prefix="w16a16_standalone_"))
    source = work / "w16a16_standalone.c"
    binary = work / "w16a16_standalone"
    source.write_text(source_text, encoding="utf-8")

    qnn_skel_so = ROOT / "tools" / "qnn-sdk" / "lib" / "hexagon-v75" / "unsigned" / "libQnnHtpV75Skel.so"
    compile_cmd = [
        str(clang), "-O2", "-mv75", "-mhvx", "-mhvx-length=128B", "-mhmx", "-DARCHV=75",
        "-I", str(h2_include),
        "-I", str(h2_kernel_include),
        "-I", str(ROOT / "example" / "handwritten_hmx_matmul" / "include"),
        "-moslib=h2",
        "-Wl,-L," + str(h2_lib),
        "-Wl,--section-start=.start=0x02000000",
        "-o", str(binary), str(source),
    ]
    compile_result = run(compile_cmd, ROOT)
    payload = {
        "schema": "w16a16_standalone_kernel.v1",
        "artifact": str(artifact),
        "shape_mkn": [m, k, n],
        "qnn_used": False,
        "native_raw_path": str(native_raw_path),
        "compile": {
            "returncode": compile_result.returncode,
            "output": compile_result.stdout.strip().splitlines(),
        },
        "pass": False,
    }
    if compile_result.returncode != 0:
        print(compile_result.stdout)
        if args.json_out:
            args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        return 1

    sim_cmd = [
        str(sim), "--mv75", "--mhmx", "1", "--simulated_returnval", "--",
        str(booter), "--ext_power", "1", "--use_ext", "1", "--fence_hi", "0xfe000000",
        str(binary),
    ]
    sim_result = run(sim_cmd, ROOT)
    out = sim_result.stdout

    import re
    diff_m = re.search(r"^\[DIFF\] vs_native diff_bytes=(\d+) first_diff=(-?\d+)$", out, re.M)
    pass_m = re.search(r"^\[PASS\] w16a16 standalone checksum=(0x[0-9a-fA-F]+) bytes=(\d+)$", out, re.M)
    raw_m = re.search(r"^\[RAW\] internal_surface_checksum=(0x[0-9a-fA-F]+)$", out, re.M)

    native_checksum = fnv1a32(native_raw)
    diff_bytes = int(diff_m.group(1)) if diff_m else None
    first_diff = int(diff_m.group(2)) if diff_m else None
    out_checksum = pass_m.group(1) if pass_m else None
    byte_exact = diff_bytes == 0 and out_checksum == native_checksum
    payload.update({
        "simulate": {"returncode": sim_result.returncode, "output": out.splitlines()},
        "native_checksum": native_checksum,
        "output_checksum": out_checksum,
        "internal_surface_checksum": raw_m.group(1) if raw_m else None,
        "diff_bytes": diff_bytes,
        "first_diff_byte": first_diff,
        "byte_exact": byte_exact,
        "pass": byte_exact,
    })
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if not args.keep_work:
        import shutil
        shutil.rmtree(work, ignore_errors=True)

    status = "BYTE-EXACT" if byte_exact else "MISMATCH"
    print(f"w16a16 standalone (QNN-free): {status}  "
          f"out={out_checksum} native={native_checksum} "
          f"diff_bytes={diff_bytes} first_diff={first_diff}")
    if not byte_exact:
        # surface the sim tail for debugging
        for line in out.splitlines()[-25:]:
            print("  |", line)
    return 0 if byte_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
