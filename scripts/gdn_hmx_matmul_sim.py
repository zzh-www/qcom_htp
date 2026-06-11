#!/usr/bin/env python3
"""De-risk M1: prove the fully-owned handwritten u8i8 HMX kernel computes a
correct 64x64x64 matmul, VTCM-resident, glue-free, in hexagon-sim (no device).

Approach B: synthesize a minimal bare-metal C harness (modelled on
example/handwritten_hmx_matmul/tools/body_entry_smoke.c), embed the packed
activation/weight/bias as C arrays, set up the 64^3 descriptors exactly as
prepare_owned_inputs.py prescribes, call hm_u8i8_v73deep_kernel, then dump the
4096-byte output VTCM surface as hex.  Parse stdout, de-pack the Crouton8
surface back to natural [m,n], compare to a numpy reference.

Modes:
  --probe   : controlled inputs (weight=1, act=1, bias ramp) to DISCOVER the
              output surface layout empirically.
  --real    : sliced real 64x64 data from the known-correct 256^3 artifact.

Reads cycle count from h2_get_pcycles() bracketing the kernel call.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "handwritten_hmx_matmul"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(SCRIPTS))

from prepare_owned_inputs import pack_w8_kmajor  # noqa: E402
from emulate_hmx_conv1x1_params import conv1x1_words  # noqa: E402

DATA_W = EXAMPLE.parent / "qnn_matmul_profile" / "output_u8i8_aligned_e2e_256" / "u8i8.onnx.wRaw_KN.npy"
DATA_BIAS = EXAMPLE.parent / "qnn_matmul_profile" / "output_u8i8_aligned_e2e_256" / "u8i8.onnx.bias_q_int32.npy"
DATA_ACT = (
    EXAMPLE.parent
    / "qnn_matmul_profile"
    / "output_u8i8_native_ref_e2e_256"
    / "runtime_inputs_native"
    / "A.raw"
)

M = K = N = 64
TILE_BYTES = 4096


# ---------------------------------------------------------------------------
# packers (replicate prepare_owned_inputs.py contract exactly)
# ---------------------------------------------------------------------------
def pack_act_crouton8(act_mk: np.ndarray) -> np.ndarray:
    """Activation crouton8 surface for the descriptor-driven u8i8 kernel.

    EMPIRICAL FINDING (M1, hexagon-sim): each K-tile must be a SEPARATE
    contiguous crouton8 tile of `m*32` bytes, laid end-to-end (kt0 at byte 0,
    kt1 at byte m*32, ...), and act_table[kt] points at kt*m*32.  Within one
    k-tile the order is row8_group(4) x m32_group(m//32) x row_sub(8) x
    col_word(8), copy 4 bytes.  Requires m%32==0, k%32==0.

    NOTE: this differs from prepare_owned_inputs.py:247-269, which interleaves
    `kt` INSIDE the row8_group loop (scattering each k-tile across the surface).
    That interleaved layout makes the kernel read the wrong activation columns
    for k>=32 (verified: k-tile1 aliased to k-tile0 columns).  The per-k-tile
    contiguous layout below is bit-exact for full-K matmuls in sim.
    """
    m, k = act_mk.shape
    assert m % 32 == 0 and k % 32 == 0
    out_buf = np.zeros(m * k, dtype=np.uint8)
    out = 0
    for kt in range(k // 32):
        k_base = kt * 32
        for row8_group in range(4):
            for m32_group in range(m // 32):
                for row_sub in range(8):
                    row = m32_group * 32 + row8_group * 8 + row_sub
                    for col_word in range(8):
                        col = k_base + col_word * 4
                        out_buf[out:out + 4] = act_mk[row, col:col + 4]
                        out += 4
    return out_buf


def pack_folded_bias(effective: np.ndarray) -> np.ndarray:
    """Per N32 tile: 32 control words 0x6000 then 32 effective-bias int32 words.

    Replicates prepare_owned_inputs.py:856-860.
    """
    assert effective.size % 32 == 0
    chunks = []
    control = np.full(32, 0x6000, dtype="<i4")
    for start in range(0, effective.size, 32):
        chunks.append(control)
        chunks.append(effective[start:start + 32].astype("<i4"))
    return np.concatenate(chunks).astype("<i4", copy=False)


def descriptor_tables(m: int, k: int, n: int):
    """Build the u8i8 crouton8 offset tables + descriptor field values.

    Mirrors prepare_owned_inputs.py::generated_descriptor_tables for u8i8 with
    output dims [1, m//32, 32, n] (surface_n=n), itemsize=1.
    """
    itemsize = 1
    m_tiles = max(1, m // 32)
    k_tiles = max(1, k // 32)
    n_tiles = max(1, n // 32)
    # u8i8 crouton8: m_tiles must be even -> group of 64 rows
    assert m_tiles % 2 == 0, "u8i8 crouton8 needs m_tiles even"
    table_m_groups = m_tiles // 2
    table_row_span = 64

    # Each k-tile is a separate contiguous crouton8 tile of m*32 bytes (M1
    # empirical finding); act_table[kt] = kt * m * 32.
    m_total = m
    activation_offsets = []
    for mt in range(table_m_groups):
        for kt in range(k_tiles):
            activation_offsets.append(kt * m_total * 32 * itemsize)
    output_offsets = []
    for mt in range(table_m_groups):
        for nt in range(n_tiles):
            output_offsets.append((mt * n_tiles + nt) * table_row_span * 32 * itemsize)

    out_desc = dict(
        out_table_stride_dwords=n_tiles,
        out_y_stride_words=m_tiles * 4,
        n_tiles_pow2=m_tiles * 4,
        m_total_minus_step=8,
        k_total_bytes=n_tiles * 32,
    )
    act_desc = dict(
        n_act_pairs=k_tiles,
        act_table_y_stride_words=m_tiles * 4,
    )
    mask_words = conv1x1_words(0x700, 0, 0, 0, 0x20)
    extra = [1, 0]
    return dict(
        activation_offsets=activation_offsets,
        output_offsets=output_offsets,
        out_desc=out_desc,
        act_desc=act_desc,
        mask_words=list(mask_words),
        extra=extra,
        m_tiles=m_tiles,
        k_tiles=k_tiles,
        n_tiles=n_tiles,
        table_m_groups=table_m_groups,
        table_row_span=table_row_span,
    )


# ---------------------------------------------------------------------------
# reference (scripts/reconstruct_hmx_u8_drain.py contract, scale=512,baseline=0)
# ---------------------------------------------------------------------------
def reference_u8(act_mk: np.ndarray, w_kn: np.ndarray, effective: np.ndarray) -> np.ndarray:
    raw = act_mk.astype(np.int64) @ w_kn.astype(np.int64)  # [m,n]
    out = raw + effective.astype(np.int64)[None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def effective_from_w(w_kn: np.ndarray, bias_q: np.ndarray) -> np.ndarray:
    return (-128 * w_kn.astype(np.int64).sum(axis=0) + bias_q.astype(np.int64)).astype("<i4")


# ---------------------------------------------------------------------------
# C harness emission
# ---------------------------------------------------------------------------
def c_u8_array(name: str, data: bytes) -> str:
    body = ",".join(str(b) for b in data)
    return f"static const uint8_t {name}[{len(data)}] = {{{body}}};"


def c_u32_array(name: str, vals) -> str:
    body = ",".join(f"{int(v) & 0xffffffff}u" for v in vals)
    return f"static const uint32_t {name}[{len(vals)}] = {{{body}}};"


HARNESS_TMPL = r"""
#include <stdint.h>
#include <stdio.h>

#include <h2.h>
#include <h2_common_info.h>
#include <h2_mxaccess.h>
#include <h2_cycles.h>

#include "handwritten_hmx_u8i8_kernel.h"

%(ARRAYS)s

#define ACT_BYTES %(ACT_BYTES)du
#define WEIGHT_BYTES %(WEIGHT_BYTES)du
#define BIAS_BYTES %(BIAS_BYTES)du
#define OUT_BYTES %(OUT_BYTES)du
#define ACT_ENTRIES %(ACT_ENTRIES)du
#define OUT_ENTRIES %(OUT_ENTRIES)du

static void copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) dst[i] = src[i];
}

int main(void) {
  unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);
  unsigned int vtcm_size = h2_info(INFO_VTCM_SIZE);
  printf("u8i8 64cubed matmul sim\n");
  printf("[Init] VTCM base=0x%%08x size=%%u KB\n", vtcm_base, vtcm_size);
  if (vtcm_base == 0 || vtcm_size < 1024u) { h2_thread_stop(1); return 1; }

  h2_mxaccess_state_t mxacc;
  h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0, CFG_HMX_CONTEXTS, 0x1);
  int mret = h2_mxaccess_acquire(&mxacc);
  printf("[Init] HMX acquired (%%d)\n", mret);

  uint8_t *base = (uint8_t *)(uintptr_t)(vtcm_base + 0x10000u);
  uint8_t *act = base + 0x00000u;
  uint8_t *weight = base + 0x20000u;
  uint8_t *bias = base + 0x40000u;
  uint8_t *out = base + 0x60000u;
  int32_t *act_table = (int32_t *)(base + 0xc0000u);
  int32_t *out_table = (int32_t *)(base + 0xc1000u);

  copy_bytes(act, k_activation, ACT_BYTES);
  copy_bytes(weight, k_packed_weight, WEIGHT_BYTES);
  copy_bytes(bias, k_folded_bias, BIAS_BYTES);
  for (uint32_t i = 0; i < OUT_BYTES; ++i) out[i] = 0u;

  uint32_t mask_words[16] __attribute__((aligned(16)));
  copy_bytes((uint8_t *)mask_words, (const uint8_t *)k_mask_control, 64u);
  uint32_t extra[2] __attribute__((aligned(16))) = {%(EXTRA0)du, %(EXTRA1)du};

  /* ONE descriptor-driven u8i8 call computes the whole 64x64x64 matmul,
     VTCM-resident.  The kernel reads all 64 activation rows and 2 k-tiles. */
  for (uint32_t i = 0; i < ACT_ENTRIES; ++i) act_table[i] = (int32_t)(uintptr_t)(act + k_activation_offsets[i]);
  for (uint32_t i = 0; i < OUT_ENTRIES; ++i) out_table[i] = (int32_t)(uintptr_t)(out + k_output_offsets[i]);
  HmU8I8OutDesc out_desc = { out_table, %(OUT_TABLE_STRIDE)du, %(OUT_Y_STRIDE)du,
      %(N_TILES_POW2)du, %(M_TOTAL_MINUS_STEP)d, %(K_TOTAL_BYTES)du };
  HmU8I8ActDesc act_desc = { act_table, %(N_ACT_PAIRS)du, %(ACT_Y_STRIDE)du };
  unsigned long long t0 = h2_get_pcycles();
  hm_u8i8_v73deep_kernel(&out_desc, &act_desc, weight, bias,
                         (const HmU8I8MaskDesc *)mask_words, extra);
  unsigned long long t1 = h2_get_pcycles();
  printf("[CYC]%%llu\n", (unsigned long long)(t1 - t0));

  printf("[OUT]");
  for (uint32_t i = 0; i < OUT_BYTES; ++i) printf("%%02x", out[i]);
  printf("\n[PASS] u8i8 64cubed body returned\n");
  h2_thread_stop(0);
  return 0;
}
"""


def emit_harness(act_packed: bytes, w_packed: bytes, bias_packed: bytes, tabs, out_bytes: int) -> str:
    arrays = "\n".join(
        [
            c_u8_array("k_activation", act_packed),
            c_u8_array("k_packed_weight", w_packed),
            c_u8_array("k_folded_bias", bias_packed),
            c_u32_array("k_activation_offsets", tabs["activation_offsets"]),
            c_u32_array("k_output_offsets", tabs["output_offsets"]),
            c_u32_array("k_mask_control", tabs["mask_words"]),
        ]
    )
    od = tabs["out_desc"]
    ad = tabs["act_desc"]
    return HARNESS_TMPL % dict(
        ARRAYS=arrays,
        ACT_BYTES=len(act_packed),
        WEIGHT_BYTES=len(w_packed),
        BIAS_BYTES=len(bias_packed),
        OUT_BYTES=out_bytes,
        ACT_ENTRIES=len(tabs["activation_offsets"]),
        OUT_ENTRIES=len(tabs["output_offsets"]),
        EXTRA0=tabs["extra"][0],
        EXTRA1=tabs["extra"][1],
        OUT_TABLE_STRIDE=od["out_table_stride_dwords"],
        OUT_Y_STRIDE=od["out_y_stride_words"],
        N_TILES_POW2=od["n_tiles_pow2"],
        M_TOTAL_MINUS_STEP=od["m_total_minus_step"],
        K_TOTAL_BYTES=od["k_total_bytes"],
        N_ACT_PAIRS=ad["n_act_pairs"],
        ACT_Y_STRIDE=ad["act_table_y_stride_words"],
    )


def tool(name: str) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found)
    cands = sorted((ROOT / "tools" / "hexagon-sdk" / "tools" / "HEXAGON_Tools").glob("*/Tools/bin/" + name))
    if cands:
        return cands[0]
    raise FileNotFoundError(name)


def resolve_h2():
    h2_install = ROOT / "tools" / "h2-install"
    h2_root = h2_install.resolve().parent if h2_install.is_symlink() else h2_install.parent
    return h2_root, h2_install


def build_and_run(c_src: str, keep: bool = False):
    clang = tool("hexagon-clang")
    sim = tool("hexagon-sim")
    h2_root, h2_install = resolve_h2()
    booter = h2_install / "bin" / "booter"
    work = Path(tempfile.mkdtemp(prefix="gdn_hmx_64_"))
    src = work / "harness.c"
    src.write_text(c_src)
    binary = work / "harness"
    compile_cmd = [
        str(clang), "-O2", "-mv75", "-mhvx", "-mhvx-length=128B", "-mhmx", "-DARCHV=75",
        "-I", str(h2_install / "include"),
        "-I", str(h2_root / "kernel" / "include"),
        "-I", str(EXAMPLE / "include"),
        "-moslib=h2",
        "-Wl,-L," + str(h2_install / "lib"),
        "-Wl,--section-start=.start=0x02000000",
        "-o", str(binary), str(src),
    ]
    cr = subprocess.run(compile_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cr.returncode != 0:
        print("COMPILE FAILED:\n" + cr.stdout)
        return None
    sim_cmd = [
        str(sim), "--mv75", "--mhmx", "1", "--simulated_returnval", "--",
        str(booter), "--ext_power", "1", "--use_ext", "1", "--fence_hi", "0xfe000000",
        str(binary),
    ]
    sr = subprocess.run(sim_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not keep:
        shutil.rmtree(work)
    else:
        print(f"work dir: {work}")
    return sr.stdout


def parse_out(stdout: str):
    o = re.search(r"^\[OUT\]([0-9a-f]*)$", stdout, re.M)
    c = re.search(r"^\[CYC\](\d+)$", stdout, re.M)
    surf = bytes.fromhex(o.group(1)) if o else None
    cyc = int(c.group(1)) if c else None
    return surf, cyc


# ---------------------------------------------------------------------------
# Output de-pack.  EMPIRICAL closed form (M1, hexagon-sim), validated bit-exact
# vs numpy on real + 3 random full-range datasets.  One u8i8 call writes the
# whole [64,64] u8 output as a 4096-byte crouton8 surface.  The byte offset of
# out[r,c] is:
#   byte = nt*2048 + r8*512 + m32*256 + rsub*32 + cw*4 + bsub
# where nt=c//32, m32=r//32, r8=(r%32)//8, rsub=r%8, cw=(c%32)//4, bsub=c%4.
# ---------------------------------------------------------------------------
def depack_output(surface: bytes, m: int, n: int) -> np.ndarray:
    arr = np.frombuffer(surface, dtype=np.uint8)
    out = np.zeros((m, n), dtype=np.int64)
    for r in range(m):
        m32 = r // 32
        r8 = (r % 32) // 8
        rsub = r % 8
        for c in range(n):
            nt = c // 32
            cw = (c % 32) // 4
            bsub = c % 4
            byte = nt * 2048 + r8 * 512 + m32 * 256 + rsub * 32 + cw * 4 + bsub
            out[r, c] = arr[byte]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "real", "random"], default="real")
    ap.add_argument("--seed", type=int, default=7, help="seed for --mode random")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    global M, K, N
    M = K = N = args.size
    tabs = descriptor_tables(M, K, N)
    out_bytes = M * N  # 4096 for 64x64 u8

    if args.mode == "probe":
        # m-ramp: act[m,:]=m+1, w[0,:]=1 -> raw=act[m,0]=m+1, eff=0 -> out=clip(m+1).
        # Distinct value per row; n-independent.  Sanity-checks the de-pack row map.
        act = np.zeros((M, K), dtype=np.uint8)
        for m in range(M):
            act[m, :] = min(m + 1, 255)
        w = np.zeros((K, N), dtype=np.int8)
        w[0, :] = 1
        bias_q = np.full(N, 128, dtype="<i4")
        effective = effective_from_w(w, bias_q)
    elif args.mode == "random":
        rng = np.random.default_rng(args.seed)
        act = rng.integers(0, 256, (M, K)).astype(np.uint8)
        w = rng.integers(-128, 128, (K, N)).astype(np.int8)
        bias_q = rng.integers(-5000, 5000, N).astype("<i4")
        effective = effective_from_w(w, bias_q)
    else:  # real: slice the known-correct 256^3 u8i8 artifact down to 64^3
        w256 = np.load(DATA_W)
        bias256 = np.load(DATA_BIAS)
        a256 = np.fromfile(DATA_ACT, dtype=np.uint8).reshape(256, 256)
        w = w256[0:K, 0:N].astype(np.int8)
        bias_q = bias256[0:N].astype("<i4")
        act = a256[0:M, 0:K].astype(np.uint8)
        effective = effective_from_w(w, bias_q)

    ref = reference_u8(act, w, effective)

    act_packed = pack_act_crouton8(act).tobytes()
    w_packed = pack_w8_kmajor(w).tobytes()
    bias_packed = pack_folded_bias(effective).tobytes()

    c_src = emit_harness(act_packed, w_packed, bias_packed, tabs, out_bytes)
    stdout = build_and_run(c_src, keep=args.keep)
    if stdout is None:
        return 1
    surface, cyc = parse_out(stdout)
    if surface is None:
        print("SIM OUTPUT (no [OUT]):\n" + stdout)
        return 1

    print(f"mode={args.mode}  cycles(pcycles)={cyc}")
    print(f"descriptor: out_desc={tabs['out_desc']} act_desc={tabs['act_desc']}")
    print(f"  m_tiles={tabs['m_tiles']} k_tiles={tabs['k_tiles']} n_tiles={tabs['n_tiles']} "
          f"table_m_groups={tabs['table_m_groups']} row_span={tabs['table_row_span']}")

    got = depack_output(surface, M, N)
    diff = np.abs(got - ref.astype(np.int64))
    max_abs = int(diff.max())
    n_mis = int((diff != 0).sum())
    print(f"de-pack vs numpy reference: max_abs_diff={max_abs}  mismatches={n_mis}/{M*N}")
    if max_abs == 0:
        print("RESULT: PASS  (bit-exact 64x64x64 u8i8 matmul in hexagon-sim, single call)")
        macs = M * N * K
        if cyc:
            print(f"cyc/MAC = {cyc}/{macs} = {cyc / macs:.6e}  "
                  f"(single call incl. one-shot kernel setup; NOT steady-state -- the "
                  f"~430 cyc fixed prologue/drain dominates this tiny shape)")
        return 0
    else:
        print("RESULT: MISMATCH")
        rows, cols = np.nonzero(diff)
        for r, c in list(zip(rows, cols))[:10]:
            print(f"  [{r},{c}] got={got[r, c]} ref={ref[r, c]}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
