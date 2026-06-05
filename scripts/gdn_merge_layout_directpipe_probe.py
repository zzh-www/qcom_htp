#!/usr/bin/env python3
"""Step-2 待办3: prove LAYOUT DIRECT-PIPE — the int16 crouton16_row4 OUTPUT surface the
kernel writes is byte-identical to the crouton16_row4 ACTIVATION surface the next matmul
reads.  If so, STEP A's output feeds STEP B's activation with ZERO repack.

This reduces to: is `deblock_a16_crouton16_row4` (the C output de-pack proven bit-exact
in run_handwritten_artifact_body_sim) the EXACT INVERSE of `pack_a16_crouton16_row4_surface`
(the activation packer)?  If pack∘deblock = id and deblock∘pack = id, the two surface
contracts are identical -> direct-pipe valid (only the WEIGHT operand needs k-major repack).

Pure index permutation check (no hexagon-sim).  deblock ported verbatim from the C harness
(run_handwritten_artifact_body_sim.py L1708-1730).

Reproduce: source scripts/env.sh && python scripts/gdn_merge_layout_directpipe_probe.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example" / "handwritten_hmx_matmul"))
from prepare_owned_inputs import pack_a16_crouton16_row4_surface


def deblock_c_harness(surface_u16: np.ndarray, m: int, n: int) -> np.ndarray:
    """Faithful port of the C harness deblock_a16_crouton16_row4 (run_handwritten_artifact_
    body_sim L1708-1730) — the de-pack PROVEN bit-exact against the real KERNEL output.
    The C code hardcodes the 256^3 tile (2048 B); generalized here as tile = (m//32)*256 B
    = (m//32)*128 u16 (crouton16 tile = m_tiles*256 B per the recipe).  All in u16 units."""
    src = surface_u16.reshape(-1)
    dst = np.zeros((m, n), dtype=surface_u16.dtype)
    n32 = n // 32
    tile_u16 = (m // 32) * 128            # 1024 @ m=256 (==2048 B); 256 @ m=64
    for row4_phase in range(8):
        for nt in range(n32):
            block = (row4_phase * n32 + nt) * tile_u16
            for m32_group in range(m // 32):
                for row_pair in range(2):
                    row0 = m32_group * 32 + row4_phase * 4 + row_pair * 2
                    row1 = row0 + 1
                    src_pair = block + (m32_group * 2 + row_pair) * 64     # 128 B = 64 u16
                    for col in range(32):
                        word = src_pair + col * 2                          # 4 B = 2 u16
                        dst[row0, nt * 32 + col] = src[word + 0]
                        dst[row1, nt * 32 + col] = src[word + 1]
    return dst


def main():
    rng = np.random.default_rng(0)
    ok = True
    # The C-harness deblock is PROVEN to de-pack the real kernel output bit-exact.  If that
    # SAME permutation inverts pack (the activation packer), then kernel-output-surface ==
    # activation-surface format -> direct-pipe.  Test: deblock_c_harness(pack(logical))==logical.
    for (m, n) in [(64, 64), (256, 256), (64, 128), (128, 64), (256, 64), (64, 192)]:
        logical = rng.integers(0, 65536, (m, n)).astype(np.uint16)
        packed = pack_a16_crouton16_row4_surface(logical).reshape(-1)     # = activation surface
        back = deblock_c_harness(packed, m, n)                            # = kernel-output de-pack
        same = np.array_equal(back, logical)
        ok = ok and same
        print(f"  m={m:3d} n={n:3d}:  C-deblock(pack(logical)) == logical : {same}")

    print()
    if ok:
        print("✅ DIRECT-PIPE PROVEN: deblock_a16_crouton16_row4 == inverse of "
              "pack_a16_crouton16_row4_surface.")
        print("   => kernel's raw int16 output surface IS byte-identical to the activation "
              "surface format.")
        print("   => STEP A crouton output feeds STEP B activation with ZERO repack; only the "
              "WEIGHT operand needs k-major.")
    else:
        print("❌ NOT mutual inverses -> output surface != activation surface; a repack IS "
              "needed between matmuls.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
