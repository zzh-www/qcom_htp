---
name: V8 C8 kernel Phase 2 — scalar matmul BIT-EXACT at 32³–1024³ 2026-04-27 night
description: Scalar reference matmul reading Crouton_8 act blocks directly + native pre-packed wt + int32 bias, writing tile-layout output, produces bit-exact match vs Python reference at every shape 32³ → 1024³ (1.04M cells total). Crouton_8 layout formula validated end-to-end.
type: project
---

# Result

**BIT-EXACT 1024/1024, 4096/4096, 16384/16384, 65536/65536, 262144/262144, 1048576/1048576** for shapes 32³, 64³, 128³, 256³, 512³, 1024³ respectively.

Full Crouton_8 input layout, native pre-packed wt layout, and tile-layout output are all consumed correctly, **end-to-end through QNN's auto-DMA + ForceFormat_Crouton + UntileToRowMajor pipeline**.

# Crouton_8 access formula (verified by bit-exact end-to-end test)

```
For act tensor [1, M/32, 32, K] with Crouton_8 + Indirect storage:
  block_rows  = min(M / 4, 64)              # 8 / 16 / 32 / 64 for S=32 / 64 / 128 / >=256
  row_groups  = M / block_rows              # = max(4, M / 64)
  k_chunks    = K / 32

  block_table_length = row_groups × k_chunks
  block_size_bytes   = block_rows × 32

  Block index b = rg × k_chunks + kc        # k_chunk INNER, row_group OUTER
  Block(b) covers:
    rows [rg * block_rows  ..  (rg+1) * block_rows - 1]
    cols [kc * 32          ..  (kc+1) * 32 - 1]

  Within a block: row-major (32 bytes per row × block_rows rows)
  byte_in_block(m_in_block, k_in_chunk) = m_in_block × 32 + k_in_chunk

  Inverse — to read act[m, k]:
    rg          = m / block_rows
    kc          = k / 32
    block_idx   = rg × k_chunks + kc
    byte_offset = (m % block_rows) × 32 + (k % 32)
    act[m, k]   = ((const uint8_t *)block_table[block_idx])[byte_offset]
```

For square M=K=N=S, derived `block_rows` per shape:

| S    | block_rows | row_groups | k_chunks | total blocks |
|-----:|:----------:|:----------:|:--------:|:------------:|
| 32   | 8          | 4          | 1        | 4            |
| 64   | 16         | 4          | 2        | 8            |
| 128  | 32         | 4          | 4        | 16           |
| 256  | 64         | 4          | 8        | 32           |
| 512  | 64         | 8          | 16       | 128          |
| 1024 | 64         | 16         | 32       | 512          |

At ≥256³ a single block spans MULTIPLE M_tiles (block_rows=64 > M_t row group=32). This was the source of confusion earlier — block boundaries don't align with M_t boundaries.

# Native pre-packed wt access (already known but re-verified)

```
For wt tensor [1, K/32, N/32, 1024] u8:
  Tile (kt, nt) at offset (kt * N_t + nt) * 1024 in flat buffer
  Within a tile, byte for cell (r, c) where r,c ∈ [0..32):
    byte_offset = (r / 4) * 128 + c * 4 + (r % 4)
  Stored value = (int8_t) wRaw_KN[kt*32 + r, nt*32 + c]
```

# Output write (tile-layout, consumed by UntileToRowMajor)

```
out_u8[m, n]  →  tile[(mt * N_t + nt) * 1024 + (m % 32) * 32 + (n % 32)]
                where mt = m/32, nt = n/32

UntileToRowMajor then maps tile bytes back to row-major final user output.
```

# Math (kernel scalar)

```
Per output cell (m, n):
  acc[m, n] = Σ_k act_u8[m, k] × wRaw_i8[k, n]                    # the matmul
            + (-ACT_ZP_FOLD × Σ_k wRaw_i8[k, n]) + bias_q[n]      # the fold
  out_u8[m, n] = saturate_u8(acc)                                 # scale=1, zp=0 in this test
```

For our quant_overrides with `act offset=0`, `ACT_ZP_FOLD = 0`, the fold collapses to `bias_q[n]` (no Σwt term needed). The fold formula is kept explicit for native alignment when act_zp ≠ 0.

`Σ_k wRaw[k, n]` is precomputed once per channel (N MACs total) before the outer matmul loop — negligible vs M×N×K MACs.

# Code

`src/HmxMatMulV9SkelOp.cpp` — new branch under `V9_KERNEL_SCALAR` flag. ~70 lines of straight C, no HMX, no HVX. Build with:
```bash
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_SCALAR" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_SCALAR" bash build_x86.sh
```

`gen_v8c8_test.py` — emits Python reference output `out_ref_u8.npy` next to the ONNX file using NumPy. Comparison loop:
```python
for S in (32, 64, 128, 256, 512, 1024):
    OUT_DIR=/tmp/v8c8_scalar_$S M=$S K=$S N=$S bash run_v8c8_phase2.sh
    device = np.round(np.fromfile(out, dtype=np.float32)).astype(np.uint8).reshape(S, S)
    ref    = np.load("v8c8.onnx.out_ref_u8.npy")
    assert (device == ref).all()
```

# What this proves

End-to-end correctness of the V8 C8 framework (3-input native-aligned: act Crouton_8 + wt + int32 bias):
- ✅ Crouton_8 indirect block layout
- ✅ Native pre-packed wt layout
- ✅ int32 bias readable
- ✅ Tile-layout output
- ✅ UntileToRowMajor permutation handled
- ✅ All shapes 32³–1024³ (full sweep, 1.04M output cells)

The math is correct. **The kernel can now be replaced with HMX inline asm without worrying about format issues**.

# Next — Phase 3: HMX inner loop

Plan:
1. Pre-compute `effective_int32[c] = -ACT_ZP × Σwt[c] + bias_q[c]` per channel (N values).
2. Convert `effective_int32` + a per-channel scale_fp16 into the V8-prod 128-B/N-tile bias VTCM layout (or native 256-B/N-tile if we can encode the int32 fold via mxmem2).
3. HMX MAC loop per (mt, nt) tile:
   - `mxclracc`
   - For each kt: `{ activation.ub = mxmem(act_tile_ptr, 0x71F):cm \n weight.b = mxmem(wt_tile_ptr, 0x3FF) }`
   - `bias = mxmem(bias_per_tile)` or `mxmem2(...)`
   - `mxmem(out_tile, 0):after:cm:sat.ub = acc`
4. Open question: can we hand HMX a Crouton_8 block pointer via `mxmem(...):cm`, or do we need a per-tile pointer of shape `[1024 B contiguous]`?
   - At ≥256³, a Crouton_8 block is 64 rows × 32 cols = 2 KB. A HMX :cm "tile" expected by the silicon is 32×32 = 1 KB.
   - Likely the HMX kernel walks block_table[bi][byte_offset_within_block] so each `:cm` reads ONE 32-row strip out of the larger block.

This is silicon-RE territory but we have the bit-exact scalar reference to validate against.
