---
name: V8 C8 kernel Phase 1 — input access fully validated, Crouton_8 layout partially decoded 2026-04-27 evening
description: Established full read-access for all 3 inputs (Crouton_8 act via block_table, native pre-packed wt via raw_data, int32 bias via raw_data). Σ_k W[k,0] bit-exact at 32³–1024³ vs Python reference. Discovered shape API breaks at >=64³ (workaround: derive S from block_table_length for square test). Identified that BbbKMajor's tile-output is permuted to row-major positions by UntileToRowMajor. Crouton_8 internal block layout is 32-col stride per row, block_rows scales with shape (8/16/32/64 capped at 2KB block).
type: project
---

# Status

V8 C8 kernel development — **Phase 1 (input access) complete**. Phase 2 (scalar reference matmul) is the next step. Significant blocker decoded: how to read each input correctly + how to map kernel writes to user-visible output.

# Findings

## 1. Input access pattern per tensor type

| Input | Sig | Access API | Works at |
|-------|-----|------------|----------|
| act | Crouton_8 + Indirect | `qhpi_tensor_block_table()` returns `void**`, `qhpi_tensor_block_table_length()` returns block count | All shapes |
| wt | Flat4 + Direct | `qhpi_tensor_raw_data()` returns flat byte buffer | All shapes |
| bias | Flat4 + Direct | `qhpi_tensor_raw_data()` returns int32 array | All shapes |

## 2. Shape API broken at ≥64³

`qhpi_tensor_shape()`, `qhpi_tensor_padded_shape()`, `qhpi_tensor_padding()`, and `qhpi_tensor_block_shape()` ALL return `rank=0` for inputs and outputs at ≥64³. Only at 32³ do they return correct values. The compile-time auto-DMA path (`q::ConvLayer.opt.weights_to_vtcm`, `q::ForceFormat_Crouton`) appears to strip logical shape metadata from VTCM-resident tensors.

**Workaround**: derive M=K=N=S from `qhpi_tensor_block_table_length(act)`:
- blocks=4   → S=32
- blocks=8   → S=64
- blocks=16  → S=128
- blocks=32  → S=256
- blocks=128 → S=512
- blocks=512 → S=1024

(Square-only mapping; non-square shapes would need a different probe.)

## 3. UntileToRowMajor permutes the marker

BbbKMajor's tile-layout output `[M_t, N_t, 1024]` gets remapped by UntileToRowMajor as:
```
out_rowmajor[(mt*32 + r) * N + nt*32 + c] = tile[(mt*N_tiles + nt)*1024 + r*32 + c]
```

So byte `r*32 + c` of BbbKMajor's tile-0 lands at row-major position `[r*N + c]`. When inspecting the device output, the marker bytes 32..63 (r=1, c=0..31 of tile 0) appear at fp32 cells `[N..N+31]`, NOT at the first 64 cells.

**Decoder helper**:
```python
def u8_at(byte_offset, S, fp32_buf):
    r, c = byte_offset // 32, byte_offset % 32
    return int(np.round(fp32_buf[r * S + c]))
```

## 4. Native pre-packed wt format verified

Wt at offset `(kt * N_t + nt) * 1024` per tile, byte at `(r/4)*128 + c*4 + (r%4)` = `wRaw_KN[kt*32+r, nt*32+c]` as int8. Σ_k W[k, 0] computed in-kernel matches Python reference at all 6 tested shapes (32³–1024³).

## 5. Crouton_8 layout — partially decoded

For square act `[1, M/32, 32, K]`:
- block_cols = 32 (always)
- block_rows = block_size / 32 = (total_bytes / blocks) / 32
- block_size capped at 2 KB (≥256³)
- Within each block: 32-byte stride per row (row-major within block)

| Shape | block count | block_size | block_rows |
|-------|:-----------:|:----------:|:----------:|
| 32³   | 4   | 256  | 8  |
| 64³   | 8   | 512  | 16 |
| 128³  | 16  | 1024 | 32 |
| 256³  | 32  | 2048 | 64 |
| 512³  | 128 | 2048 | 64 |
| 1024³ | 512 | 2048 | 64 |

At ≥256³, block_rows (64) > M_t row group (32) — single Crouton_8 block spans across multiple M_tiles. **Block ordering** when block_rows > 32 not yet verified — needs more probing.

# Code state

`src/HmxMatMulV9SkelOp.cpp` — has `V9_INPUT_PROBE` flag with full input-access probe code. Currently writes a 64-byte marker at the start of BbbKMajor's tile output. Build with:
```bash
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_INPUT_PROBE" bash build.sh
```

`gen_v8c8_test.py` — clean 3-input ONNX (act + wt + bias). No combined-static. No host-side fold.

Lowered graph 6 nodes shape-invariant: `q::*InputSlice → q::ForceFormat_Crouton → q::ConvLayer.opt.weights_to_vtcm@FB.fB. → q::ConvLayer.opt.weights_to_vtcm@Fi.fi. → BbbKMajor → UntileToRowMajor`.

# Next steps for kernel body

## Phase 2 — scalar reference matmul

Concrete plan:
1. Empirically resolve Crouton_8 block ordering at ≥256³ (block_rows=64 spans 2 M_tiles — verify block index formula).
2. Or, simpler: convert act blocks → row-major M×K buffer at the start of the kernel (HVX load+store).
3. Compute scalar matmul: `acc[m,n] = Σ_k act_u8[m,k] × wRaw_i8[k,n] + bias_q[n]`
4. Saturate to u8: `out[m,n] = clip(acc, 0, 255)`
5. Write to tile-layout output: `tile[(mt*N_t + nt)*1024 + r*32 + c]`

Validates math + I/O before HMX inline asm.

## Phase 3 — HMX inner loop

Adapt V8 prod's `hmx_v8_mac_convert` pattern:
- `mxclracc`
- K-loop: `{ activation.ub = mxmem(act, 0x71F):cm \n weight.b = mxmem(wt, 0x3FF) }`
- `mxmem(out, 0):after:cm:sat.ub = acc`

Two open questions for HMX:
- Does `mxmem :cm` directly consume Crouton_8 blocks, or do we need to repack to V8-prod tile format first?
- For native fold semantics (raw u8 act + int32 fold absorbed into bias), do we use `mxmem2` (reads 256B/tile = native fp16 pair + int32 fold)? Or compute fold ourselves into V8-prod fp16-pair (128B/tile)?

The fold computation `effective_int32[c] = -ACT_ZP × Σ_k W[k,c] + bias_q[c]` only needs N MACs — negligible overhead compared to K×M×N MACs in the matmul itself.

# Reproduce input-access probe

```bash
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_INPUT_PROBE" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_INPUT_PROBE" bash build_x86.sh
cd standard_flow/phaseB_v8/gen_out/HmxMatMulPhase3Package_Converter_Op_Package
QNN_SDK_ROOT=/path/to/qnn-sdk make
cd ../..
for S in 32 64 128 256 512 1024; do
    OUT_DIR=/tmp/v8c8_probe_$S M=$S K=$S N=$S bash run_v8c8_phase2.sh > /dev/null 2>&1
done
# Then decode using u8_at(byte_offset, S, fp32_buf) helper (see findings #3)
```
