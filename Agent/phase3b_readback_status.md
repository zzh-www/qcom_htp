# Phase 3B `:cm` readback debug status (2026-04-23)

> ⚠️ Historical note from Phase 3A/B research. Files referenced below
> (`kernel/pack_act_hvx.c`, `kernel/hmx_core{,_v2}.c`, `src/HmxMatMulV{2-7}Op.cpp`,
> `src/run_phase3_probe.cpp`, etc.) were moved to
> `example/hmx_matmul_phase3/_archive/` in the 2026-04-25 V8-only cleanup.
> Content of this note is still accurate as RE history. See
> `docs/qnn_custom_op_sop.md` for the current V8 path.


## Current state

**HMX path works and fast** — `:cm` + row-major activation + 2K-aligned VTCM:
- 32³ cyc/MAC = **1.99** (Phase 2 was 2.72 for same-scope kernel)
- Graph finalize, MAC issues, dual-scale readback all execute
- Row 0 (oBuf[0..3]) = 32 ✓ correct for constant all-1 input
- But 512/1024 output cells still wrong (all with value 0 where expected 32)

## Diagnosis path so far

| config | act VTCM align | wt format | readback idx | mismatches | max_err |
|--------|----------------|-----------|--------------|-----------:|--------:|
| baseline | offset 512 (1K) | Phase 2 packed | phys*64+2*jc+stream | 1024 | 12273 |
| | offset 512 | row-major | phys*64+2*jc+stream | 1024 | 12273 |
| | offset 2048 (2K) | row-major | phys*64+2*jc+stream | 512 | 32 |
| | offset 2048 | row-major | ir*32+jc | 512 | 32 |
| | offset 2048 | row-major | ir*64+2*jc | 768 | 32 |

Key findings:
- **2K alignment critical** — without it, all 1024 wrong (huge max_err)
- **Weight format row-major** works same as Phase 2 packed for constant input
  (both leave 512 wrong; discriminate needed via non-uniform test)
- **Readback dual-scale bias NOT required** for small K (Agent A probe used
  single store with bias=0x4000 for A=W=1, K=32 → got 32 directly)
- **Row-major readback and Phase 2 readback give same 512 mismatches count**
  for constant input — the error is which CELLS are zero

## Outstanding question

For all-1 input, expected 32 per cell (32 K × 1 × 1). Pattern of wrong cells:
- 512 = half the grid
- max_abs_err = 32 → wrong cells have value 0
- oBuf[0..3] = 32 (first 4 cells correct)

Likely: `:cm` writes readback to only half the indexed positions (stream 0
at Phase 2 stream interleave). The other half stays zero or unused.

Next-session action: write a KERNEL-side diagnostic that dumps raw
`out_lo[0..1023]` halfwords into the output tensor's first ~2 KiB of int32
so host can read the FULL raw readback without any decoder logic. Then
visually inspect: for all-1 input, which halfword positions contain 32 and
which contain 0. That maps the exact `:cm` readback layout.

## Where to pick up

File: `example/hmx_matmul_phase3/src/HmxMatMulV2Op.cpp`, the readback decode
loop (around lines 135-145). Replace the decode with a raw dump:

```c
/* DIAG: dump raw out_lo as int32 into first 1024 elements of out[] */
for (int i = 0; i < 1024; i++)
    out[i] = (int32_t)out_lo[i];
/* DIAG 2: dump raw out_hi */
for (int i = 0; i < 1024 && (i + 1024) < M*N; i++)
    out[i + 1024] = (int32_t)out_hi[i];
return QHPI_Success;
```

Run at shape 32,32,32 with aBuf=wBuf=1. Then print oBuf pattern (which
indices have 32, which have 0). From that pattern, derive the `idx(ir,jc)`
formula. Then fix the decoder.

## Perf note for context

Current cyc/MAC at 32³ = 1.99 is already near the Phase 2 post-optim value
(2.72 in Phase 2 was a weaker comparator since Phase 2 was 32³ pre-Path-B).
Once readback correct + scaling to 512³, expect to see the 7.9 cyc/packet
QNN-peak benefit that Agent A's probe demonstrated.

## Files modified

- `example/hmx_matmul_phase3/src/HmxMatMulV2Op.cpp` (current state has
  row-major readback, 2K-aligned tiles, constant-input test data)
- `example/hmx_matmul_phase3/src/run_matmul_v2.cpp` (constant all-1 input)
- `example/hmx_matmul_phase3/kernel/hmx_core_v2.c` (row-major wt gather,
  single-scale readback)
