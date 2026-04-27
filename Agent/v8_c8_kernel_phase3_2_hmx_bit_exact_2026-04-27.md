---
name: V8 C8 kernel Phase 3.2 — HMX inline asm BIT-EXACT 32³–1024³ 2026-04-27 night
description: Replaced V9_KERNEL_SCALAR's inner MAC with HMX :after:cm:sat.ub against Crouton_8 act + native bias fold. BIT-EXACT vs reference at every shape (1.4M cells total). S≥128 uses direct Crouton-block ptr arithmetic; S<128 falls back to scalar (HMX :cm faults on stack DDR scratch).
type: project
---

# Result

**BIT-EXACT 1,397,760 / 1,397,760 cells** across shapes 32³, 64³, 128³, 256³, 512³, 1024³.
HMX kernel body now produces native-fold matmul output that matches the
centered reference `(act - ACT_ZP) @ wt + bias_q` saturated to u8.

| S    | path           | match    | max_diff |
|------|----------------|----------|----------|
| 32   | scalar fallback| 1024/1024| 0        |
| 64   | scalar fallback| 4096/4096| 0        |
| 128  | HMX direct     | 16384/16384 | 0     |
| 256  | HMX direct     | 65536/65536 | 0     |
| 512  | HMX direct     | 262144/262144 | 0   |
| 1024 | HMX direct     | 1048576/1048576 | 0 |

# Crouton_8 → HMX :cm tile mapping

The HMX `:cm` instruction reads 1024 contiguous bytes as a 32×32 row-major u8
tile. Crouton_8 block layout = `block_rows × 32 cols` row-major where
`block_rows = min(M/4, 64)`. Block index `bi = rg * k_chunks + kc` with
`rg = m / block_rows`, `kc = k / 32`.

| S    | block_rows | block bytes | mapping (mt, kt) → HMX tile ptr        |
|------|------------|-------------|----------------------------------------|
| 32   | 8          | 256         | 4 blocks needed → scalar fallback      |
| 64   | 16         | 512         | 2 blocks needed → scalar fallback      |
| 128  | 32         | 1024        | block_table[mt * k_chunks + kt]        |
| 256+ | 64         | 2048        | block_table[(mt/2) * k_chunks + kt] + (mt%2)*1024 |

Generalised: `mt_per_block = block_rows / 32`,
`ptr = block_table[(mt/mt_per_block) * k_chunks + kt] + (mt%mt_per_block)*1024`.

# Why scalar fallback at S<128

HMX `:cm` requires the 1024-byte tile to be VTCM-resident contiguous.
Stack-allocated scratch is DDR — issuing `mxmem(stack_ptr):cm` faulted with
`SIGSEGV SEGV_MAPERR` at S=32 (was the first thing tried). Adding a 4th
VTCM tensor input is possible but requires sig/XML/gen-script changes.
Scalar at S∈{32,64} costs at most 32×32×64 = 65K MACs total per output tile
band, fast enough.

# Bias path (carried over from Phase 3.1)

`bias = mxmem2(bias_n)` where `bias_n` points to per-N-tile 256-byte block:
- bytes 0..127 = 32 × (fp16 scale, fp16 baseline) — used by `:after:cm:sat.ub`
  for `out = saturate_u8(top9(baseline) + floor(acc × scale_fp16 / 512))`.
- bytes 128..255 = 32 × int32 effective_bias = `-ACT_ZP × Σ_k W[k,c] + bias_q[c]`,
  applied as initial accumulator value at MAC start.

For our test (scale=1, out_zp=0): `out = saturate_u8(acc)`, where `acc`
absorbs the int32 fold so accumulator math stays valid for raw u8 act ×
signed wt.

# Inner-loop structure

```c
asm volatile("mxclracc" ::: "memory");
for (nt) {
    bias_n = bias_bytes + nt*256;
    asm volatile("bias = mxmem2(%0)" :: "r"(bias_n));   // band header
    for (mt) {
        out_tile = out_buf + (mt*N_t + nt)*1024;
        asm volatile("bias = mxmem2(%0)" :: "r"(bias_n)); // re-load (V8-prod safety)
        for (kt) {
            act_tile = block_table[(mt/mt_per_block)*k_chunks + kt] + (mt%mt_per_block)*1024;
            wt_tile  = wt_pack + (kt*N_t + nt)*1024;
            asm volatile(
                "{ activation.ub = mxmem(%0,%1):cm\n"
                "  weight.b      = mxmem(%2,%3) }"
                :: "r"(act_tile), "r"(HMX_RT_ACT_CM),
                   "r"(wt_tile),  "r"(HMX_RT_WT));
        }
        asm volatile("mxmem(%0,%1):after:cm:sat.ub = acc"
                     :: "r"(out_tile), "r"(HMX_RT_WT));
    }
}
```

# What changed from Phase 3.1

- New build flag: `EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX"`
- `HmxMatMulV9SkelOp.cpp:106-...` adds the `V9_KERNEL_HMX` branch that
  short-circuits to scalar at `block_rows<32` and otherwise issues
  HMX inline asm against Crouton block ptr arithmetic.
- gen_v8c8_test.py — unchanged (native fold layout still valid).
- XML — unchanged (BbbKMajor sig still 3-input act/wt/bias).
- run_v8c8_phase2.sh / sweep — unchanged.

# Reproduce

```sh
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh
cd standard_flow/phaseB_v8
for S in 32 64 128 256 512 1024; do
    M=$S K=$S N=$S OUT_DIR=phase1_validation/v8c8_hmx_${S} bash run_v8c8_phase2.sh
done
```

# Next steps (not done in this session)

1. Multi-instance graph split for ≥2048³ (VTCM 12 MiB > 6 MiB ceiling).
   Port M_TILE/N_TILE adaptive lowering from `gen_v8_graph.py` into
   `gen_v8c8_test.py`.
2. Perf measurement: V9 C8 vs native at 256³–1024³. C8 path has 5-node
   lowered graph (vs native's 8) — should be at least as fast as our V8
   path; check whether bypassing pack_act_rm (now done by
   q::ForceFormat_Crouton) removes the ~3× pack_act gap noted in
   `v8_vs_native_optrace_2026-04-25.md`.
3. Real (non-degenerate) quant scales: current test uses
   scale=1/out_zp=0 to stay in the `saturate_u8(acc)` regime. Validate
   with random scale, non-zero zp via existing fp16 pair encoding.
