# `set_hmx_params_conv1x1` 5-arg ABI — empirical probe results (2026-04-28)

Step 5.1 deliverable. Probe in `HmxMatMulV9SkelOp.cpp::V9_PARAMS_PROBE`,
runner `example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v9_params_probe.sh`.
Cross-.so dlsym call to `_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm`
(in libQnnHtpV75Skel.so) works via R_HEX_JMP_SLOT relocation.

## Probe data (16 cases × 0x40-byte descriptor)

```
case   (arg1,  arg2,  arg3,  arg4,  arg5)  → non-zero descriptor fields
   0   (0x0,   0x0,   0x0,   0x0,   0x0)   → +0x0c=0x1f                   +0x18=0x3ff
   1   (0x400, 0x0,   0x0,   0x0,   0x0)   → +0x04=0x400  +0x0c=0x41f     +0x18=0x3ff
   2   (0x600, 0x0,   0x0,   0x0,   0x0)   → +0x04=0x600  +0x0c=0x61f     +0x18=0x3ff
   3   (0x780, 0x0,   0x0,   0x0,   0x0)   → +0x04=0x780  +0x0c=0x79f     +0x18=0x3ff
   4   (0x7c0, 0x0,   0x0,   0x0,   0x0)   → +0x04=0x7c0  +0x0c=0x7df     +0x18=0x3ff
   5   (0x0,   0x8,   0x0,   0x0,   0x0)   → +0x0c=0x7                    +0x18=0xff
   6   (0x0,   0x20,  0x0,   0x0,   0x0)   → +0x0c=0x1f                   +0x18=0x3ff
   7   (0x0,   0x100, 0x0,   0x0,   0x0)   → +0x0c=0x1f                   +0x18=0x3ff
   8   (0x0,   0x0,   0x1,   0x0,   0x0)   → +0x00=0x20    +0x0c=0x1f     +0x18=0x3ff
   9   (0x0,   0x0,   0x10,  0x0,   0x0)   → +0x00=0x200   +0x0c=0x1f     +0x18=0x3ff
  10   (0x0,   0x0,   0x20,  0x0,   0x0)   → +0x00=0x400   +0x0c=0x1f     +0x18=0x3ff
  11   (0x0,   0x0,   0x0,   0x8,   0x20)  → +0x0c=0x1f    +0x18=0x7ff    +0x30=0x20
  12   (0x0,   0x0,   0x0,   0x20,  0x20)  → +0x0c=0x1f    +0x18=0x7ff    +0x30=0x20
  13   (0x0,   0x0,   0x0,   0x0,   0x100) → +0x0c=0x1f    +0x18=0x3ff    +0x30=0x100
  14   (0x0,   0x0,   0x0,   0x0,   0x200) → +0x0c=0x1f    +0x18=0x3ff    +0x30=0x200
  15   (0x0,   0x0,   0x0,   0x0,   0x300) → +0x08=0x1d    +0x0c=0x2      +0x18=0x5f
```

## Field-by-field semantics (mask_desc = first 0x20 bytes, +0x30 ext)

| Off  | Bytes from probe                          | mask_desc role            | Derivation rule                                |
|------|-------------------------------------------|---------------------------|------------------------------------------------|
| +0x00| `arg3<<5` low 11 bits                     | `out_check`               | `(arg3 << 5) & 0x7FF`                          |
| +0x04| `arg1` low 11 bits                        | `out_rt_mask`             | `arg1 & 0x7FF`                                 |
| +0x08| 0 normally, 0x1d when arg5 r13-field>2    | `act_check`               | path-dependent (arg5 bits 8..12)               |
| +0x0c| `(arg1 & 0x7E0) \| 0x1F` normally         | `act_rt_base` (for `:cm`) | base 0x1F + arg1 r2-field bits                 |
| +0x10| (sentinel preserved 0xCDCDCDCD)           | `filter_x_stride`         | not written by 1×1                             |
| +0x14| (sentinel preserved 0xCDCDCDCD)           | `_pad14`                  | not written                                    |
| +0x18| `0x3FF` normally; `0x7FF` if arg5 bit 5   | `alt_rt` (last-K Rt_wt)   | derived from `r9-1` (arg2 r7-field path)       |
| +0x30| `arg5` raw                                | extension (unused by 1×1) | overwritten with arg5 (lower bits cleared)     |

## Canonical args for u8×i8 1×1 matmul

Per `Agent/qnn_re/descriptor_builder_3d7920.S:3d7c00`, the descriptor builder
sets `r1 = 0x700` right before the call to `set_hmx_params_conv1x1`. Plugging
into the table:

```
arg1 = 0x700 (= QNN's r1 in the typical conv1x1 path)
arg2 = 0    or any value ≤ 32 not affecting +0x18 (gives 0x3FF)
arg3 = ?    (controls +0x00 = out_check alignment value;
             0 may be acceptable since `bitsclr(_, 0x7e0)` test passes for 0)
arg4 = 0    (only used if arg5 bit 5 set)
arg5 = 0    (basic mode, no depth/spread)
```

→ produces descriptor:
```
+0x00: 0x000      (out_check — passes bitsclr(0, 0x7e0))
+0x04: 0x700      (out_rt_mask — Rt for sat.ub store; native uses 0x700, NOT 0x3FF)
+0x08: 0x000      (act_check — passes bitsclr(0, 0x7e0))
+0x0c: 0x71F      (act_rt_base — :cm Rt — matches V8's HMX_RT_ACT_CM = 0x71F)
+0x18: 0x3FF      (alt_rt — last-K Rt_wt — matches V8's HMX_RT_WT = 0x3FF)
```

## Important discoveries

1. **`out_rt_mask` (sat.ub Rt) = 0x700, NOT 0x3FF.** V8's current
   inline-asm uses `HMX_RT_WT = 0x3FF` for sat.ub:
   ```c
   asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                :: "r"(out_tile), "r"((int32_t)HMX_RT_WT));
   ```
   But native passes `r11 = 0x700` for the same operation. Either (a)
   both work and we're paying a small perf cost with 0x3FF, or (b) the
   bit-exact V8 we have has been getting away with degenerate output
   (saturation hides any byte-mapping difference — see #2).

2. **Crouton_8 output [1, M/32, 32, N] layout is non-trivial.** Each
   block holds 8 row-chunks of 8 rows × 32 cols at row-stride 32 within
   the row-group, NOT 64 contiguous rows as V9_KERNEL_HMX assumed.
   Empirical mapping (S=256, total 32 blocks):
   ```
   block_idx = ((m % 32) / 8) * out_n_chunks + (n / 32)
   inblk_off = (m / 32) * 256 + (m % 8) * 32 + (n % 32)
   ```
   Block 0 covers logical rows {0..7, 32..39, 64..71, 96..103, 128..135,
   160..167, 192..199, 224..231} × cols 0..31 (8 sub-chunks × 256 B).

3. **V9_KERNEL_HMX bit-exact at 256³ may be artifact of saturation.**
   Reference output has only 2 unique values (0, 255) due to large
   accumulator magnitude, so any byte-permutation produces matching
   output. With non-saturating inputs, V9_KERNEL_HMX's
   `(m%32)*32 + (n%32)` mapping at offset `(mt%2)*1024` would mismatch
   the actual Crouton_8 layout. **Open issue** — verify with smaller
   weights (e.g., int8 in [-2,2]) before trusting V9_KERNEL_HMX as
   ground truth. Bit-exact memory entries assume the native-fold
   probe ran with current saturating inputs.

## Cross-reference

- Disasm: `Agent/qnn_re/set_hmx_params_conv1x1.S` (~70 lines)
- Caller: `Agent/qnn_re/descriptor_builder_3d7920.S:3d7c00..3d7cdc`
- Inner kernel ABI: `Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md`
- Probe runner: `example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v9_params_probe.sh`
- Probe artefacts: `.../phase1_validation/v9_params_probe/`
- Step 5 plan: `docs/v8c8_step5_descriptor_driven_plan.md`
