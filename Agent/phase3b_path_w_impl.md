# Phase 3B Path W — Four HVX Ops Implementation

Split the monolithic HMX MatMul custom op (Phase 2) into dataflow-graph
stages. This document captures the API, layout, and wiring decisions
for the four HVX marshalling ops that sit around the pure-HMX core.

## Files

- `example/hmx_matmul_phase3/kernel/pack_act_hvx.c`
- `example/hmx_matmul_phase3/kernel/pack_wt_hvx.c`
- `example/hmx_matmul_phase3/kernel/combine_hi_lo_hvx.c`
- `example/hmx_matmul_phase3/kernel/int4_expand_hvx.c`
- `example/hmx_matmul_phase3/src/HmxMatMulPhase3Interface.cpp` (updated)
- `example/hmx_matmul_phase3/build.sh` (updated — separate kernel compile step)
- `example/hmx_matmul_phase3/test_ops_sim.c` (sim harness)

All four new ops are registered in `qhpi_init` via `register_*` wrappers.
The ops co-exist with the existing Phase 3A `MatMulInt8xInt8Crouton` probe
and Agent B's `MatMulV2`; package now advertises six op names.

## API signatures

```c
void pack_act_hvx_kernel_body(const uint16_t *au,
                              uint8_t *out_hi, uint8_t *out_lo,
                              int M, int K);
void pack_wt_hvx_kernel_body (const int8_t *w,
                              int8_t *out,
                              int K, int N);
void combine_hi_lo_hvx_kernel_body(const int32_t *P_hi,
                                   const int32_t *P_lo,
                                   const int32_t *col_sum,
                                   int32_t *out,
                                   int M, int N);
void int4_expand_hvx_kernel_body(const uint8_t *in,
                                 int8_t *out,
                                 int K, int N);
```

All QHPI entry points declare `QHPI_RESOURCE_HVX` with `multithreaded=true`
so QHPI's self-slicer can parallelize each body across the four HVX worker
threads along the outermost tile-index axis.

### Op → tensor contract

| Op                         | Inputs                                        | Outputs                                |
|----------------------------|-----------------------------------------------|----------------------------------------|
| `PackActivationToHmxTile`  | `QUInt16 [1,1,M,K]`                           | `QUInt8 [1,M/32,K/32,2048]` ×2 (hi,lo) |
| `PackWeightToHmxTile`      | `QInt8  [1,1,K,N]`                            | `QInt8  [1,K/32,N/32,1024]`            |
| `CombineHiLo`              | `Int32 [1,1,M,N]` (P_hi), `Int32` (P_lo), `Int32 [1,1,1,N]` (col_sum) | `Int32 [1,1,M,N]` |
| `Int4Expand`               | `QUInt8 [1,1,K,N/2]`                          | `QInt8 [1,1,K,N]`                      |

All I/O is Flat4 + Direct + `DDR_OR_TCM`. No Crouton, no Indirect — these
ops are pure data movers and the HMX core is already probed separately
against Crouton (Phase 3A). Keeping them in flat layout lets QNN decide
placement; we'll re-evaluate if the compiler inserts extra ForceFormat ops
once the graph is wired.

## Implementation decisions

1. **Algorithm reuse, not rewrites.** Each op body is lifted verbatim from
   validated Phase 2 code:
   - `pack_act`: the `mask 0xFF00 + Q6_Vh_vshuff_Vh` (hi) /
     `Q6_Vh_vasl_VhR(_, 8) + Q6_Vh_vshuff_Vh` (lo) loop from
     `hmx_int4_prepack_activation_fused`.
   - `pack_wt`: two back-to-back `Q6_Vb_vshuff_Vb` per 128-byte kg chunk
     from `pack_weight_32x32` (proof: `Agent/hvx_4way_byte_transpose_re.md`).
   - `combine`: `vasl(hi,8) + vadd(lo) - vasl(col_sum,15)` per vector,
     from the tail of `hmx_int4_matmul_mn_dualacc`.
2. **Self-gather inside `pack_wt`.** Phase 2's packer assumed a contiguous
   32×32 input tile. Here the input is the full `[K,N]` tensor; we gather
   the 32×32 slice into a 1 KB stack buffer per (kt, nt) pair, then apply
   the 2×vshuff transpose. This keeps the HVX body identical to Phase 2
   at the cost of one 1 KB scratch copy per tile.
3. **Two outputs for `pack_act`.** Hi and lo streams go to separate output
   tensors rather than a doubled-size single tensor. That keeps each
   downstream HMX MAC pass (P_hi = Σ A_hi·W, P_lo = Σ A_lo·W) pointing at
   a single tensor with the natural `[M/32, K/32, 2048]` layout.
4. **`int4_expand` sign-extension.** HVX lacks a byte-arithmetic-shift.
   We use `vand(0xF0) → vlsr_Vuh(_, 4) → vand(0x0F)` for hi nibbles (the
   pre-mask kills neighbor-byte bleed), then `vcmp_gt_VubVub >= 8` → `vmux`
   a `16`-splat and `vsub_b` to sign-extend. Interleaving is via
   `Q6_W_vshuff_VVR(hi, lo, -1)` (byte granularity).
5. **Linkage.** Kernel .c files are compiled as C++ (deprecated-but-works)
   so QHPI's C++-style default-args compile; `register_*` functions wear
   `extern "C"` so the Interface can bind them by plain C name.

## Suggested graph wiring (w4a16)

```
 Activation [QUInt16 1,1,M,K]  -----> [PackActivationToHmxTile] --> A_hi [...2048]
                                                                \-> A_lo [...2048]

 Weight     [QUInt8  1,1,K,N/2] --> [Int4Expand] --> W [QInt8 1,1,K,N]
                                                       \--> [ReduceSum axis=K] --> col_sum [Int32 1,1,1,N]
                                                       \--> [PackWeightToHmxTile] --> W_tiles [...1024]

 A_hi, W_tiles --> [MatMulV2 hi-stream]  --> P_hi [Int32 M×N]
 A_lo, W_tiles --> [MatMulV2 lo-stream]  --> P_lo [Int32 M×N]

 P_hi, P_lo, col_sum --> [CombineHiLo] --> result [Int32 M×N]
```

For w4a8: drop `pack_act` (activation stays uint8 after Cast, pack via a
simpler op that still needs writing), and replace the two MatMul passes
with a single u8·i8 MatMul — `combine` degenerates into
`out = P - (col_sum << 7)` so it could be specialized but for now use
the same op with `P_hi=0`.

For w16a16: extra weight-hi/lo path mirrors activation decomp — same four
ops suffice with a slight sig change (int16 weight input) if we extend
`pack_wt` to do the same hi/lo split.

## Verification

`test_ops_sim.c` links the four kernel bodies plus a scalar reference for
each and compares byte-for-byte / int-for-int. Stubs out the QHPI tensor
accessors so the sim binary can call body functions directly without
setting up QnnHtp graphs. Intended run:

```
$HEX_CXX -O2 -mhvx -mhvx-length=128B -mv75 \
    -I $QNN_SDK_ROOT/include/QNN \
    -DPREPARE_DISABLED -DTHIS_PKG_NAME=HmxMatMulPhase3Package \
    kernel/*.c test_ops_sim.c -o test_ops_sim
hexagon-sim -mv75 test_ops_sim
```

(Not run yet — queued for the user per the task brief.)

## Build status

`bash build.sh` green as of this commit — produces
`libQnnHmxMatMulPhase3_htp.so` with all six registered ops, plus the ARM
CPU fallback and the two existing host runners.
