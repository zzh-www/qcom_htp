# int4×int16 MatMul — iterative optimization log

Device: **SM8650 v75** (ssh oneplus). Measurement harness:
`example/hmx_matmul_qnn/run_on_device.sh --shape M,K,N` → `run_int4_matmul`
with 1 warmup + 5 steady-state iters. Metric: `Accelerator (execute) time
(cycles)` from QNN profile. Min/max spread <0.01% across iters (highly
stable, no bimodality).

Comparison baselines from
`example/qnn_matmul_profile/bench_data_2026-04-19` (QNN-built-in MatMul,
via `profile_all.sh`, QHAS `timeline_cycles`):

| config  | 512³ `timeline_cycles` | cycles/MAC @ 512³ |
|---------|-----------------------:|------------------:|
| w8a8    | 66,513                 | 4.96e-4           |
| w8a16   | 108,573                | 8.09e-4           |
| fp16    | 169,382                | 1.26e-3           |
| w16a16  | 677,092                | 5.04e-3           |

(All four are HMX-resident. These are the numbers to beat.)

**HMX theoretical floor for int4×int16** (from plan):
- Per 32×32×32 tile: 2 MAC packets + 2 converts ≈ 4 HMX-issue events
  ≈ ~500 cycles.
- For 512³ = 4096 tiles × ~500 = **~2 M cycles** (≈ 1.5e-2 cycles/MAC).

## Iteration 0 — baseline (2026-04-19)

HMX kernel adapted from `int16_matmul_hmx.c`; Op.cpp tiles 32×32×32 over
(M,N) and accumulates over K. Static globals hold large buffers (moved
off stack after fixing graphExecute err 1003).

| Shape  | Avg cycles    | cycles/MAC | vs HMX floor | vs w8a16 |
|--------|--------------:|-----------:|-------------:|---------:|
| 32³    |       280,209 |      8.55  |      ~570×   |    n/a   |
| 128³   |    17,750,580 |      8.46  |      ~565×   |    n/a   |
| 256³   |   142,100,170 |      8.47  |      ~565×   |    n/a   |
| 512³   | 1,137,197,044 |      8.47  |      ~565×   | **10,474×** |

Analysis: constant ~8.47 cycles/MAC across scales means the overhead is
**per-element, not per-tile or per-graph**. Profile confirms: pack + unpack
+ scalar combine dominate. HMX MAC itself is <5% of cycles.

Suspects ranked by opportunity:
1. `hmx_int4_matmul_tile` scalar activation decomposition: per-element
   `au = (int32)a[i] + 32768; A_h[i] = au>>8; A_l[i] = au & 0xFF` — runs
   1024× per tile, ~3–5 scalar ops each → ~5000 scalar instructions/tile.
   **HVX can do this 64-at-a-time** with `Q6_Vh_vshuff` or byte unpacks.
2. `pack_activation_full` / `pack_weight_full` — strided scatter writes
   into HMX tile layout; 1024 individual byte writes per tile × 3 calls
   (weight packed per partial + both activations). HVX vstore w/ shuffle
   could fold multiple rows per store.
3. `gather_a_tile` / `gather_w_tile` (Op.cpp) — linear row copy from
   user buffer into local tile, with signed/unsigned offset shift. At
   512³ this touches 512·32 = 16K elements per tile, 4096 tiles — a huge
   amount of data motion.
4. Final combine in kernel: 1024 scalar multiply-adds per tile.

## Iteration 1 — hoist weight pack out of the per-partial loop

Change: `hmx_partial_dual_scale` was packing the weight tile redundantly
every call, but weight is the same for both A_hi and A_lo partials within
a 32³ tile. Move `pack_weight_full` up to `hmx_int4_matmul_tile`.

| Shape  | Avg cycles    | cycles/MAC | Δ vs iter 0 |
|--------|--------------:|-----------:|------------:|
| 32³    |       275,766 |      8.41  |   −1.6%     |
| 128³   |    17,470,784 |      8.33  |   −1.5%     |
| 256³   |   139,805,600 |      8.33  |   −1.6%     |
| 512³   | 1,118,570,420 |      8.33  |   −1.7%     |

Small win (≈1024 byte-writes saved per tile out of ~20K scalar ops). All
shapes still bit-exact. The 1.7% number confirms what the per-MAC cost
breakdown suggests: no single scalar hot spot dominates; everything is
distributed across pack / unpack / decompose / combine.

## Next step — K-accumulation inside the HMX kernel

The 8.33 cycles/MAC floor is dominated by *per-32³-tile* scalar overhead,
which gets paid `(M/32) × (N/32) × (K/32)` times. For 512³ that's 4096
tile invocations — each re-decomposing activation, packing both
operands, running 2 MAC packets, unpacking, and combining.

Structural fix: keep the HMX accumulator hot across K-tiles. The u8·i8
accumulator is int32 and comfortably holds `K/32 × 1M ≤ 16M` (for 512³).
New kernel shape:

```
for each (m_tile, n_tile):
    clracc
    for each k_tile:
        pack A_hi and W for this k_tile
        activation.ub = mxmem; weight.b = mxmem    # MAC accumulates
    readback -> P_hi
    clracc
    for each k_tile:
        pack A_lo and W
        activation.ub = mxmem; weight.b = mxmem
    readback -> P_lo
    out[m,n] = (P_hi << 8) + P_lo - 32768 · col_sum_w
```

This amortizes decompose / unpack / combine across all K-tiles. Expected
gain at 512³ with K=512: 16× reduction in per-tile overhead on the
scalar-dominant paths → target ≤0.5 cycles/MAC. Weight pack still runs
per (k_tile, n_tile) pair — could hoist further if we cache packed
weights in VTCM (I1 in the plan).

Target for iteration 2: **`cycles_per_MAC ≤ 0.5` at 512³**. This brings
us within ~30× of the HMX floor — still short, but closes the biggest
structural gap.

## Known issue: warning "Selecting disabled op for HmxInt4MatMulPackage"

Present at every graphFinalize. TBD whether this actually prevents HMX
scheduling or whether it's cosmetic. If HMX is being serialized to a
single thread under this flag, self-slicing (`multithreaded=true`)
combined with graph-level HVX threads would unlock 4× parallelism on
top of whatever the kernel does. Investigate by instrumenting the
kernel entry with `FARF` and counting invocations per execute.
