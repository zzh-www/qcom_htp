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

## Iteration 2 — per-(m,n) K-accumulated kernel

Rewrite `hmx_int4_matmul_tile` → `hmx_int4_matmul_mn_tile`: one
invocation consumes the full K dim for one (m_tile, n_tile) output, with
a single `mxclracc` at the start and 1 dual-scale readback at the end
per partial (instead of (K/32) × 2 readbacks). Big structural win.

| Shape  | cycles/MAC | vs iter 1 |
|--------|-----------:|----------:|
| 32³    |      8.63  |    +3%    |
| 128³   |      3.88  |   −53%    |
| 256³   |      3.11  |   −63%    |
| 512³   |    **2.74** |   **−67%** |

Cycles/MAC now decreases with scale because per-(m,n) overhead stays
constant while K grows. 32³ regresses because K=32 gives no
amortization and we added the upfront full-K decomp cost.

## Iteration 3 — pack micro-optimizations

Three small changes stacked:

- **3a**: replaced 1024 stride-4 byte writes + memset in both
  `pack_activation_32x32` and `pack_weight_32x32` with 4-byte u32 writes
  (no read-modify-write, no memset needed).
- **3b**: added a row-stride variant `pack_activation_32x32_rs` and
  removed the inner 32-memcpy sub-gather in the K loop — we now read the
  activation directly from the row-major [32×K] strip.
- **3c**: **pre-pack activation**. The full activation 32×K is
  decomposed into (A_hi, A_lo) byte streams and packed into VTCM as
  K/32 ready-to-HMX-load tiles **once per m_tile**, then reused across
  all n_tiles. Kills the per-K-iter activation pack.

Cumulative at 512³: 2.74 → 2.58 → 2.44 → **2.17** cycles/MAC.

## Iteration 4 — pre-pack weight too (REGRESSED, reverted)

Extended the prepack to also pack all K weight tiles into VTCM before
the inner HMX MAC loop — the idea was to make the MAC loop purely HMX
issues with zero scalar pack interleaved. Result: **2.17 → 3.51** at
512³ (60% slower).

Diagnosis: HMX `{activation.ub = mxmem(p); weight.b = mxmem(q)}` back-
to-back from two VTCM regions appears to stall on VTCM bank contention.
The scalar `pack_weight_32x32` in the previous iteration naturally
interleaved between MACs and *helped* by decoupling VTCM reads. Moral:
on v75, dense VTCM bandwidth is the bottleneck for this kernel, not
compute. Reverted weight to per-K-iter scalar pack.

## Iteration 2 blocked — multi-threading (self-slicing is HVX-only)

Tried `multithreaded=true` with self-slicing on the M axis. Graph
prepare rejects it: `Can't set self_slicing=6 slices on non-HVX op`.
QNN's self-slicing mechanism is gated on `QHPI_RESOURCE_HVX` alone;
`QHPI_RESOURCE_HMX` ops cannot self-slice. Adding HVX to the bitmask
produces `invalid resource flag 0x6`. Alternative multi-threading
paths (graph-level fan-out, host-side slice dispatch) deferred — this
is a structural QNN constraint, not a kernel issue.

## Summary table

| Iter | Cumulative cycles/MAC @ 512³ | cycles @ 512³ | vs baseline |
|-----:|-----------------------------:|--------------:|------------:|
| 0    | 8.47                          | 1,137 M       |     1×      |
| 1    | 8.33 (weight-pack-hoist)      | 1,118 M       | 1.02×       |
| 2    | 2.74 (K-accumulation)         |   367 M       | 3.10×       |
| 3a   | 2.58 (u32 packed writes)      |   346 M       | 3.29×       |
| 3b   | 2.44 (row-strided pack)       |   327 M       | 3.48×       |
| 3c   | **2.17** (prepack activation) | **291 M**     | **3.91×**   |

## Honest assessment vs targets

| Target          | 512³ cycles | My gap  |
|-----------------|------------:|--------:|
| HMX theo. floor | ~2 M        | ~145×   |
| w8a8 (stretch)  | 66 K        | ~4400×  |
| w8a16 (pass)    | 108 K       | ~2700×  |
| w16a16 (float)  | 677 K       | ~430×   |

Still very far from passing. The **structural** remaining wins needed
to close the gap:

1. **Honest HVX vectorization of all scalar hot paths** — pack loops,
   decomp, combine, col_sum. Each is 5-10× on its own; collectively
   maybe 10-20× overall.
2. **Graph-level parallelism** — partition M into separate subgraphs
   QNN can schedule concurrently, bypassing the self-slicing HVX gate.
3. **Native `weight.n` (int4 nibble) HMX path** — cuts weight VTCM
   traffic in half, possibly 2× throughput in memory-bound regimes.
   Tile layout is undocumented and needs reverse-engineering.

Each of these is substantial (days of work). Without them, the 2.17
cycles/MAC floor is likely the near-term ceiling for this single-thread,
scalar-pack-based implementation.

## Decoded: how QNN built-in MatMul is lowered (from chrometrace_htp.json)

Extracted from `example/qnn_matmul_profile/sweep_data_2026-04-19/s{32,128,256}/`
(tar-extracted) + `.../s512/` for w8a8, w8a16, fp16. Node count per
(size, config):

| size | w8a8 | w8a16 | fp16 |
|-----:|-----:|------:|-----:|
| 32³  |  5   |  5    |  5   |
| 128³ |  5   |  5    |  5   |
| 256³ |  5   |  5    |  5   |
| 512³ | **14** | **14** | **8** |

At ≤ 256³ the compiler emits one big `ConvLayer_s1.opt` (int) or
`ConvLayer.fp16.s1.tcm` (fp16) kernel plus weight/bias staging
(`weights_to_vtcm`, `bias_to_vtcm`, `ForceFormat_Crouton`,
`InputSlice`).

**At 512³ the matmul is tiled across the output plane:**
- `w8a8` / `w8a16`: **4-way** tiling — 2 `InputSlicePad` split activation
  on M into two 256×K halves; 2 `weights_to_vtcm` stage the two 256-N
  halves of the weight; **4 independent `ConvLayer_s1.opt` kernels**
  each compute one [256, 512, 256] sub-tile; then **2 `Concat`** ops
  stitch them back on N, then on M.
- `fp16`: 2-way (only M split) — 2 `ConvLayer.fp16.s1.tcm` each doing
  [256, 512, 512]; 1 `Concat`. Simpler because fp16 HMX tile has
  different VTCM density.

### Byte-level tile size decoding (w8a8 @ 512³)

From each node's `scalar_params.mem_{dram,vtcm}_{read,write}`:

| node                     | bytes      | decoded                                   |
|--------------------------|-----------:|-------------------------------------------|
| InputSlicePad (×2)       | 131,072 ea | 256·512·1 B = half of [512,512] int8 act  |
| weights_to_vtcm (×2)     | 131,072 ea | 512·256·1 B = half of [512,512] int8 wt   |
| bias_to_vtcm (×2)        |   2,048 ea | 256 channels × 8 B (int16 scale)          |
| ConvLayer_s1.opt (×4)    | R 264,192  | act tile 131K + wt tile 131K + bias 2K    |
|                          | W  65,536  | output 256·256·1 B = ¼ of [512,512]       |

Confirms the sub-tile: **[M=256, K=512, N=256]** per `ConvLayer_s1.opt`.

### Graph topology

```
A[1,M=512,K=512] ─┬─ InputSlicePad(M_top)  ─┬─ ConvLayer(A_top, W_left)  → C_00 [256,256]
                  │                         └─ ConvLayer(A_top, W_right) → C_01
                  └─ InputSlicePad(M_bot)  ─┬─ ConvLayer(A_bot, W_left)  → C_10
                                            └─ ConvLayer(A_bot, W_right) → C_11
W[1,K=512,N=512] ─┬─ weights_to_vtcm(W_left)
                  └─ weights_to_vtcm(W_right)
        C_00,C_01,C_10,C_11 ─ Concat_N(×2) ─ Concat_M ─ Output
```

4 ConvLayer kernels are data-independent → QNN scheduler runs them
concurrently across 4 HVX threads. The scalar-overhead-free HMX
throughput plus 4-way thread parallelism is what delivers
**w8a8 @ 512³ = 66 K cycles** (4.96e-4 cycles/MAC).

### Gap reconciliation

Our current: 2.17 cycles/MAC ≈ **4400× slower per MAC** than w8a8. Even
if we replicated the 4-way graph-level parallelism perfectly, we'd
only close a 4× gap → still ~1100× short. The QNN kernel must be
running at ~5e-4 cycles/MAC *per thread* — which means their pack /
decomp / combine / unpack paths are all HVX-vectorized, not scalar.

## Revised roadmap (user directive: 先 HMX 最优 tile 布局, 再 HVX 切片 concat)

1. **P6: HVX-vectorize `pack_activation_32x32`** — highest-leverage
   scalar hot spot. Expected 5-10× on this function alone.
2. **P7: HVX-vectorize remaining scalar paths** (pack_weight, decomp,
   unpack readback, combine, col_sum).
3. **P5 (optional scouting)**: reverse-engineer
   `libQnnHtpV75Skel.so` symbol `ConvLayer_s1.opt` to see the exact
   HVX intrinsic sequence they use — may reveal tile-layout tricks.
4. **P8: Graph-level 2×2 tiling** — emit 4 `Int4MatMulTile` nodes +
   2 `q::Concat` ops in host harness when M,N > 256. Uses QNN
   built-in `Concat`; only our tile kernel is custom.
5. (deferred) Native `weight.n` (int4 nibble) HMX path for VTCM
   bandwidth savings at very large K.

Rough gain estimate stacking P6 → P7 → P8:
- After P6 (pack_activation HVX): cycles/MAC ~1.0 (-55%)
- After P7 (all scalar HVX): cycles/MAC ~0.05 (20× more)
- After P8 (4-way graph tiling): cycles/MAC ~0.015 (4× more)
- Total target: close to w8a16's 8e-4 (17× remaining gap) and within
  reach of w8a8's 5e-4 for some shapes.

## P5 — Decoded QNN built-in kernel, found a bigger structural lever

Disassembled `libQnnHtpV75Skel.so` with `hexagon-llvm-objdump -d`,
searching for HMX instruction patterns. Three key findings:

### 1. v75 HMX instruction set usage in the skel

| suffix         | count | interpretation                        |
|----------------|------:|---------------------------------------|
| activation.ub  |  724  | uint8 activation (dominant int path)  |
| weight.b       |  448  | int8 weight (all `:dilate` — Conv use)|
| weight.n       |  232  | **int4 nibble — alive in v75!**       |
| weight.hf      |  110  | fp16 weight                           |
| activation.hf  |  110  | fp16 activation                       |
| weight.c       |   44  | (chunked?)                            |

`weight.n` is the native int4 path. Every internal use pairs with
`:dilate` modifier, so the QNN compiler uses it via Conv2D-with-dilate=1
(MatMul-as-Conv pattern). Explains why `w4a16` MatMul fails at
compose time — the native int4 HMX path is live but QNN gates it
behind Conv2D-+-LPBQ only.

### 2. Internal ConvLayer kernels have NO HVX pack in the hot loop

Sample disassembly of one of the HMX kernel bodies
(`hmx_convhnh_5x5_stride1` @ 0x214294, inside `loop0/loop1`):

```
; Inner MAC loop body (after pointer setup):
{ activation.ub = mxmem(r13, r8)
  weight.b      = mxmem(r14, r28):dilate }   ; endloop0
...
mxmem(r5, r6):before:sat.uh = acc:2x1        ; readback outside loop
```

All scalar instructions inside the loop are `memw` (pointer loads) and
`addasl` / `sub` / `or` (address arithmetic). **Zero HVX vshuff, zero
byte-level pack ops.** Data is already in HMX-tile-ready format before
the kernel runs.

### 3. QNN has a dedicated HVX pack op: ForceFormat_Crouton

Tested by changing our kernel signature from `QHPI_Layout_Flat4` →
`QHPI_Layout_Crouton_16`/`Crouton_8` + `Storage_Indirect`. Result:

```
graph_prepare.cc:186: Input 0: op=[ForceFormat_Crouton_f2c@CH.FH] output0=[...QUint16Crouton_TCM]
graph_prepare.cc:186: Input 1: op=[ForceFormat_Crouton_f2c@CB.FB] output0=[...QUint8Crouton_TCM]
```

**QNN's optimizer auto-inserts `ForceFormat_Crouton_f2c` in front of
any kernel whose input signature demands Crouton layout.** This is the
same HVX-optimized pack the built-in ConvLayer uses. Conversion is:
flat half (FH) → crouton half (CH), flat byte (FB) → crouton byte (CB).

Our kernel graph-finalizes successfully under this signature but
produces wrong output — the kernel body still calls
`qhpi_tensor_raw_data()`, which is Direct-storage-only; for
Indirect-storage Crouton input we need `qhpi_tensor_block_table()`
plus an understanding of how to iterate the 2 KB blocks.

### Crouton layout shape — `R4CroutonLayout` definition

From `tools/qnn-sdk/include/QNN/HTP/core/memory_layout.h:311`:

```cpp
class R4CroutonLayout : public ChunkedMemoryLayout<4, 0,0, 1,0, 2,0, 3,0,
                                                    1,8, 2,8, 3,32> {};
```

4-rank tensor chunked by 8 (dim 1), 8 (dim 2), 32 (dim 3). Chunk volume
= 8·8·32 = 2048 elements.
- `Crouton_8`:  2 KB per chunk — **matches HMX activation tile byte
  count exactly** (2 KB).
- `Crouton_16`: 4 KB per chunk (twice HMX tile).
- `Crouton_32`: 8 KB per chunk.

The Indirect storage block_table returns pointers to these chunks in
their storage order (VTCM-resident blocks of 2–8 KB).

## Revised plan (supersedes prior P6/P7)

**P9 (new, high-priority):** Rewrite kernel body to consume Crouton
block-table directly. After this change:
- Delete `pack_activation_32x32_rs`, `pack_weight_32x32`, and all the
  per-tile pack scratch — framework handles it via
  `ForceFormat_Crouton`.
- Delete the per-K-iter activation decomposition into sg_A_h/sg_A_l —
  handle it via a one-pass HVX operation at the tile-load boundary (or
  possibly fold it into the HMX accumulator setup; need to see if
  Crouton_16 chunk pair-layout makes it automatic).
- Kernel body becomes: for each (m_tile, n_tile): clracc → K-loop of
  HMX MAC pair loads from block-table pointers → dual-scale readback →
  combine into int32 output.

Expected impact: reduces kernel to roughly match internal ConvLayer's
scalar footprint (~few hundred cycles per tile, not tens of
thousands). If this works, single-thread cycles/MAC should collapse
from 2.17 toward O(1e-3) — close to QNN's built-in performance.

P6 (HVX-vectorize pack_activation) and P7 (HVX-vectorize rest of
scalar) are **obsoleted by P9** — the entire problem they solve gets
moved out of our kernel into QNN's framework op.

## Crouton probe result (sub-agent, 2026-04-20)

Swapped signatures to `QHPI_Layout_Crouton_{16,8}` + `Storage_Indirect`.
QNN successfully auto-inserted `ForceFormat_Crouton_f2c@{CH.FH, CB.FB}`
upstream of our op — mechanism confirmed. But Crouton output format is
NOT HMX-tile-ready:

| Tensor                     | block_table_length | block_shape       |
|----------------------------|-------------------:|-------------------|
| activation [1,1,32,32] u16 | 8                  | [1, 8, 2, 32]     |
| weight [1,1,32,32] u8      | 4                  | [1, 8, 8, 32]     |

Activation block 0 bytes: LE uint16 with adjacent-M-pair-interleaved
along K. Our HMX mxmem expects 4-byte slots of (pad, s0, pad, s1).
Not the same — a secondary pack would still be needed.

Combined with re-reading ch03's HVX+HMX example (which uses Flat4 +
Direct + plain mxmem on homogeneous fp16), the conclusion is: **stay
on Flat4 + Direct** (ch03 architecture), keep our own pack (we still
need int16→byte-split for u8·i8 HMX path), and optimize pack in-place.
Crouton route is a dead end for int16×int8 matmul.

## P6 — fused prepack_activation (completed 2026-04-20)

Eliminated the DDR round-trip by fusing decomp + pack directly from the
input VTCM uint16 tensor into the output VTCM HMX tile (no intermediate
sg_A_h / sg_A_l / a_strip writes).

| Shape  | Pre-P6 | Post-P6 | Δ     |
|--------|-------:|--------:|-------|
| 32³    |  8.47  |  7.79   | −8%   |
| 128³   |  3.88  |  3.21   | −17%  |
| 256³   |  3.11  |  2.48   | −20%  |
| 512³   |  2.17  |  2.12   | −2%   |

Big wins at mid sizes where activation prepack was a larger fraction;
negligible at 512³ where it was already amortized. All bit-exact.

Tried and REVERTED within P6:
- HVX `pack_activation_32x32_rs`: vshuffe_b de-interleaves even-indexed
  bytes (drops odd halves) — wrong semantics. Would need vshuff (not
  vshuffe). Deferred; function is <1% of runtime after fused prepack.
- Fused weight path (stride-N read from wu inside pack): **2× REGRESSION**
  (4.33 vs 2.13 @ 512³). Stride-N VTCM reads hit bank conflicts in a
  tight loop; the contiguous w_col DDR+cache path is ~2× faster.
- w_col placed in VTCM: **2.7× REGRESSION** (5.75 vs 2.13 @ 512³). VTCM
  scalar reads compete with HMX mxmem for the same VTCM banks, serializing.

**Key learning**: on HTP v75, DDR+L2-cache beats VTCM for scalar-heavy
hot loops. VTCM is strictly for HMX `mxmem` loads. Don't move data to
VTCM unless HMX will consume it.

## P7 (HVX-vectorize remaining scalars) — deferred

Net budget for all remaining scalar hot paths combined: ≤5% of runtime.
HVX effort with debug cost does not pay back here. Would become
meaningful IF we had cross-tile parallelism (P8) to overlap HVX work
with HMX compute on separate threads. Deferred.

## P8 — QHPI auto-tile callbacks (registered, not active at test scales)

Registered `shape_required` + `shape_legalized` + `build_tile`. QNN's
central tiler successfully creates multiple sub-op nodes when the
output exceeds the alignment:

- `shape_required = 32`: 16-way tiling at 512³ → **51% REGRESSION**
  (per-op setup overhead, K-accumulation amortization lost).
- `shape_required = 256`: 4-way tiling at 512³ → **5% regression**
  (sub-ops serialize on the single HMX unit).
- `shape_required = 2048`: no tiling at test scales → identical to
  single-op performance.

The machinery works (verified bit-exact at 512³ / 1024³ with tiling
enabled), but without HVX-vectorized pack/unpack to run concurrently
on separate HVX threads, tiled sub-ops just serialize on the single
HMX compute unit and the single scalar main thread. Setting
`shape_required = 2048` effectively disables tiling at test scales
while leaving the machinery ready for a future session that combines
auto-tile with HVX parallelism (P7+P8 together).

## Final state (2026-04-20)

| Shape  | baseline | final | speedup |
|--------|---------:|------:|--------:|
| 32³    |  8.47    | 7.79  | 1.09×   |
| 128³   |  3.88    | 3.21  | 1.21×   |
| 256³   |  3.11    | 2.48  | 1.25×   |
| 512³   |  8.33    | 2.12  | 3.93×   |

Total at 512³: 2.12 cycles/MAC = 285 M cycles per 512³ inference.
Bit-exact correctness at all shapes.

Gap to target: w8a16 at 108 K cycles — still **2600× away**. Physical
upper bound with our 2-packet-per-tile decomposition is ~108 K (same
packet count as w8a16). The remaining gap is almost entirely in the
HMX MAC loop cycle-per-packet — built-in kernels achieve ~16 cyc/packet
(evidence: 66K cycles for 4096 packets at 512³ w8a8), our kernel is
running at ~35 K cycles per 32³ logical tile at K=∞ asymptote, i.e.
~17 K cyc/packet. That's 1000× over built-in.

The 1000× per-packet gap suggests our HMX MAC issues are not
pipelining: every `mxmem` load stalls for VTCM latency (~100+ cycles)
with no overlap. Built-in uses `:dilate` modifier + specific access
patterns that pipeline through the VTCM port. Matching this requires
either reverse-engineering `:dilate` semantics + tile layout OR
adopting the Crouton layout (which is what ConvLayer uses). The
Crouton path was probed (found accessible but non-trivial to consume)
and left as the most promising next avenue, above P7/P8.
