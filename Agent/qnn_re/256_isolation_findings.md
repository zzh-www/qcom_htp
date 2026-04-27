# 256³ native isolation + ConvLayer_s1.opt entry RE (2026-04-26)

## Why 256³

At 256³ a u8×i8 MatMul lowers to **exactly one** `q::ConvLayer_s1.opt`
event (verified from optrace). No `@Spill/@Fill`, no multi-tile splitting,
no concurrent HMX instances. This eliminates QNN's graph-compile-time
scheduling tricks as a confound and isolates the per-kernel cost.

## End-to-end measurements (SM8650 v75, perf_profile=burst)

| Shape | Path     | Wall µs | Cycles | per-op kernel cyc | gap vs native |
|-------|----------|--------:|-------:|------------------:|---------------|
| 256³  | native   |    79   | 18,672 | 11,595 (MatMul_0) | —             |
| 256³  | V8       |   126   | 67,008 | (single mmv8)     | **1.6× wall / 3.6× cyc** |

Cycle counts above are from `qnn-profile-viewer` with the standard reader,
3rd inference (warm). The native cold-run was 497 µs / 51,340 cyc; warm
runs 2 and 3 were both ~80 µs, so the 79 µs is stable.

## Per-op breakdown (native 256³ optrace, single inference)

```
ConvLayer_s1.opt          5,964 cyc   ← the HMX kernel itself
ForceFormat_Crouton      17,909 cyc   ← act packing into Crouton tile layout
*InputSlice (×2)          6,408 cyc   ← DMA fetch + DmaCheckpointWait
*OutputSlice (×2)         5,305 cyc
ForceFormat_Flat          5,509 cyc   ← unpack tile-layout → row-major DDR
ConvLayer.opt.weights_to_vtcm  1,561 cyc
ConvLayer.opt.bias_to_vtcm     3,336 cyc
DmaCheckpointSet          ~3,000 cyc
                          ─────────
total timeline           49,946 cyc   ≈ 28 µs (graph_execute_us)
```

The kernel itself is only **5,964 cyc** — the bulk of timeline is act/wt
DMA + Crouton packing + slicing.

For matmul at 256³ K=256, M_tiles × N_tiles = 8 × 8 = 64 output tiles,
each with K/32 = 8 K-iterations of 2-MAC unrolled. So total MAC packets
≈ 64 × 8 = 512. At theoretical 7.89 cyc/packet that's ~4,040 cyc — the
observed 5,964 cyc has ~50 % overhead beyond MAC steady-state, plausibly
from sat.ub drain + outer-loop bias load.

## Where the kernel lives

Disassembly artefacts saved under `Agent/qnn_re/`:
- `skel_text_full.S` — full `.text` of `libQnnHtpV75Skel.so` (60 MB)
- `convlayer_s1_wrapper_3d7620.S` — the dispatcher
- `descriptor_builder_3d7920.S` — descriptor-construction subroutine
- `hmx_convbbb1x1_stride1_2ea740.S` — the actual MAC kernel

### Entry stack

```
ConvLayer_s1.opt graph node
   ↓
0x3d7620   wrapper (allocframe 0xd8)         ← visible name is wrong
   │       (mis-attached to "_Z17code_to_type_nameI...pkWeightsF16_TCM…")
   │       Reads input/weight/output tensor descriptors,
   │       computes per-axis tile counts,
   │       chooses 1×1 vs N×N path.
   │
   ├─ 0x3d7694  call 0x3d7920  ─────► descriptor builder
   │                                    fills 12-field param struct at r16
   │                                    (offsets 0x10..0x3c)
   │
   └─ 0x3d778c  call 0x2ea740  ─────► hmx_convbbb1x1_stride1
                args r0..r4 = (act_ptrs?, wt_struct?, descriptor, ?, ?)
```

### Descriptor (param struct at r16) populated by 0x3d7920

| offset | source                                         | likely meaning                         |
|--------|------------------------------------------------|----------------------------------------|
| 0x10   | `r14` = act->dim_x_pad << 7                    | activation x-padding                    |
| 0x14   | `r5`  = act->Cin >> 5                           | num input-channel tiles (Cin/32)        |
| 0x18   | `r26` = (Cin/32) × (in_h/8)                     | total in tiles (used as outer-K loop)   |
| 0x1c   | `r9`  = bit-packed (act stride/format bits)    | mxmem `:cm` activation mode bits        |
| 0x20   | `r7`  = bit-packed                              | activation `Rt_act` mask                |
| 0x24   | `r4`  = (out_h+0x1f) & ~0x1f                    | out_h aligned to 32                     |
| 0x28   | `r5`  = wt->dim2 (memw(r24+0xc))                | weight stride                           |
| 0x2c   | `r6`  = lsr(wt_dim,3)                           | wt tiles                                |
| 0x30   | `r27` = mpyi(wt tiles, …)                       | total wt tile count                     |
| 0x34   | `r11` = memw(r3+0xc)                            | wt base ptr or stride                   |
| 0x38   | `r8`  = computed from in dims                   |                                         |
| 0x3c   | `r12` = memw(r3+0x10)                           |                                         |

**These do not map 1-to-1 onto V8's mmv8 setup.** V8 hard-codes most of
this state in inline-asm constants (`r24 = #0x1c`, `r25 = #0x3ff`,
unrolled K=8) instead of reading it out of a heap descriptor.

### Inside the kernel hmx_convbbb1x1_stride1 (0x2ea740)

Preamble reads from arg0 (act ptr table?) and arg1 (something else):
```
r17 = memw(r0+0x4); r16 = memw(r0+0x8)
r13 = memw(r0+0x14)        ← outer K loop count
r20 = memw(r0+0xc)
r12 = memw(r0+0x10)
r0  = memw(r0+0x0)
m0  = r17                  ← used for memw(r0++m0) tile-ptr walk
r15 = memw(r1+0x8)
r4  = memw(r1+0x4)
```

Inner loop already RE'd in `Agent/qnn_vs_v8_root_cause_2026-04-24.md`:
2-MAC unroll, `:after:cm:sat.ub` direct write, Rt_wt = 0x3FF,
Rt_act = r7 | 0x1C. Same instruction sequence we use. The objdump
emits `<unknown>` for the actual HMX MAC bundles since llvm-19 doesn't
decode them, but byte-level we already byte-replicated this.

## What this isolation actually tells us — corrected

Per-op breakdown at 256³ (warm 3rd inference, qnn-profile-viewer):

| | Native (cyc) | V8 (cyc) | ratio |
|---|---:|---:|---|
| Input/IO         | 3,793  | 4,280  | parity |
| pack_act         | (in MatMul_0) | 17,049 | parity vs Native ForceFormat_Crouton 17,909 cyc |
| **MatMul payload** | **9,734**  | **34,152** mmv8 | **3.5×** |
| pack_wt + tcm2ddr| (in MatMul_0) | 6,051  | — |
| Output           | 5,145  | 2,718  | — |
| **Total**        | **18,672** | **67,008** | **3.6×** |

**The gap is in mmv8, not pack_act.**  At 256³ V8's HVX `pack_act_rm`
runs in 17,049 cyc — within 5 % of QNN's `ForceFormat_Crouton` at
17,909 cyc.  All 3.5× of the matmul-payload gap is in the inner MAC
loop.

ConvLayer_s1.opt does 64 output tiles × 8 K-tile-iters = 512 MAC
packets.  Native:  5,964 cyc / 512 ≈ **11.65 cyc/packet**.  V8 mmv8:
34,152 cyc / 512 ≈ **66.7 cyc/packet**.  The 55 cyc/packet difference
matches the previously-isolated **~58 cyc act-address-change penalty**
(`Agent/v8_perf_gap_isolated_2026-04-26.md`).  Native somehow avoids
this penalty.

## What we don't yet have

The wrapper at 0x3d7620 and the descriptor builder at 0x3d7920 build
**multiple stacked structures** that the kernel reads:

- Kernel reads `memd(r4+0x0)`, `memd(r4+0x8)`, `memw(r4+0x18)` (and
  `memw(r4+0x30)` in the unaligned/sparsity variant).  So r4 is the
  HMX-state descriptor — only ~20 bytes, not the 48-byte block we
  thought.
- `r4 = r20` at the call site, but `r20 = r29+0x60` — that stack region
  is **not visibly written** by the disassembly slice we have. It is
  filled by a virtual call `callr r2` at 0x3d7b1c — likely a
  tensor-vtable method that knows how to lay out HMX descriptors for
  the specific tensor encoding (Crouton, padding, scale).  The
  descriptor builder's r16 writes at r29+0x28..0x54 (offsets 0x10..0x3c)
  feed *into* that vtable call but are not the kernel's r4.
- Kernel also reads `memw(r0+0x0..0x14)` and `memw(r1+0x0..0x8)`. r0 = act
  tensor obj, r1 = wt tensor obj.  Each of these has its own ~6-field
  descriptor layout we have not decoded.

So three semi-independent structures gate the kernel: act tensor
descriptor, wt tensor descriptor, HMX state descriptor.  V8's mmv8
implicitly emits all three via inline-asm constants, but apparently
not in the byte layout that lets HMX skip the act-change penalty.

## Why this isolation does NOT immediately unlock a new lever

12+ experiments documented in `gap_closing_attempts_2026-04-26_session_summary.md`
have all returned 0 % improvement.  The 256³ data does not contradict
any of them; it just localizes the residual gap to the kernel's
per-packet cost, which is exactly the bucket those experiments
targeted.

## Resolution — 2026-04-26: act_rt fix

Discovery via path B (hybrid): dlsym `set_hmx_params_conv1x1` from V9
op-pkg, dump the 64-byte descriptor it produces for various args.
Probe code: `HmxMatMulV9SkelOp.cpp` `V9_DUMP_HMX_PARAMS` block.

Results for `set_hmx_params_conv1x1(desc, 0x700, 0, 0, 32, 0)`:

| Field           | V8 hand-crafted | Native (probed) |
|-----------------|----------------:|----------------:|
| 0x00 out_check  | 0               | 0               |
| **0x04 out_rt** | **0x3FF**       | **0x700**       |
| 0x08 act_check  | 0               | 0               |
| **0x0c act_rt_base** | **0x7FF**  | **0x71F**       |
| 0x18 alt_rt     | 0x3FF           | 0x3FF           |

The kernel uses `r24 = or(r7, #0x1c)` for the activation MAC Rt mask.
With `r7 = 0x71F` → `r24 = 0x71F` (0x1c bits already set).
With `r7 = 0x7FF` → `r24 = 0x7FF` — extra bits 5,6,7 set.

Bits 5-7 in HMX's act Rt encode filter-x-stride for NxN conv. For 1×1
matmul they should be clear. V8 had set them, causing HMX to insert
dead work per packet.

### Verified empirically

| Variant | mmv8 cyc @ 256³ | Output identical to baseline? |
|---------|----------------:|:------------------------------|
| V8 baseline (act 0x7FF, wt 0x3FF) | 34,152 | reference                  |
| **V8 act=0x71F**      | **7,127**  | **100.00 % identical** ✓     |
| V8 wt=0x700           | ~10K      | ✗ wrong output                 |
| V8 act=0x71F, wt=0x700 | ~10K     | ✗ wrong output                 |

Only the **activation** side ports cleanly. The weight rt 0x700 changes
HMX semantics in a way that breaks matmul correctness.

### After the fix (warm 3rd inference)

mmv8 cyc/packet at 256³: **65 → 13.6** (silicon ceiling 7.9; native 11.4).
Wall (no wait) at 256³: 22 µs → **21 µs** (native is 19 µs — essentially matched).

This was the path A/B lever all along: the descriptor builder ships a
slightly different mask shape than what V8 had derived, and that
single field controls per-packet HMX scheduling.
