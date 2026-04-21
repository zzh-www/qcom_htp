# QNN HMX pipelining RE — findings (2026-04-22)

Device: SM8650 v75 (ssh oneplus). Probe:
`example/hmx_matmul_device/probe_pipeline_device.c` +
`run_pipeline_probe.sh`.

## Source of hypotheses

Disassembly of `libQnnHtpV75Skel.so::hmx_convbbb1x1_stride1 @ 0x2ea740`
(the 1×1 conv = MatMul u8·i8 hot kernel) using
`hexagon-llvm-objdump -d --mattr=+hmxv75,+hvxv75,+hvx-length128b`.

Hot loop body (inner, `loop0(0x2ea820, r5)`):

```asm
; prologue (executed once per outer iter):
{ r24 = or(r7, #0x1c)        ; Rt_act = r7 | 0x1C
  r25 = #0x3ff }             ; Rt_wt  = 0x3FF

; loop0 body (2 HMX MAC packets per iter):
{ p0 = cmp.eq(r26, #0x2); r26 = add(r26, #-0x2)
  r6  = memw(r1++#0x8)       ; fetch act ptr 0
  r23 = memw(r1+#0x4) }      ; fetch act ptr 1
{ r8 = add(r8, #0x400)       ; weight ptr += 1024
  if (p0) r25:24 = combine(r9, r7)
  activation.ub = mxmem(r6, r24):cm
  weight.b      = mxmem(r8, r25) }
{ r8 += add(r25, #0x1)       ; weight ptr += r25+1 = 0x400
  activation.ub = mxmem(r23, r24):cm
  weight.b      = mxmem(r8, r25) } :endloop0
```

Both packets use `:cm` on activation, plain weight.b, with Rt values that
prior RE mis-identified (tried `:dilate`, `:above`, `mxswapacc` — those
belong to larger-filter Conv2D kernels, not 1×1 MatMul).

## Probe results (SM8650 v75, A=1 W=1, N=400 MACs accumulated, single readback)

| ID | pattern                                 | Rt_act       | Rt_wt  | out[0,0] | cyc/MAC |
|----|-----------------------------------------|--------------|--------|---------:|--------:|
| P1 | plain baseline                          | 2047         | 2047   |  32      | **19.68** |
| P2 | plain                                   | 2047         | 0x3FF  |  32      | **7.89**  |
| P3 | `:cm` on activation                     | 2063 (`\|0x1c`) | 2047 |  16      | 19.64   |
| P4 | `:cm` (QNN pattern)                     | 2063         | 0x3FF  |  16      | **8.03**  |
| P5 | `:cm` PAIR — 2 MACs/iter, 2 act ptrs    | 2063         | 0x3FF  |  —       | 7.98    |
| P6 | `:cm` alone (no `\|0x1c`)               | 2047         | 0x3FF  |  —       | 7.94    |

## What we learned

### 1. `Rt_wt = 0x3FF` is the 2.5× HMX pipelining unlock (not `:cm`)

Changing only the weight Rt from 2047 to 0x3FF drops cyc/MAC from **19.68
to 7.89** — a **2.5× speedup** with no code structure change, identical
functional output, and no modifier involvement.

**Mechanism hypothesis**: Rt is a byte-mask / stride hint for the HMX
mxmem load unit. 2047 (`0x7FF`) is the 2 KiB activation-tile mask; 0x3FF
is the 1 KiB weight-tile mask. Using a mask that matches the real
operand's footprint lets the HMX unit schedule VTCM fetches without
over-speculating into the next tile's bank. Using an over-sized mask
(2047 on a 1 KiB weight) likely triggers bank-conflict stalls on the
subsequent MAC's weight load.

Our prior iter-10 Rt sweep missed this because it only tested
combinations with `2047` or `32767` on weight — never the tight 0x3FF.

### 2. `:cm` is a SEMANTIC modifier, not a perf modifier

`:cm` halves the output: P1 (plain) → 32, P3 (`:cm` same Rts) → 16.
Likely interpretation: "convolution mode" consumes only ONE of the two
streams (s0 OR s1) in the activation tile's dual-stream layout, giving
half the per-logical-row accumulation. For 1×1 conv, where 16 phys_rows
× 2 streams = 32 logical rows, this matters for correctness but not for
MAC throughput.

At our Rt_wt=0x3FF: P2 (plain) = 7.89, P4 (`:cm`) = 8.03 — **within
noise**. `:cm` is a tile-geometry switch; whatever pipelining benefit
exists from it (if any) is independent of the 2.5× Rt_wt win.

### 3. Pair-mode (2 different act ptrs per iter) doesn't help

P5 runs 2 MAC packets per loop iter with activation ptrs alternating —
same structure as QNN's hot loop. Cycles/MAC = 7.98 = same as plain
single-ptr (P4, 8.03). The 2-MAC structure in QNN is there to reuse the
weight tile across 2 output rows (amortizing weight load); it is **not**
a pipelining mechanism on its own.

### 4. Prior RE mistakes corrected

| prior claim | correct finding |
|---|---|
| "Built-in achieves ~16 cyc/packet via `:dilate` + `mxswapacc` pipelining" | The 1×1 conv kernel uses neither. It's plain activation.ub + weight.b with Rt_wt=0x3FF. `:dilate`/`:above`/`mxswapacc` are Conv2D-specific (5×5 / 3×3 filter kernels). |
| ":cm is probably the streaming-mode unlock" | `:cm` is a single-stream semantic modifier. Doesn't affect cycles. |
| "~1 cyc/MAC is the HMX target" | At the PACKET level, measured ceiling is **7.89 cyc/packet** on v75. One packet = 1 MAC packet doing 32·32·32 = 32 K MACs. Per-MAC = 2.4e-4, which is competitive with QNN w8a8's per-MAC of ~4.9e-4. **We're already at QNN-packet-level throughput when Rt is right.** |

## Revised gap analysis

Built-in QNN w8a8 at 512³ = 66,513 cycles, 128 × 128 × 128 = **2048 packets
worth** of MACs (4096 32³ tiles × 0.5 tiles-per-packet pair structure
actually... let me just compute the implied cyc/packet for QNN):

- w8a8 512³: 66 K cycles ÷ 4096 packets = **16 cyc/packet** (QNN)
- our w4a16 512³: 1,118 M cycles ÷ 8192 packets (2 MAC passes × 4096) = **~136,000 cyc/packet**

So QNN runs at 16 cyc/packet, we run at 136K cyc/packet. Our probe
shows the HMX unit itself can do 7.9 cyc/packet with Rt_wt=0x3FF.
Therefore our **scalar overhead** adds ~136,000 cyc/packet while QNN
adds ~8 cyc/packet. That's the actual gap: not HMX, not pipelining —
**scalar pack/unpack/decompose/combine around each MAC**.

Implication for Phase 3: the 500× gap is **almost entirely HVX-able**
scalar overhead. Drop in Rt_wt=0x3FF first (2.5× win on the HMX loop
itself), then HVX-vectorize pack/unpack/decompose/combine.

## Action items

1. **Immediate (drop-in)**: Set `Rt_wt = 0x3FF` in
   `example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c` (all `hmx_load_pair_*`
   variants). Expected: 2.12 → ~0.85 cyc/MAC at 512³ (2.5× scalar remains
   the dominant term; this only speeds up the HMX-MAC portion).
2. **Phase 3**: HVX-vectorize `pack_activation_32x32_rs`,
   `pack_weight_32x32`, the activation hi/lo decompose, and the combine
   loop. Each uses `vshuff` (not `vshuffe_b`).
3. **Phase 2A (w16a16)**: same Rt_wt=0x3FF drop-in applies to the 4 MAC
   packets per tile in `int16_matmul_hmx.c`.
4. **Phase 4 (w4a8)**: start fresh with Rt_wt=0x3FF baked in.

## Methodology note for future RE

The two prior errors were:
- Conflating 1×1 conv with larger-filter conv's instruction patterns.
- Testing pattern variants holistically ("add :above + mxswapacc") rather
  than one knob at a time.

Fix for both: always isolate ONE knob, measure with a tight-loop probe
on silicon, and record functional output — semantic changes hide in the
output data, not the cycle count.

## Followup — `:above` and `mxswapacc` semantics (probe_dualacc_device.c)

| test pattern | result | conclusion |
|---|---|---|
| MAC_above (single) | out = 32 | `:above` writes **current** acc, NOT "other" |
| MAC_plain, store_current (no retain), swap, store_other | out2 = 0 | store without `:retain` clears **BOTH** accs |
| MAC → A; swap; MAC → B; store B :retain; swap; store A | A=32 B=32 | `:retain` preserves both accs ✓; mxswapacc truly swaps |
| above + plain + swap + store | 0 | pattern ineffective because store w/o retain wipes state |

**Corrected model**:
- Two accumulators A, B. `mxswapacc` swaps which is "current".
- HMX MAC always writes to "current"; `:above` is no-op for accumulator routing.
- `store :after.uh` — destructive to both accs (implicit clear on store).
- `store :after:retain.uh` — preserves both accs.
- Dual-scale readback idiom: load_bias_lo; store_retain; load_bias_hi;
  store_retain; swap; load_bias_lo; store_retain; load_bias_hi; store (no retain).

**Result for dualacc kernel**: correctness fixed (was broken at K ≥ 64 due to
missing `:retain` on the hi-byte store of acc A — clearing B before we
could read it). Perf: 2.12 → 2.47 (plain dualacc, worse) → 2.08 (fused hi+lo
dualacc using two accs to hold the two partials instead of two K-passes).

Double-buffering weight tiles yields **no speedup** — HMX pipelining is
NOT limited by VTCM write→read latency on a single tile address; scalar
work in the inner loop already overlaps with HMX MAC issue at the
pipelines' granularity.

## What's NOT the bottleneck at 2.08 cyc/MAC (512³)

Hypotheses systematically ruled out:
- ❌ HMX MAC issue rate (ceiling = 7.89 cyc/packet = 2.4e-4 cyc/MAC)
- ❌ VTCM bank conflict on wt_tile (double-buffer test: 0 improvement)
- ❌ Accumulator data-dep chain (dualacc: worse; fused dualacc: only 2%)
- ❌ `:above` / `:dilate` / `:cm` pipelining unlock (no modifier changes
      throughput at kernel scope)

## What IS the bottleneck

At 2.08 cyc/MAC × 131 M MACs = **272 M cycles** for 512³. Per (m_tile,
n_tile) call: 272 M / 256 = **1.06 M cycles**. For K=512 this means
**65 K cycles per K-tile pair** (1 weight pack + 2 MACs + 2 swaps).

Breakdown (via `pack_weight_32x32 → stub` ablation):
- `pack_weight_32x32`: 22 M cycles / 256 calls / 16 k-iters = 5.4 K cyc/call. **8% of total.**
- Everything else: 92%.

The "everything else" is dominated by **DDR bandwidth for `gather_w_col`
and `w_col` reads inside pack_weight** — we read 512 × 32 = 16 K bytes per
`gather_w_col` call, 16 K × 256 = 4 M byte-reads from DDR per inference.
At ~80 cyc per cache line (16 bytes) cold, that's 20 M cycles just for
w_col DDR traffic, plus similar for the activation.

**Next levers (ranked by expected impact)**:
1. Hoist `gather_w_col` out of the m loop — gather runs once per n_tile,
   not per (m, n) pair. Saves 15/16 = 94% of gather cost = ~10% overall.
2. Use VTCM for w_col staging (read once per n_tile, reused across m). This
   has been tried and regressed 2.7× — but only under the OLD MAC hot-loop.
   Retest with fused dualacc.
3. HVX-vectorize `pack_weight_32x32` — 8% → ~1.5%. Modest overall impact
   but important for w4a8 where weight pack is shared code.
4. Graph-level 2×2 tiling (4 threads) — 4× parallelism if scalar hot paths
   are replicated across HVX threads.

With all 4 stacked, realistic target: 0.05–0.10 cyc/MAC at 512³, closing
90%+ of the gap to QNN w8a16 (8.1e-4 cyc/MAC). The last 10× requires
QNN's graph-level parallelism (4-way ConvLayer concurrency).

## Methodology note

The productive approach turned out to be:
1. **Disassemble** the real hot loop with `hexagon-llvm-objdump -d --mattr=+hmxv75,+hvxv75,+hvx-length128b`.
2. **Isolate each knob** with a tight-loop silicon probe (200-400 MACs accumulated).
3. **Compare to a baseline** functionally first (correctness), then by cycle count.
4. **Ablate** kernel regions to quantify relative cost. This is what revealed
   pack_weight is only 8% — without ablation the prior RE work assumed
   scalar pack was the whole story.
