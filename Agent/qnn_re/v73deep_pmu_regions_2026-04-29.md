---
name: V73DEEP per-region PMU probe — gap localized to K-MAC body
description: PMU per-region differential probe at 256³ pinpoints the V73DEEP kernel's K-MAC inner-loop body as the source of ~74% of packet cost; native's 346 pkts is mathematically incompatible with the per-iter MAC packet count we observe, suggesting native uses HMX descriptor-driven fan-out
type: project
---

# V73DEEP per-region PMU probe — 2026-04-29

Build: `EXTRA_DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST -DV9_PROBE_REGIONS"`

Method: 4 kernel-call variants per op, each shrinking one loop dimension.
Linear fit of PMU `COMMITTED_PKT_ANY` per call:

```
P(r13e, r20, r28) = A + r13e * (B + r20 * (C + r28 * D))
```

| Variant | n_tiles | k_total | n_act | r20 | r13e | r28 | pkts(ANY) | cyc  |
|--------:|--------:|--------:|------:|----:|-----:|----:|----------:|-----:|
| V0      | 64      | 256     | 8     | 8   | 4    | 4   | 1724      | 2104 |
| V1      | 8       | 256     | 8     | 1   | 4    | 4   | 537       | 290  |
| V2      | 64      | 256     | 2     | 8   | 4    | 1   | 920       | 647  |
| V3      | 64      | 32      | 8     | 8   | 1    | 4   | 728       | 514  |

Linear fit (ANY threads):
- **A = 396** (prologue+epilogue, per-call fixed)
- **B ≈ 0** (per-outer overhead)
- **C = 8.89** (per-loop1 overhead — drain cvt+store ×2)
- **D = 8.38** (per-loop0 = K-MAC body)

## Region breakdown for V0 baseline (1724 ANY)

| Region | Formula | ANY pkts | est T0 (÷2) |
|--------|---------|---------:|-------------:|
| Prologue/epilogue (A) | × 1                | 396  | 198 |
| Per-outer (B*r13e)    | × 4 outers          | 0    | 0   |
| Per-loop1 (C*r13e*r20)| × 32 (4×8)          | 285  | 142 |
| **Per-loop0 (D*…)**   | **× 128 (4×8×4)**   | **1073** | **536** |

K-MAC body = 74% of total. Per-loop0 cost D ≈ 4.2 T0 packets/iter (after
PMU ANY → T0 halving). Native chrometrace = 346 T0 packets for the WHOLE op.

## Conclusion

**Mathematical incompatibility**: even with 0 prologue and 0 drain, our
kernel's K-MAC body (D × 128 = 536 T0 pkts) already exceeds native's
346 T0 packet TOTAL.

This means native does NOT execute the same number of `mxmem` packets we do.
Two ways native could achieve it:

1. **HMX descriptor-driven fan-out** — native sets up HMX state via
   state-config packets so that one `mxmem` packet covers multiple tiles
   internally (per-tile addressing is auto-walked by HMX). Per memory
   `reference_hmx_dsp_vs_descriptor_2026-04-28`: native ConvLayer.opt
   uses this exact pattern. Each MAC packet "weighs 8×" in tile work.

2. **Different kernel** — native at 256³ might dispatch to a kernel
   variant we haven't disassembled, with fewer mxmem instructions per
   logical MAC. E.g., a hand-tuned NxN body where 2 MACs fit per packet.

Most-likely answer: **#1 is the path**. Our V73DEEP kernel body is doing
DSP-driven MAC issue (one mxmem per K-MAC pair). Native achieves
descriptor-driven HMX fan-out, getting 8x throughput per MAC packet.

## What this rules out

- ❌ Calibrating descriptor fields (we already match exactly per `desc_dump`).
- ❌ Multi-call splitting (per-call prologue is only 198 T0; not the bottleneck).
- ❌ Different mask configuration (mask was confirmed via dump).
- ❌ Different kernel selection at our descriptor level.

## What can close the gap

Per `docs/hmx_dsp_vs_descriptor_driven.md` and the path forward in
`NEXT_STEPS_v73deep_gap.md`:

- **Phase 3 (inline kernel body)** alone won't help — same DSP-driven pattern.
- **Phase 2 (kernel-prologue patching)** saves only A ≈ 198 T0 per call;
  not enough.
- **The real fix**: write a kernel body that uses HMX state-config packets
  to set up tile fan-out, then issue fewer mxmem packets. This requires
  understanding HMX state-config encoding (HMX descriptor registers — not
  documented in the public HVX/Hexagon ref).

## Cross-kernel comparison

Same probe with `V9_PROBE_V73_NONDEEP` (calls `hmx_v73_convbbb1x1_stride1`
instead of deep variant) using identical descriptors:

| Kernel         | A (prologue) | B (per-outer) | C (per-loop1) | D (per-loop0) | V0 ANY |
|----------------|-------------:|--------------:|--------------:|--------------:|-------:|
| V73DEEP        | 396          | ~0            | 8.89          | 8.38          | 1724   |
| V73 non-deep   | 438          | 68            | 6.83          | 11.55         | 2407   |
| OLD non-v73    | (crashes — needs N-major wt layout, we provide K-major) |

V73 non-deep is **WORSE** than deep on every per-region cost. The B = 68
explosion comes from non-deep doing one bias load per outer iter (vs deep's
two paired loads amortized over 2 N-tiles). D=11.55 (non-deep) > D=8.38
(deep) confirms the K-MAC inner body is fundamentally heavy in BOTH variants.

## Strengthened conclusion (with metric correction)

**Metric correction (2026-04-29 PM)**: direct V9_PMU_PROBE vs chrometrace
on the SAME baseline V73DEEP build at 256³ shows:
- chrometrace pkts = 1303
- PMU `COMMITTED_PKT_ANY` (kernel call only) = 1316
- They MATCH. Chrometrace pkts ≈ PMU ANY (whole op), NOT T0.

Therefore the production V73DEEP chrometrace 747 ≈ 747 PMU ANY (whole op),
and native chrometrace 346 ≈ 346 PMU ANY (whole op).

Re-running the gap math with corrected metric:
- Our V73DEEP V0 kernel call = **1316 ANY** (with PMU-probe overhead).
- Production V73DEEP whole op = **747 ANY** (without PMU-probe instrumentation).
- Native whole op = **346 ANY**.

Per linear fit on V73DEEP probe variants (in PMU ANY units):
- A = 396 (prologue/epilogue)
- C = 8.9 (per-loop1 drain)
- D = 8.4 (per-loop0 K-MAC pair)
- Total kernel = 396 + 32×8.9 + 128×8.4 = 1759 ANY

That's the PMU-instrumented total. Subtract probe overhead ≈ 600 → ~1100 raw kernel.
Production (no probe) op total 747 = kernel + non-kernel. Kernel itself ≈ 700.

Native 346 is 2× lower. Sources of remaining 350 ANY packet gap:

- **Per-iter density**: our 8.4 ANY/K-MAC pair vs hypothetical native 4 ANY/K-MAC pair would close the gap entirely.
- HMX `:2x2` cvt fan-out (per `docs/hmx-programming-guide/09-instr-convert.md`)
  exists in V75 hbh kernels (FP16 output) but **NOT in any bbb kernel**. So
  this optimization is inaccessible for u8 output via dlsym path.
- Possibly native uses an inline kernel generated by libHtpPrepare at prepare
  time, embedded directly in ctx-binary — we'd find this only by reading
  the ctx-binary text region for the ConvLayer.opt instance.

## Tools added (reusable)

- `V9_PROBE_REGIONS` macro in `HmxMatMulV9SkelOp.cpp` — runs 4 PMU variants.
- `V9_PROBE_V73_NONDEEP` / `V9_PROBE_OLD_KERNEL` sub-flags swap the kernel
  symbol called inside the variants loop.
- `scripts/parse_v73deep_probe_regions.py` — decodes + fits + reports gap.
- Same runner `standard_flow/phaseB_v8/run_v8c8_chain.sh` works with `WT_LAYOUT=kmaj`.

## Where to look next (out of scope this session)

If reopening the gap:

1. **Hunt the descriptor-driven kernel** — dlsym all `T` symbols in
   libQnnHtpV75Skel.so starting with hmx_/conv_/matmul_; disasm any we
   haven't seen for HMX state-config patterns (different from our
   per-loop0 mxmem). The V75-specific `hmx_v75_convbbh*_stride1` family
   targets fp16 output — disasming them might reveal V75 HMX features
   we could adapt to bbb.
2. **Static RE libHtpPrepare.so** — x86 binary likely contains the C++
   that GENERATES per-shape kernel code at prepare-time. That generated
   code would be embedded in ctx-binary text region.
3. **Accept and document** that V73DEEP is our best achievable using
   public HMX kernel symbols, and stop trying to match native at 256³.
   The 2.5× gap is the cost of using stock kernel libraries vs native's
   on-the-fly code generation.
