---
name: V8 mmv8 perf gap fully isolated — `:cm` act-stride penalty 2026-04-26
description: 8-variant probe sweep on V8 mmv8 at 2048³. Definitive finding — 240K of 277K cyc/inst comes from `:cm` activation MAC packet stalling when activation address changes between consecutive packets. Per-change cost ~58 cyc, independent of unique-address count or change frequency. Sat.ub drain, weight stride, outer loop overhead all cost ~0. Pure MAC issue rate when act address fixed: 8.97 cyc/packet (within noise of HMX silicon ceiling 7.89). Real wall-time gap V8/QNN is 4.5× at 512³ → 14.9× at 2048³ → 5.7× at 4096³ (NOT 30× — earlier mistake from comparing different cycle metrics).
type: project
---

## Wall-time ground truth (added 2026-04-26 correction)

| shape | V8 wall µs | QNN wall µs | ratio | V8 freq | QNN freq |
|------:|-----------:|------------:|------:|--------:|---------:|
| 512³  | 193 | 43 | **4.5×** | 2.31 GHz | 2.98 GHz |
| 1024³ | 1,292 | 114 | **11.3×** | 2.30 GHz | 4.38 GHz* |
| 2048³ | 9,764 | 654 | **14.9×** | 2.11 GHz | 2.68 GHz |
| 4096³ | 79,706 | 14,101 | **5.7×** | 2.00 GHz | 2.33 GHz |

\* QNN's 1024³ effective freq looks suspiciously high — likely a measurement artifact in `Accelerator (execute) time (cycles)`, not actually 4.4 GHz silicon. Wall µs is still ground truth.

Source: `qnn-profile-viewer` "Accelerator (execute) time" output, mean of 2nd+3rd inferences (skipping warmup).

## Cycle-level analysis (within V8 only — comparisons valid only when same metric)

# V8 mmv8 perf gap — fully isolated 2026-04-26

## Setup

V8 mmv8 inner loop is byte-identical to `hmx_convbbb1x1_stride1` (verified in `Agent/v8_mmv8_byte_identical_to_convlayer_2026-04-26.md`). All probes done at 2048³, M_TILE=N_TILE=256, M_t=N_t=8, K_t=64.

Per V8 mmv8 instance: 64 (mt, nt) iters × 64 K-iters × 2 MAC packets = 4096 MAC packets.

## Probe results

Compile flag passes via `EXTRA_DEFS="-DV8_PROBE_<X>"` in `build.sh`.

| Variant | mmv8 cyc/inst | cyc/packet | What it measured |
|---------|--------------:|-----------:|------------------|
| Original (full V8 path) | **277,032** | **67.6** | baseline |
| `V8_PROBE_NO_SATUB`     | 278,696 | 68.0 | sat.ub drain costs **~0** |
| `V8_PROBE_NO_MAC`       | 3,899 | — | only setup + 64 sat.ubs ≈ 60 cyc/(mt,nt) |
| `V8_PROBE_SAME_ADDR`    | **36,780** | **8.97** | both act+wt fixed → near silicon ceiling |
| `V8_PROBE_ACT_FIXED`    | 37,005 | 9.03 | act fixed, wt strides → wt stride costs **0** |
| `V8_PROBE_WT_FIXED`     | 278,523 | 68.0 | wt fixed, act strides → **act stride = full 240K penalty** |
| `V8_PROBE_ACT_4WAY`     | 278,189 | 67.9 | act cycles 4 unique addrs → not cache size dependent |
| `V8_PROBE_ACT_PAIR_SAME`| 278,119 | 67.9 | same act in pair, different between pairs → not change frequency |
| `V8_PROBE_RT_ACT_3FC`   | 278,229 | 67.9 | smaller Rt_act → Rt isn't the lever |
| `V8_PROBE_ACT_4K_STRIDE`| 278,155 | 67.9 | act tiles 4KB stride (different banks) → **NOT VTCM banking** |

## Definitive root cause

**The `:cm` activation MAC packet pays a ~58 cyc penalty every time the activation VTCM address differs from the previous packet's.**

Properties of the penalty:
- **Per-change**, not per-unique-address. Cycling 4 addrs vs 64 — both pay full penalty.
- **Frequency-invariant**. Changing every-packet (PROBE_WT_FIXED) and changing every-other-packet (PROBE_ACT_PAIR_SAME) cost the same.
- **Survives all asm-level changes**: 3-packet body, hardware loop0, mxmem2 bias, sat.ub Rt=0x3FF, pre-baked ptr table, smaller Rt_act — none of them help.

The HMX silicon ceiling for cache-warm `:cm` MAC packets is 8.03 cyc/packet (per `Agent/qnn_hmx_pipelining.md` P4 probe). V8 hits 8.97 cyc/packet when act is fixed — within 12% of ceiling. So the inner kernel is fundamentally optimal; the gap to QNN comes entirely from somewhere V8 doesn't have visibility/control.

## Per-MAC budget at 2048³

Modeling V8's 4096 packets per instance:
- Steady-state MAC: 4096 × 8.97 = 36.7K cyc ✓ (matches SAME_ADDR)
- Drain + setup: 3.9K cyc ✓ (matches NO_MAC)
- **Penalty for 4096 act-addr changes (one per MAC): ~240K cyc**

Per change cost: 240K / 4096 ≈ 58 cyc. Penalty is **substantially larger than the MAC packet itself** (8.97 cyc).

## Why doesn't QNN's ConvLayer pay this?

QNN's wall-time ground truth at 2048³ = 654 µs, V8 = 9764 µs. **Real ratio = 14.9×**. (Previous "30×" claim was from a metric-mismatch error: chrometrace JSON's per-op `cycles` field is HMX-active-only while V8's profile.txt cycles include stall — see corrected wall-time table at top.)

QNN's act addresses also change between MACs (`r6 = memw(r1++#8)` in inner loop), yet QNN's wall is much smaller. Something about QNN's setup avoids the per-act-change stall.

The 4K-stride probe **rules out VTCM bank conflict** as the cause. With 4KB-spaced addresses (which guarantee different VTCM banks), V8 still pays the full 240K cyc penalty.

Remaining viable hypotheses (both require runtime probing of QNN to verify):

1. **HMX has an internal "act-stream tracker" that requires the address sequence to match a descriptor-defined access pattern.** When V8 changes addr arbitrarily, the tracker resets/stalls; QNN's pre-baked `act_ptr_pairs` follows a specific pattern HMX recognizes (e.g., monotonic, particular alignment). The `:cm` mode ("continuous") strongly suggests stream tracking.
2. **`hmx_convbbb1x1_stride1` sets HMX state via a register/CSR** (not exposed in our naive `mxmem` calls) before the inner loop — for example, an HMX prefetch-base register that pre-fetches subsequent act tiles into an HMX-internal cache. This would be visible only by reading the kernel's prologue carefully (we read it but the prologue does scalar setup; HMX state is touched only via `mxclracc`).
3. **The HMX MAC has dual accumulators (per `Agent/qnn_hmx_pipelining.md` §followup) and `hmx_convbbb1x1_stride1` uses `mxswapacc` between act-stream segments**, hiding the stall by accumulating into a SECOND acc while the first is being addr-stalled. The disasm at 0x2ea820 doesn't show `mxswapacc`, but maybe the prologue or other paths do.

## What WOULD beat the penalty (now narrower)

A. **Successfully call `hmx_convbbb1x1_stride1` via dlsym with correct descriptor values** — would automatically get QNN's HMX setup and avoid the penalty. Requires runtime probing of a real ConvLayer call to capture descriptor values.
B. **Re-read the FULL 492 B of `hmx_convbbb1x1_stride1`** including the prologue (0x2ea7xx range we partially RE'd) and the `_unaligned` fallback at 0x2eb180. Look for any HMX register write or `:retain` pattern not yet documented.
C. **Build an instrumentation op-pkg** that hooks before QNN's ConvLayer call and dumps the 3 descriptor structs. Then we know the values to plug into our V9 dlsym path.

VTCM banking, Rt mask tuning, MAC body byte-replication — all proven NOT the lever.

## Concrete next experiment

Path C is the highest-leverage: dump real descriptor values from a running QNN ConvLayer. Implementation:
1. Build a custom op-pkg that wraps QNN's `q::ConvLayer_s1.opt` op (intercept at QHPI registration).
2. In the wrapper, before calling through to ConvLayer's real impl, copy r0/r1/r4 (the 3 descriptor pointers) into a known location.
3. Run a tiny QNN MatMul (e.g. 64×128×64) with the wrapped op-pkg active.
4. Read back the captured descriptor bytes; match field interpretations.

Alternatively path B (more disasm) is cheaper but yields less than path C.

## Files preserved 2026-04-26 (probe variants under compile flags)

- `example/hmx_matmul_phase3/src/HmxMatMulV8Op.cpp` — branches gated by `V8_PROBE_NO_SATUB`, `V8_PROBE_NO_MAC`, `V8_PROBE_SAME_ADDR`, `V8_PROBE_ACT_FIXED`, `V8_PROBE_WT_FIXED`, `V8_PROBE_ACT_4WAY`, `V8_PROBE_ACT_PAIR_SAME`, `V8_PROBE_RT_ACT_3FC`. Default build (no flag) = production V8.
- `example/hmx_matmul_phase3/build.sh` — `EXTRA_DEFS` env injection.

To re-run a probe: `EXTRA_DEFS="-DV8_PROBE_SAME_ADDR" bash build.sh`.
