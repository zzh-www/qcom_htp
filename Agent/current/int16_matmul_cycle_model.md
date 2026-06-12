# int16 (w16a16) 64³ MatMul — definitive cycle model on v75 HTP

Authoritative cost model for an int16×int16 64³ matmul on the v75 HTP, measured from QNN-native
optrace. Many GDN-inverse decisions key off "how much does one int16 64³ matmul cost" — this pins it,
**separating the two numbers that were previously conflated and read as misleading.**

## 口径 — 纯 kernel ≠ feed-inclusive (full rules → skill `htp-cycle-metric`)

**The general cycle口径 — the 4 metrics, latency-vs-throughput, the 4 traps, the report template, and the
concept timelines — live in skill `htp-cycle-metric` (read it first).** This doc only applies them to int16.

int16 64³, device-measured, native ConvLayer, **dominant-path == PCYCLE** (per-64³ = (C/64)³-normalized):
- **纯 kernel latency = 256** (warm; → 128/64³-MAC @256³). The 4 byte-passes + drain are INSIDE the one
  `q::ConvLayer_s1.opt` op (it writes the final int16 output) — this is "the matmul itself."
- **per-call feed-inclusive = 1365 / 3867 / 8.6K–10.4K** = kernel + feed-not-hidden. You pay the feed (not 256)
  **iff the matmuls are a dependency chain** (can't pre-load the next) — that's why the inverse's int16 matmuls
  cost 8.6K, not 256.
- **Never quote "1167 / 6×" as the kernel cost** — that's the *throughput* of the custom 2×w8a16 op
  (`HmxU16I16ToU16MatMul`), not the native single-convhhh *latency* (256 = 1.45× u8i8). (skill trap #4.)

**GDN 求逆 = producer-bound (本文核心图，GDN 专属):**
```
HVX producer: ▓▓▓▓▓▓ diag(SMT隐藏)+merge quant/pack ▓▓▓▓▓▓  1.62M = 93% = wall
HMX merge:    ■   ■   ■   ■   ■                            160K = 7% (空闲多)
→ 换 int16 merge: HMX +41K(忽略); 真成本=producer pack 翻倍 +150K ⇒ ≈1.97M, oc 1.16e-3 (精度 2.7×)
```

## The one finding

A native int16 MatMul lowers to a **single** `q::ConvLayer_s1.opt` HMX op (the convhhh / "16-bit kernel",
selected by a CTRL input that also picks 4/8/16-bit; **not** 2× w8a16 — that is a separate software route).
Its Perfetto Description: *"Conv2d strides=(1,1) with an extra CTRL input … or use the 4/8/16bit kernels."*
**The bare kernel is cheap; essentially all cost is weight-prep + activation crouton/slice + format.**

## Cost table (dominant-path, per-64³)

| scenario | per-64³ | compute% | what dominates |
|---|---|---|---|
| **A. bare HMX kernel** (`q::ConvLayer_s1.opt` only) | **128–264** | 100% | 264 @64³ tile → **128 asymptote** @256³ |
| **B. static/REUSED weight, warm**  ← the GDN case | **~1365** | ~16% | weight folds to prepare (`weights_to_vtcm` once=162); QNN fuses 32→4 ConvLayer; **act crouton/slice dominate** |
| **C. dynamic weight, warm** (independent batches) | **~3867** | ~7% | + per-call `convert_weights_to_signed`(709) + bias_update + slice |
| D. single COLD serial 64³ (1 op, no amortize) | ~31700 | <7% | one-time warmup — **not** a chain steady state |
| E. big tile 256³ (dynamic, amortized) | ~580/64³ | ~22% | per-call wrapper (~3600 fixed) amortized over 64× MACs |

Derived facts:
- **Bare kernel floor ≈ 128 cyc/64³-MAC** (256³: 8168 dom / 64 = 128; 128³: 1152/8 = 144; 64³ tile: 264 incl per-op overhead).
- **Per-call wrapper ≈ 3600 cyc, ~fixed** (dynamic weight): convert_weights + crouton + reshape + slice + bias. → dominates at 64³ (93%), amortizes at 256³ (E).
- **Weight reuse removes the heavy weight path**: `convert_weights_to_signed`(6652 cold / 709 warm) + `bias_weight_update` + `bias_scale_shuff` + `Cast` → replaced by a one-time `weights_to_vtcm`(162/64³) + `bias_to_vtcm`(44). B vs C = **2.8× cut** purely from reusing the weight.
- **Cold-start is one-time** (D): first ConvLayer 1970–2725 vs warm 227–340; a long matmul chain pays it once.

## Reconcile with prior project numbers

| project figure | what it actually was | corrected by this model |
|---|---|---|
| "1167 native supertile floor" | custom `HmxU16I16ToU16MatMul` `by_htp_type` aggregate (2×w8a16 impl) | **NOT the kernel floor.** Native single-convhhh = **264/64³ (128 @256³)** dominant-path — lower. 1167 is that custom op's overhead. |
| "5844 @256³ carrier" (our hand) | hand convhhh 256³ **wall (feed-inclusive)** | ~10× above native-256³ full (E, 580/64³); the gap is exposed feed, not kernel |
| "8.6K–10.4K @64³" (our hand) | hand convhhh 64³ **wall (feed-inclusive)** | kernel is 264; the 8.3–10.1K is feed (dependency chain → not hidden) |

**The bare kernel is byte-identical** between QNN-native and our replica `our_v73deep_kernel_i16`
(= `hmx_v73_convhhh1x1_stride1`). So the 6–30× spread between routes is **entirely feed / weight-prep /
crouton-format scheduling**, not the kernel. QNN's advantage = it (a) folds static-weight prep to prepare-time,
(b) fuses batches into larger ConvLayer ops (8×64³ → 142/64³ compute), (c) pipelines crouton/slice.

## Matmul-kernel implications (valid for any matmul-bound op)

1. **Don't blame the int16 kernel.** A 64³ int16 matmul's HMX compute is ~264 cyc (130 packets @ ~2 cyc/pkt).
   Our hand path's 8.6–10.4K/64³ is ~97% feed/format, not MAC.
2. **Weight reuse is the biggest lever (2.8×)** *for the matmul in isolation*: reuse the same block across RHS
   columns → pre-pack/convert ONCE, per-matmul ≈ ConvLayer + activation feed (~1365/64³ regime).
3. **Bigger tiles amortize the fixed wrapper** (E: 256³ → 580/64³). 64³ recursion granularity is the worst case.
4. **Activation crouton/slice is the residual floor** even with static weight (~880/64³ in B); it is **2× u8i8's**
   because int16 = 2 bytes and is **not** removable by weight reuse. Hiding it under compute (DMA double-buffer)
   is the only remaining lever — and it bottoms at ~264, still above u8i8's 313 only because u8i8 feeds 1 byte.

> ⚠️ These are matmul-kernel facts. They do **not** rescue the GDN inverse — see the Decision below.

## Decision (2026-06-13, CORRECTED): re-open the int16-HMX inverse? **OPEN — the earlier KILL was a metric error.**

> **A first pass KILLed this on 2026-06-13 using "int16 = 6× u8i8 / 1365 floor". That was WRONG — it used the
> THROUGHPUT (HMX-busy) of the 2×w8a16 software decomposition, not the LATENCY of the native single-convhhh
> kernel. Device re-measure (native MatMul, dominant-path, same metric & context for both dtypes):**
>
> | native 64³ ConvLayer | u8i8 | int16 | ratio |
> |---|---|---|---|
> | **latency (dominant-path)** | **176** | **256** | **1.45×** |
> | HMX-op-sum (throughput proxy) | 292 | 371 | 1.27× |
>
> So the native int16 **kernel** is only **1.45× u8i8**, not 6×. The "1167 / 6×" was the custom
> `HmxU16I16ToU16MatMul` (2×w8a16) measured as HMX-busy — a different impl AND a different metric. The 4
> byte-passes **pipeline**, so latency ≪ throughput.

Corrected roofline (latency口径 + the real producer breakdown from `gdn_opt_ledger.md`):
- The inverse is **producer-bound: HMX only 7% busy (160K/1.78M), idle-mostly** (`gdn_solve.md:101`). When HMX is
  idle-mostly, the merge matmul's relevant cost is its **latency** (1.45× u8i8), not throughput. Swapping 512
  merge u8i8(176)→int16(256) adds only **~41K** to the HMX critical path — negligible.
- The deciding cost is the **producer weight-pack**: `gdn_opt_ledger.md:46,50` measure it at **8.4% (~150K), ON the
  wall critical path** (kmajor `vshuff`, NOT SMT-hidden, HW-irreducible). int16 weight = 2 bytes → that pack
  **~doubles → +~150K**. (The diag forward-subst is **SMT-hidden**, `:36` — it does NOT grow the wall.)
- **Corrected roofline ≈ 1.78M + 150K (pack) + 41K (kernel) ≈ ~1.97M (+11%)**, oc ~1.16e-3. NOT 2.32M (my killed
  roofline used 6× throughput — wrong) and NOT a free 1.78M win (the pack really does double).

**This is a precision-Pareto point, and a good one:** ~1.97M / oc 1.16e-3 **beats the existing precision option BP4**
(2.71M / 4.90e-3) on **both** wall and accuracy. vs shipping all-precision u8i8 (1.78M / 3.10e-3) it is +11% wall
for 2.7× better oc.

Why our hand `GDN_BR_W16` measured 4.31M (not 1.97M): the hand int16 kernel ran the byte-passes **un-pipelined**
(serial feed in the call thread) → HMX-busy blew to 3.98M (92%, throughput-bound). The native proves the SAME
byte-identical convhhh runs at **256 latency** when fed pipelined. So **S1's job = make the hand kernel pipeline
like native** (HMX back to idle-mostly) AND confirm the +150K pack estimate. Pure-HMX (diag on HMX) stays
numerically dead (A⁶⁴=0) — viable shape is **diag on HVX + int16 merge on HMX** (shipping, swap merge dtype).

**Verdict: NOT killed — a ~1.97M precision-Pareto candidate (beats BP4). Hinges on S1** = (a) a hand int16 merge
that pipelines to native's 256 latency (un-proven in our pipeline; native is the existence proof), (b) confirm the
producer pack doubles by ~150K not more. Earlier "×1.3–2.5 loses / KILL" was the throughput-confused roofline — discard it.

## Reproduce

```bash
# dynamic-weight, batched (warm steady-state, C): regenerate + run
DSSH_HOST=oneplus  # mux via scripts/dssh.sh
bash example/gdn_native/solve_op/standalone/mm_int16_shapes.sh           # 1-batch 64³ (cold serial, D) + 8x8
H=32 CS=64 bash example/gdn_native/solve_op/standalone/gdn_mm.sh         # 32-batch dynamic (but u8i8; for int16 use mm_64 dir already present)
# static/reused weight (B): one weight × BATCH matmuls
BATCH=32 bash example/gdn_native/solve_op/standalone/mm_staticw.sh
# decode any run dir:
QNN_SDK_ROOT="$PWD/tools/qnn-sdk" .venv/bin/python scripts/decode_qnn_optrace.py <dir>/out
# per-op dominant-path lives in <dir>/out/optrace/chrometrace_qnn_htp_analysis_summary.json -> htp_op_instances[].num_dominant_path_cycles
```

Artifacts: `example/gdn_native/solve_op/standalone/{mm_64,mm_128,mm_256,mm_1x1x64x64,mm_1x8x8x64,mm_staticw_64}/out/optrace/`.
Generators: `scripts/mm_int16_probe.py`, `scripts/mm_int16_staticw_probe.py`. Metric skill: `htp-cycle-metric`.
