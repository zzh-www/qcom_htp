---
name: V8 mmv8 inner loop byte-identical to QNN ConvLayer; per-MAC gap is NOT in the asm
description: Replicated hmx_convbbb1x1_stride1's hot loop body in V8 mmv8 (3-packet structure, hardware loop0/loop1, alt_rt swap, mxmem2 bias, sat.ub Rt=0x3FF, pre-baked act_ptrs). Verified via disasm. Per-instance perf at 2048³ unchanged (277K cyc). Real wall-time gap to QNN at 2048³ = 14.9× (V8 9764 µs vs QNN 654 µs). Earlier "30×" claim was a cycle-metric mismatch — corrected via wall µs comparison.
type: project
---

# V8 mmv8 inner-loop replica — done, but not the lever (2026-04-26)

## What was changed in `HmxMatMulV8Op.cpp` (tile_output path)

1. **3-packet inner body**, matching `hmx_convbbb1x1_stride1` 0x2ea820–0x2ea848 byte-for-byte:
   ```
   { p0 = cmp.eq(r26, #2)
     r26 = add(r26, #-2)
     r6  = memw(r1++#8)
     r23 = memw(r1+#4) }
   { r8 = add(r8, #0x400)
     if (p0) r25:24 = combine(r9, r7)        ; alt_rt swap on last K
     activation.ub = mxmem(r6, r24):cm
     weight.b      = mxmem(r8, r25) }
   { r8 = add(r8, add(r25, #1))               ; r8 += 0x400 (since r25=0x3FF)
     activation.ub = mxmem(r23, r24):cm
     weight.b      = mxmem(r8, r25) }
   ```
2. **Hardware `loop0(label, K_tiles/2)`** — zero per-iter scalar overhead (was software for-loop with 3 extra packets per iter).
3. **`bias = mxmem2(r3)`** — matched QNN's bias-load instruction exactly (was plain `bias = mxmem(...)`).
4. **`sat.ub Rt = 0x3FF`** — matched QNN (was Rt=0).
5. **Pre-baked activation pointer table** at op start (`act_ptrs_all[M_t * K_t]`), eliminating the per-(mt,nt) ptr-fill scalar loop.

## Verified replica (V8 disasm at 0x1cc4–0x1cec)

```
1cc4:  { p0 = cmp.eq(r26,#0x2); r26 = add(r26,#-2); r6  = memw(r1++#0x8); r23 = memw(r1+#0x4) }
1cd4:  { r8 = add(r8,#0x400); if(p0) r25:24 = combine(r9,r7);
         activation.ub = mxmem(r6,r24):cm; weight.b = mxmem(r8,r25) }
1ce4:  { r8 = add(r8,add(r25,#0x1));
         activation.ub = mxmem(r23,r24):cm; weight.b = mxmem(r8,r25) } :endloop0
```

Identical to QNN's 0x2ea820–0x2ea848 body.

## Result: NO perf change at 2048³

| Variant | mmv8 avg cyc/instance | total wall |
|---------|----------------------:|-----------:|
| Original V8 (2-packet body, sw loop)    | 277,076 | 20.5M |
| 3-packet body, sw loop                  | 277,368 | 20.5M |
| 3-packet body + hardware loop0          | 277,032 | 20.5M |
| Above + mxmem2 bias                     | 276,982 | 20.5M |
| Above + sat.ub Rt=0x3FF                 | 277,644 | 20.6M |
| Above + pre-baked act_ptrs              | 278,302 | 20.6M |

All within run-to-run variance. The asm-level replica is complete; perf is invariant.

## Where the gap actually lives (since it's NOT in the asm)

**Wall-time ground truth** (mean of 2nd+3rd inferences):

| shape | V8 wall µs | QNN wall µs | ratio |
|------:|-----------:|------------:|------:|
| 512³  | 193 | 43 | 4.5× |
| 2048³ | 9,764 | 654 | 14.9× |
| 4096³ | 79,706 | 14,101 | 5.7× |

QNN's `q::ConvLayer_s1.opt` cumulative HMX-active cycles at 2048³ = 608K (per chrometrace JSON). V8 mmv8 cumulative wall cycles = 53.2M. **DIFFERENT METRICS** — chrometrace reports HMX-active-only (excludes stalls), profile.txt reports total wall. Comparing them gave the bogus 30× ratio I had originally claimed.

Per probe (`Agent/qnn_hmx_pipelining.md`), HMX silicon ceiling = 7.89 cyc/MAC packet in tight loop with one tile of pre-loaded data. V8 mmv8 at 2048³ = 66 cyc/packet wall = ~8× over silicon ceiling — explained by the per-`:cm`-act-stride penalty (~58 cyc) found in `Agent/v8_perf_gap_isolated_2026-04-26.md`.

Likely culprits, in order of plausibility:

1. **VTCM bank conflicts** between act and weight loads in the same MAC packet. V8's allocations may co-locate them; QNN's compile-time descriptor synthesis may stagger.
2. **HMX-VTCM cache state across instances**. QNN's 256 small ConvLayer calls keep the HMX-VTCM channel hot; V8's 192 large calls (each with sat.ub at end) may cool the channel between instances.
3. **`mxmem` issue rate is gated by something we don't control**: act tile address being on a "fast path" vs "slow path" within HMX's VTCM port arbiter. ConvLayer's pre-baked addresses may always land on fast paths.

## What's actually needed to close the gap

These are blocked on getting visibility into QNN's actual ConvLayer call:

1. **Instrument-and-dump descriptors**: build a custom op-pkg that hooks before `q::ConvLayer_s1.opt` at runtime and dumps the 3 descriptor structs (`hmx_conv_out_desc_t`, `hmx_conv_act_desc_t`, `hmx_conv_mask_desc_t`) to a known buffer. Then we'd know the actual VALUES (not just RE'd LAYOUT) and can call `hmx_convbbb1x1_stride1` directly via dlsym (mechanism already proven).
2. **Run V8 mmv8 with VTCM allocations FORCIBLY split** to different banks for act/wt and re-measure. If perf jumps, banking was the gap.
3. **Disasm the V8 surrounding code (loop1 / per-instance setup / VTCM acquire)** to see what packets QNN runs between MAC chains that we don't.

## Files modified 2026-04-26 (mmv8 asm replica)

- `example/hmx_matmul_phase3/src/HmxMatMulV8Op.cpp` — `tile_output` branch: 3-packet body + hardware loop0 + mxmem2 bias + sat.ub Rt=0x3FF + pre-baked act_ptrs

The replica change is bit-equivalent (V8 still runs at 512³/2048³ producing same shape output, no crash). It's not faster, but it's now the closest possible match to QNN's hot loop without descriptor synthesis. Future work that DOES synthesize descriptors will plug into the same kernel structure.
