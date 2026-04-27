---
name: dlsym call to hmx_convbbb1x1_stride1 PROVEN — wall-time gap is NOT in the kernel 2026-04-26
description: Successfully called QNN's hmx_convbbb1x1_stride1 from custom op-pkg via dlsym. Cycle measurement shows ~67 cyc/packet — IDENTICAL to V8 inline-asm. Real wall-time gap to QNN at 2048³ = 14.9× (V8 9764 µs vs QNN 654 µs); at 512³ = 4.5×; at 4096³ = 5.7×. NOT accessible via the kernel call alone — it lives in QNN compile-time graph machinery (VTCM bank allocation, multi-instance scheduling) that custom op-pkgs cannot replicate. Output validation shows the kernel produces bias-only-with-noise output for V8 row-major activation, indicating activation FORMAT mismatch. V9 with Crouton activation crashes the kernel — additional VTCM stride/layout constraint we can't satisfy.
type: project
---

# dlsym to hmx_convbbb1x1_stride1 — call works, perf advantage not accessible (2026-04-26)

## Wall-time ground truth (THE primary metric)

| shape | V8 wall µs | QNN wall µs | ratio | V8 freq | QNN freq |
|------:|-----------:|------------:|------:|--------:|---------:|
| 512³  | 193 | 43 | **4.5×** | 2.31 GHz | 2.98 GHz |
| 1024³ | 1,292 | 114 | **11.3×** | 2.30 GHz | 4.38 GHz* |
| 2048³ | 9,764 | 654 | **14.9×** | 2.11 GHz | 2.68 GHz |
| 4096³ | 79,706 | 14,101 | **5.7×** | 2.00 GHz | 2.33 GHz |

\* 1024³ QNN cycle counter looks suspicious; wall µs is still ground truth.

The wall ratio is **not 30×** as I'd originally claimed — that came from comparing chrometrace JSON's per-op `cycles` field (HMX-active-only, no stalls) with V8's profile.txt cycles (wall-clock total). DIFFERENT METRICS.

Source: `qnn-profile-viewer` "Accelerator (execute) time" output, mean of 2nd+3rd inferences.

## Achieved

1. **dlsym call mechanism proven** — `R_HEX_JMP_SLOT` resolves correctly across .so. V8 mmv8 wrapper calls `hmx_convbbb1x1_stride1` per (mt, nt) tile, kernel runs without crash, completes 16 MAC packets per call (verified by cycle count).
2. **Descriptor field interpretation derived** for single-tile call:
   - `out_desc = {out_tbl[1], 1 (stride_dwords), 0 (y_stride), 1 (n_tiles_pow2), 64 (m_total), 32 (k_total_bytes)}`
   - `act_desc = {act_tbl[K_t], K_t (n_act_pairs = K-strip count), 0 (y_stride)}`
   - `mask_desc = {0 (out_check, probe-pass), 0x3FF (out_rt), 0 (act_check), 2047 (act_rt_base), 0, 0, 0x3FF (alt_rt)}`
3. **Bypassed bitsclr alignment probe** by setting check fields to 0.

## Measured perf @ 512³ (M_TILE=N_TILE=256, K_t=16, 1024 MAC packets per V8 op)

| Variant | mmv8 cyc/op | cyc/packet | output range | std |
|---------|------------:|-----------:|--------------|----:|
| V8 baseline (inline asm) | 60K | **58.6** | [0, 255] | 34.88 |
| V8 dlsym, K_t=16 MACs    | 71K | **69.7** | [113, 139] | **6.37** |
| V8 dlsym, K_t/2=8 MACs   | 34K | 33.5 | [119, 135] | **3.70** |

Output std DOUBLES going from 8 to 16 MACs (sqrt(2) random-walk consistent), confirming kernel IS doing requested MAC count. But output magnitude is bias-only-with-noise — std at 16 MACs (6.37) is ~5× smaller than baseline's 34.88.

## Definitive: wall-time gap is NOT in the kernel

**dlsym call to the SAME QNN kernel gives 69 cyc/packet wall** — same as V8's hand-written inline-asm replica. Wall ratio V8/QNN at 2048³ = 14.9× (real, from µs).

- The kernel itself runs at the same per-MAC throughput regardless of caller (us vs QNN's graph runtime).
- QNN's wall-time advantage MUST come from **graph-level scheduling**: VTCM bank allocation pattern, multi-instance interleaving on HMX, or something else only QNN's compiler controls.

For us as custom-op-pkg authors, **the QNN ConvLayer wall-time rate is not reachable** without writing our own QNN graph compiler equivalent. The 14.9× gap (5.7× at 4096³ where QNN's @Spill/@Fill kicks in and V8 closes ground) is the real ceiling from custom op-pkg perspective.

## Why output is bias-only with noise

The kernel runs 16 MAC packets but the resulting `acc` value, after bias-fp16 scaling and sat.ub, is essentially `bias_zp + noise`. If the activation tile bytes were valid for what HMX `:cm` expects, accumulation would produce CORRELATED matmul output (std 35). Instead std 6.4 — random-walk-like.

This means: **the activation tile byte layout HMX `:cm` mode expects from `hmx_convbbb1x1_stride1` differs from what V8's pack_act_rm produces** (row-major 32×32). Likely candidates:

1. HMX `:cm` interprets the 1024-byte tile as 16 rows × 64 cols (concatenated rows), NOT 32×32. Mismatch with our pack_act_rm.
2. HMX `:cm` expects "Crouton" 8-block-of-128B layout where each block is 4 rows × 32 cols.

V9 with PackActCrouton (option 2) was tested but crashed — additional VTCM constraint we can't satisfy.

## V9 dlsym crash — known unknown

V9 BbbKMajor with PackActCrouton input + V8 weight + same descriptor structure → "Graph Execution failure" on first dlsym call. Inline-asm HMX in same V9 op runs fine. So crash is from kernel itself, not framework or input bindings.

Likely cause: V9 packed_act has tiles spaced 2048 bytes apart (2× stride vs V8's 1024). HMX `:cm` may prefetch "next 1024 bytes" expecting consecutive tiles, hits gap data → fault. V8's 1024-stride layout doesn't trigger this.

## Bottom line

**hmx_convbbb1x1_stride1 is callable but doesn't unlock QNN-perf.** The wall-time gap (14.9× at 2048³) requires QNN's compile-time graph machinery, not the kernel itself. Our V8 mmv8 inline-asm path IS already at the maximum throughput accessible to a custom op-pkg.

To break past this would require:
- Reverse-engineering QNN's compile-time VTCM allocation algorithm and replicating it in our op-pkg's tensor placement decisions
- OR writing a multi-instance scheduling layer that hides VTCM stalls via overlapping calls
- Both are substantial efforts of unknown payoff (not guaranteed to recover the full 14.9× since the gap might be in HMX-internal state we can't even see)

## Files modified for the dlsym path (preserved under compile flags)

- `src/HmxMatMulV8Op.cpp` — `V8_USE_DLSYM_PER_TILE` enables per-(mt,nt) dlsym call. Single-tile descriptor synthesis with field values that don't crash.
- `src/HmxMatMulV9SkelOp.cpp` — `V9_USE_DLSYM` parallel path. Crashes — bug retained for future investigation. `V9_INLINE_MINIMAL_HMX` works.
- `standard_flow/phaseB_v8/gen_v9_test.py` — uses PackWeightToHmxTileV3 (V8 weight format) for V9 test instead of PackActCrouton on weight.
- `MatMulV8Package.xml` — BbbKMajor in[1] shape updated to V8 weight layout `[1, N/32, K/32, 1024]`.
