---
name: V8 apples-to-apples vs QNN — corrected wall ratios 2026-04-26
description: Made V8 consume the SAME ONNX input shape (rank-3 [1,M,K] u8) and produce the SAME output (rank-3 [1,M,N] u8 row-major) as QNN's MatMul. Required adding Reshape at input + UntileToRowMajor + Reshape at output. Real wall ratios: 10.4× at 512³, 17.5× at 1024³, 21.3× at 2048³. Earlier numbers (4.5/11.3/15.1×) were biased — V8 was skipping the Untile/Reshape that QNN does internally as part of inference.
type: project
---

# V8 apples-to-apples — corrected gap measurements 2026-04-26

## Wall-time table (mean of 2nd+3rd inferences, µs from `qnn-profile-viewer`)

| Shape | **Apples V8** µs | Tile-only V8 µs | QNN µs | Apples÷QNN | Tile-only÷QNN |
|------:|-----------------:|----------------:|-------:|-----------:|--------------:|
| 512³  | **448** | 193 | 43 | **10.4×** | 4.5× |
| 1024³ | **1998** | 1292 | 114 | **17.5×** | 11.3× |
| 2048³ | **13912** | 9849 | 654 | **21.3×** | 15.1× |

The "Apples V8" column is the **honest comparison** with QNN — both consume rank-3 [1, M, K] u8 input and produce rank-3 [1, M, N] u8 output.

## What "apples-to-apples" required

To match QNN's interface (verified by inspecting QNN's matmul.onnx):
- QNN: input A `[1, M, K]` fp32 (with quant_overrides → u8 int8 internally), output Y `[1, M, N]` fp32 → u8
- V8 production was: input `[1, 1, M, K]` u8 (rank-4), output `[1, M_t, N_t, 1024]` u8 tile-layout

Changes made (`gen_v8_apples.py`):
1. **Input rank-3** `[1, M, K]` u8 — added `Reshape` to internal rank-4 `[1, 1, M, K]`
2. **Output rank-3** `[1, M, N]` u8 row-major — added `UntileToRowMajor` per M-stripe + `Reshape` to drop the leading `1`
3. Internal tile format: **BOTH V8 production (PackActivationU8RowMajor + MatMulV8) and Crouton (PackActCrouton + BbbKMajor) work correctly** — verified bit-near-exact vs Python reference (only fp16 rounding diffs, max delta = 1). My earlier "bias-only broken" diagnosis at N_t > 8 was WRONG: the narrow output range (e.g. [126, 129] at 2048³) is just the actual matmul output distribution at small bias-scale, NOT a bug. Wall time identical between formats (~1% noise).
4. ConverterOpPackage updated: added `UntileToRowMajorShapeInference`. Built fresh `libConverterOpPackage.so`.
5. MatMulV8Package.xml: added `UntileToRowMajor` OpDef + SupplementalOpDef + SupportedOps entry.

## Decomposing the apples wall

For 2048³ (added 4063 µs vs tile-only 9849 µs):
- mmv8 (HMX): same as production (~9.9 ms)
- Reshape input rank-4: ~0 (no-op, view-only)
- **UntileToRowMajor + Reshape output: ~4 ms** — DDR-write-bandwidth bound
  - Per `untile_to_rowmajor_hvx.c` source comment: "DDR bandwidth for N-strided 32-B writes is the hard limit here"
  - Compare to QNN's q::OutputSlice: 466K cyc / 2.68 GHz = 174 µs at 2048³ → **23× faster than ours**

So our UntileToRowMajor is the new bottleneck for matching QNN at 2048³. If we improved untile to QNN's level (~174 µs), apples wall would drop to 9849+174 = 10023 µs → ratio 15.3× (close to the tile-only ratio).

## True root cause of remaining gap

Even with optimal untile, V8 mmv8 wall ≈ 9849 µs vs QNN HMX wall ≈ 530 µs at 2048³. Gap ≈ 18.6× from HMX kernel side alone. This is the actual gap between V8's HMX MAC kernel and QNN's q::ConvLayer_s1.opt — confirmed via probe data.

Per probes earlier (`Agent/v8_perf_gap_isolated_2026-04-26.md`): V8 mmv8 hits 8.97 cyc/packet when act addr is fixed (HMX silicon ceiling). With actual stride: 67 cyc/packet (8× over ceiling) due to `:cm` activation address-change penalty (~58 cyc per change).

## V9 BbbKMajor + Crouton — works correctly (corrected)

EARLIER diagnosis was wrong. I claimed "bias-only output" at multiple shapes but never compared to actual Python reference. Once I did:

| Shape | Crouton apples wall µs | Diffs vs Python ref | max delta |
|------:|----------------------:|--------------------:|----------:|
| 512³  | 453 | 1.76% | 1 |
| 1024³ | 2019 | 3.71% | 1 |
| 2048³ | 14098 | 1.90% | 1 |

All diffs are fp16 rounding (max delta = 1). Narrow ranges at large K (e.g. [126, 129] at 2048³) are NOT bugs — they're the actual reference matmul output range when K is large and per-channel bias scale ≈ 1/(K × 1.3) is tiny.

**Wall time is essentially identical between Crouton (V9 BbbKMajor) and row-major (V8 MatMulV8) paths** (within 1% noise — 14098 vs 13912 µs at 2048³). So switching to QNN-native Crouton internal format does NOT unlock perf — both paths hit the same HMX wall.

V8 apples (BOTH internal formats work correctly) — `gen_v8_apples.py` produces ONNX with PackActCrouton + BbbKMajor (the Crouton/QNN-native variant).

## Files modified for apples-to-apples

- `standard_flow/phaseB_v8/gen_v8_apples.py` (new, V8 apples-to-apples ONNX generator)
- `standard_flow/phaseB_v8/MatMulV8Package.xml` (added UntileToRowMajor OpDef)
- `standard_flow/phaseB_v8/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/ConverterOpPackage.cpp` (added UntileToRowMajorShapeInference; rebuilt libConverterOpPackage.so)
- `standard_flow/phaseB_v8/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so` (rebuilt)
- `src/HmxMatMulV9SkelOp.cpp` (V8-weight-format addressing for V9 BbbKMajor inline asm; bias load per mt iter — unsuccessful fix attempt)

## Conclusion

V8 ONNX interface AND internal Crouton tile format now match QNN-native. Wall-time gap remains 10-21× (largely from V8 mmv8 HMX kernel itself, plus slow UntileToRowMajor). Switching internal tile format from row-major → Crouton is a wash for wall time — confirmed via direct measurement.

All three are concrete and shape-bounded; none is the "30× wall miracle" we'd need to fully close to QNN.
