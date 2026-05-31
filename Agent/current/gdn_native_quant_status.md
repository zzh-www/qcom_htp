# GDN on QNN-native: float reference + all-integer HTP graph

Status of the goal **"先在 qnn native 上实现 gdn kernel对齐输出"** — implement the GDN chunk
kernel as a QNN-native graph and align its output to the fp64 reference, on the **HTP backend**,
quantized (no fp16). The executable math spec is `scripts/gdn_ref_kernel.py::gdn_chunk`; the
kernel/quant contract is `gdn_kernel_reference.md`. This page is the QNN-native realization.

## TL;DR (verified root cause)

The all-integer GDN graph **composes and runs correctly on the v75 HTP** (structure is right,
corr 0.97), but the device output is **~16% (p00) … 43% (p29)** — and the cause is now pinned:

> **`int16 × int16` MatMul OVERFLOWS the HTP int32 accumulator.** Each int16×int16 product is
> ~2³⁰; the GDN GEMMs contract over the **128-dim head**, so the accumulator sums to ~**2³⁷ ≫ 2³¹**
> (int32). It saturates/wraps on outlier-heavy dot-products → the 16–43% (data-dependent).

Proof — `scripts/gdn_faithful_sim.py` runs the exact deployed graph with a faithful int32
accumulator (real golden L00, oc relerr p00/p29):

| matmul accumulator | p00 | p29 |
|---|---|---|
| **float** (the trap — what my earlier sims used) | 0.001 | 0.001 |
| **int32-saturating, w16a16** | **0.146** | **0.693** | ← matches device 0.164 / 0.431 |
| int32-saturating, **w8a16** (one operand int8) | 0.053 | 0.049 |

So **"全程 int16" (int16×int16 matmul) is physically impossible** for the 128-dim GEMMs on HTP.
The matmul **weight-port must be int8** (w8a16) so int8×int16×128 ≈ 5.3e8 < 2³¹ fits; **activations
stay int16**. This is exactly why `gdn_kernel_reference.md` §4 / the kickoff memory locked **w8a16,
not w16a16** ("int16×int16 over contract-128 ≈ 2³⁷ > int32"). Everything else — the native L2Norm,
exp, the `(I-A)⁻¹` solve, all elementwise — is **fine at int16**.

**Earlier wrong conclusions in this file's history (now corrected):** "needs ~24-bit",
"data-dependent shrink", "deep-chain requant floor", "fp16/custom-HMX required". All were artifacts
of a non-faithful simulator: it (a) **accumulated matmuls in float** (so it never saw the int32
overflow), and (b) **quantized the l2norm internals** (`x*x`) which the device fuses into the
native L2Norm op. With both fixed, the only real blocker is the accumulator → w8a16.

## Files

| file | role |
|---|---|
| `scripts/gdn_onnx_kernel.py` | `gdn_chunk` as a static ONNX graph. `GDNChunk` (float) / `GDNChunkQ` (quant path, structural consts as inputs). l2norm uses `F.normalize` → fuses to native QNN **L2Norm**. ORT exact 3.8e-7. |
| `scripts/gdn_faithful_sim.py` | **device-faithful** sim: int32 matmul accumulator + fused L2Norm. `ACC=int32 WBITS=16/8`. Reproduces the device. |
| `scripts/gdn_quant_sim.py` | older partial sim (float-accumulate, optimistic — kept for the per-GEMM scheme study only). |
| `example/gdn_native/run_gdn_native_quant.sh` | all-int HTP path: two-pass symmetric quantize → ctx → device → compare. |
| `example/gdn_native/run_gdn_native.sh` | float native path (CPU fp32 exact; HTP fp16 overflows real data). |
| `example/gdn_native/ctxgen_check.sh` | host-only compose gate (no device). |
| `example/gdn_native/{probe_htp,probe_chain}.sh`, `scripts/gdn_probe_*.py` | isolate single HTP ops (exp, L2Norm, matmul chain) vs sim. |

## The triangular solve (static reformulation — works, not the problem)

`gdn_chunk` has a sequential `T = (I-A)⁻¹`. Made static as **block forward substitution**
(`solve_T_blocked`): logical block 16 / physical block 32 (matmuls 32-aligned — HTP rejects 16×16),
block extract by **Slice+Pad**, no Concat. The Neumann product `∏(I+A^(2^i))` was rejected (its
powers reach ~1e7). The solve quantizes fine at int16 (the faithful sim confirms).

## HTP-composition recipe (the real, reusable constraint cascade)

Getting a fully-quantized int16 graph to **compose** on HTP (per `HtpOpDefSupplement.html`):
1. **No runtime mask ops** — bake masks as numpy-backed constant initializers (no EyeLike/Compare).
2. **32-aligned matmul dims** — logical-16 / physical-32 block padding (16×16 rejected).
3. **No Concat of differently-scaled tensors** — Slice+Pad block extraction.
4. **act×act MatMul, not FullyConnected** — feed structural constants (`cumsum_U`, `sel0..3`) as
   runtime INPUTS so their matmuls stay MatMul (a constant operand → FC, int16 FC unsupported).
5. **Symmetric operands (offset −32768)** — two-pass: calibrate → `--dump_encoding_json` →
   `symmetric_overrides_from_dump` rewrites every encoding symmetric → re-convert. → ctxgen OK.
6. **l2norm must be the native op** — use `torch.nn.functional.normalize` (→ ONNX ReduceL2/Div →
   QNN `L2Norm`, verified with `qairt-dlc-info`); a hand-built LpNormalization / manual
   sum-rsqrt-mul risks decomposing into per-op-quantized primitives.

## Float native path — DONE (fp16 rejected for deployment)

- **CPU host fp32**: oc 3.5e-7 vs fp64 — the QNN-native graph is exact.
- **HTP fp16**: aligns on mild prompts but **overflows to NaN** on heavy-tailed real Qwen
  activations (the l2norm sum-of-squares; abs-max ~25) → fp16 unusable.

## Next (to discuss, not yet built)

Implement **w8a16 per-GEMM** with the §4 `ASYM_SIDE` orientation (int8 on the bounded operand,
int16 on the outlier-prone one) so the accumulator fits AND int8 lands on the better-conditioned
operand. Faithful-sim w8a16 (naive orientation) = 5%; the project's oriented w8a16 host study =
**1.28e-2** (`tests/gdn/test_gdn_layer.py`). To realize on the device-native graph, each
precision-critical matmul needs a designated int8 weight-port (today all operands are forced int16
by `symmetric_overrides_from_dump`).

## Reproduce

```bash
# device-faithful simulator — the int32 accumulator is the whole story
ACC=float WBITS=16 .venv/bin/python scripts/gdn_faithful_sim.py   # 0.001 (float accumulate = trap)
ACC=int32 WBITS=16 .venv/bin/python scripts/gdn_faithful_sim.py   # 0.146/0.693 == device (overflow)
ACC=int32 WBITS=8  .venv/bin/python scripts/gdn_faithful_sim.py   # 0.05  (w8a16 fits int32)
# float native graph is exact through the QNN runtime
./example/gdn_native/run_gdn_native.sh                            # CPU fp32: oc 3.5e-7
# all-int16 HTP device run (composes, runs, but ~16% due to the int32-accumulator overflow)
GDN_LAYER=0 TEST_PROMPT=p00 ./example/gdn_native/run_gdn_native_quant.sh
# inspect what each op became in the DLC (verify native L2Norm etc.)
qairt-dlc-to-json -i example/gdn_native/quant_w16a16_L0/gdn_quant.dlc -o /tmp/q.json
```
