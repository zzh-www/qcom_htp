# GDN on QNN-native: float reference + all-integer HTP graph

Status of the goal **"先在 qnn native 上实现 gdn kernel对齐输出"** — implement the GDN chunk
kernel as a QNN-native graph and align its output to the fp64 reference, on the **HTP backend**,
**fully quantized (all-integer, no fp16)**.

The executable spec for the math is `scripts/gdn_ref_kernel.py::gdn_chunk`; this page covers the
QNN-native realization. See `gdn_kernel_reference.md` for the kernel/quant contract.

## Files

| file | role |
|---|---|
| `scripts/gdn_onnx_kernel.py` | `gdn_chunk` as a static, ONNX-exportable graph. `GDNChunk` (float, baked-const) and `GDNChunkQ` (quantized path: structural constants are runtime inputs). Exact vs fp64 (ORT 3.7e-7). |
| `scripts/gdn_quant_sim.py` | host fixed-point simulator — quantizes every op output, two-pass calibrate/eval on real golden chunks. Iterate the scheme in seconds, no device. |
| `example/gdn_native/run_gdn_native.sh` | float native path: ONNX→DLC→qnn-net-run (`BACKEND=cpu` host fp32, or `BACKEND=htp` device fp16). |
| `example/gdn_native/run_gdn_native_quant.sh` | **all-integer HTP path**: two-pass symmetric int16 → context binary → device run → compare. |
| `example/gdn_native/ctxgen_check.sh` | host-only export→convert→quantize→ctxgen (no device) — fast HTP-composition gate. |

## The triangular solve is the crux

`gdn_chunk` has a sequential `T = (I-A)^-1` (forward substitution, [SEQ]). To make it a static
graph it must be reformulated:

- **Neumann product** `∏(I+A^(2^i))` — algebraically exact (A strictly-lower 64×64 is nilpotent)
  and fine in fp64/fp32, but the matrix powers transiently reach **~1e7** before the nilpotent
  collapse → destroys int16 and overflows fp16. **Rejected.**
- **Block forward substitution** (the deployed form, `solve_T_blocked`): every intermediate stays
  O(10). LOGICAL block 16 (small enough the per-block inverse stays bounded) but PHYSICAL block 32
  (every block padded to 32×32 so all matmuls are 32-aligned — HTP rejects 16×16). Block
  place/extract uses constant 0/1 **selector** matmuls (no Concat/Slice — HTP rejects quantized
  Concat of differently-scaled tensors).

## Float native path — DONE (but fp16 is rejected for deployment)

`run_gdn_native.sh` converts the graph and runs it through the real QNN runtime:
- **CPU host backend, fp32**: oc 3.5e-7 / S_out 6.9e-7 vs fp64 ref — the QNN-native graph is exact.
- **HTP device backend, fp16**: 2.7e-3 on well-conditioned inputs, **but overflows to NaN on real
  Qwen activations** (heavy-tailed, abs-max ~25, crest ~185). Confirms fp16 is unusable → the
  deployment must be all-integer.

## All-integer HTP path — graph COMPOSES + RUNS; accuracy tuning open

`run_gdn_native_quant.sh` produces a fully-quantized int16 graph (no float fallback: the 8 GEMMs,
Exp, Sqrt/rsqrt, and the solve all run as QNN quantized HTP kernels). Getting it to **compose on
HTP** required clearing a cascade of int16-MatMul constraints (per the HTP Backend Op Definition
Supplement, `tools/qnn-sdk/docs/.../HtpOpDefSupplement.html`):

1. **No runtime mask ops** — `torch.eye`/compares export as EyeLike/GreaterOrEqual (no QNN
   translation) → bake masks as constants / numpy-backed initializers.
2. **32-aligned matmul dims** — 16×16 block matmuls rejected → logical-16 / physical-32 padding.
3. **No Concat of differently-scaled tensors** — assemble T with selector MatMul+Add, not Concat.
4. **activation×activation MatMul, not FullyConnected** — a constant matmul operand is lowered to
   FC (int16 FC unsupported) → feed the structural constants (`cumsum_U`, `sel0..3`) as runtime
   INPUTS so they quantize as activations (`GDNChunkQ` / `const_inputs`).
5. **Symmetric operands (offset −32768)** — HTP int16 MatMul needs symmetric operands, but
   qairt-quantizer assigns offset 0 to non-negative tensors. Fix = **two-pass**: calibrate once
   (`--dump_encoding_json`), rewrite every encoding symmetric (`symmetric_overrides_from_dump`,
   offset −32768, scale = max-abs / 32767), re-convert encoding-driven. → **ctxgen OK**.

**Current device result (L00, real golden chunk):** the all-integer graph composes, runs on the
v75 HTP from a context binary, and emits structured output — but accuracy is **oc ~2.7e-1 /
S_out ~3.8e-1**, far from the host simulator's prediction.

**The simulator says this scheme should align:** all-int16, blocked solve, on real golden chunks —
asymmetric **1.9e-3**, symmetric (HTP-faithful) **3.4e-3**, worst over 6 prompts **~7e-3** — all
within the 1.5e-2 tolerance. The final DLC encodings match the intended symmetric scales exactly.

**=> The gap is precision, not a bug.** The HTP output is **highly correlated with the reference
(corr 0.97 oc / 0.93 S)** — the all-integer computation is fundamentally CORRECT. The 27%
decomposes into a **systematic ~16% magnitude shrink** (best-fit scale 0.84) + **~25% residual
quantization noise**.

### Root cause (localized by device probes)

Three isolation probes show the individual HTP ops are FAITHFUL to the simulator:
- `gdn_probe_ops.py`/`probe_htp.sh`: HTP **exp faithful** (htp-vs-quantin 2e-5); the **fused
  `LpNormalization`→L2Norm op faithful** (2e-4) — but the *manual* sumsq/rsqrt l2norm had a 1.55e-2
  error because it exposes a per-tensor-quantized sum-of-squares.
- `gdn_probe_chain.py`/`probe_chain.sh`: a 2-deep **int16 MatMul chain matches the sim** (3.4e-3).

**Fix 1 applied — fused L2Norm.** Replacing the manual l2norm with the `LpNormalization` op
(`_L2Norm`/`l2norm_lastdim`) cut device oc **27% → 17%**.

**The remaining 17% = per-tensor (head-shared) quantization of a multi-head graph + deep-chain
requant noise.** GDN runs 32 heads as a batch dim, but QNN quantizes each `[1,32,…]` tensor with
ONE scale set by the max-norm head:
- Error concentrates in **small-norm heads** (`corr(head_norm, head_relerr) = −0.53`; head 21
  norm 4e-4 → relerr 686%) — they're crushed by the shared scale. (Small absolute weight in the
  global norm, so a minor contributor.)
- **Large/well-conditioned heads still carry ~15–22%** — accumulated requant noise across the
  ~190 quantized op outputs. The deployed solve alone adds ~50 matmuls (the selector/padded-block
  realization), each output requantized. The simulator models only ~50 boundaries with exact
  transcendentals, hence its optimistic 3.4e-3.

### Definitive characterization (all levers tried)

The 16% device error decomposes as a **systematic 0.843 magnitude shrink + 4.9% residual**:
`oc ≈ 0.843·ref`, residual-after-global-scale 4.9%, residual-after-**per-head**-scale **2.8%**.

Levers tried (each a device run):
- **Fused L2Norm**: 27% → 16% (kept).
- **Leaner solve** (Slice+Pad block extraction, MatMul 91→77, no-requant): 17% → 16% (negligible).
- **Per-head pre-scale** of v,S_in (`per_head_vscale`, `vscale`/`inv_vscale` inputs; output ×inv):
  no change — the dominant error is NOT per-head input crushing.
- **Headroom** on the symmetric scales (`GDN_HEADROOM`): 1.0 (max-abs) is optimal; 2.0 → worse
  (0.455 scale, coarser → more small-value-rounding shrink); 0.5/0.7 → catastrophic (outliers clip).

**Root cause = irreducible deep-int16-chain noise.** The shrink is small values rounding toward
zero across the ~50-op recurrence/solve (worse with coarser quant), not clipping and not op-count.
It resists every global lever. A **per-head bias correction** (calibrate the 0.78–1.00 per-head
shrink offline, fold 1/s_h into `inv_vscale` — legitimate, like the HMX drain scale) recovers it to
**~2.8%**, but the **~2.8% residual is the chain-noise floor** — below it requires *less int16
requant*, which QNN-native auto-quant cannot provide (per-op requant after every node, no int32
accumulation across ops).

### Conclusion / path to 1.5% tol

The QNN-native auto-quantized int16 route gets GDN **running correctly on HTP** (corr 0.97) but has
a **~2.8% noise floor** it cannot beat for GDN's deep recurrence — short of the 1.5e-2 tolerance.
Reaching tol needs **int32 accumulation without per-op requant**, i.e. the project's **custom HMX
GEMM route** (per-head-GEMM drain / explicit scale), with QNN-native used for orchestration only.
Optionally implement the per-head bias correction to bank the ~2.8% if that is acceptable interim.

## Reproduce

```bash
# scheme validation (host, seconds): asymmetric / symmetric int16 on real golden chunks
PYTHONPATH=scripts .venv/bin/python scripts/gdn_quant_sim.py            # asym ~1.9e-3
PYTHONPATH=scripts GDN_SYM=1 .venv/bin/python scripts/gdn_quant_sim.py  # sym  ~3.4e-3
# float native (rejected fp16, but proves the graph): CPU exact, HTP overflows real data
./example/gdn_native/run_gdn_native.sh                 # CPU fp32: oc 3.5e-7
# all-integer HTP graph composes (host-only gate) + runs on device
./example/gdn_native/ctxgen_check.sh                   # -> CTXGEN OK
GDN_LAYER=0 TEST_PROMPT=p00 TEST_CHUNK=0 ./example/gdn_native/run_gdn_native_quant.sh
```
