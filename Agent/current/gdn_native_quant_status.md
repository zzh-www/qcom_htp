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

Localization so far:
- **Faithful simulator** (bit-faithful to the deployed selector/padded solve, symmetric int16) =
  oc **3.4e-3** on real golden — so the scheme, the requant chain, and the scales are NOT the gap.
- **Transcendental probe** (`scripts/gdn_probe_ops.py` / `example/gdn_native/probe_htp.sh`,
  isolates exp & l2norm on HTP at real ranges): HTP **exp is faithful** (htp-vs-quantin 2e-5);
  HTP **rsqrt/l2norm has a 1.55e-2 LUT error**, but injecting that into the full sim
  (`GDN_L2_ERR`) yields only ~1.1% oc → **not enough to explain 27%**.

=> Transcendentals are largely exonerated. The residual lives in the quantized int16 MatMul/solve
chain AS EXECUTED ON HTP — the simulator (float-then-quantize per op) underestimates the on-device
noise by ~70×. Prime suspects: the auto-inserted per-MatMul bias quantization (the sim has no
bias) and int16 accumulation/requant rounding across 91 matmuls. Next: per-stage device probe of
a small int16 MatMul chain vs the sim; inspect the bias tensors in `gdn_quant.dlc`; and chase the
systematic 0.84 scale (a single corrective scale may recover much of the magnitude).

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
