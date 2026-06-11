# GDN Kernel Reference (HTP port spec)

Concise current-state source for the **Gated DeltaNet (GDN) linear-attention kernel**
we are porting to Hexagon HTP, validated against the real `Qwen/Qwen3.5-4B` model.
This page is the design baseline now that the host **reference + quantization contract**
are settled; the next phase is the tile-level / device implementation.

Scope note: GDN and Kimi Delta Attention (KDA) differ **only in the gate** (GDN = scalar
per-head decay; KDA = per-channel `Diag(α)`); the chunked GEMM skeleton is identical. We
build a unified core and **do GDN first**.

## 1. What the kernel is (boundary)

The kernel = the chunked gated-delta-rule compute at the FLA `chunk_gated_delta_rule`
boundary inside `Qwen3_5GatedDeltaNet`. Captured I/O (post causal-conv1d+silu; GQA already
expands 16 k/q heads → 32 v-heads; q/k are L2-normalized **inside** the kernel):

| tensor | shape | notes |
|---|---|---|
| query / key / value | `[1, T, 32, 128]` | pre-l2norm; q/k normalized inside |
| g (log-decay) | `[1, T, 32]` | per (token, v-head) scalar = `-exp(A_log)·softplus(a+dt_bias)` |
| beta | `[1, T, 32]` | `sigmoid(b)`, per (token, v-head) |
| **o** (output) | `[1, T, 32, 128]` | before the output gate `norm(o,z)` (that gate is OUTSIDE the kernel) |

Out of kernel scope: conv1d, gate-param (A_log/dt_bias) derivation, output gate, GQA
model-level expand, varlen/cu_seqlens, KV-cache, and any FlashQLA/FlashKDA GPU scheduling
(context-parallel warmup, producer/consumer) — those are interface/scheduling.

Model facts (`Qwen3.5-4B` `text_config`, `model_type qwen3_5`): 32 layers, pattern
3×`linear_attention`+1×`full_attention` (`full_attention_interval=4`) → **24 GDN layers**
(`model.layers.{i}.linear_attn`, i ∉ {3,7,11,…}); `linear_num_key_heads=16`,
`linear_num_value_heads=32`, key/value `head_dim=128`, `linear_conv_kernel_dim=4`,
`attn_output_gate=true`, chunk `C=64`, `mamba_ssm_dtype=float32`.

## 2. Fixed-length kernel unit (HTP needs fixed shapes)

T is fixed at `C = 64`. The HTP kernel unit is **`gdn_chunk`**: exactly one 64-token chunk
plus the incoming state. Fixed I/O (B=1, H=32, Dk=Dv=128):

```
qc,kc,vc : [1,32,64,128]   gc,betac : [1,32,64]   S_in : [1,32,128,128]
   ->  oc : [1,32,64,128]   S_out : [1,32,128,128]
```

Variable sequence length is the **outer loop** `for i in range(ceil(T/64)): o[i],S = gdn_chunk(chunk_i, S)`
(orchestration, not the kernel); the last chunk is zero-padded to 64.

## 3. Compute = 8 GEMMs + gating + one triangular solve

Per chunk, tagged by HTP engine (`[HMX]` integer 32×32-tile matmul, `[HVX]` vector
elementwise/exp/l2norm, `[SEQ]` sequential). Two stages: **A intra-chunk (all chunks
independent → parallel)**, **B inter-chunk (sequential in state S, parallel over heads)**.

```
PRE  l2norm(q,k); q*=1/√Dk; v_β=v·β; k_β=k·β; g=cumsum(g)                       [HVX]
A 8  decay = tril(exp(g_i−g_j))                                                 [HVX]
A 9  A = −(k_β·kᵀ)·decay, strictly-lower                                        [HMX]+[HVX]
A10  T = (I−A)⁻¹  (forward substitution, 64 steps)                              [SEQ]
A11  U = T·v_β                                                                  [HMX]
A12  W = T·(k_β·e^g)                                                            [HMX]
B13  P = (q·kᵀ)·decay, causal                                                   [HMX]+[HVX]
B14  v' = W·S                                                                   [HMX]
B15  v_new = U − v'                                                             [HVX]
B16  attn_inter = (q·e^g)·S                                                     [HMX]
B17  o = attn_inter + P·v_new                                                   [HMX]+[HVX]
B18  S = S·e^{g_last} + (k·e^{g_last−g})ᵀ·v_new                                 [HVX]+[HMX]
```

GEMM tile shapes (32×32 tiles): `[64,64]`=2×2, `[64,128]`=2×4, `[128,128]`=4×4, `[128,64]`=4×2.

## 4. Quantization contract (HTP `A @ W^T`)

Hardware constraint of the current HMX kernel: every GEMM is `A @ W^T` with
**A (left) = per-tensor ASYMMETRIC int16** (activation port; scale + zero-point folded into
the drain) and **W (right) = per-tensor SYMMETRIC int8** (weight port; scale only). Quant is
positional. Our freedom is *which operand becomes A vs W* (compute `C` or `Cᵀ`): we orient
each GEMM so the **outlier-prone** operand is A (int16) and the **bounded** one is W (int8).

| GEMM | math | A = asym int16 (left) | W = sym int8 (right) | kernel computes Cᵀ |
|---|---|---|---|---|
| A_kk | k_β·kᵀ | k_β | k | no |
| A_Uv | T·v_β | v_β | T | yes |
| A_Wk | T·(k_β·e^g) | k_β·e^g | T | yes |
| B_qk | q·kᵀ | **q** | k | no |
| B_WS | W·S | S | W(=T·(k_β·e^g)) | yes |
| B_qS | (q·e^g)·S | S | q·e^g | yes |
| B_Pv | P·v_new | v_new | P | yes |
| B_kv | (k·dec)ᵀ·v_new | v_new | k·dec | yes |

- **q is asym-int16** (in `B_qk`). 6/8 GEMMs are computed transposed (Cᵀ) to keep the
  ranged operand in int16; transpose is free numerically.
- Under w8a16 each GEMM must have exactly one sym-int8 operand. In `B_qS`, S (the worst-
  conditioned operand) must stay int16, so `q·e^g` is forced to int8.
- The `[SEQ]` triangular solve and all gating/exp/l2norm stay fp16/fp on HVX (not quantized;
  fp16 is a float format, not quantization). State S is carried high-precision between
  chunks and quantized only at the GEMM inputs that read it.

## 5. Activation ranges (real Qwen3.5-4B, all 768 captures)

Boundary I/O (pre-l2norm for q/k):

| tensor | min | max | mean | std | abs-max | per-layer abs-max [lo,hi] |
|---|---|---|---|---|---|---|
| query | −0.279 | 25.1 | ~0 | 0.161 | 25.1 | [0.95, 25.1] |
| key | −0.279 | 28.1 | ~0 | 0.153 | 28.1 | [1.03, 28.1] |
| value | −0.279 | 13.6 | ~0 | 0.098 | 13.6 | [1.60, 13.6] |
| g | −42.0 | 0 | −0.30 | 0.818 | 42.0 | [2.19, 42.0] |
| beta | 0.001 | 1.000 | 0.495 | 0.277 | 1.000 | [0.99, 1.0] |
| o | −0.089 | 0.680 | ~0 | 0.0032 | 0.680 | [0.025, 0.68] |

Internal GEMM operands (post-l2norm/derived; L00 example abs-max / crest=max/RMS): k≈0.99/13,
T=1.0/7.7, q≈0.087/13, q·e^g≈0.086/17, P≈0.078/16, k_β≈0.95/21, **S≈7.0/185 (worst)**,
v_β≈9.84/72, v_new≈9.67/95.

Implications: q/k/v are heavy-tailed (crest ~150–185 from rare outliers); l2norm tames q/k
before the GEMMs, but v/S/v_new keep their range → those go to int16. Per-layer abs-max
varies ~30× → **per-layer static scales are essential**.

## 6. Validation methodology & status

- **Golden**: real `Qwen3.5-4B` forward over a frozen prompt corpus
  (`tests/gdn/prompts.json`, 32 prompts, 24 calib / 8 test, EN/ZH/code/math/narrative,
  1–5 chunks). Per-(prompt,layer) kernel I/O stored as **lossless bf16-bits** in
  `tests/gdn/golden/` (gitignored, ~2 GB, regenerable). 768 captures = 32 × 24 layers.
- **fp32 reference**: `gdn_ref_kernel.gdn_kernel` (our torch impl) reproduces FLA to
  **2e-7** — it IS the reference; the model's bf16 `o` sits at the **bf16 floor ≈ 3e-3**.
- **A single test = one GDN layer**: build STATIC per-layer scales from the layer's calib
  prompts (each layer is its own kernel), run on held-out test prompts, assert
  `relerr(o_quant, o_fp32) < tol` (default 1.5e-2). pytest-parametrized per layer.
- **Precision findings**: int8 is dead (per-tensor 10–17 dB SNR; even per-head ~22–30 dB →
  3–8% err); **int16 needed**. Scale must be **per-layer** (pooling across layers is ~5×
  pessimistic). **Shared-across-heads** scale is acceptable and *more robust* to calibration
  than per-head. **w8a16** (int8×int16) fits int32 accumulate (≈2²⁹ < 2³¹). Asymmetric on
  the int16 side is ~negligible (int16 has the range); the int8 operand dominates error.
- **Current result**: **L00 is the best-aligned layer**, fixed as the initial dev/test
  target — passes w8a16 at **1.28e-2**. Across all layers only L00 is < 1.5e-2 (rest 2–4%);
  L18/L30 worst, driven by a code-prompt outlier that exceeds calibration range (bit-width
  independent → calibration coverage, not quantization).

## 7. Files & reproduce

```bash
# regenerate golden (needs downloads/Qwen3.5-4B; mirror, no proxy)
env -u http_proxy -u https_proxy -u all_proxy .venv/bin/python scripts/gdn_extract_golden.py \
    --model downloads/Qwen3.5-4B --prompts tests/gdn/prompts.json --out tests/gdn/golden
# single-layer test (default: w8a16, focus L00); GDN_LAYER=all for every layer
.venv/bin/python tests/gdn/test_gdn_layer.py
.venv/bin/python -m pytest tests/gdn/test_gdn_layer.py -q
# exactness of the torch reference vs FLA fp32 (needs old fp32 golden) / data ranges
.venv/bin/python scripts/gdn_ref_kernel.py --validate
.venv/bin/python scripts/gdn_data_stats.py
```

| file | role |
|---|---|
| `scripts/gdn_ref_kernel.py` | torch reference: `gdn_chunk` (fixed unit) + `gdn_kernel` (loop); `mm(A,B,site)` hook |
| `scripts/gdn_extract_golden.py` | run model, capture per-layer kernel I/O (bf16-bits) |
| `scripts/gdn_data_stats.py` | shapes + activation ranges of the golden |
| `tests/gdn/prompts.json` | frozen prompt corpus (append-only) |
| `tests/gdn/test_gdn_layer.py` | single-layer static-quant test; `ASYM_SIDE` = the A/W map |

## 8. Next

Tile-level / device implementation against L00: 32×32-tile dataflow, static rescale at the
`A @ W^T` GEMM boundaries (no dynamic Q/DQ), VTCM residency of U/W/S, GQA 16→32 reuse, and
the 6/8-GEMM Cᵀ orientation. Background: see `~/.claude` memory
`project_gdn_kda_htp_kickoff_2026-05-31` and the QNN custom-op infra in
`qnn_hmx_matmul_status.md`.
