---
name: q::ConvLayer.opt.bias_to_vtcm decoded — weight-aware fold, not just type conversion 2026-04-27
description: RE'd native ConvLayer's bias_to_vtcm by feeding a known monotonic bias [0.001..0.256] through native MatMul, then byte-diffing the ctx-binary's VTCM bias region. Result: bias_to_vtcm pre-computes -act_zp × Σ_k W[k,c] per output channel, folds into the int32 bias, and lays out a per-tile (fp16 scale + fp16 baseline) || per-channel (int32 effective bias) format. NOT just int32 → fp16-pair.
type: project
---

# bias_to_vtcm RE (2026-04-27)

User question: 「咱们的 bias 就端到端都用 int32 不行嘛 / qnn 的 bias_to_vtcm 是否就是 int32→fp16 pair 搬运？」

## TL;DR

**Native bias_to_vtcm does FAR more than int32 → fp16-pair conversion.** It is a **weight-aware fold** that:

1. Reads the int32 bias initializer from DLC
2. Reads the i8 **weight matrix** (cross-tensor dependency!)
3. Computes `Σ_k W[k,c]` per output channel
4. Folds `-act_zp × Σ_k W[k,c]` into the bias to produce an "effective integer bias"
5. Computes per-tile fp16 scale (per-tensor output_scale × wt_scale × act_scale)
6. Lays out VTCM as per-tile (128 bytes fp16-pair constants) + per-channel (128 bytes int32 effective bias) = 256 bytes / N-tile

## How we proved it

Wrote `gen_native_with_bias.py` (saved at `/tmp/gen_native_with_bias.py`) that emits a MatMul with explicit fp32 bias `[0.001, 0.002, …, 0.256]` (256 channels). Random i8 weight, random u8 act. qairt-converter quantizes bias to int32; ctxgen produces a model with `q::ConvLayer.opt.bias_to_vtcm` in the graph.

Inspected ctx-binary at offset `0x19100` (just after the 65536-byte weight blob). Layout decoded:

| offset within tile | content | size |
|---|---|---|
| `0..127`     | 32 × `(fp16 scale, fp16 baseline)` — per-tile constants, all 32 channels share same value | 128 B |
| `128..255`   | 32 × `int32 effective_bias[c]` — per-channel folded value | 128 B |

So per N-tile = 256 bytes; for 8 tiles (N=256) = 2048 bytes total.

Hypothesis verified by:
```python
sum_w = W.sum(axis=0)                                  # [N], per-channel Σ_k W[k,c]
bias_q = round(bias_fp32 / bias_scale)                 # quantized int32
effective_int32 = -128 * sum_w + bias_q                # the formula
# Read actual int32 part from ctx[0x19180+t*256 : ...]
# Match across all 256 channels: diff ∈ {0, ±1} (rounding LSB only)
# Best bias_scale fitted = 6.31e-5
```

256/256 channels match within 1 LSB.

## Why native does this fold

Mathematically equivalent rewrite of quantized matmul:
```
acc[m,n] = Σ_k (a[m,k] - act_zp) × w[k,n] + bias[n]
        = Σ_k a[m,k] × w[k,n]   +  (-act_zp × Σ_k w[k,n] + bias[n])
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    folded into "effective bias", computed once at prepare time
```

Native lets HMX consume **raw u8 activation** (no per-element zp subtraction). The HMX `:cm:sat.ub` instruction reads:
- bias-register (1024 bytes loaded via `mxmem`): per-tile fp16 scale + per-channel int32 effective baseline
- accumulator: gets folded baseline added before `:sat.ub` saturates to u8

## How V8 production avoids the fold

V8 production (`PackActivationU8RowMajor` op) **subtracts act_zp from activation** (centered activation), then HMX MAC computes `centered_act × wt`. With centered activation, no `-act_zp × Σ_k w` term is needed.

V8 bias VTCM format becomes simpler — just 32 channels × (fp16 scale, fp16 baseline=0x4000) per tile, 128 bytes/tile. Compare to native's 256 bytes/tile with the int32 fold.

Both produce bit-exact same matmul output (V8 prod 32³–1024³ verified bit-exact in earlier sessions).

## Implication for "int32 end-to-end"

**Strict answer**: no path uses int32 end-to-end into HMX, because:
1. HMX `:cm:sat.ub` silicon requires fp16 scale + fp16 baseline (or per-channel int32 baseline mixed with fp16 scale) for the saturate-to-u8 step
2. Even native's "int32 part" in VTCM isn't raw int32 bias — it's `-act_zp × Σwt + bias` precomputed
3. The per-tile fp16 scale must always exist (4 bytes/tile)

**Pragmatic answer**: user-facing ONNX bias is int32 (matches native ABI), but op-pkg internally must convert into one of the two HMX-compatible VTCM layouts:
- (a) V8-prod simple: act centered upstream → bias = (scale, baseline_zp) fp16-pair only. Easier to implement.
- (b) Native fold: act raw u8 → bias = (per-tile fp16) + (per-channel int32 with `-act_zp×Σwt+bias` folded in). Byte-1:1 with native.

For our V8 C8 path (which uses C8/Crouton act layout), we can choose either (a) or (b) — both are mathematically correct. (a) is what V8 production already does.

## Files

- `/tmp/gen_native_with_bias.py` — test harness emitting MatMul with explicit fp32 bias
- `/tmp/native_with_bias/ctx/ctx_with_bias.bin` — native ctx with decodable bias bytes at 0x19100..0x1B100
- `/tmp/native_with_bias/model.onnx.{bias_fp32,wRaw_i8}.npy` — ground-truth tensors used to verify the fold

## Open questions (not pursued)

- Exact fp16 scale formula — the per-tile constants `(0x4408, 0x4040)` should encode `out_scale × wt_scale × act_scale` and the output zp shift. Not RE'd in detail since V8 prod's simpler format works.
- Whether the per-tile constants are really shared across all 32 channels (verified in this test by per-tensor output scale, but per-channel quant might layout differently).
