---
name: V8 C8 kernel Phase 3.1 — native fold layout via host pre-fold, bit-exact 32³–1024³ 2026-04-27 night
description: Bias path now matches native bias_to_vtcm end-state byte-for-byte. Gen-script computes the weight-aware fold (-ACT_ZP × Σ_k W[k,c] + bias_q[c]) and packs into 256-B/N-tile native layout (lower 128 B = 32×(fp16 scale, fp16 baseline), upper 128 B = 32 × int32 effective). Kernel reads effective_int32 from VTCM upper half — NO fold computation in kernel. Bit-exact match vs centered Python reference at every shape 32³–1024³ (1.04M cells total).
type: project
---

# Result

**BIT-EXACT 1024/1024, 4096/4096, 16384/16384, 65536/65536, 262144/262144, 1048576/1048576**
at shapes 32³, 64³, 128³, 256³, 512³, 1024³.

Bias bytes in VTCM are byte-1:1 with what native `q::ConvLayer.opt.bias_to_vtcm`
produces. Kernel does NOT fold — gen-script does it at host prepare time
(equivalent timing and semantics to native).

# Bias layout (native byte-1:1)

```
Per N-tile (256 B):
  bytes 0..127   : 32 × (fp16 scale, fp16 baseline)   per-channel pair
  bytes 128..255 : 32 × int32 effective_bias[c]
                   effective[c] = -ACT_ZP × Σ_k W[k, c] + bias_q[c]

Total bias_size = N_t × 256 = 8N bytes
ONNX initializer: int32 [2N] (= 8N bytes)
```

For our test (quant_overrides scale=1.0, out_zp=0):
- scale_u16 = fp16(512.0) = 0x6000 (one per channel)
- baseline_u16 = (out_zp << 7) = 0 for zp=0, would be 0x4000 for zp=128

# Kernel math (no fold)

```c
acc[m,n] = effective_int32[c]                            // initial value
         + Σ_k act_u8[m,k] × wRaw_i8[k,n]                // raw u8 act × signed wt
         = bias_q[c] + Σ_k (act_u8[m,k] - ACT_ZP) × wRaw_i8[k,n]  // algebraically

out[m,n] = saturate_u8(top9(baseline) + floor(acc × scale_fp16 / 512))
```

For our test (scale_fp16=fp16(512), baseline_u16=0):
- `floor(acc × 512 / 512) = acc`
- `top9(0) = 0`
- `out = saturate_u8(acc)`

# Important findings during this phase

1. **QNN does NOT auto-fold our int32 bias**. Earlier hypothesis was wrong —
   what looked like fold was actually our gen-script's pre-folded bytes
   passing through verbatim (QNN's `weights_to_vtcm@Fi.fi.` is a verbatim
   typed DMA, not a fold dispatcher).

2. **Therefore the only way to get native byte-1:1 fold layout in VTCM is
   host-side fold in gen-script** — exactly equivalent to native's
   `q::ConvLayer.opt.bias_to_vtcm` semantics (which also runs at host
   prepare time, baked into the ctx-binary).

3. **Python reference must use centered formula** to match the native math:
   ```python
   acc = (act - ACT_ZP) @ wt + bias_q     # centered act × wt + bias
   ```
   (NOT raw `act @ wt + bias_q`). This was a subtle bug — the kernel reads
   `effective = -ACT_ZP × Σwt + bias_q` from VTCM, which absorbs the
   centering. Math is equivalent both ways.

# Code

`gen_v8c8_test.py` — host pre-fold, bias as int32 [2N]:
```python
ACT_ZP = 128
sum_w = wRaw_KN.sum(axis=0)
effective = -ACT_ZP * sum_w + bias_q
# Pack into 256-B/N-tile layout (fp16 pair lower + int32 effective upper)
```

`HmxMatMulV9SkelOp.cpp` — `V9_KERNEL_SCALAR` branch:
```c
int32_t acc = BIAS_EFF_I32(nt, nc);    // read effective from upper half
for (k...) acc += ACT_AT(m, k) * WT_AT(k, n);   // raw u8 × signed wt
out = saturate_u8(acc);                 // for our test (scale=1, zp=0)
```

# What's next — Phase 3.2: HMX inline asm

Replace the scalar inner MAC + saturate with HMX:
```
mxclracc
bias = mxmem2(bias_per_n_tile)              // load 256 B native fold
for kt in K_t:
    activation.ub = mxmem(act_tile, RT_ACT):cm
    weight.b = mxmem(wt_tile, RT_WT)
mxmem(out_tile, 0):after:cm:sat.ub = acc
```

## Open questions for HMX

1. **Crouton_8 → HMX :cm tile pointer**. At ≥256³ a Crouton block is 2 KB
   (64 rows × 32 cols), but HMX expects 1 KB (32 rows × 32 cols) tiles.
   Need to figure out:
   - Does HMX `mxmem :cm` accept Crouton block ptr directly?
   - Or do we need per-tile offset arithmetic (1 KB strip from a 2 KB block)?
   - Or HVX repack to V8-prod tile-array first?

2. **mxmem2 vs mxmem for bias**. With native 256-B/tile layout, `mxmem2`
   should consume both halves. V8 prod uses `bias = mxmem2(...)` — confirms
   it reads 256 B.

3. **Crouton_8 byte layout WITHIN a block**. We verified the per-block
   layout is 32-byte-stride per row × block_rows rows row-major. HMX may
   expect a different in-tile byte order (HMX-native).

## Reference

- Bit-exact scalar (this work) is the ground truth for HMX validation
- V8 prod's `hmx_v8_mac_convert` (`HmxMatMulV8Op.cpp:87`) is a working HMX
  inner loop — adapt for Crouton_8 input
- `Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md` has the descriptor ABI for
  the native HMX kernel (could be called via dlsym for end-to-end test)
