---
name: Session 2026-04-27 — V8 C8 framework alignment unblocked + native bias_to_vtcm decoded
description: Single-session log of the 2026-04-27 work that (1) cracked the err 6006 blocker for V8 C8 Phase 2 by folding bias into the wt static, (2) decoded native ConvLayer's bias_to_vtcm format byte-exact via byte-diff with a known-bias native MatMul. Detailed spec of native bias VTCM layout for the next session to either replicate (byte-1:1) or skip (functional equivalence via V8 prod's centered-act path).
type: project
---

# Session 2026-04-27 summary

## Where we started this session

V8 C8 Phase 2 was blocked at device runtime err 6006 (`Dma execution failed on the skel side`) when we tried to declare BbbKMajor's in[0] as Crouton_8 + Indirect — the auto-inserted `q::ConvLayer.opt.weights_to_vtcm × 3` for our wt+bias+scratch statics would fail before our kernel ever ran.

## Where we ended up

**Working device path (256³ NOOP body):**

```
ONNX (act_raw + 1 combined wt_bias static)
  ↓
qairt-converter → DLC
  ↓
qnn-context-binary-generator with our op-pkg
  ↓
ctx-binary contains 4-node lowered graph:
  q::*InputSlice → q::ForceFormat_Crouton → q::ConvLayer.opt.weights_to_vtcm → BbbKMajor
  ↓
device: "Finished Executing Graphs"  ✓ no err 6006
```

This is the **structural alignment** with native ConvLayer (4 of native's 8 nodes, missing the output untile and the separate bias_to_vtcm). Activation flows through the SAME `q::ForceFormat_Crouton` auto-insertion native uses.

## The two pieces of new knowledge unlocked this session

### 1. err 6006 root cause — slot-2 dispatch trap

When in[0]=Crouton_8 triggers ConvLayer-like classification in QNN's host prepare, each static initializer slot picks a specialized `weights_to_vtcm` variant by SLOT POSITION (not by tensor metadata):

- slot 1 (wt slot) → weight-style descriptor → works for `[1, K_t, N_t, 1024]`-shaped pre-packed weight
- slot 2 (bias slot) → bias-slot descriptor that doesn't match our `[1, 1, N/32, 128]` u16 bias → DMA reads go OOB → err 6006
- slot 3 (scratch slot) → similar bias-slot-style misfit

QHPI v1 sig has NO role/kind field to override this. early_rewrite path is also closed: `qhpi_op_create()` rejects internal `q::*` op names ("invalid operator name, internal package prefix is not allowed") — only `q::QNN_*` public ops accepted, and that whitelist contains no DMA helpers.

### 2. Workaround that works — combined wt+bias single static

Fold bias bytes into the wt static buffer; have only ONE static input. The slot-1 weights_to_vtcm path works correctly for any `[1, K_t', N_t, 1024]`-shaped buffer, regardless of what's stored in those bytes.

```python
# 256³ example
extra_ktiles = ceil(bias_size / (N_t * 1024))           # 1 for 256³
total_ktiles = K_t + extra_ktiles                        # 9 for 256³

combined = np.zeros((1, total_ktiles, N_t, 1024), np.int8)
combined_flat[0:wt_size]                  = pre_packed_native_weight_bytes   # 65536 bytes
combined_flat[wt_size:wt_size+bias_size]  = bias_bytes_in_some_format        # 2048 bytes
```

Op sig: 2 inputs (act + wt_bias), all TCM_Only, single auto-DMA → bias lands in VTCM at offset wt_size from the start of the wt VTCM region. End-state functionally equivalent to native (both wt and bias VTCM-resident, accessible via `mxmem`).

# Native bias_to_vtcm — byte-decoded format

This is the **load-bearing new finding** for the next session. We RE'd native bias_to_vtcm by feeding a known-monotonic bias and byte-diffing the produced ctx-binary.

## Test setup (reproducible)

```bash
# /tmp/gen_native_with_bias.py builds a MatMul with explicit fp32 bias
# bias_fp32 = [0.001, 0.002, …, 0.256]  (256 channels)
# random i8 weight in [-127, 127]
# random u8 act in [0, 255]
# qairt-converter → DLC (bias quantized to int32, 32-bit-symmetric)
# qnn-context-binary-generator → ctx_with_bias.bin

# Outputs preserved at /tmp/native_with_bias/
```

## Decoded VTCM layout

Per N-tile (32 output channels, **256 bytes** in VTCM):

```
[byte 0..127]    32 × (fp16 scale, fp16 baseline_zp_encoded)   -- per-tile constants
                 All 32 channels in a tile share the SAME 4-byte (scale, baseline_zp).
                 Only varies between tiles if quant scale is per-channel; per-tensor → constant across all tiles.

[byte 128..255]  32 × int32 effective_bias[c]                    -- per-output-channel folded value
                 effective_int32[c] = -act_zp × Σ_k W[k,c] + bias_quantized[c]
                 i.e., includes the weight-column-sum × act_zp PRECOMPUTED at host prepare time.
```

For N=256: 256 bytes/tile × 8 tiles = 2048 bytes total.

## The fold formula (256/256 channels match within ±1 LSB)

```
sum_w[c]            = Σ_k W[k, c]                                # per output channel
bias_quantized[c]   = round(bias_fp32[c] / bias_scale)
                      where bias_scale ≈ wt_scale × act_scale (fitted = 6.31e-5 in our test)
effective_int32[c]  = -act_zp × sum_w[c] + bias_quantized[c]     # the fold

# for u8 input with zp=128:
# effective_int32[c]  = -128 × sum_w[c] + bias_quantized[c]
```

Verified: across 256 channels of our test, the int32 part of native ctx VTCM at `0x19180+t*256 .. 0x19200+t*256` matches `effective_int32[c]` with at most ±1 LSB diff (rounding noise). 8 N-tiles, all match.

## Why native folds Σwt into the bias

Quantized matmul algebraic identity:

```
acc[m,n] = Σ_k (a[m,k] - act_zp) × w[k,n] + bias[n]
        = Σ_k a[m,k] × w[k,n]  +  ( -act_zp × Σ_k w[k,n] + bias[n] )
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   precomputed once at host prepare = effective_bias[n]
```

By absorbing the `-act_zp × Σwt` term into the bias, native lets HMX `mxmem` consume **raw u8 activation** (no per-element zp subtraction needed). The accumulator gets initialized with the folded effective_bias and HMX does pure `Σ a×w` MACs on top of it.

## The per-tile fp16 pair (bytes 0..127 of each 256-byte tile)

In our test, all 32 channels of tile 0 had byte pattern `08 44 40 40` = (u16 0x4408, u16 0x4040):

- `0x4408` fp16 = 4.03125 (probably encodes `output_scale × wt_scale × act_scale × 512` per `:cm:sat.ub` semantics)
- `0x4040` fp16 = 2.125 (probably encodes the output zp baseline / 0x4000 + per-tile shift)

Exact encoding formula not RE'd in this session — V8 prod's simpler fp16-pair format `(fp16(512×scale), 0x4000)` works because V8 prod uses centered activation (subtracts act_zp upstream). Native's per-tile pair encodes both scale AND post-fold output-zp shift, slightly different.

## V8 production's alternative (what we already do)

V8 production avoids the fold by **subtracting act_zp from activation in `PackActivationU8RowMajor`** (centered activation upstream). Then:

```
acc[m,n] = Σ_k centered_act[m,k] × w[k,n] + bias[n]
```

Bias VTCM format becomes simpler — just per-channel `(fp16 scale, fp16 baseline=0x4000)`, 4 bytes/channel × 32 channels = 128 bytes/tile. **Half the bytes per tile vs native** because V8 doesn't need the fold's int32 part.

Bit-exact equivalent matmul output (V8 prod 32³–1024³ already verified earlier).

# Current op-pkg state at end-of-session

## Files modified (under `V9_C8_ALIGNMENT_TEST` flag)

- `example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp`
  - `BbbKMajor` op registered with 2-input sig (act + wt_bias), kernel `hmx_matmul_v9_kernel`
  - act sig: `Crouton_8 + Indirect + TCM_Only`
  - wt_bias sig: `Flat4 + Direct + TCM_Only`
  - kernel body: `V9_KERNEL_NOOP` / `V9_C8_ALIGNMENT_TEST` branch writes 16-byte marker prologue + zeroes the rest. **No actual matmul yet.**

- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/MatMulV8Package.xml`
  - BbbKMajor: 2 inputs declared with shapes `[1, M/32, 32, K]` and `[1, (K+bias_K)/32, N/32, 1024]`

- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_v8c8_test.py`
  - Builds combined `[1, K_t+extra, N_t, 1024]` initializer (256³: `[1, 9, 8, 1024]`)
  - Weight bytes (offset 0..wt_size) in pre-packed native ConvLayer layout
  - Bias bytes (offset wt_size..wt_size+bias_size) in V8 prod's u16 fp16-pair format
    (Note: this is the V8-prod-simple format, NOT the native fold format)

- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_out/.../ConverterOpPackage.cpp`
  - BbbKMajorShapeInference updated for 2-input case

- `example/hmx_matmul_phase3/build_x86.sh`
  - Added `${EXTRA_DEFS:-}` propagation so host metadata .so picks up sig changes when EXTRA_DEFS is set

- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v8c8_phase2.sh`
  - Test runner: gen → convert → ctxgen → push → device run → logcat capture → output marker decode

## Verified working (256³)

- ctxgen produces 4 nodes: InputSlice + ForceFormat_Crouton + weights_to_vtcm × 1 + BbbKMajor
- device run "Finished Executing Graphs"
- output 65536 bytes (fp32-dequantized to 262144 by qnn-net-run)
- kernel writes 16-byte marker (verified ~14 nonzero bytes in dequantized output)

## NOT done yet

| Work item | Why it matters | Effort |
|---|---|---|
| **Real HMX kernel body** (matmul calculation) | Currently NOOP — output is zeros + marker | medium |
| **Output untile** (tile-major → row-major) | Native has q::ForceFormat_Flat + Reshape × 2; we still emit `[1, M/32, N/32, 1024]` | small |
| **int32 bias ABI alignment with native** | We pre-bake fp16-pair in gen_v8c8_test.py instead of accepting int32 | small-medium |
| **Bit-exact validation vs native** | Need real kernel first | medium |
| **Shape-adaptive (32³–4096³)** | Currently 256³ hardcoded test | small |
| **Perf comparison** | Need real kernel first | small once kernel works |

# Next-session pickup plan

## Recommended order

### 1. Wire up real HMX kernel body (highest value)

Take V8 production's mmv8 inner loop (already proven bit-exact in V8 prod) and adapt it for the new op:

- Input layout differences from V8 prod:
  - act: now Crouton_8-blocked `[1, M/32, 32, K]` (after ForceFormat_Crouton). V8 prod uses tile-array `[1, M/32, K/32, 1024]`. Need to figure out HMX `mxmem` access pattern for Crouton_8 layout.
  - wt: `[1, K_t + extra, N_t, 1024]` — same 1024-byte tile layout as V8 prod's PackWeightToHmxTileV3 output but with extra K-tiles holding bias bytes.
  - bias: not a separate tensor; lives at offset `wt_size` of the wt_bias VTCM region. Read as `(uint16_t*)(wt_bias_vtcm + wt_size)`, indexed `bias_n = bias_vtcm + nt * 128` (V8 prod format).
  - output: still tile-major `[1, M/32, N/32, 1024]` (V8 prod tile layout).
- Reuse V8 prod's HMX MAC + `:cm:sat.ub` pattern verbatim — the bias format we use is V8 prod's fp16-pair simple format, not the native fold format. So V8's `:cm:sat.ub` semantics should produce correct output.
- ⚠ Subtlety: the bias FORMAT decision (V8 prod simple vs native fold) directly determines what bytes need to go into the combined static and what `:cm:sat.ub` will compute. Decide this before writing the kernel.

### 2. Decide the bias format strategy

Two paths, **pick one before kernel work**:

**(A) Stay with V8 prod simple bias format** (current code, easier path):
- Bias bytes in combined static: `(fp16(512 × output_scale_per_channel), 0x4000)` per channel
- HMX kernel must consume centered activation (act - 128). Need to verify q::ForceFormat_Crouton output IS centered. **Suspicion: it likely isn't — Crouton just reformats layout, doesn't subtract zp.** So we'd need to subtract zp ourselves — but where? In our op kernel's start? That defeats the "let QNN do the work" goal.
- Or: declare act tensor as signed int8 in QHPI sig (forces upstream to deliver signed bytes, equivalent to subtracting 128).
- Risk: may not produce correct numerical result without further work.

**(B) Switch to native fold bias format** (1:1 native, more work):
- Bias bytes in combined static: per-tile (fp16 scale, fp16 baseline) + per-channel int32 effective_bias
- Effective_bias = `-act_zp × Σ_k W[k,c] + bias_q[c]` — host-side fold computed in `gen_v8c8_test.py` (we have access to weight bytes there since we pre-pack them anyway).
- HMX kernel reads (fp16, fp16) per-tile + int32 per-channel via the combined VTCM region. Need to figure out the mxmem access pattern (probably TWO mxmem reads per HMX kernel iteration: one for fp16 pair, one for int32 baseline?).
- Pro: 1:1 with native VTCM bytes. ABI-compatible with int32 user-facing bias from qairt-converter.
- Con: kernel access pattern more complex; requires more silicon-level RE of HMX bias-load instruction variants.

**Recommendation**: start with (A) since V8 prod already works. If output is wrong, debug-step the act-zp issue. Switching to (B) is reversible.

### 3. Output untile

Add an internal tile→row-major op or use `q::ForceFormat_Flat` (if early_rewrite can insert it — public op?). Native uses 2 reshapes + ForceFormat_Flat. Not on critical path for kernel correctness — output stays tile-major for now.

### 4. ABI cleanup (int32 user-facing bias)

If we want users to feed standard ONNX with int32 bias (matching qairt-converter convention), do the int32 → (whatever VTCM format we picked) conversion inside `gen_v8c8_test.py` or a host-side pre-process hook. We have all the weight bytes there for path (B).

# Critical file paths for pickup

- Active code: `example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp` (search for `V9_C8_ALIGNMENT_TEST`)
- Op-pkg XML: `example/hmx_matmul_phase3/standard_flow/phaseB_v8/MatMulV8Package.xml`
- Test gen: `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_v8c8_test.py`
- Test runner: `example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v8c8_phase2.sh`
- Build: `EXTRA_DEFS=-DV9_C8_ALIGNMENT_TEST bash build.sh && bash build_x86.sh`
- V8 production reference (mmv8 HMX inner loop, bias format, kernel structure): `example/hmx_matmul_phase3/src/HmxMatMulV8Op.cpp`

# Native bias byte-decode artefacts (for next-session re-verification)

If the next session needs to re-verify the native bias formula or check additional edge cases:

```bash
# Reproduce native bias VTCM bytes
python /tmp/gen_native_with_bias.py --out /tmp/native_with_bias/model.onnx
cd /tmp/native_with_bias
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter -i model.onnx \
    --quantization_overrides quant_overrides.json -o model.dlc
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path model.dlc --binary_file ctx_with_bias --output_dir ctx \
    --config_file htp_config.json

# Decode bias bytes at 0x19180..0x19200 (tile 0, int32 fold part)
python3 -c "
import numpy as np
ctx = open('/tmp/native_with_bias/ctx/ctx_with_bias.bin','rb').read()
W = np.load('/tmp/native_with_bias/model.onnx.wRaw_i8.npy').astype(np.int32)
bias_fp32 = np.arange(1, 257, dtype=np.float32) * 0.001
bias_q = (bias_fp32 / 6.31e-5).round().astype(np.int32)
expected = -128 * W.sum(axis=0) + bias_q
actual = np.zeros(256, np.int32)
for t in range(8):
    actual[t*32:(t+1)*32] = np.frombuffer(ctx[0x19180+t*256:0x19180+t*256+128], np.int32)
print('match:', (np.abs(actual - expected) <= 1).sum(), '/256')
"
# Should print: match: 256 /256
```

# Where to look in memory

- **`Agent/v8_c8_alignment_phase2_BREAKTHROUGH_2026-04-27.md`** — original session-1 breakthrough (mixed memloc workaround, before bisect)
- **`Agent/v8_c8_bias_combined_static_2026-04-27.md`** — combined wt+bias static (current production approach)
- **`Agent/qnn_re/bias_to_vtcm_decoded_2026-04-27.md`** — native bias byte-decoded
- **`Agent/qnn_re/op_registration_slot_semantics_2026-04-27.md`** — why we can't use early_rewrite for bias
- **`Agent/qnn_re/weights_to_vtcm_RE_2026-04-27.md`** — weights_to_vtcm RE (cost tables not implementation)
- **`Agent/v8_c8_alignment_phase1_2026-04-27.md`** — Phase 1 work (Crouton_8 sig discovery)

# One-liner status

**V8 C8 framework is unblocked, device runs cleanly, bias path lands in VTCM via combined-static workaround (V8 prod's simple fp16-pair format). Native bias byte-format decoded as weight-aware fold. Real kernel body still needs to be wired up; format choice (V8 prod simple vs native fold) is the gating decision before kernel work.**
