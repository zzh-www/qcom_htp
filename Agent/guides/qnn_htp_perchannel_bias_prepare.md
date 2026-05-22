# QNN HTP Per-Channel Bias Prepare

This document records the recovered QNN HTP prepare algorithm for per-channel
Conv1x1 MatMul kernels.  It explains how a quantized DLC Conv bias `B` is
converted into the final HTP `bias_to_vtcm` sidecar effective-bias field, and
how that common prepared value is projected by each observed per-channel
kernel family.

Implementation-level numeric details are in
[docs/qnn_htp_perchannel_bias_prepare_algorithm.md](/home/zzh/work/qcom_htp/docs/qnn_htp_perchannel_bias_prepare_algorithm.md).

The covered families are:

- `u8i8`
- `w4a8_per_channel`
- `w8a16`
- `w4a16_per_channel`

LPBQ and W4 per-group kernels are intentionally out of scope.

## Boundary

Keep the QNN layers separate:

- Source ONNX is a float graph.  `Conv(A, W, B)` has a float `B`
  initializer; bias is not a runtime tensor.
- DLC quantization converts that float `B` into the static quantized Conv bias
  tensor.  In the checked cases, quantized DLC `B` equals generated `bias_q`.
- HTP prepare lowers quantized DLC `B` plus Conv scale metadata into the
  sidecar consumed by `q::ConvLayer.opt.bias_to_vtcm`.
- Device runtime receives activation raw data for the native reference.  Bias,
  scales, and zero-points have already been serialized into the context.

Therefore small integer deltas between DLC `B` and the final sidecar bias are
introduced before device execution.  They are not runtime-input issues, weight
layout issues, activation layout issues, or qairt-quantizer bias bugs.

## Call Chain

The relevant lowering path is visible with the ctxgen gdb helper
[scripts/gdb_dump_u8i8_bias_prepare.py](/home/zzh/work/qcom_htp/scripts/gdb_dump_u8i8_bias_prepare.py):

```text
Conv2d_w_scale(..., B sFxp32, scale float32)
  -> dequantize_bias
  -> find_bias_scale
  -> scale_normalizing
  -> requant_bias
  -> ConvLayer.opt.convert_bias
  -> ConvLayer.opt.adjust_bias
  -> ConvLayer.v73.opt.convert_bias
  -> ConvLayer.opt.bias_scale_shuff
  -> ConvLayer.opt.bias_to_vtcm
```

The arithmetic source is the const-fold/evaluator path before
`bias_to_vtcm`.  `bias_to_vtcm` is the materialized sidecar boundary: final
bottom mapping shows it consumes an `Int32 [1, N/32, 1, 64]` or
`Int32 [1, N/32, 1, 128]` const and produces the VTCM sidecar input for the
native Conv layer.

## Common Prepare Algorithm

All covered per-channel kernels share the same scalar prepare rule before the
family-specific sidecar projection:

```python
global_bias_scale = float32(act_scale) * max(float32(weight_scale))
dequant = float32(DLC_B * global_bias_scale)

find_bias_scale = float32(max(abs(dequant)) * 16.0 / 2**32)
requant_mul = float32(1.0 / find_bias_scale)
expanded = nearbyintf(float32(dequant * requant_mul))

restored = float32(float32(expanded * find_bias_scale) / global_bias_scale)
prepared_bias_i32 = trunc(restored)
```

If `max(abs(dequant)) == 0`, QNN emits all-zero prepared bias.

Important details:

- The scale in `dequantize_bias` is global:
  `act_scale * max(weight_scale)`.  It is not
  `act_scale * weight_scale[channel]`.
- `nearbyintf` is modeled with `np.rint` under ctxgen's default
  round-to-nearest-even environment.
- The restore path must preserve QNN's float32 boundaries:
  multiply by `find_bias_scale`, round to float32, divide by
  `global_bias_scale`, round to float32, then truncate toward zero.
- Collapsing the restore into one algebraic ratio can reintroduce `+/-1`
  mismatches.

Public implementation:

```python
from scripts.qnn_htp_bias_prepare import (
    qnn_htp_perchannel_prepare_bias_stages,
    qnn_htp_perchannel_a8_sidecar_bias_q,
    qnn_htp_perchannel_a16_sidecar_bias_q,
)

stages = qnn_htp_perchannel_prepare_bias_stages(
    dlc_bias_q,
    act_scale,
    weight_scale,
)
prepared_bias_i32 = stages.final_bias_i32
```

The old `qnn_htp_u8i8_*` names remain only as compatibility aliases.

## Effective Bias

The HMX sidecar stores the bias already combined with activation zero-point
folding.  For the covered kernels the activation zero point is `128`.

```python
folded_activation_zp_i32 = -128 * sum_k(weight_q[k, channel])
effective_i32 = folded_activation_zp_i32 + sidecar_bias_i32
```

The family-specific question is only how `prepared_bias_i32` becomes
`sidecar_bias_i32`.

## Kernel Matrix

| Family | Bias scale schema | Sidecar record | Sidecar bias projection | Current validation |
| --- | --- | --- | --- | --- |
| `u8i8` | per-channel W8 scale | 256B per 32 output channels | `sidecar_bias_i32 = prepared_bias_i32` | `normal_random` custom/native sidecar `2048/2048`; same-hardware output exact |
| `w4a8_per_channel` | per-channel W4 scale | 256B per 32 output channels | `effective_i32 = 16 * (-128 * sum_w4 + prepared_bias_i32)` and drain scale divided by `16` | 7/7 correctness cases full 256B sidecar byte-exact |
| `w8a16` | per-channel W8 scale | 512B per 32 output channels | `sidecar_bias_i32 = trunc(prepared_bias_i32 / 256)` | 7/7 native-match cases output exact with generated sidecar |
| `w4a16_per_channel` | per-channel W4 scale | 512B per 32 output channels | `sidecar_bias_i32 = trunc(prepared_bias_i32 / 256)` | 7/7 native-match cases full 512B sidecar byte-exact and output exact |

For current A16 correctness cases,
`trunc(prepared_bias_i32 / 256)` equals `trunc(DLC_B / 256)`.  The
implementation still runs the full common prepare rule so future boundary
cases cannot silently bypass QNN behavior.

## `u8i8`

`u8i8` is the reference custom/native alignment path for this rule.

Inputs:

- Activation: U8, zero point `128`.
- Weight: signed I8 with per-channel scale.
- Bias source: quantized DLC Conv `B` int32.
- Output: U8, so the HMX drain record is 256 bytes per N tile.

Record layout per 32 output channels:

```text
bytes 0..127:
  channel c: uint16 drain_scale_f16, uint16 drain_control
bytes 128..255:
  channel c: int32 effective_i32
```

Bias algorithm:

```python
prepared = qnn_htp_perchannel_prepare_bias_q(dlc_B, act_scale, weight_scale)
effective = -128 * sum_k(weight_i8) + prepared
```

Device evidence for `u8i8/normal_random` at `256x256x256`:

```text
native vs generated bytes: 2048/2048
scale match: 256/256
control match: 256/256
effective delta counts: {'0': 256}
same-hardware custom/native output: 65536/65536, maxdiff=0
```

## `w4a8_per_channel`

`w4a8_per_channel` uses the same common prepared bias as `u8i8`, but projects
the final sidecar into the W4 HMX accumulator domain.

Inputs:

- Activation: U8, zero point `128`.
- Weight: W4 per-channel scale; this section does not cover W4 LPBQ/per-group.
- Bias source: quantized DLC Conv `B` int32.
- Output: U8, so the sidecar record is 256 bytes per N tile.

Bias algorithm:

```python
prepared = qnn_htp_perchannel_prepare_bias_q(dlc_B, act_scale, weight_scale)
effective = 16 * (-128 * sum_k(weight_w4_dequant_codes) + prepared)
```

The U8 drain scale/control fields use
[scripts/qnn_htp_u8_drain.py](/home/zzh/work/qcom_htp/scripts/qnn_htp_u8_drain.py).
This is separate from bias prepare, but it lives in the same 256B record.

Drain scale/control rule:

```python
exact_scale = float32(512) * act_scale * weight_scale[channel] / output_scale / 16
if exact_scale > 65504:
    encoded_scale = float32(exact_scale / 2**32)
else:
    encoded_scale = exact_scale

scale_u16, control_u16 = fp16_quarter_encode(encoded_scale)
```

`control_u16` is `0x0040` for the outer quarters of an fp16 interval and
`0x8040` for the middle half.  The overflow path was confirmed from the
`ConvLayer.opt.bias_scale_shuff` call chain; it is not an empirical clamp.

Validation on context data:

```text
bias_only          bytes=256/256 scale=32/32 ctrl=32/32 eff=32/32
negative_boundary  bytes=256/256 scale=32/32 ctrl=32/32 eff=32/32
normal_random      bytes=256/256 scale=32/32 ctrl=32/32 eff=32/32
positive_boundary  bytes=256/256 scale=32/32 ctrl=32/32 eff=32/32
scale_only         bytes=256/256 scale=32/32 ctrl=32/32 eff=32/32
single_k_impulse   bytes=256/256 scale=32/32 ctrl=32/32 eff=32/32
zp_neutral         bytes=256/256 scale=32/32 ctrl=32/32 eff=32/32
```

## `w8a16`

`w8a16` uses the common prepare rule, then projects the result for the U16/A16
native record.

Inputs:

- Activation: U16/A16 path with the same folded zero-point convention in the
  decoded sidecar.
- Weight: signed I8 with per-channel scale.
- Bias source: quantized DLC Conv `B` int32.
- Output: U16, so the native sidecar record is 512 bytes per N tile.

Bias algorithm:

```python
prepared = qnn_htp_perchannel_prepare_bias_q(dlc_B, act_scale, weight_scale)
sidecar_bias = trunc(prepared / 256.0)
effective = -128 * sum_k(weight_i8) + sidecar_bias
```

The decoded native A16 W8 record uses family-specific scale/control constants.
Those constants are not the per-channel bias prepare rule.  The currently
validated arithmetic field is the effective int32 field.

Validation:

```text
w8a16 native-match cases:
  normal_random, zp_neutral, positive_boundary, negative_boundary, single_k_impulse, bias_only, scale_only
  custom sidecar source: native final drain/control bytes plus generated effective fields
  same-hardware output: 65536/65536 for each case
```

Do not use QNN Native's final A16 sidecar as the byte-level custom-wrapper
input.  In `zp_neutral`, that native final effective-bias correction shifts
channel 243 by +256 in the custom HMX body.  The native final control/drain
bytes are still useful: they remove the small +/-1 rounding differences seen
when the fully generated sidecar is used for normal random data.

## `w4a16_per_channel`

`w4a16_per_channel` follows the same A16 projection as `w8a16`, with W4
per-channel weights.  It is distinct from W4 LPBQ/per-group, whose scale
schema and sidecar handling are intentionally not covered here.

Inputs:

- Activation: U16/A16 path with the same folded zero-point convention in the
  decoded sidecar.
- Weight: W4 per-channel scale.
- Bias source: quantized DLC Conv `B` int32.
- Output: U16, so the native sidecar record is 512 bytes per N tile.

Bias algorithm:

```python
prepared = qnn_htp_perchannel_prepare_bias_q(dlc_B, act_scale, weight_scale)
sidecar_bias = trunc(prepared / 256.0)
effective = -128 * sum_k(weight_w4_dequant_codes) + sidecar_bias
```

W4A16 uses the same normalized two-stage A16 drain scale path as W8A16:

```python
max_w = max(float32(weight_scale))
normalized = float32(weight_scale / max_w)
max_channel_scale = float32(float32(max_w * act_scale) / output_scale)
drain_scale = float32(normalized * max_channel_scale)
```

The four A16 control/drain words are then generated with the same
`qnn_htp_w8a16_drain_control_words` packing helper used by W8A16.  Fixed A16
constants are insufficient: they can leave the effective-bias field correct
while the 512-byte sidecar control/drain bytes still differ from QNN Native.

Validation:

```text
w4a16_per_channel native-match cases:
  normal_random, zp_neutral, positive_boundary, negative_boundary, single_k_impulse, bias_only, scale_only
  generated sidecar: 4096/4096 bytes for each case
  control bytes: 2048/2048 for each case
  effective fields: 256/256 for each case
  same-hardware output: 65536/65536 for each case
```

## Verification Commands

Syntax and static checks:

```bash
uv run python -m py_compile \
  scripts/qnn_htp_bias_prepare.py \
  scripts/qnn_htp_u8_drain.py \
  scripts/analyze_u8i8_native_bias_record.py \
  scripts/dump_u8i8_bias_prepare_stages.py
```

Stage dump validation:

```bash
uv run python scripts/dump_u8i8_bias_prepare_stages.py \
  --case-dir /tmp/qcom_htp_u8i8_prepare_impl_check/output_u8i8_native_match_normal_random_ci/cases/u8i8/normal_random \
  --native-dir /tmp/qcom_htp_u8i8_prepare_impl_check/output_u8i8_native_match_normal_random_ci/native_normal_random \
  --gdb-dump-dir /tmp/qcom_htp_u8i8_prepare_impl_check/output_u8i8_native_match_normal_random_ci/analysis/gdb_requant_bias_loop \
  --out-dir /tmp/qcom_htp_u8i8_prepare_impl_check/output_u8i8_native_match_normal_random_ci/analysis/prepare_stage_recheck_formula \
  --qnn-sdk-root tools/qnn-sdk
```

Expected summary:

```text
gdb dequant vs global-max model: 256/256 words, maxabs=0
bias_scale_shuff_trunc vs native sidecar bias_q: {'0': 256}
native effective vs expected: {'0': 256}
```

Same-hardware custom/native gate without native sidecar injection:

```bash
OUT_ROOT=/tmp/qcom_htp_u8i8_formula_gate USE_NATIVE_BIAS_RECORD=0 \
  BUILD_PACKAGES=0 DEVICE=oneplus CHAIN=1 CASE_NAME=normal_random \
  M=256 K=256 N=256 bash scripts/run_u8i8_python_case_custom_native_match.sh
```

Expected result:

```text
native vs custom bytes: 2048/2048
native vs custom scale/control bytes: 1024/1024
native vs custom effective bytes: 1024/1024
same-hardware custom/native output: 65536/65536, maxdiff=0
```

## Public Files

- [scripts/qnn_htp_bias_prepare.py](/home/zzh/work/qcom_htp/scripts/qnn_htp_bias_prepare.py):
  common prepare algorithm and A8/A16 projection helpers.
- [scripts/qnn_htp_u8_drain.py](/home/zzh/work/qcom_htp/scripts/qnn_htp_u8_drain.py):
  U8 drain scale/control encoding, including the A8 overflow boundary.
- [scripts/analyze_u8i8_native_bias_record.py](/home/zzh/work/qcom_htp/scripts/analyze_u8i8_native_bias_record.py):
  context-sidecar extractor and comparator for U8 records.
- [scripts/dump_u8i8_bias_prepare_stages.py](/home/zzh/work/qcom_htp/scripts/dump_u8i8_bias_prepare_stages.py):
  diagnostic stage dump for comparing generated arrays, DLC `B`, gdb evidence,
  and final sidecar fields.
- [example/qnn_hmx_matmul_u8i8/bias_prepare_probe/](/home/zzh/work/qcom_htp/example/qnn_hmx_matmul_u8i8/bias_prepare_probe/):
  reproducible u8i8 probe for the DLC-to-context boundary.

## Non-Goals

- This is not a runtime bias ABI.
- This does not describe W4 LPBQ or W4 per-group bias handling.
- This does not replace QNN Native-vs-Python tolerance policy.  Cross-hardware
  or Python/native checks may allow small differences, but same-hardware
  custom/native and handwritten/custom comparisons remain exact-output gates.
