# QNN HTP Per-Channel Bias Prepare Algorithm

This document is the implementation-oriented companion to
`Agent/guides/qnn_htp_perchannel_bias_prepare.md`.  It focuses on the exact
numeric algorithm needed to reproduce QNN HTP's per-channel Conv1x1
`bias_to_vtcm` sidecar bytes.

The implementation in this repository is:

- `scripts/qnn_htp_bias_prepare.py`: common bias prepare and A8/A16 projection.
- `scripts/qnn_htp_u8_drain.py`: U8 drain scale/control encoding.

## Inputs

The common prepare function takes only values that exist before HTP prepare:

```text
bias_q_i32     int32[N]     quantized DLC Conv B
act_scale      float32      activation encoding scale
weight_scale   float32[N]   per-channel weight encoding scale
```

The function does not take runtime activation data, output scale, activation
zero point, or packed HMX weights.  Those values belong to later sidecar
packing or accumulator-drain stages.

Use the DLC quantized `B` tensor as `bias_q_i32`.  Do not pass the original
float ONNX bias and do not re-quantize bias at runtime.

## Outputs

The common algorithm returns:

```text
prepared_bias_i32 int32[N]
```

This is not always the exact value stored in the final record.  The final
record uses a family-specific projection:

```text
u8i8:                   sidecar_bias_i32 = prepared_bias_i32
w4a8_per_channel:       effective_i32 uses 16 * (-128 * sum_w4 + prepared_bias_i32)
A16/U16-output kernels: sidecar_bias_i32 = trunc(prepared_bias_i32 / 256)
```

The final HMX effective-bias field then folds activation zero point:

```text
effective_i32 = -128 * sum_k(weight_q[k, channel]) + sidecar_bias_i32
```

For W4 kernels, `weight_q` means the signed logical W4 values after decoding
the 4-bit payload into the kernel's integer domain.

## Exact Common Algorithm

The important implementation rule is that QNN rounds at several float32
boundaries.  Keep those boundaries explicit.

Reference implementation:

```python
import numpy as np


def prepare_bias_i32(
    bias_q_i32: np.ndarray,
    act_scale: np.float32 | float,
    weight_scale: np.ndarray,
) -> np.ndarray:
    bias_q = bias_q_i32.astype(np.int32, copy=False)

    global_bias_scale = (
        np.float32(act_scale) * np.max(weight_scale.astype(np.float32))
    ).astype(np.float32)

    dequant = (bias_q.astype(np.float32) * global_bias_scale).astype(np.float32)
    max_abs = np.max(np.abs(dequant).astype(np.float32)).astype(np.float32)
    if max_abs == np.float32(0.0):
        return np.zeros_like(bias_q, dtype=np.int32)

    find_bias_scale = (
        max_abs * np.float32(16.0) / np.float32(2.0**32)
    ).astype(np.float32)

    requant_mul = (np.float32(1.0) / find_bias_scale).astype(np.float32)
    pre_nearby = (dequant * requant_mul).astype(np.float32)

    expanded = np.rint(pre_nearby)
    expanded = np.clip(
        expanded,
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    ).astype(np.int32)

    restored = (
        (expanded.astype(np.float32) * find_bias_scale).astype(np.float32)
        / global_bias_scale
    ).astype(np.float32)

    return np.trunc(restored).astype(np.int32)
```

The repository implementation additionally returns intermediate arrays through
`qnn_htp_perchannel_prepare_bias_stages`.

## Stage Details

### `global_bias_scale`

QNN uses one scalar for the whole bias vector:

```python
global_bias_scale = float32(act_scale) * max(float32(weight_scale))
```

It does not use per-channel `act_scale * weight_scale[channel]` here.  This is
the main trap when trying to reproduce native sidecar bias from normal
quantization formulas.

### `dequant`

QNN dequantizes DLC bias with the global scalar:

```python
dequant[channel] = float32(bias_q[channel] * global_bias_scale)
```

The multiplication is represented as float32.  The resulting vector is the
input to `find_bias_scale`.

### `find_bias_scale`

QNN finds the largest absolute dequantized value, then scales it down:

```python
find_bias_scale = float32(max(abs(dequant)) * 16 / 2**32)
```

Equivalent:

```python
find_bias_scale = float32(max(abs(dequant)) / 2**28)
```

When `max(abs(dequant)) == 0`, QNN's effective result is all-zero prepared
bias.  Do not divide by zero to get NaNs and then clean them up later.

### `requant_bias`

The active ctxgen const-fold loop behaves like:

```python
expanded = nearbyintf(float32(dequant * float32(1 / find_bias_scale)))
expanded = clamp_to_int32(expanded)
```

`nearbyintf` follows the process default round-to-nearest-even environment.
`np.rint` matches the observed outputs in the checked ctxgen runs.

### `bias_scale_shuff` restore

The final sidecar-bias value is not produced by a single simplified ratio.
QNN restores through float32 stages:

```python
tmp = float32(expanded * find_bias_scale)
restored = float32(tmp / global_bias_scale)
prepared_bias_i32 = trunc(restored)
```

The `trunc` is toward zero.  Replacing this with `round`, `floor`, or a fused
double-precision expression can change boundary channels by `+/-1`.

## A8 Projection

U8-output kernels currently covered:

- `u8i8`
- `w4a8_per_channel`

`u8i8` projection:

```python
sidecar_bias_i32 = prepared_bias_i32
effective_i32 = -128 * sum_k(weight_i8[k, channel]) + sidecar_bias_i32
```

`w4a8_per_channel` projection:

```python
sidecar_bias_i32 = prepared_bias_i32
effective_i32 = 16 * (-128 * sum_k(weight_w4[k, channel]) + sidecar_bias_i32)
drain_exact_scale = float32(512) * act_scale * weight_scale / output_scale / 16
```

Record shape:

```text
Int32 [1, N/32, 1, 64]
256 bytes per 32 output channels
```

Byte layout per 32-channel tile:

```text
0x000..0x07f  32 pairs of uint16 drain_scale_f16, uint16 drain_control
0x080..0x0ff  32 int32 effective_i32 values
```

Implementation entrypoint:

```python
from scripts.qnn_htp_bias_prepare import qnn_htp_perchannel_a8_sidecar_bias_q

sidecar_bias_i32 = qnn_htp_perchannel_a8_sidecar_bias_q(
    dlc_bias_q,
    act_scale,
    weight_scale,
)
```

## A16 Projection

A16/U16-output kernels currently covered:

- `w8a16`
- `w4a16_per_channel`

Projection:

```python
sidecar_bias_i32 = trunc(prepared_bias_i32 / 256.0)
effective_i32 = -128 * sum_k(weight_q[k, channel]) + sidecar_bias_i32
```

Record shape:

```text
Int32 [1, N/32, 1, 128]
512 bytes per 32 output channels
```

Implementation entrypoint:

```python
from scripts.qnn_htp_bias_prepare import qnn_htp_perchannel_a16_sidecar_bias_q

sidecar_bias_i32 = qnn_htp_perchannel_a16_sidecar_bias_q(
    dlc_bias_q,
    act_scale,
    weight_scale,
)
```

In the current correctness cases, this projection equals
`trunc(DLC_B / 256)`, but that is not the implementation contract.  Keep the
full common prepare path so future cases that cross a boundary follow QNN.

## U8 Drain Scale/Control

The U8 drain scale/control pair is separate from bias prepare but stored in the
same A8 256-byte record.  It depends on output scale and per-channel
`weight_scale`.

Inputs:

```text
act_scale       float32
weight_scale    float32[N]
output_scale    float32
```

Exact scale:

```python
exact_scale = float32(512) * act_scale * weight_scale[channel] / output_scale
```

Overflow normalization:

```python
if exact_scale > 65504:
    encoded_scale = float32(exact_scale / 2**32)
else:
    encoded_scale = exact_scale
```

Then encode `encoded_scale` against its enclosing fp16 interval:

```python
nearest = float16(encoded_scale)
lower = previous_fp16(nearest) if float32(nearest) > encoded_scale else nearest
upper = next_fp16(lower)
frac = (encoded_scale - float32(lower)) / (float32(upper) - float32(lower))

scale_u16 = upper_bits if frac >= 0.75 else lower_bits
control_u16 = 0x8040 if 0.25 <= frac < 0.75 else 0x0040
```

Implementation entrypoint:

```python
from scripts.qnn_htp_u8_drain import qnn_htp_u8_drain_scale_control

scale_u16, control_u16 = qnn_htp_u8_drain_scale_control(exact_scale)
```

This overflow behavior was recovered from the `ConvLayer.opt.bias_scale_shuff`
ctxgen path.  Do not use `np.float16(exact_scale)` saturation to `0x7bff` for
the overflow case.

## Per-Kernel Packing Recipes

### `u8i8`

```python
scale_u16, control_u16 = qnn_htp_u8_drain_scale_control(
    float32(512) * act_scale * weight_scale / output_scale
)
sidecar_bias = qnn_htp_perchannel_a8_sidecar_bias_q(dlc_B, act_scale, weight_scale)
effective = -128 * weight_i8_kn.sum(axis=0) + sidecar_bias
```

Pack one 256-byte tile for every 32 output channels:

```text
tile[4*c + 0:4*c + 2]       = uint16(scale_u16[channel])
tile[4*c + 2:4*c + 4]       = uint16(control_u16[channel])
tile[128 + 4*c:132 + 4*c]   = int32(effective[channel])
```

### `w4a8_per_channel`

W4A8 uses the same common prepared bias as `u8i8`, but the HMX accumulator
domain is scaled by 16.  QNN compensates by dividing the drain scale by 16 and
multiplying the folded effective-bias field by 16:

```python
scale_u16, control_u16 = qnn_htp_u8_drain_scale_control(
    float32(512) * act_scale * weight_scale / output_scale / 16
)
sidecar_bias = qnn_htp_perchannel_a8_sidecar_bias_q(dlc_B, act_scale, weight_scale)
effective = 16 * (-128 * weight_w4_kn.sum(axis=0) + sidecar_bias)
```

The `negative_boundary` case specifically verifies that the `2**32` drain
overflow path is required for byte-exact scale/control.

### A16 Drain Scale/Control

The A16 kernels covered here do not use the direct per-channel expression for
drain scale:

```python
direct = float32(act_scale * weight_scale[channel] / output_scale)
```

QNN normalizes weight scales first and then multiplies through a shared maximum
channel scale:

```python
max_w = max(float32(weight_scale))
normalized = float32(weight_scale / max_w)
max_channel_scale = float32(float32(max_w * act_scale) / output_scale)
drain_scale = float32(normalized * max_channel_scale)
```

The resulting `drain_scale` is packed with
`qnn_htp_w8a16_drain_control_words`.  The helper name is historical; the same
packing path is used for the currently covered `w8a16` and
`w4a16_per_channel` records.

### `w8a16`

```python
sidecar_bias = qnn_htp_perchannel_a16_sidecar_bias_q(dlc_B, act_scale, weight_scale)
effective = -128 * weight_i8_kn.sum(axis=0) + sidecar_bias
```

The custom W8A16 HMX wrapper now uses a fully generated 512-byte sidecar.  The
normalized two-stage float32 path removes the former 0/1/2 ULP control mismatches in
`normal_random`.  `native_final` and `hybrid` remain diagnostic sidecar sources
in `scripts/build_w8a16_custom_a16_sidecar.py`; the promoted correctness gate
uses `generated`.

### `w4a16_per_channel`

```python
sidecar_bias = qnn_htp_perchannel_a16_sidecar_bias_q(dlc_B, act_scale, weight_scale)
effective = -128 * weight_w4_kn.sum(axis=0) + sidecar_bias
```

W4A16 uses the same normalized two-stage A16 drain scale/control path as
W8A16.  This is required for the full 512-byte native sidecar to match; fixed
control/drain constants only validate the effective-bias half of the record.

This does not apply to W4 per-group/LPBQ.  Per-group scale handling needs a
separate contract.

## Common Failure Modes

- Passing float ONNX bias instead of quantized DLC `B`.
- Using per-channel `act_scale * weight_scale[channel]` in `dequantize_bias`.
- Computing the restore path in float64 or as one fused ratio.
- Using `round` or `floor` instead of `nearbyintf` and final `trunc`.
- Saturating U8 drain overflow to fp16 max instead of dividing by `2**32`.
- Applying the A8 direct projection to A16 records.
- Treating QNN Native's final W8A16 A16 sidecar as the byte-level custom HMX
  wrapper input.
- Computing A16 drain scale directly as
  `act_scale * weight_scale[channel] / output_scale`; use the normalized
  two-stage float32 path instead.
- Treating `bias_to_vtcm` as the arithmetic source instead of the materialized
  sidecar boundary.

## Validation Targets

Known matching results:

```text
u8i8 normal_random:
  native/custom sidecar bytes: 2048/2048
  same-hardware output: 65536/65536

w4a8_per_channel:
  7/7 correctness cases full 256B sidecar byte-exact

w8a16:
  7/7 native-match cases output exact with generated custom sidecar

w4a16_per_channel:
  7/7 native-match cases full 512B sidecar byte-exact with generated sidecar
  same-hardware output exact for all 7 cases
```

Reproduce these on device (evidence is regenerated, not stored).  Set
`KERNEL_E2E_OUT_ROOT` to any writable directory; the per-case summary lands at
`$KERNEL_E2E_OUT_ROOT/output_<family>_native_match_ci/analysis/custom_native_compare_summary.json`:

```bash
KERNEL_E2E_OUT_ROOT="$PWD/ci_evidence" \
  tests/qnn_kernel_e2e/correctness/test_w8a16_per_channel_native_match_e2e.sh
KERNEL_E2E_OUT_ROOT="$PWD/ci_evidence" \
  tests/qnn_kernel_e2e/correctness/test_w4a16_per_channel_native_match_e2e.sh
```

Useful checks:

```bash
uv run python -m py_compile \
  scripts/qnn_htp_bias_prepare.py \
  scripts/qnn_htp_u8_drain.py \
  scripts/analyze_u8i8_native_bias_record.py \
  scripts/dump_u8i8_bias_prepare_stages.py
```

For generated context artifacts, compare sidecar fields from
`case_native_ctx.bin` against the output of `scripts/qnn_htp_bias_prepare.py`
and `scripts/qnn_htp_u8_drain.py` before looking at device output.
