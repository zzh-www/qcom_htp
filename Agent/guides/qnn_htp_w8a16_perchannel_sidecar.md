# QNN HTP W8A16 Per-Channel Sidecar

This note records the promoted W8A16 per-channel sidecar contract used by the
custom `HmxU16I8ToU16MatMul` wrapper and the evidence that it matches QNN
Native output.

## Scope

Covered kernel:

```text
family: w8a16
custom op: QnnHmxMatMulW8A16Package::HmxU16I8ToU16MatMul
native op: QNN Native Conv1x1 lowered to q::ConvLayer_s1.opt
shape gate: M=256 K=256 N=256 CHAIN=1
weight schema: signed I8, per-output-channel scale
activation/output: U16/A16 runtime tensors
```

This is separate from W4A16 and W4 LPBQ/per-group handling.

## Two Different Sidecars

There are two sidecar boundaries that must not be conflated:

| Name | Producer | Consumer | Use |
| --- | --- | --- | --- |
| QNN Native final sidecar | QNN ctxgen `q::ConvLayer.opt.bias_to_vtcm` | native `q::ConvLayer_s1.opt` | Native HTP Conv execution |
| Custom wrapper sidecar | `scripts/build_w8a16_custom_a16_sidecar.py --source generated` | custom `HmxU16I8ToU16MatMul` wrapper | Custom/native exactness gate |

The native final sidecar is useful as a reference artifact, but it is not the
byte-level input contract for the custom wrapper.  The historical `zp_neutral`
failure showed this directly: importing the native final 512-byte record into
the custom wrapper made one output channel differ by `+256`, even though the
bytes were native-exact.  The generated custom sidecar produces the same output
as QNN Native.

## Record Layout

The W8A16 per-channel sidecar is 512 bytes per 32 output channels:

```text
record shape: Int32 [1, N/32, 1, 128]
tile bytes:   512 bytes per N32 tile
```

Each tile is split into even-channel and odd-channel halves:

```text
half 0 bytes 0x000..0x0ff: even channels 0,2,...,30
half 1 bytes 0x100..0x1ff: odd channels 1,3,...,31
```

Within each half, each lane uses 8 bytes:

```text
control/drain words:
  half_base + 8 * lane + 0: uint16 word0
  half_base + 8 * lane + 2: uint16 word1
  half_base + 8 * lane + 4: uint16 word2
  half_base + 8 * lane + 6: uint16 word3

effective bias field:
  half_base + 128 + 8 * lane: int32 effective_i32
```

Natural channel order is reconstructed by:

```python
parity = channel & 1
lane = (channel % 32) // 2
half_base = parity * 256
```

## Effective Bias

The custom wrapper effective-bias field is:

```python
sidecar_bias = qnn_htp_perchannel_a16_sidecar_bias_q(
    dlc_bias_q,
    act_scale,
    weight_scale,
)
effective_i32 = -128 * sum_k(weight_i8[k, channel]) + sidecar_bias[channel]
```

Important details:

- `dlc_bias_q` is QNN's quantized DLC Conv `B` int32 tensor, not the original
  float ONNX bias.
- `act_scale`, `output_scale`, and `weight_scale` are synced from the quantized
  QNN Native DLC encodings before building the custom sidecar.
- For the current 7 correctness cases, the generated effective field is the
  custom wrapper ABI input.  The QNN Native final sidecar can differ in an edge
  case (`zp_neutral`: one effective field differs), while the generated custom
  sidecar still gives output-exact custom/native results.

Implementation entrypoint:

```python
from scripts.analyze_a16_native_bias_record import expected_record

_, generated_record, effective_i32 = expected_record(case_dir)
```

## Drain Scale

The recovered W8A16 drain scale is not the direct per-channel expression:

```python
wrong = float32(act_scale * weight_scale[channel] / output_scale)
```

QNN ctxgen materializes it through a normalized two-stage float32 path:

```python
max_w = max(float32(weight_scale))
normalized = float32(weight_scale / max_w)
max_channel_scale = float32(float32(max_w * act_scale) / output_scale)
drain_scale = float32(normalized * max_channel_scale)
```

The two float32 rounding points are required.  The direct expression matched
some channels but caused `0/1/2` ULP mismatches in `normal_random`, which then
changed A16 control/drain words.

Implementation entrypoint:

```python
from scripts.qnn_htp_u8_drain import qnn_htp_w8a16_drain_scale

drain_scale = qnn_htp_w8a16_drain_scale(
    act_scale,
    weight_scale,
    output_scale,
)
```

The scale is then packed by the recovered QNN materializer logic:

```python
from scripts.qnn_htp_u8_drain import qnn_htp_w8a16_drain_control_words

word0, word1, word2, word3 = qnn_htp_w8a16_drain_control_words(drain_scale)
```

The packing implementation mirrors the observed QNN HTP x86 prepare path:

```text
0x1048 -> 0x1064
materializer/helper: 0x14e37cf / 0x14e3843
scale_ctl: 0x8000
mode: 0
split: 1
```

## Custom Build Modes

`scripts/build_w8a16_custom_a16_sidecar.py` supports three modes:

| Mode | Meaning | Status |
| --- | --- | --- |
| `generated` | Generated control/drain plus generated effective-bias fields | promoted default |
| `hybrid` | Native final control/drain plus generated effective-bias fields | diagnostic only |
| `native_final` | QNN Native final sidecar copied byte-for-byte | diagnostic only |

The promoted path is:

```bash
uv run python scripts/build_w8a16_custom_a16_sidecar.py \
  --case-dir "$CASE_DIR" \
  --native-sidecar-raw "$NATIVE_RECORD" \
  --source generated \
  --out-raw "$CUSTOM_SIDECAR"
```

`--native-sidecar-raw` is still passed so the script can emit comparison
metrics, but `generated` does not copy native final bytes into the custom ABI
input.

## Correctness CI

The W8A16 per-channel correctness gate is part of the default correctness CI:

```bash
tests/qnn_kernel_e2e/correctness/test_w8a16_per_channel_native_match_e2e.sh
```

That wrapper delegates to:

```bash
scripts/run_qnn_kernel_e2e_ci.sh w8a16_per_channel_native_match
```

The full correctness group runs it through:

```bash
tests/qnn_kernel_e2e/run_correctness.sh
```

The gate runs these 7 cases at `256x256x256`:

```text
normal_random
zp_neutral
positive_boundary
negative_boundary
single_k_impulse
bias_only
scale_only
```

Current device evidence:

```text
command:
  KERNEL_E2E_OUT_ROOT="$PWD/ci_evidence" \
    BUILD_PACKAGES=0 DEVICE=oneplus \
    tests/qnn_kernel_e2e/correctness/test_w8a16_per_channel_native_match_e2e.sh

summary:
  $KERNEL_E2E_OUT_ROOT/output_w8a16_per_channel_native_match_ci/analysis/custom_native_compare_summary.json

result:
  all 7 cases passed
  every case exact: 65536/65536
  every case maxabs: 0
  custom_sidecar_source: generated
```

The machine-readable promoted status is in:

```text
tests/qnn_kernel_e2e/correctness/status.json
```
