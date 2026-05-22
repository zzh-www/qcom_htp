# U8I8 Bias Prepare Probe

This example documents the recovered u8i8 QNN Native Conv1x1 bias-prepare
rule.  It is intentionally scoped to the prepare/context stage: no device
execution is required because the historical mismatch is visible in
`case_native_ctx.bin`.

Durable per-channel algorithm documentation lives in
`Agent/guides/qnn_htp_perchannel_bias_prepare.md`; the reusable implementation is
`scripts/qnn_htp_bias_prepare.py`.  This probe remains u8i8-focused, but the
shared implementation also exposes the validated w4a8 and A16 projections.

## Run

```bash
example/qnn_hmx_matmul_u8i8/bias_prepare_probe/run_bias_prepare_probe.sh \
  --out-root /tmp/qcom_htp_u8i8_bias_prepare_probe
```

Useful shorter loops:

```bash
example/qnn_hmx_matmul_u8i8/bias_prepare_probe/run_bias_prepare_probe.sh --only vector32
example/qnn_hmx_matmul_u8i8/bias_prepare_probe/run_bias_prepare_probe.sh --only sweep160
```

Outputs:

- `cases/`: generated zero-sum-weight u8i8 cases.
- `native/`: float ONNX, encoded DLC, quantized DLC, and HTP context artifacts.
- `analysis/summary.json`: structured comparison of generated `bias_q`, DLC
  static Conv `B`, and final `bias_to_vtcm` sidecar.
- `analysis/*.native_bias_record.raw`: extracted native sidecar records.

## What It Proves

The probe keeps `sum_k(weight_q) = 0` per output channel.  Therefore the custom
effective-bias formula

```text
effective_i32 = -128 * sum_k(weight_q) + bias_q
```

reduces to `effective_i32 = bias_q`.  Any final sidecar delta is then caused by
HTP prepare's bias conversion path, not activation zero-point folding.

Current evidence:

- The quantized DLC static Conv `B` matches generated `bias_q` exactly.
- The final HTP context sidecar can differ from DLC `B` by `+/-1`.
- The deltas first appear when graph-before `Conv2d_w_scale(scale float32,
  B sFxp32)` is lowered into final `q::ConvLayer.opt.bias_to_vtcm`.

Raw native-sidecar extraction is now only a diagnostic bridge.  The accepted
fix is to reproduce the HTP prepare rule that converts DLC `B` plus Conv scale
metadata into the final sidecar effective-bias field.

## Current Rule Boundary

Validated negative facts:

- Not a runtime-input issue: bias is a float ONNX initializer and a static
  quantized DLC tensor.
- Not a qairt-quantizer bias issue: DLC `B == generated bias_q`.
- Not a folded-weight-sum issue: zero-sum weights still reproduce the delta.
- Not a device execution issue: the delta is already in the generated context.
- Direct context-binary extraction proves the DLC-to-context boundary:
  `case_native_ctx.bin` contains the selected native bias record at byte offset
  `58624` for the zero-sum sweep.  The extracted 1280 bytes are byte-identical
  to `analysis/context_binary_extracted_bias_to_vtcm.raw`; within that record
  scale/control match the generated expectation for all 160 channels, while
  effective bias already differs from DLC `B` by `{-1: 16, 0: 128, +1: 16}`.
  Therefore the `+/-1` mismatch is already present in the context binary, before
  device execution.
- Not a missing-HTP-node issue: a device `qnn-net-run` optrace for the zero-sum
  sweep shows `q::ConvLayer.opt.bias_to_vtcm` as an HTP runtime event for
  `qnn_op=conv1x1`, followed later by `q::ConvLayer_s1.opt`.  In that run the
  event costs were `bias_to_vtcm=1999` cycles, `weights_to_vtcm=2876` cycles,
  and `ConvLayer_s1.opt=3460` cycles.  Final bottom mapping agrees that
  `bias_to_vtcm` consumes and produces an `Int32 [1,5,1,64]` sidecar for
  `N=160`.

Recovered prepare rule:

- The relevant HTP code is in the `libQnnHtp.so` source-string cluster
  `hexagon/ops/2048-byte/int/conv_fused_biasscale.cc`.
- `dequantize_bias(Replacement&, OpRef const&, OpRef const&, OpRef const&)`
  at `0x1ba81b2` converts the quantized DLC bias back to a float const during
  `GraphPrepare`: it reads an output scale from `OutputDef + 0x4c`, scans a
  scale tensor with `fmaxf`, multiplies those scales, then writes float values
  with `GraphPrepare::gen_Const_1D_array<(DType)4>`.
- `find_bias_scale(Replacement&, OpRef const&)` at `0x1ba84cb` computes a new
  bias scale from that float bias, not from the original DLC int32 values:

  ```text
  bias_scale = max(abs(float_bias)) * 16 / pow(2.0, 32.0)
             = max(abs(float_bias)) / 2^28
  ```

  The constants are visible in the disassembly as `-inf` max initialization,
  `8.0` followed by `addss %xmm0,%xmm0`, and `pow(2.0, 32.0)`.
- A targeted callsite trace confirms that graph prepare first builds the
  scale/bias re-quantization rule:

  ```text
  scale_normalizing(Scale, Max_scale)
  requant_bias(Bias, scale_normalizing(...))
  ```

  The same run shows `ConvLayer.opt.convert_bias.simple` being built with
  `WeightScale`, `TotalScale`, `OutOff`, and `ConvCtrl` labels, followed by
  the downstream bias path:

  ```text
  ConvLayer.opt.convert_bias.simple
  ConvLayer.opt.adjust_bias
  ConvLayer.v73.opt.convert_bias
  ConvLayer.opt.bias_scale_shuff
  ConvLayer.opt.bias_to_vtcm
  requant_bias
  ```

- The active numeric body is the internal scalar loop at `0x1ba8116`, reached
  from `GraphPrepare::const_prop`.  Its disassembly is the `requant_bias`
  stage: it evaluates `nearbyintf(input_f32 * encoding_mul + offset)`, clips,
  then writes int32 with `cvttss2si`.
- GDB dumps of this loop resolve the final rule.  QNN first expands the float
  bias into an int32 range using `find_bias_scale`, then
  `bias_scale_shuff.int` restores the sidecar bias with a second float32
  multiply/divide and toward-zero truncation:

  ```python
  global_bias_scale = float32(act_scale) * max(float32(weight_scale))
  dequant = float32(DLC_B * global_bias_scale)
  find_bias_scale = float32(max(abs(dequant)) * 16.0 / 2**32)
  expanded = nearbyintf(float32(dequant * float32(1.0 / find_bias_scale)))
  final_bias = trunc(float32(float32(expanded * find_bias_scale) / global_bias_scale))
  ```

  This reproduces the QNN Native final sidecar bias exactly for both probes:
  `zero_sum_sweep160_m80_p79` is `160/160`, and `normal_random 256^3` is
  `256/256`.
- The ordinary `gen_Const_1D_array<int32>` / common int32 const dumps do not
  contain a 256-element DLC or final sidecar bias vector.  The only notable
  int32 dump in the current run is a 25-int shape/control const beginning with
  `[4, 4, 1, 65536, ...]`.  This excludes the currently instrumented generic
  int32 const path as the final bias serializer.
- Final mapping gives a cleaner boundary than writer-pattern probing.  In the
  final graph, `q::ConvLayer.opt.bias_to_vtcm` consumes a const tensor
  `data_type=50 dims=[1,8,1,64]` and produces the VTCM sidecar tensor consumed
  as input 2 by `q::ConvLayer_s1.opt`.  The const tensor has 512 int32 words,
  exactly the 2048-byte sidecar record size for `N=256`.  The numeric rule is
  the const-fold/evaluator for `bias_scale_shuff.int`, not the `bias_to_vtcm`
  DMA op.

The root cause is HTP prepare's two-stage bias re-quantization chain:

```text
DLC Conv B int32
  -> dequantize_bias: int32 bias back to float
  -> find_bias_scale: derive a new bias scale from max(abs(float_bias))
  -> scale_normalizing(Scale, Max_scale)
  -> requant_bias: nearbyintf(float_bias / find_bias_scale) to expanded int32
  -> convert_bias / adjust_bias graph-prep path
  -> bias_scale_shuff.int: restore with find_bias_scale/global_bias_scale,
     trunc to final Int32 [1,8,1,64]
  -> bias_to_vtcm: DMA materialization of that final const sidecar
```

For custom/native alignment, the clean rule is therefore: do not expect the
QNN Native final sidecar bias to equal the DLC int32 `B` exactly.  The custom
op input generator now reproduces this HTP prepare float re-quantization rule
directly.

## Function-Level Dump

The current debug dump entrypoints are:

```bash
uv run python scripts/gdb_dump_u8i8_bias_prepare.py \
  --native-dir /tmp/qcom_htp_u8i8_sidecar_gate_check/output_u8i8_native_match_normal_random_ci/native_normal_random \
  --out-dir /tmp/qcom_htp_u8i8_sidecar_gate_check/output_u8i8_native_match_normal_random_ci/analysis/gdb_prepare_dump

uv run python scripts/dump_u8i8_bias_prepare_stages.py \
  --case-dir /tmp/qcom_htp_u8i8_sidecar_gate_check/output_u8i8_native_match_normal_random_ci/cases/u8i8/normal_random \
  --native-dir /tmp/qcom_htp_u8i8_sidecar_gate_check/output_u8i8_native_match_normal_random_ci/native_normal_random \
  --gdb-dump-dir /tmp/qcom_htp_u8i8_sidecar_gate_check/output_u8i8_native_match_normal_random_ci/analysis/gdb_prepare_dump \
  --out-dir /tmp/qcom_htp_u8i8_sidecar_gate_check/output_u8i8_native_match_normal_random_ci/analysis/prepare_stage_dump
```

The gdb helper intentionally avoids broad pattern or memory scans.  It only
records named prepare functions, targeted graph-builder callsites, and generated
const buffers.

For `u8i8/normal_random` at `256x256x256`, gdb confirms the prepare stages:

```text
dequantize_bias_out == DLC_B * act_scale * max(weight_scale): 256/256 words
act_scale = 0.006692663766443729
max(weight_scale) = 0.007873821072280407
global_bias_scale = 5.269683606456965e-05
find_bias_scale = 1.8472866292196244e-10
gdb int32 const best matches:
DLC={'file': None, 'match': 0, 'offset': None},
native={'file': None, 'match': 0, 'offset': None}
requant_bias encoding_mul = 5413345280.0
bias_scale_shuff_trunc vs native sidecar bias_q: {'0': 256}
native effective vs expected: {'0': 256}
```

This replaces the earlier per-channel assumption for `dequantize_bias`.
`act_scale * weight_scale[channel]` differs from the gdb dump by up to
`0.000766851`.  The final sidecar bias still differs from DLC `B` by
`{-1: 7, 0: 238, +1: 11}`, but that delta is now explained by the recovered
two-stage QNN prepare rule.

## Current Custom Gate

The promoted same-hardware exactness gate can now use the generated
HTP-prepare sidecar directly:

```bash
USE_NATIVE_BIAS_RECORD=0 bash scripts/run_u8i8_python_case_custom_native_match.sh
```

Fresh validation:

```text
OUT_ROOT=/tmp/qcom_htp_u8i8_formula_gate USE_NATIVE_BIAS_RECORD=0 \
  BUILD_PACKAGES=0 DEVICE=oneplus CHAIN=1 CASE_NAME=normal_random \
  M=256 K=256 N=256 bash scripts/run_u8i8_python_case_custom_native_match.sh

native vs custom bytes: 2048/2048
native vs custom scale/control bytes: 1024/1024
native vs custom effective bytes: 1024/1024
same-hardware custom/native output: 65536/65536, maxdiff=0
```

The old `--bias-record-raw` path remains useful as a diagnostic override, but it
is no longer required for the canonical `normal_random` gate.
