---
name: qnn-native-op-flow
description: Build or review QNN native op generation flows in qcom_htp. Use when creating float ONNX native Conv/MatMul references, quantization_overrides, QAIRT converter/quantizer/context commands, native raw input/output runs, or when checking that scale/zero-point/bias handling is in the right QNN layer rather than runtime tensors.
---

# QNN Native Op Flow

Use this skill when generating or reviewing QNN native-op artifacts in
`qcom_htp`. This is a process guide only: do not turn it into an end-to-end
tool unless the user explicitly asks for one.

## Contract

Keep these layers separate:

1. **Source model**: a float ONNX graph. Native op parameters such as weights
   and bias are ordinary float initializers here.
2. **Encoding/quantization**: `quantization_overrides.json` is consumed by
   `qairt-converter`; `qairt-quantizer` turns the encoded DLC into a quantized
   DLC/context. Scale, zero-point, bitwidth, and packing belong here.
3. **Runtime**: `qnn-net-run --retrieve_context` consumes native raw activation
   inputs and emits native raw outputs. Do not pass scales, zero-points, or bias
   as runtime tensors unless the source model/op ABI actually declares them as
   graph inputs.

For a native Conv1x1 MatMul reference, prefer `Conv(A, W, B)` when bias is part
of the op semantics. `B` is a float source-model initializer. The lowered HTP
graph may show `q::ConvLayer.opt.bias_to_vtcm` with an Int32 const; that is the
quantized/lowered kernel side of the same static parameter, not a user runtime
input.

## Standard Flow

1. Generate float tensors:
   - activation `A_float`;
   - weight `W_float`;
   - optional bias `B_float`;
   - a Python oracle for `Y_float = A_float @ W_float.T + B_float`.
2. Derive quantization parameters from float data:
   - activation/output: per-tensor affine encoding, using the kernel family
     zero-point policy;
   - weight: signed symmetric encoding, usually per output channel for native
     Conv axis 0;
   - bias: model it as a float op parameter; when validating the quantized path,
     simulate int32 bias with `bias_scale = act_scale * weight_scale` for
     per-channel weights.
3. Build a float ONNX source model.
4. Write `quantization_overrides.json` for source tensor names.
5. Run `qairt-converter -i model.onnx --quantization_overrides ...`.
6. Run `qairt-quantizer` on the encoded DLC, normally with
   `--enable_float_fallback --bias_bitwidth 32`.
7. Generate context with `qnn-context-binary-generator` and save backend op
   mapping.
8. Run `qnn-net-run` with native raw activation input only.
9. Inspect logs and optrace before comparing results broadly.

## Python Source Model Pattern

Example for logical `X[M,K] @ W[N,K].T + B[N]` as Conv1x1:

```python
from onnx import TensorProto, helper, numpy_helper

# Runtime native raw input will be A_q laid out as [1, K, 1, M].
# Source ONNX remains float.
a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, K, 1, M])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, N, 1, M])
w_init = numpy_helper.from_array(W_float.reshape(N, K, 1, 1).astype("float32"), name="W")
b_init = numpy_helper.from_array(B_float.astype("float32"), name="B")
conv = helper.make_node(
    "Conv",
    ["A", "W", "B"],
    ["Y"],
    name="conv1x1",
    pads=[0, 0, 0, 0],
    strides=[1, 1],
)
graph = helper.make_graph([conv], "native_conv1x1_ref", [a], [y], [w_init, b_init])
```

Write the runtime native input separately:

```python
# A_q is logical [M, K] uint8/uint16. Native Conv input is [1, K, 1, M].
A_q.T.reshape(1, K, 1, M).astype("uint8").tofile("runtime_inputs_native/A.raw")
Path("runtime_input_list.txt").write_text("A:=runtime_inputs_native/A.raw\n")
```

## Encoding Pattern

Use source-model tensor names. For v1 overrides, array form is accepted and is
clear for generated files:

```json
{
  "version": "1.0.0",
  "activation_encodings": [
    {
      "name": "A",
      "enc_type": "PER_TENSOR",
      "bw": 8,
      "dtype": "INT",
      "is_sym": false,
      "scale": [0.0066883437],
      "offset": [-128],
      "min": [-0.856108],
      "max": [0.849420]
    },
    {
      "name": "Y",
      "enc_type": "PER_TENSOR",
      "bw": 8,
      "dtype": "INT",
      "is_sym": false,
      "scale": [0.0479665074],
      "offset": [0],
      "min": [0.0],
      "max": [12.231459]
    }
  ],
  "param_encodings": [
    {
      "name": "W",
      "enc_type": "PER_CHANNEL",
      "bw": 8,
      "dtype": "INT",
      "is_sym": true,
      "scale": [0.0078285569],
      "offset": [0],
      "min": [-0.9942267],
      "max": [0.9942267],
      "axis": 0
    }
  ]
}
```

Do not add a separate bias encoding unless the source tensor and converter path
require it and the log proves it is consumed as intended. If the source ONNX
has `Conv(A,W,B)`, the normal path is to let the quantizer handle the float bias
parameter with `--bias_bitwidth 32`.

## Commands

Use the shared helper flow when available:

```bash
source scripts/qairt_quant_flow.sh

"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter" \
  -i "$OUT_DIR/model.onnx" \
  --target_backend HTP \
  --enable_framework_trace \
  --quantization_overrides "$OUT_DIR/quant_overrides.json" \
  --source_model_input_layout A NONTRIVIAL \
  --desired_input_layout A NONTRIVIAL \
  --source_model_output_layout Y NONTRIVIAL \
  --desired_output_layout Y NONTRIVIAL \
  -o "$OUT_DIR/model_encoded.dlc" \
  > "$OUT_DIR/_convert.log" 2>&1

qairt_quantize_encoded_dlc \
  "$OUT_DIR/model_encoded.dlc" \
  "$OUT_DIR/model.dlc" \
  8 8 32 0 \
  "$OUT_DIR/_quantize.log"
```

Generate context and preserve mapping:

```bash
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" \
  --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
  --dlc_path "$OUT_DIR/model.dlc" \
  --binary_file model_native_ctx \
  --output_dir "$OUT_DIR/ctx" \
  --config_file "$OUT_DIR/htp_config.json" \
  --profiling_level detailed \
  --profiling_option optrace \
  --save_backend_op_mapping \
  > "$OUT_DIR/_ctxgen.log" 2>&1
```

Runtime uses native raw I/O only:

```bash
qnn-net-run \
  --backend ../libQnnHtp.so \
  --retrieve_context model_native_ctx.bin \
  --input_list runtime_input_list.txt \
  --profiling_level detailed \
  --profiling_option optrace \
  --output_dir out \
  --config_file htp_config.json \
  --use_native_input_files \
  --use_native_output_files
```

## Validation Gates

Before trusting output comparisons:

- `_convert.log` should say the expected number of encodings were processed.
- Missing encodings are float fallback risks. Tensors without encoding can
  become fp16 by default.
- `_quantize.log` may warn that `--enable_float_fallback` disables a second
  quantization pass; that is expected for encoded DLC flow.
- `ctx/*bottom_mapping*.json` and optrace should show the intended native op,
  for example `Conv2d_w_scale` and `q::ConvLayer_s1.opt`.
- Check for accidental `q::Add.fp16`, `q::Dequantize`, or `q::Quantize` around
  the compute op. Those usually mean the source graph or encodings are wrong.
- For Conv1x1, `q::ConvLayer.opt.bias_to_vtcm` with Int32 data is normal when
  bias is present; it is not evidence that bias was a runtime input.
- Decode optrace with:

```bash
QNN_SDK_ROOT="$PWD/tools/qnn-sdk" \
  uv run python scripts/decode_qnn_optrace.py "$OUT_DIR"
```

## Lessons

- Do not pass quantization parameters to runtime. They belong in encodings and
  the quantized DLC/context.
- Do not replace source float bias with a manually prepared runtime int32 bias
  unless working on a custom-op ABI that explicitly takes that folded tensor.
- Do not compare against a Python oracle that uses raw float bias while QNN is
  using int32 quantized bias. For per-channel weight, model
  `bias_q = round(bias_float / (act_scale * weight_scale))` and dequantize it
  back for the quantized-path oracle.
- Small 1-LSB differences can be rounding-policy differences after the flow is
  otherwise correct. Prove layout/optrace first before changing data layout.
- For per-group/block W4, use documented blockwise encodings. Do not flatten
  `[output_channel, group]` scales into per-channel encodings.

## Related Files

- `scripts/qairt_quant_flow.sh`: two-stage encoded DLC helper commands.
- `example/qnn_matmul_profile/run_matched_native_a8_ref.sh`: native runtime
  command structure with context binary and native raw I/O.
- `example/qnn_matmul_profile/gen_native_w4a16_conv.py`: float Conv source
  model and native Conv input layout pattern.
- `example/qnn_matmul_profile/gen_matched_native_a8.py`: MatMul/Add native
  reference pattern.
- `example/qnn_hmx_matmul_common/gen_quant_chain.py`: custom-op folded
  bias/control examples; use this only when working on custom-op ABI, not as
  the default native-op bias model.
- `tools/qnn-sdk/docs/QAIRT-Docs/QNN/general/applyencodings.html`: encoding
  inputs, target-specific fallback behavior, and per-channel/block examples.
- `tools/qnn-sdk/docs/QAIRT-Docs/QNN/general/converters.html`: converter
  quantization overrides and fallback behavior.
- `tools/qnn-sdk/docs/QAIRT-Docs/QNN/general/tools.html`: quantizer options
  such as `--bias_bitwidth`.
