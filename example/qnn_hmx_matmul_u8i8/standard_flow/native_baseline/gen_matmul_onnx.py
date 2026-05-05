#!/usr/bin/env python3
"""
Generate a plain fp32 MatMul ONNX model for the native-baseline flow. The QNN
converter quantizes it to a u8-style MatMul so we can compare the custom
HmxU8I8ToU8MatMul path against QNN native behavior.

Output tree:
    model.onnx                   fp32 MatMul with baked-in random weights
    inputs_fp32/input_0.raw      calibration activation (u8-range floats)
    input_list.txt               calibration file list
    runtime_inputs_u8/a.raw      u8 inference-time input for qnn-net-run
    runtime_input_list.txt       inference file list (native u8)
"""
import os, sys, struct
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
M = K = N = 512
np.random.seed(0xB17E)

# Static weight: use the same deterministic i8-style pattern as the custom
# chain generator, scaled to fp32. The converter will re-quantize, but the
# numerical range is what matters for the native-baseline trace.
wRaw_i8 = np.array([((i * 13) % 15) - 7 for i in range(K * N)], dtype=np.int8).reshape(K, N)
w_fp32 = wRaw_i8.astype(np.float32) / 127.0          # fp32 scale → weights_bitwidth=8 picks scale=1/127 approx

# Activation: deterministic u8 0..255 pattern, converted to fp32 [0,1].
aRaw_u8 = np.array([(i * 37) & 0xFF for i in range(M * K)], dtype=np.uint8).reshape(1, M, K)
a_fp32 = aRaw_u8.astype(np.float32) / 255.0

# ---------- ONNX fp32 MatMul: [1, M, K] × [K, N] → [1, M, N] ----------
input_info  = helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [1, M, K])
output_info = helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [1, M, N])
w_init = numpy_helper.from_array(w_fp32, name="weight_0")
node = helper.make_node("MatMul", inputs=["input_0", "weight_0"], outputs=["output_0"], name="MatMul_0")
graph = helper.make_graph([node], "matmul_512x512x512",
                           [input_info], [output_info], [w_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(HERE, "model.onnx"))
print(f"  -> {HERE}/model.onnx  (MatMul {1}x{M}x{K} @ [{K},{N}] = [{1},{M},{N}])")

# ---------- Calibration set (fp32 .raw, little-endian float32) ----------
calib_dir = os.path.join(HERE, "inputs_fp32")
os.makedirs(calib_dir, exist_ok=True)
a_fp32.astype("<f4").tofile(os.path.join(calib_dir, "input_0.raw"))
# Add a second calibration sample with different seed to give quantizer some range
a2 = (np.random.randint(0, 256, size=(1, M, K)).astype(np.float32) / 255.0)
a2.astype("<f4").tofile(os.path.join(calib_dir, "input_1.raw"))
with open(os.path.join(HERE, "input_list.txt"), "w") as f:
    f.write("input_0:=inputs_fp32/input_0.raw\n")
    f.write("input_0:=inputs_fp32/input_1.raw\n")
print(f"  -> {HERE}/inputs_fp32/input_{{0,1}}.raw  +  input_list.txt")

# ---------- Inference-time native input (u8, same as V8) ----------
u8_dir = os.path.join(HERE, "runtime_inputs_u8")
os.makedirs(u8_dir, exist_ok=True)
aRaw_u8.tofile(os.path.join(u8_dir, "a.raw"))
with open(os.path.join(HERE, "runtime_input_list.txt"), "w") as f:
    f.write("input_0:=runtime_inputs_u8/a.raw\n")
print(f"  -> {HERE}/runtime_inputs_u8/a.raw  +  runtime_input_list.txt")
