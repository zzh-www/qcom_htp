#!/usr/bin/env python3
"""Marker-weight 256³ native MatMul generator.

Builds two ONNX models with single MatMul, M=K=N=256, w8a8.
Each model has a distinct, recoverable wRaw_KN pattern so we can
hex-search the prepared ctx-binary's weights blob and recover the
deep-variant's wt repack formula.

Outputs:
  s256_marker_K/   wRaw[k,n] = (k & 0x7F) - 64       (range [-64, 63] int8)
                   → each cell value tells us its source K row.
  s256_marker_N/   wRaw[k,n] = (n & 0x7F) - 64
                   → each cell value tells us its source N column.

256 distinct values fit in int8 because we use only 128 buckets — k%128.
Combined with the second test (n-marker), we recover the (k%128, n%128)
mapping per dst byte. For 256-wide dimensions, ambiguity is between
{k=0,k=128} and similar — break with a third test if needed.

Usage:
  python gen_marker_wt_256.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import json

HERE = os.path.dirname(os.path.abspath(__file__))
M = K = N = 256

def build(out_dir, wRaw_i8):
    OUT = os.path.join(HERE, out_dir)
    os.makedirs(OUT, exist_ok=True)
    # Save wRaw for later RE.
    np.save(os.path.join(OUT, "wRaw_KN.npy"), wRaw_i8)

    # Activation: distinct identity-like u8 (doesn't really matter for wt RE).
    aRaw_u8 = np.arange(M * K, dtype=np.uint32).astype(np.uint8).reshape(1, M, K)
    a_fp32 = aRaw_u8.astype(np.float32) / 255.0

    w_fp32 = wRaw_i8.astype(np.float32) / 127.0

    inp = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, M, K])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, M, N])
    w_init = numpy_helper.from_array(w_fp32, name="W")
    node = helper.make_node("MatMul", inputs=["A", "W"], outputs=["Y"], name="MatMul_0")
    graph = helper.make_graph([node], "model", [inp], [out], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(OUT, "model.onnx"))

    # Calibration / runtime inputs.
    calib = os.path.join(OUT, "inputs_fp32"); os.makedirs(calib, exist_ok=True)
    a_fp32.astype("<f4").tofile(os.path.join(calib, "A.raw"))
    a2 = (np.random.RandomState(0).randint(0, 256, size=(1, M, K)).astype(np.float32) / 255.0)
    a2.astype("<f4").tofile(os.path.join(calib, "input_1.raw"))
    with open(os.path.join(OUT, "input_list.txt"), "w") as f:
        f.write("A:=inputs_fp32/A.raw\n")
        f.write("A:=inputs_fp32/input_1.raw\n")
    u8 = os.path.join(OUT, "runtime_inputs_u8"); os.makedirs(u8, exist_ok=True)
    aRaw_u8.tofile(os.path.join(u8, "a.raw"))
    with open(os.path.join(OUT, "runtime_input_list.txt"), "w") as f:
        f.write("A:=runtime_inputs_u8/a.raw\n")

    # Quant overrides: int8 sym for both A, Y, W (mirrors baseline_s512).
    sym8 = {
        "bitwidth": 8, "dtype": "int", "is_symmetric": "True",
        "scale": 0.007874015748031496, "offset": -128, "min": -1.0, "max": 1.0,
    }
    overrides = {
        "activation_encodings": {"A": [sym8], "Y": [sym8]},
        "param_encodings": {"W": [sym8]},
    }
    with open(os.path.join(OUT, "quant_overrides.json"), "w") as f:
        json.dump(overrides, f, indent=2)
    print(f"  {out_dir}: wRaw range [{wRaw_i8.min()}, {wRaw_i8.max()}]  "
          f"sample wRaw[0:4,0:4]=\n{wRaw_i8[:4,:4]}")

# Marker K: per-row identity. wRaw[k, n] depends on k only.
# Use (k % 128) - 64 → range [-64, 63], stored unique per row.
# Note: With k mod 128, rows 0 and 128 collide (same value), rows 1 and 129 collide, etc.
# That's OK for recovering layout — we only need to identify the row up to a 128-period.
wRaw_K = np.tile((np.arange(K, dtype=np.int32) % 128 - 64).astype(np.int8).reshape(K, 1), (1, N))
build("s256_marker_K", wRaw_K)

# Marker N: per-col identity. wRaw[k, n] depends on n only.
wRaw_N = np.tile((np.arange(N, dtype=np.int32) % 128 - 64).astype(np.int8).reshape(1, N), (K, 1))
build("s256_marker_N", wRaw_N)

print("DONE. Now run run_native_marker_256.sh to ctxgen both.")
