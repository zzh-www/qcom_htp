#!/usr/bin/env python3
"""Native baseline chain variant: N MatMul ops in series sharing the
same weight tensor. Mirrors custom_u8i8/gen_u8i8_chain.py so we can
profile native QNN MatMul vs HmxU8I8ToU8MatMul with the same chain
methodology.

Usage:
    python gen_matmul_onnx_chain.py --size 256 --chain 8 --out s256_chain8

Output (in --out subdir):
    model.onnx                ONNX graph (chain of N MatMul ops)
    inputs_fp32/A.raw, input_1.raw    fp32 calibration inputs
    runtime_inputs_u8/a.raw   u8 runtime input
    input_list.txt / runtime_input_list.txt
"""
import argparse, json, os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=None, help="cube shape; sets M=K=N")
ap.add_argument("--M", type=int, default=None)
ap.add_argument("--K", type=int, default=None)
ap.add_argument("--N", type=int, default=None)
ap.add_argument("--chain", type=int, default=8,
                help="number of MatMul ops in series (each consumes previous output)")
ap.add_argument("--shared_w", action="store_true",
                help="all MatMul ops share the same W tensor (default: per-op W to avoid compiler-side folding)")
ap.add_argument("--mode", choices=["chain", "independent"], default="chain",
                help="chain: Y_i = MatMul(Y_{i-1}, W_i) — output feeds next input. "
                     "independent: each MatMul gets fresh A_i and W_i (no data dep) — "
                     "isolates pure HMX kernel cost from saturation/cache effects.")
ap.add_argument("--out", required=True, help="output subdir relative to this script")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, args.out)
os.makedirs(OUT, exist_ok=True)

if args.size is not None:
    M = K = N = args.size
else:
    assert args.M and args.K and args.N, "must pass --size or --M --K --N"
    M, K, N = args.M, args.K, args.N

assert M == K == N, "chain only works for square (output shape == input shape)"
chain = max(1, int(args.chain))

np.random.seed(0xB17E)

def make_a(idx):
    """Fresh per-instance act bytes (uint8 [1,M,K])."""
    seed = (idx + 1) * 374761393
    return np.array([((i * 37 + seed) & 0xFF) for i in range(M * K)],
                    dtype=np.uint8).reshape(1, M, K)

def make_w(idx):
    """Fresh per-instance weight bytes (int8 [K,N])."""
    seed = (idx + 1) * 1000003
    return np.array([((seed + i * 13) % 15) - 7 for i in range(K * N)],
                    dtype=np.int8).reshape(K, N)

aRaw_u8 = make_a(0)
a_fp32  = aRaw_u8.astype(np.float32) / 255.0

# Always 1 model input "A". For --mode independent we extend with A_1..A_{chain-1}.
input_info_list = [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, M, K])]
output_info_list = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, M, N])]

# Build the graph based on mode.
w_inits = []
nodes = []
shared_W_name = None
if args.shared_w:
    wRaw_i8 = make_w(0)
    w_fp32  = wRaw_i8.astype(np.float32) / 127.0
    shared_W_name = "W"
    w_inits.append(numpy_helper.from_array(w_fp32, name=shared_W_name))

if args.mode == "chain":
    # Chain: Y_i = MatMul(Y_{i-1}, W_i). Each MatMul consumes previous
    # output. Subject to int8 saturation propagation in intermediates.
    prev = "A"
    for i in range(chain):
        out_name = f"Y_{i}" if i < chain - 1 else "Y"
        if args.shared_w:
            w_name = shared_W_name
        else:
            wRaw_i8 = make_w(i)
            w_fp32  = wRaw_i8.astype(np.float32) / 127.0
            w_name = f"W_{i}"
            w_inits.append(numpy_helper.from_array(w_fp32, name=w_name))
        nodes.append(helper.make_node(
            "MatMul", inputs=[prev, w_name], outputs=[out_name],
            name=f"MatMul_{i}",
        ))
        prev = out_name
elif args.mode == "independent":
    # Independent: each MatMul gets fresh A_i (model input) and fresh W_i
    # (static). No data dep between MatMuls. HMX is a single physical
    # unit so they serialise at the hardware level. To keep all 8 MatMuls
    # alive in the graph (no DCE), the graph has 8 outputs Y_0..Y_{n-1}.
    # First input keeps name "A" so the runner script doesn't change.
    output_info_list = []
    for i in range(chain):
        if i == 0:
            a_name = "A"
        else:
            a_name = f"A_{i}"
            input_info_list.append(helper.make_tensor_value_info(
                a_name, TensorProto.FLOAT, [1, M, K]))
        if args.shared_w:
            w_name = shared_W_name
        else:
            wRaw_i8 = make_w(i)
            w_fp32  = wRaw_i8.astype(np.float32) / 127.0
            w_name = f"W_{i}"
            w_inits.append(numpy_helper.from_array(w_fp32, name=w_name))
        out_name = f"Y_{i}"
        nodes.append(helper.make_node(
            "MatMul", inputs=[a_name, w_name], outputs=[out_name],
            name=f"MatMul_{i}",
        ))
        output_info_list.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, [1, M, N]))

graph = helper.make_graph(nodes, "model_chain",
                          input_info_list, output_info_list, w_inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(OUT, "model.onnx"))

calib = os.path.join(OUT, "inputs_fp32"); os.makedirs(calib, exist_ok=True)
a_fp32.astype("<f4").tofile(os.path.join(calib, "A.raw"))
# Two-line input_list.txt for calibration sweeps. Runtime list below uses u8.
input_list_lines = []
runtime_list_lines = []
u8 = os.path.join(OUT, "runtime_inputs_u8"); os.makedirs(u8, exist_ok=True)

if args.mode == "independent":
    # Generate per-input fresh data both for calibration (fp32) and runtime (u8).
    parts = []
    runtime_parts = []
    for i in range(chain):
        a_i_u8 = make_a(i)
        a_i_fp32 = a_i_u8.astype(np.float32) / 255.0
        a_i_fp32.astype("<f4").tofile(os.path.join(calib, f"A_{i}.raw" if i else "A.raw"))
        a_i_u8.tofile(os.path.join(u8, f"a_{i}.raw" if i else "a.raw"))
        name = "A" if i == 0 else f"A_{i}"
        parts.append(f"{name}:=inputs_fp32/{'A.raw' if i==0 else f'A_{i}.raw'}")
        runtime_parts.append(f"{name}:=runtime_inputs_u8/{'a.raw' if i==0 else f'a_{i}.raw'}")
    # 2 calibration samples (each line = one inference), so duplicate parts twice.
    input_list_lines = [" ".join(parts), " ".join(parts)]
    runtime_list_lines = [" ".join(runtime_parts)]
else:
    a2 = (np.random.randint(0, 256, size=(1, M, K)).astype(np.float32) / 255.0)
    a2.astype("<f4").tofile(os.path.join(calib, "input_1.raw"))
    aRaw_u8.tofile(os.path.join(u8, "a.raw"))
    input_list_lines = ["A:=inputs_fp32/A.raw", "A:=inputs_fp32/input_1.raw"]
    runtime_list_lines = ["A:=runtime_inputs_u8/a.raw"]

with open(os.path.join(OUT, "input_list.txt"), "w") as f:
    f.write("\n".join(input_list_lines) + "\n")
with open(os.path.join(OUT, "runtime_input_list.txt"), "w") as f:
    f.write("\n".join(runtime_list_lines) + "\n")

with open(os.path.join(OUT, "native_io.json"), "w") as f:
    json.dump(
        {
            "input_name": "A",
            "output_name": "Y" if args.mode == "chain" else [f"Y_{i}" for i in range(chain)],
            "native_input": "runtime_inputs_u8/a.raw",
            "runtime_input_list": "runtime_input_list.txt",
            "native_input_storage": "uint8",
            "native_input_bytes": int(M * K),
            "expected_native_output_storage": "uint8",
            "expected_native_output_bytes": int(M * N),
            "shape": [1, M, K],
            "output_shape": [1, M, N],
        },
        f,
        indent=2,
    )

# Write quant_overrides.json with int8 specs for ALL intermediate tensors.
# Without this, QNN falls back to fp16 ConvLayer for intermediates (Y_0..Y_{n-2})
# because they don't have explicit int8 encodings defined. With this, every
# MatMul in the chain uses int8 q::ConvLayer_s1.opt, matching the custom
# u8/i8 chain's int8 arithmetic surface.
import json as _json
sym8 = {
    "bitwidth": 8, "dtype": "int", "is_symmetric": "True",
    "scale": 0.007874015748031496, "offset": -128, "min": -1.0, "max": 1.0,
}
act_encs = {"A": [sym8]}
if args.mode == "chain":
    act_encs["Y"] = [sym8]
    for i in range(chain - 1):
        act_encs[f"Y_{i}"] = [sym8]
else:  # independent
    for i in range(chain):
        if i > 0:
            act_encs[f"A_{i}"] = [sym8]
        act_encs[f"Y_{i}"] = [sym8]
if args.shared_w:
    param_encs = {"W": [sym8]}
else:
    param_encs = {f"W_{i}": [sym8] for i in range(chain)}
overrides = {
    "activation_encodings": act_encs,
    "param_encodings": param_encs,
}
with open(os.path.join(OUT, "quant_overrides.json"), "w") as f:
    _json.dump(overrides, f, indent=2)

print(f"  size={M} chain={chain} -> {OUT}/model.onnx + inputs")
print(f"  graph: {chain} × MatMul (shared W), input [1,{M},{K}] → output [1,{M},{N}]")
print(f"  quant_overrides.json: int8 sym for A, Y, Y_0..Y_{chain-2} ({chain+1} tensors total)")
