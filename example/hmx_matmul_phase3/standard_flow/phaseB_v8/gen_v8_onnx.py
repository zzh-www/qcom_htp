#!/usr/bin/env python3
"""
Phase B: author custom-domain ONNX matching run_matmul_v8_graph.cpp semantics.

Tensor types (matching V8 host binary verbatim so kernels see identical
memory layouts):
    act_raw   [1,1,512,512] UINT8   input (APP_WRITE)
    wt_raw    [1,1,512,512] UFIXED_POINT_8  STATIC  (baked .bin)
    bias_fp16 [1,1,16,128]  UINT16  APP_WRITE   (fed via input_list)
    scratch   [1,1,1,2048]  UINT8   APP_WRITE   (fed via input_list)
    packed_{act,wt} / out_tile / out  UINT8  intermediate/output

ONNX can't express APP_WRITE directly, so bias and scratch are declared as
graph inputs (not initializers). input_list.txt feeds all three.
"""
import os, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
M = K = N = 512
M_T, K_T, N_T = M // 32, K // 32, N // 32
DOMAIN = "hmx"

# Static weight (kept inline as initializer, matches V8's STATIC wt_raw)
wRaw = np.array([((i * 13) % 15) - 7 for i in range(K * N)], dtype=np.int8).reshape(1, 1, K, N)
wRaw_u8 = wRaw.view(np.uint8)

# ---------- tensor info (inputs + static initializers) ----------
act_in   = helper.make_tensor_value_info("act_raw",   TensorProto.UINT8, [1, 1, M, K])
bias_in  = helper.make_tensor_value_info("bias_fp16", TensorProto.UINT16, [1, 1, N_T, 128])
scr_in   = helper.make_tensor_value_info("scratch",   TensorProto.UINT8,  [1, 1, 1, 2048])
out      = helper.make_tensor_value_info("out",       TensorProto.UINT8, [1, M_T, N_T, 1024])
wt_init  = numpy_helper.from_array(wRaw_u8, name="wt_raw")

packed_act = helper.make_tensor_value_info("packed_act", TensorProto.UINT8, [1, M_T, K_T, 1024])
packed_wt  = helper.make_tensor_value_info("packed_wt",  TensorProto.UINT8, [1, N_T, K_T, 1024])
out_tile   = helper.make_tensor_value_info("out_tile",   TensorProto.UINT8, [1, M_T, N_T, 1024])

n_pa = helper.make_node("PackActivationU8RowMajor", ["act_raw"], ["packed_act"],
                        name="pack_act", domain=DOMAIN)
n_pw = helper.make_node("PackWeightToHmxTileV3",    ["wt_raw"],  ["packed_wt"],
                        name="pack_wt",  domain=DOMAIN)
n_mm = helper.make_node("MatMulV8",
                        ["packed_act", "packed_wt", "bias_fp16", "scratch"], ["out_tile"],
                        name="mmv8", domain=DOMAIN)
n_cp = helper.make_node("TcmDramCopy", ["out_tile"], ["out"],
                        name="tcm2ddr", domain=DOMAIN)

graph = helper.make_graph(
    [n_pa, n_pw, n_mm, n_cp],
    "matmul_v8_512",
    [act_in, bias_in, scr_in],   # bias/scratch now runtime inputs
    [out],
    [wt_init],                   # only weight is static
    value_info=[packed_act, packed_wt, out_tile],
)
model = helper.make_model(
    graph,
    producer_name="v8_phaseB",
    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
)
model.ir_version = 8
onnx.save(model, os.path.join(HERE, "v8_model.onnx"))
print(f"  -> {HERE}/v8_model.onnx")

# ---------- runtime input files ----------
u8_dir = os.path.join(HERE, "runtime_inputs_u8")
os.makedirs(u8_dir, exist_ok=True)
aRaw = np.array([(i * 37) & 0xFF for i in range(M * K)], dtype=np.uint8).reshape(1, 1, M, K)
aRaw.tofile(os.path.join(u8_dir, "act.raw"))

bias = np.zeros((1, 1, N_T, 128), dtype=np.uint16)
def fp32_to_fp16(f):
    return np.float16(f).view(np.uint16).item()
for nt in range(N_T):
    for c in range(32):
        n = nt * 32 + c
        scale = 1.0 / (K * (1.0 + 0.1 * (n % 7)))
        bias[0, 0, nt, 2*c]     = fp32_to_fp16(512.0 * scale)
        bias[0, 0, nt, 2*c + 1] = 0x4000
bias.tofile(os.path.join(u8_dir, "bias.raw"))

scratch = np.zeros((1, 1, 1, 2048), dtype=np.uint8)
scratch.tofile(os.path.join(u8_dir, "scratch.raw"))

with open(os.path.join(HERE, "input_list.txt"), "w") as f:
    f.write("act_raw:=runtime_inputs_u8/act.raw bias_fp16:=runtime_inputs_u8/bias.raw scratch:=runtime_inputs_u8/scratch.raw\n")
print(f"  -> runtime_inputs_u8/{{act,bias,scratch}}.raw (+ input_list.txt)")
