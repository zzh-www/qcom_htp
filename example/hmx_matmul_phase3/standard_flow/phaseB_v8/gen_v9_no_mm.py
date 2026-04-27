#!/usr/bin/env python3
"""V9 isolation test: PackActCrouton → TcmDramCopy (no matmul) — confirms
the V9 op-pkg ctxgen issue is in MatMulV9 specifically vs the pack stack."""
import os, numpy as np, onnx
from onnx import helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAIN = "hmx"
M, K = 32, 128

act_in = helper.make_tensor_value_info("act_raw", TensorProto.UINT8, [1, 1, M, K])
out    = helper.make_tensor_value_info("out", TensorProto.UINT8, [1, K//32, M//4, 128])

n_pa = helper.make_node("PackActCrouton", ["act_raw"], ["packed_act"], name="pack_act", domain=DOMAIN)
n_cp = helper.make_node("TcmDramCopy", ["packed_act"], ["out"], name="tcm2ddr", domain=DOMAIN)
g = helper.make_graph([n_pa, n_cp], "v9_no_mm", [act_in], [out],
                      value_info=[helper.make_tensor_value_info("packed_act", TensorProto.UINT8, [1, K//32, M//4, 128])])
m = helper.make_model(g, producer_name="v9_no_mm",
                      opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)])
m.ir_version = 8
onnx.save(m, os.path.join(HERE, "v9_no_mm.onnx"))

aRaw = np.array([(i*37) & 0xFF for i in range(M*K)], dtype=np.uint8).reshape(1,1,M,K)
os.makedirs(os.path.join(HERE, "runtime_inputs_v9_nomm"), exist_ok=True)
aRaw.tofile(os.path.join(HERE, "runtime_inputs_v9_nomm", "act.raw"))
with open(os.path.join(HERE, "input_list_v9_nomm.txt"), "w") as f:
    f.write("act_raw:=runtime_inputs_v9_nomm/act.raw\n")
print(f"  -> v9_no_mm.onnx")
