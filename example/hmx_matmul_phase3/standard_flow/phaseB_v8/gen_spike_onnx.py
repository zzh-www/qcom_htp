#!/usr/bin/env python3
"""
Phase 1 dlsym spike — single-op ONNX for CroutonPackSpike.

Builds tiny graph:
    act_raw [1,1,32,128] u8
        ↓ CroutonPackSpike
    crouton_out [1,1,4,1024] u8       (skel-call output)
    spike_stats [1,1,1,4]    u8       [skel_done, ref_done, n_diffs, max_diff]

Both outputs become graph outputs so qnn-net-run dumps them, and we read
spike_stats to confirm bit-exact match against the in-kernel reference.
"""
import os, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAIN = "hmx"

# Inputs — random 32×128 u8.  Shape is fixed for the spike.
act_in       = helper.make_tensor_value_info("act_raw",       TensorProto.UINT8, [1, 1, 32, 128])
crouton_out  = helper.make_tensor_value_info("crouton_out",   TensorProto.UINT8, [1, 1, 4, 1024])
spike_stats  = helper.make_tensor_value_info("spike_stats",   TensorProto.UINT8, [1, 1, 1, 4])

n_spike = helper.make_node(
    "CroutonPackSpike",
    inputs=["act_raw"],
    outputs=["crouton_out", "spike_stats"],
    name="spike",
    domain=DOMAIN,
)

graph = helper.make_graph(
    [n_spike],
    "crouton_pack_spike_test",
    [act_in],
    [crouton_out, spike_stats],
)
model = helper.make_model(
    graph,
    producer_name="crouton_spike",
    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
)
model.ir_version = 8
onnx.save(model, os.path.join(HERE, "spike_model.onnx"))
print(f"  -> {HERE}/spike_model.onnx")

# Runtime input — deterministic pattern (same `(i*13+7) & 0xff` as the kernel
# uses internally for its reference comparison).
u8_dir = os.path.join(HERE, "runtime_inputs_spike")
os.makedirs(u8_dir, exist_ok=True)
M, K = 32, 128
aRaw = np.array([(i * 13 + 7) & 0xff for i in range(M * K)], dtype=np.uint8).reshape(1, 1, M, K)
aRaw.tofile(os.path.join(u8_dir, "act.raw"))

with open(os.path.join(HERE, "input_list_spike.txt"), "w") as f:
    f.write("act_raw:=runtime_inputs_spike/act.raw\n")

print(f"  -> runtime_inputs_spike/act.raw, input_list_spike.txt")
