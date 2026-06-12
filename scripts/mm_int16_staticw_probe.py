#!/usr/bin/env python3
"""int16 MatMul with a STATIC weight (B = graph initializer) vs dynamic, to measure how much of
QNN's per-call wrapper (weight-convert / bias / crouton) folds to prepare-time when the weight is reused.

Usage: mm_int16_staticw_probe.py <outdir>   # emits 64x64 A@Bconst -> Cout, A runtime input only
"""
import sys, os, json
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

outdir = sys.argv[1]
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # reuse one static weight across BATCH matmuls
os.makedirs(outdir, exist_ok=True)
M=K=N=64
rng=np.random.default_rng(0)
A=(rng.integers(-8,8,size=(1,BATCH,M,K)).astype(np.float32))*0.03
B=(rng.integers(-6,7,size=(K,N)).astype(np.float32))*0.05   # STATIC weight [K,N], reused over BATCH
A.tofile(os.path.join(outdir,"A.raw"))
Cf=np.matmul(A.astype(np.float64),B.astype(np.float64))

a=helper.make_tensor_value_info("A",TensorProto.FLOAT,[1,BATCH,M,K])
y=helper.make_tensor_value_info("Cout",TensorProto.FLOAT,[1,BATCH,M,N])
b_init=numpy_helper.from_array(B.astype(np.float32),name="B")     # B as initializer (constant), broadcast
mm=helper.make_node("MatMul",["A","B"],["Cout"],name="mm0")
m=helper.make_model(helper.make_graph([mm],"mm_staticw",[a],[y],[b_init]),
                    opset_imports=[helper.make_opsetid("",17)])
onnx.save(m,os.path.join(outdir,"mm.onnx"))

aA=max(abs(A).max(),1e-6); aB=max(abs(B).max(),1e-6); aC=max(abs(Cf).max(),1e-6)
json.dump({"version":"2.0.0","encodings":[
    {"name":"A","output_dtype":"uint16","y_scale":2*aA/65535.0,"y_zero_point":32768},
    {"name":"B","output_dtype":"int16","y_scale":aB/32767.0},
    {"name":"Cout","output_dtype":"uint16","y_scale":2*aC/65535.0,"y_zero_point":32768}]},
    open(os.path.join(outdir,"ovr.json"),"w"))
print(f"static-weight MatMul A[1,1,64,64] @ B-const[1,1,64,64] -> {outdir}/mm.onnx (only A is a runtime input)")
