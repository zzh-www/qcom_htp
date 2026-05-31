#!/usr/bin/env python3
"""FULLY-FAITHFUL fixed-point simulator: quantizes the output of EVERY compute op (matmul, mul,
add, sub, exp, rsqrt, ...) of the exact deployed graph (gdn_onnx_kernel.GDNChunkQ) at BW bits,
per-tensor symmetric, via a TorchFunctionMode. Data-movement ops (transpose/reshape/cat/pad/...)
preserve encoding (no requant), matching HTP. Run: BW=16 .venv/bin/python scripts/gdn_faithful_sim.py

KEY RESULT (real golden L00): bit-width needed for GDN's deep recurrence to align (oc relerr):
  int16 -> 0.39/1.04/0.89 (FAILS) | int20 -> 0.20/0.13/0.07 | int24 -> 0.004/0.003/0.008 (ALIGNS)
  int32 -> ~0. => GDN needs ~24-bit intermediates; QNN-native caps activations at int16, so it
  cannot align GDN. int32 accumulation without per-op requant (custom HMX) provides the precision.
"""
import os; BW=int(os.environ.get("BW","16"))
import os, sys, glob, json
import numpy as np, torch
sys.path.insert(0,"scripts")
from torch.overrides import TorchFunctionMode
import gdn_onnx_kernel as G
from gdn_onnx_kernel import gdn_chunk_onnx, GDNChunkQ, const_inputs, per_head_vscale, _golden_chunk_args, CONST_INPUT_NAMES
from gdn_ref_kernel import gdn_chunk, relerr

# Quantize the OUTPUT of every compute op (matmul/mul/add/sub/exp/rsqrt/div/neg), per-tensor
# symmetric int16, with per-call-index calibrated ranges. Skip data-movement ops (they preserve
# encoding on device: transpose/reshape/unsqueeze/squeeze/slice/cat/pad/to/clone/getitem).
COMPUTE = {"matmul","mul","add","sub","exp","rsqrt","div","neg","sum","pow","reciprocal","sqrt","mean"}
SKIP = {"transpose","reshape","unsqueeze","squeeze","cat","pad","to","clone","contiguous","view","permute","flatten","__get__"}
class QMode(TorchFunctionMode):
    def __init__(self, ranges, mode): self.r=ranges; self.mode=mode; self.i=0
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}
        out=func(*args,**kwargs)
        name=getattr(func,"__name__","")
        if name in COMPUTE and isinstance(out,torch.Tensor) and out.is_floating_point() and out.numel()>1:
            k=self.i; self.i+=1
            mx=float(out.abs().max())
            if self.mode=="calib":
                self.r[k]=max(self.r.get(k,0.0),mx)
            else:
                s=self.r.get(k,mx)/((1<<(BW-1))-1) or 1.0
                out=torch.clamp(torch.round(out/s),-(1<<(BW-1)),(1<<(BW-1))-1)*s
        return out

def runcase(args, ranges, mode):
    m=GDNChunkQ().eval().double()
    cs=const_inputs(); cargs=[torch.from_numpy(cs[n]).double() for n in CONST_INPUT_NAMES]
    vs,ivs=per_head_vscale(args[2],args[5])
    with QMode(ranges,mode):
        oc,S=m(*args,*cargs,vs.double(),ivs.double())
    return oc,S

GOLD="tests/gdn/golden"
man=json.load(open(f"{GOLD}/manifest.json"))
calibf=[r["file"] for r in man["records"] if r["layer"]==0 and r["split"]=="calib"][:12]
ranges={}
for f in calibf:
    for ch in [0]:
        runcase(list(_golden_chunk_args(f"{GOLD}/{f}",ch)), ranges, "calib")
print("calibrated", len(ranges), "compute-op outputs")
for P in ["p00","p15","p29"]:
    fp=sorted(glob.glob(f"{GOLD}/{P}*_L00.npz"))[0]
    a=list(_golden_chunk_args(fp,0))
    ocr,Sr=gdn_chunk(*a)
    ocq,Sq=runcase(a, ranges, "eval")
    print(f"  {P}: faithful-sim oc relerr {relerr(ocq,ocr):.3f}")
