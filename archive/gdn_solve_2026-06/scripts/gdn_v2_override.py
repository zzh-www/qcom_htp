#!/usr/bin/env python3
"""Generate a COMPLETE v2.0.0 quantization override for the GDN quant-path ONNX, with quant
params computed DIRECTLY IN TORCH/ORT (no qairt calibration), per the qnn-native-op-flow skill.

Datatype rule (proven by gdn_faithful_sim.py to give oc relerr ~1.3e-2):
  * I8 = every tensor that is used as a MatMul **in[1]** -> int8 SYMMETRIC.
      - inter-chunk deep GEMMs become UFX16 × SFX8 (the w8a16 primitive; in[0] outliers were
        already swapped to in[0] in gdn_onnx_kernel so the bounded operand lands on in[1]);
      - solve blocks become SFX8 × SFX8 (a block tensor that is one matmul's in1 and another's in0
        is int8 in both — legal, and dodges the per-tensor sharing conflict).
  * everything else -> uint16 ASYMMETRIC (the HTP 16-bit activation port = UFX16).

Ranges are the per-tensor min/max over real golden chunks, evaluated through the float ONNX with
onnxruntime (all intermediate tensors exposed as outputs).

Usage: gdn_v2_override.py model.onnx out_v2.json [--golden DIR] [--layer 0] [--calib 12]
"""
import argparse, json, glob, os, sys
import numpy as np, onnx, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_onnx_kernel as G
from gdn_onnx_kernel import (INPUT_NAMES, CONST_INPUT_NAMES, VSCALE_NAMES, const_inputs,
                             per_head_vscale, _golden_chunk_args)


def i8_tensor_set(model):
    """Every tensor consumed as a MatMul in[1]."""
    return {n.input[1] for n in model.graph.node if n.op_type == "MatMul" and len(n.input) >= 2}


# Internal ops of the fused native L2Norm (F.normalize -> ReduceL2/Clip/Expand/Div) and the
# 1/sqrt(Dk) constant chain (Pow/Reciprocal/Sqrt). Giving these intermediates an encoding BREAKS
# the L2Norm fusion on HTP (the op gets split into per-tensor-quantized primitives, destroying the
# qc/kc normalization — observed as a ~19.5x output error). Skip them so they fall back to float
# and L2Norm fuses; its quantized OUTPUT (the Div result) still gets an encoding.
L2NORM_INTERNAL = {"ReduceL2", "Clip", "Expand", "Pow", "Reciprocal", "Sqrt", "ReduceSum", "ReduceMean"}


def fused_internal_tensors(model):
    return {o for n in model.graph.node if n.op_type in L2NORM_INTERNAL for o in n.output}


def expose_all_outputs(model):
    """Return an ORT session that emits every intermediate tensor + the list of names."""
    m = onnx.ModelProto(); m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    produced = [o for node in m.graph.node for o in node.output if o]
    for name in produced:
        if name not in have:
            vi = onnx.ValueInfoProto(); vi.name = name
            m.graph.output.append(vi); have.add(name)
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess, [o.name for o in sess.get_outputs()]


def calib_feeds(golden, layer, n):
    pr = os.environ.get("GDN_CALIB_PROMPT")
    if pr:                                                     # self-calibrate on a specific prompt
        files = [os.path.basename(f) for f in glob.glob(os.path.join(golden, f"{pr}*_L{layer:02d}.npz"))]
    else:
        man = json.load(open(os.path.join(golden, "manifest.json")))
        files = [r["file"] for r in man["records"] if r["layer"] == layer and r["split"] == "calib"]
    cs = {k: v.astype(np.float32) for k, v in const_inputs().items()}
    feeds = []
    for f in files:
        p = os.path.join(golden, f)
        T = int(np.load(p)["query"].shape[1])
        for chunk in ([0, 1] if T > G.CHUNK else [0]):
            if len(feeds) >= n:
                return feeds
            a = _golden_chunk_args(p, chunk)
            vs, ivs = per_head_vscale(a[2], a[5])
            feed = {nm: t.float().numpy().astype(np.float32) for nm, t in zip(INPUT_NAMES, a)}
            feed.update(cs)
            feed["vscale"] = vs.numpy().astype(np.float32)
            feed["inv_vscale"] = ivs.numpy().astype(np.float32)
            feeds.append(feed)
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx"); ap.add_argument("out")
    ap.add_argument("--golden", default="tests/gdn/golden")
    ap.add_argument("--layer", type=int, default=0); ap.add_argument("--calib", type=int, default=12)
    a = ap.parse_args()
    model = onnx.load(a.onnx)
    skip = fused_internal_tensors(model)                      # don't encode fused-L2Norm internals
    skip |= {t for t in os.environ.get("GDN_SKIP", "").split(",") if t}  # extra float-fallback tensors
    sess, out_names = expose_all_outputs(model)
    valid_in = {i.name for i in sess.get_inputs()}             # graph surgery may have dropped inputs
    feeds = [{k: v for k, v in f.items() if k in valid_in}
             for f in calib_feeds(a.golden, a.layer, a.calib)]
    # i8 = every MatMul in[1]; ALSO in[0] of solve blocks (K<=32) so the solve runs int8×int8
    # (both operands int8 -> legal SFX8×SFX8, and avoids the residual UFX16×UFX16 on Ap@Ap).
    shp = dict(zip(out_names, sess.run(out_names, feeds[0]))); shp.update(feeds[0])
    # i8 = every MatMul in[1]; ALSO in[0] of solve blocks (K<=32) so the solve runs int8×int8
    # (both operands int8 -> legal SFX8×SFX8, and avoids the residual UFX16×UFX16 on Ap@Ap).
    # NOTE: relaxing the non-self solve matmuls to i8×i16 (in[0] int16) was tried and is WORSE on
    # device (T 4.4e-3→8.7e-3) — the all-int8 solve is the better operating point here.
    i8 = set()
    deep_i8 = os.environ.get("GDN_DEEP_I8") == "1"   # EXPERIMENT: also int8 the deep-GEMM in[0]
    for n in model.graph.node:
        if n.op_type == "MatMul" and len(n.input) >= 2:
            i8.add(n.input[1])
            k = shp[n.input[0]].shape[-1] if n.input[0] in shp and hasattr(shp[n.input[0]], "shape") else 999
            if k <= 32 or deep_i8:
                i8.add(n.input[0])
    rng = {}
    def upd(nm, arr):
        lo, hi = float(np.min(arr)), float(np.max(arr))
        p = rng.get(nm)
        rng[nm] = (lo, hi) if p is None else (min(p[0], lo), max(p[1], hi))
    for feed in feeds:
        for nm, arr in feed.items():
            upd(nm, arr)
        for nm, arr in zip(out_names, sess.run(out_names, feed)):
            upd(nm, arr)
    encs = []
    n8 = 0
    for nm, (lo, hi) in rng.items():
        if nm in skip:                                        # fused-op internal -> float fallback
            continue
        if nm in i8:                                          # int8 symmetric
            amax = max(abs(lo), abs(hi)) or 1e-6
            encs.append({"name": nm, "output_dtype": "int8", "y_scale": amax / 127.0})
            n8 += 1
        else:                                                 # 16-bit non-matmul-in1 tensor
            mode = "sym" if os.environ.get("GDN_I16_SYM") == "1" else os.environ.get("IN16_MODE", "sym")
            amax = max(abs(lo), abs(hi)) or 1e-6
            if mode == "sym":                                 # int16 symmetric (SFX16, zp 0)
                encs.append({"name": nm, "output_dtype": "int16", "y_scale": amax / 32767.0})
            elif mode == "mid":                               # uint16, zp pinned to midpoint 32768
                encs.append({"name": nm, "output_dtype": "uint16", "y_scale": amax / 32768.0, "y_zero_point": 32768})
            else:                                             # uint16 asymmetric (zp fit to range)
                lo, hi = min(lo, 0.0), max(hi, 0.0); scale = (hi - lo) / 65535.0 or 1e-6
                encs.append({"name": nm, "output_dtype": "uint16", "y_scale": scale,
                             "y_zero_point": max(0, min(65535, int(round(-lo / scale))))})
    json.dump({"version": "2.0.0", "encodings": encs}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(encs)} tensors ({n8} int8 / {len(encs)-n8} uint16), "
          f"{len(i8)} matmul-in1 tensors targeted")


if __name__ == "__main__":
    main()
