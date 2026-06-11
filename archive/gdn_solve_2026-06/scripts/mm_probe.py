#!/usr/bin/env python3
"""Minimal single-MatMul probe: find the exact recipe that makes the HTP backend compile a
MatMul as  UFX16(asym) x SFX8(sym) -> UFX16  (int16 activation x int8 weight, the w8a16 primitive
that keeps the int32 accumulator from overflowing on a 128-deep contraction).

Two source graphs (K=128 contraction, the overflow-prone case):
  --kind weight : out = A @ W ,  A is an input (act),  W is a constant initializer (weight)
  --kind actact : out = A @ B ,  BOTH A and B are inputs (activation x activation, like GDN)

Usage:
  mm_probe.py --emit DIR --kind weight|actact      # write onnx + calib set
  mm_probe.py --check DIR/ctx/<name>_bottom_mapping_graph_before.json   # decode HTP-layer datatypes
"""
import argparse, os, json, glob
import numpy as np, torch, torch.nn as nn

M, K, N = 32, 128, 32          # K=128 = the deep contraction that overflows int32 at int16xint16


class MMWeight(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.w = nn.Parameter(torch.randn(K, N, generator=g) * 0.3)   # constant weight
    def forward(self, a):
        return torch.matmul(a, self.w)


class MMActAct(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def build_qdq(out_dir):
    """MatMul(A, B) with an explicit int8 QuantizeLinear/DequantizeLinear on B (QAT-style).
    The converter is documented to honor QDQ datatypes from the source model, so B should land
    as SFX8 even though it is a dynamic activation (no weight)."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, M, K])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, K, N])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, M, N])
    bscale = numpy_helper.from_array(np.array(5.0 / 127, np.float32), "B_scale")
    bzp = numpy_helper.from_array(np.array(0, np.int8), "B_zp")        # int8 symmetric zero-point
    nodes = [helper.make_node("QuantizeLinear", ["B", "B_scale", "B_zp"], ["Bq"]),
             helper.make_node("DequantizeLinear", ["Bq", "B_scale", "B_zp"], ["Bdq"]),
             helper.make_node("MatMul", ["A", "Bdq"], ["Y"])]
    graph = helper.make_graph(nodes, "qdq", [A, B], [Y], [bscale, bzp])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, os.path.join(out_dir, "mm.onnx"))


def emit(out_dir, kind, nsamp=12):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    # A_POS=1 makes the in[0] tensor SINGLE-SIDED (all-positive, like eg/decay) so its zero-point
    # lands far from the 32768 midpoint — tests whether an off-midpoint asymmetric in[0] breaks.
    posA = os.environ.get("A_POS") == "1"
    def shape_a(nm, a):
        return (np.abs(a) + 0.05).astype("<f4") if (posA and nm == "A") else a
    if kind == "weight":
        m = MMWeight().eval()
        inputs = {"A": (1, 1, M, K)}
        names = ["A"]
        torch.onnx.export(m, (torch.randn(1, 1, M, K),), os.path.join(out_dir, "mm.onnx"),
                          opset_version=17, input_names=names, output_names=["Y"], dynamo=False)
    elif kind == "actact":
        m = MMActAct().eval()
        inputs = {"A": (1, 1, M, K), "B": (1, 1, K, N)}
        torch.onnx.export(m, (torch.randn(1, 1, M, K), torch.randn(1, 1, K, N)),
                          os.path.join(out_dir, "mm.onnx"),
                          opset_version=17, input_names=["A", "B"], output_names=["Y"], dynamo=False)
    else:                                                     # qdq: explicit int8 QDQ on B
        inputs = {"A": (1, 1, M, K), "B": (1, 1, K, N)}
        build_qdq(out_dir)
    # calibration set
    lines = []
    for i in range(nsamp):
        sub = os.path.join(out_dir, f"s{i:02d}"); os.makedirs(sub, exist_ok=True)
        parts = []
        for nm, shp in inputs.items():
            a = shape_a(nm, torch.randn(*shp, generator=g).numpy().astype("<f4"))
            a.tofile(os.path.join(sub, f"{nm}.raw"))
            parts.append(f"{nm}:=s{i:02d}/{nm}.raw")
        lines.append(" ".join(parts))
    open(os.path.join(out_dir, "calib_list.txt"), "w").write("\n".join(lines) + "\n")
    # one held-out TEST sample + fp32 reference Y = A @ B (for on-device numerical check)
    gt = torch.Generator().manual_seed(99)
    At = shape_a("A", torch.randn(1, 1, M, K, generator=gt).numpy().astype("<f4"))
    Bt = torch.randn(1, 1, K, N, generator=gt).numpy().astype("<f4")
    At.tofile(os.path.join(out_dir, "A.raw")); Bt.tofile(os.path.join(out_dir, "B.raw"))
    np.matmul(At, Bt).astype("<f4").tofile(os.path.join(out_dir, "Y_ref.raw"))
    tl = "A:=A.raw" + ("" if kind == "weight" else " B:=B.raw")
    open(os.path.join(out_dir, "test_list.txt"), "w").write(tl + "\n")
    print(f"emitted {kind} MatMul ({M}x{K} @ {K}x{N}) + {nsamp} calib + 1 test sample -> {out_dir}")


def write_v2_override(out_dir, i8="B"):
    """Compute quantization params DIRECTLY IN TORCH (no qairt calibration) and emit a v2.0.0
    override that hard-pins each tensor's datatype via `output_dtype`:
      i8 tensor  -> int8  SYMMETRIC   (zero_point 0)
      others     -> uint16 ASYMMETRIC (the HTP 16-bit activation port = UFX16)
    v2 schema (output_dtype) is honored per-tensor for activations; v1 `bitwidth` is not."""
    rng = {}
    for s in sorted(glob.glob(os.path.join(out_dir, "s*"))):
        A = np.fromfile(os.path.join(s, "A.raw"), "<f4").reshape(1, 1, M, K)
        B = np.fromfile(os.path.join(s, "B.raw"), "<f4").reshape(1, 1, K, N)
        Y = A @ B
        for nm, t in (("A", A), ("B", B), ("Y", Y)):
            lo, hi = float(t.min()), float(t.max())
            rng[nm] = (min(rng.get(nm, (lo, hi))[0], lo), max(rng.get(nm, (lo, hi))[1], hi))
    in0_mode = os.environ.get("IN0_MODE", "int16sym")         # int16sym | uint16mid | uint16asym
    encs = []
    for nm, (lo, hi) in rng.items():
        amax = max(abs(lo), abs(hi)) or 1.0
        if nm == i8:                                          # in[1] int8 symmetric
            encs.append({"name": nm, "output_dtype": "int8", "y_scale": amax / 127.0, "y_zero_point": 0})
        elif nm == "A":                                       # in[0] activation — the variable under test
            if in0_mode == "int16sym":                        # SFX16, zp=0
                encs.append({"name": nm, "output_dtype": "int16", "y_scale": amax / 32767.0})
            elif in0_mode == "uint16mid":                     # UFX16, zp pinned to midpoint 32768
                encs.append({"name": nm, "output_dtype": "uint16", "y_scale": amax / 32768.0, "y_zero_point": 32768})
            else:                                             # uint16asym: UFX16, zp fit to range
                l2, h2 = min(lo, 0.0), max(hi, 0.0); sc = (h2 - l2) / 65535.0 or 1e-6
                encs.append({"name": nm, "output_dtype": "uint16", "y_scale": sc,
                             "y_zero_point": max(0, min(65535, int(round(-l2 / sc))))})
        else:                                                 # out / others: uint16 asymmetric
            lo, hi = min(lo, 0.0), max(hi, 0.0); scale = (hi - lo) / 65535.0 or 1e-6
            encs.append({"name": nm, "output_dtype": "uint16", "y_scale": scale,
                         "y_zero_point": max(0, min(65535, int(round(-lo / scale))))})
    path = os.path.join(out_dir, "v2_ovr.json")
    json.dump({"version": "2.0.0", "encodings": encs}, open(path, "w"), indent=1)
    print(f"  wrote {path}: " + ", ".join(f"{e['name']}={e['output_dtype']}" for e in encs))


DT = {776: "SFX8", 790: "SFX16", 818: "SFX32", 1032: "UFX8", 1046: "UFX16",
      562: "FP32", 534: "FP16", 1074: "UFX32"}


def check(json_path):
    d = json.load(open(json_path))
    g = d["graph"]; tens = g["tensors"]; nodes = g["nodes"]
    def info(tid):
        t = tens.get(tid, {})
        return DT.get(t.get("data_type"), t.get("data_type")), t.get("dims")
    found = False
    for nid, n in nodes.items():
        if n.get("type") == "MatMul":
            found = True
            ins = n["input_names"]; outs = n["output_names"]
            d0, s0 = info(ins[0]); d1, s1 = info(ins[1]); do, _ = info(outs[0])
            print(f"HTP MatMul {nid}:  in0={d0}{s0}  in1={d1}{s1}  ->  out={do}")
    if not found:
        print("no MatMul node found at HTP layer; op_types:", g.get("op_types"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit"); ap.add_argument("--kind", choices=["weight", "actact", "qdq"], default="weight")
    ap.add_argument("--write-v2", metavar="DIR", help="torch-compute params + emit v2.0.0 override")
    ap.add_argument("--i8", default="B", help="tensor to force int8-symmetric in --write-v2")
    ap.add_argument("--check")
    a = ap.parse_args()
    if a.emit:
        emit(a.emit, a.kind)
    if a.write_v2:
        write_v2_override(a.write_v2, a.i8)
    if a.check:
        check(a.check)
