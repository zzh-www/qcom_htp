#!/usr/bin/env python3
"""Probe a 2-deep int16 MatMul CHAIN on HTP vs the float-then-quantize simulator.

The full-graph all-integer device output is correct (corr 0.97) but ~27% noisy, while the
simulator predicts 3e-3. This isolates whether HTP's quantized int16 MatMul CHAIN (int32
accumulate + requant + the auto bias / zero-point correction) diverges from the sim's
quantize-each-output model. Real GDN magnitudes (k_beta, k, v of a golden chunk).

  Y1 = A @ B^T   (A=k_beta [.,64,128], B=k [.,64,128] -> Y1 [.,64,64])
  Y2 = Y1 @ C    (C=v [.,64,128]      -> Y2 [.,64,128])

Compare device Y2 to: exact (fp64), and sim (symmetric int16 quantize each operand+output).
"""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_ref_kernel import bf16_to_f32, relerr            # noqa
from gdn_onnx_kernel import _golden_chunk_args, EPS        # noqa


class Chain(nn.Module):
    def forward(self, A, B, C):
        Y1 = torch.matmul(A, B.transpose(-1, -2))
        Y2 = torch.matmul(Y1, C)
        return Y1, Y2


def _abc(golden_dir, layer, prompt, chunk):
    files = sorted(glob.glob(os.path.join(golden_dir, f"*_L{layer:02d}.npz")))
    f = [p for p in files if os.path.basename(p).startswith(prompt)] or files
    f = f[0]
    q, k, v, gc, beta, S = _golden_chunk_args(f, chunk)
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)   # l2norm(k)
    k_beta = kn * beta.unsqueeze(-1)
    return k_beta.float(), kn.float(), v.float()               # A, B, C


def export(path, gd, layer, prompt, chunk):
    A, B, C = _abc(gd, layer, prompt, chunk)
    torch.onnx.export(Chain().eval(), (A, B, C), path, opset_version=17,
                      input_names=["A", "B", "C"], output_names=["Y1", "Y2"], dynamo=False)
    print(f"wrote {path}")


def emit_io(out, gd, layer, prompt, chunk, ncal=12):
    os.makedirs(out, exist_ok=True)
    A, B, C = _abc(gd, layer, prompt, chunk)
    for n, t in (("A", A), ("B", B), ("C", C)):
        t.numpy().astype("<f4").tofile(os.path.join(out, f"{n}.raw"))
    open(os.path.join(out, "input_list.txt"), "w").write("A:=A.raw B:=B.raw C:=C.raw\n")
    files = sorted(glob.glob(os.path.join(gd, f"*_L{layer:02d}.npz")))
    lines = []
    for i, fp in enumerate(files[:ncal]):
        sub = os.path.join(out, f"c{i:02d}"); os.makedirs(sub, exist_ok=True)
        a, b, c = _abc(gd, layer, os.path.basename(fp).split("_")[0], 0)
        for n, t in (("A", a), ("B", b), ("C", c)):
            t.numpy().astype("<f4").tofile(os.path.join(sub, f"{n}.raw"))
        lines.append(f"A:=c{i:02d}/A.raw B:=c{i:02d}/B.raw C:=c{i:02d}/C.raw")
    open(os.path.join(out, "calib_list.txt"), "w").write("\n".join(lines) + "\n")
    Y1, Y2 = Chain()(A.double(), B.double(), C.double())
    Y2.float().numpy().astype("<f4").tofile(os.path.join(out, "Y2_exact.raw"))
    # sim: symmetric int16 quantize each operand + each matmul output (ranges from calib)
    rng = {}
    for fp in files[:ncal]:
        a, b, c = _abc(gd, layer, os.path.basename(fp).split("_")[0], 0)
        y1 = torch.matmul(a.double(), b.double().transpose(-1, -2)); y2 = torch.matmul(y1, c.double())
        for n, t in (("A", a), ("B", b), ("C", c), ("Y1", y1), ("Y2", y2)):
            m = float(t.abs().max()); rng[n] = max(rng.get(n, 0), m)
    def qs(t, n):
        s = rng[n] / 32767; return torch.clamp(torch.round(t / s), -32768, 32767) * s
    a, b, c = qs(A.double(), "A"), qs(B.double(), "B"), qs(C.double(), "C")
    y1 = qs(torch.matmul(a, b.transpose(-1, -2)), "Y1"); y2 = qs(torch.matmul(y1, c), "Y2")
    y2.float().numpy().astype("<f4").tofile(os.path.join(out, "Y2_sim.raw"))
    print(f"emitted chain IO -> {out}/ (A absmax {A.abs().max():.2f} Y2 absmax {Y2.abs().max():.2f})")


def compare(out, result):
    ex = np.fromfile(os.path.join(out, "Y2_exact.raw"), "<f4")
    sim = np.fromfile(os.path.join(out, "Y2_sim.raw"), "<f4")
    h = None
    for c in (os.path.join(result, "Y2.raw"), os.path.join(result, "Result_0", "Y2.raw")):
        if os.path.exists(c): h = np.fromfile(c, "<f4")
    if h is None:
        hh = glob.glob(os.path.join(result, "**", "Y2*.raw"), recursive=True); h = np.fromfile(hh[0], "<f4") if hh else None
    if h is None:
        print("  Y2: MISSING"); return
    t = lambda x: torch.from_numpy(x.copy())
    print(f"  Y2: htp-vs-exact {relerr(t(h),t(ex)):.2e} | htp-vs-SIM {relerr(t(h),t(sim)):.2e} "
          f"| sim-vs-exact {relerr(t(sim),t(ex)):.2e}")
    print(f"      => {'HTP int16 chain DIVERGES from sim (chain execution is the gap)' if relerr(t(h),t(sim))>0.05 else 'HTP chain matches sim'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export"); ap.add_argument("--emit-io"); ap.add_argument("--compare"); ap.add_argument("--result")
    ap.add_argument("--golden", default="tests/gdn/golden"); ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="p00"); ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()
    if a.export: export(a.export, a.golden, a.layer, a.prompt, a.chunk)
    if a.emit_io: emit_io(a.emit_io, a.golden, a.layer, a.prompt, a.chunk)
    if a.compare: compare(a.compare, a.result or a.compare)


if __name__ == "__main__":
    main()
