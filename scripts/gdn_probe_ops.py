#!/usr/bin/env python3
"""Probe HTP's quantized transcendentals (exp, rsqrt/l2norm) in isolation.

Builds a minimal int16 graph computing only exp(g) and l2norm(x) on REAL GDN ranges, runs it
on HTP, and separates two error sources:
  exact      = exp(g) / l2norm(x)            (fp64 truth)
  quantin    = exp(dq(Q(g))) / l2norm(dq(Q(x)))   (what gdn_quant_sim models: exact op of the
               quantized input)
  htp        = device output (dequantized)
If htp ≈ quantin but both differ from exact -> error is just input quantization (sim already
captures it).  If htp differs from quantin -> HTP's fixed-point exp/rsqrt LUT is the culprit
(NOT in the sim) -> explains the ~27% full-graph gap.

Build + emit:  .venv/bin/python scripts/gdn_probe_ops.py --export probe.onnx --emit-io <dir>
Compare:       .venv/bin/python scripts/gdn_probe_ops.py --compare <dir> --result <dev_out> \
                 --gscale <s> --xscale <s>
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_ref_kernel import gdn_chunk, bf16_to_f32, relerr  # noqa
from gdn_onnx_kernel import _golden_chunk_args, CHUNK, EPS  # noqa


class ProbeOps(nn.Module):
    """exp(g) and l2norm(x) — the GDN transcendentals, nothing else."""
    def forward(self, g, x):
        exp_g = torch.exp(g)
        l2_x = x * torch.rsqrt((x * x).sum(-1, keepdim=True) + EPS)
        return exp_g, l2_x


def _real_inputs(golden_dir, layer, prompt, chunk):
    """Real g (cumsum, range ~[-264,0]) and x=query (pre-l2norm) for one golden chunk."""
    files = sorted(glob.glob(os.path.join(golden_dir, f"*_L{layer:02d}.npz")))
    f = [p for p in files if os.path.basename(p).startswith(prompt)] or files
    f = max(files, key=lambda p: int(np.load(p)["query"].shape[1])) if not prompt else f[0]
    q, k, v, gc, beta, S = _golden_chunk_args(f, chunk)
    g = torch.cumsum(gc, dim=-1)                                 # [1,32,64], real decay range
    x = q                                                        # [1,32,64,128] pre-l2norm query
    return g.float(), x.float()


def export(path, golden_dir, layer, prompt, chunk, opset=17):
    g, x = _real_inputs(golden_dir, layer, prompt, chunk)
    torch.onnx.export(ProbeOps().eval(), (g, x), path, opset_version=opset,
                      input_names=["g", "x"], output_names=["exp_g", "l2_x"], dynamo=False)
    print(f"wrote {path}")
    return g, x


def emit_io(out_dir, golden_dir, layer, prompt, chunk, ncalib=12):
    os.makedirs(out_dir, exist_ok=True)
    g, x = _real_inputs(golden_dir, layer, prompt, chunk)
    g.numpy().astype("<f4").tofile(os.path.join(out_dir, "g.raw"))
    x.numpy().astype("<f4").tofile(os.path.join(out_dir, "x.raw"))
    with open(os.path.join(out_dir, "input_list.txt"), "w") as f:
        f.write("g:=g.raw x:=x.raw\n")
    # calibration set (several chunks) so qairt sees the real ranges
    files = sorted(glob.glob(os.path.join(golden_dir, f"*_L{layer:02d}.npz")))
    lines = []
    for i, fp in enumerate(files[:ncalib]):
        sub = os.path.join(out_dir, f"c{i:02d}"); os.makedirs(sub, exist_ok=True)
        cg, cx = _real_inputs(golden_dir, layer, os.path.basename(fp).split("_")[0], 0)
        cg.numpy().astype("<f4").tofile(os.path.join(sub, "g.raw"))
        cx.numpy().astype("<f4").tofile(os.path.join(sub, "x.raw"))
        lines.append(f"g:=c{i:02d}/g.raw x:=c{i:02d}/x.raw")
    open(os.path.join(out_dir, "calib_list.txt"), "w").write("\n".join(lines) + "\n")
    # references
    eg, l2 = ProbeOps()(g.double(), x.double())
    eg.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "exp_g_exact.raw"))
    l2.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "l2_x_exact.raw"))
    print(f"emitted probe IO -> {out_dir}/ (g range [{g.min():.1f},{g.max():.1f}], x abs-max {x.abs().max():.1f})")


def _qsym(t, scale, bits=16):
    qm = (1 << (bits - 1)) - 1
    return torch.clamp(torch.round(t / scale), -qm - 1, qm) * scale


def compare(out_dir, result_dir, gscale=None, xscale=None):
    g = torch.from_numpy(np.fromfile(os.path.join(out_dir, "g.raw"), "<f4").copy())
    x = torch.from_numpy(np.fromfile(os.path.join(out_dir, "x.raw"), "<f4").copy()).reshape(1, 32, 64, 128)
    eg_ex = np.fromfile(os.path.join(out_dir, "exp_g_exact.raw"), "<f4")
    l2_ex = np.fromfile(os.path.join(out_dir, "l2_x_exact.raw"), "<f4")

    def dev(name):
        for c in (os.path.join(result_dir, f"{name}.raw"), os.path.join(result_dir, "Result_0", f"{name}.raw")):
            if os.path.exists(c):
                return np.fromfile(c, "<f4")
        h = glob.glob(os.path.join(result_dir, "**", f"{name}*.raw"), recursive=True)
        return np.fromfile(h[0], "<f4") if h else None

    # quant-input baseline (what the sim models): exact op of the symmetric-quantized input
    g_q = _qsym(g.double(), gscale) if gscale else g.double()
    x_q = _qsym(x.double(), xscale) if xscale else x.double()
    eg_qi, l2_qi = ProbeOps()(g_q, x_q.reshape(1, 32, 64, 128))
    eg_qi = eg_qi.flatten().numpy(); l2_qi = l2_qi.flatten().numpy()

    for name, ex, qi in (("exp_g", eg_ex, eg_qi), ("l2_x", l2_ex, l2_qi)):
        h = dev(name)
        if h is None:
            print(f"  {name}: MISSING"); continue
        e_exact = relerr(torch.from_numpy(h.copy()), torch.from_numpy(ex.copy()))
        e_quantin = relerr(torch.from_numpy(h.copy()), torch.from_numpy(qi.copy()))
        e_inq = relerr(torch.from_numpy(qi.copy()), torch.from_numpy(ex.copy()))
        verdict = "HTP-LUT error!" if e_quantin > 5 * max(e_inq, 1e-4) else "HTP faithful (input-quant only)"
        print(f"  {name}: htp-vs-exact {e_exact:.2e} | htp-vs-quantin {e_quantin:.2e} "
              f"| quantin-vs-exact {e_inq:.2e}  => {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export"); ap.add_argument("--emit-io")
    ap.add_argument("--compare"); ap.add_argument("--result")
    ap.add_argument("--gscale", type=float); ap.add_argument("--xscale", type=float)
    ap.add_argument("--golden", default="tests/gdn/golden")
    ap.add_argument("--layer", type=int, default=0); ap.add_argument("--prompt", default="p00")
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()
    if a.export:
        export(a.export, a.golden, a.layer, a.prompt, a.chunk)
    if a.emit_io:
        emit_io(a.emit_io, a.golden, a.layer, a.prompt, a.chunk)
    if a.compare:
        compare(a.compare, a.result or a.compare, a.gscale, a.xscale)


if __name__ == "__main__":
    main()
