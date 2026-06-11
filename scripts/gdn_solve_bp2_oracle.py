#!/usr/bin/env python3
"""BP2/BP4 byte-pass merge precision oracle — predicts device oc BEFORE implementing on device.

Route: keep the 313-cyc u8i8 kernel; precision comes from byte-decomposing the OPERANDS,
combining the per-pass int8 drains on HVX (gdn_solve.md s4 supertile route).

Modes (mirror the GdnSolveBR16.cpp static chain exactly: scales / g1 / clips):
  u8i8: act u8@sAa, wt i8@sTw, K-stack drain @sP=128*556*d*sAa*sTw/127, final force@sTw -> T codes i8.
  bp2 : act 16-bit (q15@sAa/256 -> Ah/Al, both passes drain at the same g1, Sacc16=256*Ph+Pl @sP/256);
        wt stays i8 (T codes16 @sTw/256 >>8); final merge 2-pass (T_ii q15 split) -> T codes16 @sTw/256.
  bp4 : bp2 + wt 16-bit (Wh/Wl, drop the LL pass: 3 passes inner, 3 passes final).
oc = e2e GDN chunk relerr vs fp64 (same metric as device 1.17e-2), all heads of one real chunk.

Reproduce: source scripts/env.sh && python scripts/gdn_solve_bp2_oracle.py
"""
import os, sys, glob
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import gdn_onnx_kernel as gok; gok.CHUNK = 256
import gdn_ref_kernel as grk; grk.CHUNK = 256
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import GOLDEN

C, BL, NB = 256, 64, 4
sAa, sTw, sTa, COLABS, sSacc = 4.895068e-03, 7.874256e-03, 7.874256e-03, 556, 3.742e-03
TI = 2.0 / 32767.0
sTw16, sAa16, sSacc16 = sTw / 256, sAa / 256, sSacc / 256

npz = next(f for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))) if np.load(f)["query"].shape[1] >= C)
qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]
H = A.shape[0]
Texact = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])
print(f"golden={os.path.basename(npz)} H={H}")

def gdn_out(T):
    dt = torch.float64
    q, k, v = qc.to(dt), kc.to(dt), vc.to(dt); gg = torch.cumsum(gc.to(dt), dim=-1)
    qn = (q / (q.norm(dim=-1, keepdim=True) + 1e-12)) * (1.0 / (q.shape[-1] ** 0.5))
    knn = k / (k.norm(dim=-1, keepdim=True) + 1e-12)
    v_beta = v * betac.to(dt).unsqueeze(-1); kb = knn * betac.to(dt).unsqueeze(-1)
    z = torch.zeros((), dtype=dt)
    trl = torch.tril(torch.ones(C, C, dtype=torch.bool))
    dec = torch.exp(torch.where(trl, gg.unsqueeze(-1) - gg.unsqueeze(-2), z)) * trl.to(dt)
    attn = torch.from_numpy(T).unsqueeze(0)
    U = attn @ v_beta; W = attn @ (kb * torch.exp(gg).unsqueeze(-1))
    P = (qn @ knn.transpose(-1, -2)) * dec
    P = P.masked_fill(torch.triu(torch.ones(C, C, dtype=torch.bool), 1), 0.0)
    v_new = U - W @ S_in.to(dt)
    return (qn * torch.exp(gg).unsqueeze(-1)) @ S_in.to(dt) + P @ v_new

O_ref = gdn_out(Texact)

def split16(q15):                     # int16 codes -> hi/lo both zp128 (q15 = 256*Ah + Al + 128; the
    u = q15.astype(np.int64) + 32768  # +128 constant rides the eff term on device -> exact)
    return (u >> 8) - 128, (u & 0xFF) - 128

def drain(raw, g1):                   # int8 output drain (round + clip)
    return np.clip(np.round(raw * g1), -127, 127)

def solve(h, mode):
    blk = lambda M, i, j: M[i*BL:(i+1)*BL, j*BL:(j+1)*BL]
    Tii = [np.round(np.linalg.inv(np.eye(BL) - blk(A[h], i, i)) / TI).clip(-32767, 32767) for i in range(NB)]
    Tw = {}                           # off-diag T codes: u8i8 -> i8 @sTw; bp -> i16 @sTw16
    for d in range(1, NB):
        for j in range(NB - d):
            i = j + d
            maxP = 128.0 * COLABS * d; g1 = 127.0 / maxP; sP = maxP * sAa * sTw / 127.0
            rawHH = np.zeros((BL, BL)); rawLH = np.zeros((BL, BL)); rawHL = np.zeros((BL, BL))
            raw8 = np.zeros((BL, BL)); csum = np.zeros(BL)
            for k in range(j, i):
                if mode == "u8i8":
                    W8 = np.round(Tii[k] * TI / sTw).clip(-127, 127) if k == j else Tw[(k, j)]
                    a8 = np.round(blk(A[h], i, k) / sAa).clip(-127, 127)
                    raw8 += a8 @ W8
                    continue
                w16 = np.round(Tii[k] * TI / sTw16).clip(-32639, 32639) if k == j else Tw[(k, j)]
                Ah_, Al_ = split16(np.round(blk(A[h], i, k) / sAa16).clip(-32767, 32767))
                if mode == "bp2":
                    W8 = np.round(w16 / 256).clip(-127, 127)
                    rawHH += Ah_ @ W8; rawLH += Al_ @ W8; csum += 128 * W8.sum(axis=0)
                else:                  # bp4: wt byte-split, LL pass dropped; +128 const exact post-drain
                    Wh = np.round(w16 / 256).clip(-127, 127); Wl = w16 - 256 * Wh
                    rawHH += Ah_ @ Wh; rawLH += Al_ @ Wh; rawHL += Ah_ @ Wl; csum += 128 * Wh.sum(axis=0)
            if mode == "u8i8":
                Sacc = drain(raw8, g1); sS = sP
            else:
                B, LI = 4, 4                            # B: hi drain boost (16-bit Sacc absorbs it; B=8 clips
                gb = g1 * B                             # some heads, max code 232). LI: lo-act passes carry
                Sacc = (256 * drain(rawHH, gb)          # ~2x the hi raw (uniform bytes) -> own gain gb/LI.
                        + LI * drain(rawLH, gb / LI) + 8 * drain(rawHL, gb / 8) + np.round(csum * gb))
                Sacc = np.clip(Sacc, -32767, 32767); sS = sP / (256 * B)
            # ---- final merge: T_ij = T_ii @ Sacc, drains forced so hi pass lands @sTw ----
            if mode == "u8i8":
                Wq = np.round(Sacc * sS / sSacc).clip(-127, 127)
                a = np.round(Tii[i] * TI / sTa).clip(-127, 127)
                Tw[(i, j)] = drain(a @ Wq, (sTa * sSacc) / sTw)
            else:
                Wq16 = np.round(Sacc * sS / sSacc16).clip(-32639, 32639)
                Ah_, Al_ = split16(Tii[i])
                gH = (256 * TI) * sSacc16 / sTw           # hi-act x hi-wt -> codes @sTw
                LF = 32                                       # final lo-act raw ~20x hi raw -> gain /LF
                if mode == "bp2":
                    Wh = np.round(Wq16 / 256).clip(-127, 127); Wl = np.zeros_like(Wh)
                else:
                    Wh = np.round(Wq16 / 256).clip(-127, 127); Wl = Wq16 - 256 * Wh
                Oh = drain(Ah_ @ Wh, gH * 256)
                Ol = LF * drain(Al_ @ Wh, gH * 256 / LF) + 8 * drain(Ah_ @ Wl, gH * 32) \
                   + np.round(128 * Wh.sum(axis=0) * gH * 256)
                Tw[(i, j)] = np.clip(256 * Oh + Ol, -32639, 32639)
    T = np.zeros((C, C))
    for i in range(NB): T[i*BL:(i+1)*BL, i*BL:(i+1)*BL] = Tii[i] * TI
    sT = sTw if mode == "u8i8" else sTw16
    for (i, j), cd in Tw.items(): T[i*BL:(i+1)*BL, j*BL:(j+1)*BL] = cd * sT
    return T

for mode in ("u8i8", "bp2", "bp4"):
    Tq = np.stack([solve(h, mode) for h in range(H)])
    oc = float((gdn_out(Tq) - O_ref).norm() / (O_ref.norm() + 1e-12))
    Tre = float(np.linalg.norm(Tq - Texact) / np.linalg.norm(Texact))
    print(f"{mode:5s}  oc={oc:.4e}  Trelerr={Tre:.4e}")
