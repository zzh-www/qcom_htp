#!/usr/bin/env python3
"""End-to-end precision of the user's pure-HMX D&C scheme on REAL 256x256 GDN data.

Block-recursive triangular inverse of L=I-A (C=256, 4x4 of 64-blocks):
  diagonal T_ii  = Taylor(p=3)+Newton(K) on A_ii            (the user's scheme)
  off-diag  T_ij = T_ii @ Sum_{k=j}^{i-1} A_ik @ T_kj       (block forward-subst merge)
Every matmul + every stored iterate is quantized to int16 the way the w16a16 HMX
path forces it (int16 inputs -> int32 acc EXACT -> drain back to int16). We test
two int16 granularities: single global scale (what one act-scale gives) and the
most generous per-ROW scale, plus fp32 as the ceiling. Metric = relerr(T,T_true)
per head, and the worst diagonal block, on real chunks. C=256.
"""
import sys, os, glob, argparse
import numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts',1)[0]
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--samples", type=int, default=40)
ap.add_argument("--newton", type=int, default=4); a = ap.parse_args()
import gdn_onnx_kernel as gok; gok.CHUNK = 256
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
C = 256; BL = 64; NB = 4
tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
def buildA(npz):
    qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1,1,C,C)).squeeze(-2)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff*tl)*tl
    return ((-torch.matmul(k_beta, kn.transpose(-1,-2))*decay)*sl).double().numpy()[0]
def relerr(x,y): return np.linalg.norm(x-y)/(np.linalg.norm(y)+1e-12)
def q_global(x,nb):
    if nb is None: return x
    lim=(1<<(nb-1))-1; m=np.abs(x).max()
    return x if m==0 else np.round(x/(m/lim)).clip(-lim,lim)*(m/lim)
def q_perrow(x,nb):
    if nb is None: return x
    lim=(1<<(nb-1))-1; m=np.abs(x).max(axis=1,keepdims=True); m[m==0]=1
    return np.round(x/(m/lim)).clip(-lim,lim)*(m/lim)
def q_bf16(x,nb):                       # round-to-nearest-even bf16 (8-bit exp like fp32 -> huge range)
    u=x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u=((u+0x7FFF+((u>>16)&1))&0xFFFF0000).astype(np.uint32)
    return u.view(np.float32).astype(np.float64)
G_CLIPB=[1.0]                           # per-head clip bound (set to max|T_true|, oracle best-case)
def q_clip(x,nb):                       # int16 with FIXED scale B/lim + saturate (user's "clip the big values")
    lim=(1<<(nb-1))-1; B=G_CLIPB[0]; s=B/lim
    return np.round(np.clip(x,-B,B)/s).clip(-lim,lim)*s
def tn_inv(Aii,p,K,q):                 # Taylor(p)+Newton(K) 64-block inverse, quant q each store
    n=Aii.shape[0]; I=np.eye(n); A=q(Aii,16) if q!=(lambda x,nb:x) else Aii
    X=I.copy()
    for _ in range(p): X=q(I+A@q(X,16),16)
    M=I-A; pk=np.abs(X).max()
    for _ in range(K):
        MX=q(M@q(X,16),16); X=q(q(X,16)@q(2*I-MX,16),16); pk=max(pk,np.abs(X).max())
    return X,pk
def solve_dc(A,p,K,q):                  # full block-recursive C=256 inverse, the user's scheme
    T=np.zeros((C,C)); worstpk=0
    blk=lambda M,i,j: M[i*BL:(i+1)*BL, j*BL:(j+1)*BL]
    Tii=[]
    for i in range(NB):
        Xi,pk=tn_inv(blk(A,i,i),p,K,q); Tii.append(Xi); worstpk=max(worstpk,pk)
        T[i*BL:(i+1)*BL,i*BL:(i+1)*BL]=Xi
    for d in range(1,NB):
        for j in range(NB-d):
            i=j+d; Sacc=np.zeros((BL,BL))
            for k in range(j,i):
                Sacc=q(Sacc+q(blk(A,i,k)@blk(T,k,j),16),16)
            Tij=q(Tii[i]@Sacc,16)
            T[i*BL:(i+1)*BL, j*BL:(j+1)*BL]=Tij
    return T,worstpk
import os as _os
files=[f for f in sorted(glob.glob(os.path.join(ROOT,"tests/gdn/golden/*.npz"))) if np.load(f)["query"].shape[1]>=C]
samp=files[::max(1,len(files)//a.samples)]
# Precompute A + true inverse ONCE per head (was the slowness: buildA re-ran per head per config).
HA=[]
for f in samp:
    Af=buildA(f)
    for h in range(Af.shape[0]):
        A=Af[h]; HA.append((A, np.linalg.inv(np.eye(C)-A), max(np.abs(np.linalg.inv(np.eye(C)-A)).max(),1e-6)))
print(f"{len(HA)} real heads cached, C=256")
identity=lambda x,nb:x
if _os.environ.get("SWEEP_NEWTON"):
    print(f"Taylor3 + Newton×K, int16(w16a16) global-scale end-to-end  (relerr vs fp64)")
    print(f"{'K':>3} {'#mm/head':>9} | {'int16 median':>13} {'p90':>9} {'max':>10} {'%heads<1.4e-2':>13}")
    for K in [1,2,3,4,5,6]:
        res=np.array([relerr(solve_dc(A,3,K,q_global)[0],Tt) for A,Tt,_ in HA]); mm=4*(3+2*K)+20
        print(f"{K:>3} {mm:>9} | {np.median(res):>13.2e} {np.percentile(res,90):>9.2e} {res.max():>10.2e} {100*np.mean(res<1.4e-2):>12.1f}%")
    raise SystemExit(0)
print(f"C=256, Taylor3+Newton x{a.newton}  (relerr vs fp64-exact inverse)")
print(f"{'compute':>20} | {'relerr median':>14} {'p90':>9} {'max':>9} | {'%heads<1.4e-2':>13} {'iter peak':>10}")
for name,q in [("fp32",identity),("bf16",q_bf16),("int16 global",q_global),("int16 clip(oracle B)",q_clip)]:
    res=[]; pks=[]
    for A,Tt,B in HA:
        G_CLIPB[0]=B   # oracle clip bound: only the overshoot ever saturates, never the true answer
        T,pk=solve_dc(A,3,a.newton,q); res.append(relerr(T,Tt)); pks.append(pk)
    res=np.array(res)
    print(f"{name:>20} | {np.median(res):>14.2e} {np.percentile(res,90):>9.2e} {res.max():>9.2e} | {100*np.mean(res<1.4e-2):>12.1f}% {max(pks):>10.1e}")
