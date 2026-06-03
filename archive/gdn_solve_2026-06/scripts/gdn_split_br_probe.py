#!/usr/bin/env python3
"""M6: C=256 block-recursive triangular inverse T=(I-A)^-1 expressed as a GRAPH of SEPARATE ops so
QNN schedules the HVX diagonal solve and the HMX merge matmuls on DIFFERENT unit-threads (HVX tids
512-515 vs HMX tid 256) so they can OVERLAP.  (Contrast: the prior single fused GdnSolveBR op ran
single-threaded with zero cross-unit overlap.)

Algorithm (nb=4, BL=64, lower-tri inverse of L=I-A, A strictly-lower):
  T_ii = inv(I - A_ii)                                       -- HVX GdnSolve, 1 batched node over H*4
  T_ij = T_ii @ ( sum_{k=j}^{i-1} A_ik @ T_kj )   (i>j)      -- HMX MatMul nodes, grouped by level d=i-j
Off-diag blocks by dependency level:
  d=1: T10,T21,T32 (1 inner term)   d=2: T20,T31 (2 terms)   d=3: T30 (3 terms)

Graph design (intermediates kept VTCM-resident via op chaining, no DDR round-trip):
  Adiag [1,H*4,64,64] uint16 -> GdnSolve -> Tdiag uint16 (4 diag blocks/head).
  All merge operands are int8 (MatMul in[1] must be midpoint-symmetric int8; in[0] free -> we use int8
  on both).  Each merge term A_ik @ T_kj is one MatMul; inner terms summed by Add; final T_ii @ S one more.
  Diagonal T_ii is requantized uint16->int8 once (the converter inserts the requant at the Slice output,
  declared int8).  Off-diag A blocks are int8 graph inputs.

Three graphs at the SAME H (comparable device walls):
  split : Adiag->GdnSolve + 16 merges -> assembled Tout   (HVX + HMX, the M6 graph)
  diag  : Adiag->GdnSolve->Tdiag only                     (HVX-only standalone load)
  merge : the 16 merges only, fed Tdiag(int8) + Aoff direct inputs   (HMX-only standalone load)

Usage: gdn_split_br_probe.py <outdir> <H> <ref_A.raw>
"""
import sys, os, json, numpy as np, onnx
from onnx import helper, TensorProto

outdir, H, refA = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
C = 256; BL = 64; NB = 4
np.random.seed(7)

a0 = np.fromfile(refA, dtype=np.float32).reshape(-1, 64, 64)
def mkA(h):
    big = np.tile(a0[h % a0.shape[0]], (NB, NB))[:C, :C]
    return np.tril(big * 0.7, -1).astype(np.float64)
A = np.stack([mkA(h) for h in range(H)])                       # [H,C,C]
Tref = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])

def blk(M, i, j): return M[:, i*BL:(i+1)*BL, j*BL:(j+1)*BL]    # [H,64,64]

# diagonal blocks -> Adiag [1, H*NB, 64,64] head-major (slot = h*NB + i)
Adiag = np.zeros((H, NB, BL, BL))
for i in range(NB): Adiag[:, i] = blk(A, i, i)
Adiag = Adiag.reshape(1, H*NB, BL, BL)

sA = max(abs(A).max() / 32767.0, 1e-12)
sT = 2.0 / 32767.0
# net-run reads input raws as float32 and the quantized context applies the input quantization from the
# override (do NOT pre-quantize the raws).  Adiag laid out [1,H,NB*64,64] to match the graph input shape.
Adiag.reshape(1, H, NB*BL, BL).astype(np.float32).tofile(os.path.join(outdir, "Adiag.raw"))

offpairs = [(1,0),(2,0),(2,1),(3,0),(3,1),(3,2)]
def s16(x): return max(abs(x).max() / 32767.0, 1e-12)
sAoff = {(i,k): s16(blk(A,i,k)) for (i,k) in offpairs}    # int16-symmetric scale (act operand)
for (i,k) in offpairs:                                    # float32 raw [1,H,64,64]; context quantizes
    blk(A,i,k).reshape(1, H, BL, BL).astype(np.float32).tofile(os.path.join(outdir, f"Aoff_{i}{k}.raw"))

# ---- fp golden of every node (for scale chaining) ----
Tblk = {(i,i): np.stack([np.linalg.inv(np.eye(BL) - blk(A,i,i)[h]) for h in range(H)]) for i in range(NB)}
merge_terms = {(j+d, j): [((j+d,k),(k,j)) for k in range(j, j+d)] for d in range(1,NB) for j in range(NB-d)}
inner_fp = {}; sum_fp = {}
for d in range(1, NB):
    for j in range(NB - d):
        i = j + d; S = np.zeros((H, BL, BL))
        for (aik, tkj) in merge_terms[(i,j)]:
            term = np.matmul(blk(A, aik[0], aik[1]), Tblk[tkj]); inner_fp[(i, aik[1], j)] = term; S += term
        sum_fp[(i,j)] = S; Tblk[(i,j)] = np.matmul(Tblk[(i,i)], S)
def smax(x): return max(abs(x).max() / 127.0, 1e-12)     # int8 storage scale for merge intermediates

# emit Tdiag int8 raw for the merge-only graph (the diagonal blocks the merges consume as wts/acts).
# merge-only uses ONE global int8 scale for the whole Tdiag_i8 input tensor (sTglob); the split graph
# uses per-block scales via the GdnSolve uint16 output requant (more accurate).  merge-only is a
# unit-LOAD probe (its accuracy is not the gate), so a single scale is fine.
sTd = {i: smax(Tblk[(i,i)]) for i in range(NB)}
sTglob = max(sTd.values())
# Tdiag external input (merge-only graph): float32 raw [1,H,NB*64,64]; context quantizes to uint16.
Tdiag_f = np.zeros((H, NB, BL, BL))
for i in range(NB):
    Tdiag_f[:, i] = Tblk[(i,i)]
Tdiag_f.reshape(1, H, NB*BL, BL).astype(np.float32).tofile(os.path.join(outdir, "Tdiag_i8.raw"))

# ---- ONNX builders ----
def vi(name, shape): return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
def ci64(nodes, name, arr):
    nodes.append(helper.make_node("Constant", [], [name],
                  value=helper.make_tensor(name, TensorProto.INT64, [len(arr)], arr)))

def build(emit_solve):
    """Returns (nodes, inputs, outputs, encs).  emit_solve=True -> Adiag->GdnSolve; False -> Tdiag_i8 input."""
    nodes = []; encs = []
    def eu16(n, s): encs.append({"name": n, "output_dtype": "uint16", "y_scale": float(s), "y_zero_point": 32768})
    def ei16(n, s): encs.append({"name": n, "output_dtype": "int16",  "y_scale": float(s), "y_zero_point": 0})
    def ei8(n, s):  encs.append({"name": n, "output_dtype": "int8",  "y_scale": float(s), "y_zero_point": 0})

    if emit_solve:
        # Adiag uint16 external input [1,H,NB*64,64] (16-bit; preserves diagonal-solve precision); reshape
        # to [1,H*NB,64,64] for GdnSolve.  All external inputs are kept 16-bit (Aoff = int16) so net-run
        # reads every raw with a uniform 16-bit element size (mixed 8/16-bit caused a batch-size mismatch).
        eu16("Adiag", sA); eu16("Adiag_r", sA); eu16("Tdiag", sT)
        ci64(nodes, "shp_solve", [1, H*NB, BL, BL])
        nodes.append(helper.make_node("Reshape", ["Adiag", "shp_solve"], ["Adiag_r"], name="rs_Adiag"))
        nodes.append(helper.make_node("GdnSolve", ["Adiag_r"], ["Tdiag"], name="GdnSolve_0", domain="gdn"))
        diag_src = "Tdiag"
    else:
        # merge-only: Tdiag_i8 is a direct uint16 input [1,H,NB*64,64] (16-bit for a uniform input set);
        # the converter requantizes to the per-block int8 the merges consume.  Single global scale.
        eu16("Tdiag_i8", sTglob)
        diag_src = "Tdiag_i8"

    # reshape diag source [1,H*NB,64,64] -> [1,H,NB,64,64], slice per block i -> [1,H,64,64] int8
    ci64(nodes, "shp_HNB", [1, H, NB, BL, BL])
    nodes.append(helper.make_node("Reshape", [diag_src, "shp_HNB"], ["diag5"], name="rs_diag"))
    if emit_solve: eu16("diag5", sT)
    else: eu16("diag5", sTglob)
    ci64(nodes, "shp_H1", [1, H, BL, BL])
    Tii = {}                                  # i -> int8 tensor name [1,H,64,64] (merge operand)
    Tii_u16 = {}                              # i -> uint16 tensor name [1,H,64,64] (assembly block)
    for i in range(NB):
        ci64(nodes, f"st_{i}", [i]); ci64(nodes, f"en_{i}", [i+1]); ci64(nodes, f"ax_{i}", [2]); ci64(nodes, f"sp_{i}", [1])
        s5 = f"Tii_{i}_s5"; o = f"Tii_{i}"; ou = f"Tii_{i}_u16"
        nodes.append(helper.make_node("Slice", ["diag5", f"st_{i}", f"en_{i}", f"ax_{i}", f"sp_{i}"], [s5], name=f"sl_{i}"))
        # int8 alias for merge operand use
        nodes.append(helper.make_node("Reshape", [s5, "shp_H1"], [o], name=f"rs_{i}"))
        # uint16 alias at scale sT for assembly (diagonal block of the output T)
        nodes.append(helper.make_node("Reshape", [s5, "shp_H1"], [ou], name=f"rsu_{i}"))
        sblk = sTd[i] if emit_solve else sTglob
        eu16(s5, sT if emit_solve else sTglob)
        ei8(o, sblk); Tii[i] = o
        eu16(ou, sT); Tii_u16[i] = ou

    # off-diag A blocks: int16 external inputs (16-bit, uniform with the uint16 Adiag/Tdiag so net-run
    # reads all raws at one element size).  Used as MatMul in[0] (act, free dtype) -> w8a16 on HMX.
    for (i,k) in offpairs: ei16(f"Aoff_{i}{k}", sAoff[(i,k)])

    # internal merge intermediates stay int8 (small terms); the FINAL off-diag T_ij and all assembled
    # blocks are uint16 at the T scale sT (|T|~1-2 needs >8-bit dynamic range; int8 at sT would clip).
    def matmul_i8(out, act, wt, sout):    # int8 intermediate output
        nodes.append(helper.make_node("MatMul", [act, wt], [out], name="mm_"+out)); ei8(out, sout)
    def matmul_u16(out, act, wt, sout):   # uint16 result output (holds |T|~1-2)
        nodes.append(helper.make_node("MatMul", [act, wt], [out], name="mm_"+out)); eu16(out, sout)

    # int8 copies of the diagonal T_ii to feed as MatMul wt operands (in[1] must be symmetric int8).
    # Tii[i] is uint16; declare a requantized int8 alias used only as a wt.
    Tw = {}
    for i in range(NB):
        Tw[i] = Tii[i]  # Tii already int8 (declared above) -> usable as wt
    Tname = {(i,i): Tii[i] for i in range(NB)}     # int8 diag (act/wt operand for merges)
    for d in range(1, NB):
        for j in range(NB - d):
            i = j + d
            term_outs = []
            for (aik, tkj) in merge_terms[(i,j)]:
                # A_ik @ T_kj : act = Aoff (int16, in[0] free), wt = T_kj (int8, in[1] symmetric)
                tout = f"term_{i}{j}_{aik[1]}"
                matmul_i8(tout, f"Aoff_{aik[0]}{aik[1]}", Tname[tkj], smax(inner_fp[(i, aik[1], j)]))
                term_outs.append(tout)
            if len(term_outs) == 1:
                Sname = term_outs[0]
            else:
                acc = term_outs[0]
                for n2 in range(1, len(term_outs)):
                    outn = f"sum_{i}{j}_{n2}"
                    nodes.append(helper.make_node("Add", [acc, term_outs[n2]], [outn], name="ad_"+outn))
                    ei8(outn, smax(sum_fp[(i,j)])); acc = outn
                Sname = acc
            # T_ij = T_ii @ S : act = T_ii (int8, in[0]), wt = S (int8, in[1]); result uint16 (assembled block)
            outT = f"T_{i}{j}"
            matmul_u16(outT, Tii[i], Sname, sT)
            Tname[(i,j)] = outT

    # Output the 10 lower-tri blocks stacked on the head axis -> Tout [1, ntri*H, 64, 64] uint16, and
    # reconstruct the assembled T host-side.  (Avoids a large [1,H,256,256] Concat that fails ctxgen at
    # higher H; the full-C assembly is pure output glue, not part of the HVX/HMX compute under test.)
    tri = [(i,j) for i in range(NB) for j in range(i+1)]   # 10 blocks, row-major lower-tri
    blocks = []
    for (i,j) in tri:
        blocks.append(Tii_u16[i] if i == j else Tname[(i,j)])
    nodes.append(helper.make_node("Concat", blocks, ["Tout"], axis=1, name="cat_Tout")); eu16("Tout", sT)

    aoff = [vi(f"Aoff_{i}{k}", [1,H,BL,BL]) for (i,k) in offpairs]
    # ALL external graph inputs are int8 (net-run reads every raw as int8 uniformly; mixing a uint16
    # input made it misread the int8 raws -> "batch size" mismatch).  Adiag enters as int8 and the
    # converter requantizes int8->uint16 at the GdnSolve input boundary (declared uint16 in the override).
    ntri = NB*(NB+1)//2
    if emit_solve:
        inputs = [vi("Adiag", [1, H, NB*BL, BL])] + aoff
    else:
        inputs = [vi("Tdiag_i8", [1, H, NB*BL, BL])] + aoff
    return nodes, inputs, [vi("Tout", [1, ntri*H, BL, BL])], encs

def diag_only():
    nodes = []; encs = []
    encs.append({"name":"Adiag","output_dtype":"uint16","y_scale":float(sA),"y_zero_point":32768})
    encs.append({"name":"Tdiag","output_dtype":"uint16","y_scale":float(sT),"y_zero_point":32768})
    nodes.append(helper.make_node("GdnSolve", ["Adiag"], ["Tdiag"], name="GdnSolve_0", domain="gdn"))
    return nodes, [vi("Adiag", [1,H*NB,BL,BL])], [vi("Tdiag", [1,H*NB,BL,BL])], encs

def save(name, nodes, inputs, outputs, encs):
    m = helper.make_model(helper.make_graph(nodes, name, inputs, outputs),
                          opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
    onnx.save(m, os.path.join(outdir, name + ".onnx"))
    json.dump({"version": "2.0.0", "encodings": encs}, open(os.path.join(outdir, name + ".ovr.json"), "w"))

save("split", *build(emit_solve=True))
save("merge", *build(emit_solve=False))
save("diag",  *diag_only())

# golden assembled split-T (fp math) for the accuracy comparison
Tsplit = np.zeros((H, C, C))
for i in range(NB):
    for j in range(i+1):
        Tsplit[:, i*BL:(i+1)*BL, j*BL:(j+1)*BL] = Tblk[(i,j)]
Tref.astype(np.float32).tofile(os.path.join(outdir, "Tref.raw"))

rel = np.linalg.norm(Tsplit - Tref) / np.linalg.norm(Tref)
nM = sum(1 for n in build(True)[0] if n.op_type=="MatMul")
print(f"H={H} C={C} NB={NB}: split fp-math relerr vs np.linalg.inv={rel:.3e} (structure check)")
print(f"  GdnSolve diag batch=[1,{H*NB},{BL},{BL}]  {len(offpairs)} off-diag A inputs  {nM} MatMuls")
