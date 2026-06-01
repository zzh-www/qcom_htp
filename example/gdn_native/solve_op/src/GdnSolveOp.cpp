/*
 * GdnSolveOp.cpp — QNN/QHPI custom op "GdnSolve": T = (I - A)^-1 for a strictly-lower A.
 *
 *   input[0]  A : QInt16  [B,H,C,C]  (per-tensor scale sA = quant_parameters.stepsize)
 *   output[0] T : QInt16  [B,H,C,C]  (per-tensor scale sT)
 *
 * Why a custom op: QNN has no native matrix-inverse, so the pure-graph route fakes it with an
 * int8-matmul Neumann chain (iterated Ap@Ap, requant every step) → T's value degrades to ~4.4e-3.
 * This op does the inverse the KDA way (fla/ops/kda/chunk_intra.py): per-BL(=16) diagonal-block
 * forward substitution (int16 codes, int32 accumulation, requant once per row) + block-triangular
 * merge.  Internal arithmetic is int16/int32; we control where requant happens (NOT every matmul),
 * so T comes out near-exact (host bit-model scripts/gdn_solve_int16_model.py: T relerr 3.6e-5,
 * int32 accumulator peak 5.4e8 < 2^31).  Bit-faithful to that golden + gdn_solve_ref.c.
 *
 * Scope: solve only (A->T).  Upstream A build + downstream U/W consume stay in the QNN graph;
 * downstream casts T to int8 as needed.
 */
#include "HTP/core/qhpi.h"
#include "gdn_solve_core.h"   /* scalar gdn_solve_head_q() — host-validated golden + x86 fallback */

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

#define GDN_MAX_SLICES 8     /* per-slice scratch slots (>= HVX threads); avoids races + DSP stack */

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hexagon_protos.h>
/*
 * HVX path: full C=64 forward substitution T[i,:] = e_i + sum_{k<i} A[i,k]*T[k,:], vectorized over the
 * 64 columns (2 IEEE-fp32 HVX vectors per row), int accumulation in fp32 (the precision is set by the
 * int16-quantized A and int16 output T — the accumulator type is immaterial for these bounded values,
 * and fp32 avoids the int16 even/odd width-double shuffle).  One Tf scratch per slice (thread).
 */
static int32_t __attribute__((aligned(128))) g_Tc[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];   /* int32 T codes (scale sT) */
static int32_t __attribute__((aligned(128))) g_Afx[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];  /* folded A codes (scale 2^-GDN_F); low 16b used */

/* Per-CALL scratch ownership.  qhpi_slice_number is 0-based PER op-instance, so concurrent tile-ops
 * (the central tiler splits H into several tiles that run on different HVX threads at the same time)
 * all see slice 0 and would race on g_Tf[0]/g_Af[0].  Claim a globally-unique free slot per call
 * (atomic CAS) and release it at the end — distinct concurrent threads always get distinct scratch. */
static volatile int g_slot_busy[GDN_MAX_SLICES];
static int gdn_claim_slot(void) {
    for (;;)
        for (int s = 0; s < GDN_MAX_SLICES; ++s)
            if (__sync_bool_compare_and_swap(&g_slot_busy[s], 0, 1)) return s;
}
static inline void gdn_free_slot(int s) { __sync_lock_release(&g_slot_busy[s]); }

#define GDN_F 15   /* fold scale: Afx = round(A*2^GDN_F); keeps |Afx| < 2^15 (valid int16 mul operand) */

/* HVX kernel — PURE INTEGER (no fp internal).  A,T are int16 codes; the per-row requant ×sA is FOLDED
 * into A as a power-of-2 fixed-point (Afx = round(A·2^F)), so requant is an arithmetic shift, not a
 * multiply.  AXPY: int16×int16 -> int32 via Q6_Vw_vmpyiacc_VwVwRh (A scalar in a register, lane-clean,
 * no splat/widen/even-odd-shuffle).  Requant: Tc = (acc + 2^(F-1)) >> F (Q6_Vw_vasr).  T is genuinely
 * lower-triangular so acc[c>i]=0 -> no explicit mask.  int32 acc peak ~5e8 < 2^31 (host-validated). */
static void gdn_solve_head_hvx(const uint16_t *Au, int C, int zpA, float sA,
                               float sT, int zpT, uint16_t *Tu, int slot) {
    int32_t *Tc = g_Tc[slot];
    int32_t *Afx = g_Afx[slot];
    const int NV = (C + 31) / 32;

    /* 1) fold A -> int32 codes at scale 2^-F: Afx = round((Au-zpA)*sA*2^F).  The float scale sA*2^F is
     *    turned into a fixed-point multiplier (M*2^-S, M in int16 range) ONCE per call (standard quantized-
     *    kernel setup; the only float, scalar not vector); the per-element path is pure integer:
     *    Afx = (A_code*M + 2^(S-1)) >> S.  vzxt splits even/odd cols, vshuff restores order. */
    {
        float sf = sA * (float)(1 << GDN_F);              /* fold factor; normalize to M in [2^14, 2^15) */
        int S = 14;
        while (S < 30 && sf * (float)(1 << (S + 1)) < 30000.0f) ++S;
        while (S >  0 && sf * (float)(1 <<  S)      > 32760.0f) --S;
        const int M = (int)(sf * (float)(1 << S) + 0.5f);
        const int Mrep = (M & 0xFFFF) * 0x10001;          /* vmpyi_VwRh uses both scalar halfwords */
        const HVX_Vector vzp = Q6_V_vsplat_R(zpA), vrndS = Q6_V_vsplat_R(1 << (S - 1));
        const HVX_UVector *Av = (const HVX_UVector *)Au;
        HVX_Vector *Afxv = (HVX_Vector *)Afx;             /* 32 int32 / vector */
        const int nblk = (C*C) / 64;
        for (int b = 0; b < nblk; ++b) {
            HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(Av[b]);    /* uint16 -> int32 (even/odd cols) */
            HVX_Vector c0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), vzp);   /* A code (int32) */
            HVX_Vector c1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), vzp);
            HVX_Vector i0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c0, Mrep), vrndS), S);
            HVX_Vector i1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c1, Mrep), vrndS), S);
            HVX_VectorPair s = Q6_W_vshuff_VVR(i1, i0, -4); /* even/odd -> sequential cols */
            Afxv[2*b]   = Q6_V_lo_W(s);
            Afxv[2*b+1] = Q6_V_hi_W(s);
        }
    }

    /* 2) forward substitution: acc = Σ_{k<i} Afx[i,k]·Tc[k,:] (int32), Tc[i,:] = (acc + 2^(F-1)) >> F.
     *    vmpyi_VwRh consumes BOTH halfwords of the scalar reg (even lanes h0, odd lanes h1), so the
     *    A code is replicated into both halves (x * 0x10001) — otherwise odd columns multiply by 0. */
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (GDN_F - 1));
    const int ei = (int)(1.0f / sT + 0.5f);               /* diagonal e_i code = round(1/sT) */
    const int two = (NV > 1);                             /* C=64 -> 2 col-vectors; C<=32 -> 1 */
    for (int i = 0; i < C; ++i) {
        /* SCALAR (not array) accumulators so they stay in the VRF across the k-loop — an a0[2]/a1[2]
         * array is spilled to stack + reloaded every MAC (~2.5 pkt/MAC); named locals -> ~1 pkt/MAC.
         * Two accumulators (k-even/k-odd) per col-vector for ILP; col-vec 1 only used when C>32. */
        HVX_Vector e0 = Q6_V_vzero(), o0 = Q6_V_vzero(), e1 = Q6_V_vzero(), o1 = Q6_V_vzero();
        int k = 0;
        for (; k + 1 < i; k += 2) {
            int s0 = (Afx[i*C + k] & 0xFFFF) * 0x10001, s1 = (Afx[i*C + k + 1] & 0xFFFF) * 0x10001;
            const HVX_Vector *T0 = (const HVX_Vector *)(Tc + k*C), *T1 = (const HVX_Vector *)(Tc + (k+1)*C);
            e0 = Q6_Vw_vmpyiacc_VwVwRh(e0, T0[0], s0);
            o0 = Q6_Vw_vmpyiacc_VwVwRh(o0, T1[0], s1);
            if (two) { e1 = Q6_Vw_vmpyiacc_VwVwRh(e1, T0[1], s0);
                       o1 = Q6_Vw_vmpyiacc_VwVwRh(o1, T1[1], s1); }
        }
        for (; k < i; ++k) {
            int s0 = (Afx[i*C + k] & 0xFFFF) * 0x10001;
            const HVX_Vector *T0 = (const HVX_Vector *)(Tc + k*C);
            e0 = Q6_Vw_vmpyiacc_VwVwRh(e0, T0[0], s0);
            if (two) e1 = Q6_Vw_vmpyiacc_VwVwRh(e1, T0[1], s0);
        }
        HVX_Vector *Ti = (HVX_Vector *)(Tc + i*C);
        Ti[0] = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(e0, o0), vrnd), GDN_F);
        if (two) Ti[1] = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(e1, o1), vrnd), GDN_F);
        Tc[i*C + i] += ei;                                /* + e_i (diagonal); acc[c>i]==0 already */
    }

    /* 3) output: Tu = Tc + zpT, narrow int32->uint16 saturating.  Upper-tri Tc==0 -> zpT (correct). */
    const HVX_Vector vzpT = Q6_V_vsplat_R(zpT);
    for (int r = 0; r < C; ++r) {
        HVX_Vector q0 = Q6_Vw_vadd_VwVw(*(HVX_Vector *)(Tc + r*C),      vzpT);
        HVX_Vector q1 = Q6_Vw_vadd_VwVw(*(HVX_Vector *)(Tc + r*C + 32), vzpT);
        *(HVX_UVector *)(Tu + r*C) = Q6_Vuh_vpack_VwVw_sat(q1, q0);
    }
}
#endif  /* __hexagon__ */

static void gdn_solve_head_scalar(const uint16_t *Au, int C, int zpA, float sA,
                                  float sT, int zpT, uint16_t *Tu, int slot) {
    static int16_t g_As[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];
    static int16_t g_Ts[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];
    int16_t *As = g_As[slot], *Ts = g_Ts[slot];
    for (int i = 0; i < C*C; ++i) As[i] = (int16_t)((int32_t)Au[i] - zpA);
    gdn_solve_head_q<int16_t>(As, C, sA, sT, 32767.0f, Ts);
    for (int i = 0; i < C*C; ++i) Tu[i] = (uint16_t)((int32_t)Ts[i] + zpT);
}

static uint32_t gdn_solve_kernel(
        QHPI_RuntimeHandle *handle,
        uint32_t num_outputs, QHPI_Tensor **outputs,
        uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
    (void)num_outputs; (void)num_inputs;
    if (!outputs || !outputs[0] || !inputs || !inputs[0]) return QHPI_Success;

    /* A and T are int16 acts carried as uint16-midpoint on HTP (zero_offset ~= 32768). */
    const uint16_t *Au = (const uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    uint16_t *Tu = (uint16_t *)qhpi_tensor_raw_data(outputs[0]);
    if (!Au || !Tu) return QHPI_Success;

    const QHPI_Quant_Parameters qa = qhpi_tensor_quant_parameters(inputs[0]);
    const QHPI_Quant_Parameters qt = qhpi_tensor_quant_parameters(outputs[0]);
    const float sA = qa.stepsize, sT = qt.stepsize;
    const int32_t zpA = qa.zero_offset, zpT = qt.zero_offset;

    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);   /* [B,H,C,C] */
    int C = (s.rank >= 1) ? (int)s.dims[s.rank - 1] : GDN_CMAX;
    if (C <= 0 || C > GDN_CMAX || (C % GDN_BL)) return QHPI_Success;
    uint32_t heads = 1;
    for (uint32_t d = 0; d + 2 < s.rank; ++d) heads *= s.dims[d];

    /* self-slice across HVX threads: this thread handles heads [h0, h1) */
    uint32_t ns = qhpi_num_slices(handle), sl = qhpi_slice_number(handle);
    if (ns == 0) ns = 1;
    uint32_t h0 = (uint64_t)heads * sl / ns, h1 = (uint64_t)heads * (sl + 1) / ns;
#if defined(__hexagon__)
    int slot = gdn_claim_slot();                       /* unique scratch per concurrent thread */
#else
    int slot = (int)(sl < GDN_MAX_SLICES ? sl : 0);
#endif
    for (uint32_t h = h0; h < h1; ++h) {
        const uint16_t *Ah = Au + (size_t)h*C*C;
        uint16_t *Th = Tu + (size_t)h*C*C;
#if defined(__hexagon__)
        gdn_solve_head_hvx(Ah, C, zpA, sA, sT, zpT, Th, slot);
#else
        gdn_solve_head_scalar(Ah, C, zpA, sA, sT, zpT, Th, slot);
#endif
    }
#if defined(__hexagon__)
    gdn_free_slot(slot);
#endif
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static float gdn_solve_cost(uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
    if (!inputs || num_inputs < 1 || !inputs[0]) return 1.0f;
    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);
    float n = 1.0f; for (uint32_t d = 0; d < s.rank; ++d) n *= (float)s.dims[d];
    return n;                                       /* ~ elements; refine after HVX */
}

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::gdn_solve_kernel",
        gdn_solve_kernel,
        QHPI_RESOURCE_HVX,                        /* HVX-vectorized */
        false, true,  false, false,                /* multithreaded=true (engages when tiler can split, e.g. isolated) *
                                                   * ~11M; QNN won't self-slice a custom op whose output feeds the
                                                   * non-tiled Transpose/matmul). Real multi-HVX needs manual qurt. */
        1, sig_inputs,
        1, sig_outputs,
        gdn_solve_cost,
        0,
        0, nullptr, nullptr,
        nullptr,
    },
};

/* Tiling: opt into the central tiler so GdnSolve is split over the head dim and its tile-ops run on
 * multiple HVX threads.  C,C must NOT tile (each tile is a full 64x64 solve over a head-subset). */
static QHPI_Shape gdn_shape_required(const QHPI_Op *op) {
    (void)op;
    QHPI_Shape req; req.rank = 4;
    req.dims[0] = 1;                 /* B */
    req.dims[1] = 8;                 /* H: force tiles of 8 heads -> 4 parallel tile-ops */
    req.dims[2] = QHPI_DO_NOT_TILE;  /* C rows: keep whole */
    req.dims[3] = QHPI_DO_NOT_TILE;  /* C cols: keep whole */
    return req;
}

static const QHPI_Op *gdn_build_tile(const QHPI_Op *op, const QHPI_Shape *out_start,
                                     const QHPI_Shape *out_extent) {
    /* input A and output T share shape; the tile's input slice == output slice */
    QHPI_OpRef in = qhpi_op_input(op, 0);
    QHPI_Shape s = *out_start, e = *out_extent;
    QHPI_OpRef in_slice = qhpi_op_slice(in, &s, &e);
    QHPI_OpRef inputs[] = { in_slice };
    QHPI_OutputDef o0 = qhpi_op_output(op, 0);
    QHPI_OutputDef outputs[] = { { o0.type, o0.quant_parameters, *out_extent } };
    return qhpi_op_create(op, qhpi_op_name(op), 1, inputs, 1, outputs);
}

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::GdnSolve",
        1, sg_kernels,
        nullptr,                 /* early_rewrite */
        gdn_shape_required,      /* shape_required */
        nullptr,                 /* shape_legalized */
        0,                       /* tile_output index */
        gdn_build_tile,          /* build_tile */
        nullptr,                 /* late_rewrite */
    },
};

extern "C" void register_gdn_solve_op(void) {
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
