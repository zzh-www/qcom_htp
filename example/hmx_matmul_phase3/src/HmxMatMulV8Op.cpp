/*
 * HmxMatMulV8Op.cpp — Phase 3D.4: pure-HMX replica of QNN's q::ConvLayer_s1.opt.
 *
 * One-shot u8×i8→u8 matmul using HMX's :after:cm:sat.ub store.
 * Hardware formula (silicon-verified by probe_hmx_formula.c, 2026-04-24):
 *   out[m][c] = saturate_u8( (bias_raw[2c+1] >> 7)
 *                          + floor(acc[m][c] × bias_fp16[2c+1] / 512) )
 *   with acc = Σ_k act_u8[m][k] × wt_i8[k][c]    (activation is PLAIN u8)
 *   and col c uses bias lane (2c+1) — odd-indexed fp16 entries only.
 * Both zero_point and scale are encoded into the per-column fp16 bias:
 *   baseline (zp) = top 9 bits of raw u16 (= 8×exp_biased for zero mantissa)
 *   slope (scale) = fp16 value / 512
 * For u8 output with zp=128 + per-channel scale, pick bias such that
 * exp_biased=16 (bias ∈ [2.0, 4.0)); mantissa encodes channel scale variation.
 *
 * Signatures:
 *   Input 0: packed_act  [1, M_tiles, K_tiles, 1024] u8   TCM_Only
 *            (row-major 32×32 tile for :cm activation)
 *   Input 1: packed_wt   [1, N_tiles, K_tiles, 1024] u8   TCM_Only
 *            (Phase 2 P2 4-K-row × 32-col pack)
 *   Input 2: bias_scale  [1, 1, N_tiles, 32] u16/fp16     TCM_Only
 *            (per-channel scale folded = 512 × scale_out[n])
 *   Output 0: out        [1, 1, M, N] u8                  DDR_OR_TCM
 *            (u8 directly, zero_offset=128)
 *
 * Kernel body = 4 HMX inline-asm instructions per output tile, no HVX.
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

#if defined(__hexagon__)
/* Rt masks. The act_rt value 0x71F was reverse-engineered by dlsym-probing
 * QNN's set_hmx_params_conv1x1 — the previous V8 value 0x7FF set bits 5-7
 * (filter-stride encoding) which apply to NxN convs but NOT to 1×1 matmul.
 * Stripping those bits is bit-exact at 256³/512³/1024³ and 2× faster mmv8
 * cycles (Agent/qnn_re/256_isolation_findings.md §"act_rt fix"). */
#define HMX_RT_ACT_CM  0x71F           /* native act_rt_base for 1×1 matmul */
#define HMX_RT_WT      0x3FF           /* plain weight Rt (do NOT use native 0x700 — breaks output) */

/* Descriptor structs for QNN's hmx_convbbb1x1_stride1 dlsym call.
 * Per Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md §6. */
typedef struct {
    int32_t *out_tile_ptr_table;
    uint32_t out_table_stride_dwords;
    uint32_t out_y_stride_words;
    uint32_t n_tiles_pow2;
    int32_t  m_total_minus_step;
    uint32_t k_total_bytes;
} hmx_conv_out_desc_t;

typedef struct {
    int32_t *act_ptr_pairs;
    uint32_t n_act_pairs;
    uint32_t act_table_y_stride_words;
} hmx_conv_act_desc_t;

typedef struct {
    int32_t  out_check;
    uint32_t out_rt_mask;
    int32_t  act_check;
    uint32_t act_rt_base;
    uint32_t filter_x_stride;
    uint32_t _pad14;
    uint32_t alt_rt;
} hmx_conv_mask_desc_t;

#if defined(V8_USE_DLSYM_PER_TILE)
extern "C" void hmx_convbbb1x1_stride1(
    const hmx_conv_out_desc_t  *out_desc,
    const hmx_conv_act_desc_t  *act_desc,
    const void                 *weight_base,
    const void                 *bias_base,
    const hmx_conv_mask_desc_t *mask_desc);
#endif

static inline __attribute__((always_inline))
void hmx_v8_mac_convert(
    const void *act_tile,              /* 1 KiB row-major */
    const void *wt_tile,               /* 1 KiB P2 packed */
    const void *bias_scale,            /* 128 fp16 values */
    void       *out_tile)              /* 1 KiB u8 destination */
{
    asm volatile("bias = mxmem(%0)" :: "r"(bias_scale) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "  weight.b      = mxmem(%2, %3) }"
        :: "r"(act_tile), "r"(HMX_RT_ACT_CM),
           "r"(wt_tile),  "r"(HMX_RT_WT)
        : "memory");
    asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                 :: "r"(out_tile), "r"(0) : "memory");
}

static inline __attribute__((always_inline))
void hmx_v8_mac_accumulate(
    const void *act_tile,
    const void *wt_tile)
{
    asm volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "  weight.b      = mxmem(%2, %3) }"
        :: "r"(act_tile), "r"(HMX_RT_ACT_CM),
           "r"(wt_tile),  "r"(HMX_RT_WT)
        : "memory");
}
#endif

static inline uint32_t dim_at_v8(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

static uint32_t hmx_matmul_v8_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    uint8_t *out = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    uint32_t total = 1;
    for (uint32_t i = 0; i < os.rank; i++) total *= os.dims[i];
    memset(out, 128, total);
    return QHPI_Success;
#else
    const uint8_t  *packed_act = (const uint8_t  *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t  *packed_wt  = (const uint8_t  *)qhpi_tensor_raw_data(inputs[1]);
    const uint16_t *bias_all   = (const uint16_t *)qhpi_tensor_raw_data(inputs[2]);
    uint8_t        *vtcm_stg   = (uint8_t *)qhpi_tensor_raw_data(inputs[3]);
    uint8_t        *out        = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
#if defined(V8_ACT_KMAJOR)
    /* PackActCrouton (Crouton K-major) layout: [1, K_t, M_t, 1024]. */
    const uint32_t K_tiles = dim_at_v8(as, 1);
    const uint32_t M_tiles = dim_at_v8(as, 2);
#else
    /* PackActivationU8RowMajor layout: [1, M_t, K_t, 1024]. */
    const uint32_t M_tiles = dim_at_v8(as, 1);
    const uint32_t K_tiles = dim_at_v8(as, 2);
#endif
    const uint32_t N_tiles = dim_at_v8(ws, 1);
    const uint32_t N       = dim_at_v8(os, os.rank - 1);

    /* Tile-layout output: HMX sat.ub writes 1 KiB/tile directly. */
    const uint32_t out_last_dim = dim_at_v8(os, os.rank - 1);
    const bool tile_output = (out_last_dim == 1024);
    if (tile_output) {
        (void)vtcm_stg;
        (void)N;
        /* QNN-style loop order: N-TILE OUTER, M-TILE INNER.
         * Mimics hmx_convbbb1x1_stride1 @ 0x2ea740 in libQnnHtpV75Skel.so.
         * Benefits:
         *   - Load bias ONCE per nt, reuse across M_tiles iterations
         *     (previously we reloaded bias each (mt, nt) tile — K_tiles
         *      × M_tiles × N_tiles tiles re-loaded = M_tiles×N_tiles×~50 cyc
         *      wasted).
         *   - Weight tile for fixed nt is consumed repeatedly — its VTCM
         *     lines stay cache-hot.
         *   - Activation tiles stream through as fast as VTCM can deliver.
         * K inner loop is 2-MAC unroll (matches QNN's hot-loop packet
         * structure). */
        /* Prologue mxclracc — per QNN disasm, acc is cleared ONCE at op
         * entry; subsequent per-tile sat.ub implicitly clears acc for the
         * next MAC sequence ("after" on non-retain variant clears both
         * accs per probe_dualacc_device RE). */
        /* Pre-bake activation pointer table ONCE per kernel call.
         * Layout: act_ptrs_all[mt * K_tiles + kt] = packed_act + (mt*K_t + kt)*1024.
         * This avoids re-running the K-loop pointer-fill per (mt, nt) and may
         * help VTCM port scheduling by removing scalar table-build packets
         * interleaved with HMX MACs. */
        int32_t act_ptrs_all[128 * 8] __attribute__((aligned(16)));  /* up to M_t=8, K_t=128 (4096³) */
        for (uint32_t mt = 0; mt < M_tiles; mt++) {
            for (uint32_t k = 0; k < K_tiles; k++) {
#if defined(V8_ACT_KMAJOR)
                /* Crouton K-major: tile (mt, kt) at packed_act + (k * M_tiles + mt) * 1024 */
                act_ptrs_all[mt * K_tiles + k] =
                    (int32_t)(uintptr_t)(packed_act + (k * M_tiles + mt) * 1024);
#else
                act_ptrs_all[mt * K_tiles + k] =
                    (int32_t)(uintptr_t)(packed_act + (mt * K_tiles + k) * 1024);
#endif
            }
        }

        asm volatile("mxclracc" ::: "memory");
        for (uint32_t nt = 0; nt < N_tiles; nt++) {
            const uint8_t  *wt_tiles = packed_wt + (nt * K_tiles) * 1024;
            const uint16_t *bias_n   = bias_all  + nt * 128;
            /* Bias loaded ONCE per nt band — use mxmem2 to match QNN's
             * hmx_convbbb1x1_stride1 exactly (bias = mxmem2(r3)). */
            asm volatile("bias = mxmem2(%0)" :: "r"(bias_n) : "memory");
            for (uint32_t mt = 0; mt < M_tiles; mt++) {
                const int32_t *act_ptrs = &act_ptrs_all[mt * K_tiles];
                uint8_t       *out_tile  = out + (mt * N_tiles + nt) * 1024;

                /* Per-tile inner K loop replicates QNN's
                 * hmx_convbbb1x1_stride1 loop0 body byte-for-byte
                 * (Agent/qnn_hmx_pipelining.md disasm). The 3-packet
                 * structure puts ptr post-inc + alt_rt swap IN-PACKET with
                 * MAC issue, eliminating scalar bubble cycles between MACs.
                 *
                 * Body (per loop0 iter, consumes 2 K-tiles):
                 *   { p0 = cmp.eq(r26, #2); r26 -= 2
                 *     r6  = memw(r1++#8)         ; act ptr 0, post-inc 1 pair
                 *     r23 = memw(r1+#4) }        ; act ptr 1
                 *   { r8 += 0x400                ; weight ptr += 1024
                 *     if (p0) r25:24 = combine(r9,r7)  ; alt_rt swap on last K
                 *     activation.ub = mxmem(r6, r24):cm
                 *     weight.b      = mxmem(r8, r25) }
                 *   { r8 += add(r25,#1)          ; r8 += 0x400
                 *     activation.ub = mxmem(r23, r24):cm
                 *     weight.b      = mxmem(r8, r25) }
                 *
                 * We pre-bake the act-ptr-pair table on the stack per (nt,
                 * mt) tile drain and use raw asm with post-inc walking. */
#if defined(V8_USE_DLSYM_PER_TILE)
                {
                    /* Single-tile call to hmx_convbbb1x1_stride1.
                     * Geometry: 1 N-tile × 1 M-tile × K_tiles K-strips → 1 output tile.
                     * Descriptor values per RE in
                     * Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md and updated
                     * understanding from disasm at 0x2ea740 + outer loop is
                     * actually N-loop (bias advance per iter). */
                    static int32_t out_tbl[1] __attribute__((aligned(64)));
                    out_tbl[0] = (int32_t)(uintptr_t)out_tile;
                    static int32_t act_tbl[64] __attribute__((aligned(64)));
                    for (uint32_t k = 0; k < K_tiles; k++) {
                        act_tbl[k] = act_ptrs[k];
                    }
                    /* For act_rt_base = 2047 (0x7FF):
                     *   r28 = 0x7FF & 0x7E0 = 0x7E0; ct0=5; r18 = 0
                     *   r21 = 1<<0 = 1
                     *   r19 = 6-0 = 6; r22 = 1<<6 = 64
                     * Loop1 trip = (n_tiles_pow2 + 0) >> 0 = n_tiles_pow2
                     * Loop1 exits when r17 (init r12) <= 0 after -= r22 each iter
                     * For 1 loop1 iter: m_total_minus_step = 64 (loop1 runs once: r17 = 64-64 = 0, exit)
                     */
                    hmx_conv_out_desc_t  od = {
                        out_tbl,            /* out_tile_ptr_table */
                        1,                  /* out_table_stride_dwords */
                        0,                  /* out_y_stride_words */
#if defined(V8_DLSYM_NTILES_KT)
                        K_tiles,            /* n_tiles_pow2 = K_t */
#else
                        1,                  /* n_tiles_pow2 (1 loop1 iter) */
#endif
#if defined(V8_DLSYM_MTOTAL_KT32)
                        (int32_t)(K_tiles * 32), /* m_total = K (in case it controls K-loop) */
#elif defined(V8_DLSYM_MTOTAL_1)
                        1,                  /* m_total = 1 */
#else
                        64,                 /* m_total_minus_step (= r22 for 1 loop1 iter) */
#endif
#if defined(V8_DLSYM_KBYTES_BIG)
                        K_tiles * 32,       /* k_total_bytes = K */
#else
                        32,                 /* k_total_bytes */
#endif
                    };
                    hmx_conv_act_desc_t  ad = {
                        act_tbl,            /* act_ptr_pairs */
#if defined(V8_DLSYM_NPAIRS_K2)
                        K_tiles / 2,        /* n_act_pairs = K_t/2 (pair count) */
#elif defined(V8_DLSYM_NPAIRS_2K)
                        K_tiles * 2,        /* n_act_pairs = 2*K_t */
#else
                        K_tiles,            /* n_act_pairs = K_t (default) */
#endif
                        0,                  /* act_table_y_stride_words */
                    };
                    hmx_conv_mask_desc_t md = {
                        0,                  /* out_check = 0 (always satisfies bitsclr probe) */
                        HMX_RT_WT,          /* out_rt_mask = 0x3FF */
                        0,                  /* act_check = 0 (probe-pass) */
                        2047,               /* act_rt_base = 0x7FF */
                        0,                  /* filter_x_stride (unused 1x1) */
                        0,                  /* _pad */
                        HMX_RT_WT,          /* alt_rt = 0x3FF */
                    };
                    hmx_convbbb1x1_stride1(&od, &ad, wt_tiles, bias_n, &md);
                }
                continue;
#endif
                {
                    /* act_ptrs already pre-baked at op start. */

                    /* Hexagon hardware loop0 — same as QNN ConvLayer.
                     * Zero overhead per iter (vs software loop's 3 extra
                     * packets per iter = ~3 cyc/MAC scalar bookkeeping).
                     *
                     * loop0 trip count = K_tiles / 2. For K_tiles even the
                     * pair body runs trip times. For odd K_tiles, tail MAC
                     * runs after. */
                    const uint32_t loop0_trip = K_tiles / 2;
                    const uint32_t k_odd = K_tiles & 1;
                    register int32_t r1  asm("r1")  = (int32_t)(uintptr_t)act_ptrs;
                    register int32_t r8  asm("r8")  = (int32_t)(uintptr_t)wt_tiles - 0x400;
                    register int32_t r9  asm("r9")  = (int32_t)HMX_RT_WT;
                    register int32_t r7  asm("r7")  = (int32_t)HMX_RT_ACT_CM;
                    register int32_t r24 asm("r24") = (int32_t)HMX_RT_ACT_CM;
                    register int32_t r25 asm("r25") = (int32_t)HMX_RT_WT;
                    register int32_t r26 asm("r26") = (int32_t)K_tiles;

#if !defined(V8_PROBE_NO_MAC)
#if defined(V8_PROBE_SAME_ADDR) || defined(V8_PROBE_ACT_FIXED) || defined(V8_PROBE_WT_FIXED)
                    /* Probe variants — isolates VTCM stride contribution. */
                    if (loop0_trip > 0) {
#if defined(V8_PROBE_SAME_ADDR)
                        register int32_t r6_a  asm("r6")  = act_ptrs[0];
                        register int32_t r23_a asm("r23") = act_ptrs[0];
                        register int32_t r8_w  asm("r8")  = (int32_t)(uintptr_t)wt_tiles;
                        asm volatile(
                            "  loop0(1f, %5)\n1:\n"
                            "{ activation.ub = mxmem(%0,%3):cm\n  weight.b = mxmem(%2,%4) }\n"
                            "{ activation.ub = mxmem(%1,%3):cm\n  weight.b = mxmem(%2,%4) }:endloop0"
                            :: "r"(r6_a), "r"(r23_a), "r"(r8_w),
                               "r"((int32_t)HMX_RT_ACT_CM), "r"((int32_t)HMX_RT_WT),
                               "r"(loop0_trip)
                            : "lc0", "sa0", "memory");
#elif defined(V8_PROBE_ACT_FIXED)
                        /* Act fixed (cache-warm), wt sweeps K_tiles via post-inc r8. */
                        register int32_t r6_a  asm("r6")  = act_ptrs[0];
                        register int32_t r23_a asm("r23") = act_ptrs[0];
                        register int32_t r8_w  asm("r8")  = (int32_t)(uintptr_t)wt_tiles - 0x400;
                        asm volatile(
                            "  loop0(1f, %4)\n1:\n"
                            "{ r8 = add(r8, #0x400)\n"
                            "  activation.ub = mxmem(r6,%2):cm\n  weight.b = mxmem(r8,%3) }\n"
                            "{ r8 = add(r8, #0x400)\n"
                            "  activation.ub = mxmem(r23,%2):cm\n  weight.b = mxmem(r8,%3) }:endloop0"
                            : "+r"(r8_w), "+r"(r6_a)
                            : "r"((int32_t)HMX_RT_ACT_CM), "r"((int32_t)HMX_RT_WT),
                              "r"(loop0_trip), "r"(r23_a)
                            : "lc0", "sa0", "memory");
#elif defined(V8_PROBE_WT_FIXED)
                        /* Wt fixed (cache-warm), act strides via C-side loop. */
                        const int32_t wt_fixed = (int32_t)(uintptr_t)wt_tiles;
                        for (uint32_t kk = 0; kk + 1 < K_tiles; kk += 2) {
                            asm volatile(
                                "{ activation.ub = mxmem(%0,%2):cm\n  weight.b = mxmem(%4,%3) }\n"
                                "{ activation.ub = mxmem(%1,%2):cm\n  weight.b = mxmem(%4,%3) }"
                                :
                                : "r"(act_ptrs[kk]), "r"(act_ptrs[kk+1]),
                                  "r"((int32_t)HMX_RT_ACT_CM), "r"((int32_t)HMX_RT_WT),
                                  "r"(wt_fixed)
                                : "memory");
                        }
#elif defined(V8_PROBE_ACT_4WAY)
                        /* Cycle through ONLY 4 unique act addresses (act_ptrs[0..3])
                         * regardless of K_tiles. Tests whether penalty is per
                         * new address or per address-mismatch event. */
                        const int32_t wt_fixed = (int32_t)(uintptr_t)wt_tiles;
                        for (uint32_t kk = 0; kk + 1 < K_tiles; kk += 2) {
                            asm volatile(
                                "{ activation.ub = mxmem(%0,%2):cm\n  weight.b = mxmem(%4,%3) }\n"
                                "{ activation.ub = mxmem(%1,%2):cm\n  weight.b = mxmem(%4,%3) }"
                                :
                                : "r"(act_ptrs[kk & 3]), "r"(act_ptrs[(kk+1) & 3]),
                                  "r"((int32_t)HMX_RT_ACT_CM), "r"((int32_t)HMX_RT_WT),
                                  "r"(wt_fixed)
                                : "memory");
                        }
#elif defined(V8_PROBE_ACT_PAIR_SAME)
                        /* Same act addr WITHIN pair (r6=r23), but stride between pairs. */
                        const int32_t wt_fixed = (int32_t)(uintptr_t)wt_tiles;
                        for (uint32_t kk = 0; kk + 1 < K_tiles; kk += 2) {
                            asm volatile(
                                "{ activation.ub = mxmem(%0,%2):cm\n  weight.b = mxmem(%4,%3) }\n"
                                "{ activation.ub = mxmem(%0,%2):cm\n  weight.b = mxmem(%4,%3) }"
                                :
                                : "r"(act_ptrs[kk]), "r"(act_ptrs[kk+1]),
                                  "r"((int32_t)HMX_RT_ACT_CM), "r"((int32_t)HMX_RT_WT),
                                  "r"(wt_fixed)
                                : "memory");
                        }
#elif defined(V8_PROBE_RT_ACT_3FC)
                        /* Use Rt_act = 0x3FC (analog to Rt_wt = 0x3FF unlock). */
                        const int32_t wt_fixed = (int32_t)(uintptr_t)wt_tiles;
                        const int32_t rt_act = 0x3FF | 0x1C;
                        for (uint32_t kk = 0; kk + 1 < K_tiles; kk += 2) {
                            asm volatile(
                                "{ activation.ub = mxmem(%0,%2):cm\n  weight.b = mxmem(%4,%3) }\n"
                                "{ activation.ub = mxmem(%1,%2):cm\n  weight.b = mxmem(%4,%3) }"
                                :
                                : "r"(act_ptrs[kk]), "r"(act_ptrs[kk+1]),
                                  "r"(rt_act), "r"((int32_t)HMX_RT_WT),
                                  "r"(wt_fixed)
                                : "memory");
                        }
#elif defined(V8_PROBE_ACT_4K_STRIDE)
                        /* Read act tiles at 4096-byte stride (every 4th K-tile).
                         * Probes whether 1024-byte stride causes VTCM bank
                         * conflict — if yes, 4K stride hits different banks
                         * and runs faster. Result is wrong but cycle count tells. */
                        const int32_t wt_fixed = (int32_t)(uintptr_t)wt_tiles;
                        for (uint32_t kk = 0; kk + 1 < K_tiles; kk += 2) {
                            const int32_t a0 = act_ptrs[(kk * 4) % K_tiles];
                            const int32_t a1 = act_ptrs[((kk + 1) * 4) % K_tiles];
                            asm volatile(
                                "{ activation.ub = mxmem(%0,%2):cm\n  weight.b = mxmem(%4,%3) }\n"
                                "{ activation.ub = mxmem(%1,%2):cm\n  weight.b = mxmem(%4,%3) }"
                                :
                                : "r"(a0), "r"(a1),
                                  "r"((int32_t)HMX_RT_ACT_CM), "r"((int32_t)HMX_RT_WT),
                                  "r"(wt_fixed)
                                : "memory");
                        }
#endif
                    }
                    (void)r1; (void)r8; (void)r24; (void)r25; (void)r26;
                    (void)r7; (void)r9;
#else
                    if (loop0_trip > 0) {
                        asm volatile(
                            "  loop0(1f, %5)\n"
                            "1:\n"
                            "{ p0 = cmp.eq(r26, #2)\n"
                            "  r26 = add(r26, #-2)\n"
                            "  r6  = memw(r1++#8)\n"
                            "  r23 = memw(r1+#4) }\n"
                            "{ r8 = add(r8, #0x400)\n"
                            "  if (p0) r25:24 = combine(r9, r7)\n"
                            "  activation.ub = mxmem(r6, r24):cm\n"
                            "  weight.b      = mxmem(r8, r25) }\n"
                            "{ r8 = add(r8, add(r25, #1))\n"
                            "  activation.ub = mxmem(r23, r24):cm\n"
                            "  weight.b      = mxmem(r8, r25) }:endloop0"
                            : "+r"(r1), "+r"(r8), "+r"(r24), "+r"(r25), "+r"(r26)
                            : "r"(loop0_trip), "r"(r7), "r"(r9)
                            : "p0", "r6", "r23", "lc0", "sa0", "memory");
                    }
#endif
                    if (k_odd) {
                        const uint8_t *a_tail = (const uint8_t*)(uintptr_t)act_ptrs[K_tiles - 1];
                        asm volatile(
                            "{ r8 = add(r8, #0x400)\n"
                            "  activation.ub = mxmem(%0, %1):cm\n"
                            "  weight.b      = mxmem(r8, %2) }"
                            : "+r"(r8)
                            : "r"(a_tail), "r"((int32_t)HMX_RT_ACT_CM),
                              "r"((int32_t)HMX_RT_WT)
                            : "memory");
                    }
#endif
#if !defined(V8_PROBE_NO_SATUB)
                    asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                                 :: "r"(out_tile), "r"((int32_t)HMX_RT_WT) : "memory");
#endif
                }
            }
        }
    } else {
        /* Legacy row-major path: VTCM staging + scalar scatter. */
        for (uint32_t mt = 0; mt < M_tiles; mt++) {
            for (uint32_t nt = 0; nt < N_tiles; nt++) {
                const uint8_t  *act_tiles = packed_act + (mt * K_tiles) * 1024;
                const uint8_t  *wt_tiles  = packed_wt  + (nt * K_tiles) * 1024;
                const uint16_t *bias_n    = bias_all   + nt * 128;
                uint8_t        *out_tile  = out + (mt * 32) * N + nt * 32;

                asm volatile("bias = mxmem(%0)" :: "r"(bias_n) : "memory");
                asm volatile("mxclracc" ::: "memory");
                for (uint32_t kt = 0; kt < K_tiles; kt++) {
                    hmx_v8_mac_accumulate(
                        act_tiles + kt * 1024,
                        wt_tiles  + kt * 1024);
                }
                asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                             :: "r"(vtcm_stg), "r"(0) : "memory");
                for (uint32_t r = 0; r < 32; r++) {
                    memcpy(&out_tile[r * N], &vtcm_stg[r * 32], 32);
                }
            }
        }
    }
    return QHPI_Success;
#endif
}

static QHPI_Tensor_Signature_v1 sig_inputs_v8[] = {
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* packed_act (row-major) */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* packed_wt (P2) */
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* bias fp16 */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* vtcm staging (1 KiB aligned) */
};
static QHPI_Tensor_Signature_v1 sig_outputs_v8[] = {
    /* TCM-only output.  In tile-layout mode, sat.ub writes 1 KiB per
     * tile directly to this VTCM buffer — matches QNN ConvLayer_s1.opt.
     * A follow-up TcmDramCopy op bulk-memcpys to DDR if the graph end
     * requires DDR. */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels_v8[] = {
    {
        THIS_PKG_NAME_STR "::hmx_matmul_v8",
        hmx_matmul_v8_kernel,
        QHPI_RESOURCE_HMX,
#if defined(V8_MMV8_MULTITHREADED)
        false, true, false, false,      /* multithreaded=true probe */
#else
        false, false, false, false,
#endif
        4, sig_inputs_v8,
        1, sig_outputs_v8,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v8[] = {
    {
        THIS_PKG_NAME_STR "::MatMulV8",
        1, sg_kernels_v8,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_hmx_matmul_v8_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_v8) / sizeof(sg_ops_v8[0]),
                         sg_ops_v8, THIS_PKG_NAME_STR);
}
