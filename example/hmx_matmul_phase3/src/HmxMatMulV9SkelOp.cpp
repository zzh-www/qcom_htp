/*
 * HmxMatMulV9SkelOp.cpp — V10/Route1 matmul.
 *
 * Same HMX inner loop as MatMulV8 (matches hmx_convbbb1x1_stride1 silicon
 * ceiling on v75) but consumes K-major Crouton-byte layout produced by
 * PackActCrouton (which wraps libQnnHtpV75Skel.so::convert_to_crouton_b).
 *
 * Layout difference:
 *   V8 packed_act: [1, M_tiles, K_tiles, 1024]  tile(mt,kt) @ (mt*K_t + kt)*1024
 *   V9 packed_act: [1, K_tiles, M_grp,  128 ]  tile(mt,kt) @ (kt*M_t + mt)*1024
 *   V8 packed_wt:  [1, N_tiles, K_tiles, 1024]  tile(nt,kt) @ (nt*K_t + kt)*1024
 *   V9 packed_wt:  [1, K_tiles, N_grp,  128 ]  tile(nt,kt) @ (kt*N_t + nt)*1024
 *
 * Where M_grp = M/4 (8 h-iters per tile), N_grp = N/4 likewise.
 *
 * Signatures (matching V8 except input layouts):
 *   in[0]: u8 packed_act [1, K_t, M_grp, 128]   TCM_Only
 *   in[1]: u8 packed_wt  [1, K_t, N_grp, 128]   TCM_Only
 *   in[2]: u16 bias      [1, 1, N_t, 128]       TCM_Only
 *   in[3]: u8 vtcm_stg   [1, 1, 1, 2048]        TCM_Only  (legacy fallback)
 *   out[0]: u8 out_tile  [1, M_t, N_t, 1024]    TCM_Only  (tile-layout)
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#if defined(__hexagon__) && (defined(V9_PMU_PROBE) || defined(V9_PROBE_REGIONS))
/* Hexagon PMU access via QURT API. v75 PMU enum from
 * computev75/include/qurt/qurt_consts.h. We don't include that header
 * directly to avoid pulling all of QURT — just re-declare the symbols
 * and constants we need. */
extern "C" void         qurt_pmu_set(int reg_id, unsigned int reg_value);
extern "C" unsigned int qurt_pmu_get(int reg_id);
extern "C" void         qurt_pmu_enable(int enable);
#define QURT_PMUCNT0    0
#define QURT_PMUCNT1    1
#define QURT_PMUCNT2    2
#define QURT_PMUCNT3    3
#define QURT_PMUCFG     4
#define QURT_PMUEVTCFG  5
/* PMU event opcodes from itrace example comments / itrace_dsp_events_pmu.h
 * (low byte of itrace ID = raw PMU opcode). NB: NOT the same as the 0x80XX
 * itrace IDs — those are itrace-internal table indices. */
#define PMU_COMMITTED_PKT_ANY 0x03   /* total committed packets (all HW threads) */
#define PMU_COMMITTED_PKT_BSB 0x04
#define PMU_COMMITTED_PKT_B2B 0x07
#define PMU_COMMITTED_PKT_SMT 0x08
#define PMU_COMMITTED_PKT_T0  0x0a   /* thread 0 only (itrace 0x800a low byte) */
#define PMU_COMMITTED_PKT_T1  0x0b
#define PMU_COMMITTED_PKT_T2  0x0c
#define PMU_AXI_WRITE_REQUEST 0x42
#define PMU_AXI_LINE128_READ  0x3f
#define PMU_COMMITTED_FPS     0x50
#define PMU_COMMITTED_INSTS   0x25   /* committed instructions (not packets) */
#define PMU_DISPATCHED_PKTS   0x2f
#define PMU_HVX_ACTIVE        0xCC   /* HVX active — 9 bits, may not fit */
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

#if defined(__hexagon__)
/* See HmxMatMulV8Op.cpp note: 0x71F is the native value; 0x7FF was a buggy
 * V8 carry-over that set filter-stride bits 5-7 not applicable to 1×1 matmul. */
#define HMX_RT_ACT_CM  0x71F           /* native act_rt_base for 1×1 matmul */
#define HMX_RT_WT      0x3FF
#endif

/* libQnnHtpV75Skel.so::hmx_convbbb1x1_stride1 ABI per
 * Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md §6.
 * Cross-.so dlsym is empirically validated for this skel
 * (Agent/dlsym_spike_PASS_2026-04-25.md). */
typedef struct {
    int32_t *out_tile_ptr_table;     /* +0x00 */
    uint32_t out_table_stride_dwords;/* +0x04 */
    uint32_t out_y_stride_words;     /* +0x08, internally <<= 2 */
    uint32_t n_tiles_pow2;           /* +0x0c */
    int32_t  m_total_minus_step;     /* +0x10 */
    uint32_t k_total_bytes;          /* +0x14 */
} hmx_conv_out_desc_t;

typedef struct {
    int32_t *act_ptr_pairs;             /* +0x00 list of {ptr_lo, ptr_hi} */
    uint32_t n_act_pairs;               /* +0x04 */
    uint32_t act_table_y_stride_words;  /* +0x08, internally <<= 2 */
} hmx_conv_act_desc_t;

typedef struct {
    int32_t  out_check;       /* +0x00 — must satisfy bitsclr(_, 0x7e0) */
    uint32_t out_rt_mask;     /* +0x04 — Rt for sat.ub store */
    int32_t  act_check;       /* +0x08 */
    uint32_t act_rt_base;     /* +0x0c — base Rt for activation MAC */
    uint32_t filter_x_stride; /* +0x10 unused by 1x1 */
    uint32_t _pad14;          /* +0x14 */
    uint32_t alt_rt;          /* +0x18 — Rt for last-K MAC + odd-tail */
} hmx_conv_mask_desc_t;

#if defined(__hexagon__) && (defined(V9_USE_DLSYM) || defined(V9_USE_NATIVE_KERNEL) || defined(V9_PARAMS_PROBE))
extern "C" void hmx_convbbb1x1_stride1(
    const hmx_conv_out_desc_t  *out_desc,
    const hmx_conv_act_desc_t  *act_desc,
    const void                 *weight_base,
    const void                 *bias_base,
    const hmx_conv_mask_desc_t *mask_desc);

/* hmx_v73_convbbb1x1_stride1 (0x2eadc0, 936 B) — uses `:deep:cm` activation
 * mxmem modifier (non-v73 uses plain `:cm`). Disasm shows 6-arg signature:
 *   r0..r4 same as non-v73 + r5 = pointer to extra_param[2]:
 *     extra_param[0] (uint32) — marker; if == 1, take fast path with
 *                               cvt.ub = acc(r31) + mxmem.cm = cvt drain.
 *                               Otherwise → slow non-fast branch (still :deep:cm).
 *     extra_param[1] (uint32) — r31 used as cvt config arg. Likely 0 / unused
 *                               for u8 sat case (mxmem2 bias preloads acc).
 * If extra_param[0] also has bit 5 set in mask_desc[+0x30], v73 dispatcher
 * jumps to hmx_v73_convbbb1x1deep_stride1 (the real "deep K-fanout" body). */
extern "C" void hmx_v73_convbbb1x1_stride1(
    const hmx_conv_out_desc_t  *out_desc,
    const hmx_conv_act_desc_t  *act_desc,
    const void                 *weight_base,
    const void                 *bias_base,
    const hmx_conv_mask_desc_t *mask_desc,
    const uint32_t             *extra_param);

/* hmx_v73_convbbb1x1deep_stride1 (0x2ebe40, 1132 B) — REAL deep variant.
 *   `:deep:cm` activation + `:deep` weight mxmem modifiers
 *   2 cvt.ub drains per loop1 iter (M-fanout = 2 output tiles per pass)
 *   Two bias = mxmem2 loads per outer iter (r3+0x101 then r3+0xff)
 *   Wt stride r22 = (alt_rt+1)/2 * N_t = N_t * 512 bytes per K-step
 *   extra_param[] is a per-K-iter array (r5 advances inside the loop)
 * Reached via hmx_v73_convbbb1x1_stride1 if mask_desc[+0x30] bit 5 set,
 * but typically called directly via own dispatch. */
extern "C" void hmx_v73_convbbb1x1deep_stride1(
    const hmx_conv_out_desc_t  *out_desc,
    const hmx_conv_act_desc_t  *act_desc,
    const void                 *weight_base,
    const void                 *bias_base,
    const hmx_conv_mask_desc_t *mask_desc,
    const uint32_t             *extra_param);

/* hmx_v73_convbbb1x1deep_stride1_sparsity (0x2ebb00, 820 B). Disasm
 * (Step 3 candidate): activation MAC = `:cm` (NOT `:deep:cm`), weight
 * MAC = `:deep`. Bias single load per outer (r3 += 0x100, like non-deep).
 * Inner drain has small inner loop indexed by r23 (= r26 = extra_param[0]).
 * extra_param[] consumed periodically: `r26 = memw(r5++#4); r27 = memw(r5++#4)`
 * ABI same 6-arg as deep variant — try first with extra={1, 0, ...}. */
extern "C" void hmx_v73_convbbb1x1deep_stride1_sparsity(
    const hmx_conv_out_desc_t  *out_desc,
    const hmx_conv_act_desc_t  *act_desc,
    const void                 *weight_base,
    const void                 *bias_base,
    const hmx_conv_mask_desc_t *mask_desc,
    const uint32_t             *extra_param);

/* Untested kernel variants in libQnnHtpV75Skel.so. ABI assumed compatible
 * with v73 1x1 stride1 family (5 or 6 args). For NxN variants the descriptor
 * format may differ but with filter_x_stride=0 and stride1 they may handle
 * 1x1 as degenerate case. */
extern "C" void hmx_v73_convbbb1x1_stride1_unaligned(
    const hmx_conv_out_desc_t *, const hmx_conv_act_desc_t *,
    const void *, const void *, const hmx_conv_mask_desc_t *,
    const uint32_t *);
extern "C" void hmx_convbbb1x1_stride1_unaligned(
    const hmx_conv_out_desc_t *, const hmx_conv_act_desc_t *,
    const void *, const void *, const hmx_conv_mask_desc_t *);
extern "C" void hmx_v73_convbbb_stride1(
    const hmx_conv_out_desc_t *, const hmx_conv_act_desc_t *,
    const void *, const void *, const hmx_conv_mask_desc_t *,
    const uint32_t *);
extern "C" void hmx_v73_convbbb_stride1_aligned(
    const hmx_conv_out_desc_t *, const hmx_conv_act_desc_t *,
    const void *, const void *, const hmx_conv_mask_desc_t *,
    const uint32_t *);
#endif

#if defined(__hexagon__) && (defined(V9_DUMP_HMX_PARAMS) || defined(V9_PARAMS_PROBE) || defined(V9_USE_NATIVE_KERNEL))
/* libQnnHtpV75Skel.so :: set_hmx_params_conv1x1 — fills a 0x40-byte
 * hmx_params descriptor block at *out_params. GLOBAL DEFAULT export
 * (verified via llvm-readelf), reachable from our op-pkg .so via the
 * R_HEX_JMP_SLOT cross-.so call mechanism (Agent/dlsym_spike_PASS_2026-04-25.md). */
extern "C" void _Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm(
    void     *out_params,  /* r0 — 64-byte buffer */
    uint32_t  arg1,        /* r1 */
    uint32_t  arg2,        /* r2 */
    uint32_t  arg3,        /* r3 */
    uint32_t  arg4,        /* r4 */
    uint32_t  arg5);       /* r5 */
#define set_hmx_params_conv1x1 _Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm
#endif

static inline uint32_t dim_at_v9(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

static uint32_t hmx_matmul_v9_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    /* Host stub — ctxgen may invoke this; do not touch buffers (QNN host
     * may pass null/dummy ones). */
    return QHPI_Success;
#elif defined(V9_KERNEL_HMX)
    /* Phase 3.2 — HMX inline asm against Crouton_8 act + native bias fold.
     *
     * Bias VTCM bytes are byte-1:1 with native q::ConvLayer.opt.bias_to_vtcm
     * end-state. Per N-tile (256 B):
     *   bytes 0..127   : 32 × (fp16 scale, fp16 baseline)   — used by sat.ub
     *   bytes 128..255 : 32 × int32 effective_bias[c]      — initial acc fold
     * Both halves are loaded by `bias = mxmem2(...)`.
     *
     * Crouton_8 act → HMX :cm tile mapping:
     *   block_rows = min(M/4, 64)
     *   block_cols = 32  (always)
     *   Block index = rg * k_chunks + kc;  rg = m // block_rows, kc = k // 32
     *   In-block offset = (m % block_rows) * 32 + (k % 32)
     *
     * For each HMX tile (mt, kt) we need rows [mt*32 .. mt*32+31] × cols [kt*32 .. kt*32+31]:
     *   block_rows = 64 (S ≥ 256³): 1 block holds 2 HMX tiles → ptr = block + (mt%2)*1024
     *   block_rows = 32 (S = 128³): 1 block = 1 HMX tile        → ptr = block
     *   block_rows < 32 (S ∈ {32,64}): 2 or 4 blocks per HMX tile → repack to scratch
     *
     * Layouts:
     *   wt[1, N_t, K_t, 1024]    N-tile-outer, native pre-pack (host-side, Step 2)
     *   bias[2N] int32           native-fold layout (256 B per N-tile)
     *   out[1, M_t, N_t, 1024]   tile-layout, sat.ub direct write
     */
    {
        void **act_blocks = qhpi_tensor_block_table(inputs[0]);
        const uint8_t *wt_pack    = (const uint8_t *)qhpi_tensor_raw_data(inputs[1]);
        const uint8_t *bias_bytes = (const uint8_t *)qhpi_tensor_raw_data(inputs[2]);
        /* Output is now Crouton_8 + Indirect → use block_table for writes. */
        void **out_blocks = qhpi_tensor_block_table(outputs[0]);
        if (!act_blocks || !wt_pack || !bias_bytes || !out_blocks) return QHPI_Success;


        uint32_t blocks = qhpi_tensor_block_table_length(inputs[0]);
        uint32_t S = 0;
        if (blocks == 4)         S = 32;
        else if (blocks == 8)    S = 64;
        else if (blocks == 16)   S = 128;
        else if (blocks == 32)   S = 256;
        else if (blocks == 128)  S = 512;
        else if (blocks == 512)  S = 1024;
        if (S == 0) return QHPI_Success;

        const uint32_t M = S, K = S, N = S;
        const uint32_t M_t = M / 32, N_t = N / 32, K_t = K / 32;
        const uint32_t block_rows = (M / 4) < 64 ? (M / 4) : 64;
        const uint32_t k_chunks   = K / 32;

        /* No output zero needed: HMX :after:cm:sat.ub writes every byte
         * of every M_t × N_t × 1024 tile. Scalar fallback below also
         * writes every cell. Earlier defensive zero was costing ~40K cyc
         * at 256³ (compiler did NOT vectorize the byte-loop). */

        /* Scalar fallback for S ∈ {32, 64} where block_rows < 32 means a
         * single HMX :cm tile would span multiple non-contiguous Crouton
         * blocks. HMX :cm requires contiguous VTCM-resident bytes; stack
         * scratch faults (DDR addr). At these shapes total MAC volume is
         * tiny (≤2 K-tiles × ≤4 N-tiles × ≤2 M-tiles), so scalar is fine. */
        /* Output Crouton_8 byte layout (mirror of input):
         *   block_index_out = rg_o * out_n_chunks + kc_o
         *     rg_o = m / out_block_rows;  kc_o = n / 32
         *     out_block_rows = block_rows (same as input since M=N square)
         *   in-block offset: (m % out_block_rows) * 32 + (n % 32)
         */
        const uint32_t out_n_chunks = N / 32;

        if (block_rows < 32) {
            /* Scalar fallback at S∈{32,64}: write per-cell into output
             * block_table (Crouton_8). For S=32 each block is 256 B
             * (8 rows × 32 cols); S=64 each block is 512 B. */
            for (uint32_t nt = 0; nt < N_t; nt++) {
                const int32_t *eff32 =
                    (const int32_t *)(bias_bytes + nt * 256 + 128);
                for (uint32_t mt = 0; mt < M_t; mt++) {
                    for (uint32_t mr = 0; mr < 32; mr++) {
                        for (uint32_t nc = 0; nc < 32; nc++) {
                            uint32_t m = mt * 32 + mr;
                            uint32_t n = nt * 32 + nc;
                            int32_t acc = eff32[nc];
                            for (uint32_t k = 0; k < K; k++) {
                                uint32_t rg = m / block_rows;
                                uint32_t kc = k >> 5;
                                uint32_t bi = rg * k_chunks + kc;
                                uint32_t bo = (m % block_rows) * 32 + (k & 31);
                                uint8_t  a = ((const uint8_t *)act_blocks[bi])[bo];
                                uint32_t kt = k >> 5, kr = k & 31;
                                uint32_t nt2 = n >> 5, nc2 = n & 31;
                                uint32_t dst = (kr / 4) * 128 + nc2 * 4 + (kr % 4);
                                int8_t   w = (int8_t)wt_pack[(nt2 * K_t + kt) * 1024 + dst];
                                acc += (int32_t)a * (int32_t)w;
                            }
                            if (acc < 0) acc = 0; else if (acc > 255) acc = 255;
                            uint32_t rg_o = m / block_rows;
                            uint32_t kc_o = n / 32;
                            uint32_t bo_o = (m % block_rows) * 32 + (n % 32);
                            ((uint8_t *)out_blocks[rg_o * out_n_chunks + kc_o])[bo_o] =
                                (uint8_t)acc;
                        }
                    }
                }
            }
            return QHPI_Success;
        }

        /* Pre-bake act_ptrs[M_t][K_t] and wt_ptrs[N_t][K_t] into stack arrays.
         * Per-mt act tiles come from disjoint VTCM blocks (Crouton_8 block_table
         * lookups). For wt under the N-outer layout `[1, N_t, K_t, 1024]`
         * (Step 2 — matches native q::ConvLayer.opt.weights_to_vtcm@FB.fB.
         * end-state byte ordering) the K-tiles for fixed nt are 1024 B apart.
         *
         * Note on inline r8 += 0x400 post-inc (Step 2B attempt, abandoned):
         * native disasm shows a 4-packet body where `r8 = add(r8, 0x400)` and
         * `weight.b = mxmem(r8, r25)` are in DIFFERENT packets. We tried
         * fusing them into a 3-packet body (1 ALU + 2 mxmem in one packet);
         * the assembler accepts it but Hexagon V75 HMX hardware faults at
         * runtime — modifying the address register in the same packet as
         * the HMX load is not allowed. The 4-packet prebake body below
         * (2 cyc/MAC) matches native's 4-packet split-MAC body (also
         * 2 cyc/MAC) — no perf delta from a 3-packet form (which is
         * impossible on this silicon).
         *
         * Stack: 2 × M_t × K_t × 4 B, max 8 KB at S=1024. */
        int32_t act_ptrs_all[32 * 32] __attribute__((aligned(16)));  /* M_t × K_t */
        int32_t wt_ptrs_all[32 * 32]  __attribute__((aligned(16)));  /* N_t × K_t */

        const uint32_t mt_per_block = block_rows / 32;  /* ∈ {1, 2} when ≥ 128 */
        for (uint32_t mt = 0; mt < M_t; mt++) {
            const uint32_t rg = mt / mt_per_block;
            const uint32_t mt_in_block = mt % mt_per_block;
            for (uint32_t kt = 0; kt < K_t; kt++) {
                act_ptrs_all[mt * K_t + kt] = (int32_t)(uintptr_t)(
                    (const uint8_t *)act_blocks[rg * k_chunks + kt]
                    + mt_in_block * 1024);
            }
        }
        for (uint32_t nt = 0; nt < N_t; nt++) {
            for (uint32_t kt = 0; kt < K_t; kt++) {
                /* N-outer wt addressing: tile (nt, kt) at (nt*K_t + kt)*1024 */
                wt_ptrs_all[nt * K_t + kt] = (int32_t)(uintptr_t)(
                    wt_pack + (nt * K_t + kt) * 1024);
            }
        }

        asm volatile("mxclracc" ::: "memory");

        /* Loop order: mt outer, nt inner.
         *
         * At S ≥ 256³ the output Crouton_8 block_rows = 64, so each 2 KiB block
         * holds two HMX 1 KiB tiles (mt % 2 = 0 at offset 0, mt % 2 = 1 at
         * offset 1024). With the alternative nt-outer order each pair of
         * adjacent mt writes (mt=0,1 / 2,3 / 4,5 / 6,7) hits the same block
         * ~30 cyc apart (one MAC sweep), causing ~+3K cyc of VTCM cache-line
         * conflict at 256³ vs the (now-unused) pre-Crouton-8 tile-array
         * output. With mt outer the second write to the same block is delayed
         * by N_t whole nt-sweeps (~640 cyc at 256³), eliminating the conflict.
         *
         * Cost: M_t × N_t bias reloads instead of N_t — but each reload is
         * one mxmem2 instruction (~5 cyc), worst case ≈ 1024 × 5 ≈ 5K cyc at
         * S=1024, while the conflict pattern at S=1024 would cost ≈ 13K.
         * Net win at every shape; per-(mt,nt) reload also makes bias
         * persistence robust (HMX bias state has empirically expired after
         * ≥8 sat.ub events on some configs). */
        const uint32_t mt_per_blk_o = block_rows / 32;
        for (uint32_t mt = 0; mt < M_t; mt++) {
            const uint32_t bi_o_row = (mt / mt_per_blk_o) * out_n_chunks;
            const uint32_t off_o    = (mt % mt_per_blk_o) * 1024;
            const int32_t *act_ptrs = &act_ptrs_all[mt * K_t];
            for (uint32_t nt = 0; nt < N_t; nt++) {
                const uint8_t *bias_n  = bias_bytes + nt * 256;
                const int32_t *wt_ptrs = &wt_ptrs_all[nt * K_t];
                asm volatile("bias = mxmem2(%0)" :: "r"(bias_n) : "memory");
                uint8_t *out_tile = (uint8_t *)out_blocks[bi_o_row + nt] + off_o;

                /* 4-packet body / 2 MAC = 2 cyc/MAC. Matches native silicon
                 * ceiling — see Step 2 note above for why a 3-packet form
                 * is not possible on this hardware.
                 *   { r6  = memw(r1++#8); r8  = memw(r3++#8) }                ; load 2 act + 2 wt ptrs
                 *   { r23 = memw(r1+#-4); r9  = memw(r3+#-4) }                ; (in 2 packets, max 2 mem/pkt)
                 *   { activation.ub = mxmem(r6,r24):cm; weight.b = mxmem(r8,r25) }   ; MAC 1
                 *   { activation.ub = mxmem(r23,r24):cm; weight.b = mxmem(r9,r25) }:endloop0   ; MAC 2 + endloop */
                const uint32_t loop0_trip = K_t / 2;
                register int32_t r1  asm("r1")  = (int32_t)(uintptr_t)act_ptrs;
                register int32_t r3  asm("r3")  = (int32_t)(uintptr_t)wt_ptrs;
                register int32_t r24 asm("r24") = (int32_t)HMX_RT_ACT_CM;
                register int32_t r25 asm("r25") = (int32_t)HMX_RT_WT;
                asm volatile(
                    "  loop0(1f, %2)\n"
                    "1:\n"
                    "{ r6  = memw(r1++#8)\n"
                    "  r8  = memw(r3++#8) }\n"
                    "{ r23 = memw(r1+#-4)\n"
                    "  r9  = memw(r3+#-4) }\n"
                    "{ activation.ub = mxmem(r6, r24):cm\n"
                    "  weight.b      = mxmem(r8, r25) }\n"
                    "{ activation.ub = mxmem(r23, r24):cm\n"
                    "  weight.b      = mxmem(r9, r25) }:endloop0"
                    : "+r"(r1), "+r"(r3)
                    : "r"(loop0_trip), "r"(r24), "r"(r25)
                    : "r6", "r8", "r9", "r23", "lc0", "sa0", "memory");
                asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                             :: "r"(out_tile), "r"((int32_t)HMX_RT_WT) : "memory");
            }
        }
    }
    return QHPI_Success;
#elif defined(V9_KERNEL_SCALAR)
    /* Scalar reference matmul (Phase 2 / Phase 3 prep) — NATIVE bias layout.
     *
     * Bias VTCM bytes are pre-folded by gen_v8c8_test.py (host prepare time,
     * matching native q::ConvLayer.opt.bias_to_vtcm semantics). Per N-tile
     * (256 B):
     *   bytes 0..127   : 32 × (fp16 scale, fp16 baseline)
     *   bytes 128..255 : 32 × int32 effective_bias[c]
     * effective_int32[c] = -ACT_ZP × Σ_k W[k,c] + bias_q[c]
     *
     * Math (silicon-verified V8 prod formula, generalised to RAW act +
     * effective_int32 fold absorbed into accumulator init):
     *   acc[m,n] = effective_int32[n]                                  (init from bias upper)
     *            + Σ_k act_u8[m,k] × wRaw_i8[k,n]                      (HMX MAC equivalent)
     *   out[m,n] = saturate_u8( top9(baseline_n) +
     *                           floor(acc[m,n] × scale_fp16_n / 512) )
     *
     * Layouts (unchanged):
     *   act in[0] : Crouton_8 + Indirect, block_rows = min(M/4, 64)
     *   wt  in[1] : Direct, native pre-pack
     *   bias in[2]: Direct, int32 [2*N] = N_t × 64 int32 = N_t × 256 bytes (native fold) */
    {
        void **act_blocks = qhpi_tensor_block_table(inputs[0]);
        const uint8_t *wt_pack    = (const uint8_t *)qhpi_tensor_raw_data(inputs[1]);
        const uint8_t *bias_bytes = (const uint8_t *)qhpi_tensor_raw_data(inputs[2]);
        uint8_t       *out_buf    = (uint8_t       *)qhpi_tensor_raw_data(outputs[0]);
        if (!act_blocks || !wt_pack || !bias_bytes || !out_buf) return QHPI_Success;

        /* Derive S from block_table_length (square test only). */
        uint32_t blocks = qhpi_tensor_block_table_length(inputs[0]);
        uint32_t S = 0;
        if (blocks == 4)         S = 32;
        else if (blocks == 8)    S = 64;
        else if (blocks == 16)   S = 128;
        else if (blocks == 32)   S = 256;
        else if (blocks == 128)  S = 512;
        else if (blocks == 512)  S = 1024;
        if (S == 0) return QHPI_Success;

        const uint32_t M = S, K = S, N = S;
        const uint32_t M_t = M / 32, N_t = N / 32, K_t = K / 32;
        const uint32_t block_rows = (M / 4) < 64 ? (M / 4) : 64;
        const uint32_t k_chunks   = K / 32;

        /* Zero output buffer (size = M_t × N_t × 1024). */
        for (size_t i = 0; i < (size_t)M_t * N_t * 1024; i++) out_buf[i] = 0;

        /* Inline Crouton_8 act lookup */
        #define ACT_AT(m, k) ({                                               \
            uint32_t _rg = (m) / block_rows;                                  \
            uint32_t _kc = (k) >> 5;                                          \
            uint32_t _bi = _rg * k_chunks + _kc;                              \
            uint32_t _bo = ((m) % block_rows) * 32 + ((k) & 31);              \
            ((const uint8_t *)act_blocks[_bi])[_bo];                          \
        })
        #define WT_AT(k, n) ({                                                \
            uint32_t _kt = (k) >> 5, _kr = (k) & 31;                          \
            uint32_t _nt = (n) >> 5, _nc = (n) & 31;                          \
            uint32_t _dst = (_kr / 4) * 128 + _nc * 4 + (_kr % 4);            \
            (int32_t)(int8_t)wt_pack[(_nt * K_t + _kt) * 1024 + _dst];        \
        })

        /* Native bias layout helpers — bias_bytes[n_tile][256B] */
        #define BIAS_TILE_BASE(nt) (bias_bytes + (nt) * 256)
        /* Lower 128 B per tile = 32 × (fp16 scale, fp16 baseline) per channel */
        #define BIAS_SCALE_FP16(nt, c) \
            (*(const uint16_t *)(BIAS_TILE_BASE(nt) + (c) * 4 + 0))
        #define BIAS_BASELINE_U16(nt, c) \
            (*(const uint16_t *)(BIAS_TILE_BASE(nt) + (c) * 4 + 2))
        /* Upper 128 B per tile = 32 × int32 effective_bias */
        #define BIAS_EFF_I32(nt, c) \
            (*(const int32_t *)(BIAS_TILE_BASE(nt) + 128 + (c) * 4))

        /* fp16-bits → fp32 (subset, no NaN/Inf handling — fine for our test). */
        auto fp16_to_fp32 = [](uint16_t h) -> float {
            uint32_t s = (h >> 15) & 1;
            int32_t  e = (h >> 10) & 0x1F;
            uint32_t m = h & 0x3FF;
            uint32_t bits;
            if (e == 0) {
                if (m == 0) { bits = s << 31; }
                else { /* subnormal */
                    int shift = 0; while ((m & 0x400) == 0) { m <<= 1; shift++; }
                    e = 1 - shift;
                    m &= 0x3FF;
                    bits = (s << 31) | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
                }
            } else if (e == 31) {
                bits = (s << 31) | (0xFF << 23) | (m << 13);
            } else {
                bits = (s << 31) | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
            }
            float f; __builtin_memcpy(&f, &bits, 4); return f;
        };

        /* Main matmul loop. */
        for (uint32_t mt = 0; mt < M_t; mt++) {
            for (uint32_t nt = 0; nt < N_t; nt++) {
                uint8_t *tile = out_buf + (mt * N_t + nt) * 1024;
                for (uint32_t mr = 0; mr < 32; mr++) {
                    for (uint32_t nc = 0; nc < 32; nc++) {
                        uint32_t m = mt * 32 + mr;
                        uint32_t n = nt * 32 + nc;
                        /* acc init = effective_int32[c] (host fold) */
                        int32_t acc = BIAS_EFF_I32(nt, nc);
                        /* MAC */
                        for (uint32_t k = 0; k < K; k++) {
                            acc += (int32_t)ACT_AT(m, k) * WT_AT(k, n);
                        }
                        /* Direct saturate (scale=1, zp=0 in our test).
                         * TODO: use BIAS_SCALE_FP16/BIAS_BASELINE_U16 once
                         * the scale path is validated separately. */
                        if (acc < 0)        acc = 0;
                        else if (acc > 255) acc = 255;
                        tile[mr * 32 + nc] = (uint8_t)acc;
                    }
                }
            }
        }

        #undef ACT_AT
        #undef WT_AT
        #undef BIAS_TILE_BASE
        #undef BIAS_SCALE_FP16
        #undef BIAS_BASELINE_U16
        #undef BIAS_EFF_I32
    }
    return QHPI_Success;
#elif defined(V9_INPUT_PROBE)
    /* Phase-1 input-access probe: dump layout/shape/storage info for each
     * input + the FIRST 4 BYTES of its actual data into the output marker.
     * Validates that we can read all 3 input tensors before doing matmul.
     *
     * Output marker layout (first 64 B of output, after that all zero):
     *   [0]    = 0xA5
     *   [1..3] = input[0] layout, storage, block_table_length (low 8 b each)
     *   [4..7] = input[0] block[0] first 4 bytes (act)
     *   [8..15]= input[1] layout, storage; first 4 bytes of wt; act_zp_byte_0
     *   [16..19] = input[2] layout, storage, dims[0] low; bias_q[0] low byte
     *   [20..23] = bias_q[0] (raw int32 bytes — full 32 bits)
     *   [24..27] = bias_q[1]
     *   [28..31] = bias_q[N-1]   (last channel)
     *   [32..35] = sum_w[0]      (Σ_k W[k,0] computed scalar)
     *   [63]   = 0x5A */
    {
        uint8_t *out_buf = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);
        QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
        size_t out_bytes = 1;
        for (uint32_t i = 0; i < os.rank; i++) out_bytes *= os.dims[i];
        for (size_t i = 0; i < out_bytes; i++) out_buf[i] = 0;
        if (!out_buf || out_bytes < 64) return QHPI_Success;

        out_buf[0] = 0xA5;
        out_buf[1] = (uint8_t)qhpi_tensor_layout(inputs[0]);
        out_buf[2] = (uint8_t)qhpi_tensor_placement(inputs[0]);
        out_buf[3] = (uint8_t)qhpi_tensor_block_table_length(inputs[0]);

        /* Probe Crouton_8 layout: dump first 4 bytes of multiple blocks,
         * also block[0] byte 32 and 64 (to check row-stride). */
        void **act_blocks = qhpi_tensor_block_table(inputs[0]);
        uint32_t n_blocks = qhpi_tensor_block_table_length(inputs[0]);
        if (act_blocks && n_blocks > 0) {
            const uint8_t *blk0 = (const uint8_t *)act_blocks[0];
            if (blk0) {
                /* First 4 bytes of block 0 (gen produces (i*37)%256 row-major). */
                out_buf[4] = blk0[0];
                out_buf[5] = blk0[1];
                out_buf[6] = blk0[2];
                out_buf[7] = blk0[3];
            }
            /* Block[1] first byte at out_buf[52] (only if rank-1 block exists) */
            if (n_blocks > 1) {
                const uint8_t *blk1 = (const uint8_t *)act_blocks[1];
                if (blk1) out_buf[52] = blk1[0];
            }
            /* Block[N_blocks-1] first byte at out_buf[53] */
            const uint8_t *blkLast = (const uint8_t *)act_blocks[n_blocks - 1];
            if (blkLast) out_buf[53] = blkLast[0];

            /* Block[0] byte at offset 32 (stride probe — should be act[0,32]
             * if block is row-major 8-rows × K-cols, else act[1,0]) */
            if (blk0) out_buf[54] = blk0[32];
            if (blk0) out_buf[55] = blk0[33];
        } else {
            out_buf[4] = 0xDD; /* no block table */
        }

        out_buf[8] = (uint8_t)qhpi_tensor_layout(inputs[1]);
        out_buf[9] = (uint8_t)qhpi_tensor_placement(inputs[1]);
        const uint8_t *wt = (const uint8_t *)qhpi_tensor_raw_data(inputs[1]);
        if (wt) {
            out_buf[10] = wt[0]; out_buf[11] = wt[1];
            out_buf[12] = wt[2]; out_buf[13] = wt[3];
        } else {
            out_buf[10] = 0xCC;
        }

        out_buf[16] = (uint8_t)qhpi_tensor_layout(inputs[2]);
        out_buf[17] = (uint8_t)qhpi_tensor_placement(inputs[2]);

        /* Derive S from act block_table_length BEFORE any bias[] reads
         * (bias[last] indexed via bs.dims[0] when rank=0 caused subsequent
         * writes to be lost on Hexagon at ≥64³ — likely OOB read fault). */
        out_buf[48] = 0xC0;
        uint32_t blocks = qhpi_tensor_block_table_length(inputs[0]);
        out_buf[49] = (uint8_t)(blocks & 0xFF);
        uint32_t S = 0;
        if (blocks == 4)         S = 32;
        else if (blocks == 8)    S = 64;
        else if (blocks == 16)   S = 128;
        else if (blocks == 32)   S = 256;
        else if (blocks == 128)  S = 512;
        else if (blocks == 512)  S = 1024;
        out_buf[50] = (uint8_t)(S & 0xFF);
        out_buf[51] = (uint8_t)((S >> 8) & 0xFF);

        uint32_t K_t = S / 32;
        uint32_t N_t = S / 32;
        /* Use byte stores instead of int32 — Hexagon HVX may be reordering
         * misaligned int32 writes in surprising ways. */
        out_buf[36] = (uint8_t)(S & 0xFF);
        out_buf[37] = (uint8_t)((S >> 8) & 0xFF);
        out_buf[38] = (uint8_t)((S >> 16) & 0xFF);
        out_buf[39] = (uint8_t)((S >> 24) & 0xFF);
        out_buf[40] = out_buf[36]; out_buf[41] = out_buf[37];
        out_buf[42] = out_buf[38]; out_buf[43] = out_buf[39];
        out_buf[44] = out_buf[36]; out_buf[45] = out_buf[37];
        out_buf[46] = out_buf[38]; out_buf[47] = out_buf[39];

        /* Bias reads — bounded by S (= N for square test). */
        const int32_t *bias = (const int32_t *)qhpi_tensor_raw_data(inputs[2]);
        if (bias && S > 0) {
            int32_t b0 = bias[0];
            out_buf[19] = (uint8_t)(b0 & 0xFF);
            *(int32_t *)(out_buf + 20) = b0;
            *(int32_t *)(out_buf + 24) = (S > 1) ? bias[1] : b0;
            *(int32_t *)(out_buf + 28) = bias[S - 1];
        } else if (bias) {
            *(int32_t *)(out_buf + 20) = bias[0];
        } else {
            out_buf[20] = 0xBB;
        }

        /* Σ_k W[k, 0] over native pre-packed wt */
        int32_t sum_w_c0 = 0;
        if (wt && K_t > 0 && N_t > 0) {
            for (uint32_t kt = 0; kt < K_t; kt++) {
                const uint8_t *tile = wt + (kt * N_t + 0) * 1024;
                for (uint32_t r = 0; r < 32; r++) {
                    uint32_t dst = (r / 4) * 128 + (r % 4);
                    sum_w_c0 += (int32_t)(int8_t)tile[dst];
                }
            }
        }
        *(int32_t *)(out_buf + 32) = sum_w_c0;

        out_buf[63] = 0x5A;
    }
    return QHPI_Success;
#elif defined(V9_USE_NATIVE_KERNEL)
    /* Step 5.2 — descriptor-driven HMX kernel via dlsym to libQnnHtpV75Skel.so::
     * hmx_convbbb1x1_stride1 (= q::ConvLayer_s1.opt body for u8×i8 1×1 conv).
     *
     * Per-(mt, nt) tile call: 1 output tile via K_t MACs.
     *   out_desc.n_tiles_pow2 = 1  (loop1 = 1 sat.ub drain)
     *   act_desc.n_act_pairs = K_t (loop0 walks K_t/2 pairs, 2 MACs each)
     *   k_total_bytes = 32         (single outer K-iter — all K covered by loop0)
     *
     * Mask descriptor: built once via dlsym set_hmx_params_conv1x1(0x700,0,0,0,0)
     * per Step 5.1 probe characterization (Agent/qnn_re/set_hmx_params_conv1x1_probe_2026-04-28.md):
     *   +0x04 out_rt_mask  = 0x700  (sat.ub Rt — native, NOT V8's 0x3FF)
     *   +0x0c act_rt_base  = 0x71F  (:cm Rt — matches V8's HMX_RT_ACT_CM)
     *   +0x18 alt_rt       = 0x3FF  (last-K Rt_wt — matches V8's HMX_RT_WT)
     */
    {
#if defined(V9_PROBE_KERNEL_CYC)
        uint64_t cyc_op_start;
        asm volatile ("%0 = C15:14" : "=r"(cyc_op_start));
#endif
#if defined(V9_PMU_PROBE)
        /* Configure PMU. PMUCFG=0x400 (bit 10) enables counting per docs.
         * Setting PMUEVTCFG resets cnt0..3; assigns events to counters. */
        qurt_pmu_enable(1);
        qurt_pmu_set(QURT_PMUCFG, 0x400);
        /* cnt0 = COMMITTED_PKT_ANY (all threads)
         * cnt1 = COMMITTED_PKT_T0  (thread 0 only — should match per-op count)
         * cnt2 = COMMITTED_INSTS   (sub-packet — counts individual instructions)
         * cnt3 = DISPATCHED_PKTS   (dispatched, including squashed) */
        uint32_t _evtcfg = ((uint32_t)PMU_DISPATCHED_PKTS   << 24)
                         | ((uint32_t)PMU_COMMITTED_INSTS   << 16)
                         | ((uint32_t)PMU_COMMITTED_PKT_T0  <<  8)
                         | ((uint32_t)PMU_COMMITTED_PKT_ANY      );
        qurt_pmu_set(QURT_PMUEVTCFG, _evtcfg);
        uint32_t pkt_op_start = qurt_pmu_get(QURT_PMUCNT0);
        uint64_t cyc_op_start_64;
        asm volatile ("%0 = C15:14" : "=r"(cyc_op_start_64));
#endif
        void **act_blocks = qhpi_tensor_block_table(inputs[0]);
        const uint8_t *wt_pack    = (const uint8_t *)qhpi_tensor_raw_data(inputs[1]);
        const uint8_t *bias_bytes = (const uint8_t *)qhpi_tensor_raw_data(inputs[2]);
        void **out_blocks = qhpi_tensor_block_table(outputs[0]);
        if (!act_blocks || !wt_pack || !bias_bytes || !out_blocks) return QHPI_Success;

        uint32_t blocks = qhpi_tensor_block_table_length(inputs[0]);
        uint32_t S = 0;
        if (blocks == 4)         S = 32;
        else if (blocks == 8)    S = 64;
        else if (blocks == 16)   S = 128;
        else if (blocks == 32)   S = 256;
        else if (blocks == 128)  S = 512;
        else if (blocks == 512)  S = 1024;
        if (S == 0) return QHPI_Success;

        const uint32_t M = S, K = S, N = S;
        const uint32_t M_t = M / 32, N_t = N / 32, K_t = K / 32;
        const uint32_t block_rows = (M / 4) < 64 ? (M / 4) : 64;
        const uint32_t mt_per_block = block_rows / 32;
        const uint32_t k_chunks = K_t;  /* same as input K-tile count */

        /* Build mask_desc once (kernel-static — Rt masks don't depend on (mt,nt,kt)).
         *
         * 2026-04-28 RE finding: caller sites in libHtpPrepare.so::0xd998f0 use
         * arg1=0x700 in some blocks, 0x70b in others. The 0xb low nibble (bits
         * 0,1,3) is the suspected "deep mode" flag — V73DEEP path needs the
         * same mask config native uses. */
        static uint32_t mask_buf[16] __attribute__((aligned(16)));
        static int mask_initialized = 0;
        if (!mask_initialized) {
            for (uint32_t i = 0; i < 16; i++) mask_buf[i] = 0;
#if defined(V9_NATIVE_V73DEEP)
            /* Deep variant: from set_hmx_params_conv1x1 disasm at 0xd998f0:
             *   mask[+0x30] = arg5  (this is the "deep" flag the v73
             *   dispatcher checks via tstbit(r4, #5))
             *   mask[+0x0c] (act_rt_base) also depends on arg5 bit 5:
             *     test $0x20, %r9b ; cmove → r15d
             *
             * arg1 (0x700 vs 0x70b) controls bit-shuffle of mask[+0x08]
             * and mask[+0x0c] when arg1 & 3 != 0. Deep call sites in
             * libHtpPrepare.so use 0x70b.
             *
             * arg4: when arg5 bit 5 set, arg4 ≥ 8 bumps mask[+0x18] alt_rt
             * from 0x3FF → 0x7FF, doubling wt MAC stride from 4096 to 8192.
             * V73DEEP_ARG4 default 0 (alt_rt=0x3FF). Compile with
             * -DV73DEEP_ARG4=8 to force alt_rt=0x7FF and see effect. */
#ifndef V73DEEP_ARG4
#define V73DEEP_ARG4 0
#endif
#ifndef V73DEEP_ARG1
#define V73DEEP_ARG1 0x70b
#endif
#ifndef V73DEEP_ARG5
#define V73DEEP_ARG5 0x20
#endif
            set_hmx_params_conv1x1(mask_buf, V73DEEP_ARG1, 0, 0, V73DEEP_ARG4, V73DEEP_ARG5);
#else
            set_hmx_params_conv1x1(mask_buf, 0x700, 0, 0, 0, 0);
#endif
            mask_initialized = 1;
        }
        const hmx_conv_mask_desc_t *md = (const hmx_conv_mask_desc_t *)mask_buf;

#if defined(V9_PROBE_KERNEL_CYC)
        uint64_t cyc_after_setup;
        asm volatile ("%0 = C15:14" : "=r"(cyc_after_setup));
#endif

#if defined(V9_NATIVE_PER_NT_V2)
        /* Failed batching attempt — see commit history + memory entry. */
#elif defined(V9_NATIVE_SINGLE_CALL)
        /* Single-call descriptor-driven matmul. ONE kernel call covers entire
         * matmul (M_t × N_t × K_t MACs). Discovery via full HMX disasm of
         * hmx_convbbb1x1_stride1 (Agent/qnn_re/hmx_convbbb1x1_stride1_FULL.S
         * with --mattr=+hvxv75,+hmx,+hvx-length128b decoded `<unknown>`):
         *
         * Outer K-iter at 2ea800:
         *   r3 += 0x100;  r13 -= 0x20;  r17 = r12;  bias = mxmem2(r3)
         *   → bias auto-walks 256 B per K-iter (matches N-tile bias stride!)
         *
         * Loop1 end at 2ea86c:
         *   r27:26 = combine(r8, r4);  r8 = r2
         *   → r27 captures wt end position (= r2 + K_t*0x400 after loop1 done)
         * Outer K-iter end at 2ea888-94:
         *   r2 = r27
         *   → wt auto-walks K_t*1024 B per K-iter (matches [N_t, K_t, 1024] layout!)
         *
         * Setup for one matmul (FIXED 2026-04-28: m_total_minus_step was wrong):
         *   k_total_bytes      = N_t * 32  → N_t outer iters (= one per nt; bias
         *                                    auto-walks +0x100 per outer = 1 N-tile)
         *   m_total_minus_step = 8         → r17 = 8, sub r22=8 → 0, 1 K-iter
         *                                    (was M_t*64 = 64 redundant K-iters!)
         *   n_tiles_pow2       = M_t * 8   → r20 = M_t loop1 iters (one drain per mt)
         *   act_pairs flat     = M_t × K_t (mt outer, kt inner) — kernel reads K_t
         *                                    consecutive ptrs per loop1 iter, then
         *                                    advances by K_t for next mt
         *   act_table_y_stride = 0         → r15 = 0 (irrelevant since 1 K-iter)
         *   n_act_pairs        = K_t       → K_t MACs per loop1 iter
         *   out_table_stride   = N_t       → m0 = N_t*4 bytes; loop1 walks rows of
         *                                    out_tbl[mt * N_t + nt] for fixed nt
         *   out_tbl[mt*N_t+nt] = ptr for (mt,nt); outer-iter advances base by 4 (=
         *                                    column nt+1)
         *   wt_base            = wt_pack — kernel walks K_t*1024 per outer via
         *                                    r27 = r8_after_loop1; matches N-outer
         *                                    wt[nt][kt] layout (gen_v8c8_test.py)
         *   bias_base          = bias_bytes — kernel walks +0x100 per outer (256B
         *                                    per N-tile = native bias_to_vtcm fold)
         *
         * Total drains = N_t outer × 1 K-iter × M_t loop1 = M_t × N_t ✓
         * Total MACs   = total drains × K_t = M_t × N_t × K_t ✓
         */
        int32_t act_tbl_all[1024] __attribute__((aligned(64))); /* M_t × K_t up to 32×32 = 1024 (1024³) */
        int32_t out_tbl_all[1024] __attribute__((aligned(64))); /* M_t × N_t up to 32×32 = 1024 (1024³) */
        /* Lane A round 2 (2026-04-28): split by mt_per_block so the divide
         * `M_t / mt_per_block` and inb branch go away. mt_per_block ∈ {1,2}
         * for all our supported shapes (S ∈ [128, 1024]). The `void **`
         * block table is pointer-sized (32-bit on Hexagon), so we can read
         * its entries as int32_t directly with no cast pipeline.
         *
         * Round 1 (separate inb loop) was ~250 pkts at 256³ — compiler
         * generated 2-packet loop body + udivsi3 call. Round 2 unrolls inb
         * so each load fans into both writes in one packet. */
        const int32_t *act_src_int = (const int32_t *)act_blocks;
        const int32_t *out_src_int = (const int32_t *)out_blocks;
#if defined(__hexagon__)
        /* HVX fast path: when K_t * mt_blocks * sizeof(int32_t) >= 128 (= 1 vector),
         * we batch the {ptr, ptr+1024} interleave with one vshuff per 32-ptr chunk.
         * For 256³ exactly 32 ptrs, 1 vector each side. mt_per_block must be 2.
         * Rt=32 in vshuff = 32-byte (= 8 int32) chunk swap pass — exactly K_t=8
         * for 256³, so output pattern is dst[0..8]=src+0, dst[8..16]=src+1024,
         * dst[16..24]=src+0(next chunk), dst[24..32]=src+1024(next chunk), ...
         * which is the desired interleave by inb. */
        if (mt_per_block == 2 && K_t == 8 && N_t == 8) {
#if defined(V73D_NATIVE_LAYOUT)
            /* P1.5 native layout: 32 entries, each = first M-tile of M-pair.
             * Direct copy of act_blocks (4 groups * 8 K-tiles = 32 ptrs). */
            std::memcpy(act_tbl_all, act_blocks, 32 * sizeof(int32_t));
            std::memcpy(out_tbl_all, out_blocks, 32 * sizeof(int32_t));
#else
            HVX_Vector v_acts, v_outs;
            std::memcpy(&v_acts, act_blocks, sizeof(HVX_Vector));
            std::memcpy(&v_outs, out_blocks, sizeof(HVX_Vector));
            HVX_Vector v_1024 = Q6_V_vsplat_R(1024);
            HVX_Vector v_acts_hi = Q6_Vw_vadd_VwVw(v_acts, v_1024);
            HVX_Vector v_outs_hi = Q6_Vw_vadd_VwVw(v_outs, v_1024);
            HVX_VectorPair vp_acts = Q6_W_vshuff_VVR(v_acts_hi, v_acts, 32);
            HVX_VectorPair vp_outs = Q6_W_vshuff_VVR(v_outs_hi, v_outs, 32);
            HVX_Vector vp_acts_lo = Q6_V_lo_W(vp_acts), vp_acts_hi2 = Q6_V_hi_W(vp_acts);
            HVX_Vector vp_outs_lo = Q6_V_lo_W(vp_outs), vp_outs_hi2 = Q6_V_hi_W(vp_outs);
            std::memcpy(&act_tbl_all[ 0], &vp_acts_lo,  sizeof(HVX_Vector));
            std::memcpy(&act_tbl_all[32], &vp_acts_hi2, sizeof(HVX_Vector));
            std::memcpy(&out_tbl_all[ 0], &vp_outs_lo,  sizeof(HVX_Vector));
            std::memcpy(&out_tbl_all[32], &vp_outs_hi2, sizeof(HVX_Vector));
#endif
        } else
#endif
        if (mt_per_block == 2) {
            const uint32_t mt_blocks = M_t >> 1;
#if defined(V73D_NATIVE_LAYOUT)
            /* P1.5 native layout: 32-entry tables, each entry = ptr to 2KB
             * region (M-pair packed). Native uses arg1=0x700 → mask[+0x0c]=0x71f
             * which makes :deep:cm fetch 2 M-tiles from a single 2KB pointer. */
            for (uint32_t rg = 0; rg < mt_blocks; rg++) {
                const int32_t *__restrict__ a_src = act_src_int + rg * K_t;
                const int32_t *__restrict__ o_src = out_src_int + rg * N_t;
                int32_t *__restrict__ a_dst = act_tbl_all + rg * K_t;
                int32_t *__restrict__ o_dst = out_tbl_all + rg * N_t;
                for (uint32_t kt = 0; kt < K_t; kt++) a_dst[kt] = a_src[kt];
                for (uint32_t nt = 0; nt < N_t; nt++) o_dst[nt] = o_src[nt];
            }
#else
            for (uint32_t rg = 0; rg < mt_blocks; rg++) {
                const int32_t *__restrict__ a_src = act_src_int + rg * K_t;
                const int32_t *__restrict__ o_src = out_src_int + rg * N_t;
                int32_t *__restrict__ a_dst0 = act_tbl_all + (rg << 1) * K_t;
                int32_t *__restrict__ a_dst1 = a_dst0 + K_t;
                int32_t *__restrict__ o_dst0 = out_tbl_all + (rg << 1) * N_t;
                int32_t *__restrict__ o_dst1 = o_dst0 + N_t;
                for (uint32_t kt = 0; kt < K_t; kt++) {
                    int32_t a = a_src[kt];
                    a_dst0[kt] = a;
                    a_dst1[kt] = a + 1024;
                }
                for (uint32_t nt = 0; nt < N_t; nt++) {
                    int32_t o = o_src[nt];
                    o_dst0[nt] = o;
                    o_dst1[nt] = o + 1024;
                }
            }
#endif
        } else { /* mt_per_block == 1 */
            for (uint32_t rg = 0; rg < M_t; rg++) {
                const int32_t *__restrict__ a_src = act_src_int + rg * K_t;
                const int32_t *__restrict__ o_src = out_src_int + rg * N_t;
                int32_t *__restrict__ a_dst = act_tbl_all + rg * K_t;
                int32_t *__restrict__ o_dst = out_tbl_all + rg * N_t;
                for (uint32_t kt = 0; kt < K_t; kt++) a_dst[kt] = a_src[kt];
                for (uint32_t nt = 0; nt < N_t; nt++) o_dst[nt] = o_src[nt];
            }
        }
#if defined(V9_PROBE_KERNEL_CYC)
        uint64_t cyc_after_table;
        asm volatile ("%0 = C15:14" : "=r"(cyc_after_table));
#endif
        /* Descriptor sweep knobs (override via -D). Defaults are the
         * baseline that gives 100% bit-exact at 747 pkts for V73DEEP. */
#ifndef V73D_N_TILES_POW2
#define V73D_N_TILES_POW2 (M_t * 8)        /* +0x0c: r20 = (val+7)>>3 → loop1 trip */
#endif
#ifndef V73D_M_TOTAL_MINUS_STEP
#define V73D_M_TOTAL_MINUS_STEP 8           /* +0x10: → r17 = val, sub r22=8 → K-iters */
#endif
#ifndef V73D_K_TOTAL_BYTES
#define V73D_K_TOTAL_BYTES (N_t * 32)       /* +0x14: r13 = (val+0x1f)>>5 → outer trip */
#endif
#ifndef V73D_N_ACT_PAIRS
#define V73D_N_ACT_PAIRS K_t                /* +0x04 act_desc: r4 = val → loop0 (val/2) */
#endif
#ifndef V73D_EXTRA_PARAM_0
#define V73D_EXTRA_PARAM_0 1u
#endif
#ifndef V73D_EXTRA_PARAM_1
#define V73D_EXTRA_PARAM_1 0u
#endif
#ifndef V73D_OUT_Y_STRIDE
#define V73D_OUT_Y_STRIDE 0u           /* P1.4: native uses 32 (= M_t * 4 dwords) */
#endif
#ifndef V73D_AD_ACT_Y_STRIDE
#define V73D_AD_ACT_Y_STRIDE 0u        /* P1.4: native uses 32 (= M_t * 4 dwords) */
#endif
        hmx_conv_out_desc_t od = {
            out_tbl_all,
            (uint32_t)N_t,                  /* +0x04: out_table_stride_dwords = N_t */
            (uint32_t)(V73D_OUT_Y_STRIDE),  /* +0x08: out_y_stride_words */
            (uint32_t)(V73D_N_TILES_POW2),  /* +0x0c */
            (int32_t)(V73D_M_TOTAL_MINUS_STEP), /* +0x10 */
            (uint32_t)(V73D_K_TOTAL_BYTES)  /* +0x14 */
        };
        hmx_conv_act_desc_t ad = {
            act_tbl_all,
            (uint32_t)(V73D_N_ACT_PAIRS),   /* +0x04 */
            (uint32_t)(V73D_AD_ACT_Y_STRIDE) /* +0x08 */
        };
#if defined(V9_NATIVE_V73DEEP)
        /* hmx_v73_convbbb1x1deep_stride1 — REAL deep variant: :deep on both
         * activation+weight, 2 cvt drains per loop1 iter, N-fanout=2.
         *
         * 2026-04-28: ctx-binary diff RE finding (Agent/qnn_re/
         * v73deep_wt_layout_DECODED_2026-04-28.md): native uses K-major outer
         * `[K_t, N_t, 1024]` layout for wt (vs our V8C8 N-major
         * `[N_t, K_t, 1024]`). Within-tile = same 4-row interleave.
         *
         * Caller (gen_v8c8_chain.py) MUST pack wt in K-major layout when
         * V9_NATIVE_V73DEEP is enabled (--wt_layout=kmaj). We pass the bytes
         * verbatim — no runtime repack (would overflow op-pkg .bss).
         */
        uint32_t extra_param[16] __attribute__((aligned(16))) = { V73D_EXTRA_PARAM_0, V73D_EXTRA_PARAM_1 };

#if defined(V73D_MASK_38_EXTRA_PTR)
        /* P1.4 finding: native sets mask[+0x38] = extra_param ptr. Mask is
         * cached static, but we patch it each call since extra_param is
         * stack-local. */
        ((uint32_t *)mask_buf)[0x38/4] = (uint32_t)(uintptr_t)extra_param;
#endif

#if defined(V9_DESC_DUMP)
        /* Dump pre-call descriptors to the Crouton_8 output and skip the
         * kernel call. After UntileToRowMajor, the values land at row r,
         * cols 0..127 of the row-major output (r ∈ 0..4). Decode with
         * scripts/parse_v73deep_desc_dump.py.
         *
         * Row layout (each 128 bytes; first u32 is a magic 0xD0DE000r):
         *   row 0: shape header (M,N,K,M_t,N_t,K_t, wt_pack, bias_bytes) +
         *          mask_buf[0..63]
         *   row 1: od (out_table, stride, y_stride, n_tiles_pow2,
         *          m_total_minus_step, k_total_bytes) + ad +
         *          extra_param[0..15]
         *   row 2: act_tbl_all[0..30] (the M_t×K_t pointers we built —
         *          first 31 entries; full table is in actual ad.act_pairs)
         *   row 3: out_tbl_all[0..30]
         *   row 4: derived "candidate sd tile count" values per the wrapper
         *          formula sd_tile_count = (sd[0x18]>>3)*(sd[0x1c]>>3)*(sd[0x20]>>5)
         *          — multiple guesses for what (H,W,D) maps to (M,K,N) */
        {
            const uint32_t out_n_chunks = N / 32;
            const uint32_t total_out_blocks = qhpi_tensor_block_table_length(outputs[0]);
            const uint32_t bytes_per_block = total_out_blocks > 0 ? (uint32_t)(M * N) / total_out_blocks : 0;
            for (uint32_t bi = 0; bi < total_out_blocks; bi++) {
                uint8_t *blk = (uint8_t *)out_blocks[bi];
                if (blk) for (uint32_t k = 0; k < bytes_per_block; k++) blk[k] = 0;
            }

            uint8_t row_bytes[5][128];
            for (uint32_t i = 0; i < 5; i++)
                for (uint32_t j = 0; j < 128; j++) row_bytes[i][j] = 0;

            uint32_t *r0w = (uint32_t *)row_bytes[0];
            r0w[0] = 0xD0DE0000u;
            r0w[1] = M; r0w[2] = N; r0w[3] = K;
            r0w[4] = M_t; r0w[5] = N_t; r0w[6] = K_t;
            r0w[7] = (uint32_t)(uintptr_t)wt_pack;
            r0w[8] = (uint32_t)(uintptr_t)bias_bytes;
            r0w[9] = mt_per_block;
            r0w[10] = block_rows;
            r0w[11] = (uint32_t)(uintptr_t)act_blocks;
            r0w[12] = (uint32_t)(uintptr_t)out_blocks;
            for (uint32_t i = 0; i < 16; i++) r0w[16 + i] = mask_buf[i];

            uint32_t *r1w = (uint32_t *)row_bytes[1];
            r1w[0] = 0xD0DE0001u;
            r1w[1] = (uint32_t)(uintptr_t)od.out_tile_ptr_table;
            r1w[2] = od.out_table_stride_dwords;
            r1w[3] = od.out_y_stride_words;
            r1w[4] = od.n_tiles_pow2;
            r1w[5] = (uint32_t)od.m_total_minus_step;
            r1w[6] = od.k_total_bytes;
            r1w[7] = (uint32_t)(uintptr_t)ad.act_ptr_pairs;
            r1w[8] = ad.n_act_pairs;
            r1w[9] = ad.act_table_y_stride_words;
            for (uint32_t i = 0; i < 16; i++) r1w[10 + i] = extra_param[i];

            uint32_t *r2w = (uint32_t *)row_bytes[2];
            r2w[0] = 0xD0DE0002u;
            const uint32_t n_act_dump = (M_t * K_t) < 31 ? (M_t * K_t) : 31;
            for (uint32_t i = 0; i < n_act_dump; i++) r2w[1 + i] = (uint32_t)act_tbl_all[i];

            uint32_t *r3w = (uint32_t *)row_bytes[3];
            r3w[0] = 0xD0DE0003u;
            const uint32_t n_out_dump = (M_t * N_t) < 31 ? (M_t * N_t) : 31;
            for (uint32_t i = 0; i < n_out_dump; i++) r3w[1 + i] = (uint32_t)out_tbl_all[i];

            uint32_t *r4w = (uint32_t *)row_bytes[4];
            r4w[0] = 0xD0DE0004u;
            r4w[1] = (uint32_t)(M_t * K_t);
            r4w[2] = (uint32_t)(N_t * K_t);
            r4w[3] = (uint32_t)(M_t * N_t);
            /* Wrapper-formula candidates for sd[0x18,0x1c,0x20] semantics */
            r4w[4] = (uint32_t)((M >> 3) * (K >> 3) * (K >> 5));   /* HWD = M,K,K */
            r4w[5] = (uint32_t)((K >> 3) * (N >> 3) * (N >> 5));   /* HWD = K,N,N */
            r4w[6] = (uint32_t)((M >> 3) * (1u) * (K >> 5));        /* HWD = M,1,K (W=1) */
            r4w[7] = (uint32_t)((1u) * (K >> 3) * (N >> 5));        /* HWD = 1,K,N */

            for (uint32_t r = 0; r < 5; r++) {
                const uint32_t m_row = r;
                for (uint32_t n_col = 0; n_col < 128 && n_col < N; n_col++) {
                    uint32_t block_idx = ((m_row % 32) / 8) * out_n_chunks + (n_col / 32);
                    uint32_t inblk_off = (m_row / 32) * 256 + (m_row % 8) * 32 + (n_col % 32);
                    if (block_idx >= total_out_blocks) continue;
                    uint8_t *blk = (uint8_t *)out_blocks[block_idx];
                    if (blk) blk[inblk_off] = row_bytes[r][n_col];
                }
            }
        }
        return QHPI_Success;
#endif

#if defined(V9_PROBE_REGIONS)
        /* Differential PMU probe: run the V73DEEP kernel with 4 descriptor
         * variants that selectively shrink each loop dimension, and record
         * committed-packet count per variant. Solve a linear model:
         *
         *   P(r13_eff, r20, r28) = A + r13_eff * (B + r20 * (C + r28 * D))
         *
         * where r13_eff = (r13 + 1) / 2 (kernel decrements r13 by 2 per outer),
         * A = prologue+epilogue, B = per-outer overhead (bias loads + setup),
         * C = per-loop1 overhead (drain prep + 2 drain stores),
         * D = per-loop0 packet count (~K-MAC per loop0 iter).
         *
         * 4 variants, output writes to first 4 logical "rows" via Crouton_8 scatter.
         * Each row 64 bytes contains: magic | variant_id | n_tiles_pow2 | k_total_bytes |
         *                              n_act_pairs | pkt_count | cyc_count | inst_count
         */
        {
            qurt_pmu_enable(1);
            qurt_pmu_set(QURT_PMUCFG, 0x400);
            uint32_t _evtcfg = ((uint32_t)PMU_DISPATCHED_PKTS   << 24)
                             | ((uint32_t)PMU_COMMITTED_INSTS   << 16)
                             | ((uint32_t)PMU_COMMITTED_PKT_T0  <<  8)
                             | ((uint32_t)PMU_COMMITTED_PKT_ANY      );
            qurt_pmu_set(QURT_PMUEVTCFG, _evtcfg);

            const uint32_t out_n_chunks = N / 32;
            const uint32_t total_out_blocks_p = qhpi_tensor_block_table_length(outputs[0]);
            const uint32_t bytes_per_block_p = total_out_blocks_p > 0 ? (uint32_t)(M * N) / total_out_blocks_p : 0;
            for (uint32_t bi = 0; bi < total_out_blocks_p; bi++) {
                uint8_t *blk = (uint8_t *)out_blocks[bi];
                if (blk) for (uint32_t k = 0; k < bytes_per_block_p; k++) blk[k] = 0;
            }

            struct Variant {
                uint32_t n_tiles_pow2;
                uint32_t k_total_bytes;
                uint32_t n_act_pairs;
            };
            const Variant variants[4] = {
                /* V0 baseline: r20=8, r13=8 (4 outers), r28=4 — full work */
                { (uint32_t)(M_t * 8),  (uint32_t)(N_t * 32), (uint32_t)K_t },
                /* V1 small M: r20=1 (1 loop1) */
                { 8,                    (uint32_t)(N_t * 32), (uint32_t)K_t },
                /* V2 small K-inner: r28=1 (1 loop0 iter) */
                { (uint32_t)(M_t * 8),  (uint32_t)(N_t * 32), 2 },
                /* V3 small N-outer: r13=1 (1 outer iter) */
                { (uint32_t)(M_t * 8),  32,                   (uint32_t)K_t },
            };

            uint32_t pkts_v[4]  = {0, 0, 0, 0};   /* PMU_COMMITTED_PKT_ANY */
            uint32_t pkts_t0_v[4] = {0, 0, 0, 0}; /* PMU_COMMITTED_PKT_T0  */
            uint32_t cycs_v[4]  = {0, 0, 0, 0};
            uint32_t insts_v[4] = {0, 0, 0, 0};

            /* Phase 1: run all 4 variants, capturing PMU values per variant.
             * Kernel calls clobber the output buffer, so we write probe rows
             * AFTER all variants have run. */
            for (uint32_t v = 0; v < 4; v++) {
                hmx_conv_out_desc_t od_v = {
                    out_tbl_all,
                    (uint32_t)N_t,
                    0,
                    variants[v].n_tiles_pow2,
                    8,                                 /* keep m_total_minus_step = 8 */
                    variants[v].k_total_bytes
                };
                hmx_conv_act_desc_t ad_v = {
                    act_tbl_all,
                    variants[v].n_act_pairs,
                    0
                };

                uint32_t any_b  = qurt_pmu_get(QURT_PMUCNT0);
                uint32_t t0_b   = qurt_pmu_get(QURT_PMUCNT1);
                uint32_t inst_b = qurt_pmu_get(QURT_PMUCNT2);
                uint64_t cyc_b;
                asm volatile ("%0 = C15:14" : "=r"(cyc_b));

#if defined(V9_PROBE_V73_NONDEEP)
                /* probe non-deep V73 kernel with same descriptors */
                {
                    const uint32_t ep2[2] = { 1u, 0u };
                    hmx_v73_convbbb1x1_stride1(&od_v, &ad_v, wt_pack, bias_bytes, md, ep2);
                }
#elif defined(V9_PROBE_V73_UNALIGNED)
                /* probe v73 unaligned kernel with same descriptors */
                {
                    const uint32_t ep2[2] = { 1u, 0u };
                    hmx_v73_convbbb1x1_stride1_unaligned(&od_v, &ad_v, wt_pack, bias_bytes, md, ep2);
                }
#elif defined(V9_PROBE_OLD_KERNEL)
                /* probe OLD non-v73 kernel with same descriptors */
                hmx_convbbb1x1_stride1(&od_v, &ad_v, wt_pack, bias_bytes, md);
#else
                hmx_v73_convbbb1x1deep_stride1(&od_v, &ad_v, wt_pack, bias_bytes, md, extra_param);
#endif

                uint64_t cyc_a;
                asm volatile ("%0 = C15:14" : "=r"(cyc_a));
                uint32_t any_a  = qurt_pmu_get(QURT_PMUCNT0);
                uint32_t t0_a   = qurt_pmu_get(QURT_PMUCNT1);
                uint32_t inst_a = qurt_pmu_get(QURT_PMUCNT2);

                pkts_v[v]    = any_a  - any_b;
                pkts_t0_v[v] = t0_a   - t0_b;
                cycs_v[v]    = (uint32_t)(cyc_a - cyc_b);
                insts_v[v]   = inst_a - inst_b;
            }

            /* Phase 2: zero output then scatter probe rows. */
            for (uint32_t bi = 0; bi < total_out_blocks_p; bi++) {
                uint8_t *blk = (uint8_t *)out_blocks[bi];
                if (blk) for (uint32_t k = 0; k < bytes_per_block_p; k++) blk[k] = 0;
            }
            for (uint32_t v = 0; v < 4; v++) {
                uint8_t row_p[128];
                for (uint32_t i = 0; i < 128; i++) row_p[i] = 0;
                uint32_t *rw = (uint32_t *)row_p;
                rw[0] = 0xC0DE0000u | v;
                rw[1] = v;
                rw[2] = variants[v].n_tiles_pow2;
                rw[3] = variants[v].k_total_bytes;
                rw[4] = variants[v].n_act_pairs;
                rw[5] = pkts_v[v];
                rw[6] = cycs_v[v];
                rw[7] = insts_v[v];
                rw[8]  = (variants[v].n_tiles_pow2 + 7) >> 3;
                rw[9]  = (variants[v].k_total_bytes + 0x1f) >> 5;
                rw[10] = (rw[9] + 1) / 2;
                rw[11] = variants[v].n_act_pairs >> 1;
                rw[12] = pkts_t0_v[v];

                const uint32_t m_row = v;
                for (uint32_t n_col = 0; n_col < 128 && n_col < N; n_col++) {
                    uint32_t block_idx = ((m_row % 32) / 8) * out_n_chunks + (n_col / 32);
                    uint32_t inblk_off = (m_row / 32) * 256 + (m_row % 8) * 32 + (n_col % 32);
                    if (block_idx >= total_out_blocks_p) continue;
                    uint8_t *blk = (uint8_t *)out_blocks[block_idx];
                    if (blk) blk[inblk_off] = row_p[n_col];
                }
            }
        }
        return QHPI_Success;
#endif

#if defined(V9_NATIVE_V73DEEP_SPARSITY)
        /* DEAD-END 2026-04-28: tested but crashes on device. Disasm shows
         * activation MAC = `:cm` non-deep, wt `:deep`, single bias load per
         * outer. SIGSEGV on device — sparsity variant requires sparse-format
         * weight bytes (compressed sparse mask + values) which we can't easily
         * provide. Per packet count analysis, this variant also has SAME loop
         * structure as non-deep (1 drain/loop1) so wouldn't help anyway. */
        hmx_v73_convbbb1x1deep_stride1_sparsity(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#elif defined(V9_KERNEL_V73_UNALIGNED)
        hmx_v73_convbbb1x1_stride1_unaligned(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#elif defined(V9_KERNEL_OLD_UNALIGNED)
        hmx_convbbb1x1_stride1_unaligned(&od, &ad, wt_pack, bias_bytes, md);
#elif defined(V9_KERNEL_V73_BBB_NXN)
        hmx_v73_convbbb_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#elif defined(V9_KERNEL_V73_BBB_ALIGNED)
        hmx_v73_convbbb_stride1_aligned(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#elif defined(V9_KERNEL_V73DEEP_SPLIT)
        /* DEAD-END: see commit history. */
        hmx_v73_convbbb1x1deep_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#elif defined(V9_KERNEL_V73DEEP_PER_M)
        /* DEAD-END 2026-04-28 PM: tested per-M-tile multi-call (M_t=8 calls,
         * each with n_tiles_pow2=8 = 1 loop1 iter). Result: dur=3676 / pkts=1466
         * — WORSE than single-call baseline. Per-call kernel prologue/epilogue
         * overhead dominates: each call is ~183 packets (vs estimated ~46).
         * Multi-call only works if the kernel has a way to skip prologue
         * across calls (which it doesn't from outside). */
        hmx_v73_convbbb1x1deep_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#elif defined(V9_KERNEL_OLD_V73DESC)
        /* OLD non-v73 kernel called with V73DEEP-style descriptor.
         * Native q::ConvLayer_s1.opt at 256³ calls 0x2ea740 (per wrapper
         * disasm 0x3dc440). Mask args same as deep (arg5=0x20 sets deep
         * flag — but OLD kernel doesn't check that; it just reads its
         * Rt fields). */
        hmx_convbbb1x1_stride1(&od, &ad, wt_pack, bias_bytes, md);
#elif defined(V9_PROBE_KERNEL_CYC)
        /* Fine-grained pcycle probe. Stash 4× uint32 (16 B), all little-endian:
         *   bytes  0..3  = kernel_cyc (kernel call body)
         *   bytes  4..7  = desc_cyc   (od/ad/extra_param construction post-table)
         *   bytes  8..11 = table_cyc  (act_tbl_all + out_tbl_all build — Lane A target)
         *   bytes 12..15 = setup_cyc  (op-start through mask init incl. qhpi calls) */
        uint64_t cyc_before_kernel, cyc_after_kernel;
        asm volatile ("%0 = C15:14" : "=r"(cyc_before_kernel));
        hmx_v73_convbbb1x1deep_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
        asm volatile ("%0 = C15:14" : "=r"(cyc_after_kernel));
        uint32_t kernel_cyc = (uint32_t)(cyc_after_kernel - cyc_before_kernel);
        uint32_t desc_cyc   = (uint32_t)(cyc_before_kernel - cyc_after_table);
        uint32_t table_cyc  = (uint32_t)(cyc_after_table   - cyc_after_setup);
        uint32_t setup_cyc  = (uint32_t)(cyc_after_setup   - cyc_op_start);
        if (out_blocks[0]) {
            uint8_t *dst = (uint8_t *)out_blocks[0];
            for (int b = 0; b < 4; b++) dst[ 0+b] = (uint8_t)((kernel_cyc >> (8*b)) & 0xFF);
            for (int b = 0; b < 4; b++) dst[ 4+b] = (uint8_t)((desc_cyc   >> (8*b)) & 0xFF);
            for (int b = 0; b < 4; b++) dst[ 8+b] = (uint8_t)((table_cyc  >> (8*b)) & 0xFF);
            for (int b = 0; b < 4; b++) dst[12+b] = (uint8_t)((setup_cyc  >> (8*b)) & 0xFF);
        }
#elif defined(V9_PMU_PROBE)
        /* PMU probe: directly count COMMITTED_PKT_ANY (event 0x02) and
         * HVX_ACTIVE cycles (event 0xCC) around the kernel call. Compare
         * to chrometrace's reported pkts/cyc to verify metric semantics.
         *
         * Stash to out_buf[0..23] (6× uint32 LE):
         *   [0..3]   pkt_kernel       (committed packets DURING kernel call)
         *   [4..7]   cyc_kernel       (pcycle delta DURING kernel call)
         *   [8..11]  hvx_kernel       (HVX active cycles DURING kernel call)
         *   [12..15] pkt_op_total     (committed packets WHOLE op)
         *   [16..19] cyc_op_total     (pcycle delta WHOLE op)
         *   [20..23] hvx_op_total     (HVX active WHOLE op) */
        {
            /* Read all 4 counters before/after kernel call. */
            uint32_t any_b  = qurt_pmu_get(QURT_PMUCNT0);
            uint32_t t0_b   = qurt_pmu_get(QURT_PMUCNT1);
            uint32_t inst_b = qurt_pmu_get(QURT_PMUCNT2);
            uint32_t disp_b = qurt_pmu_get(QURT_PMUCNT3);
            uint64_t cyc_b;
            asm volatile ("%0 = C15:14" : "=r"(cyc_b));
            hmx_v73_convbbb1x1deep_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
            uint64_t cyc_a;
            asm volatile ("%0 = C15:14" : "=r"(cyc_a));
            uint32_t any_a  = qurt_pmu_get(QURT_PMUCNT0);
            uint32_t t0_a   = qurt_pmu_get(QURT_PMUCNT1);
            uint32_t inst_a = qurt_pmu_get(QURT_PMUCNT2);
            uint32_t disp_a = qurt_pmu_get(QURT_PMUCNT3);
            uint32_t any_kernel  = any_a  - any_b;
            uint32_t t0_kernel   = t0_a   - t0_b;
            uint32_t inst_kernel = inst_a - inst_b;
            uint32_t disp_kernel = disp_a - disp_b;
            uint32_t cyc_kernel  = (uint32_t)(cyc_a - cyc_b);
            uint32_t any_op   = any_a  - pkt_op_start;
            uint32_t t0_op    = t0_a;  /* assumes 0 at op start; adjust if needed */
            uint32_t cyc_op   = (uint32_t)(cyc_a - cyc_op_start_64);
            if (out_blocks[0]) {
                uint8_t *dst = (uint8_t *)out_blocks[0];
                #define STORE_LE(off, v) for (int b=0;b<4;b++) dst[(off)+b] = (uint8_t)(((v)>>(8*b)) & 0xFF)
                STORE_LE( 0, any_kernel);   /* PMU committed pkts ANY (all threads) */
                STORE_LE( 4, t0_kernel);    /* PMU committed pkts T0 (thread 0) */
                STORE_LE( 8, inst_kernel);  /* PMU committed instructions */
                STORE_LE(12, disp_kernel);  /* PMU dispatched packets */
                STORE_LE(16, cyc_kernel);   /* pcycle delta — kernel only */
                STORE_LE(20, any_op);       /* whole op ANY */
                STORE_LE(24, t0_op);        /* whole op T0 */
                STORE_LE(28, cyc_op);       /* whole op pcycle */
                #undef STORE_LE
            }
        }
#else
        hmx_v73_convbbb1x1deep_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#endif
#elif defined(V9_NATIVE_V73)
        /* hmx_v73_convbbb1x1_stride1 — `:deep:cm` activation MAC. 6th param =
         * extra[2] = { marker=1, cvt_arg=0 }. Bit-exact + native-level cyc/pkt
         * (3.06 cyc/pkt @ 512³). 1.17–2.27× over SINGLE_CALL. */
        const uint32_t extra_param[2] = { 1u, 0u };
        hmx_v73_convbbb1x1_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
#else
        hmx_convbbb1x1_stride1(&od, &ad, wt_pack, bias_bytes, md);
#endif
#else
        /* Per-(mt, nt) tile call (bit-exact baseline). */
        for (uint32_t mt = 0; mt < M_t; mt++) {
            const uint32_t rg = mt / mt_per_block;
            const uint32_t mt_in_block = mt % mt_per_block;

            /* Pre-bake act tile addresses for this mt across all K-tiles. */
            int32_t act_tbl[32 * 2] __attribute__((aligned(16)));
            for (uint32_t kt = 0; kt < K_t; kt++) {
                act_tbl[kt] = (int32_t)(uintptr_t)(
                    (const uint8_t *)act_blocks[rg * k_chunks + kt]
                    + mt_in_block * 1024);
            }

            /* Per-(mt, nt) tile call. Slow (~1.8K cyc dlsym overhead × M_t*N_t
             * calls). Use V9_NATIVE_SINGLE_CALL for production (1.4× speedup
             * over V9_KERNEL_HMX inline asm at 256³). This default branch is
             * kept as bit-exact reference. */
            for (uint32_t nt = 0; nt < N_t; nt++) {
                int32_t out_tbl[2] __attribute__((aligned(16)));
                out_tbl[0] = (int32_t)(uintptr_t)(
                    (uint8_t *)out_blocks[rg * (N / 32) + nt]
                    + mt_in_block * 1024);
                const uint8_t *wt_for_n = wt_pack + (nt * K_t) * 1024;
                const uint8_t *bias_n   = bias_bytes + nt * 256;
                /* Prefetch bias (256 B) + first wt tile (1 KB) for warm L2 */
                hmx_conv_out_desc_t od = { out_tbl, 1, 0, 1, 32, 32 };
                hmx_conv_act_desc_t ad = { act_tbl, K_t, 0 };
                hmx_convbbb1x1_stride1(&od, &ad, wt_for_n, bias_n, md);
            }
        }
#endif
    }
    return QHPI_Success;
#elif defined(V9_PARAMS_PROBE)
    /* Step 5.1 — characterize set_hmx_params_conv1x1 args.
     *
     * Calls dlsym'd set_hmx_params_conv1x1 with a sweep of arg combinations,
     * dumps each resulting 0x40-byte descriptor into the output Crouton_8
     * tensor at predictable row/col positions so the post-Untile row-major
     * output reveals each case at out[case_idx][0..127].
     *
     * Layout per case (128 bytes / row):
     *   bytes 0..3   : arg1
     *   bytes 4..7   : arg2
     *   bytes 8..11  : arg3
     *   bytes 12..15 : arg4
     *   bytes 16..19 : arg5
     *   bytes 20..23 : 0xDEADBEEF magic
     *   bytes 24..87 : 64-byte descriptor output from set_hmx_params_conv1x1
     *   bytes 88..127: zeros
     *
     * Crouton_8 output [1, M/32, 32, N] real layout (empirically reverse-
     * engineered via initial probe iteration). Each block holds 8 row-chunks
     * at row-stride 32 (not 64-contiguous as a naive HMX-tile-major model
     * would suggest):
     *   block_idx = ((m % 32) / 8) * out_n_chunks + (n / 32)
     *   inblk_off = (m / 32) * 256 + (m % 8) * 32 + (n % 32)
     * out_n_chunks = N / 32. Block 0 covers logical rows
     *   {0..7, 32..39, 64..71, 96..103, 128..135, 160..167, 192..199, 224..231}
     * × cols 0..31 (8 sub-chunks × 8 rows × 32 cols = 2048 B).
     *
     * Run with: EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_PARAMS_PROBE" */
    {
        void **out_blocks = qhpi_tensor_block_table(outputs[0]);
        if (!out_blocks) return QHPI_Success;

        uint32_t blocks = qhpi_tensor_block_table_length(inputs[0]);
        uint32_t S = 0;
        if (blocks == 4)         S = 32;
        else if (blocks == 8)    S = 64;
        else if (blocks == 16)   S = 128;
        else if (blocks == 32)   S = 256;
        else if (blocks == 128)  S = 512;
        else if (blocks == 512)  S = 1024;
        if (S == 0) return QHPI_Success;

        const uint32_t M = S, N = S;
        const uint32_t out_n_chunks = N / 32;

        /* Use actual block_table_length to bound writes safely. */
        const uint32_t total_out_blocks = qhpi_tensor_block_table_length(outputs[0]);
        const uint32_t bytes_per_block =
            total_out_blocks > 0 ? (uint32_t)(M * N) / total_out_blocks : 0;
        for (uint32_t bi = 0; bi < total_out_blocks; bi++) {
            uint8_t *blk = (uint8_t *)out_blocks[bi];
            if (blk) for (uint32_t k = 0; k < bytes_per_block; k++) blk[k] = 0;
        }

        /* Probe-arg sweep. The 5 args are well-aligned to mask_desc semantics
         * via internal bit-twiddling; we vary each parameter to characterize
         * its contribution to output bytes. */
        struct ProbeArgs {
            uint32_t r1, r2, r3, r4, r5;
        };
        const ProbeArgs cases[] = {
            /* Baseline + isolated bit sweeps */
            {0x000, 0x000, 0x000, 0x000, 0x000},   /*  0 — all zero */
            {0x400, 0x000, 0x000, 0x000, 0x000},   /*  1 — arg1 bit 10 (r2 = 32 path) */
            {0x600, 0x000, 0x000, 0x000, 0x000},   /*  2 — arg1 r2-field = 48 */
            {0x780, 0x000, 0x000, 0x000, 0x000},   /*  3 — arg1 r2-field = 60 */
            {0x7C0, 0x000, 0x000, 0x000, 0x000},   /*  4 — arg1 r2-field = 62 */
            /* arg2 sweep (likely K-related count) */
            {0x000, 0x008, 0x000, 0x000, 0x000},   /*  5 — arg2 = 8 */
            {0x000, 0x020, 0x000, 0x000, 0x000},   /*  6 — arg2 = 32 */
            {0x000, 0x100, 0x000, 0x000, 0x000},   /*  7 — arg2 = 256 (K total) */
            /* arg3 (stored at +0x00 << 5) */
            {0x000, 0x000, 0x001, 0x000, 0x000},   /*  8 — arg3 = 1 */
            {0x000, 0x000, 0x010, 0x000, 0x000},   /*  9 — arg3 = 16 */
            {0x000, 0x000, 0x020, 0x000, 0x000},   /* 10 — arg3 = 32 */
            /* arg4 (negated, low 3 bits via if-r5-bit5) */
            {0x000, 0x000, 0x000, 0x008, 0x020},   /* 11 — arg4=8, arg5 bit5 set */
            {0x000, 0x000, 0x000, 0x020, 0x020},   /* 12 — arg4=32 */
            /* arg5: r13 = (r5>>8)&0x1F (depth/spread) */
            {0x000, 0x000, 0x000, 0x000, 0x100},   /* 13 — arg5 r13-field = 1 */
            {0x000, 0x000, 0x000, 0x000, 0x200},   /* 14 — arg5 r13-field = 2 */
            {0x000, 0x000, 0x000, 0x000, 0x300},   /* 15 — arg5 r13-field = 3 (path A) */
        };
        const uint32_t n_cases = sizeof(cases) / sizeof(cases[0]);

        for (uint32_t i = 0; i < n_cases && i < 16; i++) {
            uint8_t desc[64] __attribute__((aligned(16)));
            for (uint32_t j = 0; j < 64; j++) desc[j] = 0xCD;
            set_hmx_params_conv1x1(desc, cases[i].r1, cases[i].r2,
                                   cases[i].r3, cases[i].r4, cases[i].r5);

            /* Pack header (24 B) + descriptor (64 B) + zero pad (40 B) = 128 B. */
            uint8_t row[128];
            for (uint32_t j = 0; j < 128; j++) row[j] = 0;
            *(uint32_t *)(row + 0)  = cases[i].r1;
            *(uint32_t *)(row + 4)  = cases[i].r2;
            *(uint32_t *)(row + 8)  = cases[i].r3;
            *(uint32_t *)(row + 12) = cases[i].r4;
            *(uint32_t *)(row + 16) = cases[i].r5;
            *(uint32_t *)(row + 20) = 0xDEADBEEF;
            for (uint32_t j = 0; j < 64; j++) row[24 + j] = desc[j];

            /* Scatter row[0..127] into Crouton_8 output blocks using the
             * empirically-correct layout. */
            const uint32_t m = i;
            for (uint32_t n = 0; n < 128 && n < N; n++) {
                uint32_t block_idx = ((m % 32) / 8) * out_n_chunks + (n / 32);
                uint32_t inblk_off = (m / 32) * 256 + (m % 8) * 32 + (n % 32);
                if (block_idx >= total_out_blocks) continue;
                uint8_t *blk = (uint8_t *)out_blocks[block_idx];
                if (blk) blk[inblk_off] = row[n];
            }
        }
    }
    return QHPI_Success;
#elif defined(V9_KERNEL_NOOP) || defined(V9_C8_ALIGNMENT_TEST)
    {
        /* Marker prologue so we can prove the kernel actually executed:
         *   [0]    = 0xA5 (start magic)
         *   [1]    = inputs[0] layout enum
         *   [2]    = output layout enum
         *   [3]    = num_inputs
         *   [4]    = output rank
         *   [5..8] = output dims[0..3] (low byte)
         *   [15]   = 0x5A (end magic) */
        uint8_t *out_buf = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);
        QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
        size_t bytes = 1;
        for (uint32_t i = 0; i < os.rank; i++) bytes *= os.dims[i];
        for (size_t i = 0; i < bytes; i++) out_buf[i] = 0;
        if (out_buf && bytes >= 16) {
            out_buf[0] = 0xA5;
            out_buf[1] = (uint8_t)qhpi_tensor_layout(inputs[0]);
            out_buf[2] = (uint8_t)qhpi_tensor_layout(outputs[0]);
            out_buf[3] = (uint8_t)num_inputs;
            out_buf[4] = (uint8_t)os.rank;
            for (uint32_t i = 0; i < 4 && i < os.rank; i++)
                out_buf[5 + i] = (uint8_t)(os.dims[i] & 0xFF);
            out_buf[15] = 0x5A;
        }
    }
    return QHPI_Success;
#else
    const uint8_t  *packed_act = (const uint8_t  *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t  *packed_wt  = (const uint8_t  *)qhpi_tensor_raw_data(inputs[1]);
    const uint16_t *bias_all   = (const uint16_t *)qhpi_tensor_raw_data(inputs[2]);
    uint8_t        *out        = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    /* Shapes — packed dims: [1, K_t, M_t, 1024] / [1, K_t, N_t, 1024].
     * Last dim 1024 = full HMX tile; M_t = M/32, N_t = N/32. */
    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);

    /* packed_act [1, K_t, M_t, 1024]; packed_wt [1, N_t, K_t, 1024] (V8 P2 format). */
    const uint32_t K_tiles = dim_at_v9(as, 1);
    const uint32_t M_tiles = dim_at_v9(as, 2);
    const uint32_t N_tiles = dim_at_v9(ws, 1);

#if defined(V9_INLINE_MINIMAL_HMX)
    /* Probe: minimal HMX inline asm (no dlsym). One MAC + drain per output tile.
     * Expects V8 weight format. */
    asm volatile("mxclracc" ::: "memory");
    for (uint32_t nt = 0; nt < N_tiles; nt++) {
        const uint8_t  *wt_for_n = packed_wt + nt * K_tiles * 1024;
        const uint16_t *bias_n   = bias_all  + nt * 128;
        asm volatile("bias = mxmem2(%0)" :: "r"(bias_n) : "memory");
        for (uint32_t mt = 0; mt < M_tiles; mt++) {
            uint8_t *out_tile = out + (mt * N_tiles + nt) * 1024;
            /* 1 MAC then sat.ub — proves HMX inputs are accessible. */
            const uint8_t *a0 = packed_act + (0 * M_tiles + mt) * 1024;
            asm volatile(
                "{ activation.ub = mxmem(%0, %1):cm\n"
                "  weight.b      = mxmem(%2, %3) }"
                :: "r"(a0), "r"((int32_t)HMX_RT_ACT_CM),
                   "r"(wt_for_n), "r"((int32_t)HMX_RT_WT)
                : "memory");
            asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                         :: "r"(out_tile), "r"((int32_t)HMX_RT_WT) : "memory");
        }
    }
    return QHPI_Success;
#endif
#if defined(V9_DUMP_HMX_PARAMS)
    /* Dump set_hmx_params_conv1x1 output for various arg combos.
     * Output layout: out[i*128 .. i*128+127] = {6×u32 args, 64×u8 desc, 36 pad}
     * where i indexes the arg combo (0..15).
     *
     * Args derived from descriptor builder static RE
     * (Agent/qnn_re/descriptor_builder_full.S):
     *   r1 = 0x700 (constant on the conv1x1 path)
     *   r2 = and(r20, 0x1c000) — runtime flags; we sweep typical values
     *   r3 = 0 (constant)
     *   r4 = sub(0, orig_arg3) — the wrapper's transformed arg
     *   r5 = and(r24, 7) — 3-bit thing from arg metadata
     *
     * For 256³ matmul: K=256, M=N=256. Likely candidates for r4 are
     * 32, 64, 128, 256 (K-related). r5 in {0, 1, 7}. */
    {
        for (size_t i = 0; i < M_tiles * N_tiles * 1024; i++) ((uint8_t*)out)[i] = 0;
        struct ProbeArgs { uint32_t r1, r2, r3, r4, r5; const char *tag; };
        const ProbeArgs cases[] = {
            {0x700, 0x0000, 0,   32, 0, "r2=0,r4=32,r5=0"},
            {0x700, 0x4000, 0,   32, 0, "r2=4000,r4=32,r5=0"},
            {0x700, 0x8000, 0,   32, 0, "r2=8000,r4=32,r5=0"},
            {0x700, 0xc000, 0,   32, 0, "r2=c000,r4=32,r5=0"},
            {0x700, 0x0000, 0,   64, 0, "r2=0,r4=64,r5=0"},
            {0x700, 0x0000, 0,  128, 0, "r2=0,r4=128,r5=0"},
            {0x700, 0x0000, 0,  256, 0, "r2=0,r4=256,r5=0"},
            {0x700, 0x0000, 0,(uint32_t)-32, 0, "r2=0,r4=-32,r5=0"},
            {0x700, 0x0000, 0,   32, 1, "r2=0,r4=32,r5=1"},
            {0x700, 0x0000, 0,   32, 7, "r2=0,r4=32,r5=7"},
            {0x700, 0x4004, 0,   32, 0, "r2=4004,r4=32,r5=0"},  /* r3 in 0x2004 mode */
            {0x700, 0x2004, 0,   32, 0, "r2=2004,r4=32,r5=0"},
            {0x000, 0x0000, 0,   32, 0, "all-zero"},
            {0x704, 0x0000, 0,   32, 0, "r1=704"},
            {0x707, 0x0000, 0,   32, 0, "r1=707"},
            {0x700, 0x0000, 1,   32, 0, "r3=1"},
        };
        const size_t n_cases = sizeof(cases) / sizeof(cases[0]);

        uint8_t *out_buf = (uint8_t *)out;
        for (size_t i = 0; i < n_cases; i++) {
            uint8_t desc[64] __attribute__((aligned(16)));
            for (size_t j = 0; j < 64; j++) desc[j] = 0xCD;  /* sentinel */
            set_hmx_params_conv1x1(desc, cases[i].r1, cases[i].r2,
                                   cases[i].r3, cases[i].r4, cases[i].r5);
            /* Pack: 24 bytes args header + 64 bytes desc + 40 pad = 128 bytes */
            uint32_t *hdr = (uint32_t *)(out_buf + i * 128);
            hdr[0] = cases[i].r1;
            hdr[1] = cases[i].r2;
            hdr[2] = cases[i].r3;
            hdr[3] = cases[i].r4;
            hdr[4] = cases[i].r5;
            hdr[5] = 0xDEADBEEF;
            for (size_t j = 0; j < 64; j++) {
                out_buf[i * 128 + 24 + j] = desc[j];
            }
        }
        return QHPI_Success;
    }
#endif
#if defined(V9_USE_DLSYM)
    /* Per-(mt, nt) call to hmx_convbbb1x1_stride1.
     * V9 packed_act [1, K_t, M_t, 1024]; V9 packed_wt [1, N_t, K_t, 1024]. */
#if defined(V9_DLSYM_FIRST_ONLY)
    /* Probe: only do (mt=0, nt=0) tile to isolate crash. Other tiles get zero. */
    for (uint32_t i = 0; i < M_tiles * N_tiles * 1024; i++) ((uint8_t*)out)[i] = 0;
    for (uint32_t nt = 0; nt < 1; nt++) {
        const uint8_t  *wt_for_n = packed_wt + nt * K_tiles * 1024;
        const uint16_t *bias_n   = bias_all  + nt * 128;
        for (uint32_t mt = 0; mt < 1; mt++) {
#else
    for (uint32_t nt = 0; nt < N_tiles; nt++) {
        const uint8_t  *wt_for_n = packed_wt + nt * K_tiles * 1024;
        const uint16_t *bias_n   = bias_all  + nt * 128;
        for (uint32_t mt = 0; mt < M_tiles; mt++) {
#endif
            uint8_t *out_tile = out + (mt * N_tiles + nt) * 1024;
            static int32_t out_tbl[1] __attribute__((aligned(64)));
            out_tbl[0] = (int32_t)(uintptr_t)out_tile;
            static int32_t act_tbl[64] __attribute__((aligned(64)));
            for (uint32_t k = 0; k < K_tiles; k++) {
                act_tbl[k] = (int32_t)(uintptr_t)(packed_act + (k * M_tiles + mt) * 1024);
            }
            hmx_conv_out_desc_t od = {
                out_tbl, 1, 0, 1, 64, 32,
            };
            hmx_conv_act_desc_t ad = {
                act_tbl, K_tiles, 0,
            };
            hmx_conv_mask_desc_t md = {
                0, HMX_RT_WT, 0, 2047, 0, 0, HMX_RT_WT,
            };
#if defined(V9_DLSYM_PROBE_ACT_AS_WT)
            hmx_convbbb1x1_stride1(&od, &ad, (const void*)packed_act, bias_n, &md);
#else
            hmx_convbbb1x1_stride1(&od, &ad, wt_for_n, bias_n, &md);
#endif
        }
    }
    return QHPI_Success;
#endif

    /* QNN-style loop order: N outer, M inner, K innermost (2-MAC unroll).
     * packed_act is K-major (Crouton from PackActCrouton): tile (kt, mt) at (kt*M_t+mt)*1024.
     * packed_wt is N-major (V8 PackWeightToHmxTileV3): tile (nt, kt) at (nt*K_t+kt)*1024.
     *
     * IMPORTANT: HMX bias register only holds for ~8 sat.ub events. If M_tiles > 8,
     * we MUST reload bias every mt iter (or every 8 mt iters) to avoid stale bias.
     * V8 production with M_TILE=256 had M_t=8 max, so this was hidden. */
    asm volatile("mxclracc" ::: "memory");
    for (uint32_t nt = 0; nt < N_tiles; nt++) {
        const uint8_t  *wt_for_n = packed_wt + (nt * K_tiles) * 1024;
        const uint16_t *bias_n   = bias_all + nt * 128;

        for (uint32_t mt = 0; mt < M_tiles; mt++) {
            uint8_t *out_tile = out + (mt * N_tiles + nt) * 1024;
            /* Reload bias each mt iter — cheap (1 mxmem) and safe for M_t > 8. */
            asm volatile("bias = mxmem(%0)" :: "r"(bias_n) : "memory");

            uint32_t kt = 0;
            for (; kt + 1 < K_tiles; kt += 2) {
                const uint8_t *a0 = packed_act + ((kt    ) * M_tiles + mt) * 1024;
                const uint8_t *a1 = packed_act + ((kt + 1) * M_tiles + mt) * 1024;
                const uint8_t *w0 = wt_for_n + (kt    ) * 1024;
                const uint8_t *w1 = wt_for_n + (kt + 1) * 1024;
                asm volatile(
                    "{ activation.ub = mxmem(%0, %1):cm\n"
                    "  weight.b      = mxmem(%2, %3) }\n"
                    "{ activation.ub = mxmem(%4, %1):cm\n"
                    "  weight.b      = mxmem(%5, %3) }"
                    :: "r"(a0), "r"(HMX_RT_ACT_CM),
                       "r"(w0), "r"(HMX_RT_WT),
                       "r"(a1), "r"(w1)
                    : "memory");
            }
            if (kt < K_tiles) {
                const uint8_t *a0 = packed_act + (kt * M_tiles + mt) * 1024;
                const uint8_t *w0 = wt_for_n + kt * 1024;
                asm volatile(
                    "{ activation.ub = mxmem(%0, %1):cm\n"
                    "  weight.b      = mxmem(%2, %3) }"
                    :: "r"(a0), "r"(HMX_RT_ACT_CM),
                       "r"(w0), "r"(HMX_RT_WT)
                    : "memory");
            }
            asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                         :: "r"(out_tile), "r"(0) : "memory");
        }
    }
    return QHPI_Success;
#endif
}

/* C8 alignment experiment 2026-04-27: declare act as Crouton_8 layout to
 * match native q::ConvLayer_s1.opt input format. ONNX shape changes from
 * [1, K_t, M_t, 1024] (raw byte tile-array, Flat4) to native logical shape
 * [1, M/32, 32, K] (Crouton_8 blocked). Goal: have QNN compiler auto-insert
 * q::ForceFormat_Crouton between DDR input and our op, eliminating our
 * hand-rolled PackActCrouton + tile-array byte layout.
 *
 * Weight: also flat [1, 1, K, N] u8 (matches native [1, 1, 256, 256] for 256³).
 * Bias: keep current u16 fp16-pair format for now (orthogonal change).
 * Scratch: removed (kernel reads/writes only act+wt+bias→out). */
/* C8 alignment 2026-04-27 Phase 2 try-3 — pre-packed weight + Crouton_8 act.
 * Layout for weight decoded by diffing native ctx-bin (see
 * Agent/qnn_re/weights_to_vtcm_RE_2026-04-27.md). gen_v8c8_test.py now writes
 * the wt_flat initializer in native ConvLayer's K-tile-major / 4-row-group
 * 1024-byte-tile layout. Weights_to_vtcm DMA should now linear-copy correct
 * pre-packed bytes into VTCM, and HMX kernel should read them directly. */
#if defined(V9_C8_ALIGNMENT_TEST)
/* C8-aligned 3-input sig — native ConvLayer alignment.
 *
 *   in[0] act:        Crouton_8 + Indirect + TCM_Only
 *                            → triggers q::ForceFormat_Crouton
 *   in[1] wt_packed:  Flat4 + Direct + TCM_Only
 *                     shape [1, K/32, N/32, 1024]  (native pre-pack)
 *                            → triggers q::ConvLayer.opt.weights_to_vtcm@FB.fB.
 *   in[2] bias:       Int32 + Flat4 + Direct + TCM_Only
 *                     shape [N]  (raw int32 STATIC, native shape)
 *                            → triggers q::ConvLayer.opt.weights_to_vtcm@Fi.fi.
 *
 * Output: u8 Crouton_8 + Indirect + TCM_Only [1, M/32, 32, N] (Step 3).
 *   HMX :after:cm:sat.ub writes 32×32 row-major u8 tiles. By organising
 *   the writes through the output block_table (1 KB per write at S=128,
 *   2 KB per Crouton block at S≥256), the resulting bytes ARE valid
 *   Crouton_8 tensor data. QNN compiler then auto-inserts
 *   q::ForceFormat_Flat (HVX in-place VTCM) + framework Output op
 *   (HVX bulk DDR copy) downstream — replacing our scalar
 *   UntileToRowMajor (441 K cyc at 256³ → ~80 K cyc native equivalent).
 *
 * NOTE: qhpi_tensor_shape() returns rank=0 for all auto-DMA'd tensors at
 * >=64³. Kernel derives M=K=N (square test) from act block_table_length. */
static QHPI_Tensor_Signature_v1 sig_inputs_v9[] = {
    {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
    {QHPI_Int32,  QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs_v9[] = {
    {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
};
static const uint32_t SIG_NUM_IN_V9 = 3;
#else
static QHPI_Tensor_Signature_v1 sig_inputs_v9[] = {
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs_v9[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static const uint32_t SIG_NUM_IN_V9 = 4;
#endif

/* Prepare-time hooks (Step 4 — bring custom op into QNN's scheduler).
 *
 * Without these, QNN treats BbbKMajor as opaque and serialises everything
 * around it (no overlap with q::ForceFormat_Crouton, weights_to_vtcm DMAs,
 * or the output side ForceFormat_Flat). At 256³ this leaves ~10 K cyc of
 * "custom-op tax" on the wire vs native ConvLayer_s1.opt.
 *
 * native ConvLayer_s1.opt provides cost_function + shape_required +
 * shape_legalized + tile_output (= 0). We mirror that surface so QNN
 * can pipeline the surrounding ops. */

#if defined(V9_C8_ALIGNMENT_TEST)
/* cost_function: rough cycle estimate. QNN uses this to decide
 * scheduling priority and overlap budgets. We approximate
 * cyc ≈ M_t × N_t × K_t × 16 (= ~16 cyc per HMX MAC packet, slightly
 * pessimistic vs measured 19 cyc/MAC at 256³ to give scheduler room).
 *
 * Inputs[0] is Crouton_8 act with rank-0 shape at preparation, but we
 * can read OutputDef shape via qhpi_op_output. Since we're called at
 * prepare time with QHPI_Tensor* inputs, we read tensor shape there. */
static float bbb_cost_function(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_inputs;
    if (!inputs || !inputs[0]) return 1.0f;
    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);
    /* act shape at prepare time: try [1, M/32, 32, K]; rank-3 fallback [1, M, K]. */
    uint32_t M = 256, K = 256;
    if (s.rank == 4) { M = s.dims[1] * s.dims[2]; K = s.dims[3]; }
    else if (s.rank == 3) { M = s.dims[1]; K = s.dims[2]; }
    /* Square test only for now: assume N == K. */
    uint32_t N = K;
    uint32_t M_t = (M + 31) / 32, N_t = (N + 31) / 32, K_t = (K + 31) / 32;
    float macs = (float)M_t * (float)N_t * (float)K_t;
#if defined(V9_COST_HIGH)
    return macs * 64.0f;   /* tell QNN we are 4× more expensive — encourages overlap */
#elif defined(V9_COST_LOW)
    return macs * 4.0f;    /* tell QNN we are cheap — encourages tight packing */
#else
    return macs * 16.0f;   /* baseline ~16 cyc/MAC packet */
#endif
}

/* shape_required: M and N output dims must be multiples of 32 (HMX tile).
 * QHPI_DO_NOT_TILE on K (we keep K whole; only M/N tile). */
static QHPI_Shape bbb_shape_required(const QHPI_Op *op)
{
    (void)op;
    QHPI_Shape req = {0};
    req.rank = 4;
    /* Output shape is [1, M/32, 32, N]; the tile constraints are on M_t (dim1)
     * and N (dim3). Set M_t = 1 (tile by 1 M-tile = 32 rows), N step = 32. */
    req.dims[0] = 1;
    req.dims[1] = 1;       /* tile dim1 = M_t in 1-tile units */
    req.dims[2] = 32;      /* row chunk size */
    req.dims[3] = 32;      /* tile N in 32-col units */
    return req;
}

/* shape_legalized: round proposed shape up to required multiples. */
static QHPI_Shape bbb_shape_legalized(const QHPI_Op *op, const QHPI_Shape *proposed)
{
    (void)op;
    QHPI_Shape s = *proposed;
    if (s.rank >= 4) {
        if (s.dims[1] < 1) s.dims[1] = 1;
        if (s.dims[2] < 32) s.dims[2] = 32;
        if (s.dims[3] < 32) s.dims[3] = 32;
        /* Round dim3 (N) up to multiple of 32. */
        if (s.dims[3] % 32) s.dims[3] = ((s.dims[3] + 31) / 32) * 32;
    }
    return s;
}
#endif

static QHPI_Kernel_v1 sg_kernels_v9[] = {
    {
        THIS_PKG_NAME_STR "::hmx_matmul_v9",
        hmx_matmul_v9_kernel,
        QHPI_RESOURCE_HMX,
        false,
#if defined(V9_BBB_MULTITHREAD)
        true,                     /* multithreaded probe */
#else
        false,
#endif
        false, false,
        SIG_NUM_IN_V9, sig_inputs_v9,
        1, sig_outputs_v9,
#if defined(V9_C8_ALIGNMENT_TEST)
        bbb_cost_function,
#else
        nullptr,
#endif
        0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v9[] = {
    {
        THIS_PKG_NAME_STR "::BbbKMajor",
        1, sg_kernels_v9,
#if defined(V9_C8_ALIGNMENT_TEST)
        nullptr,                  /* early_rewrite */
        bbb_shape_required,
        bbb_shape_legalized,
        0,                        /* tile_output: tile output[0] */
        nullptr,                  /* build_tile (let QNN's default tile work) */
        nullptr,                  /* late_rewrite */
#else
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
#endif
    },
};

extern "C" void register_hmx_matmul_v9_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops_v9) / sizeof(sg_ops_v9[0]), sg_ops_v9, THIS_PKG_NAME_STR);
}
