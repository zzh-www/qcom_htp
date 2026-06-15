/* w16a16_mm.h — w16a16 matmul PRIMITIVE on the M=256 carrier (device-side, VTCM).
 *
 * Shape M=256 x K=64 x N=64: the device-verified byte-exact shape (verified == QNN native via the
 * custom op qnn_hmx_matmul_w16a16, accepted 65536/65536). 256 act rows = 4 independent
 * 64-row blocks sharing one 64x64 weight = Phase-4 fan-out batching (amortizes per-op setup).
 * This carrier's act/out crouton = `pack_act_crouton16(.,256,.)` = 8 row4-groups x 32-row m32
 * blocks; descriptor out_y=256/m_total=1/n_tiles=256/act_y=128.
 *
 * ⚠️ STALE-CLAIM CORRECTED (cron#66, 2026-06-14): the old comment "M<256 / single-64^3 descriptors
 * never byte-exact" is FALSE. A single M=64 64^3 IS bit-exact to QNN native `ConvLayer_s1`
 * (max|d|=0) using native's M=64 descriptor (out_y=4/m_total=8/n_tiles=8/act_y=4, 4 tiles @mod4) +
 * the CLOSED-FORM `crouton_pos(r,c)` bit-permutation layout + native control 0x804035F3/0x4000023E.
 * See `Agent/current/pure_hmx_solve_build.md` 末节 + `project_dense_n8_matmul_bitexact_2026-06-14`.
 *
 * PER-CALL cyc (口径④, resident, back-to-back, this device): M=256 carrier n_tiles=256 = 42333
 * (=10583/64^3-equiv); cron#42 n_tiles=32 trim = 5547 (=1387/64^3-equiv, = 8 tile/block x4 = already
 * the dense floor via batching); single M=64 dense n_tiles=8 = 1576/64^3. ⇒ the carrier's n_tiles=32
 * already achieves the per-matmul floor; M=64 dense is the bit-exact-to-native reference + single-block
 * (no 4-way batch) option, NOT a per-matmul speedup over the batched carrier.
 *
 * Pipeline: C packers (w16a16_pack.h, byte-exact) -> our_v73deep_kernel_i16 (sha256 == device-
 * byte-exact custom op) -> crouton16 row4 deblock -> linear u16.
 *
 * Quant contract (standalone's): act u16 zp 32768, wt q16 clipped ±32639, out u16 zp 32768,
 * gain = product/32767 (constant 2-pow drain, extra={1,1536}, control 0x00404420 = 1/32767 drain;
 * native mm1ex uses 0x804035F3/0x4000023E = sA*sB/sC drain). Scale tracking SOFTWARE (caller owns sA/sW/sY).
 */
#ifndef W16A16_MM_H
#define W16A16_MM_H
#include <stdint.h>
#include "w16a16_pack.h"
#include "../baremetal/inc/v73deep_conv1x1_kernel_i16.h"

#define W16MM_M 256
#define W16MM_K 64
#define W16MM_N 64
#define W16MM_ACT_BYTES   (W16MM_M * W16MM_K * 2)   /* 32KB crouton16 act */
#define W16MM_WT_BYTES    (W16MM_K * W16MM_N * 2)   /* 8KB 4-pass stream */
#define W16MM_BIAS_BYTES  ((W16MM_N / 16) * 256)    /* 1KB */
#define W16MM_OUT_BYTES   (W16MM_M * W16MM_N * 2)   /* 32KB crouton16 out */
#define W16A16_MM_VTCM_BYTES 0x16000u               /* 88KB total incl tables */

typedef struct {
    uint8_t  *act, *wt, *out;
    int32_t  *bias, *atab, *otab;
    uint32_t *ep, *mb;
    void     *od, *ad;
} w16a16_mm_t;

/* q16 weights must be clipped to ±32639 (=127*256+127; native saturates the hi-byte there). */
static inline int16_t w16a16_clip_q16(int v) { return (int16_t)(v > 32639 ? 32639 : (v < -32639 ? -32639 : v)); }

#if defined(__hexagon__)
/* init: carve VTCM + fill the proven M=256,K=64,N=64 single-full-N descriptors (op formula:
 * out {tbl, N_t, M, M, 1, N}, act {tbl, K_t, K_t*64}; tables rg 0..63 x {kt|nt}). */
static void w16a16_mm_init(w16a16_mm_t *b, uint8_t *vtcm, void *desc_mem) {
    b->act  = vtcm;                                /* 0x0..0x18000 (96K: up to K=192 stack) */
    b->wt   = vtcm + 0x18000;                      /* 24K */
    b->bias = (int32_t *)(vtcm + 0x1E000);
    b->atab = (int32_t *)(vtcm + 0x1E800);         /* up to 384 entries (K-stack d=3) */
    b->otab = (int32_t *)(vtcm + 0x1F400);
    b->ep   = (uint32_t *)(vtcm + 0x1F800);
    b->mb   = (uint32_t *)(vtcm + 0x1F840);
    b->out  = vtcm + 0x20000;                      /* 32K */
    for (int rg = 0; rg < 64; ++rg) for (int t = 0; t < 2; ++t) {
        b->atab[rg * 2 + t] = (int32_t)(uintptr_t)(b->act + (((rg & 7) * 2 + t) * 2048));
        b->otab[rg * 2 + t] = (int32_t)(uintptr_t)(b->out + (((rg & 7) * 2 + t) * 2048));
    }
    static const uint32_t MASK[16] = { 0x0u,0x700u,0x0u,0x77cu,0x0u,0x0u,0x3ffu,0x0u,
                                       0x0u,0x0u,0x0u,0x0u,0x80u,0x0u,0x0u,0x0u };
    b->ep[0] = 1u; b->ep[1] = 1536u;
    for (int i = 0; i < 16; ++i) b->mb[i] = MASK[i];
    b->mb[14] = (uint32_t)(uintptr_t)b->ep;
    hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)desc_mem;
    hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)((uint8_t *)desc_mem + 64);
    od->out_tile_ptr_table = b->otab; od->out_table_stride_dwords = 2u; od->out_y_stride_words = 256u;
    /* DEFAULT n_tiles = the EXACT minimum for this shape (cron#67): ceil(M/32)*ceil(N/32)*byte_pass,
     * byte_pass=2 for w16a16. M=256,N=64 -> 8*2*2 = 32. (Was hardcoded 256 = 8x over-walk; trimming to
     * the minimum is byte-identical output + ~4x faster per-call. See docs/w16a16_kernel_mechanism.md §5.) */
    od->n_tiles_pow2 = (uint32_t)(((W16MM_M + 31) / 32) * ((W16MM_N + 31) / 32) * 2);
    od->m_total_minus_step = 1; od->k_total_bytes = 64u;
    ad->act_ptr_pairs = b->atab; ad->n_act_pairs = 2u; ad->act_table_y_stride_words = 128u;
    b->od = od; b->ad = ad;
}

/* full primitive: linear A(u16 256x64) x W(q16 64x64, pre-clipped) -> linear Y(u16 256x64). */
static void w16a16_mm(w16a16_mm_t *b, const uint16_t *A, const int16_t *W, uint16_t *Y) {
    w16a16_pack_act_crouton16(A, (uint16_t *)b->act, W16MM_M, W16MM_K);
    w16a16_pack_wt_kmajor(W, b->wt, W16MM_K, W16MM_N);
    w16a16_pack_bias(W, b->bias, W16MM_K, W16MM_N);
    our_v73deep_kernel_i16((const hmx_conv_out_desc_t *)b->od, (const hmx_conv_act_desc_t *)b->ad,
                           b->wt, (const uint8_t *)b->bias, (const hmx_conv_mask_desc_t *)b->mb, b->ep);
    w16a16_depack_crouton16((const uint16_t *)b->out, Y, W16MM_M, W16MM_N);
}

#ifndef GP_NO_LEANMM
/* Stage B (cron#83): lean from-scratch 8-tile MAC consumer is the PRODUCTION DEFAULT (cron#83); build
 * with -DGP_NO_LEANMM to escape to the native our_v73deep_kernel_i16. Production byte-identical. Replaces
 * our_v73deep_kernel_i16 at ALL 3 call sites via this single hot-path wrapper. */
#include "lean_mm64.h"
#endif

/* kernel-only (operands already packed) — Phase-4 hot path. */
static inline void w16a16_mm_run(w16a16_mm_t *b) {
#ifndef GP_NO_LEANMM
    lean_mm64(b);          /* lean is the production default; non-nt8 shapes runtime-fallback to native inside lean_mm64 */
#else
    our_v73deep_kernel_i16((const hmx_conv_out_desc_t *)b->od, (const hmx_conv_act_desc_t *)b->ad,
                           b->wt, (const uint8_t *)b->bias, (const hmx_conv_mask_desc_t *)b->mb, b->ep);
#endif
}
#endif /* __hexagon__ */

#endif /* W16A16_MM_H */
