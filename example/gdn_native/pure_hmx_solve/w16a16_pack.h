/* w16a16_pack.h — REAL C packers for the w16a16 HMX matmul (crouton16 act, dilated-kmajor int16 wt,
 * native bias record). Portable scalar C (host-testable). Byte-identical to the proven Python packers
 * (example/handwritten_hmx_matmul/prepare_owned_inputs.py:pack_a16_crouton16_row4_surface,
 *  scripts/generate_w16a16_weight_sidecar.py:generate_sidecar/generate_bias_sidecar).
 * These are the device-side packers the pure-HMX solve needs (the standalone packs in Python offline). */
#ifndef W16A16_PACK_H
#define W16A16_PACK_H
#include <stdint.h>
#include <stddef.h>

/* activation: MxK uint16 -> Crouton16 row4 surface (pairs two adjacent u16 rows per 32-bit lane). */
static void w16a16_pack_act_crouton16(const uint16_t *A, uint16_t *out, int M, int K) {
    int o = 0;
    for (int row4 = 0; row4 < 8; ++row4)
        for (int kt = 0; kt < K / 32; ++kt) {
            int kb = kt * 32;
            for (int m32 = 0; m32 < M / 32; ++m32)
                for (int rp = 0; rp < 2; ++rp) {
                    int row0 = m32 * 32 + row4 * 4 + rp * 2, row1 = row0 + 1;
                    for (int col = kb; col < kb + 32; ++col) {
                        out[o++] = A[row0 * K + col];
                        out[o++] = A[row1 * K + col];
                    }
                }
        }
}

/* weight: KxN int16 codes (q16) -> dilated low/high byte 4-pass k-major stream.
 * Per 8 q16: low[0:4], rounded_high[0:4], low[4:8], rounded_high[4:8]; rounded_high=(q16+128)>>8. */
static void w16a16_pack_wt_kmajor(const int16_t *q16, uint8_t *out, int K, int N) {
    int o = 0;
    for (int nbase = 0; nbase < N; nbase += 128) {
        int tiles = (N - nbase) / 32; if (tiles > 4) tiles = 4;
        for (int nt = 0; nt < tiles; ++nt)
            for (int half = 0; half < 2; ++half)
                for (int kt = 0; kt < K / 32; ++kt)
                    for (int grp = 0; grp < 8; ++grp) {
                        int vals[64], vi = 0;
                        for (int lane = 0; lane < 4; ++lane) {
                            int ch = grp * 8 + half * 4 + lane;
                            for (int off = ch * 16; off < (ch + 1) * 16; ++off) {
                                int rgrp = off / 128, rem = off % 128;
                                int col = rem / 4, rmod = rem % 4;
                                int row = rgrp * 4 + rmod;
                                vals[vi++] = (int)q16[(kt * 32 + row) * N + (nbase + nt * 32 + col)];
                            }
                        }
                        for (int idx = 0; idx < vi; idx += 8) {
                            for (int j = 0; j < 4; ++j) out[o++] = (uint8_t)(vals[idx + j] & 0xff);
                            for (int j = 0; j < 4; ++j) out[o++] = (uint8_t)(((vals[idx + j] + 128) >> 8) & 0xff);
                            for (int j = 4; j < 8; ++j) out[o++] = (uint8_t)(vals[idx + j] & 0xff);
                            for (int j = 4; j < 8; ++j) out[o++] = (uint8_t)(((vals[idx + j] + 128) >> 8) & 0xff);
                        }
                    }
    }
}

/* bias record: per 16-N-col group, 64 int32 = [control(32)] + [eff,0]*16, eff = (-colsum(q16))/2. */
static void w16a16_pack_bias(const int16_t *q16, int32_t *out, int K, int N) {
    static const int32_t control[32] = {
        0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000,
        0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000,
        0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000,
        0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000,0x00404420,0x40000000 };
    int o = 0;
    for (int nbase = 0; nbase < N; nbase += 128) {
        int present = N - nbase; if (present > 128) present = 128;
        int ngroups = present / 16;
        for (int g = 0; g < ngroups; ++g) {
            for (int i = 0; i < 32; ++i) out[o + i] = control[i];
            for (int idx = 0; idx < 16; ++idx) {
                long colsum = 0; int n = nbase + g * 16 + idx;
                for (int k = 0; k < K; ++k) colsum += (int)q16[k * N + n];
                /* python: ((-colsum)//2) with floor division */
                long v = -colsum; long eff = (v >= 0) ? (v / 2) : -(((-v) + 1) / 2);  /* floor div by 2 */
                out[o + 32 + idx * 2] = (int32_t)eff;
                out[o + 32 + idx * 2 + 1] = 0;
            }
            o += 64;
        }
    }
}

/* output deblock: Crouton16 row4 surface (MxN u16) -> linear MxN u16 (proven inverse of the act packer's
 * row4 layout; same one the byte-exact standalone uses to recover the QNN-native Y).
 * Block size scales with M: (M/32) pair-slab pairs x 128B = M*4 bytes (2048B at M=256, 512B at M=64). */
static void w16a16_depack_crouton16(const uint16_t *src, uint16_t *dst, int M, int N) {
    int blk_u16 = (M / 32) * 128;
    for (int row4 = 0; row4 < 8; ++row4)
        for (int nt = 0; nt < N / 32; ++nt) {
            const uint16_t *block = src + (size_t)(row4 * (N / 32) + nt) * blk_u16;
            for (int m32 = 0; m32 < M / 32; ++m32)
                for (int rp = 0; rp < 2; ++rp) {
                    int row0 = m32 * 32 + row4 * 4 + rp * 2, row1 = row0 + 1;
                    const uint16_t *sp = block + (size_t)(m32 * 2 + rp) * 64;     /* 128 B = 64 u16 / pair */
                    for (int col = 0; col < 32; ++col) {
                        dst[row0 * N + nt * 32 + col] = sp[col * 2];
                        dst[row1 * N + nt * 32 + col] = sp[col * 2 + 1];
                    }
                }
        }
}

#endif /* W16A16_PACK_H */
