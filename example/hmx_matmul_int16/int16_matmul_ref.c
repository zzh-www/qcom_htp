/*
 * int16_matmul_ref.c — plain-C references.
 *
 *   im_matmul_ref                : oracle (exact int16 in int64)
 *   im_matmul_ref_top_only       : mirrors the HMX milestone-1 top-byte
 *                                  kernel, must match HMX output bit-exactly
 *   im_matmul_ref_f16_path       : kept so older tests still build; computes
 *                                  the f16 experimental path
 */

#include "int16_matmul.h"
#include <stdint.h>

static inline int16_t sat_i16_shift(int64_t prod, int shift)
{
    int64_t half = (shift == 0) ? 0 : ((int64_t)1 << (shift - 1));
    int64_t v    = (prod + half) >> shift;
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

void im_matmul_ref(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq)
{
    for (int i = 0; i < IM_TILE_M; i++) {
        for (int j = 0; j < IM_TILE_N; j++) {
            int64_t acc = 0;
            for (int k = 0; k < IM_TILE_K; k++) {
                acc += (int64_t)a[i * IM_TILE_K + k]
                     * (int64_t)w[k * IM_TILE_N + j];
            }
            out[i * IM_TILE_N + j] =
                sat_i16_shift(acc * (int64_t)rq.mul, rq.shift);
        }
    }
}

/*
 * Top-byte-only reference (must match HMX milestone-1 kernel bit-exactly).
 *   A_h = (uint8)((a + 32768) >> 8)
 *   W_h = (int8) (w >> 8)
 *   base[i][j]       = sum_k A_h[i][k] * W_h[k][j]          (u8 * i8)
 *   base_wrapped[i][j] = (int16_t)(uint16_t)base           <-- mimic uint16 round-trip
 *   top[i][j]        = base_wrapped - 128 * ColSum(W_h)
 *   int32_sum        = top * 65536
 *   out              = sat_i16(int32_sum * mul >> shift)
 */
void im_matmul_ref_top_only(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    int                     input_shift)
{
    (void)input_shift;   /* unused — kept for API compat */

    int32_t col_sum_wh[IM_TILE_N];
    for (int j = 0; j < IM_TILE_N; j++) {
        int32_t s = 0;
        for (int k = 0; k < IM_TILE_K; k++)
            s += (int32_t)(int8_t)(w[k * IM_TILE_N + j] >> 8);
        col_sum_wh[j] = s;
    }

    for (int i = 0; i < IM_TILE_M; i++) {
        for (int j = 0; j < IM_TILE_N; j++) {
            int32_t base = 0;
            for (int k = 0; k < IM_TILE_K; k++) {
                uint8_t ah = (uint8_t)(((int32_t)a[i * IM_TILE_K + k] + 32768) >> 8);
                int8_t  wh = (int8_t) (w[k * IM_TILE_N + j] >> 8);
                base += (int32_t)ah * (int32_t)wh;
            }
            /* Mirror the uint16 output wraparound: low 16 bits,
             * reinterpret as signed int16.  No extra 257/256 scale when
             * the activation tile is properly packed (only valid byte
             * positions touched; ignored positions zeroed). */
            int16_t wrapped = (int16_t)(uint16_t)(base & 0xFFFF);

            int32_t top = (int32_t)wrapped - 128 * col_sum_wh[j];
            int64_t int32_sum = (int64_t)top * 65536;
            out[i * IM_TILE_N + j] =
                sat_i16_shift(int32_sum * (int64_t)rq.mul, rq.shift);
        }
    }
}

/* Kept as a thin alias so the test harness keeps building without changes. */
void im_matmul_ref_f16_path(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    int                     input_shift)
{
    im_matmul_ref_top_only(out, a, w, rq, input_shift);
}
