/*
 * v73deep_conv1x1_kernel.h -- owned V73DEEP Conv1x1 kernel replica from
 * libQnnHtpV75Skel.so
 *  (VMA 0x2ebe40, 1132 bytes).
 *
 * Phase B step 1: own-the-bytes. Runtime no longer dlsym\'s the QNN binary; we
 * jump into our embedded copy. Code is position-independent (PC-relative branches
 * only, no calls out, no GOT/PLT references) so embedding works.
 *
 * Phase B step 2 (completed): the full 1132-byte kernel body
 *   prologue (P1..P11) → main K-MAC body (M1..M18) → final-reduce
 *   (F1..F11) → epilogue (E1..E4), plus Alt-A/B/C dispatch arms
 * is now hand-written Hexagon inline asm (see v73deep_conv1x1_kernel.inc).
 * Each packet conversion was byte-verified against the native 1132 bytes via
 * /tmp/v73deep_fullasm/verify.sh.
 *
 * To regenerate the .byte fallback (sanity check): extract via
 *   dd if=tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
 *      of=/tmp/v73deep_kernel.bin bs=1 skip=3063360 count=1132
 * then run scripts/extract_v73deep_bytes.py.
 *
 * ABI matches :
 *   r0 = od ptr, r1 = ad ptr, r2 = wt ptr, r3 = bias ptr,
 *   r4 = mask desc ptr, r5 = extra_param ptr.
 */

#ifndef V73DEEP_CONV1X1_KERNEL_H
#define V73DEEP_CONV1X1_KERNEL_H

#include <stdint.h>

struct hmx_conv_out_desc_t;
struct hmx_conv_act_desc_t;
struct hmx_conv_mask_desc_t;

#if defined(__hexagon__)

__attribute__((naked, aligned(64), noinline))
static void our_v73deep_kernel(
    const struct hmx_conv_out_desc_t  *od,
    const struct hmx_conv_act_desc_t  *ad,
    const uint8_t                     *wt,
    const uint8_t                     *bias,
    const struct hmx_conv_mask_desc_t *mask,
    const uint32_t                    *extra_param)
{
    __asm__ volatile (
#include "v73deep_conv1x1_kernel.inc"
    );
}

#endif /* __hexagon__ */

#endif /* V73DEEP_CONV1X1_KERNEL_H */
