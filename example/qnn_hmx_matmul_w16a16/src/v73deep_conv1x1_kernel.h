/*
 * v73deep_conv1x1_kernel.h -- embedded native HMX Conv1x1-family body.
 *
 * This package keeps the same C ABI boundary as the validated u8i8 path:
 *   r0 = od ptr, r1 = ad ptr, r2 = wt ptr, r3 = bias ptr,
 *   r4 = mask desc ptr, r5 = extra_param ptr.
 *
 * The included .inc file is a hybrid readable asm/word replica of the native
 * W16A16 slice from libQnnHtpV75Skel.so. Byte-proven packets are written as
 * inline asm; unknown HMX or branch-sensitive packets remain raw words.
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
