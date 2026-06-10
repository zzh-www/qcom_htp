/*
 * v73deep_conv1x1_kernel_i16.h -- byte-proven W16A16 (int16xint16->uint16) HMX body.
 *
 * The included .inc is a VERBATIM copy (sha256-identical) of the device-byte-exact
 * custom-op kernel example/qnn_hmx_matmul_w16a16/src/v73deep_conv1x1_kernel.inc.
 * That op is byte-exact on real v75 (accepted_256: native-exact 65536/65536 maxdiff=0),
 * so this asm body is byte-verified on device. Renamed our_v73deep_kernel_i16 to coexist
 * with the int8 our_v73deep_kernel (same hmx_conv_*_desc_t ABI, r0..r5 register contract).
 */

#ifndef V73DEEP_CONV1X1_KERNEL_I16_H
#define V73DEEP_CONV1X1_KERNEL_I16_H

#include <stdint.h>

struct hmx_conv_out_desc_t;
struct hmx_conv_act_desc_t;
struct hmx_conv_mask_desc_t;

#if defined(__hexagon__)

__attribute__((naked, aligned(64), noinline))
static void our_v73deep_kernel_i16(
    const struct hmx_conv_out_desc_t  *od,        /* r0 */
    const struct hmx_conv_act_desc_t  *ad,        /* r1 */
    const uint8_t                     *wt,        /* r2  (4-pass hi/lo k-major stream) */
    const uint8_t                     *bias,      /* r3  (int16 native folded-bias record) */
    const struct hmx_conv_mask_desc_t *mask,      /* r4  (mask words from ARG1=0x70b,ARG5=0x80) */
    const uint32_t                    *extra_param) /* r5  ({1,1536}) */
{
    __asm__ volatile (
#include "v73deep_conv1x1_kernel_i16.inc"
    );
}

#endif /* __hexagon__ */

#endif /* V73DEEP_CONV1X1_KERNEL_I16_H */
