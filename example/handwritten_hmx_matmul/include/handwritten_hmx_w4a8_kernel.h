#ifndef HANDWRITTEN_HMX_W4A8_KERNEL_H_
#define HANDWRITTEN_HMX_W4A8_KERNEL_H_

#include <stdint.h>

#include "handwritten_hmx_kernel_abi.h"

typedef struct HmW4A8OutDesc {
  int32_t *out_tile_ptr_table;
  uint32_t out_table_stride_dwords;
  uint32_t out_y_stride_words;
  uint32_t n_tiles_pow2;
  int32_t m_total_minus_step;
  uint32_t k_total_bytes;
} HmW4A8OutDesc;

typedef struct HmW4A8ActDesc {
  int32_t *act_ptr_pairs;
  uint32_t n_act_pairs;
  uint32_t act_table_y_stride_words;
} HmW4A8ActDesc;

typedef struct HmW4A8MaskDesc {
  int32_t out_check;
  uint32_t out_rt_mask;
  int32_t act_check;
  uint32_t act_rt_base;
  uint32_t filter_x_stride;
  uint32_t pad14;
  uint32_t alt_rt;
} HmW4A8MaskDesc;

HM_ASSERT_CONV1X1_ABI(HmW4A8OutDesc, HmW4A8ActDesc, HmW4A8MaskDesc);

#if defined(__hexagon__)

__attribute__((naked, aligned(64), noinline))
static void hm_w4a8_v73deep_kernel(
    const HmW4A8OutDesc *out_desc,
    const HmW4A8ActDesc *act_desc,
    const uint8_t *packed_weight,
    const uint8_t *folded_bias,
    const HmW4A8MaskDesc *mask_desc,
    const uint32_t *extra_param) {
  __asm__ volatile(
#include "../kernels/w4a8/v73deep_conv1x1_kernel.inc"
  );
}

#endif  // defined(__hexagon__)

#endif  // HANDWRITTEN_HMX_W4A8_KERNEL_H_
