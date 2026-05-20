#ifndef HANDWRITTEN_HMX_U8I8_KERNEL_H_
#define HANDWRITTEN_HMX_U8I8_KERNEL_H_

#include <stdint.h>

#include "handwritten_hmx_kernel_abi.h"

typedef struct HmU8I8OutDesc {
  int32_t *out_tile_ptr_table;
  uint32_t out_table_stride_dwords;
  uint32_t out_y_stride_words;
  uint32_t n_tiles_pow2;
  int32_t m_total_minus_step;
  uint32_t k_total_bytes;
} HmU8I8OutDesc;

typedef struct HmU8I8ActDesc {
  int32_t *act_ptr_pairs;
  uint32_t n_act_pairs;
  uint32_t act_table_y_stride_words;
} HmU8I8ActDesc;

typedef struct HmU8I8MaskDesc {
  int32_t out_check;
  uint32_t out_rt_mask;
  int32_t act_check;
  uint32_t act_rt_base;
  uint32_t filter_x_stride;
  uint32_t pad14;
  uint32_t alt_rt;
} HmU8I8MaskDesc;

HM_ASSERT_CONV1X1_ABI(HmU8I8OutDesc, HmU8I8ActDesc, HmU8I8MaskDesc);

#if defined(__hexagon__)

__attribute__((naked, aligned(64), noinline))
static void hm_u8i8_v73deep_kernel(
    const HmU8I8OutDesc *out_desc,
    const HmU8I8ActDesc *act_desc,
    const uint8_t *packed_weight,
    const uint8_t *folded_bias,
    const HmU8I8MaskDesc *mask_desc,
    const uint32_t *extra_param) {
  __asm__ volatile(
#include "../kernels/u8i8/v73deep_conv1x1_kernel.inc"
  );
}

#endif  // defined(__hexagon__)

#endif  // HANDWRITTEN_HMX_U8I8_KERNEL_H_
