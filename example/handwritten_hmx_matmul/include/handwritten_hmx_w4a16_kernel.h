#ifndef HANDWRITTEN_HMX_W4A16_KERNEL_H_
#define HANDWRITTEN_HMX_W4A16_KERNEL_H_

#include <stdint.h>

#include "handwritten_hmx_kernel_abi.h"

typedef struct HmW4A16OutDesc {
  int32_t *out_tile_ptr_table;
  uint32_t out_table_stride_dwords;
  uint32_t out_y_stride_words;
  uint32_t n_tiles_pow2;
  int32_t m_total_minus_step;
  uint32_t k_total_bytes;
} HmW4A16OutDesc;

typedef struct HmW4A16ActDesc {
  int32_t *act_ptr_pairs;
  uint32_t n_act_pairs;
  uint32_t act_table_y_stride_words;
} HmW4A16ActDesc;

typedef struct HmW4A16MaskDesc {
  int32_t out_check;
  uint32_t out_rt_mask;
  int32_t act_check;
  uint32_t act_rt_base;
  uint32_t filter_x_stride;
  uint32_t pad14;
  uint32_t alt_rt;
} HmW4A16MaskDesc;

HM_ASSERT_CONV1X1_ABI(HmW4A16OutDesc, HmW4A16ActDesc, HmW4A16MaskDesc);

#if defined(__hexagon__)

__attribute__((naked, aligned(64), noinline))
static void hm_w4a16_v73deep_kernel(
    const HmW4A16OutDesc *out_desc,
    const HmW4A16ActDesc *act_desc,
    const uint8_t *packed_weight,
    const uint8_t *folded_bias,
    const HmW4A16MaskDesc *mask_desc,
    const uint32_t *extra_param) {
  __asm__ volatile(
#include "../kernels/w4a16/v73deep_conv1x1_kernel.inc"
  );
}

__attribute__((naked, aligned(64), noinline))
static void hm_w4a16_v73wrapper_entry_kernel(
    const HmW4A16OutDesc *out_desc,
    const HmW4A16ActDesc *act_desc,
    const uint8_t *packed_weight,
    const uint8_t *folded_bias,
    const HmW4A16MaskDesc *mask_desc,
    const uint32_t *extra_param) {
  __asm__ volatile(
      "{ r8 = memw(r4+#0x30) }\n"
      "{ p1 = tstbit(r8,#0x5)\n"
      "  if (p1.new) jump:t 1f\n"
      "  r7:6 = memd(r4+#0x8)\n"
      "  r11:10 = memd(r4+#0x0) }\n"
      "{ jump 1f }\n"
      "1:\n"
#include "../kernels/w4a16/v73deep_conv1x1_kernel.inc"
  );
}

__attribute__((naked, aligned(64), noinline))
static void hm_w4a16_v73wrapper_nondeep_kernel(
    const HmW4A16OutDesc *out_desc,
    const HmW4A16ActDesc *act_desc,
    const uint8_t *packed_weight,
    const uint8_t *folded_bias,
    const HmW4A16MaskDesc *mask_desc,
    const uint32_t *extra_param) {
  __asm__ volatile(
#include "../kernels/w4a16/v73wrapper_nondeep_conv1x1_kernel.inc"
  );
}

#endif  // defined(__hexagon__)

#endif  // HANDWRITTEN_HMX_W4A16_KERNEL_H_
