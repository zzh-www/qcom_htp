#ifndef HANDWRITTEN_HMX_KERNEL_ABI_H_
#define HANDWRITTEN_HMX_KERNEL_ABI_H_

#include <stddef.h>

#if defined(__cplusplus)
#define HM_STATIC_ASSERT static_assert
#else
#define HM_STATIC_ASSERT _Static_assert
#endif

#if defined(__hexagon__)
#define HM_ASSERT_CONV1X1_ABI(OUT_DESC, ACT_DESC, MASK_DESC)                         \
  HM_STATIC_ASSERT(sizeof(OUT_DESC) == 24, "out descriptor size changed");           \
  HM_STATIC_ASSERT(offsetof(OUT_DESC, out_tile_ptr_table) == 0,                      \
                   "out descriptor table offset changed");                          \
  HM_STATIC_ASSERT(offsetof(OUT_DESC, out_table_stride_dwords) == 4,                 \
                   "out descriptor stride offset changed");                         \
  HM_STATIC_ASSERT(offsetof(OUT_DESC, out_y_stride_words) == 8,                      \
                   "out descriptor y stride offset changed");                       \
  HM_STATIC_ASSERT(offsetof(OUT_DESC, n_tiles_pow2) == 12,                           \
                   "out descriptor tile-count offset changed");                     \
  HM_STATIC_ASSERT(offsetof(OUT_DESC, m_total_minus_step) == 16,                     \
                   "out descriptor m-span offset changed");                         \
  HM_STATIC_ASSERT(offsetof(OUT_DESC, k_total_bytes) == 20,                          \
                   "out descriptor k-byte offset changed");                         \
  HM_STATIC_ASSERT(sizeof(ACT_DESC) == 12, "activation descriptor size changed");    \
  HM_STATIC_ASSERT(offsetof(ACT_DESC, act_ptr_pairs) == 0,                           \
                   "activation descriptor table offset changed");                   \
  HM_STATIC_ASSERT(offsetof(ACT_DESC, n_act_pairs) == 4,                             \
                   "activation descriptor pair-count offset changed");              \
  HM_STATIC_ASSERT(offsetof(ACT_DESC, act_table_y_stride_words) == 8,                \
                   "activation descriptor y stride offset changed");                \
  HM_STATIC_ASSERT(sizeof(MASK_DESC) == 28, "mask descriptor size changed");         \
  HM_STATIC_ASSERT(offsetof(MASK_DESC, out_check) == 0,                              \
                   "mask descriptor out_check offset changed");                     \
  HM_STATIC_ASSERT(offsetof(MASK_DESC, out_rt_mask) == 4,                            \
                   "mask descriptor out_rt_mask offset changed");                   \
  HM_STATIC_ASSERT(offsetof(MASK_DESC, act_check) == 8,                              \
                   "mask descriptor act_check offset changed");                     \
  HM_STATIC_ASSERT(offsetof(MASK_DESC, act_rt_base) == 12,                           \
                   "mask descriptor act_rt_base offset changed");                   \
  HM_STATIC_ASSERT(offsetof(MASK_DESC, filter_x_stride) == 16,                       \
                   "mask descriptor filter_x_stride offset changed");               \
  HM_STATIC_ASSERT(offsetof(MASK_DESC, pad14) == 20,                                 \
                   "mask descriptor pad14 offset changed");                         \
  HM_STATIC_ASSERT(offsetof(MASK_DESC, alt_rt) == 24,                                \
                   "mask descriptor alt_rt offset changed")
#else
#define HM_ASSERT_CONV1X1_ABI(OUT_DESC, ACT_DESC, MASK_DESC)
#endif

#endif  // HANDWRITTEN_HMX_KERNEL_ABI_H_
