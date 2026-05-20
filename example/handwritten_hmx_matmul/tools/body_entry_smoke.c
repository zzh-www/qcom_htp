#include <stdint.h>
#include <stdio.h>

#include <h2.h>
#include <h2_common_info.h>
#include <h2_mxaccess.h>

#include "handwritten_hmx_u8i8_kernel.h"
#include "handwritten_hmx_w4a16_kernel.h"
#include "handwritten_hmx_w4a8_kernel.h"
#include "handwritten_hmx_w8a16_kernel.h"

#define TILE_BYTES 4096u
#define TABLE_WORDS 64u

static uint32_t ptr32(const void *ptr) {
  return (uint32_t)(uintptr_t)ptr;
}

static void fill_bytes(uint8_t *dst, uint32_t bytes, uint8_t value) {
  for (uint32_t i = 0; i < bytes; ++i) {
    dst[i] = value;
  }
}

static void fill_table(int32_t *table, uint32_t words, void *ptr) {
  const int32_t value = (int32_t)ptr32(ptr);
  for (uint32_t i = 0; i < words; ++i) {
    table[i] = value;
  }
}

static void init_mask_words(uint32_t *mask_words) {
  for (uint32_t i = 0; i < 16u; ++i) {
    mask_words[i] = 0u;
  }
}

static void init_common(
    int32_t *out_table,
    int32_t *act_table,
    void *out_surface,
    void *act_surface) {
  fill_table(out_table, TABLE_WORDS, out_surface);
  fill_table(act_table, TABLE_WORDS, act_surface);
}

static int run_u8i8(uint8_t *base) {
  uint8_t *act = base + 0x00000u;
  uint8_t *weight = base + 0x10000u;
  uint8_t *bias = base + 0x20000u;
  uint8_t *out = base + 0x30000u;
  int32_t *act_table = (int32_t *)(base + 0x40000u);
  int32_t *out_table = (int32_t *)(base + 0x41000u);
  uint32_t mask_words[16] __attribute__((aligned(16)));
  uint32_t extra[2] __attribute__((aligned(16))) = {1u, 0u};

  fill_bytes(act, TILE_BYTES, 1u);
  fill_bytes(weight, TILE_BYTES, 1u);
  fill_bytes(bias, TILE_BYTES, 0u);
  fill_bytes(out, TILE_BYTES, 0u);
  init_common(out_table, act_table, out, act);
  init_mask_words(mask_words);

  HmU8I8OutDesc out_desc = {out_table, 1u, 1u, 1u, 8, 64u};
  HmU8I8ActDesc act_desc = {act_table, 2u, 1u};
  hm_u8i8_v73deep_kernel(
      &out_desc, &act_desc, weight, bias, (const HmU8I8MaskDesc *)mask_words, extra);
  return 0;
}

static int run_w4a8(uint8_t *base) {
  uint8_t *act = base + 0x00000u;
  uint8_t *weight = base + 0x10000u;
  uint8_t *bias = base + 0x20000u;
  uint8_t *out = base + 0x30000u;
  int32_t *act_table = (int32_t *)(base + 0x40000u);
  int32_t *out_table = (int32_t *)(base + 0x41000u);
  uint32_t mask_words[16] __attribute__((aligned(16)));
  uint32_t extra[2] __attribute__((aligned(16))) = {1u, 0u};

  fill_bytes(act, TILE_BYTES, 1u);
  fill_bytes(weight, TILE_BYTES, 0x11u);
  fill_bytes(bias, TILE_BYTES, 0u);
  fill_bytes(out, TILE_BYTES, 0u);
  init_common(out_table, act_table, out, act);
  init_mask_words(mask_words);

  HmW4A8OutDesc out_desc = {out_table, 1u, 1u, 1u, 8, 64u};
  HmW4A8ActDesc act_desc = {act_table, 2u, 1u};
  hm_w4a8_v73deep_kernel(
      &out_desc, &act_desc, weight, bias, (const HmW4A8MaskDesc *)mask_words, extra);
  return 0;
}

static int run_w8a16(uint8_t *base) {
  uint8_t *act = base + 0x00000u;
  uint8_t *weight = base + 0x10000u;
  uint8_t *bias = base + 0x20000u;
  uint8_t *out = base + 0x30000u;
  int32_t *act_table = (int32_t *)(base + 0x40000u);
  int32_t *out_table = (int32_t *)(base + 0x41000u);
  uint32_t mask_words[16] __attribute__((aligned(16)));
  uint32_t extra[2] __attribute__((aligned(16))) = {1u, 0u};

  fill_bytes(act, TILE_BYTES, 1u);
  fill_bytes(weight, TILE_BYTES, 1u);
  fill_bytes(bias, TILE_BYTES, 0u);
  fill_bytes(out, TILE_BYTES, 0u);
  init_common(out_table, act_table, out, act);
  init_mask_words(mask_words);

  HmW8A16OutDesc out_desc = {out_table, 1u, 1u, 1u, 8, 64u};
  HmW8A16ActDesc act_desc = {act_table, 2u, 1u};
  hm_w8a16_v75deep_kernel(
      &out_desc, &act_desc, weight, bias, (const HmW8A16MaskDesc *)mask_words, extra);
  return 0;
}

static int run_w4a16(uint8_t *base) {
  uint8_t *act = base + 0x00000u;
  uint8_t *weight = base + 0x10000u;
  uint8_t *bias = base + 0x20000u;
  uint8_t *out = base + 0x30000u;
  int32_t *act_table = (int32_t *)(base + 0x40000u);
  int32_t *out_table = (int32_t *)(base + 0x41000u);
  uint32_t mask_words[16] __attribute__((aligned(16)));
  uint32_t extra[2] __attribute__((aligned(16))) = {1u, 0u};

  fill_bytes(act, TILE_BYTES, 1u);
  fill_bytes(weight, TILE_BYTES, 0x11u);
  fill_bytes(bias, TILE_BYTES, 0u);
  fill_bytes(out, TILE_BYTES, 0u);
  init_common(out_table, act_table, out, act);
  init_mask_words(mask_words);

  HmW4A16OutDesc out_desc = {out_table, 1u, 1u, 1u, 8, 64u};
  HmW4A16ActDesc act_desc = {act_table, 2u, 1u};
  hm_w4a16_v73deep_kernel(
      &out_desc, &act_desc, weight, bias, (const HmW4A16MaskDesc *)mask_words, extra);
  return 0;
}

static int run_family(const char *name, int (*fn)(uint8_t *), uint8_t *base) {
  printf("[RUN] %s body entry\n", name);
  int rc = fn(base);
  if (rc == 0) {
    printf("[PASS] %s body entry returned\n", name);
  } else {
    printf("[FAIL] %s body entry rc=%d\n", name, rc);
  }
  return rc == 0 ? 0 : 1;
}

int main(void) {
  int fail = 0;
  unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);
  unsigned int vtcm_size = h2_info(INFO_VTCM_SIZE);
  printf("Handwritten HMX body-entry simulator smoke\n");
  printf("[Init] VTCM base=0x%08x size=%u KB\n", vtcm_base, vtcm_size);
  if (vtcm_base == 0 || vtcm_size < 1024u) {
    printf("[FAIL] missing VTCM\n");
    h2_thread_stop(1);
    return 1;
  }

  h2_mxaccess_state_t mxacc;
  h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0, CFG_HMX_CONTEXTS, 0x1);
  int mret = h2_mxaccess_acquire(&mxacc);
  printf("[Init] HMX acquired (%d)\n", mret);

  uint8_t *base = (uint8_t *)(uintptr_t)(vtcm_base + 0x10000u);
  fail += run_family("u8i8", run_u8i8, base);
  fail += run_family("w4a8", run_w4a8, base);
  fail += run_family("w8a16", run_w8a16, base);
  fail += run_family("w4a16", run_w4a16, base);

  printf("Results: %d PASS / %d FAIL\n", 4 - fail, fail);
  h2_thread_stop(fail);
  return fail;
}
