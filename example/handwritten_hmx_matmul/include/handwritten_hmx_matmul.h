#ifndef HANDWRITTEN_HMX_MATMUL_H_
#define HANDWRITTEN_HMX_MATMUL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HmStatus {
  HM_STATUS_OK = 0,
  HM_STATUS_INVALID_ARGUMENT = 1,
  HM_STATUS_UNSUPPORTED = 2,
  HM_STATUS_RUNTIME_ERROR = 3,
} HmStatus;

typedef struct HmBuffer {
  void *data;
  size_t bytes;
} HmBuffer;

typedef struct HmConstBuffer {
  const void *data;
  size_t bytes;
} HmConstBuffer;

typedef struct HmPreparedRun {
  uint32_t m;
  uint32_t k;
  uint32_t n;
  uint32_t chain;
  HmConstBuffer activation;
  HmConstBuffer packed_weight;
  HmConstBuffer folded_bias;
  HmConstBuffer control;
  HmConstBuffer extra_control;
  HmConstBuffer activation_table;
  HmConstBuffer output_table;
  HmConstBuffer descriptor;
  HmConstBuffer mask_control;
  HmBuffer scratch;
  HmBuffer output;
} HmPreparedRun;

HmStatus hm_u8i8_run_prepared(const HmPreparedRun *run);
HmStatus hm_w4a8_run_prepared(const HmPreparedRun *run);
HmStatus hm_w8a16_run_prepared(const HmPreparedRun *run);
HmStatus hm_w4a16_run_prepared(const HmPreparedRun *run);
HmStatus hm_w16a16_run_prepared(const HmPreparedRun *run);

const char *hm_status_string(HmStatus status);

#ifdef __cplusplus
}
#endif

#endif  // HANDWRITTEN_HMX_MATMUL_H_
