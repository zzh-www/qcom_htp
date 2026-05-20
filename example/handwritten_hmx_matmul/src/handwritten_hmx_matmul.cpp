#include "handwritten_hmx_matmul.h"

#include <algorithm>
#include <cstring>

namespace {

HmStatus ValidatePreparedRun(const HmPreparedRun *run) {
  if (run == nullptr) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->m == 0 || run->k == 0 || run->n == 0 || run->chain == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->activation.data == nullptr || run->activation.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->packed_weight.data == nullptr || run->packed_weight.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->folded_bias.data == nullptr || run->folded_bias.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->control.data == nullptr || run->control.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->extra_control.data == nullptr || run->extra_control.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->activation_table.data == nullptr || run->activation_table.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->output_table.data == nullptr || run->output_table.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->descriptor.data == nullptr || run->descriptor.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->mask_control.data == nullptr || run->mask_control.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  if (run->output.data == nullptr || run->output.bytes == 0) {
    return HM_STATUS_INVALID_ARGUMENT;
  }
  return HM_STATUS_OK;
}

HmStatus CopySmokeRun(const HmPreparedRun *run) {
  HmStatus status = ValidatePreparedRun(run);
  if (status != HM_STATUS_OK) {
    return status;
  }
  const size_t bytes = std::min(run->activation.bytes, run->output.bytes);
  std::memcpy(run->output.data, run->activation.data, bytes);
  if (bytes < run->output.bytes) {
    std::memset(static_cast<unsigned char *>(run->output.data) + bytes, 0,
                run->output.bytes - bytes);
  }
  return HM_STATUS_OK;
}

}  // namespace

extern "C" HmStatus hm_u8i8_run_prepared(const HmPreparedRun *run) {
  return CopySmokeRun(run);
}

extern "C" HmStatus hm_w4a8_run_prepared(const HmPreparedRun *run) {
  return CopySmokeRun(run);
}

extern "C" HmStatus hm_w8a16_run_prepared(const HmPreparedRun *run) {
  return CopySmokeRun(run);
}

extern "C" HmStatus hm_w4a16_run_prepared(const HmPreparedRun *run) {
  return CopySmokeRun(run);
}

extern "C" HmStatus hm_w16a16_run_prepared(const HmPreparedRun *run) {
  return CopySmokeRun(run);
}

extern "C" const char *hm_status_string(HmStatus status) {
  switch (status) {
    case HM_STATUS_OK:
      return "ok";
    case HM_STATUS_INVALID_ARGUMENT:
      return "invalid_argument";
    case HM_STATUS_UNSUPPORTED:
      return "unsupported";
    case HM_STATUS_RUNTIME_ERROR:
      return "runtime_error";
  }
  return "unknown";
}
