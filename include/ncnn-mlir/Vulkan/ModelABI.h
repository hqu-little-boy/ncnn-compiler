#pragma once

#include <stddef.h>
#include <stdint.h>

#define NCNN_VK_ABI_VERSION UINT32_C(1)
#define NCNN_VK_DEVICE_AUTO UINT32_MAX
#define NCNN_VK_OPTIONS_V1_SIZE ((uint32_t)sizeof(ncnn_vk_options))

#ifdef __cplusplus
using ncnn_vk_status = uint32_t;
#else
typedef uint32_t ncnn_vk_status;
#endif

enum {
  NCNN_VK_STATUS_SUCCESS = 0,
  NCNN_VK_STATUS_INVALID_ARGUMENT = 1,
  NCNN_VK_STATUS_UNSUPPORTED_DEVICE = 2,
  NCNN_VK_STATUS_OUT_OF_MEMORY = 3,
  NCNN_VK_STATUS_DEVICE_LOST = 4,
  NCNN_VK_STATUS_API_ERROR = 5,
  NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH = 6
};

#ifdef __cplusplus
struct ncnn_vk_options {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t device_index;
  uint32_t flags;
};
#else
typedef struct ncnn_vk_options {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t device_index;
  uint32_t flags;
} ncnn_vk_options;
#endif

static inline void ncnn_vk_options_init(ncnn_vk_options* options) {
#ifdef __cplusplus
  if (options == nullptr) {
#else
  if (options == NULL) {
#endif
    return;
  }
  options->struct_size = NCNN_VK_OPTIONS_V1_SIZE;
  options->abi_version = NCNN_VK_ABI_VERSION;
  options->device_index = NCNN_VK_DEVICE_AUTO;
  options->flags = 0;
}
