#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ncnn-mlir/Vulkan/ModelABI.h"
#include "relu_scalar.h"

_Static_assert(NCNN_VK_ABI_VERSION == 1, "unexpected Vulkan ABI version");
_Static_assert(sizeof(ncnn_vk_status) == 4, "unexpected status width");
_Static_assert(sizeof(ncnn_vk_options) == 16, "unexpected options layout");
_Static_assert(offsetof(ncnn_vk_options, struct_size) == 0,
               "unexpected struct_size offset");
_Static_assert(offsetof(ncnn_vk_options, abi_version) == 4,
               "unexpected abi_version offset");
_Static_assert(offsetof(ncnn_vk_options, device_index) == 8,
               "unexpected device_index offset");
_Static_assert(offsetof(ncnn_vk_options, flags) == 12,
               "unexpected flags offset");
_Static_assert(NCNN_VK_OPTIONS_V1_SIZE == 16, "unexpected options size");
_Static_assert(NCNN_VK_DEVICE_AUTO == UINT32_MAX,
               "unexpected auto device value");

#define FIXTURE_INPUT_COUNT 1
#define FIXTURE_OUTPUT_COUNT 1
#define FIXTURE_INPUT1_RANK 3
#define FIXTURE_INPUT1_DIM0 INT64_C(3)
#define FIXTURE_INPUT1_DIM1 INT64_C(4)
#define FIXTURE_INPUT1_DIM2 INT64_C(5)
#define FIXTURE_INPUT1_ELEMENTS UINT64_C(60)
#define FIXTURE_OUTPUT1_RANK 3
#define FIXTURE_OUTPUT1_DIM0 INT64_C(3)
#define FIXTURE_OUTPUT1_DIM1 INT64_C(4)
#define FIXTURE_OUTPUT1_DIM2 INT64_C(5)
#define FIXTURE_OUTPUT1_ELEMENTS UINT64_C(60)

_Static_assert(FIXTURE_INPUT_COUNT == RELU_SCALAR_INPUT_COUNT &&
                 FIXTURE_OUTPUT_COUNT == RELU_SCALAR_OUTPUT_COUNT,
               "Vulkan and CPU fixture I/O counts differ");
_Static_assert(FIXTURE_INPUT1_RANK == RELU_SCALAR_INPUT1_RANK &&
                 FIXTURE_INPUT1_DIM0 == RELU_SCALAR_INPUT1_DIM0 &&
                 FIXTURE_INPUT1_DIM1 == RELU_SCALAR_INPUT1_DIM1 &&
                 FIXTURE_INPUT1_DIM2 == RELU_SCALAR_INPUT1_DIM2 &&
                 FIXTURE_INPUT1_ELEMENTS == RELU_SCALAR_INPUT1_ELEMENTS,
               "Vulkan and CPU fixture input contracts differ");
_Static_assert(FIXTURE_OUTPUT1_RANK == RELU_SCALAR_OUTPUT1_RANK &&
                 FIXTURE_OUTPUT1_DIM0 == RELU_SCALAR_OUTPUT1_DIM0 &&
                 FIXTURE_OUTPUT1_DIM1 == RELU_SCALAR_OUTPUT1_DIM1 &&
                 FIXTURE_OUTPUT1_DIM2 == RELU_SCALAR_OUTPUT1_DIM2 &&
                 FIXTURE_OUTPUT1_ELEMENTS == RELU_SCALAR_OUTPUT1_ELEMENTS,
               "Vulkan and CPU fixture output contracts differ");
_Static_assert(_Generic(&relu_scalar,
                 int (*)(const float*, float*): 1,
                 default: 0),
               "unexpected CPU fixture C function signature");

typedef struct fixture_vk_context fixture_vk_context;

ncnn_vk_status fixture_vk_create(const ncnn_vk_options* options,
                                 fixture_vk_context** context,
                                 char* diagnostic,
                                 size_t diagnostic_capacity);
ncnn_vk_status fixture_vk_run(fixture_vk_context* context,
                              const float* input1,
                              size_t input1_elements,
                              float* output1,
                              size_t output1_elements,
                              char* diagnostic,
                              size_t diagnostic_capacity);
ncnn_vk_status fixture_vk_destroy(fixture_vk_context* context);

/* Test-only ABI stub; this file does not exercise a Vulkan runtime. */
struct fixture_vk_context {
  int live;
};

static ncnn_vk_status (*create_signature)(const ncnn_vk_options*,
                                          fixture_vk_context**,
                                          char*,
                                          size_t) = fixture_vk_create;
static ncnn_vk_status (*run_signature)(fixture_vk_context*,
                                       const float*,
                                       size_t,
                                       float*,
                                       size_t,
                                       char*,
                                       size_t) = fixture_vk_run;
static ncnn_vk_status (*destroy_signature)(fixture_vk_context*) =
  fixture_vk_destroy;

static ncnn_vk_status report(ncnn_vk_status status,
                             char* diagnostic,
                             size_t diagnostic_capacity,
                             const char* message) {
  if (diagnostic_capacity > 0 && diagnostic != NULL) {
    const size_t length = strlen(message);
    const size_t copy_length =
      length < diagnostic_capacity - 1 ? length : diagnostic_capacity - 1;
    memcpy(diagnostic, message, copy_length);
    diagnostic[copy_length] = '\0';
  }
  return status;
}

ncnn_vk_status fixture_vk_create(const ncnn_vk_options* options,
                                 fixture_vk_context** context,
                                 char* diagnostic,
                                 size_t diagnostic_capacity) {
  if (context == NULL) {
    return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                  diagnostic,
                  diagnostic_capacity,
                  "missing context output");
  }
  *context = NULL;
  if (diagnostic_capacity > 0 && diagnostic == NULL) {
    return NCNN_VK_STATUS_INVALID_ARGUMENT;
  }
  if (diagnostic_capacity > 0) {
    diagnostic[0] = '\0';
  }
  if (options == NULL || options->struct_size < NCNN_VK_OPTIONS_V1_SIZE ||
      options->abi_version != NCNN_VK_ABI_VERSION || options->flags != 0) {
    return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                  diagnostic,
                  diagnostic_capacity,
                  "invalid options");
  }
  if (options->device_index != NCNN_VK_DEVICE_AUTO &&
      options->device_index != 0) {
    return report(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
                  diagnostic,
                  diagnostic_capacity,
                  "device is not supported");
  }
  fixture_vk_context* created = malloc(sizeof(*created));
  if (created == NULL) {
    return report(NCNN_VK_STATUS_OUT_OF_MEMORY,
                  diagnostic,
                  diagnostic_capacity,
                  "cannot create context");
  }
  created->live = 1;
  *context = created;
  return NCNN_VK_STATUS_SUCCESS;
}

ncnn_vk_status fixture_vk_run(fixture_vk_context* context,
                              const float* input1,
                              size_t input1_elements,
                              float* output1,
                              size_t output1_elements,
                              char* diagnostic,
                              size_t diagnostic_capacity) {
  if (diagnostic_capacity > 0 && diagnostic == NULL) {
    return NCNN_VK_STATUS_INVALID_ARGUMENT;
  }
  if (diagnostic_capacity > 0) {
    diagnostic[0] = '\0';
  }
  if (context == NULL || !context->live || input1 == NULL || output1 == NULL) {
    return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                  diagnostic,
                  diagnostic_capacity,
                  "invalid run argument");
  }
  if (input1_elements != FIXTURE_INPUT1_ELEMENTS ||
      output1_elements != FIXTURE_OUTPUT1_ELEMENTS) {
    return report(NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
                  diagnostic,
                  diagnostic_capacity,
                  "element count does not match static shape");
  }
  for (size_t index = 0; index < FIXTURE_OUTPUT1_ELEMENTS; ++index) {
    output1[index] = input1[index];
  }
  return NCNN_VK_STATUS_SUCCESS;
}

ncnn_vk_status fixture_vk_destroy(fixture_vk_context* context) {
  if (context == NULL) {
    return NCNN_VK_STATUS_SUCCESS;
  }
  if (!context->live) {
    return NCNN_VK_STATUS_INVALID_ARGUMENT;
  }
  context->live = 0;
  free(context);
  return NCNN_VK_STATUS_SUCCESS;
}

int main(void) {
  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  fixture_vk_context* context = NULL;
  fixture_vk_context* other_context = NULL;
  char diagnostic[2] = {'x', 'x'};
  float input[FIXTURE_INPUT1_ELEMENTS] = {0};
  float output[FIXTURE_OUTPUT1_ELEMENTS] = {0};

  if (create_signature == NULL || run_signature == NULL ||
      destroy_signature == NULL ||
      fixture_vk_destroy(NULL) != NCNN_VK_STATUS_SUCCESS) {
    return 1;
  }
  if (fixture_vk_create(NULL, &context, diagnostic, sizeof(diagnostic)) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != NULL || diagnostic[1] != '\0') {
    return 2;
  }

  ncnn_vk_options_init(&options);
  options.flags = 1;
  if (fixture_vk_create(&options, &context, NULL, 0) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != NULL) {
    return 3;
  }

  ncnn_vk_options_init(&options);
  options.abi_version = NCNN_VK_ABI_VERSION + 1;
  if (fixture_vk_create(&options, &context, NULL, 0) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != NULL) {
    return 4;
  }

  ncnn_vk_options_init(&options);
  options.struct_size = NCNN_VK_OPTIONS_V1_SIZE - 1;
  if (fixture_vk_create(&options, &context, NULL, 0) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != NULL) {
    return 4;
  }

  ncnn_vk_options_init(&options);
  options.device_index = 1;
  if (fixture_vk_create(&options, &context, NULL, 0) !=
        NCNN_VK_STATUS_UNSUPPORTED_DEVICE ||
      context != NULL) {
    return 4;
  }

  ncnn_vk_options_init(&options);
  options.struct_size = NCNN_VK_OPTIONS_V1_SIZE + 8;
  if (fixture_vk_create(&options, &context, NULL, 0) !=
        NCNN_VK_STATUS_SUCCESS ||
      context == NULL ||
      fixture_vk_create(&options, &other_context, NULL, 0) !=
        NCNN_VK_STATUS_SUCCESS ||
      other_context == NULL || other_context == context) {
    return 5;
  }
  if (fixture_vk_run(NULL,
                     input,
                     FIXTURE_INPUT1_ELEMENTS,
                     output,
                     FIXTURE_OUTPUT1_ELEMENTS,
                     NULL,
                     0) != NCNN_VK_STATUS_INVALID_ARGUMENT) {
    return 6;
  }
  if (fixture_vk_run(context,
                     NULL,
                     FIXTURE_INPUT1_ELEMENTS,
                     output,
                     FIXTURE_OUTPUT1_ELEMENTS,
                     NULL,
                     0) != NCNN_VK_STATUS_INVALID_ARGUMENT) {
    return 7;
  }
  if (fixture_vk_run(context,
                     input,
                     FIXTURE_INPUT1_ELEMENTS - 1,
                     output,
                     FIXTURE_OUTPUT1_ELEMENTS,
                     NULL,
                     0) != NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH) {
    return 8;
  }
  input[0] = 42.0f;
  if (fixture_vk_run(context,
                     input,
                     FIXTURE_INPUT1_ELEMENTS,
                     output,
                     FIXTURE_OUTPUT1_ELEMENTS,
                     NULL,
                     0) != NCNN_VK_STATUS_SUCCESS ||
      output[0] != input[0]) {
    return 9;
  }
  if (fixture_vk_destroy(context) != NCNN_VK_STATUS_SUCCESS ||
      fixture_vk_destroy(other_context) != NCNN_VK_STATUS_SUCCESS) {
    return 10;
  }
  return 0;
}
