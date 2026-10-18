#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lifecycle_fixture_vulkan.h"

_Static_assert(LIFECYCLE_FIXTURE_VULKAN_INPUT_COUNT == 1,
               "unexpected lifecycle fixture input count");
_Static_assert(LIFECYCLE_FIXTURE_VULKAN_OUTPUT_COUNT == 1,
               "unexpected lifecycle fixture output count");
_Static_assert(LIFECYCLE_FIXTURE_VULKAN_INPUT1_RANK == 1,
               "unexpected lifecycle fixture input rank");
_Static_assert(LIFECYCLE_FIXTURE_VULKAN_INPUT1_ELEMENTS == UINT64_C(17),
               "unexpected lifecycle fixture input extent");
_Static_assert(LIFECYCLE_FIXTURE_VULKAN_OUTPUT1_RANK == 1,
               "unexpected lifecycle fixture output rank");
_Static_assert(LIFECYCLE_FIXTURE_VULKAN_OUTPUT1_ELEMENTS == UINT64_C(17),
               "unexpected lifecycle fixture output extent");

enum { LIFECYCLE_DIAGNOSTIC_CAPACITY = 2 };

typedef ncnn_vk_status (*lifecycle_create_fn)(const ncnn_vk_options*,
                                              lifecycle_fixture_vk_context**,
                                              char*,
                                              size_t);
typedef ncnn_vk_status (*lifecycle_run_fn)(lifecycle_fixture_vk_context*,
                                           const float*,
                                           size_t,
                                           float*,
                                           size_t,
                                           char*,
                                           size_t);
typedef ncnn_vk_status (*lifecycle_destroy_fn)(lifecycle_fixture_vk_context*);

struct harness_options {
  const char* library_path;
  const char* result_path;
  /* This is recorded as an expectation; device selection uses device_index. */
  const char* expected_device_match;
  uint32_t device_index;
  int no_device;
  int validation;
};

struct harness_result {
  ncnn_vk_status create_status;
  ncnn_vk_status destroy_status;
  ncnn_vk_status destroy_null_status;
  ncnn_vk_status options_null_status;
  ncnn_vk_status context_output_null_status;
  ncnn_vk_status options_bad_size_status;
  ncnn_vk_status options_bad_version_status;
  ncnn_vk_status options_bad_flags_status;
  ncnn_vk_status invalid_ordinal_status;
  ncnn_vk_status run_status;
  ncnn_vk_status run_null_context_status;
  ncnn_vk_status run_null_input_status;
  ncnn_vk_status run_null_output_status;
  ncnn_vk_status run_bad_input_count_status;
  ncnn_vk_status run_bad_output_count_status;
  unsigned int created_contexts;
  int independent_contexts;
  int repeated_create_destroy;
  int run_output_unchanged;
  int invalid_run_sentinels_unchanged;
  int diagnostic_bounded;
  int passed;
};

static int parse_options(int argc,
                         char** argv,
                         struct harness_options* options) {
  int index;
  options->library_path = NULL;
  options->result_path = NULL;
  options->expected_device_match = "";
  options->device_index = NCNN_VK_DEVICE_AUTO;
  options->no_device = 0;
  options->validation = 0;
  for (index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "--library") == 0 && index + 1 < argc) {
      options->library_path = argv[++index];
    } else if (strcmp(argv[index], "--result") == 0 && index + 1 < argc) {
      options->result_path = argv[++index];
    } else if (strcmp(argv[index], "--device-match") == 0 && index + 1 < argc) {
      options->expected_device_match = argv[++index];
    } else if (strcmp(argv[index], "--ordinal") == 0 && index + 1 < argc) {
      char* end = NULL;
      unsigned long value;
      errno = 0;
      value = strtoul(argv[++index], &end, 10);
      if (errno != 0 || end == argv[index] || *end != '\0' ||
          value > UINT32_MAX) {
        return 0;
      }
      options->device_index = (uint32_t)value;
    } else if (strcmp(argv[index], "--no-device") == 0) {
      options->no_device = 1;
    } else if (strcmp(argv[index], "--validation") == 0) {
      options->validation = 1;
    } else {
      return 0;
    }
  }
  return options->library_path != NULL && options->result_path != NULL;
}

static int load_function(void* library,
                         const char* name,
                         void* function,
                         size_t function_size) {
  void* address;
  dlerror();
  address = dlsym(library, name);
  if (dlerror() != NULL || address == NULL ||
      function_size != sizeof(address)) {
    return 0;
  }
  memcpy(function, &address, function_size);
  return 1;
}

static int write_json_string(FILE* output, const char* value) {
  static const char kHexDigits[] = "0123456789abcdef";
  const unsigned char* cursor = (const unsigned char*)value;
  if (fputc('"', output) == EOF) {
    return 0;
  }
  for (; *cursor != '\0'; ++cursor) {
    switch (*cursor) {
      case '"':
        if (fputs("\\\"", output) == EOF) {
          return 0;
        }
        break;
      case '\\':
        if (fputs("\\\\", output) == EOF) {
          return 0;
        }
        break;
      case '\b':
        if (fputs("\\b", output) == EOF) {
          return 0;
        }
        break;
      case '\f':
        if (fputs("\\f", output) == EOF) {
          return 0;
        }
        break;
      case '\n':
        if (fputs("\\n", output) == EOF) {
          return 0;
        }
        break;
      case '\r':
        if (fputs("\\r", output) == EOF) {
          return 0;
        }
        break;
      case '\t':
        if (fputs("\\t", output) == EOF) {
          return 0;
        }
        break;
      default:
        if (*cursor < 0x20U) {
          if (fputs("\\u00", output) == EOF ||
              fputc(kHexDigits[*cursor >> 4U], output) == EOF ||
              fputc(kHexDigits[*cursor & 0x0fU], output) == EOF) {
            return 0;
          }
        } else if (fputc(*cursor, output) == EOF) {
          return 0;
        }
        break;
    }
  }
  return fputc('"', output) != EOF;
}

static int write_result(FILE* output,
                        const struct harness_options* options,
                        const struct harness_result* result) {
  if (output == NULL ||
      fprintf(output,
              "{\n"
              "  \"schema_version\": 1,\n"
              "  \"kind\": \"ncnn.vulkan_lifecycle_native_result\",\n"
              "  \"fixture_id\": \"lifecycle_fixture\",\n"
              "  \"expected_device_match\": ") < 0 ||
      !write_json_string(output, options->expected_device_match) ||
      fprintf(output,
              ",\n"
              "  \"requested_device_index\": %" PRIu32 ",\n"
              "  \"mode\": \"%s\",\n"
              "  \"status\": \"%s\",\n"
              "  \"validation_enabled\": %s,\n"
              "  \"create_status\": %" PRIu32 ",\n"
              "  \"destroy_status\": %" PRIu32 ",\n"
              "  \"destroy_null_status\": %" PRIu32 ",\n"
              "  \"options_null_status\": %" PRIu32 ",\n"
              "  \"context_output_null_status\": %" PRIu32 ",\n"
              "  \"options_bad_size_status\": %" PRIu32 ",\n"
              "  \"options_bad_version_status\": %" PRIu32 ",\n"
              "  \"options_bad_flags_status\": %" PRIu32 ",\n"
              "  \"invalid_ordinal_status\": %" PRIu32 ",\n"
              "  \"run_status\": %" PRIu32 ",\n"
              "  \"run_null_context_status\": %" PRIu32 ",\n"
              "  \"run_null_input_status\": %" PRIu32 ",\n"
              "  \"run_null_output_status\": %" PRIu32 ",\n"
              "  \"run_bad_input_count_status\": %" PRIu32 ",\n"
              "  \"run_bad_output_count_status\": %" PRIu32 ",\n"
              "  \"created_contexts\": %u,\n"
              "  \"independent_contexts\": %s,\n"
              "  \"repeated_create_destroy\": %s,\n"
              "  \"run_output_unchanged\": %s,\n"
              "  \"invalid_run_sentinels_unchanged\": %s,\n"
              "  \"diagnostic_bounded\": %s,\n"
              "  \"candidate_execution\": \"not_run\",\n"
              "  \"upload_count\": \"not_measured\"\n"
              "}\n",
              options->device_index,
              options->no_device ? "no-device" : "device",
              result->passed ? "passed" : "failed",
              options->validation ? "true" : "false",
              result->create_status,
              result->destroy_status,
              result->destroy_null_status,
              result->options_null_status,
              result->context_output_null_status,
              result->options_bad_size_status,
              result->options_bad_version_status,
              result->options_bad_flags_status,
              result->invalid_ordinal_status,
              result->run_status,
              result->run_null_context_status,
              result->run_null_input_status,
              result->run_null_output_status,
              result->run_bad_input_count_status,
              result->run_bad_output_count_status,
              result->created_contexts,
              result->independent_contexts ? "true" : "false",
              result->repeated_create_destroy ? "true" : "false",
              result->run_output_unchanged ? "true" : "false",
              result->invalid_run_sentinels_unchanged ? "true" : "false",
              result->diagnostic_bounded ? "true" : "false") < 0) {
    return 0;
  }
  return !ferror(output);
}

static int test_result_writer_json_escape(void) {
  const char expected[] = "\"\\\"quote\\\\slash\\nline\\ttab\\u0001\\\"\"";
  const char special_match[] = "\"quote\\slash\nline\ttab\001\"";
  struct harness_options options = {
    .library_path = "",
    .result_path = "",
    .expected_device_match = special_match,
    .device_index = NCNN_VK_DEVICE_AUTO,
    .no_device = 0,
    .validation = 0,
  };
  struct harness_result result = {0};
  char contents[4096];
  FILE* output = tmpfile();
  size_t size;
  if (output == NULL || !write_result(output, &options, &result) ||
      fflush(output) != 0 || fseek(output, 0, SEEK_END) != 0) {
    if (output != NULL) {
      fclose(output);
    }
    return 0;
  }
  const long end = ftell(output);
  if (end < 0 || (size_t)end >= sizeof(contents) ||
      fseek(output, 0, SEEK_SET) != 0) {
    fclose(output);
    return 0;
  }
  size = fread(contents, 1, (size_t)end, output);
  fclose(output);
  if (size != (size_t)end) {
    return 0;
  }
  contents[size] = '\0';
  return strstr(contents, expected) != NULL;
}

static int test_bad_options(lifecycle_create_fn create,
                            struct harness_result* result) {
  ncnn_vk_options options;
  lifecycle_fixture_vk_context* context = (lifecycle_fixture_vk_context*)1;
  char diagnostic[LIFECYCLE_DIAGNOSTIC_CAPACITY] = {'x', 'x'};

  ncnn_vk_options_init(&options);
  options.struct_size = NCNN_VK_OPTIONS_V1_SIZE - 1;
  result->options_bad_size_status =
    create(&options, &context, diagnostic, sizeof(diagnostic));
  if (result->options_bad_size_status != NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != NULL || diagnostic[sizeof(diagnostic) - 1] != '\0') {
    return 0;
  }
  result->diagnostic_bounded = 1;

  ncnn_vk_options_init(&options);
  options.abi_version = NCNN_VK_ABI_VERSION + 1;
  context = (lifecycle_fixture_vk_context*)1;
  result->options_bad_version_status = create(&options, &context, NULL, 0);
  if (result->options_bad_version_status != NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != NULL) {
    return 0;
  }

  ncnn_vk_options_init(&options);
  options.flags = 1;
  context = (lifecycle_fixture_vk_context*)1;
  result->options_bad_flags_status = create(&options, &context, NULL, 0);
  return result->options_bad_flags_status == NCNN_VK_STATUS_INVALID_ARGUMENT &&
         context == NULL;
}

static int test_create_pointer_arguments(lifecycle_create_fn create,
                                         struct harness_result* result) {
  ncnn_vk_options options;
  lifecycle_fixture_vk_context* context = (lifecycle_fixture_vk_context*)1;
  ncnn_vk_options_init(&options);
  result->options_null_status = create(NULL, &context, NULL, 0);
  if (result->options_null_status != NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != NULL) {
    return 0;
  }
  result->context_output_null_status = create(&options, NULL, NULL, 0);
  return result->context_output_null_status == NCNN_VK_STATUS_INVALID_ARGUMENT;
}

static int test_invalid_ordinal(lifecycle_create_fn create,
                                ncnn_vk_status* status) {
  ncnn_vk_options options;
  lifecycle_fixture_vk_context* context = NULL;
  ncnn_vk_options_init(&options);
  options.device_index = UINT32_MAX - 1;
  *status = create(&options, &context, NULL, 0);
  return *status == NCNN_VK_STATUS_UNSUPPORTED_DEVICE && context == NULL;
}

static void fill_output_sentinel(float* output, size_t count) {
  size_t index;
  for (index = 0; index < count; ++index) {
    output[index] = -1234.5F;
  }
}

static int output_is_sentinel(const float* output, size_t count) {
  size_t index;
  for (index = 0; index < count; ++index) {
    if (output[index] != -1234.5F) {
      return 0;
    }
  }
  return 1;
}

static int test_invalid_run_arguments(lifecycle_run_fn run,
                                      lifecycle_fixture_vk_context* context,
                                      const float* input,
                                      float* output,
                                      struct harness_result* result) {
  const size_t elements = LIFECYCLE_FIXTURE_VULKAN_INPUT1_ELEMENTS;
  int sentinels_unchanged = 1;

  fill_output_sentinel(output, elements);
  result->run_null_context_status =
    run(NULL, input, elements, output, elements, NULL, 0);
  sentinels_unchanged =
    sentinels_unchanged && output_is_sentinel(output, elements);

  fill_output_sentinel(output, elements);
  result->run_null_input_status =
    run(context, NULL, elements, output, elements, NULL, 0);
  sentinels_unchanged =
    sentinels_unchanged && output_is_sentinel(output, elements);

  fill_output_sentinel(output, elements);
  result->run_null_output_status =
    run(context, input, elements, NULL, elements, NULL, 0);
  sentinels_unchanged =
    sentinels_unchanged && output_is_sentinel(output, elements);

  fill_output_sentinel(output, elements);
  result->run_bad_input_count_status =
    run(context, input, elements - 1, output, elements, NULL, 0);
  sentinels_unchanged =
    sentinels_unchanged && output_is_sentinel(output, elements);

  fill_output_sentinel(output, elements);
  result->run_bad_output_count_status =
    run(context, input, elements, output, elements - 1, NULL, 0);
  sentinels_unchanged =
    sentinels_unchanged && output_is_sentinel(output, elements);

  result->invalid_run_sentinels_unchanged = sentinels_unchanged;
  return result->run_null_context_status == NCNN_VK_STATUS_INVALID_ARGUMENT &&
         result->run_null_input_status == NCNN_VK_STATUS_INVALID_ARGUMENT &&
         result->run_null_output_status == NCNN_VK_STATUS_INVALID_ARGUMENT &&
         result->run_bad_input_count_status ==
           NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH &&
         result->run_bad_output_count_status ==
           NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH &&
         sentinels_unchanged;
}

int main(int argc, char** argv) {
  struct harness_options harness;
  struct harness_result result = {0};
  void* library = NULL;
  FILE* result_file = NULL;
  const char* loader_error = NULL;
  int result_written = 0;
  lifecycle_create_fn create = NULL;
  lifecycle_run_fn run = NULL;
  lifecycle_destroy_fn destroy = NULL;
  lifecycle_fixture_vk_context* context = NULL;
  lifecycle_fixture_vk_context* second_context = NULL;
  float input[LIFECYCLE_FIXTURE_VULKAN_INPUT1_ELEMENTS] = {0.0F};
  float output[LIFECYCLE_FIXTURE_VULKAN_OUTPUT1_ELEMENTS];
  char diagnostic[LIFECYCLE_DIAGNOSTIC_CAPACITY] = {'x', 'x'};

  result.create_status = UINT32_MAX;
  result.destroy_status = UINT32_MAX;
  result.destroy_null_status = UINT32_MAX;
  result.options_null_status = UINT32_MAX;
  result.context_output_null_status = UINT32_MAX;
  result.options_bad_size_status = UINT32_MAX;
  result.options_bad_version_status = UINT32_MAX;
  result.options_bad_flags_status = UINT32_MAX;
  result.invalid_ordinal_status = UINT32_MAX;
  result.run_status = UINT32_MAX;
  result.run_null_context_status = UINT32_MAX;
  result.run_null_input_status = UINT32_MAX;
  result.run_null_output_status = UINT32_MAX;
  result.run_bad_input_count_status = UINT32_MAX;
  result.run_bad_output_count_status = UINT32_MAX;
  fill_output_sentinel(output, LIFECYCLE_FIXTURE_VULKAN_OUTPUT1_ELEMENTS);

  if (!parse_options(argc, argv, &harness)) {
    fprintf(stderr,
            "usage: vulkan_lifecycle_c_harness --library PATH --result PATH "
            "[--device-match EXPECTED-NAME] [--ordinal N] [--no-device] "
            "[--validation]\n");
    return EXIT_FAILURE;
  }
  if (!test_result_writer_json_escape()) {
    fprintf(stderr, "lifecycle result JSON string escaping self-test failed\n");
    return EXIT_FAILURE;
  }
  library = dlopen(harness.library_path, RTLD_NOW | RTLD_LOCAL);
  if (library == NULL ||
      !load_function(
        library, "lifecycle_fixture_vk_create", &create, sizeof(create)) ||
      !load_function(library, "lifecycle_fixture_vk_run", &run, sizeof(run)) ||
      !load_function(
        library, "lifecycle_fixture_vk_destroy", &destroy, sizeof(destroy))) {
    loader_error = dlerror();
    fprintf(stderr,
            "cannot load lifecycle ABI: %s\n",
            loader_error == NULL ? "missing candidate symbol" : loader_error);
    if (library != NULL) {
      dlclose(library);
    }
    return EXIT_FAILURE;
  }

  if (!test_bad_options(create, &result) ||
      !test_create_pointer_arguments(create, &result)) {
    goto finish;
  }
  result.destroy_null_status = destroy(NULL);
  if (result.destroy_null_status != NCNN_VK_STATUS_SUCCESS) {
    goto finish;
  }

  if (!harness.no_device) {
    ncnn_vk_options options;
    if (!test_invalid_ordinal(create, &result.invalid_ordinal_status)) {
      goto finish;
    }
    ncnn_vk_options_init(&options);
    options.device_index = harness.device_index;
    result.create_status =
      create(&options, &context, diagnostic, sizeof(diagnostic));
    if (result.create_status == NCNN_VK_STATUS_SUCCESS && context != NULL) {
      ++result.created_contexts;
      ncnn_vk_options_init(&options);
      options.device_index = NCNN_VK_DEVICE_AUTO;
      if (create(&options, &second_context, NULL, 0) ==
            NCNN_VK_STATUS_SUCCESS &&
          second_context != NULL && second_context != context) {
        result.independent_contexts = 1;
        ++result.created_contexts;
        if (!test_invalid_run_arguments(run, context, input, output, &result)) {
          goto finish;
        }
        fill_output_sentinel(output, LIFECYCLE_FIXTURE_VULKAN_OUTPUT1_ELEMENTS);
        result.run_status = run(context,
                                input,
                                LIFECYCLE_FIXTURE_VULKAN_INPUT1_ELEMENTS,
                                output,
                                LIFECYCLE_FIXTURE_VULKAN_OUTPUT1_ELEMENTS,
                                diagnostic,
                                sizeof(diagnostic));
        result.run_output_unchanged =
          output_is_sentinel(output, LIFECYCLE_FIXTURE_VULKAN_OUTPUT1_ELEMENTS);
        if (destroy(second_context) != NCNN_VK_STATUS_SUCCESS) {
          goto finish;
        }
        second_context = NULL;
        result.destroy_status = destroy(context);
        context = NULL;
        if (result.destroy_status != NCNN_VK_STATUS_SUCCESS) {
          goto finish;
        }
        result.repeated_create_destroy = 1;
        ncnn_vk_options_init(&options);
        options.device_index = NCNN_VK_DEVICE_AUTO;
        context = NULL;
        if (create(&options, &context, NULL, 0) != NCNN_VK_STATUS_SUCCESS ||
            context == NULL) {
          goto finish;
        }
        ++result.created_contexts;
        if (destroy(context) != NCNN_VK_STATUS_SUCCESS) {
          goto finish;
        }
        context = NULL;
      } else {
        goto finish;
      }
      if (result.run_status != NCNN_VK_STATUS_API_ERROR ||
          !result.run_output_unchanged) {
        goto finish;
      }
      result.passed = 1;
    } else if (context != NULL) {
      goto finish;
    }
  } else {
    ncnn_vk_options options;
    ncnn_vk_options_init(&options);
    result.create_status =
      create(&options, &context, diagnostic, sizeof(diagnostic));
    if (result.create_status != NCNN_VK_STATUS_UNSUPPORTED_DEVICE ||
        context != NULL || diagnostic[sizeof(diagnostic) - 1] != '\0' ||
        diagnostic[0] == '\0') {
      if (context != NULL) {
        (void)destroy(context);
        context = NULL;
      }
      goto finish;
    }
    result.diagnostic_bounded = 1;
    result.passed = 1;
  }

finish:
  if (second_context != NULL) {
    (void)destroy(second_context);
  }
  if (context != NULL) {
    (void)destroy(context);
  }
  result_file = fopen(harness.result_path, "wb");
  if (result_file != NULL) {
    result_written = write_result(result_file, &harness, &result);
    if (fclose(result_file) != 0) {
      result_written = 0;
    }
  }
  if (library != NULL) {
    dlclose(library);
  }
  return result.passed && result_written ? EXIT_SUCCESS : EXIT_FAILURE;
}
