add_library(NCNNProfileRuntime STATIC
    profile_runtime.c
    profile_json_writer.c
)
set_target_properties(NCNNProfileRuntime PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON)
target_compile_options(NCNNProfileRuntime PRIVATE -fvisibility=hidden)
