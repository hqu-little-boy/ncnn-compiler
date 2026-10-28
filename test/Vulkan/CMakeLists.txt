if (NCNN_ENABLE_VULKAN_PRODUCT OR NCNN_ENABLE_VULKAN_V0_TESTS)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(NCNN_VULKAN_FIXEDRUN_LEGACY_ROOT "" CACHE PATH
        "Fresh independently built pre-VA4 Vulkan compiler reference root")
    if(NOT NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT)
        set(NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT "${CMAKE_BINARY_DIR}")
    endif()
    if(NOT NCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT)
        set(NCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT
            "${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}/test/Vulkan/evidence/va2")
    endif()
    set(NCNN_VULKAN_DEVICE_HELPER_STANDALONE_ROOT
        "${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}/test/Vulkan/device-helper-standalone")
    set(NCNN_VULKAN_DEVICE_HELPER_PRODUCT_ONLY_ROOT
        "${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}/test/Vulkan/device-helper-product-only")
    add_subdirectory(native)

    add_test(NAME vulkan-device-helper-unit
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_device_helper.py
            --mode unit
            --unit-executable $<TARGET_FILE:vulkan_device_helper_unit_test>
            --stage-root ${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}
            --evidence-root ${NCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT})
    set_tests_properties(vulkan-device-helper-unit PROPERTIES
        LABELS "vulkan;unit;offline;device-helper"
        RUN_SERIAL TRUE
        TIMEOUT 120)

    add_test(NAME vulkan-device-helper-no-device
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_device_helper.py
            --mode no-device
            --runner $<TARGET_FILE:vulkan_device_helper_runner>
            --stage-root ${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}
            --evidence-root ${NCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT})
    set_tests_properties(vulkan-device-helper-no-device PROPERTIES
        LABELS "vulkan;unit;offline;device-helper;no-device"
        RUN_SERIAL TRUE
        TIMEOUT 60)

    set(_ncnn_vulkan_device_helper_independence_args
        --mode independence
        --stage-root ${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}
        --evidence-root ${NCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT}
        --source-root ${PROJECT_SOURCE_DIR}
        --standalone-cmake ${CMAKE_CURRENT_SOURCE_DIR}/native
        --standalone-build-root ${NCNN_VULKAN_DEVICE_HELPER_STANDALONE_ROOT}
        --product-only-root ${NCNN_VULKAN_DEVICE_HELPER_PRODUCT_ONLY_ROOT}
        --cmake ${CMAKE_COMMAND}
        --python ${Python3_EXECUTABLE}
        --evidence-driver ${CMAKE_CURRENT_SOURCE_DIR}/device_helper_evidence.py
        --jobs 2)
    foreach(_cmake_arg IN ITEMS
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
            "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
            "-DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}"
            "-DLLVM_DIR=${LLVM_DIR}"
            "-DMLIR_DIR=${MLIR_DIR}"
            "-DNCNN_LIT_PY=${NCNN_LIT_PY}"
            "-DNCNN_SANITIZER_OPENMP_LIBRARY_DIR=${NCNN_SANITIZER_OPENMP_LIBRARY_DIR}"
            "-DCOMPILER_USE_CCACHE=${COMPILER_USE_CCACHE}"
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
            "-DCOMPILER_ENABLE_FORMAT_TARGETS=OFF"
            "-DBUILD_TESTING=ON")
        list(APPEND _ncnn_vulkan_device_helper_independence_args
            --cmake-arg=${_cmake_arg})
    endforeach()
    if(COMPILER_SANITIZERS)
        string(JOIN "," _ncnn_helper_sanitizers_csv ${COMPILER_SANITIZERS})
        list(APPEND _ncnn_vulkan_device_helper_independence_args
            --sanitizers ${_ncnn_helper_sanitizers_csv})
    endif()
    if(NCNN_VULKAN_DEVICE_HELPER_SKIP_PRODUCT_ONLY_CHECK)
        list(APPEND _ncnn_vulkan_device_helper_independence_args
            --skip-product-only-check)
    endif()
    add_test(NAME vulkan-device-helper-independence
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_device_helper.py
            ${_ncnn_vulkan_device_helper_independence_args})
    set_tests_properties(vulkan-device-helper-independence PROPERTIES
        LABELS "vulkan;unit;offline;tooling;device-helper;independence"
        RUN_SERIAL TRUE
        TIMEOUT 1800)

    add_test(NAME vulkan-device-helper-evidence-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_device_helper.py
            --mode evidence-tests
            --stage-root ${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}
            --evidence-root ${NCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT}
            --evidence-test-script ${CMAKE_CURRENT_SOURCE_DIR}/test_device_helper_evidence.py)
    set_tests_properties(vulkan-device-helper-evidence-tests PROPERTIES
        LABELS "vulkan;unit;offline;tooling;device-helper;evidence"
        RUN_SERIAL TRUE
        TIMEOUT 120)

    add_test(NAME vulkan-v5-comparison-validation
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_run_v5.py)
    set_tests_properties(vulkan-v5-comparison-validation PROPERTIES
        LABELS "vulkan;unit;v5;offline"
        TIMEOUT 30)

    add_test(NAME vulkan-release-evidence-validation
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_release_evidence.py)
    set_tests_properties(vulkan-release-evidence-validation PROPERTIES
        LABELS "vulkan;unit;offline;tooling"
        TIMEOUT 60)

    foreach(_validation IN ITEMS va0-evidence va0-baseline model-dependency-evidence
            sanitizer-toolchain-inventory sanitizer-toolchain-v2)
        string(REPLACE "-" "_" _validation_module "${_validation}")
        add_test(NAME vulkan-${_validation}-validation
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_${_validation_module}.py)
        set_tests_properties(vulkan-${_validation}-validation PROPERTIES
            LABELS "vulkan;unit;offline;tooling"
            TIMEOUT 60)
    endforeach()

    add_test(NAME vulkan-gate-validation
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_run_gate.py)
    set_tests_properties(vulkan-gate-validation PROPERTIES
        LABELS "vulkan;unit;offline;tooling"
        TIMEOUT 60)
endif()

if(NCNN_ENABLE_VULKAN_PRODUCT)
    add_test(NAME vulkan-host-program-evidence-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_host_program_evidence.py)
    set_tests_properties(vulkan-host-program-evidence-tests PROPERTIES
        LABELS "vulkan;unit;offline;tooling;host-program"
        TIMEOUT 60)

    set(_ncnn_vulkan_offline_test_args)
    foreach(_dependency IN LISTS NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
        list(APPEND _ncnn_vulkan_offline_test_args
            --allowed-dependency=${_dependency})
    endforeach()
    foreach(_dependency IN LISTS NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
        list(APPEND _ncnn_vulkan_offline_test_args
            --sanitizer-dependency=${_dependency})
    endforeach()
    if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
        foreach(_cxx_arg IN LISTS NCNN_VULKAN_HOST_SANITIZER_ARGS)
            list(APPEND _ncnn_vulkan_offline_test_args --cxx-arg=${_cxx_arg})
        endforeach()
        list(APPEND _ncnn_vulkan_offline_test_args
            --sanitizer-runtime-dir ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
    endif()
    add_test(NAME ncnn-compile-vulkan-offline
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../Native/check_ncnn_compile_vulkan.py
            --offline
            --compiler $<TARGET_FILE:ncnn-compile>
            --driver $<TARGET_FILE:ncnn-mlir-driver>
            --opt $<TARGET_FILE:ncnn-mlir-opt>
            --mlir-opt ${NCNN_MLIR_OPT_EXECUTABLE}
            --spirv-val ${NCNN_SPIRV_VAL_EXECUTABLE}
            --spirv-dis ${NCNN_SPIRV_DIS_EXECUTABLE}
            --spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
            --clang ${CLANG}
            --nm ${LLVM_NM}
            --readelf ${LLVM_READELF}
            --host-cxx ${CMAKE_CXX_COMPILER}
            --vulkan-include-dir ${PROJECT_SOURCE_DIR}/include
            --loader-source ${CMAKE_CURRENT_SOURCE_DIR}/../Native/vulkan_product_loader.cpp
            --llvm-major ${LLVM_VERSION_MAJOR}
            --llvm-lib-dir ${LLVM_LIBRARY_DIR}
            ${_ncnn_vulkan_offline_test_args}
            --param ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
            --bin ${CMAKE_CURRENT_BINARY_DIR}/../Numerical/fixtures/empty.bin
            --unsupported-param ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/convolution.param
            --unsupported-bin ${CMAKE_CURRENT_BINARY_DIR}/../Numerical/fixtures/convolution.bin
            --work-dir ${CMAKE_CURRENT_BINARY_DIR}/../Native/vulkan-product-offline)
    set_tests_properties(ncnn-compile-vulkan-offline PROPERTIES
        LABELS "native;integration;cli;vulkan;product;offline"
        RUN_SERIAL TRUE
        TIMEOUT 900)

    add_test(NAME vulkan-fixedrun-capture-publish
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_fixedrun_capture_publish.py
            --capture-tool ${NCNN_VULKAN_FIXEDRUN_CAPTURE}
            --param ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
            --bin ${NCNN_VULKAN_FIXEDRUN_ROOT}/empty.bin
            --driver $<TARGET_FILE:ncnn-mlir-driver>
            --opt $<TARGET_FILE:ncnn-mlir-opt>
            --translate ${MLIR_TRANSLATE}
            --clang ${CLANG}
            --nm ${LLVM_NM}
            --readelf ${LLVM_READELF}
            --llvm-as ${LLVM_AS}
            --mlir-opt ${NCNN_MLIR_OPT_EXECUTABLE}
            --spirv-val ${NCNN_SPIRV_VAL_EXECUTABLE}
            --spirv-dis ${NCNN_SPIRV_DIS_EXECUTABLE}
            --spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
            --host-cxx ${CMAKE_CXX_COMPILER}
            --work-dir ${CMAKE_CURRENT_BINARY_DIR}/fixedrun-capture-publish)
    set_tests_properties(vulkan-fixedrun-capture-publish PROPERTIES
        LABELS "vulkan;unit;offline;integration;fixedrun;capture"
        RUN_SERIAL TRUE
        TIMEOUT 900)
    set(_ncnn_fixedrun_capture_publish_library_dirs)
    if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
        list(APPEND _ncnn_fixedrun_capture_publish_library_dirs
            ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
    endif()
    if(NCNN_VULKAN_USES_SHARED_LLVM)
        list(APPEND _ncnn_fixedrun_capture_publish_library_dirs
            ${LLVM_LIBRARY_DIR})
    endif()
    if(DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        list(APPEND _ncnn_fixedrun_capture_publish_library_dirs
            "$ENV{LD_LIBRARY_PATH}")
    endif()
    if(_ncnn_fixedrun_capture_publish_library_dirs)
        string(JOIN ":" _ncnn_fixedrun_capture_publish_library_path
            ${_ncnn_fixedrun_capture_publish_library_dirs})
        set_tests_properties(vulkan-fixedrun-capture-publish PROPERTIES
            ENVIRONMENT
                "LD_LIBRARY_PATH=${_ncnn_fixedrun_capture_publish_library_path}")
    endif()

    set(NCNN_VULKAN_FIXEDRUN_EVIDENCE_ROOT
        "${CMAKE_CURRENT_BINARY_DIR}/evidence/va4")
    add_test(NAME vulkan-fixedrun-evidence-validation
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_fixedrun_evidence.py)
    set_tests_properties(vulkan-fixedrun-evidence-validation PROPERTIES
        LABELS "vulkan;unit;offline;tooling;fixedrun;evidence"
        TIMEOUT 120)

    add_test(NAME vulkan-fixedrun-legacy-reference-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_fixedrun_legacy_reference.py)
    set_tests_properties(vulkan-fixedrun-legacy-reference-tests PROPERTIES
        LABELS "vulkan;unit;offline;tooling;fixedrun;legacy-reference"
        TIMEOUT 120)

    add_test(NAME vulkan-fixedrun-producer-tool-role-validation
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_run_fixedrun.py)
    set_tests_properties(vulkan-fixedrun-producer-tool-role-validation PROPERTIES
        LABELS "vulkan;unit;offline;tooling;fixedrun;producer"
        TIMEOUT 60)

    add_test(NAME vulkan-fixedrun-independence
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/fixedrun_evidence.py
            --mode independence
            --stage-root ${CMAKE_BINARY_DIR}
            --source-root ${PROJECT_SOURCE_DIR})
    set_tests_properties(vulkan-fixedrun-independence PROPERTIES
        LABELS "vulkan;unit;offline;tooling;fixedrun;independence"
        RUN_SERIAL TRUE
        TIMEOUT 180)

    # ── VA5: clean-directory deployment, reproducibility, failed-compile
    # protection.  The device profiles run the deployment loader against each
    # configured ICD; the offline profile proves loadability plus an explicit
    # create failure without a device.
    set(VA5_CONV_BIN ${CMAKE_CURRENT_BINARY_DIR}/va5-conv-weights.bin)
    add_custom_command(
        OUTPUT ${VA5_CONV_BIN}
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_fixedrun_conv_bin.py"
            "${VA5_CONV_BIN}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_fixedrun_conv_bin.py"
        VERBATIM)
    # Built with ALL so a plain `cmake --build` produces the weights before
    # the deployment tests run; CTest has no target-level dependency here.
    add_custom_target(va5_conv_weights_fixture ALL DEPENDS ${VA5_CONV_BIN})

    set(VA5_DEPLOYMENT_ARGS
        --compiler $<TARGET_FILE:ncnn-compile>
        --host-cxx ${CMAKE_CXX_COMPILER}
        --loader-source ${CMAKE_CURRENT_SOURCE_DIR}/../Native/vulkan_product_loader.cpp
        --vulkan-include-dir ${PROJECT_SOURCE_DIR}/include
        --param ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
        --conv-param ${CMAKE_CURRENT_SOURCE_DIR}/fixedrun_conv_relu.param
        --empty-bin ${NCNN_VULKAN_FIXEDRUN_ROOT}/empty.bin
        --conv-bin ${VA5_CONV_BIN})
    # The deployment loader must match the model instrumentation profile so
    # the sanitizer runtime initializes before the dlopen'd DSO pulls it in.
    foreach(_va5_cxx_arg IN LISTS NCNN_VULKAN_HOST_SANITIZER_ARGS)
        list(APPEND VA5_DEPLOYMENT_ARGS --cxx-arg=${_va5_cxx_arg})
    endforeach()
    if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
        list(APPEND VA5_DEPLOYMENT_ARGS
            --sanitizer-runtime-dir ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
    endif()
    if(NCNN_VULKAN_USES_SHARED_LLVM)
        list(APPEND VA5_DEPLOYMENT_ARGS --llvm-lib-dir ${LLVM_LIBRARY_DIR})
    endif()

    add_test(NAME vulkan-va5-deployment-no-device
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_va5_deployment.py
            ${VA5_DEPLOYMENT_ARGS}
            --work-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/va5/no-device-work
            --output ${CMAKE_CURRENT_BINARY_DIR}/evidence/va5/no-device/deployment-evidence.json)
    set_tests_properties(vulkan-va5-deployment-no-device PROPERTIES
        LABELS "vulkan;integration;va5;deployment;no-device"
        RUN_SERIAL TRUE
        TIMEOUT 900)

    add_test(NAME vulkan-va5-deployment-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_va5_deployment.py)
    set_tests_properties(vulkan-va5-deployment-tests PROPERTIES
        LABELS "vulkan;unit;offline;tooling;va5;deployment"
        TIMEOUT 120)

    set(NCNN_VULKAN_LIFECYCLE_EVIDENCE_ROOT
        "${CMAKE_CURRENT_BINARY_DIR}/evidence/va3")
    add_test(NAME vulkan-lifecycle-evidence-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_lifecycle_evidence.py
            --build-dir ${CMAKE_BINARY_DIR})
    set_tests_properties(vulkan-lifecycle-evidence-tests PROPERTIES
        LABELS "vulkan;unit;offline;tooling;lifecycle;evidence"
        RUN_SERIAL TRUE
        TIMEOUT 120)

    add_test(NAME vulkan-lifecycle-independence
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/lifecycle_evidence.py
            --mode independence
            --stage-root ${CMAKE_BINARY_DIR}
            --source-root ${PROJECT_SOURCE_DIR}
            --output ${NCNN_VULKAN_LIFECYCLE_EVIDENCE_ROOT}/independence/independence-evidence.json
            --nm ${LLVM_NM}
            --readelf ${LLVM_READELF}
            --ar ${CMAKE_AR})
    set_tests_properties(vulkan-lifecycle-independence PROPERTIES
        LABELS "vulkan;unit;offline;tooling;lifecycle;independence"
        RUN_SERIAL TRUE
        TIMEOUT 120)

    add_test(NAME vulkan-lifecycle-no-device
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/lifecycle_evidence.py
            --mode no-device
            --stage-root ${CMAKE_BINARY_DIR}
            --harness $<TARGET_FILE:vulkan_lifecycle_c_harness>
            --library ${CMAKE_CURRENT_BINARY_DIR}/native/lifecycle/weighted/libvulkan_lifecycle_weighted.so
            --result ${NCNN_VULKAN_LIFECYCLE_EVIDENCE_ROOT}/no-device/native-result.json
            --capture ${NCNN_VULKAN_LIFECYCLE_EVIDENCE_ROOT}/no-device/capture.json)
    set_tests_properties(vulkan-lifecycle-no-device PROPERTIES
        LABELS "vulkan;unit;offline;integration;lifecycle;c-abi;no-device"
        RUN_SERIAL TRUE
        TIMEOUT 120)

    # ── VA6: full support matrix and robustness migration.  The product
    # matrix drives the public CLI over negative/positive support rows; the
    # host-faults test executes the CLI-generated wrapper against the fake
    # Vulkan test double; both run without a device. ──
    set(VA6_MATRIX_ARGS
        --compiler $<TARGET_FILE:ncnn-compile>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/va6/matrix-work
        --output ${CMAKE_CURRENT_BINARY_DIR}/evidence/va6/matrix/product-matrix-evidence.json)
    if(NCNN_VULKAN_USES_SHARED_LLVM)
        list(APPEND VA6_MATRIX_ARGS --llvm-lib-dir ${LLVM_LIBRARY_DIR})
    endif()
    if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
        list(APPEND VA6_MATRIX_ARGS
            --sanitizer-runtime-dir ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
    endif()
    add_test(NAME vulkan-va6-product-matrix
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_va6_product_matrix.py
            ${VA6_MATRIX_ARGS})
    set_tests_properties(vulkan-va6-product-matrix PROPERTIES
        LABELS "vulkan;integration;offline;cli;va6;matrix"
        RUN_SERIAL TRUE
        TIMEOUT 900)
    if(_ncnn_fixedrun_capture_publish_library_dirs)
        set_tests_properties(vulkan-va6-product-matrix PROPERTIES
            ENVIRONMENT
                "LD_LIBRARY_PATH=${_ncnn_fixedrun_capture_publish_library_path}")
    endif()

    set(VA6_FAULTS_ARGS
        --compiler $<TARGET_FILE:ncnn-compile>
        --host-cxx ${CMAKE_CXX_COMPILER}
        --nm ${LLVM_NM}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/va6_faults.param
        --conv-bin ${VA5_CONV_BIN}
        --fault-test-source ${CMAKE_CURRENT_SOURCE_DIR}/native/va6_product_fault_test.cpp
        --fake-vulkan-source ${CMAKE_CURRENT_SOURCE_DIR}/native/fakeVulkan/fake_vulkan.cpp
        --helper-archive $<TARGET_FILE:NCNNVulkanDeviceHelper>
        --native-dir ${CMAKE_CURRENT_SOURCE_DIR}/native
        --project-include ${PROJECT_SOURCE_DIR}/include
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/va6/faults-work
        --output ${CMAKE_CURRENT_BINARY_DIR}/evidence/va6/faults/host-faults-evidence.json)
    foreach(_va6_cxx_arg IN LISTS NCNN_VULKAN_HOST_SANITIZER_ARGS)
        list(APPEND VA6_FAULTS_ARGS --cxx-arg=${_va6_cxx_arg})
    endforeach()
    if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
        list(APPEND VA6_FAULTS_ARGS
            --sanitizer-runtime-dir ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
    endif()
    if(NCNN_VULKAN_USES_SHARED_LLVM)
        list(APPEND VA6_FAULTS_ARGS --llvm-lib-dir ${LLVM_LIBRARY_DIR})
    endif()
    add_test(NAME vulkan-va6-host-faults
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_va6_host_faults.py
            ${VA6_FAULTS_ARGS})
    set_tests_properties(vulkan-va6-host-faults PROPERTIES
        LABELS "vulkan;integration;offline;va6;faults;fixedrun"
        RUN_SERIAL TRUE
        TIMEOUT 900)
    if(_ncnn_fixedrun_capture_publish_library_dirs)
        set_tests_properties(vulkan-va6-host-faults PROPERTIES
            ENVIRONMENT
                "LD_LIBRARY_PATH=${_ncnn_fixedrun_capture_publish_library_path}")
    endif()

    add_test(NAME vulkan-va6-evidence-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_va6_evidence.py)
    set_tests_properties(vulkan-va6-evidence-tests PROPERTIES
        LABELS "vulkan;unit;offline;tooling;va6;evidence"
        TIMEOUT 120)
endif()

if(NCNN_ENABLE_VULKAN_V0_TESTS)
    find_package(Vulkan REQUIRED)
    find_package(Threads REQUIRED)
    find_program(SPIRV_VAL NAMES spirv-val REQUIRED)
    find_program(SPIRV_DIS NAMES spirv-dis REQUIRED)
    find_program(VULKANINFO NAMES vulkaninfo REQUIRED)
    set(V0_LLVMPIPE_ICD
        "${CMAKE_CURRENT_BINARY_DIR}/unrequested-llvmpipe-icd.json")
    set(V0_NVIDIA_ICD
        "${CMAKE_CURRENT_BINARY_DIR}/unrequested-nvidia-icd.json")
    if("llvmpipe" IN_LIST NCNN_VULKAN_TEST_DEVICES)
        find_file(V0_LLVMPIPE_ICD_PATH
            NAMES lvp_icd.json
            PATHS /usr/share/vulkan/icd.d /etc/vulkan/icd.d
            REQUIRED)
        set(V0_LLVMPIPE_ICD "${V0_LLVMPIPE_ICD_PATH}")
    endif()
    if("nvidia" IN_LIST NCNN_VULKAN_TEST_DEVICES)
        find_file(V0_NVIDIA_SYSTEM_ICD
            NAMES nvidia_icd.json
            PATHS /usr/share/vulkan/icd.d /etc/vulkan/icd.d
            REQUIRED)
        find_library(V0_NVIDIA_EGL_LIBRARY
            NAMES EGL_nvidia libEGL_nvidia.so.0
            REQUIRED)
#NVIDIA supports the EGL frontend as a headless Vulkan ICD.The GLX
#frontend in driver 615.71.09 leaks a libdrm device on unload; EGL
#cleans up normally.Only the explicitly requested NVIDIA layer looks
#up this frontend.
        file(READ "${V0_NVIDIA_SYSTEM_ICD}" V0_NVIDIA_ICD_DOCUMENT)
        string(JSON V0_NVIDIA_ICD_DOCUMENT SET "${V0_NVIDIA_ICD_DOCUMENT}"
            ICD library_path "\"${V0_NVIDIA_EGL_LIBRARY}\"")
        set(V0_NVIDIA_ICD "${CMAKE_CURRENT_BINARY_DIR}/nvidia-egl-icd.json")
        file(WRITE "${V0_NVIDIA_ICD}" "${V0_NVIDIA_ICD_DOCUMENT}\n")
    endif()
    set(NCNN_VULKAN_DEVICE_LLVMPIPE_ICD "${V0_LLVMPIPE_ICD}" PARENT_SCOPE)
    set(NCNN_VULKAN_DEVICE_NVIDIA_ICD "${V0_NVIDIA_ICD}" PARENT_SCOPE)
    set(NCNN_VULKANINFO "${VULKANINFO}" PARENT_SCOPE)

    if(NCNN_ENABLE_VULKAN_PRODUCT)
        function(ncnn_add_vulkan_lifecycle_device_test device device_match icd)
            if(NOT device IN_LIST NCNN_VULKAN_TEST_DEVICES)
                return()
            endif()
            add_test(NAME vulkan-lifecycle-${device}
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/lifecycle_evidence.py
                    --mode device
                    --stage-root ${CMAKE_BINARY_DIR}
                    --device-match ${device_match}
                    --harness $<TARGET_FILE:vulkan_lifecycle_c_harness>
                    --library ${CMAKE_CURRENT_BINARY_DIR}/native/lifecycle/weighted/libvulkan_lifecycle_weighted_validation.so
                    --icd ${icd}
                    --result ${NCNN_VULKAN_LIFECYCLE_EVIDENCE_ROOT}/${device}/native-result.json
                    --capture ${NCNN_VULKAN_LIFECYCLE_EVIDENCE_ROOT}/${device}/capture.json
                    --vulkaninfo ${VULKANINFO})
            set_tests_properties(vulkan-lifecycle-${device} PROPERTIES
                LABELS "vulkan;device;integration;lifecycle;c-abi;${device}"
                RUN_SERIAL TRUE
                TIMEOUT 300)
        endfunction()
        ncnn_add_vulkan_lifecycle_device_test(nvidia NVIDIA ${V0_NVIDIA_ICD})
        ncnn_add_vulkan_lifecycle_device_test(llvmpipe llvmpipe ${V0_LLVMPIPE_ICD})

        if(COMPILER_SANITIZERS MATCHES "(^|;)(address|undefined)(;|$)")
            set(_ncnn_fixedrun_profile sanitize)
        else()
            set(_ncnn_fixedrun_profile release)
        endif()
        function(ncnn_add_vulkan_fixedrun_device_test device device_match icd)
            if(NOT device IN_LIST NCNN_VULKAN_TEST_DEVICES)
                return()
            endif()
            if(device STREQUAL "nvidia")
                set(device_layer hardware)
            else()
                set(device_layer software)
            endif()
            set(test_args
                --output ${NCNN_VULKAN_FIXEDRUN_EVIDENCE_ROOT}/${device}/fixedrun-raw.json
                --icd ${device}
                --profile ${_ncnn_fixedrun_profile}
                --icd-manifest ${icd}
                --source-root ${PROJECT_SOURCE_DIR}
                --build-root ${CMAKE_BINARY_DIR}
                --candidate-root ${NCNN_VULKAN_FIXEDRUN_ROOT}
                --capture-tool ${NCNN_VULKAN_FIXEDRUN_CAPTURE}
                --compiler $<TARGET_FILE:ncnn-compile>
                --loader-relu $<TARGET_FILE:vulkan_fixedrun_loader_relu>
                --loader-add-relu $<TARGET_FILE:vulkan_fixedrun_loader_add_relu>
                --loader-conv-relu $<TARGET_FILE:vulkan_fixedrun_loader_conv_relu>
                --loader-add-split-relu $<TARGET_FILE:vulkan_fixedrun_loader_add_split_relu>
                --loader-split-concat $<TARGET_FILE:vulkan_fixedrun_loader_split_concat>
                --loader-reshape $<TARGET_FILE:vulkan_fixedrun_loader_reshape>
                --loader-transpose $<TARGET_FILE:vulkan_fixedrun_loader_transpose>
                --driver $<TARGET_FILE:ncnn-mlir-driver>
                --opt $<TARGET_FILE:ncnn-mlir-opt>
                --translate ${MLIR_TRANSLATE}
                --clang ${CLANG}
                --nm ${LLVM_NM}
                --readelf ${LLVM_READELF}
                --llvm-as ${LLVM_AS}
                --mlir-opt ${NCNN_MLIR_OPT_EXECUTABLE}
                --spirv-val ${NCNN_SPIRV_VAL_EXECUTABLE}
                --spirv-dis ${NCNN_SPIRV_DIS_EXECUTABLE}
                --spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
                --host-cxx ${CMAKE_CXX_COMPILER}
                --vulkaninfo ${VULKANINFO})
            if(NCNN_VULKAN_FIXEDRUN_LEGACY_ROOT)
                list(APPEND test_args --legacy-root ${NCNN_VULKAN_FIXEDRUN_LEGACY_ROOT})
            endif()
            foreach(_cxx_arg IN LISTS NCNN_VULKAN_HOST_SANITIZER_ARGS)
                list(APPEND test_args --cxx-arg=${_cxx_arg})
            endforeach()
            if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
                list(APPEND test_args
                    --sanitizer-runtime-dir ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
            endif()
            add_test(NAME vulkan-fixedrun-${device}
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_fixedrun.py
                    ${test_args})
            set_tests_properties(vulkan-fixedrun-${device} PROPERTIES
                LABELS "vulkan;device;integration;fixedrun;${device_layer}"
                RUN_SERIAL TRUE
                TIMEOUT 900)
        endfunction()
        ncnn_add_vulkan_fixedrun_device_test(nvidia NVIDIA ${V0_NVIDIA_ICD})
        ncnn_add_vulkan_fixedrun_device_test(llvmpipe llvmpipe ${V0_LLVMPIPE_ICD})

        function(ncnn_add_va5_deployment_device_test device icd)
            if(NOT device IN_LIST NCNN_VULKAN_TEST_DEVICES)
                return()
            endif()
            add_test(NAME vulkan-va5-deployment-${device}
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_va5_deployment.py
                    ${VA5_DEPLOYMENT_ARGS}
                    --device
                    --icd ${icd}
                    --work-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/va5/${device}-work
                    --output ${CMAKE_CURRENT_BINARY_DIR}/evidence/va5/${device}/deployment-evidence.json)
            set_tests_properties(vulkan-va5-deployment-${device} PROPERTIES
                LABELS "vulkan;device;integration;va5;deployment;${device}"
                RUN_SERIAL TRUE
                TIMEOUT 900)
        endfunction()
        ncnn_add_va5_deployment_device_test(nvidia ${V0_NVIDIA_ICD})
        ncnn_add_va5_deployment_device_test(llvmpipe ${V0_LLVMPIPE_ICD})

        # VA6.3: one fixed candidate DSO per fixture runs under every
        # configured ICD plus the no-ICD environment; the scenario needs at
        # least two ICDs, so register only when both devices are enabled.
        if("nvidia" IN_LIST NCNN_VULKAN_TEST_DEVICES AND
           "llvmpipe" IN_LIST NCNN_VULKAN_TEST_DEVICES)
            set(VA6_CROSS_ICD_ARGS
                --compiler $<TARGET_FILE:ncnn-compile>
                --host-cxx ${CMAKE_CXX_COMPILER}
                --loader-source ${CMAKE_CURRENT_SOURCE_DIR}/native/va6_cross_icd_loader.cpp
                --vulkan-include-dir ${PROJECT_SOURCE_DIR}/include
                --vulkaninfo ${VULKANINFO}
                --icd nvidia=${V0_NVIDIA_ICD}:NVIDIA
                --icd llvmpipe=${V0_LLVMPIPE_ICD}:llvmpipe
                --param ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
                --conv-param ${CMAKE_CURRENT_SOURCE_DIR}/fixedrun_conv_relu.param
                --empty-bin ${NCNN_VULKAN_FIXEDRUN_ROOT}/empty.bin
                --conv-bin ${VA5_CONV_BIN}
                --work-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/va6/cross-icd-work
                --output ${CMAKE_CURRENT_BINARY_DIR}/evidence/va6/cross-icd/cross-icd-evidence.json)
            foreach(_va6_cxx_arg IN LISTS NCNN_VULKAN_HOST_SANITIZER_ARGS)
                list(APPEND VA6_CROSS_ICD_ARGS --cxx-arg=${_va6_cxx_arg})
            endforeach()
            if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
                list(APPEND VA6_CROSS_ICD_ARGS
                    --sanitizer-runtime-dir ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
            endif()
            if(NCNN_VULKAN_USES_SHARED_LLVM)
                list(APPEND VA6_CROSS_ICD_ARGS --llvm-lib-dir ${LLVM_LIBRARY_DIR})
            endif()
            add_test(NAME vulkan-va6-cross-icd
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_va6_cross_icd.py
                    ${VA6_CROSS_ICD_ARGS})
            set_tests_properties(vulkan-va6-cross-icd PROPERTIES
                LABELS "vulkan;device;integration;va6;cross-icd"
                RUN_SERIAL TRUE
                TIMEOUT 900)
        endif()
    endif()

add_executable(vulkan_add_runner vulkan_add_runner.cpp)
target_link_libraries(vulkan_add_runner PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
target_compile_options(vulkan_add_runner PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_executable(vulkan_runtime_runner
    vulkan_runtime_runner.cpp
    vulkan_runtime_abi_fixture.cpp)
target_link_libraries(vulkan_runtime_runner PRIVATE
    NCNNVulkanRuntime ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(vulkan_runtime_runner PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_executable(vulkan_runtime_weight_runner vulkan_runtime_weight_runner.cpp)
target_link_libraries(vulkan_runtime_weight_runner PRIVATE
    NCNNVulkanRuntime ${CMAKE_DL_LIBS})
target_compile_options(vulkan_runtime_weight_runner PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_executable(vulkan_v0_spirv_extract extract_vulkan_spirv.cpp)
target_include_directories(vulkan_v0_spirv_extract SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})
target_compile_options(vulkan_v0_spirv_extract PRIVATE -fno-rtti)
target_link_libraries(vulkan_v0_spirv_extract PRIVATE
    MLIRGPUDialect MLIRSPIRVDialect MLIRParser MLIRIR MLIRSupport LLVMSupport)
set_target_properties(vulkan_v0_spirv_extract PROPERTIES
    BUILD_RPATH "${LLVM_LIBRARY_DIR}")

set(V0_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-add)
set(V0_CPU_LIBRARY ${V0_CPU_DIR}/libv0_add.so)
set(V0_CPU_HEADER ${V0_CPU_DIR}/v0_add.h)
set(V0_CPU_MANIFEST ${V0_CPU_DIR}/v0_add.json)
set(V0_EMPTY_BIN ${CMAKE_CURRENT_BINARY_DIR}/empty.bin)
file(WRITE ${V0_EMPTY_BIN} "")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/generate_bins.py
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_conv1x1_bin.py)
add_custom_command(
    OUTPUT ${V0_CPU_LIBRARY} ${V0_CPU_HEADER} ${V0_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V0_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/add.param
        --bin ${V0_EMPTY_BIN}
        --model-name v0_add
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V0_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/add.param
    VERBATIM)
add_custom_target(v0_cpu_add_fixture
    DEPENDS ${V0_CPU_LIBRARY} ${V0_CPU_HEADER} ${V0_CPU_MANIFEST})
add_custom_target(vulkan-v0-build
    DEPENDS vulkan_add_runner vulkan_v0_spirv_extract v0_cpu_add_fixture)

function(ncnn_add_vulkan_device_helper_add_test device)
    if(NOT device IN_LIST NCNN_VULKAN_TEST_DEVICES)
        return()
    endif()
    if(device STREQUAL "nvidia")
        set(device_match NVIDIA)
        set(device_icd ${V0_NVIDIA_ICD})
        set(device_layer hardware)
    else()
        set(device_match llvmpipe)
        set(device_icd ${V0_LLVMPIPE_ICD})
        set(device_layer software)
    endif()
    add_test(NAME vulkan-device-helper-add-${device}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_device_helper.py
            --mode add
            --runner $<TARGET_FILE:vulkan_device_helper_runner>
            --device ${device}
            --device-match ${device_match}
            --icd ${device_icd}
            --mlir-opt ${MLIR_OPT}
            --spirv-extract $<TARGET_FILE:vulkan_v0_spirv_extract>
            --spirv-val ${SPIRV_VAL}
            --spirv-dis ${SPIRV_DIS}
            --kernel ${CMAKE_CURRENT_SOURCE_DIR}/gpu_func_add_probe.mlir
            --cpu-library ${V0_CPU_LIBRARY}
            --cpu-symbol v0_add
            --stage-root ${NCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT}
            --evidence-root ${NCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT}
            --validation)
    set_tests_properties(vulkan-device-helper-add-${device} PROPERTIES
        LABELS "vulkan;device;integration;device-helper;${device_layer}"
        RUN_SERIAL TRUE
        TIMEOUT 300)
endfunction()

ncnn_add_vulkan_device_helper_add_test(nvidia)
ncnn_add_vulkan_device_helper_add_test(llvmpipe)

set(V2_RELU_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-relu)
set(V2_RELU_CPU_LIBRARY ${V2_RELU_CPU_DIR}/libv2_relu.so)
set(V2_RELU_CPU_HEADER ${V2_RELU_CPU_DIR}/v2_relu.h)
set(V2_RELU_CPU_MANIFEST ${V2_RELU_CPU_DIR}/v2_relu.json)
add_custom_command(
    OUTPUT ${V2_RELU_CPU_LIBRARY} ${V2_RELU_CPU_HEADER} ${V2_RELU_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_RELU_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_relu
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_RELU_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
    VERBATIM)
add_custom_target(v2_cpu_relu_fixture
    DEPENDS ${V2_RELU_CPU_LIBRARY} ${V2_RELU_CPU_HEADER} ${V2_RELU_CPU_MANIFEST})

set(V2_IDENTITY_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-identity)
set(V2_IDENTITY_CPU_LIBRARY ${V2_IDENTITY_CPU_DIR}/libv2_identity.so)
set(V2_IDENTITY_CPU_HEADER ${V2_IDENTITY_CPU_DIR}/v2_identity.h)
set(V2_IDENTITY_CPU_MANIFEST ${V2_IDENTITY_CPU_DIR}/v2_identity.json)
add_custom_command(
    OUTPUT ${V2_IDENTITY_CPU_LIBRARY} ${V2_IDENTITY_CPU_HEADER} ${V2_IDENTITY_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_IDENTITY_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/identity.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_identity
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_IDENTITY_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/identity.param
    VERBATIM)
add_custom_target(v2_cpu_identity_fixture
    DEPENDS ${V2_IDENTITY_CPU_LIBRARY} ${V2_IDENTITY_CPU_HEADER} ${V2_IDENTITY_CPU_MANIFEST})

set(V2_TRANSPOSE_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-transpose)
set(V2_TRANSPOSE_CPU_LIBRARY ${V2_TRANSPOSE_CPU_DIR}/libv2_transpose.so)
set(V2_TRANSPOSE_CPU_HEADER ${V2_TRANSPOSE_CPU_DIR}/v2_transpose.h)
set(V2_TRANSPOSE_CPU_MANIFEST ${V2_TRANSPOSE_CPU_DIR}/v2_transpose.json)
add_custom_command(
    OUTPUT ${V2_TRANSPOSE_CPU_LIBRARY} ${V2_TRANSPOSE_CPU_HEADER} ${V2_TRANSPOSE_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_TRANSPOSE_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/transpose.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_transpose
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_TRANSPOSE_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/transpose.param
    VERBATIM)
add_custom_target(v2_cpu_transpose_fixture
    DEPENDS ${V2_TRANSPOSE_CPU_LIBRARY} ${V2_TRANSPOSE_CPU_HEADER} ${V2_TRANSPOSE_CPU_MANIFEST})

set(V2_RESHAPE_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-reshape)
set(V2_RESHAPE_CPU_LIBRARY ${V2_RESHAPE_CPU_DIR}/libv2_reshape.so)
set(V2_RESHAPE_CPU_HEADER ${V2_RESHAPE_CPU_DIR}/v2_reshape.h)
set(V2_RESHAPE_CPU_MANIFEST ${V2_RESHAPE_CPU_DIR}/v2_reshape.json)
add_custom_command(
    OUTPUT ${V2_RESHAPE_CPU_LIBRARY} ${V2_RESHAPE_CPU_HEADER} ${V2_RESHAPE_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_RESHAPE_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/reshape.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_reshape
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_RESHAPE_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/reshape.param
    VERBATIM)
add_custom_target(v2_cpu_reshape_fixture
    DEPENDS ${V2_RESHAPE_CPU_LIBRARY} ${V2_RESHAPE_CPU_HEADER} ${V2_RESHAPE_CPU_MANIFEST})

set(V2_CONV3X3_BIN ${CMAKE_CURRENT_BINARY_DIR}/convolution-3x3.bin)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/generate_bins.py
        ${V2_CONV3X3_BIN} convolution
    COMMAND_ERROR_IS_FATAL ANY)
set(V2_CONV3X3_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/convolution.param)
set(V2_CONV3X3_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-3x3)
set(V2_CONV3X3_CPU_LIBRARY ${V2_CONV3X3_CPU_DIR}/libv2_conv3x3.so)
set(V2_CONV3X3_CPU_HEADER ${V2_CONV3X3_CPU_DIR}/v2_conv3x3.h)
set(V2_CONV3X3_CPU_MANIFEST ${V2_CONV3X3_CPU_DIR}/v2_conv3x3.json)
add_custom_command(
    OUTPUT ${V2_CONV3X3_CPU_LIBRARY} ${V2_CONV3X3_CPU_HEADER} ${V2_CONV3X3_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV3X3_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV3X3_PARAM}
        --bin ${V2_CONV3X3_BIN}
        --model-name v2_conv3x3
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV3X3_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV3X3_PARAM} ${V2_CONV3X3_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv3x3_fixture
    DEPENDS ${V2_CONV3X3_CPU_LIBRARY} ${V2_CONV3X3_CPU_HEADER}
            ${V2_CONV3X3_CPU_MANIFEST})

set(V2_CONV3X3_NO_BIAS_BIN ${CMAKE_CURRENT_BINARY_DIR}/convolution-3x3-no-bias.bin)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/generate_bins.py
        ${V2_CONV3X3_NO_BIAS_BIN} convolution_no_bias
    COMMAND_ERROR_IS_FATAL ANY)
set(V2_CONV3X3_NO_BIAS_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/convolution_no_bias.param)
set(V2_CONV3X3_NO_BIAS_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-3x3-no-bias)
set(V2_CONV3X3_NO_BIAS_CPU_LIBRARY ${V2_CONV3X3_NO_BIAS_CPU_DIR}/libv2_conv3x3_no_bias.so)
set(V2_CONV3X3_NO_BIAS_CPU_HEADER ${V2_CONV3X3_NO_BIAS_CPU_DIR}/v2_conv3x3_no_bias.h)
set(V2_CONV3X3_NO_BIAS_CPU_MANIFEST ${V2_CONV3X3_NO_BIAS_CPU_DIR}/v2_conv3x3_no_bias.json)
add_custom_command(
    OUTPUT ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY} ${V2_CONV3X3_NO_BIAS_CPU_HEADER} ${V2_CONV3X3_NO_BIAS_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV3X3_NO_BIAS_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV3X3_NO_BIAS_PARAM}
        --bin ${V2_CONV3X3_NO_BIAS_BIN}
        --model-name v2_conv3x3_no_bias
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV3X3_NO_BIAS_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV3X3_NO_BIAS_PARAM} ${V2_CONV3X3_NO_BIAS_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv3x3_no_bias_fixture
    DEPENDS ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY} ${V2_CONV3X3_NO_BIAS_CPU_HEADER}
            ${V2_CONV3X3_NO_BIAS_CPU_MANIFEST})

set(V2_CONV3X3_STRIDE2_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/convolution_3x3_stride2.param)
set(V2_CONV3X3_STRIDE2_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-3x3-stride2)
set(V2_CONV3X3_STRIDE2_CPU_LIBRARY ${V2_CONV3X3_STRIDE2_CPU_DIR}/libv2_conv3x3_stride2.so)
set(V2_CONV3X3_STRIDE2_CPU_HEADER ${V2_CONV3X3_STRIDE2_CPU_DIR}/v2_conv3x3_stride2.h)
set(V2_CONV3X3_STRIDE2_CPU_MANIFEST ${V2_CONV3X3_STRIDE2_CPU_DIR}/v2_conv3x3_stride2.json)
add_custom_command(
    OUTPUT ${V2_CONV3X3_STRIDE2_CPU_LIBRARY} ${V2_CONV3X3_STRIDE2_CPU_HEADER} ${V2_CONV3X3_STRIDE2_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV3X3_STRIDE2_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV3X3_STRIDE2_PARAM}
        --bin ${V2_CONV3X3_BIN}
        --model-name v2_conv3x3_stride2
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV3X3_STRIDE2_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV3X3_STRIDE2_PARAM} ${V2_CONV3X3_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv3x3_stride2_fixture
    DEPENDS ${V2_CONV3X3_STRIDE2_CPU_LIBRARY} ${V2_CONV3X3_STRIDE2_CPU_HEADER}
            ${V2_CONV3X3_STRIDE2_CPU_MANIFEST})

set(V2_CONV1X1_BIN ${CMAKE_CURRENT_BINARY_DIR}/convolution-1x1.bin)
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_conv1x1_bin.py
            ${V2_CONV1X1_BIN}
    COMMAND_ERROR_IS_FATAL ANY)
set(V2_CONV1X1_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/convolution_1x1.param)
set(V2_CONV1X1_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-1x1)
set(V2_CONV1X1_CPU_LIBRARY ${V2_CONV1X1_CPU_DIR}/libv2_conv1x1.so)
set(V2_CONV1X1_CPU_HEADER ${V2_CONV1X1_CPU_DIR}/v2_conv1x1.h)
set(V2_CONV1X1_CPU_MANIFEST ${V2_CONV1X1_CPU_DIR}/v2_conv1x1.json)
add_custom_command(
    OUTPUT ${V2_CONV1X1_CPU_LIBRARY} ${V2_CONV1X1_CPU_HEADER} ${V2_CONV1X1_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV1X1_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV1X1_PARAM}
        --bin ${V2_CONV1X1_BIN}
        --model-name v2_conv1x1
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV1X1_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV1X1_PARAM} ${V2_CONV1X1_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv1x1_fixture
    DEPENDS ${V2_CONV1X1_CPU_LIBRARY} ${V2_CONV1X1_CPU_HEADER}
            ${V2_CONV1X1_CPU_MANIFEST})

add_custom_target(vulkan-v2-build ALL
    DEPENDS vulkan-v0-build v2_cpu_relu_fixture v2_cpu_identity_fixture
            v2_cpu_transpose_fixture v2_cpu_reshape_fixture
            v2_cpu_conv1x1_fixture v2_cpu_conv3x3_fixture
            v2_cpu_conv3x3_no_bias_fixture v2_cpu_conv3x3_stride2_fixture)

set(V3_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-add-relu)
set(V3_CPU_LIBRARY ${V3_CPU_DIR}/libv3_add_relu.so)
set(V3_CPU_HEADER ${V3_CPU_DIR}/v3_add_relu.h)
set(V3_CPU_MANIFEST ${V3_CPU_DIR}/v3_add_relu.json)
add_custom_command(
    OUTPUT ${V3_CPU_LIBRARY} ${V3_CPU_HEADER} ${V3_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V3_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/add_relu.param
        --bin ${V0_EMPTY_BIN}
        --model-name v3_add_relu
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V3_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${CMAKE_CURRENT_SOURCE_DIR}/add_relu.param ${V0_EMPTY_BIN}
    VERBATIM)
add_custom_target(v3_cpu_add_relu_fixture
    DEPENDS ${V3_CPU_LIBRARY} ${V3_CPU_HEADER} ${V3_CPU_MANIFEST})
add_custom_target(vulkan-v3-build ALL
    DEPENDS vulkan-v2-build vulkan_runtime_runner
            vulkan_runtime_weight_runner v3_cpu_add_relu_fixture)

set(V6_DUAL_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-add-split-relu)
set(V6_DUAL_CPU_LIBRARY ${V6_DUAL_CPU_DIR}/libv6_add_split_relu.so)
set(V6_DUAL_CPU_HEADER ${V6_DUAL_CPU_DIR}/v6_add_split_relu.h)
set(V6_DUAL_CPU_MANIFEST ${V6_DUAL_CPU_DIR}/v6_add_split_relu.json)
add_custom_command(
    OUTPUT ${V6_DUAL_CPU_LIBRARY} ${V6_DUAL_CPU_HEADER} ${V6_DUAL_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V6_DUAL_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/add_split_relu.param
        --bin ${V0_EMPTY_BIN}
        --model-name v6_add_split_relu
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V6_DUAL_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${CMAKE_CURRENT_SOURCE_DIR}/add_split_relu.param ${V0_EMPTY_BIN}
    VERBATIM)
add_custom_target(v6_cpu_add_split_relu_fixture
    DEPENDS ${V6_DUAL_CPU_LIBRARY} ${V6_DUAL_CPU_HEADER} ${V6_DUAL_CPU_MANIFEST})
target_include_directories(vulkan_runtime_runner PRIVATE ${V6_DUAL_CPU_DIR})
add_dependencies(vulkan_runtime_runner v6_cpu_add_split_relu_fixture)
add_dependencies(vulkan-v3-build v6_cpu_add_split_relu_fixture)

function(ncnn_add_vulkan_v0_test test_name device_match output_name)
    if(device_match STREQUAL "NVIDIA")
        set(device_name nvidia)
        set(device_layer hardware)
        set(device_icd ${V0_NVIDIA_ICD})
    else()
        set(device_name llvmpipe)
        set(device_layer software)
        set(device_icd ${V0_LLVMPIPE_ICD})
    endif()
    if(NOT device_name IN_LIST NCNN_VULKAN_TEST_DEVICES)
        return()
    endif()
    set(test_args
        --mlir-opt ${MLIR_OPT}
        --spirv-val ${SPIRV_VAL}
        --spirv-dis ${SPIRV_DIS}
        --vulkaninfo ${VULKANINFO}
        --kernel ${CMAKE_CURRENT_SOURCE_DIR}/gpu_func_add_probe.mlir
        --spirv-extract $<TARGET_FILE:vulkan_v0_spirv_extract>
        --runner $<TARGET_FILE:vulkan_add_runner>
        --cpu-library ${V0_CPU_LIBRARY}
        --cpu-symbol v0_add
        --device-match ${device_match}
        --icd ${device_icd}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/gpu-func/${output_name}
        --validation)
    if(ARGC GREATER 3)
        list(APPEND test_args ${ARGN})
    endif()
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_v0.py
            ${test_args})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "vulkan;device;integration;${device_layer}"
        RUN_SERIAL TRUE
        TIMEOUT 180)
endfunction()

ncnn_add_vulkan_v0_test(vulkan-v0-add-nvidia NVIDIA nvidia --run-negative)
ncnn_add_vulkan_v0_test(vulkan-v0-add-llvmpipe llvmpipe llvmpipe)

function(ncnn_add_vulkan_v2_test test_name device_match output_name operation
         kernel_file binary_symbol cpu_library cpu_symbol model_param model_bin)
    if(device_match STREQUAL "NVIDIA")
        set(device_name nvidia)
        set(device_layer hardware)
        set(device_icd ${V0_NVIDIA_ICD})
    else()
        set(device_name llvmpipe)
        set(device_layer software)
        set(device_icd ${V0_LLVMPIPE_ICD})
    endif()
    if(NOT device_name IN_LIST NCNN_VULKAN_TEST_DEVICES)
        return()
    endif()
    set(test_args
        --ncnn-driver $<TARGET_FILE:ncnn-mlir-driver>
        --ncnn-mlir-opt $<TARGET_FILE:ncnn-mlir-opt>
        --mlir-opt ${MLIR_OPT}
        --spirv-val ${SPIRV_VAL}
        --spirv-dis ${SPIRV_DIS}
        --vulkaninfo ${VULKANINFO}
        --spirv-extract $<TARGET_FILE:vulkan_v0_spirv_extract>
        --runner $<TARGET_FILE:vulkan_add_runner>
        --operation ${operation}
        --elements 257
        --binary-symbol ${binary_symbol}
        --cpu-library ${cpu_library}
        --cpu-symbol ${cpu_symbol}
        --model-param ${CMAKE_CURRENT_SOURCE_DIR}/${model_param}
        --model-bin ${model_bin}
        --device-match ${device_match}
        --icd ${device_icd}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/v2/${output_name}
        --validation)
    if(NOT kernel_file STREQUAL "from-model")
        list(APPEND test_args
            --kernel ${CMAKE_CURRENT_SOURCE_DIR}/../Conversion/NCNNToVulkan/${kernel_file})
    endif()
    if(ARGC GREATER 10)
        list(APPEND test_args ${ARGN})
    endif()
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_v2.py
            ${test_args})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "vulkan;device;integration;v2;${device_layer}"
        RUN_SERIAL TRUE
        TIMEOUT 180)
endfunction()

function(ncnn_add_vulkan_v3_runtime_test test_name device_match output_name)
    if(device_match STREQUAL "NVIDIA")
        set(device_name nvidia)
        set(device_layer hardware)
        set(device_icd ${V0_NVIDIA_ICD})
    else()
        set(device_name llvmpipe)
        set(device_layer software)
        set(device_icd ${V0_LLVMPIPE_ICD})
    endif()
    if(NOT device_name IN_LIST NCNN_VULKAN_TEST_DEVICES)
        return()
    endif()
    set(test_args
        --ncnn-driver $<TARGET_FILE:ncnn-mlir-driver>
        --ncnn-mlir-opt $<TARGET_FILE:ncnn-mlir-opt>
        --mlir-opt ${MLIR_OPT}
        --spirv-val ${SPIRV_VAL}
        --spirv-dis ${SPIRV_DIS}
        --vulkaninfo ${VULKANINFO}
        --spirv-extract $<TARGET_FILE:vulkan_v0_spirv_extract>
        --v2-runner $<TARGET_FILE:vulkan_add_runner>
        --runtime-runner $<TARGET_FILE:vulkan_runtime_runner>
        --weight-runner $<TARGET_FILE:vulkan_runtime_weight_runner>
        --conv-cpu-library ${V2_CONV3X3_CPU_LIBRARY}
        --conv-model-param ${V2_CONV3X3_PARAM}
        --conv-model-bin ${V2_CONV3X3_BIN}
        --cpu-library ${V3_CPU_LIBRARY}
        --cpu-symbol v3_add_relu
        --add-cpu-library ${V0_CPU_LIBRARY}
        --add-cpu-symbol v0_add
        --relu-cpu-library ${V2_RELU_CPU_LIBRARY}
        --relu-cpu-symbol v2_relu
        --empty-bin ${V0_EMPTY_BIN}
        --device-match ${device_match}
        --icd ${device_icd}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/v3/${output_name}
        --validation)
    if(ARGC GREATER 3)
        list(APPEND test_args ${ARGN})
    endif()
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_v3.py
            ${test_args})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "vulkan;device;integration;v3;${device_layer}"
        RUN_SERIAL TRUE
        TIMEOUT 300
        ENVIRONMENT
            "NCNN_V6_DUAL_CPU_LIBRARY=${V6_DUAL_CPU_LIBRARY};NCNN_V6_DUAL_CPU_SYMBOL=v6_add_split_relu")
endfunction()

ncnn_add_vulkan_v3_runtime_test(
    vulkan-v3-runtime-nvidia NVIDIA nvidia )
ncnn_add_vulkan_v3_runtime_test(
    vulkan-v3-runtime-llvmpipe llvmpipe llvmpipe )

ncnn_add_vulkan_v2_test(
    vulkan-v2-add-nvidia NVIDIA add-nvidia add from-model model_vulkan
    ${V0_CPU_LIBRARY} v0_add add.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-add-llvmpipe llvmpipe add-llvmpipe add from-model model_vulkan
    ${V0_CPU_LIBRARY} v0_add add.param ${V0_EMPTY_BIN} )
ncnn_add_vulkan_v2_test(
    vulkan-v2-relu-nvidia NVIDIA relu-nvidia relu from-model model_vulkan
    ${V2_RELU_CPU_LIBRARY} v2_relu relu.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-relu-llvmpipe llvmpipe relu-llvmpipe relu from-model model_vulkan
    ${V2_RELU_CPU_LIBRARY} v2_relu relu.param ${V0_EMPTY_BIN}
    )
ncnn_add_vulkan_v2_test(
    vulkan-v2-copy-nvidia NVIDIA copy-nvidia copy lower-copy.mlir copy_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-copy-llvmpipe llvmpipe copy-llvmpipe copy lower-copy.mlir copy_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-identity-nvidia NVIDIA identity-nvidia identity from-model model_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-identity-llvmpipe llvmpipe identity-llvmpipe identity from-model model_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-transpose-nvidia NVIDIA transpose-nvidia transpose from-model model_vulkan
    ${V2_TRANSPOSE_CPU_LIBRARY} v2_transpose transpose.param ${V0_EMPTY_BIN}
    --elements 255 --input-shape 51x5x1 --permutation 1,0,2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-transpose-llvmpipe llvmpipe transpose-llvmpipe transpose from-model model_vulkan
    ${V2_TRANSPOSE_CPU_LIBRARY} v2_transpose transpose.param ${V0_EMPTY_BIN}
    --elements 255
    --input-shape 51x5x1 --permutation 1,0,2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-reshape-nvidia NVIDIA reshape-nvidia reshape from-model model_vulkan
    ${V2_RESHAPE_CPU_LIBRARY} v2_reshape ../Numerical/fixtures/reshape.param
    ${V0_EMPTY_BIN} --elements 8)
ncnn_add_vulkan_v2_test(
    vulkan-v2-reshape-llvmpipe llvmpipe reshape-llvmpipe reshape from-model model_vulkan
    ${V2_RESHAPE_CPU_LIBRARY} v2_reshape ../Numerical/fixtures/reshape.param
    ${V0_EMPTY_BIN} --elements 8)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-nvidia NVIDIA conv3x3-nvidia conv2d from-model model_vulkan
    ${V2_CONV3X3_CPU_LIBRARY} v2_conv3x3 ../Numerical/fixtures/convolution.param
    ${V2_CONV3X3_BIN} --elements 32 --input-shape 1x4x4
    --kernel-size 3x3 --stride 1x1 --padding 1,1,1,1
    --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-llvmpipe llvmpipe conv3x3-llvmpipe conv2d from-model model_vulkan
    ${V2_CONV3X3_CPU_LIBRARY} v2_conv3x3 ../Numerical/fixtures/convolution.param
    ${V2_CONV3X3_BIN} --elements 32
    --input-shape 1x4x4 --kernel-size 3x3 --stride 1x1
    --padding 1,1,1,1 --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-no-bias-nvidia NVIDIA conv3x3-no-bias-nvidia conv2d
    from-model model_vulkan ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY} v2_conv3x3_no_bias
    ../Numerical/fixtures/convolution_no_bias.param ${V2_CONV3X3_NO_BIAS_BIN}
    --elements 32 --input-shape 1x4x4 --kernel-size 3x3 --stride 1x1
    --padding 1,1,1,1 --output-channels 2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-no-bias-llvmpipe llvmpipe conv3x3-no-bias-llvmpipe
    conv2d from-model model_vulkan ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY}
    v2_conv3x3_no_bias ../Numerical/fixtures/convolution_no_bias.param
    ${V2_CONV3X3_NO_BIAS_BIN} --elements 32
    --input-shape 1x4x4 --kernel-size 3x3 --stride 1x1
    --padding 1,1,1,1 --output-channels 2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-stride2-nvidia NVIDIA conv3x3-stride2-nvidia conv2d
    from-model model_vulkan ${V2_CONV3X3_STRIDE2_CPU_LIBRARY} v2_conv3x3_stride2
    convolution_3x3_stride2.param ${V2_CONV3X3_BIN} --elements 8
    --input-shape 1x4x4 --kernel-size 3x3 --stride 2x2
    --padding 1,1,1,1 --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-stride2-llvmpipe llvmpipe conv3x3-stride2-llvmpipe
    conv2d from-model model_vulkan ${V2_CONV3X3_STRIDE2_CPU_LIBRARY}
    v2_conv3x3_stride2 convolution_3x3_stride2.param ${V2_CONV3X3_BIN}
    --elements 8 --input-shape 1x4x4
    --kernel-size 3x3 --stride 2x2 --padding 1,1,1,1
    --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv1x1-nvidia NVIDIA conv1x1-nvidia conv2d from-model model_vulkan
    ${V2_CONV1X1_CPU_LIBRARY} v2_conv1x1 convolution_1x1.param
    ${V2_CONV1X1_BIN} --elements 18 --input-shape 2x3x5
    --kernel-size 1x1 --stride 2x2 --padding 0,0,0,0
    --output-channels 3 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv1x1-llvmpipe llvmpipe conv1x1-llvmpipe conv2d from-model model_vulkan
    ${V2_CONV1X1_CPU_LIBRARY} v2_conv1x1 convolution_1x1.param
    ${V2_CONV1X1_BIN} --elements 18
    --input-shape 2x3x5 --kernel-size 1x1 --stride 2x2
    --padding 0,0,0,0 --output-channels 3 --conv-bias)
endif()

if(NCNN_ENABLE_VULKAN_PRODUCT)
    add_test(NAME vulkan-math-exp-runner-pipeline
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_math_exp_pipeline.py
            --mlir-opt ${MLIR_OPT}
            --spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
            --spirv-val ${NCNN_SPIRV_VAL_EXECUTABLE}
            --spirv-dis ${NCNN_SPIRV_DIS_EXECUTABLE}
            --probe ${CMAKE_CURRENT_SOURCE_DIR}/math_exp_probe.mlir
            --evidence ${CMAKE_CURRENT_BINARY_DIR}/evidence/math-exp)
    set_tests_properties(vulkan-math-exp-runner-pipeline PROPERTIES
        LABELS "vulkan;tooling;v5;offline"
        TIMEOUT 60)

#V5.1 product multi - I / O fixtures use the vendored CPU - only ncnn reference
#target directly; they deliberately do not link numerical_test_support.
    add_executable(vulkan_model_runner vulkan_model_runner.cpp)
    target_link_libraries(vulkan_model_runner PRIVATE ncnn)
    target_compile_options(vulkan_model_runner PRIVATE
        -Wall -Wextra -Wpedantic)
    add_test(NAME vulkan-v5-reference-rank-roundtrip
        COMMAND $<TARGET_FILE:vulkan_model_runner> --self-test-ranks)
    set_tests_properties(vulkan-v5-reference-rank-roundtrip PROPERTIES
        LABELS "vulkan;unit;v5;offline"
        TIMEOUT 30)
    set_source_files_properties(vulkan_model_runner.cpp PROPERTIES
        COMPILE_OPTIONS "-fexceptions")

    if(NCNN_ENABLE_VULKAN_V0_TESTS)
    set(V5_CXX_SANITIZER_ARGS)
    set(V5_SANITIZER_RUNTIME_DIR)
    if(COMPILER_SANITIZERS MATCHES "(^|;)(address|undefined)(;|$)")
        list(APPEND V5_CXX_SANITIZER_ARGS
            --cxx-arg=-fsanitize=address,undefined
            --cxx-arg=-fno-sanitize-recover=all
            --cxx-arg=-fno-omit-frame-pointer)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND V5_CXX_SANITIZER_ARGS --cxx-arg=-shared-libasan)
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} --print-resource-dir
                OUTPUT_VARIABLE V5_CLANG_RESOURCE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
            set(V5_SANITIZER_RUNTIME_DIR
                "${V5_CLANG_RESOURCE_DIR}/lib/linux")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE V5_ASAN_LIBRARY
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
            get_filename_component(V5_SANITIZER_RUNTIME_DIR
                "${V5_ASAN_LIBRARY}" DIRECTORY)
        endif()
        if(V5_SANITIZER_RUNTIME_DIR AND
           NOT IS_DIRECTORY "${V5_SANITIZER_RUNTIME_DIR}")
            message(FATAL_ERROR
                "V5 sanitizer runtime directory is missing: ${V5_SANITIZER_RUNTIME_DIR}")
        endif()
    endif()
    set(V5_CONFIGURED_DEPENDENCY_ARGS)
    # VA5: the model DSO must not carry an LLVM dependency in any compiler
    # build mode; only sanitizer SONAMEs extend the configured allowlist.
    foreach(dependency IN LISTS NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
        list(APPEND V5_CONFIGURED_DEPENDENCY_ARGS
            --allowed-dependency=${dependency}
            --sanitizer-dependency=${dependency})
    endforeach()

    set(V5_TEST_LIBRARY_DIRS)
    if(NCNN_VULKAN_USES_SHARED_LLVM)
        list(APPEND V5_TEST_LIBRARY_DIRS ${LLVM_LIBRARY_DIR})
    endif()
    if(V5_SANITIZER_RUNTIME_DIR)
        list(APPEND V5_TEST_LIBRARY_DIRS ${V5_SANITIZER_RUNTIME_DIR})
    endif()
    if(DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        list(APPEND V5_TEST_LIBRARY_DIRS "$ENV{LD_LIBRARY_PATH}")
    endif()
    string(JOIN ":" V5_TEST_LIBRARY_PATH ${V5_TEST_LIBRARY_DIRS})

    set(V5_SQUEEZENET_PARAM
        ${PROJECT_SOURCE_DIR}/test/third_party/ncnn/examples/squeezenet_v1.1.param)
    set(V5_SQUEEZENET_BIN
        ${PROJECT_SOURCE_DIR}/test/third_party/ncnn/examples/squeezenet_v1.1.bin)
    function(ncnn_add_vulkan_v5_test test_name device_match output_name icd suite)
        if(device_match STREQUAL "NVIDIA")
            set(device_name nvidia)
            set(device_layer hardware)
        else()
            set(device_name llvmpipe)
            set(device_layer software)
        endif()
        if(NOT device_name IN_LIST NCNN_VULKAN_TEST_DEVICES)
            return()
        endif()
        set(sanitizer_runtime_args)
        if(V5_SANITIZER_RUNTIME_DIR)
            list(APPEND sanitizer_runtime_args
                --sanitizer-runtime-dir=${V5_SANITIZER_RUNTIME_DIR})
        endif()
        set(model_resource_args)
        if(suite STREQUAL "v5.3")
            list(APPEND model_resource_args
                --model-param ${V5_SQUEEZENET_PARAM}
                --model-bin ${V5_SQUEEZENET_BIN})
        endif()
        add_test(NAME ${test_name}
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/run_v5.py
                --compiler $<TARGET_FILE:ncnn-compile>
                --driver $<TARGET_FILE:ncnn-mlir-driver>
                --opt $<TARGET_FILE:ncnn-mlir-opt>
                --mlir-opt ${MLIR_OPT}
                --translate ${MLIR_TRANSLATE}
                --clang ${CLANG}
                --nm ${LLVM_NM}
                --readelf ${LLVM_READELF}
                --llvm-as ${LLVM_AS}
                --spirv-val ${SPIRV_VAL}
                --spirv-dis ${SPIRV_DIS}
                --spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
                --host-cxx ${CMAKE_CXX_COMPILER}
                --llvm-major ${LLVM_VERSION_MAJOR}
                --runner $<TARGET_FILE:vulkan_model_runner>
                --vulkaninfo ${VULKANINFO}
                --project-include ${PROJECT_SOURCE_DIR}/include
                --device-match ${device_match}
                --icd ${icd}
                ${V5_CXX_SANITIZER_ARGS}
                ${sanitizer_runtime_args}
                ${V5_CONFIGURED_DEPENDENCY_ARGS}
                ${model_resource_args}
                --suite ${suite}
                --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/v5/${output_name})
        set_tests_properties(${test_name} PROPERTIES
            LABELS "vulkan;device;integration;v5;product;${device_layer}"
            RUN_SERIAL TRUE
            TIMEOUT 900)
        if(V5_TEST_LIBRARY_PATH)
            set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT "LD_LIBRARY_PATH=${V5_TEST_LIBRARY_PATH}")
        endif()
    endfunction()

ncnn_add_vulkan_v5_test(
    vulkan-v5-multiop-nvidia NVIDIA nvidia ${V0_NVIDIA_ICD} v5.1)
ncnn_add_vulkan_v5_test(
    vulkan-v5-multiop-llvmpipe llvmpipe llvmpipe ${V0_LLVMPIPE_ICD} v5.1)
ncnn_add_vulkan_v5_test(
    vulkan-v5-pooling-softmax-nvidia NVIDIA pooling-softmax-nvidia
    ${V0_NVIDIA_ICD} v5.2)
ncnn_add_vulkan_v5_test(
    vulkan-v5-pooling-softmax-llvmpipe llvmpipe pooling-softmax-llvmpipe
    ${V0_LLVMPIPE_ICD} v5.2)
ncnn_add_vulkan_v5_test(
    vulkan-v5-squeezenet-nvidia NVIDIA squeezenet-nvidia
    ${V0_NVIDIA_ICD} v5.3)
ncnn_add_vulkan_v5_test(
    vulkan-v5-squeezenet-llvmpipe llvmpipe squeezenet-llvmpipe
    ${V0_LLVMPIPE_ICD} v5.3)
    endif()
endif()
