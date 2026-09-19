cmake_minimum_required(VERSION 3.28)
project(gevva LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Both supported cards (RTX PRO 6000 Blackwell and RTX 5090) use SM120.
set(CMAKE_CUDA_ARCHITECTURES 120a)

set(GEVVA_CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "CUTLASS 4.8+ source directory (pinned setup is documented in SETUP.md)")
if(NOT EXISTS "${GEVVA_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS headers not found at ${GEVVA_CUTLASS_DIR}. See SETUP.md setup instructions.")
endif()

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_library(CUDNN_LIBRARY cudnn REQUIRED)

add_executable(gevva-engine
  src/main.cpp
  src/image.cpp
  src/image_gpu.cu
  src/multimodal.cpp
  src/runtime.cpp
  src/decisions.cpp
  src/safetensors.cpp
  src/tokenizer.cpp
  src/cudnn_attention.cpp
  src/gpu.cu
  src/gpu/device.cpp
  src/gpu/execution_context.cpp
  src/gpu/weight_arena.cpp
  src/tiled_attention.cu
  src/nvfp4_cutlass.cu
)
target_include_directories(gevva-engine PRIVATE
  include
  "${GEVVA_CUTLASS_DIR}/include"
  "${GEVVA_CUTLASS_DIR}/tools/util/include"
  "${CMAKE_SOURCE_DIR}/third_party/cudnn-frontend/include"
)
target_link_libraries(gevva-engine PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  CUDA::nvjpeg
  CUDA::nvrtc
  ${CUDNN_LIBRARY}
  nlohmann_json::nlohmann_json
  JPEG::JPEG
  PNG::PNG
  OpenMP::OpenMP_CXX
  Threads::Threads
)
target_compile_options(gevva-engine PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo>
)
set_source_files_properties(src/nvfp4_cutlass.cu PROPERTIES
  COMPILE_OPTIONS "--expt-relaxed-constexpr;-ftemplate-backtrace-limit=0;-diag-suppress=2908;-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1;-DCUTLASS_ENABLE_GDC_FOR_SM100=1"
)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
  add_test(NAME gpu_guard COMMAND gevva-engine gpu-info)
  add_test(NAME decision_shared_kv_lifetime_and_isolation COMMAND gevva-engine decision-kv-test)
  add_test(NAME decision_segmented_global_attention COMMAND gevva-engine decision-attention-test)
  add_test(NAME multimodal_prefill_attention COMMAND gevva-engine prefill-attention-test)
  add_test(NAME text_prefill_attention COMMAND gevva-engine text-prefill-attention-test)
  add_test(NAME experimental_cudnn_text_prefill_attention COMMAND gevva-engine text-prefill-attention-test 1057)
  set_tests_properties(experimental_cudnn_text_prefill_attention PROPERTIES ENVIRONMENT "GEVVA_CUDNN_TEXT_PREFILL=1")
  add_test(NAME experimental_cudnn_text_prefill_chunks COMMAND gevva-engine text-prefill-attention-test 1057)
  set_tests_properties(experimental_cudnn_text_prefill_chunks PROPERTIES ENVIRONMENT "GEVVA_CUDNN_TEXT_PREFILL=1;GEVVA_TEST_TEXT_PREFILL_CHUNKS=1")
  add_test(NAME sliding_kv_large_append_correctness COMMAND gevva-engine sliding-kv-append-test)
  set_tests_properties(sliding_kv_large_append_correctness PROPERTIES PASS_REGULAR_EXPRESSION "sliding_kv_append.checked_tokens=1024.*sliding_kv_append.key_max_abs_error=0.*sliding_kv_append.value_max_abs_error=0")
  add_test(NAME kv_cache_layout COMMAND gevva-engine kv-cache-info 128)
  set_tests_properties(kv_cache_layout PROPERTIES PASS_REGULAR_EXPRESSION "kv.global_capacity=128")
endif()
install(TARGETS gevva-engine RUNTIME DESTINATION bin)
