cmake_minimum_required(VERSION 3.28)
project(g4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This engine intentionally targets only the workstation's Blackwell GPU.
set(CMAKE_CUDA_ARCHITECTURES 120a)

set(G4_CUTLASS_DIR "${CMAKE_SOURCE_DIR}/external/cutlass" CACHE PATH
    "CUTLASS 4.8+ source directory (pinned setup is documented in README.md)")
if(NOT EXISTS "${G4_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS headers not found at ${G4_CUTLASS_DIR}. See README.md setup instructions.")
endif()

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(Python3 COMPONENTS Interpreter QUIET)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_library(CUDNN_LIBRARY cudnn REQUIRED)

add_executable(g4
  src/main.cpp
  src/image.cpp
  src/multimodal.cpp
  src/model_config.cpp
  src/oracle.cpp
  src/openai_server.cpp
  src/runtime.cpp
  src/service.cpp
  src/safetensors.cpp
  src/tokenizer.cpp
  src/cudnn_attention.cpp
  src/sm120_attention.cpp
  src/gpu.cu
  src/gpu/device.cpp
  src/gpu/execution_context.cpp
  src/gpu/weight_arena.cpp
  src/tiled_attention.cu
  src/nvfp4_cutlass.cu
)
set(G4_SM120_DSL_PYTHON "" CACHE FILEPATH "Optional build-time CuTeDSL Python for SM120 attention")
if(G4_SM120_DSL_PYTHON)
  execute_process(COMMAND ${G4_SM120_DSL_PYTHON} -m cutlass.cute.export.aot_config --libdir
    OUTPUT_VARIABLE sm120_runtime_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  set(sm120_runtime "${sm120_runtime_dir}/libcuda_dialect_runtime_static.a")
  if(NOT EXISTS "${sm120_runtime}")
    message(FATAL_ERROR "CuTeDSL static CUDA runtime not found: ${sm120_runtime}")
  endif()
  set(sm120_dir "${CMAKE_BINARY_DIR}/sm120-attention")
  set(sm120_generated)
  foreach(qtile IN ITEMS 64 128)
    add_custom_command(
      OUTPUT "${sm120_dir}/attention${qtile}.h" "${sm120_dir}/attention${qtile}.o"
      COMMAND ${CMAKE_COMMAND} -E env
        CUDA_DEVICE_ORDER=PCI_BUS_ID
        CUDA_VISIBLE_DEVICES=GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d
        CUTE_DSL_ARCH=sm_120a CUDA_HOME=${CUDAToolkit_LIBRARY_ROOT}
        ${G4_SM120_DSL_PYTHON} ${CMAKE_SOURCE_DIR}/tools/export_sm120_attention.py
        --dynamic --q-tile ${qtile} --name attention${qtile} --output ${sm120_dir}
      DEPENDS tools/export_sm120_attention.py tools/sm120_attention_launch.py
        external/cudnn-frontend/python/cudnn/sdpa/fwd/kernels/prefill_f16_sm120.py
        external/cudnn-frontend/python/cudnn/sdpa/fwd/config_sm120.py
      VERBATIM)
    list(APPEND sm120_generated "${sm120_dir}/attention${qtile}.o" "${sm120_dir}/attention${qtile}.h")
    set_source_files_properties("${sm120_dir}/attention${qtile}.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  endforeach()
  target_sources(g4 PRIVATE ${sm120_generated})
  target_include_directories(g4 PRIVATE "${sm120_dir}")
  target_compile_definitions(g4 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:G4_HAVE_SM120_ATTENTION=1>)
  target_link_libraries(g4 PRIVATE "${sm120_runtime}")
  add_executable(g4-sm120-attention-bench tools/bench_sm120_attention.cpp
    src/sm120_attention.cpp src/cudnn_attention.cpp src/gpu/device.cpp
    ${sm120_generated})
  target_compile_definitions(g4-sm120-attention-bench PRIVATE G4_HAVE_SM120_ATTENTION=1)
  target_include_directories(g4-sm120-attention-bench PRIVATE include
    "${sm120_dir}" external/cudnn-frontend/include)
  target_link_libraries(g4-sm120-attention-bench PRIVATE "${sm120_runtime}"
    CUDA::cudart CUDA::cuda_driver CUDA::nvrtc ${CUDNN_LIBRARY} nlohmann_json::nlohmann_json)
endif()
target_include_directories(g4 PRIVATE
  include
  "${G4_CUTLASS_DIR}/include"
  "${G4_CUTLASS_DIR}/tools/util/include"
  "${CMAKE_SOURCE_DIR}/external/cudnn-frontend/include"
)
target_link_libraries(g4 PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  CUDA::nvrtc
  ${CUDNN_LIBRARY}
  nlohmann_json::nlohmann_json
  JPEG::JPEG
  PNG::PNG
  OpenMP::OpenMP_CXX
  Threads::Threads
)
target_compile_options(g4 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=znver5;-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo;-Xcompiler=-march=znver5>
)
set_source_files_properties(src/nvfp4_cutlass.cu PROPERTIES
  COMPILE_OPTIONS "--expt-relaxed-constexpr;-ftemplate-backtrace-limit=0;-diag-suppress=2908;-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1;-DCUTLASS_ENABLE_GDC_FOR_SM100=1"
)

enable_testing()
add_test(NAME ragged_direct_attention_kv_growth COMMAND g4 target-batch-bench 8 8225 4)
set_tests_properties(ragged_direct_attention_kv_growth PROPERTIES
  ENVIRONMENT "G4_FIXED_SHORT_CONTEXT=4096"
  LABELS "correctness;hardware;serving;slow" TIMEOUT 120)
if(G4_SM120_DSL_PYTHON)
  add_test(NAME experimental_sm120_text_prefill
    COMMAND g4-sm120-attention-bench 8 65 1089)
  add_test(NAME experimental_sm120_text_prefill_partial_batch
    COMMAND g4-sm120-attention-bench 3 129 1153)
  add_test(NAME experimental_sm120_text_prefill_large_tile
    COMMAND g4-sm120-attention-bench 8 513 1537)
  add_test(NAME experimental_sm120_text_prefill_chunks
    COMMAND g4 text-prefill-attention-test 1057)
  set_tests_properties(experimental_sm120_text_prefill_chunks PROPERTIES
    ENVIRONMENT "G4_CUDNN_TEXT_PREFILL=1;G4_SM120_TEXT_PREFILL=1;G4_TEST_TEXT_PREFILL_CHUNKS=1")
  set_tests_properties(experimental_sm120_text_prefill
    experimental_sm120_text_prefill_partial_batch experimental_sm120_text_prefill_chunks
    experimental_sm120_text_prefill_large_tile
    PROPERTIES LABELS "correctness;hardware;experimental" TIMEOUT 60)
endif()
add_test(NAME mtp_acceptance COMMAND g4 mtp-accept-test)
set_tests_properties(mtp_acceptance PROPERTIES LABELS "correctness;mtp")
add_test(NAME gpu_guard COMMAND g4 gpu-info)
set_tests_properties(gpu_guard PROPERTIES LABELS "hardware")
add_test(NAME tokenizer_hello COMMAND g4 tokenize "Hello world!")
set_tests_properties(tokenizer_hello PROPERTIES
  PASS_REGULAR_EXPRESSION "ids=9259,1902,236888"
  LABELS "correctness")
add_test(NAME multimodal_prompt_layout
  COMMAND g4 multimodal-layout "<|image|>Describe this image." 266)
set_tests_properties(multimodal_prompt_layout PROPERTIES
  PASS_REGULAR_EXPRESSION "tokens=272.*image_offset=1.*image_tokens=266.*head=255999,258880.*tail=258882,82858,672,2471,236761"
  LABELS "correctness;multimodal")
add_test(NAME multimodal_prefill_attention COMMAND g4 prefill-attention-test)
set_tests_properties(multimodal_prefill_attention PROPERTIES
  LABELS "correctness;hardware;multimodal")
add_test(NAME text_prefill_attention COMMAND g4 text-prefill-attention-test)
add_test(NAME rotary_table_correctness COMMAND g4 rope-table-test)
set_tests_properties(rotary_table_correctness PROPERTIES
  PASS_REGULAR_EXPRESSION "rope_table.exact_cases=32"
  LABELS "correctness;hardware;experimental")
set_tests_properties(text_prefill_attention PROPERTIES LABELS "correctness;hardware")
add_test(NAME experimental_cudnn_text_prefill_attention
  COMMAND g4 text-prefill-attention-test 1057)
set_tests_properties(experimental_cudnn_text_prefill_attention PROPERTIES
  ENVIRONMENT "G4_CUDNN_TEXT_PREFILL=1" LABELS "correctness;hardware;experimental")
add_test(NAME experimental_cudnn_text_prefill_chunks
  COMMAND g4 text-prefill-attention-test 1057)
set_tests_properties(experimental_cudnn_text_prefill_chunks PROPERTIES
  ENVIRONMENT "G4_CUDNN_TEXT_PREFILL=1;G4_TEST_TEXT_PREFILL_CHUNKS=1"
  LABELS "correctness;hardware;experimental")
if(EXISTS "/mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg")
  add_test(NAME image_preprocess_geometry COMMAND g4 image-preprocess
    /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg)
  set_tests_properties(image_preprocess_geometry PROPERTIES
    PASS_REGULAR_EXPRESSION "image.resized_width=672.*image.resized_height=912.*image.patch_count=2394.*image.soft_token_count=266"
    LABELS "correctness;multimodal")
  if(Python3_Interpreter_FOUND AND
     EXISTS "/mnt/SSD/g4-models/gemma4-26b-a4b-trtllm/manifest.json")
    add_test(NAME multimodal_serve_repeatability
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/test_multimodal_serve.py
        $<TARGET_FILE:g4>
        /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg)
    set_tests_properties(multimodal_serve_repeatability PROPERTIES
      PASS_REGULAR_EXPRESSION "multimodal_serve.outputs_repeatable=true"
      LABELS "correctness;hardware;multimodal;mtp;serving;very-slow"
      TIMEOUT 180)
    add_test(NAME multimodal_batch_serve_repeatability
      COMMAND g4 batch-serve-test
        /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg
        8)
    set_tests_properties(multimodal_batch_serve_repeatability PROPERTIES
      PASS_REGULAR_EXPRESSION "batch_serve.sessions=8.*batch_serve.prompt_tokens=551.*batch_serve.repeatable=true"
      LABELS "correctness;hardware;multimodal;mtp;serving;very-slow"
      TIMEOUT 180)
    add_test(NAME multimodal_batch_mtp_repeatability
      COMMAND ${CMAKE_COMMAND} -E env G4_BATCH_SERVE_TEST_TOKENS=64
        $<TARGET_FILE:g4> batch-serve-test
        /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg
        8)
    set_tests_properties(multimodal_batch_mtp_repeatability PROPERTIES
      PASS_REGULAR_EXPRESSION "batch_serve.sessions=8.*batch_serve.prompt_tokens=551.*batch_serve.repeatable=true"
      LABELS "correctness;hardware;multimodal;mtp;serving;very-slow"
      TIMEOUT 180)
  endif()
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/goldens/generated/image/pixel_values.f32")
  add_test(NAME image_preprocess_differential COMMAND g4 image-preprocess
    /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg
    280 ${CMAKE_SOURCE_DIR}/goldens/generated/image)
  set_tests_properties(image_preprocess_differential PROPERTIES
    PASS_REGULAR_EXPRESSION "image.oracle_above_one_pixel=0"
    LABELS "correctness;multimodal;differential")
  add_test(NAME vision_patch_differential COMMAND g4 vision-patch-bench
    /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg
    ${CMAKE_SOURCE_DIR}/goldens/generated/image)
  set_tests_properties(vision_patch_differential PROPERTIES
    LABELS "correctness;hardware;multimodal;differential;slow")
  add_test(NAME vision_layer0_differential COMMAND g4 vision-layer-bench
    ${CMAKE_SOURCE_DIR}/goldens/generated/image full)
  set_tests_properties(vision_layer0_differential PROPERTIES
    LABELS "correctness;hardware;multimodal;differential;slow")
  add_test(NAME vision_encoder_differential COMMAND g4 vision-encoder-bench
    /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg
    ${CMAKE_SOURCE_DIR}/goldens/generated/image)
  set_tests_properties(vision_encoder_differential PROPERTIES
    LABELS "correctness;hardware;multimodal;differential;slow")
  add_test(NAME vision_encoder_legacy_differential
    COMMAND ${CMAKE_COMMAND} -E env G4_LEGACY_VISION_ATTENTION=1
      $<TARGET_FILE:g4> vision-encoder-bench
      /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg
      ${CMAKE_SOURCE_DIR}/goldens/generated/image)
  set_tests_properties(vision_encoder_legacy_differential PROPERTIES
    LABELS "correctness;hardware;multimodal;differential;slow")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/goldens/generated/multimodal_prefill.json")
  add_test(NAME multimodal_prefill_differential COMMAND g4 multimodal-prefill
    /mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg
    "<|image|>Describe this image.")
  set_tests_properties(multimodal_prefill_differential PROPERTIES
    PASS_REGULAR_EXPRESSION "multimodal_prefill.oracle_token_match=true"
    LABELS "correctness;hardware;multimodal;differential;slow"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
add_test(NAME primitive_correctness COMMAND g4 primitive-bench)
add_test(NAME attention_softmax_correctness COMMAND g4 attention-softmax-test)
add_test(NAME fast_attention_softmax_correctness COMMAND g4 fast-softmax-test)
add_test(NAME attention_unpack_correctness COMMAND g4 attention-unpack-test)
set_tests_properties(attention_unpack_correctness PROPERTIES
  PASS_REGULAR_EXPRESSION "attention_unpack.exact_cases=19"
  LABELS "correctness;hardware;mtp")
set_tests_properties(fast_attention_softmax_correctness PROPERTIES
  PASS_REGULAR_EXPRESSION "fast_softmax.exact_cases=36"
  LABELS "correctness;hardware;mtp")
set_tests_properties(attention_softmax_correctness PROPERTIES
  PASS_REGULAR_EXPRESSION "attention_softmax.cases=8"
  LABELS "correctness;hardware;mtp")
set_tests_properties(primitive_correctness PROPERTIES LABELS "correctness;hardware")
add_test(NAME decode_attention_correctness COMMAND g4 decode-attention-bench 128)
set_tests_properties(decode_attention_correctness PROPERTIES LABELS "correctness;hardware")
add_test(NAME turboquant_global_kv_sanity COMMAND g4 turboquant-kv-bench 4096)
set_tests_properties(turboquant_global_kv_sanity PROPERTIES
  PASS_REGULAR_EXPRESSION "turboquant.compression_ratio=3.938"
  LABELS "correctness;hardware;quantization;experimental")
add_test(NAME decode_attention_batch_correctness
  COMMAND g4 decode-attention-batch-bench 4096)
set_tests_properties(decode_attention_batch_correctness PROPERTIES
  LABELS "correctness;hardware;mtp")
add_test(NAME sliding_kv_large_append_correctness
  COMMAND g4 sliding-kv-append-test)
set_tests_properties(sliding_kv_large_append_correctness PROPERTIES
  PASS_REGULAR_EXPRESSION "sliding_kv_append.checked_tokens=1024.*sliding_kv_append.key_max_abs_error=0.*sliding_kv_append.value_max_abs_error=0"
  LABELS "correctness;hardware;long-context")
add_test(NAME kv_cache_layout COMMAND g4 kv-cache-info 128)
set_tests_properties(kv_cache_layout PROPERTIES
  PASS_REGULAR_EXPRESSION "kv.global_capacity=128"
  LABELS "correctness;hardware")
add_test(NAME bf16_matvec_correctness COMMAND g4 matvec-bench)
set_tests_properties(bf16_matvec_correctness PROPERTIES LABELS "correctness;hardware")
if(EXISTS "${CMAKE_SOURCE_DIR}/goldens/generated/manifest.json")
  add_test(NAME transformers_frontend_differential COMMAND g4 differential)
  set_tests_properties(transformers_frontend_differential PROPERTIES
    LABELS "correctness;hardware;differential"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
if(EXISTS "/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4/model.safetensors.index.json")
  add_test(NAME nvfp4_decode_matvec_correctness COMMAND g4 nvfp4-bench
    --model /mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4)
  set_tests_properties(nvfp4_decode_matvec_correctness PROPERTIES
    LABELS "correctness;hardware;nvfp4")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/goldens/generated/nvfp4_layer0_output.f32")
  add_test(NAME nvfp4_cutlass_expert_differential COMMAND g4 cutlass-expert-bench 0)
  set_tests_properties(nvfp4_cutlass_expert_differential PROPERTIES
    LABELS "correctness;hardware;differential;nvfp4;slow"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
if(EXISTS "/mnt/SSD/g4-models/gemma4-26b-a4b-trtllm/manifest.json")
  add_test(NAME nvfp4_expert_repeatability
    COMMAND g4 expert-repeat-test 5 272 8)
  set_tests_properties(nvfp4_expert_repeatability PROPERTIES
    LABELS "correctness;hardware;nvfp4;slow")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/goldens/generated/assistant_state_ctx4096.f32" AND
   EXISTS "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant/model.safetensors" AND
   EXISTS "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8/model.safetensors")
  add_test(NAME assistant_step_differential COMMAND g4 assistant-step-bench 4096)
  set_tests_properties(assistant_step_differential PROPERTIES
    PASS_REGULAR_EXPRESSION "assistant_step.oracle_token_match=true"
    LABELS "correctness;hardware;differential;mtp;slow"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/goldens/generated/assistant_state_ctx4096_drafts4.f32" AND
   EXISTS "/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4/model.safetensors.index.json" AND
   EXISTS "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8/model.safetensors")
  add_test(NAME assistant_cycle_differential COMMAND g4 assistant-cycle-bench 4096)
  set_tests_properties(assistant_cycle_differential PROPERTIES
    PASS_REGULAR_EXPRESSION "assistant_step.oracle_token_match=true"
    LABELS "correctness;hardware;differential;mtp;slow"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
