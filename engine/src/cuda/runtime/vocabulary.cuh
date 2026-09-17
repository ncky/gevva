// Private implementation fragment; included only by src/gpu.cu.
struct CompressedVocabRunner::Impl {
  int columns{};
  int candidates_per_tile{4};
  const __nv_bfloat16* exact{};
  const std::int8_t* packed{};
  const float* row_scales{};
  std::int8_t* quantized{};
  float* activation_scale{};
  std::int32_t* accumulated{};
  int* candidate_ids{};
  float* corrected{};
  cublasHandle_t handle{};

  Impl(const DeviceModel& exact_model, const std::string& exact_tensor,
       const DeviceModel& int8_model, cudaStream_t stream) {
    constexpr int kVocab = 262144;
    const auto exact_view = exact_model.tensor(exact_tensor);
    const auto packed_view = int8_model.tensor("weight");
    const auto scales_view = int8_model.tensor("scale");
    if (exact_view.info->dtype != "BF16" || exact_view.info->shape.size() != 2 ||
        exact_view.info->shape[0] != kVocab || packed_view.info->dtype != "I8" ||
        packed_view.info->shape != exact_view.info->shape ||
        scales_view.info->dtype != "F32" ||
        scales_view.info->shape != std::vector<std::uint64_t>{kVocab})
      throw std::runtime_error("invalid compressed vocabulary runner tensors");
    columns = static_cast<int>(exact_view.info->shape[1]);
    if (const char* value = std::getenv("G4_VOCAB_CANDIDATES_PER_TILE")) {
      candidates_per_tile = std::atoi(value);
      if (candidates_per_tile != 1 && candidates_per_tile != 2 &&
          candidates_per_tile != 4)
        throw std::runtime_error(
            "G4_VOCAB_CANDIDATES_PER_TILE must be 1, 2, or 4");
    }
    exact = reinterpret_cast<const __nv_bfloat16*>(exact_view.data);
    packed = reinterpret_cast<const std::int8_t*>(packed_view.data);
    row_scales = reinterpret_cast<const float*>(scales_view.data);
    constexpr int kMaximumRows = 40;
    check(cudaMalloc(&quantized, kMaximumRows * columns), "cudaMalloc(compressed vocab input)");
    check(cudaMalloc(&activation_scale, kMaximumRows * sizeof(float)),
          "cudaMalloc(compressed vocab scale)");
    check(cudaMalloc(&accumulated, kMaximumRows * kVocab * sizeof(std::int32_t)),
          "cudaMalloc(compressed vocab sums)");
    check(cudaMalloc(&candidate_ids, kMaximumRows * 1024 * sizeof(int)),
          "cudaMalloc(compressed vocab candidates)");
    check(cudaMalloc(&corrected, kMaximumRows * 1024 * sizeof(float)),
          "cudaMalloc(compressed vocab corrected)");
    check(cublasCreate(&handle), "cublasCreate(compressed vocab)");
    check(cublasSetStream(handle, stream), "cublasSetStream(compressed vocab)");
  }

  ~Impl() {
    if (handle) cublasDestroy(handle);
    cudaFree(corrected); cudaFree(candidate_ids);
    cudaFree(accumulated); cudaFree(activation_scale); cudaFree(quantized);
  }

  void launch(const __nv_bfloat16* activation, int* selected, int tokens,
              cudaStream_t stream) const {
    constexpr int kVocab = 262144;
    if (tokens < 1 || tokens > 40)
      throw std::runtime_error("compressed vocabulary batch must be 1..40");
    check(cublasSetStream(handle, stream), "cublasSetStream(compressed vocab)");
    quantize_int8_rows_kernel<<<tokens, 256, 0, stream>>>(
        activation, quantized, activation_scale, columns, tokens);
    const std::int32_t alpha = 1;
    const std::int32_t beta = 0;
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kVocab, tokens, columns,
                       &alpha, packed, CUDA_R_8I, columns, quantized, CUDA_R_8I,
                       columns, &beta, accumulated, CUDA_R_32I, kVocab,
                       CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "compressed vocabulary projection");
    if (candidates_per_tile == 1)
      local_top_accumulated_rows_kernel<1><<<tokens * 256, 256, 0, stream>>>(
          accumulated, row_scales, activation_scale, candidate_ids, tokens);
    else if (candidates_per_tile == 2)
      local_top_accumulated_rows_kernel<2><<<tokens * 256, 256, 0, stream>>>(
          accumulated, row_scales, activation_scale, candidate_ids, tokens);
    else
      local_top_accumulated_rows_kernel<4><<<tokens * 256, 256, 0, stream>>>(
          accumulated, row_scales, activation_scale, candidate_ids, tokens);
    const int candidates = candidates_per_tile * 256;
    correct_vocab_candidates_rows_kernel<<<tokens * candidates, 256, 0, stream>>>(
        exact, activation, columns, candidate_ids, corrected, tokens,
        candidates);
    argmax_corrected_rows_kernel<<<tokens, 256, 0, stream>>>(
        corrected, candidate_ids, selected, tokens, candidates);
  }

  void warmup(int maximum_tokens, cudaStream_t stream) const {
    constexpr int kVocab = 262144;
    if (maximum_tokens < 1 || maximum_tokens > 40)
      throw std::runtime_error("invalid compressed vocabulary warmup rows");
    check(cublasSetStream(handle, stream),
          "cublasSetStream(compressed vocab warmup)");
    check(cudaMemsetAsync(quantized, 0,
                          static_cast<std::size_t>(maximum_tokens) * columns,
                          stream),
          "clear compressed vocabulary warmup input");
    const std::int32_t alpha = 1;
    const std::int32_t beta = 0;
    for (int tokens = 1; tokens <= maximum_tokens; ++tokens)
      check(cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, kVocab, tokens, columns,
                &alpha, packed, CUDA_R_8I, columns, quantized, CUDA_R_8I,
                columns, &beta, accumulated, CUDA_R_32I, kVocab,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "warm compressed vocabulary projection");
    check(cudaStreamSynchronize(stream),
          "synchronize compressed vocabulary warmup");
  }
};

CompressedVocabRunner::CompressedVocabRunner(
    const DeviceModel& exact_model, const std::string& exact_tensor,
    const DeviceModel& int8_model, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(exact_model, exact_tensor, int8_model, stream)) {}
CompressedVocabRunner::~CompressedVocabRunner() = default;

void CompressedVocabRunner::launch(const void* activation, int* selected_token,
                                   cudaStream_t stream) const {
  impl_->launch(static_cast<const __nv_bfloat16*>(activation), selected_token, 1, stream);
}

void CompressedVocabRunner::launch_batch(const void* activations,
                                         int* selected_tokens, int tokens,
                                         cudaStream_t stream) const {
  impl_->launch(static_cast<const __nv_bfloat16*>(activations), selected_tokens,
                tokens, stream);
}

void CompressedVocabRunner::warmup(int maximum_tokens,
                                   cudaStream_t stream) const {
  impl_->warmup(maximum_tokens, stream);
}

