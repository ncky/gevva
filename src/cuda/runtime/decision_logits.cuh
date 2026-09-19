// Private implementation fragment; included only by src/gpu.cu.
DecisionHeadView prepare_decision_head(const RuntimeWeights& weights,
    std::span<const std::uint32_t> alias_ids, GpuExecutionContext& context,
    DeviceScratchPool& storage) {
  if (alias_ids.empty() || alias_ids.size() > 255)
    throw std::runtime_error("invalid persistent decision aliases");
  storage.reset();
  auto* aliases = static_cast<std::uint32_t*>(storage.allocate(alias_ids.size_bytes()));
  auto* selected = static_cast<__nv_bfloat16*>(storage.allocate(alias_ids.size() * 2816 * 2));
  check(cudaMemcpyAsync(aliases, alias_ids.data(), alias_ids.size_bytes(), cudaMemcpyHostToDevice,
                        context.stream()), "initialize decision aliases");
  gather_decision_embeddings_kernel<<<(alias_ids.size() * 2816 + 255) / 256, 256, 0, context.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(weights.embedding().data), aliases, selected, alias_ids.size());
  check(cudaStreamSynchronize(context.stream()), "prepare permanent decision head");
  return {std::vector<std::uint32_t>(alias_ids.begin(), alias_ids.end()), aliases, selected};
}

std::vector<DecisionReadout> decision_scores(
    const RuntimeWeights& weights, std::span<const float> hidden_states,
    std::span<const std::uint32_t> alias_ids,
    GpuExecutionContext& context, DeviceScratchPool& scratch,
    const void* device_hidden_states, int device_rows, bool options_only,
    const DecisionHeadView* prepared_head) {
  constexpr int hidden = 2816, vocab = 262144;
  if ((device_hidden_states ? (device_rows < 1 || device_rows > 256 || !hidden_states.empty())
                            : (hidden_states.empty() || hidden_states.size() % hidden || hidden_states.size() > 256 * hidden)) || alias_ids.empty() || alias_ids.size() > 255)
    throw std::runtime_error("decision readout requires 1..256 hidden rows and 1..255 aliases");
  const int rows = device_hidden_states ? device_rows : static_cast<int>(hidden_states.size() / hidden);
  const int alias_count = static_cast<int>(alias_ids.size());
  const int columns = options_only ? alias_count : vocab;
  const auto embedding = weights.embedding();
  if (embedding.info->dtype != "BF16" ||
      embedding.info->shape != std::vector<std::uint64_t>{vocab, hidden})
    throw std::runtime_error("decision readout requires BF16 tied embeddings");
  scratch.reset();
  auto* input = device_hidden_states ? static_cast<const __nv_bfloat16*>(device_hidden_states)
                                    : static_cast<const __nv_bfloat16*>(scratch.allocate(rows * hidden * 2));
  auto* normalized = static_cast<__nv_bfloat16*>(scratch.allocate(rows * hidden * 2));
  auto* output = static_cast<float*>(scratch.allocate(rows * columns * sizeof(float)));
  if (prepared_head && (prepared_head->alias_ids.size() < alias_ids.size() ||
      !std::equal(alias_ids.begin(), alias_ids.end(), prepared_head->alias_ids.begin())))
    throw std::runtime_error("persistent decision alias mismatch");
  const auto* aliases = prepared_head ? prepared_head->device_aliases
      : static_cast<std::uint32_t*>(scratch.allocate(alias_count * sizeof(std::uint32_t)));
  auto* summary = static_cast<float*>(scratch.allocate(rows * (alias_count + 2) * sizeof(float)));
  const auto stream = context.stream();
  if (!device_hidden_states) {
  auto* staging = static_cast<__nv_bfloat16*>(scratch.host_staging(rows * hidden * 2));
  for (std::size_t i = 0; i < hidden_states.size(); ++i)
    staging[i] = __float2bfloat16_rn(hidden_states[i]);
  check(cudaMemcpyAsync(const_cast<__nv_bfloat16*>(input), staging, rows * hidden * 2,
                        cudaMemcpyHostToDevice, stream), "upload decision states");
  }
  rmsnorm_2816_kernel<<<rows, 256, 0, stream>>>(
      input, static_cast<const __nv_bfloat16*>(weights.final_norm()),
      normalized, rows, 1e-6F);
  if (!prepared_head) check(cudaMemcpyAsync(const_cast<std::uint32_t*>(aliases), alias_ids.data(), alias_count * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, stream), "upload decision aliases");
  const void* head = embedding.data;
  if (options_only && prepared_head) head = prepared_head->embeddings;
  else if (options_only) {
    auto* selected = static_cast<__nv_bfloat16*>(scratch.allocate(alias_count * hidden * 2));
    gather_decision_embeddings_kernel<<<(alias_count * hidden + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(embedding.data), aliases, selected, alias_count);
    head = selected;
  }
  const float alpha = 1.0F, beta = 0.0F;
  check(cublasGemmEx(static_cast<cublasHandle_t>(context.blas_handle()), CUBLAS_OP_T, CUBLAS_OP_N,
      columns, rows, hidden, &alpha, head, CUDA_R_16BF, hidden,
      normalized, CUDA_R_16BF, hidden, &beta, output, CUDA_R_32F, columns,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "decision vocabulary projection");
  if (options_only) summarize_decision_options_kernel<<<rows, 256, 0, stream>>>(output, summary, alias_count);
  else summarize_decision_logits_kernel<<<rows, 256, 0, stream>>>(output, aliases, summary, alias_count);
  const std::size_t result_count = rows * (alias_count + 2);
  auto* result = static_cast<float*>(scratch.host_staging(result_count * sizeof(float)));
  check(cudaMemcpyAsync(result, summary, result_count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream), "read decision logits");
  check(cudaStreamSynchronize(stream), "finish decision readout");
  std::vector<DecisionReadout> readouts(rows);
  for (int row = 0; row < rows; ++row) {
    readouts[row].full_vocabulary = !options_only;
    readouts[row].log_normalizer = result[row * (alias_count + 2)];
    readouts[row].argmax_token = static_cast<std::uint32_t>(result[row * (alias_count + 2) + 1]);
    readouts[row].option_logits.assign(result + row * (alias_count + 2) + 2, result + (row + 1) * (alias_count + 2));
    if (std::any_of(readouts[row].option_logits.begin(), readouts[row].option_logits.end(), [](float value) { return !std::isfinite(value); }) ||
        (!options_only && !std::isfinite(readouts[row].log_normalizer)))
      throw std::runtime_error("non-finite decision normalization");
  }
  return readouts;
}
