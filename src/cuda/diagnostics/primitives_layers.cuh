// Private implementation fragment; included only by src/gpu.cu.
PrimitiveTestResult test_and_benchmark_primitives() {
  constexpr int kRows = 64;
  constexpr int kHidden = 2816;
  constexpr int kExperts = 128;
  constexpr int kTop = 8;
  std::vector<__nv_bfloat16> input(kRows * kHidden), norm_weight(kHidden);
  std::vector<__nv_bfloat16> logits(kRows * kExperts), expert_scale(kExperts);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = __float2bfloat16_rn(std::sin(static_cast<float>(i) * 0.013F) * 1.7F);
  for (int i = 0; i < kHidden; ++i)
    norm_weight[i] = __float2bfloat16_rn(0.7F + (i % 29) * 0.017F);
  for (int row = 0; row < kRows; ++row) {
    for (int expert = 0; expert < kExperts; ++expert) {
      // A row-dependent permutation with BF16-separated values avoids ambiguous
      // tie ordering while still exercising every expert id.
      const int rank = (expert * 37 + row * 19) & 127;
      logits[row * kExperts + expert] = __float2bfloat16_rn((rank - 64) * 0.0625F);
    }
  }
  for (int i = 0; i < kExperts; ++i)
    expert_scale[i] = __float2bfloat16_rn(0.5F + (i % 17) * 0.07F);

  __nv_bfloat16 *d_input{}, *d_norm_weight{}, *d_norm_output{}, *d_logits{}, *d_scale{};
  float* d_route_weights{};
  int* d_route_ids{};
  check(cudaMalloc(&d_input, input.size() * sizeof(input[0])), "cudaMalloc(input)");
  check(cudaMalloc(&d_norm_weight, norm_weight.size() * sizeof(norm_weight[0])), "cudaMalloc(norm weight)");
  check(cudaMalloc(&d_norm_output, input.size() * sizeof(input[0])), "cudaMalloc(norm output)");
  check(cudaMalloc(&d_logits, logits.size() * sizeof(logits[0])), "cudaMalloc(logits)");
  check(cudaMalloc(&d_scale, expert_scale.size() * sizeof(expert_scale[0])), "cudaMalloc(scale)");
  check(cudaMalloc(&d_route_weights, kRows * kTop * sizeof(float)), "cudaMalloc(route weights)");
  check(cudaMalloc(&d_route_ids, kRows * kTop * sizeof(int)), "cudaMalloc(route ids)");
  check(cudaMemcpy(d_input, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice), "copy input");
  check(cudaMemcpy(d_norm_weight, norm_weight.data(), norm_weight.size() * sizeof(norm_weight[0]), cudaMemcpyHostToDevice), "copy norm weight");
  check(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(logits[0]), cudaMemcpyHostToDevice), "copy logits");
  check(cudaMemcpy(d_scale, expert_scale.data(), expert_scale.size() * sizeof(expert_scale[0]), cudaMemcpyHostToDevice), "copy scale");

  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  auto launch_norm = [&] { rmsnorm_2816_kernel<<<kRows, 256, 0, stream>>>(d_input, d_norm_weight, d_norm_output, kRows, 1e-6F); };
  auto launch_route = [&] { route_128_top8_kernel<<<kRows, 128, 0, stream>>>(d_logits, d_scale, d_route_weights, d_route_ids, kRows); };
  launch_norm();
  launch_route();
  check(cudaGetLastError(), "primitive kernel launch");
  check(cudaStreamSynchronize(stream), "primitive synchronization");

  std::vector<__nv_bfloat16> norm_output(input.size());
  std::vector<float> route_weights(kRows * kTop);
  std::vector<int> route_ids(kRows * kTop);
  check(cudaMemcpy(norm_output.data(), d_norm_output, norm_output.size() * sizeof(norm_output[0]), cudaMemcpyDeviceToHost), "copy norm output");
  check(cudaMemcpy(route_weights.data(), d_route_weights, route_weights.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy route weights");
  check(cudaMemcpy(route_ids.data(), d_route_ids, route_ids.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy route ids");

  float norm_error = 0.0F;
  for (int row = 0; row < kRows; ++row) {
    float square_sum = 0.0F;
    for (int col = 0; col < kHidden; ++col) {
      const float x = __bfloat162float(input[row * kHidden + col]);
      square_sum += x * x;
    }
    const float inv_rms = 1.0F / std::sqrt(square_sum / kHidden + 1e-6F);
    for (int col = 0; col < kHidden; ++col) {
      const float expected = __bfloat162float(__float2bfloat16_rn(
          __bfloat162float(input[row * kHidden + col]) * inv_rms *
          __bfloat162float(norm_weight[col])));
      norm_error = std::max(norm_error, std::abs(expected - __bfloat162float(norm_output[row * kHidden + col])));
    }
  }

  float route_error = 0.0F;
  bool ids_match = true;
  for (int row = 0; row < kRows; ++row) {
    std::vector<int> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(), [&](int a, int b) {
      return __bfloat162float(logits[row * kExperts + a]) > __bfloat162float(logits[row * kExperts + b]);
    });
    const float maximum = __bfloat162float(logits[row * kExperts + order[0]]);
    float denominator = 0.0F;
    for (int k = 0; k < kTop; ++k)
      denominator += std::exp(__bfloat162float(logits[row * kExperts + order[k]]) - maximum);
    for (int k = 0; k < kTop; ++k) {
      ids_match &= route_ids[row * kTop + k] == order[k];
      const float expected = std::exp(__bfloat162float(logits[row * kExperts + order[k]]) - maximum) /
                             denominator * __bfloat162float(expert_scale[order[k]]);
      route_error = std::max(route_error, std::abs(expected - route_weights[row * kTop + k]));
    }
  }

  const float norm_us = event_benchmark(stream, launch_norm, 10, 1000);
  const float route_us = event_benchmark(stream, launch_route, 10, 1000);
  cudaStreamDestroy(stream);
  cudaFree(d_route_ids); cudaFree(d_route_weights); cudaFree(d_scale); cudaFree(d_logits);
  cudaFree(d_norm_output); cudaFree(d_norm_weight); cudaFree(d_input);
  return {norm_error, route_error, ids_match, norm_us, route_us};
}

MatvecTestResult test_bf16_matvec(std::span<const std::byte> row_major_weight,
                                 int output_features, int input_features) {
  const auto expected_bytes = static_cast<std::size_t>(output_features) * input_features * 2;
  if (row_major_weight.size() != expected_bytes) {
    throw std::runtime_error("BF16 matvec weight byte count does not match shape");
  }
  const auto* host_weight = reinterpret_cast<const __nv_bfloat16*>(row_major_weight.data());
  std::vector<__nv_bfloat16> input(input_features), output(output_features);
  for (int i = 0; i < input_features; ++i) {
    input[i] = __float2bfloat16_rn(std::sin(i * 0.019F) + std::cos(i * 0.007F) * 0.31F);
  }
  __nv_bfloat16 *d_weight{}, *d_input{}, *d_output{};
  check(cudaMalloc(&d_weight, expected_bytes), "cudaMalloc(matvec weight)");
  check(cudaMalloc(&d_input, input.size() * 2), "cudaMalloc(matvec input)");
  check(cudaMalloc(&d_output, output.size() * 2), "cudaMalloc(matvec output)");
  check(cudaMemcpy(d_weight, host_weight, expected_bytes, cudaMemcpyHostToDevice), "copy matvec weight");
  check(cudaMemcpy(d_input, input.data(), input.size() * 2, cudaMemcpyHostToDevice), "copy matvec input");
  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  check(cublasCreate(&handle), "cublasCreate");
  check(cublasSetStream(handle, stream), "cublasSetStream");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto launch = [&] {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                       output_features, 1, input_features,
                       &alpha, d_weight, CUDA_R_16BF, input_features,
                       d_input, CUDA_R_16BF, input_features,
                       &beta, d_output, CUDA_R_16BF, output_features,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(BF16 matvec)");
  };
  launch();
  check(cudaStreamSynchronize(stream), "matvec synchronization");
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost), "copy matvec output");

  float max_error = 0.0F;
  for (int row = 0; row < output_features; ++row) {
    float expected = 0.0F;
    for (int col = 0; col < input_features; ++col) {
      expected += __bfloat162float(host_weight[row * input_features + col]) *
                  __bfloat162float(input[col]);
    }
    expected = __bfloat162float(__float2bfloat16_rn(expected));
    max_error = std::max(max_error, std::abs(expected - __bfloat162float(output[row])));
  }
  const float microseconds = event_benchmark(stream, launch, 20, 1000);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  cudaFree(d_output); cudaFree(d_input); cudaFree(d_weight);
  return {max_error, microseconds};
}

MatvecTestResult test_nvfp4_matvec(std::span<const std::byte> packed_weight,
                                  std::span<const std::byte> block_scale,
                                  std::span<const std::byte> global_scale,
                                  int output_features, int input_features) {
  if (input_features % 16 != 0 || packed_weight.size() !=
          static_cast<std::size_t>(output_features) * input_features / 2 ||
      block_scale.size() != static_cast<std::size_t>(output_features) * input_features / 16 ||
      global_scale.size() != sizeof(float)) {
    throw std::runtime_error("NVFP4 matvec tensor shape mismatch");
  }
  const auto* weight = reinterpret_cast<const std::uint8_t*>(packed_weight.data());
  const auto* scales = reinterpret_cast<const std::uint8_t*>(block_scale.data());
  const float global = *reinterpret_cast<const float*>(global_scale.data());
  std::vector<__nv_bfloat16> input(input_features), output(output_features);
  for (int i = 0; i < input_features; ++i)
    input[i] = __float2bfloat16_rn(std::sin(i * 0.019F) + std::cos(i * 0.007F) * 0.31F);
  std::uint8_t *d_weight{}, *d_scales{};
  __nv_bfloat16 *d_input{}, *d_output{};
  check(cudaMalloc(&d_weight, packed_weight.size()), "cudaMalloc(NVFP4 weight)");
  check(cudaMalloc(&d_scales, block_scale.size()), "cudaMalloc(NVFP4 scales)");
  check(cudaMalloc(&d_input, input.size() * 2), "cudaMalloc(NVFP4 input)");
  check(cudaMalloc(&d_output, output.size() * 2), "cudaMalloc(NVFP4 output)");
  check(cudaMemcpy(d_weight, weight, packed_weight.size(), cudaMemcpyHostToDevice), "copy NVFP4 weight");
  check(cudaMemcpy(d_scales, scales, block_scale.size(), cudaMemcpyHostToDevice), "copy NVFP4 scales");
  check(cudaMemcpy(d_input, input.data(), input.size() * 2, cudaMemcpyHostToDevice), "copy NVFP4 input");
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate(NVFP4)");
  auto launch = [&] {
    nvfp4_matvec_kernel<256><<<output_features, 256, 0, stream>>>(
        d_weight, d_scales, global, d_input, d_output, output_features, input_features);
  };
  launch();
  check(cudaGetLastError(), "NVFP4 matvec launch");
  check(cudaStreamSynchronize(stream), "NVFP4 matvec synchronization");
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy NVFP4 output");
  float max_error = 0.0F;
  for (int row = 0; row < output_features; ++row) {
    float expected = 0.0F;
    for (int col = 0; col < input_features; ++col) {
      const unsigned packed = weight[row * (input_features / 2) + col / 2];
      const unsigned nibble = (col & 1) ? packed >> 4 : packed & 15U;
      const float scale = decode_e4m3(scales[row * (input_features / 16) + col / 16]);
      expected += __bfloat162float(input[col]) * decode_e2m1(nibble) * scale * global;
    }
    expected = __bfloat162float(__float2bfloat16_rn(expected));
    max_error = std::max(max_error, std::abs(expected - __bfloat162float(output[row])));
  }
  const float microseconds = event_benchmark(stream, launch, 20, 1000);
  cudaStreamDestroy(stream);
  cudaFree(d_output); cudaFree(d_input); cudaFree(d_scales); cudaFree(d_weight);
  return {max_error, microseconds};
}

MatvecTestResult test_nvfp4_grouped_matvec(
    const std::vector<std::span<const std::byte>>& packed_weights,
    const std::vector<std::span<const std::byte>>& block_scales,
    const std::vector<std::span<const std::byte>>& global_scales,
    int output_features, int input_features) {
  const int groups = static_cast<int>(packed_weights.size());
  if (groups == 0 || block_scales.size() != packed_weights.size() ||
      global_scales.size() != packed_weights.size() || input_features % 16 != 0) {
    throw std::runtime_error("invalid NVFP4 grouped matvec arguments");
  }
  const std::size_t weight_bytes = static_cast<std::size_t>(output_features) * input_features / 2;
  const std::size_t scale_bytes = static_cast<std::size_t>(output_features) * input_features / 16;
  std::vector<float> globals(groups);
  std::uint8_t *d_weights{}, *d_scales{};
  check(cudaMalloc(&d_weights, groups * weight_bytes), "cudaMalloc(routed weight pack)");
  check(cudaMalloc(&d_scales, groups * scale_bytes), "cudaMalloc(routed scale pack)");
  for (int group = 0; group < groups; ++group) {
    if (packed_weights[group].size() != weight_bytes || block_scales[group].size() != scale_bytes ||
        global_scales[group].size() != sizeof(float))
      throw std::runtime_error("NVFP4 grouped tensor shape mismatch");
    globals[group] = *reinterpret_cast<const float*>(global_scales[group].data());
    check(cudaMemcpy(d_weights + group * weight_bytes, packed_weights[group].data(), weight_bytes,
                     cudaMemcpyHostToDevice), "copy group weight");
    check(cudaMemcpy(d_scales + group * scale_bytes, block_scales[group].data(), scale_bytes,
                     cudaMemcpyHostToDevice), "copy group scales");
  }
  std::vector<__nv_bfloat16> input(static_cast<std::size_t>(groups) * input_features);
  for (int group = 0; group < groups; ++group)
    for (int i = 0; i < input_features; ++i)
      input[group * input_features + i] = __float2bfloat16_rn(
          std::sin(i * 0.019F + group * 0.11F) + std::cos(i * 0.007F) * 0.31F);
  __nv_bfloat16 *d_input{}, *d_output{};
  float* d_globals{};
  int* d_expert_ids{};
  std::vector<int> expert_ids(groups);
  std::iota(expert_ids.begin(), expert_ids.end(), 0);
  check(cudaMalloc(&d_globals, groups * sizeof(float)), "cudaMalloc(routed globals)");
  check(cudaMalloc(&d_expert_ids, groups * sizeof(int)), "cudaMalloc(routed ids)");
  check(cudaMalloc(&d_input, input.size() * 2), "cudaMalloc(grouped input)");
  check(cudaMalloc(&d_output, static_cast<std::size_t>(groups) * output_features * 2),
        "cudaMalloc(grouped output)");
  check(cudaMemcpy(d_input, input.data(), input.size() * 2, cudaMemcpyHostToDevice),
        "copy grouped input");
  check(cudaMemcpy(d_globals, globals.data(), groups * sizeof(float), cudaMemcpyHostToDevice),
        "copy routed global scales");
  check(cudaMemcpy(d_expert_ids, expert_ids.data(), groups * sizeof(int), cudaMemcpyHostToDevice),
        "copy routed expert ids");
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate(grouped)");
  auto launch = [&] {
    // Keeping one expert per launch preserves locality in the tiny M=1 regime.
    // The seemingly more compact 2-D grid interleaves independent allocations
    // and is ~3x slower on this workstation's SM120 GPU.
    for (int route = 0; route < groups; ++route) {
      nvfp4_routed_matvec_kernel<<<output_features, 256, 0, stream>>>(
          d_weights, d_scales, d_globals, d_expert_ids, route,
          d_input, d_output, output_features, input_features);
    }
  };
  launch();
  check(cudaStreamSynchronize(stream), "grouped NVFP4 synchronization");
  std::vector<__nv_bfloat16> output(static_cast<std::size_t>(groups) * output_features);
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy grouped output");
  float max_error = 0.0F;
  for (int group = 0; group < groups; ++group) {
    const auto* weight = reinterpret_cast<const std::uint8_t*>(packed_weights[group].data());
    const auto* scale = reinterpret_cast<const std::uint8_t*>(block_scales[group].data());
    for (int row = 0; row < output_features; ++row) {
      float expected = 0.0F;
      for (int col = 0; col < input_features; ++col) {
        const unsigned packed = weight[row * (input_features / 2) + col / 2];
        const unsigned nibble = (col & 1) ? packed >> 4 : packed & 15U;
        expected += __bfloat162float(input[group * input_features + col]) * decode_e2m1(nibble) *
                    decode_e4m3(scale[row * (input_features / 16) + col / 16]) * globals[group];
      }
      expected = __bfloat162float(__float2bfloat16_rn(expected));
      max_error = std::max(max_error,
          std::abs(expected - __bfloat162float(output[group * output_features + row])));
    }
  }
  const float microseconds = event_benchmark(stream, launch, 20, 1000);
  cudaStreamDestroy(stream);
  cudaFree(d_expert_ids); cudaFree(d_globals);
  cudaFree(d_output); cudaFree(d_input);
  cudaFree(d_scales); cudaFree(d_weights);
  return {max_error, microseconds};
}

ExpertBlockTestResult test_nvfp4_expert_block(
    const std::vector<std::span<const std::byte>>& gate_weights,
    const std::vector<std::span<const std::byte>>& gate_scales,
    const std::vector<std::span<const std::byte>>& gate_globals,
    const std::vector<std::span<const std::byte>>& up_weights,
    const std::vector<std::span<const std::byte>>& up_scales,
    const std::vector<std::span<const std::byte>>& up_globals,
    const std::vector<std::span<const std::byte>>& down_weights,
    const std::vector<std::span<const std::byte>>& down_scales,
    const std::vector<std::span<const std::byte>>& down_globals) {
  constexpr int kExperts = 128;
  constexpr int kRoutes = 8;
  constexpr int kHidden = 2816;
  constexpr int kIntermediate = 704;
  const std::size_t weight_bytes = static_cast<std::size_t>(kHidden) * kIntermediate / 2;
  const std::size_t scale_bytes = static_cast<std::size_t>(kHidden) * kIntermediate / 16;
  struct PackedProjection {
    std::uint8_t* weights{};
    std::uint8_t* scales{};
    float* globals{};
  };
  auto pack = [&](const std::vector<std::span<const std::byte>>& weights,
                  const std::vector<std::span<const std::byte>>& scales,
                  const std::vector<std::span<const std::byte>>& globals,
                  const char* name) {
    if (weights.size() != kExperts || scales.size() != kExperts || globals.size() != kExperts)
      throw std::runtime_error(std::string(name) + " expert count mismatch");
    PackedProjection result;
    check(cudaMalloc(&result.weights, kExperts * weight_bytes), "cudaMalloc(expert weight pack)");
    check(cudaMalloc(&result.scales, kExperts * scale_bytes), "cudaMalloc(expert scale pack)");
    check(cudaMalloc(&result.globals, kExperts * sizeof(float)), "cudaMalloc(expert globals)");
    std::vector<float> host_globals(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
      if (weights[expert].size() != weight_bytes || scales[expert].size() != scale_bytes ||
          globals[expert].size() != sizeof(float))
        throw std::runtime_error(std::string(name) + " expert tensor shape mismatch");
      check(cudaMemcpy(result.weights + expert * weight_bytes, weights[expert].data(), weight_bytes,
                       cudaMemcpyHostToDevice), "pack expert weight");
      check(cudaMemcpy(result.scales + expert * scale_bytes, scales[expert].data(), scale_bytes,
                       cudaMemcpyHostToDevice), "pack expert scale");
      host_globals[expert] = *reinterpret_cast<const float*>(globals[expert].data());
    }
    check(cudaMemcpy(result.globals, host_globals.data(), kExperts * sizeof(float),
                     cudaMemcpyHostToDevice), "pack expert globals");
    return result;
  };
  const auto gate = pack(gate_weights, gate_scales, gate_globals, "gate");
  const auto up = pack(up_weights, up_scales, up_globals, "up");
  const auto down = pack(down_weights, down_scales, down_globals, "down");

  const std::vector<int> expert_ids{52, 105, 55, 99, 116, 60, 3, 103};
  const std::vector<float> route_weights{0.24F, 0.19F, 0.15F, 0.12F, 0.10F, 0.08F, 0.07F, 0.05F};
  std::vector<__nv_bfloat16> input(static_cast<std::size_t>(kRoutes) * kHidden);
  for (int route = 0; route < kRoutes; ++route)
    for (int col = 0; col < kHidden; ++col)
      input[route * kHidden + col] = __float2bfloat16_rn(
          std::sin(col * 0.019F) + std::cos(col * 0.007F) * 0.31F);
  int* d_ids{};
  float* d_route_weights{};
  __nv_bfloat16 *d_input{}, *d_gate{}, *d_up{}, *d_product{}, *d_routed{}, *d_output{};
  check(cudaMalloc(&d_ids, kRoutes * sizeof(int)), "cudaMalloc(expert ids)");
  check(cudaMalloc(&d_route_weights, kRoutes * sizeof(float)), "cudaMalloc(route weights)");
  check(cudaMalloc(&d_input, input.size() * 2), "cudaMalloc(expert block input)");
  check(cudaMalloc(&d_gate, kRoutes * kIntermediate * 2), "cudaMalloc(expert gates)");
  check(cudaMalloc(&d_up, kRoutes * kIntermediate * 2), "cudaMalloc(expert ups)");
  check(cudaMalloc(&d_product, kRoutes * kIntermediate * 2), "cudaMalloc(expert products)");
  check(cudaMalloc(&d_routed, kRoutes * kHidden * 2), "cudaMalloc(routed outputs)");
  check(cudaMalloc(&d_output, kHidden * 2), "cudaMalloc(expert block output)");
  check(cudaMemcpy(d_ids, expert_ids.data(), kRoutes * sizeof(int), cudaMemcpyHostToDevice),
        "copy expert ids");
  check(cudaMemcpy(d_route_weights, route_weights.data(), kRoutes * sizeof(float),
                   cudaMemcpyHostToDevice), "copy route weights");
  check(cudaMemcpy(d_input, input.data(), input.size() * 2, cudaMemcpyHostToDevice),
        "copy expert block input");
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate(expert block)");
  auto launch = [&] {
    for (int route = 0; route < kRoutes; ++route) {
      nvfp4_routed_matvec_kernel<<<kIntermediate, 256, 0, stream>>>(
          gate.weights, gate.scales, gate.globals, d_ids, route, d_input, d_gate,
          kIntermediate, kHidden);
      nvfp4_routed_matvec_kernel<<<kIntermediate, 256, 0, stream>>>(
          up.weights, up.scales, up.globals, d_ids, route, d_input, d_up,
          kIntermediate, kHidden);
    }
    gelu_tanh_multiply_kernel<<<(kRoutes * kIntermediate + 255) / 256, 256, 0, stream>>>(
        d_gate, d_up, d_product, kRoutes * kIntermediate);
    for (int route = 0; route < kRoutes; ++route) {
      nvfp4_routed_matvec_kernel<<<kHidden, 256, 0, stream>>>(
          down.weights, down.scales, down.globals, d_ids, route, d_product, d_routed,
          kHidden, kIntermediate);
    }
    reduce_routed_experts_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(
        d_routed, d_route_weights, d_output);
  };
  launch();
  check(cudaGetLastError(), "expert block launch");
  check(cudaStreamSynchronize(stream), "expert block synchronization");
  std::vector<__nv_bfloat16> output(kHidden);
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy expert block output");

  auto cpu_project = [&](const std::vector<std::span<const std::byte>>& weights,
                         const std::vector<std::span<const std::byte>>& scales,
                         const std::vector<std::span<const std::byte>>& globals,
                         int expert, std::span<const __nv_bfloat16> activation,
                         int outputs, int inputs) {
    std::vector<__nv_bfloat16> result(outputs);
    const auto* weight = reinterpret_cast<const std::uint8_t*>(weights[expert].data());
    const auto* scale = reinterpret_cast<const std::uint8_t*>(scales[expert].data());
    const float global = *reinterpret_cast<const float*>(globals[expert].data());
    for (int row = 0; row < outputs; ++row) {
      float sum = 0.0F;
      for (int col = 0; col < inputs; ++col) {
        const unsigned packed = weight[row * (inputs / 2) + col / 2];
        const unsigned nibble = (col & 1) ? packed >> 4 : packed & 15U;
        sum += __bfloat162float(activation[col]) * decode_e2m1(nibble) *
               decode_e4m3(scale[row * (inputs / 16) + col / 16]) * global;
      }
      result[row] = __float2bfloat16_rn(sum);
    }
    return result;
  };
  std::vector<__nv_bfloat16> expected(kHidden, __float2bfloat16_rn(0.0F));
  for (int route = 0; route < kRoutes; ++route) {
    const int expert = expert_ids[route];
    const auto activation = std::span<const __nv_bfloat16>(input).subspan(route * kHidden, kHidden);
    auto gate_result = cpu_project(gate_weights, gate_scales, gate_globals, expert,
                                   activation, kIntermediate, kHidden);
    auto up_result = cpu_project(up_weights, up_scales, up_globals, expert,
                                 activation, kIntermediate, kHidden);
    std::vector<__nv_bfloat16> product(kIntermediate);
    for (int i = 0; i < kIntermediate; ++i) {
      const float x = __bfloat162float(gate_result[i]);
      constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
      const float gelu = 0.5F * x * (1.0F + std::tanh(kSqrtTwoOverPi * (x + 0.044715F * x * x * x)));
      product[i] = __float2bfloat16_rn(gelu * __bfloat162float(up_result[i]));
    }
    const auto down_result = cpu_project(down_weights, down_scales, down_globals, expert,
                                         product, kHidden, kIntermediate);
    for (int i = 0; i < kHidden; ++i) {
      const auto contribution = __float2bfloat16_rn(
          __bfloat162float(down_result[i]) * route_weights[route]);
      expected[i] = __float2bfloat16_rn(
          __bfloat162float(expected[i]) + __bfloat162float(contribution));
    }
  }
  float error = 0.0F;
  for (int i = 0; i < kHidden; ++i)
    error = std::max(error, std::abs(__bfloat162float(output[i]) - __bfloat162float(expected[i])));
  const float eager_microseconds = event_benchmark(stream, launch, 10, 200);
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin expert graph capture");
  launch();
  check(cudaStreamEndCapture(stream, &graph), "end expert graph capture");
  check(cudaGraphInstantiate(&graph_exec, graph), "instantiate expert graph");
  const float graph_microseconds = event_benchmark(
      stream, [&] { check(cudaGraphLaunch(graph_exec, stream), "launch expert graph"); }, 10, 500);
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(d_output); cudaFree(d_routed); cudaFree(d_product); cudaFree(d_up); cudaFree(d_gate);
  cudaFree(d_input); cudaFree(d_route_weights); cudaFree(d_ids);
  cudaFree(down.globals); cudaFree(down.scales); cudaFree(down.weights);
  cudaFree(up.globals); cudaFree(up.scales); cudaFree(up.weights);
  cudaFree(gate.globals); cudaFree(gate.scales); cudaFree(gate.weights);
  return {error, eager_microseconds, graph_microseconds};
}

FrontendTestResult test_bf16_frontend(std::span<const std::byte> embedding,
                                     std::span<const std::byte> norm_weight,
                                     std::span<const std::int32_t> input_ids,
                                     std::span<const float> expected_embedding,
                                     std::span<const float> expected_norm) {
  constexpr int kHidden = 2816;
  if (norm_weight.size() != kHidden * 2 ||
      expected_embedding.size() != input_ids.size() * kHidden ||
      expected_norm.size() != input_ids.size() * kHidden) {
    throw std::runtime_error("frontend differential input shape mismatch");
  }
  const auto* table = reinterpret_cast<const __nv_bfloat16*>(embedding.data());
  const auto* norm = reinterpret_cast<const __nv_bfloat16*>(norm_weight.data());
  const auto scale = __float2bfloat16_rn(std::sqrt(static_cast<float>(kHidden)));
  std::vector<__nv_bfloat16> host_hidden(input_ids.size() * kHidden);
  float embedding_error = 0.0F;
  for (std::size_t row = 0; row < input_ids.size(); ++row) {
    if (input_ids[row] < 0 ||
        (static_cast<std::uint64_t>(input_ids[row]) + 1) * kHidden * 2 > embedding.size()) {
      throw std::runtime_error("input id outside embedding table");
    }
    for (int col = 0; col < kHidden; ++col) {
      const auto value = __float2bfloat16_rn(
          __bfloat162float(table[static_cast<std::size_t>(input_ids[row]) * kHidden + col]) *
          __bfloat162float(scale));
      host_hidden[row * kHidden + col] = value;
      embedding_error = std::max(embedding_error,
          std::abs(__bfloat162float(value) - expected_embedding[row * kHidden + col]));
    }
  }

  __nv_bfloat16 *d_hidden{}, *d_norm{}, *d_output{};
  check(cudaMalloc(&d_hidden, host_hidden.size() * 2), "cudaMalloc(frontend hidden)");
  check(cudaMalloc(&d_norm, norm_weight.size()), "cudaMalloc(frontend norm)");
  check(cudaMalloc(&d_output, host_hidden.size() * 2), "cudaMalloc(frontend output)");
  check(cudaMemcpy(d_hidden, host_hidden.data(), host_hidden.size() * 2, cudaMemcpyHostToDevice),
        "copy frontend hidden");
  check(cudaMemcpy(d_norm, norm, norm_weight.size(), cudaMemcpyHostToDevice),
        "copy frontend norm");
  rmsnorm_2816_kernel<<<static_cast<int>(input_ids.size()), 256>>>(
      d_hidden, d_norm, d_output, static_cast<int>(input_ids.size()), 1e-6F);
  check(cudaGetLastError(), "frontend RMSNorm launch");
  std::vector<__nv_bfloat16> output(host_hidden.size());
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy frontend output");
  float norm_error = 0.0F;
  for (std::size_t i = 0; i < output.size(); ++i) {
    norm_error = std::max(norm_error,
                          std::abs(__bfloat162float(output[i]) - expected_norm[i]));
  }
  cudaFree(d_output);
  cudaFree(d_norm);
  cudaFree(d_hidden);
  return {embedding_error, norm_error};
}

AttentionTestResult test_bf16_layer0_attention(
    std::span<const std::byte> q_weight, std::span<const std::byte> k_weight,
    std::span<const std::byte> v_weight, std::span<const std::byte> o_weight,
    std::span<const std::byte> q_norm_weight, std::span<const std::byte> k_norm_weight,
    std::span<const float> normalized_input, std::span<const float> expected_output,
    std::span<const float> expected_probabilities) {
  constexpr int kTokens = 3;
  constexpr int kHidden = 2816;
  constexpr int kHeads = 16;
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 256;
  constexpr int kQWidth = kHeads * kHeadDim;
  constexpr int kKvWidth = kKvHeads * kHeadDim;
  auto require_bytes = [](std::span<const std::byte> value, std::size_t wanted,
                          const char* name) {
    if (value.size() != wanted) throw std::runtime_error(std::string(name) + " shape mismatch");
  };
  require_bytes(q_weight, static_cast<std::size_t>(kQWidth) * kHidden * 2, "q_proj");
  require_bytes(k_weight, static_cast<std::size_t>(kKvWidth) * kHidden * 2, "k_proj");
  require_bytes(v_weight, static_cast<std::size_t>(kKvWidth) * kHidden * 2, "v_proj");
  require_bytes(o_weight, static_cast<std::size_t>(kHidden) * kQWidth * 2, "o_proj");
  require_bytes(q_norm_weight, kHeadDim * 2, "q_norm");
  require_bytes(k_norm_weight, kHeadDim * 2, "k_norm");
  if (normalized_input.size() != kTokens * kHidden ||
      expected_output.size() != kTokens * kHidden ||
      expected_probabilities.size() != kHeads * kTokens * kTokens) {
    throw std::runtime_error("attention oracle shape mismatch");
  }

  std::vector<__nv_bfloat16> input(normalized_input.size());
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = __float2bfloat16_rn(normalized_input[i]);
  __nv_bfloat16 *d_input{}, *d_qw{}, *d_kw{}, *d_vw{}, *d_ow{}, *d_qnorm{}, *d_knorm{};
  __nv_bfloat16 *d_q{}, *d_k{}, *d_v{}, *d_kcache{}, *d_vcache{};
  __nv_bfloat16 *d_context{}, *d_output{}, *d_prob{};
  auto allocate_copy = [](__nv_bfloat16** pointer, std::span<const std::byte> source,
                          const char* name) {
    check(cudaMalloc(pointer, source.size()), name);
    check(cudaMemcpy(*pointer, source.data(), source.size(), cudaMemcpyHostToDevice), name);
  };
  check(cudaMalloc(&d_input, input.size() * 2), "cudaMalloc(attention input)");
  check(cudaMemcpy(d_input, input.data(), input.size() * 2, cudaMemcpyHostToDevice),
        "copy attention input");
  allocate_copy(&d_qw, q_weight, "q weight");
  allocate_copy(&d_kw, k_weight, "k weight");
  allocate_copy(&d_vw, v_weight, "v weight");
  allocate_copy(&d_ow, o_weight, "o weight");
  allocate_copy(&d_qnorm, q_norm_weight, "q norm weight");
  allocate_copy(&d_knorm, k_norm_weight, "k norm weight");
  check(cudaMalloc(&d_q, kTokens * kQWidth * 2), "cudaMalloc(q)");
  check(cudaMalloc(&d_k, kTokens * kKvWidth * 2), "cudaMalloc(k)");
  check(cudaMalloc(&d_v, kTokens * kKvWidth * 2), "cudaMalloc(v)");
  constexpr std::size_t kSlidingStorageTokens = 3 * 1024 + 4;
  check(cudaMalloc(&d_kcache, kSlidingStorageTokens * kKvWidth * 2),
        "cudaMalloc(sliding K cache)");
  check(cudaMalloc(&d_vcache, kSlidingStorageTokens * kKvWidth * 2),
        "cudaMalloc(sliding V cache)");
  check(cudaMalloc(&d_context, kTokens * kQWidth * 2), "cudaMalloc(context)");
  check(cudaMalloc(&d_output, kTokens * kHidden * 2), "cudaMalloc(attention output)");
  check(cudaMalloc(&d_prob, kHeads * kTokens * kTokens * 2), "cudaMalloc(probabilities)");

  cublasHandle_t handle{};
  check(cublasCreate(&handle), "cublasCreate(attention)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* weight, int width, __nv_bfloat16* output) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, width, kTokens, kHidden,
                       &alpha, weight, CUDA_R_16BF, kHidden,
                       d_input, CUDA_R_16BF, kHidden, &beta,
                       output, CUDA_R_16BF, width, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention projection");
  };
  project(d_qw, kQWidth, d_q);
  project(d_kw, kKvWidth, d_k);
  project(d_vw, kKvWidth, d_v);
  for (int token = 0; token < kTokens; ++token)
    launch_qkv_transform(false, d_q + token * kQWidth, d_k + token * kKvWidth,
                         d_v + token * kKvWidth, d_qnorm, d_knorm, token,
                         d_kcache, d_vcache, nullptr);
  attention_3x256_kernel<<<dim3(kHeads, kTokens), 256>>>(
      d_q, d_kcache, d_vcache, d_context, d_prob);
  check(cudaGetLastError(), "layer0 attention kernels");
  check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kHidden, kTokens, kQWidth,
                     &alpha, d_ow, CUDA_R_16BF, kQWidth,
                     d_context, CUDA_R_16BF, kQWidth, &beta,
                     d_output, CUDA_R_16BF, kHidden, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention output projection");
  std::vector<__nv_bfloat16> output(expected_output.size());
  std::vector<__nv_bfloat16> probabilities(expected_probabilities.size());
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy attention output");
  check(cudaMemcpy(probabilities.data(), d_prob, probabilities.size() * 2, cudaMemcpyDeviceToHost),
        "copy probabilities");
  float output_error = 0.0F;
  float probability_error = 0.0F;
  for (std::size_t i = 0; i < output.size(); ++i)
    output_error = std::max(output_error, std::abs(__bfloat162float(output[i]) - expected_output[i]));
  for (std::size_t i = 0; i < probabilities.size(); ++i)
    probability_error = std::max(probability_error,
        std::abs(__bfloat162float(probabilities[i]) - expected_probabilities[i]));
  cublasDestroy(handle);
  cudaFree(d_prob); cudaFree(d_output); cudaFree(d_context); cudaFree(d_vcache); cudaFree(d_kcache);
  cudaFree(d_v); cudaFree(d_k); cudaFree(d_q);
  cudaFree(d_knorm); cudaFree(d_qnorm); cudaFree(d_ow); cudaFree(d_vw); cudaFree(d_kw); cudaFree(d_qw);
  cudaFree(d_input);
  return {output_error, probability_error};
}

AttentionTestResult test_bf16_layer5_attention(
    std::span<const std::byte> q_weight, std::span<const std::byte> k_weight,
    std::span<const std::byte> o_weight, std::span<const std::byte> q_norm_weight,
    std::span<const std::byte> k_norm_weight, std::span<const float> normalized_input,
    std::span<const float> expected_output, std::span<const float> expected_probabilities) {
  constexpr int kTokens = 3;
  constexpr int kHidden = 2816;
  constexpr int kHeads = 16;
  constexpr int kKvHeads = 2;
  constexpr int kHeadDim = 512;
  constexpr int kQWidth = kHeads * kHeadDim;
  constexpr int kKvWidth = kKvHeads * kHeadDim;
  auto require_bytes = [](std::span<const std::byte> value, std::size_t wanted,
                          const char* name) {
    if (value.size() != wanted)
      throw std::runtime_error(std::string(name) + " shape mismatch");
  };
  require_bytes(q_weight, static_cast<std::size_t>(kQWidth) * kHidden * 2, "global q_proj");
  require_bytes(k_weight, static_cast<std::size_t>(kKvWidth) * kHidden * 2, "global k_proj");
  require_bytes(o_weight, static_cast<std::size_t>(kHidden) * kQWidth * 2, "global o_proj");
  require_bytes(q_norm_weight, kHeadDim * 2, "global q_norm");
  require_bytes(k_norm_weight, kHeadDim * 2, "global k_norm");
  if (normalized_input.size() != kTokens * kHidden ||
      expected_output.size() != kTokens * kHidden ||
      expected_probabilities.size() != kHeads * kTokens * kTokens)
    throw std::runtime_error("global attention oracle shape mismatch");

  std::vector<__nv_bfloat16> input(normalized_input.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = __float2bfloat16_rn(normalized_input[i]);
  std::vector<void*> allocations;
  auto allocate = [&](std::size_t bytes, const char* name) {
    void* pointer{};
    check(cudaMalloc(&pointer, bytes), name);
    allocations.push_back(pointer);
    return static_cast<__nv_bfloat16*>(pointer);
  };
  auto copy_weight = [&](std::span<const std::byte> source, const char* name) {
    auto* pointer = allocate(source.size(), name);
    check(cudaMemcpy(pointer, source.data(), source.size(), cudaMemcpyHostToDevice), name);
    return pointer;
  };
  auto* d_input = allocate(input.size() * 2, "cudaMalloc(global attention input)");
  check(cudaMemcpy(d_input, input.data(), input.size() * 2, cudaMemcpyHostToDevice),
        "copy global attention input");
  auto* d_qw = copy_weight(q_weight, "copy global q weight");
  auto* d_kw = copy_weight(k_weight, "copy global k weight");
  auto* d_ow = copy_weight(o_weight, "copy global o weight");
  auto* d_qnorm = copy_weight(q_norm_weight, "copy global q norm");
  auto* d_knorm = copy_weight(k_norm_weight, "copy global k norm");
  auto* d_q = allocate(kTokens * kQWidth * 2, "cudaMalloc(global q)");
  auto* d_k = allocate(kTokens * kKvWidth * 2, "cudaMalloc(global k)");
  auto* d_v = allocate(kTokens * kKvWidth * 2, "cudaMalloc(global v)");
  auto* d_kcache = allocate(kTokens * kKvWidth * 2, "cudaMalloc(global K cache)");
  auto* d_vcache = allocate(kTokens * kKvWidth * 2, "cudaMalloc(global V cache)");
  auto* d_context = allocate(kTokens * kQWidth * 2, "cudaMalloc(global context)");
  auto* d_output = allocate(kTokens * kHidden * 2, "cudaMalloc(global output)");
  auto* d_prob = allocate(kHeads * kTokens * kTokens * 2, "cudaMalloc(global probabilities)");
  cublasHandle_t handle{};
  check(cublasCreate(&handle), "cublasCreate(global attention)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* weight, int width, __nv_bfloat16* output) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, width, kTokens, kHidden,
                       &alpha, weight, CUDA_R_16BF, kHidden, d_input, CUDA_R_16BF,
                       kHidden, &beta, output, CUDA_R_16BF, width, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP), "global attention projection");
  };
  project(d_qw, kQWidth, d_q);
  project(d_kw, kKvWidth, d_k);
  for (int token = 0; token < kTokens; ++token)
    launch_qkv_transform(true, d_q + token * kQWidth, d_k + token * kKvWidth,
                         d_v + token * kKvWidth, d_qnorm, d_knorm, token,
                         d_kcache, d_vcache, nullptr);
  attention_3x512_kernel<<<dim3(kHeads, kTokens), 512>>>(
      d_q, d_kcache, d_vcache, d_context, d_prob);
  check(cudaGetLastError(), "layer5 attention kernels");
  check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kHidden, kTokens, kQWidth,
                     &alpha, d_ow, CUDA_R_16BF, kQWidth, d_context, CUDA_R_16BF,
                     kQWidth, &beta, d_output, CUDA_R_16BF, kHidden,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "global attention output projection");
  std::vector<__nv_bfloat16> output(expected_output.size());
  std::vector<__nv_bfloat16> probabilities(expected_probabilities.size());
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy global attention output");
  check(cudaMemcpy(probabilities.data(), d_prob, probabilities.size() * 2,
                   cudaMemcpyDeviceToHost), "copy global probabilities");
  AttentionTestResult result{};
  for (std::size_t i = 0; i < output.size(); ++i)
    result.output_max_abs_error = std::max(
        result.output_max_abs_error,
        std::abs(__bfloat162float(output[i]) - expected_output[i]));
  for (std::size_t i = 0; i < probabilities.size(); ++i)
    result.probability_max_abs_error = std::max(
        result.probability_max_abs_error,
        std::abs(__bfloat162float(probabilities[i]) - expected_probabilities[i]));
  cublasDestroy(handle);
  for (void* pointer : allocations) cudaFree(pointer);
  return result;
}

float test_bf16_layer0_dense_mlp(std::span<const std::byte> gate_weight,
                                 std::span<const std::byte> up_weight,
                                 std::span<const std::byte> down_weight,
                                 std::span<const float> input,
                                 std::span<const float> expected_output) {
  constexpr int kTokens = 3;
  constexpr int kHidden = 2816;
  constexpr int kIntermediate = 2112;
  if (gate_weight.size() != static_cast<std::size_t>(kIntermediate) * kHidden * 2 ||
      up_weight.size() != gate_weight.size() ||
      down_weight.size() != static_cast<std::size_t>(kHidden) * kIntermediate * 2 ||
      input.size() != kTokens * kHidden || expected_output.size() != input.size()) {
    throw std::runtime_error("dense MLP differential shape mismatch");
  }
  std::vector<__nv_bfloat16> host_input(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) host_input[i] = __float2bfloat16_rn(input[i]);
  __nv_bfloat16 *d_input{}, *d_gate_w{}, *d_up_w{}, *d_down_w{};
  __nv_bfloat16 *d_gate{}, *d_up{}, *d_product{}, *d_output{};
  auto allocate_copy = [](__nv_bfloat16** pointer, std::span<const std::byte> source,
                          const char* name) {
    check(cudaMalloc(pointer, source.size()), name);
    check(cudaMemcpy(*pointer, source.data(), source.size(), cudaMemcpyHostToDevice), name);
  };
  check(cudaMalloc(&d_input, host_input.size() * 2), "cudaMalloc(MLP input)");
  check(cudaMemcpy(d_input, host_input.data(), host_input.size() * 2, cudaMemcpyHostToDevice),
        "copy MLP input");
  allocate_copy(&d_gate_w, gate_weight, "copy gate weight");
  allocate_copy(&d_up_w, up_weight, "copy up weight");
  allocate_copy(&d_down_w, down_weight, "copy down weight");
  check(cudaMalloc(&d_gate, kTokens * kIntermediate * 2), "cudaMalloc(gate)");
  check(cudaMalloc(&d_up, kTokens * kIntermediate * 2), "cudaMalloc(up)");
  check(cudaMalloc(&d_product, kTokens * kIntermediate * 2), "cudaMalloc(product)");
  check(cudaMalloc(&d_output, kTokens * kHidden * 2), "cudaMalloc(MLP output)");
  cublasHandle_t handle{};
  check(cublasCreate(&handle), "cublasCreate(MLP)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto input_projection = [&](const __nv_bfloat16* weight, __nv_bfloat16* output) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kIntermediate, kTokens, kHidden,
                       &alpha, weight, CUDA_R_16BF, kHidden,
                       d_input, CUDA_R_16BF, kHidden, &beta,
                       output, CUDA_R_16BF, kIntermediate, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP), "MLP input projection");
  };
  input_projection(d_gate_w, d_gate);
  input_projection(d_up_w, d_up);
  constexpr int kElements = kTokens * kIntermediate;
  gelu_tanh_multiply_kernel<<<(kElements + 255) / 256, 256>>>(d_gate, d_up, d_product, kElements);
  check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kHidden, kTokens, kIntermediate,
                     &alpha, d_down_w, CUDA_R_16BF, kIntermediate,
                     d_product, CUDA_R_16BF, kIntermediate, &beta,
                     d_output, CUDA_R_16BF, kHidden, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP), "MLP down projection");
  std::vector<__nv_bfloat16> output(expected_output.size());
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy MLP output");
  float error = 0.0F;
  for (std::size_t i = 0; i < output.size(); ++i)
    error = std::max(error, std::abs(__bfloat162float(output[i]) - expected_output[i]));
  cublasDestroy(handle);
  cudaFree(d_output); cudaFree(d_product); cudaFree(d_up); cudaFree(d_gate);
  cudaFree(d_down_w); cudaFree(d_up_w); cudaFree(d_gate_w); cudaFree(d_input);
  return error;
}

float test_bf16_layer0_experts(std::span<const std::byte> gate_up_weights,
                               std::span<const std::byte> down_weights,
                               std::span<const float> input,
                               std::span<const float> top_weights,
                               std::span<const float> top_ids,
                               std::span<const float> expected_output) {
  constexpr int kExperts = 128;
  constexpr int kTokens = 3;
  constexpr int kTop = 8;
  constexpr int kHidden = 2816;
  constexpr int kIntermediate = 704;
  constexpr int kGateUp = 2 * kIntermediate;
  const std::size_t gate_up_expert_bytes = static_cast<std::size_t>(kGateUp) * kHidden * 2;
  const std::size_t down_expert_bytes = static_cast<std::size_t>(kHidden) * kIntermediate * 2;
  if (gate_up_weights.size() != kExperts * gate_up_expert_bytes ||
      down_weights.size() != kExperts * down_expert_bytes ||
      input.size() != kTokens * kHidden || top_weights.size() != kTokens * kTop ||
      top_ids.size() != kTokens * kTop || expected_output.size() != input.size()) {
    throw std::runtime_error("expert differential shape mismatch");
  }
  std::vector<__nv_bfloat16> host_input(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) host_input[i] = __float2bfloat16_rn(input[i]);
  __nv_bfloat16 *d_gate_up_w{}, *d_down_w{}, *d_input{};
  __nv_bfloat16 *d_gate_up{}, *d_product{}, *d_expert_output{}, *d_output{};
  check(cudaMalloc(&d_gate_up_w, gate_up_weights.size()), "cudaMalloc(expert gate-up weights)");
  check(cudaMalloc(&d_down_w, down_weights.size()), "cudaMalloc(expert down weights)");
  check(cudaMemcpy(d_gate_up_w, gate_up_weights.data(), gate_up_weights.size(), cudaMemcpyHostToDevice),
        "copy expert gate-up weights");
  check(cudaMemcpy(d_down_w, down_weights.data(), down_weights.size(), cudaMemcpyHostToDevice),
        "copy expert down weights");
  check(cudaMalloc(&d_input, host_input.size() * 2), "cudaMalloc(expert input)");
  check(cudaMemcpy(d_input, host_input.data(), host_input.size() * 2, cudaMemcpyHostToDevice),
        "copy expert input");
  check(cudaMalloc(&d_gate_up, kGateUp * 2), "cudaMalloc(expert gate-up)");
  check(cudaMalloc(&d_product, kIntermediate * 2), "cudaMalloc(expert product)");
  check(cudaMalloc(&d_expert_output, kHidden * 2), "cudaMalloc(expert output)");
  check(cudaMalloc(&d_output, kTokens * kHidden * 2), "cudaMalloc(combined expert output)");
  check(cudaMemset(d_output, 0, kTokens * kHidden * 2), "clear combined expert output");
  cublasHandle_t handle{};
  check(cublasCreate(&handle), "cublasCreate(experts)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  for (int token = 0; token < kTokens; ++token) {
    for (int position = 0; position < kTop; ++position) {
      const int expert = static_cast<int>(top_ids[token * kTop + position]);
      if (expert < 0 || expert >= kExperts) throw std::runtime_error("oracle expert id out of range");
      const auto* gate_up_weight = reinterpret_cast<const __nv_bfloat16*>(
          reinterpret_cast<const std::byte*>(d_gate_up_w) + expert * gate_up_expert_bytes);
      const auto* down_weight = reinterpret_cast<const __nv_bfloat16*>(
          reinterpret_cast<const std::byte*>(d_down_w) + expert * down_expert_bytes);
      check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kGateUp, 1, kHidden,
                         &alpha, gate_up_weight, CUDA_R_16BF, kHidden,
                         d_input + token * kHidden, CUDA_R_16BF, kHidden, &beta,
                         d_gate_up, CUDA_R_16BF, kGateUp, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), "expert gate-up projection");
      gelu_split_kernel<<<(kIntermediate + 255) / 256, 256>>>(d_gate_up, d_product);
      check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kHidden, 1, kIntermediate,
                         &alpha, down_weight, CUDA_R_16BF, kIntermediate,
                         d_product, CUDA_R_16BF, kIntermediate, &beta,
                         d_expert_output, CUDA_R_16BF, kHidden, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), "expert down projection");
      weighted_bf16_add_kernel<<<(kHidden + 255) / 256, 256>>>(
          d_output + token * kHidden, d_expert_output, top_weights[token * kTop + position]);
    }
  }
  std::vector<__nv_bfloat16> output(expected_output.size());
  check(cudaMemcpy(output.data(), d_output, output.size() * 2, cudaMemcpyDeviceToHost),
        "copy combined expert output");
  float error = 0.0F;
  for (std::size_t i = 0; i < output.size(); ++i)
    error = std::max(error, std::abs(__bfloat162float(output[i]) - expected_output[i]));
  cublasDestroy(handle);
  cudaFree(d_output); cudaFree(d_expert_output); cudaFree(d_product); cudaFree(d_gate_up);
  cudaFree(d_input); cudaFree(d_down_w); cudaFree(d_gate_up_w);
  return error;
}

RouterTestResult test_bf16_layer0_router(
    std::span<const std::byte> post_attention_norm_weight,
    std::span<const std::byte> pre_feedforward_norm_weight,
    std::span<const std::byte> router_scale, std::span<const std::byte> router_weight,
    std::span<const std::byte> per_expert_scale, std::span<const float> residual,
    std::span<const float> attention_output, std::span<const float> expected_pre_ff_norm,
    std::span<const float> expected_router_input, std::span<const float> expected_logits,
    std::span<const float> expected_probabilities, std::span<const float> expected_top_weights,
    std::span<const float> expected_top_ids) {
  constexpr int kRows = 3;
  constexpr int kHidden = 2816;
  constexpr int kExperts = 128;
  constexpr int kTop = 8;
  if (post_attention_norm_weight.size() != kHidden * 2 ||
      pre_feedforward_norm_weight.size() != kHidden * 2 || router_scale.size() != kHidden * 2 ||
      router_weight.size() != kExperts * kHidden * 2 || per_expert_scale.size() != kExperts * 2 ||
      residual.size() != kRows * kHidden || attention_output.size() != residual.size() ||
      expected_pre_ff_norm.size() != residual.size() ||
      expected_router_input.size() != residual.size() || expected_logits.size() != kRows * kExperts ||
      expected_probabilities.size() != kRows * kExperts ||
      expected_top_weights.size() != kRows * kTop || expected_top_ids.size() != kRows * kTop) {
    throw std::runtime_error("router differential shape mismatch");
  }
  std::vector<__nv_bfloat16> host_residual(residual.size()), host_attention(attention_output.size());
  for (std::size_t i = 0; i < residual.size(); ++i) {
    host_residual[i] = __float2bfloat16_rn(residual[i]);
    host_attention[i] = __float2bfloat16_rn(attention_output[i]);
  }
  __nv_bfloat16 *d_residual{}, *d_attention{}, *d_post_weight{}, *d_pre_weight{};
  __nv_bfloat16 *d_router_scale{}, *d_router_weight{}, *d_expert_scale{};
  __nv_bfloat16 *d_norm_attention{}, *d_combined{}, *d_pre_norm{}, *d_router_input{}, *d_logits{};
  __nv_bfloat16 *d_fused_combined{}, *d_fused_pre{}, *d_fused_expert{},
      *d_fused_router{};
  float *d_probabilities{}, *d_top_weights{};
  int* d_top_ids{};
  auto copy_bf16 = [](__nv_bfloat16** destination, std::span<const std::byte> source,
                      const char* name) {
    check(cudaMalloc(destination, source.size()), name);
    check(cudaMemcpy(*destination, source.data(), source.size(), cudaMemcpyHostToDevice), name);
  };
  check(cudaMalloc(&d_residual, host_residual.size() * 2), "cudaMalloc(router residual)");
  check(cudaMalloc(&d_attention, host_attention.size() * 2), "cudaMalloc(router attention)");
  check(cudaMemcpy(d_residual, host_residual.data(), host_residual.size() * 2, cudaMemcpyHostToDevice),
        "copy router residual");
  check(cudaMemcpy(d_attention, host_attention.data(), host_attention.size() * 2, cudaMemcpyHostToDevice),
        "copy router attention");
  copy_bf16(&d_post_weight, post_attention_norm_weight, "copy post-attention norm");
  copy_bf16(&d_pre_weight, pre_feedforward_norm_weight, "copy pre-ff norm");
  copy_bf16(&d_router_scale, router_scale, "copy router scale");
  copy_bf16(&d_router_weight, router_weight, "copy router weight");
  copy_bf16(&d_expert_scale, per_expert_scale, "copy per-expert scale");
  check(cudaMalloc(&d_norm_attention, residual.size() * 2), "cudaMalloc(norm attention)");
  check(cudaMalloc(&d_combined, residual.size() * 2), "cudaMalloc(combined residual)");
  check(cudaMalloc(&d_pre_norm, residual.size() * 2), "cudaMalloc(pre-ff norm)");
  check(cudaMalloc(&d_router_input, residual.size() * 2), "cudaMalloc(router input)");
  check(cudaMalloc(&d_fused_combined, residual.size() * 2),
        "cudaMalloc(fused combined residual)");
  check(cudaMalloc(&d_fused_pre, residual.size() * 2),
        "cudaMalloc(fused pre-ff norm)");
  check(cudaMalloc(&d_fused_expert, residual.size() * 2),
        "cudaMalloc(fused expert pre-ff norm)");
  check(cudaMalloc(&d_fused_router, residual.size() * 2),
        "cudaMalloc(fused router input)");
  check(cudaMalloc(&d_logits, kRows * kExperts * 2), "cudaMalloc(router logits)");
  check(cudaMalloc(&d_probabilities, kRows * kExperts * sizeof(float)), "cudaMalloc(router probs)");
  check(cudaMalloc(&d_top_weights, kRows * kTop * sizeof(float)), "cudaMalloc(top weights)");
  check(cudaMalloc(&d_top_ids, kRows * kTop * sizeof(int)), "cudaMalloc(top ids)");
  rmsnorm_2816_kernel<<<kRows, 256>>>(d_attention, d_post_weight, d_norm_attention, kRows, 1e-6F);
  bf16_add_kernel<<<(kRows * kHidden + 255) / 256, 256>>>(
      d_residual, d_norm_attention, d_combined, kRows * kHidden);
  rmsnorm_2816_kernel<<<kRows, 256>>>(d_combined, d_pre_weight, d_pre_norm, kRows, 1e-6F);
  router_input_kernel<<<kRows, 256>>>(d_combined, d_router_scale, d_router_input, kRows);
  rmsnorm_add_2816_kernel<<<kRows, 256>>>(
      d_attention, d_post_weight, d_residual, d_fused_combined, kRows);
  dual_rmsnorm_router_2816_kernel<<<kRows, 256>>>(
      d_fused_combined, d_pre_weight, d_pre_weight, d_router_scale,
      d_fused_pre, d_fused_expert, d_fused_router, kRows);
  cublasHandle_t handle{};
  check(cublasCreate(&handle), "cublasCreate(router)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kExperts, kRows, kHidden,
                     &alpha, d_router_weight, CUDA_R_16BF, kHidden,
                     d_router_input, CUDA_R_16BF, kHidden, &beta,
                     d_logits, CUDA_R_16BF, kExperts, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP), "router projection");
  softmax_128_kernel<<<kRows, kExperts>>>(d_logits, d_probabilities, kRows);
  route_128_top8_kernel<<<kRows, kExperts>>>(
      d_logits, d_expert_scale, d_top_weights, d_top_ids, kRows);
  std::vector<__nv_bfloat16> pre_norm(expected_pre_ff_norm.size());
  std::vector<__nv_bfloat16> router_input_result(expected_router_input.size());
  std::vector<__nv_bfloat16> combined_result(residual.size());
  std::vector<__nv_bfloat16> fused_combined(residual.size());
  std::vector<__nv_bfloat16> fused_pre(residual.size());
  std::vector<__nv_bfloat16> fused_router(residual.size());
  std::vector<__nv_bfloat16> logits_result(expected_logits.size());
  std::vector<float> probabilities(expected_probabilities.size()), top_result(expected_top_weights.size());
  std::vector<int> ids(expected_top_ids.size());
  check(cudaMemcpy(pre_norm.data(), d_pre_norm, pre_norm.size() * 2, cudaMemcpyDeviceToHost),
        "copy pre-ff norm result");
  check(cudaMemcpy(router_input_result.data(), d_router_input, router_input_result.size() * 2,
                   cudaMemcpyDeviceToHost), "copy router input result");
  check(cudaMemcpy(combined_result.data(), d_combined, combined_result.size() * 2,
                   cudaMemcpyDeviceToHost), "copy combined residual result");
  check(cudaMemcpy(fused_combined.data(), d_fused_combined, fused_combined.size() * 2,
                   cudaMemcpyDeviceToHost), "copy fused combined residual");
  check(cudaMemcpy(fused_pre.data(), d_fused_pre, fused_pre.size() * 2,
                   cudaMemcpyDeviceToHost), "copy fused pre-ff norm");
  check(cudaMemcpy(fused_router.data(), d_fused_router, fused_router.size() * 2,
                   cudaMemcpyDeviceToHost), "copy fused router input");
  check(cudaMemcpy(logits_result.data(), d_logits, logits_result.size() * 2,
                   cudaMemcpyDeviceToHost), "copy router logits result");
  check(cudaMemcpy(probabilities.data(), d_probabilities, probabilities.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "copy router probabilities");
  check(cudaMemcpy(top_result.data(), d_top_weights, top_result.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "copy top weights");
  check(cudaMemcpy(ids.data(), d_top_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy top ids");
  RouterTestResult result{};
  result.top_ids_match = true;
  for (std::size_t i = 0; i < pre_norm.size(); ++i)
    result.pre_feedforward_norm_max_abs_error = std::max(
        result.pre_feedforward_norm_max_abs_error,
        std::abs(__bfloat162float(pre_norm[i]) - expected_pre_ff_norm[i]));
  for (std::size_t i = 0; i < router_input_result.size(); ++i)
    result.input_max_abs_error = std::max(
        result.input_max_abs_error,
        std::abs(__bfloat162float(router_input_result[i]) - expected_router_input[i]));
  for (std::size_t i = 0; i < fused_pre.size(); ++i) {
    result.fused_max_abs_error = std::max(
        result.fused_max_abs_error,
        std::abs(__bfloat162float(fused_combined[i]) -
                 __bfloat162float(combined_result[i])));
    result.fused_max_abs_error = std::max(
        result.fused_max_abs_error,
        std::abs(__bfloat162float(fused_pre[i]) -
                 __bfloat162float(pre_norm[i])));
    result.fused_max_abs_error = std::max(
        result.fused_max_abs_error,
        std::abs(__bfloat162float(fused_router[i]) -
                 __bfloat162float(router_input_result[i])));
  }
  for (std::size_t i = 0; i < logits_result.size(); ++i)
    result.logits_max_abs_error = std::max(
        result.logits_max_abs_error,
        std::abs(__bfloat162float(logits_result[i]) - expected_logits[i]));
  for (std::size_t i = 0; i < probabilities.size(); ++i)
    result.probability_max_abs_error = std::max(
        result.probability_max_abs_error, std::abs(probabilities[i] - expected_probabilities[i]));
  for (std::size_t i = 0; i < top_result.size(); ++i) {
    result.top_weight_max_abs_error = std::max(
        result.top_weight_max_abs_error, std::abs(top_result[i] - expected_top_weights[i]));
    result.top_ids_match &= ids[i] == static_cast<int>(expected_top_ids[i]);
  }
  cublasDestroy(handle);
  cudaFree(d_top_ids); cudaFree(d_top_weights); cudaFree(d_probabilities); cudaFree(d_logits);
  cudaFree(d_fused_router); cudaFree(d_fused_expert); cudaFree(d_fused_pre);
  cudaFree(d_fused_combined); cudaFree(d_router_input); cudaFree(d_pre_norm);
  cudaFree(d_combined); cudaFree(d_norm_attention);
  cudaFree(d_expert_scale); cudaFree(d_router_weight); cudaFree(d_router_scale);
  cudaFree(d_pre_weight); cudaFree(d_post_weight); cudaFree(d_attention); cudaFree(d_residual);
  return result;
}

LayerTailTestResult test_bf16_layer0_tail(
    std::span<const std::byte> dense_norm_weight,
    std::span<const std::byte> expert_norm_weight,
    std::span<const std::byte> combined_norm_weight,
    std::span<const std::byte> post_attention_norm_weight,
    std::span<const std::byte> layer_scalar,
    std::span<const float> embedding, std::span<const float> attention_output,
    std::span<const float> dense_output, std::span<const float> expert_output,
    std::span<const float> expected_dense_norm, std::span<const float> expected_expert_norm,
    std::span<const float> expected_combined_norm, std::span<const float> expected_layer_output) {
  constexpr int kRows = 3;
  constexpr int kHidden = 2816;
  constexpr int kElements = kRows * kHidden;
  if (dense_norm_weight.size() != kHidden * 2 || expert_norm_weight.size() != kHidden * 2 ||
      combined_norm_weight.size() != kHidden * 2 || post_attention_norm_weight.size() != kHidden * 2 ||
      layer_scalar.size() != 2 ||
      embedding.size() != kElements || attention_output.size() != kElements ||
      dense_output.size() != kElements || expert_output.size() != kElements ||
      expected_dense_norm.size() != kElements || expected_expert_norm.size() != kElements ||
      expected_combined_norm.size() != kElements || expected_layer_output.size() != kElements) {
    throw std::runtime_error("layer tail differential shape mismatch");
  }
  auto to_bf16 = [](std::span<const float> source) {
    std::vector<__nv_bfloat16> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) result[i] = __float2bfloat16_rn(source[i]);
    return result;
  };
  const auto host_embedding = to_bf16(embedding);
  const auto host_attention = to_bf16(attention_output);
  const auto host_dense = to_bf16(dense_output);
  const auto host_expert = to_bf16(expert_output);
  std::vector<__nv_bfloat16*> allocations;
  auto allocate = [&](std::size_t bytes, const char* name) {
    __nv_bfloat16* pointer{};
    check(cudaMalloc(&pointer, bytes), name);
    allocations.push_back(pointer);
    return pointer;
  };
  auto device_vector = [&](const std::vector<__nv_bfloat16>& source, const char* name) {
    auto* pointer = allocate(source.size() * 2, name);
    check(cudaMemcpy(pointer, source.data(), source.size() * 2, cudaMemcpyHostToDevice), name);
    return pointer;
  };
  auto device_weight = [&](std::span<const std::byte> source, const char* name) {
    auto* pointer = allocate(source.size(), name);
    check(cudaMemcpy(pointer, source.data(), source.size(), cudaMemcpyHostToDevice), name);
    return pointer;
  };
  auto* d_embedding = device_vector(host_embedding, "copy tail embedding");
  auto* d_attention = device_vector(host_attention, "copy tail attention");
  auto* d_dense = device_vector(host_dense, "copy tail dense");
  auto* d_expert = device_vector(host_expert, "copy tail expert");
  auto* d_dense_weight = device_weight(dense_norm_weight, "copy dense norm weight");
  auto* d_expert_weight = device_weight(expert_norm_weight, "copy expert norm weight");
  auto* d_combined_weight = device_weight(combined_norm_weight, "copy combined norm weight");
  auto* d_post_attention_weight =
      device_weight(post_attention_norm_weight, "copy post-attention norm weight");
  auto* d_layer_scalar = device_weight(layer_scalar, "copy layer scalar");
  auto* d_dense_norm = allocate(kElements * 2, "cudaMalloc(dense norm)");
  auto* d_expert_norm = allocate(kElements * 2, "cudaMalloc(expert norm)");
  auto* d_sum = allocate(kElements * 2, "cudaMalloc(tail sum)");
  auto* d_combined_norm = allocate(kElements * 2, "cudaMalloc(combined norm)");
  auto* d_attention_norm = allocate(kElements * 2, "cudaMalloc(tail attention norm)");
  auto* d_residual = allocate(kElements * 2, "cudaMalloc(tail residual)");
  auto* d_layer_output = allocate(kElements * 2, "cudaMalloc(layer output)");
  auto* d_fused_residual = allocate(kElements * 2, "cudaMalloc(fused tail residual)");
  auto* d_fused_output = allocate(kElements * 2, "cudaMalloc(fused tail output)");
  rmsnorm_2816_kernel<<<kRows, 256>>>(d_dense, d_dense_weight, d_dense_norm, kRows, 1e-6F);
  rmsnorm_2816_kernel<<<kRows, 256>>>(d_expert, d_expert_weight, d_expert_norm, kRows, 1e-6F);
  bf16_add_kernel<<<(kElements + 255) / 256, 256>>>(d_dense_norm, d_expert_norm, d_sum, kElements);
  rmsnorm_2816_kernel<<<kRows, 256>>>(d_sum, d_combined_weight, d_combined_norm, kRows, 1e-6F);
  rmsnorm_2816_kernel<<<kRows, 256>>>(
      d_attention, d_post_attention_weight, d_attention_norm, kRows, 1e-6F);
  bf16_add_kernel<<<(kElements + 255) / 256, 256>>>(
      d_embedding, d_attention_norm, d_residual, kElements);
  bf16_add_kernel<<<(kElements + 255) / 256, 256>>>(
      d_residual, d_combined_norm, d_layer_output, kElements);
  const auto scalar = *reinterpret_cast<const __nv_bfloat16*>(layer_scalar.data());
  bf16_scale_kernel<<<(kElements + 255) / 256, 256>>>(d_layer_output, kElements, scalar);
  rmsnorm_add_2816_kernel<<<kRows, 256>>>(
      d_attention, d_post_attention_weight, d_embedding, d_fused_residual,
      kRows);
  finalize_feedforward_2816_kernel<<<kRows, 256>>>(
      d_dense, d_dense_weight, d_expert, d_expert_weight, d_combined_weight,
      d_fused_residual, d_layer_scalar, d_fused_output, kRows);
  std::vector<__nv_bfloat16> dense_norm(kElements), expert_norm(kElements), combined_norm(kElements),
      layer_output(kElements), residual_result(kElements),
      fused_residual(kElements), fused_output(kElements);
  check(cudaMemcpy(dense_norm.data(), d_dense_norm, kElements * 2, cudaMemcpyDeviceToHost),
        "copy dense norm");
  check(cudaMemcpy(expert_norm.data(), d_expert_norm, kElements * 2, cudaMemcpyDeviceToHost),
        "copy expert norm");
  check(cudaMemcpy(combined_norm.data(), d_combined_norm, kElements * 2, cudaMemcpyDeviceToHost),
        "copy combined norm");
  check(cudaMemcpy(layer_output.data(), d_layer_output, kElements * 2, cudaMemcpyDeviceToHost),
        "copy layer output");
  check(cudaMemcpy(residual_result.data(), d_residual, kElements * 2,
                   cudaMemcpyDeviceToHost), "copy tail residual");
  check(cudaMemcpy(fused_residual.data(), d_fused_residual, kElements * 2,
                   cudaMemcpyDeviceToHost), "copy fused tail residual");
  check(cudaMemcpy(fused_output.data(), d_fused_output, kElements * 2,
                   cudaMemcpyDeviceToHost), "copy fused tail output");
  LayerTailTestResult result{};
  for (int i = 0; i < kElements; ++i) {
    result.dense_norm_max_abs_error = std::max(
        result.dense_norm_max_abs_error,
        std::abs(__bfloat162float(dense_norm[i]) - expected_dense_norm[i]));
    result.expert_norm_max_abs_error = std::max(
        result.expert_norm_max_abs_error,
        std::abs(__bfloat162float(expert_norm[i]) - expected_expert_norm[i]));
    result.combined_norm_max_abs_error = std::max(
        result.combined_norm_max_abs_error,
        std::abs(__bfloat162float(combined_norm[i]) - expected_combined_norm[i]));
    result.layer_output_max_abs_error = std::max(
        result.layer_output_max_abs_error,
        std::abs(__bfloat162float(layer_output[i]) - expected_layer_output[i]));
    result.fused_max_abs_error = std::max(
        result.fused_max_abs_error,
        std::abs(__bfloat162float(fused_residual[i]) -
                 __bfloat162float(residual_result[i])));
    result.fused_max_abs_error = std::max(
        result.fused_max_abs_error,
        std::abs(__bfloat162float(fused_output[i]) -
                 __bfloat162float(layer_output[i])));
  }
  for (auto* pointer : allocations) cudaFree(pointer);
  return result;
}

