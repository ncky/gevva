// Private implementation fragment; included only by src/gpu.cu.
__global__ void summarize_decision_logits_kernel(
    const float* input, const std::uint32_t* aliases, float* summary, int alias_count) {
  constexpr int vocab = 262144;
  __shared__ float maxima[256], sums[256];
  __shared__ int ids[256];
  const auto* row = input + static_cast<std::size_t>(blockIdx.x) * vocab;
  float maximum = -INFINITY;
  int best = vocab;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float value = 30.0F * tanhf(row[i] / 30.0F);
    if (value > maximum || (value == maximum && i < best)) { maximum = value; best = i; }
  }
  maxima[threadIdx.x] = maximum;
  ids[threadIdx.x] = best;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int other = threadIdx.x + stride;
      if (maxima[other] > maxima[threadIdx.x] ||
          (maxima[other] == maxima[threadIdx.x] && ids[other] < ids[threadIdx.x])) {
        maxima[threadIdx.x] = maxima[other]; ids[threadIdx.x] = ids[other];
      }
    }
    __syncthreads();
  }
  maximum = maxima[0];
  float total = 0;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x)
    total += expf(30.0F * tanhf(row[i] / 30.0F) - maximum);
  sums[threadIdx.x] = total;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  auto* result = summary + blockIdx.x * (alias_count + 2);
  if (threadIdx.x == 0) { result[0] = maximum + logf(sums[0]); result[1] = ids[0]; }
  if (threadIdx.x < alias_count) result[2 + threadIdx.x] = 30.0F * tanhf(row[aliases[threadIdx.x]] / 30.0F);
}

// Pad only attention workspaces. Model hidden rows, RoPE positions and the
// valid causal attention graph remain unchanged; dummy rows are discarded.
__global__ void pack_decision_queries_kernel(
    const uint4* source, uint4* destination, const int* offsets,
    int sessions, int query_tokens, int vectors_per_token, bool unpack) {
  const std::size_t size = static_cast<std::size_t>(sessions) * query_tokens * vectors_per_token;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    const int col = i % vectors_per_token;
    const int row = i / vectors_per_token;
    const int session = row / query_tokens, token = row % query_tokens;
    const bool valid = token < offsets[session + 1] - offsets[session];
    const auto packed = static_cast<std::size_t>(offsets[session] + token) * vectors_per_token + col;
    if (unpack) { if (valid) destination[packed] = source[i]; }
    else destination[i] = valid ? source[packed] : make_uint4(0, 0, 0, 0);
  }
}

__global__ void pack_decision_keys_kernel(
    const uint4* keys, const uint4* values, const uint4* past_keys, const uint4* past_values,
    uint4* combined_keys, uint4* combined_values, __nv_bfloat16** key_views,
    __nv_bfloat16** value_views, const int* offsets, int sessions,
    int query_tokens, int prefix_tokens, bool global, int key_tokens) {
  const int vectors_per_token = (global ? 1024 : 2048) / 8;
  const int past = global ? prefix_tokens : min(prefix_tokens, 1024);
  const std::size_t size = static_cast<std::size_t>(sessions) * key_tokens * vectors_per_token;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    const int col = i % vectors_per_token;
    const int row = i / vectors_per_token;
    const int session = row / key_tokens, token = row % key_tokens;
    uint4 k = make_uint4(0, 0, 0, 0), v = k;
    if (token < past) {
      const int position = global ? token : ((prefix_tokens - past + token) & 1023);
      const auto index = static_cast<std::size_t>(position) * vectors_per_token + col;
      k = past_keys[index]; v = past_values[index];
    } else if (token - past < offsets[session + 1] - offsets[session]) {
      const auto index = static_cast<std::size_t>(offsets[session] + token - past) * vectors_per_token + col;
      k = keys[index]; v = values[index];
    }
    combined_keys[i] = k; combined_values[i] = v;
    if (token == 0 && col == 0) {
      key_views[session] = reinterpret_cast<__nv_bfloat16*>(combined_keys + i);
      value_views[session] = reinterpret_cast<__nv_bfloat16*>(combined_values + i);
    }
  }
}

// Prefix matrices group every question's Q rows for each head. The small
// suffix matrices remain independent, while both write one head-major score
// tensor. This avoids replicating the immutable global prefix K/V.
template <int kHeadDim, int kKvHeads>
__global__ void prepare_segmented_attention_pointers_kernel(
    const __nv_bfloat16* queries, const __nv_bfloat16* prefix_keys,
    const __nv_bfloat16* prefix_values, const __nv_bfloat16* suffix_keys,
    const __nv_bfloat16* suffix_values, __nv_bfloat16* scores, float* output,
    void** pointers, int sessions, int query_tokens, int prefix_tokens, int key_stride) {
  constexpr int kQueryWidth = 16 * kHeadDim, kKvWidth = kKvHeads * kHeadDim;
  const int matrices = 16 * (sessions + 1);
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= matrices) return;
  const bool prefix = id < 16;
  const int head = prefix ? id : (id - 16) / sessions;
  const int session = prefix ? 0 : (id - 16) % sessions;
  const int kv_head = head / (16 / kKvHeads);
  const auto query_row = static_cast<std::size_t>(session) * query_tokens;
  const auto score_row = (static_cast<std::size_t>(head) * sessions + session) * query_tokens;
  const auto* keys = prefix ? prefix_keys : suffix_keys + query_row * kKvWidth;
  const auto* values = prefix ? prefix_values : suffix_values + query_row * kKvWidth;
  pointers[id] = const_cast<__nv_bfloat16*>(keys + kv_head * kHeadDim);
  pointers[matrices + id] = const_cast<__nv_bfloat16*>(queries + query_row * kQueryWidth + head * kHeadDim);
  pointers[2 * matrices + id] = scores + score_row * key_stride + (prefix ? 0 : prefix_tokens);
  pointers[3 * matrices + id] = const_cast<__nv_bfloat16*>(values + kv_head * kHeadDim);
  pointers[4 * matrices + id] = pointers[2 * matrices + id];
  pointers[5 * matrices + id] = output + score_row * kHeadDim;
}

template <int kHeadDim>
__global__ void decision_float_to_bf16_kernel(const float* input, __nv_bfloat16* output, std::size_t count, int tokens) {
  constexpr int kQueryWidth = 16 * kHeadDim;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    output[i] = __float2bfloat16_rn(input[((i % kQueryWidth) / kHeadDim * tokens + i / kQueryWidth) * kHeadDim + i % kHeadDim]);
}

__global__ void decision_segmented_softmax_kernel(__nv_bfloat16* scores, int queries, int stride,
                                                  int past, int storage_past, bool sliding) {
  __shared__ float reduction[256];
  const int query = blockIdx.x % queries;
  auto* row = scores + static_cast<std::size_t>(blockIdx.x) * stride;
  auto allowed = [&](int key) {
    if (key >= past && key < storage_past) return false;
    const int position = key < past ? key : past + key - storage_past;
    return position <= past + query && (!sliding || position > past + query - 1024);
  };
  float maximum = -INFINITY;
  for (int key = threadIdx.x; key < stride; key += blockDim.x)
    if (allowed(key)) maximum = fmaxf(maximum, __bfloat162float(row[key]));
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int step = 128; step; step >>= 1) {
    if (threadIdx.x < step) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + step]);
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float total = 0;
  for (int key = threadIdx.x; key < stride; key += blockDim.x)
    if (allowed(key)) total += expf(__bfloat162float(row[key]) - maximum);
  reduction[threadIdx.x] = total;
  __syncthreads();
  for (int step = 128; step; step >>= 1) {
    if (threadIdx.x < step) reduction[threadIdx.x] += reduction[threadIdx.x + step];
    __syncthreads();
  }
  total = reduction[0];
  for (int key = threadIdx.x; key < stride; key += blockDim.x)
    row[key] = __float2bfloat16_rn(allowed(key) ? expf(__bfloat162float(row[key]) - maximum) / total : 0.0F);
}

template <int kHeadDim, int kKvHeads>
void launch_segmented_attention(
    cublasHandle_t handle, const __nv_bfloat16* queries,
    const __nv_bfloat16* prefix_keys, const __nv_bfloat16* prefix_values,
    const __nv_bfloat16* suffix_keys, const __nv_bfloat16* suffix_values,
    __nv_bfloat16* scores,
    float* accumulation, __nv_bfloat16* output, void* pointer_workspace,
    int sessions, int query_tokens, int prefix_tokens, int key_stride, bool sliding, cudaStream_t stream) {
  constexpr int kQueryWidth = 16 * kHeadDim, kKvWidth = kKvHeads * kHeadDim;
  const int matrices = 16 * (sessions + 1);
  const int storage_prefix = (prefix_tokens + 7) & ~7;
  auto** pointers = static_cast<void**>(pointer_workspace);
  prepare_segmented_attention_pointers_kernel<kHeadDim, kKvHeads><<<(matrices + 127) / 128, 128, 0, stream>>>(
      queries, prefix_keys, prefix_values, suffix_keys, suffix_values, scores, accumulation,
      pointers, sessions, query_tokens, storage_prefix, key_stride);
  const float one = 1.0F, zero = 0.0F;
  auto qk = [&](bool prefix) {
    const int offset = prefix ? 0 : 16;
    check(cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        prefix ? prefix_tokens : query_tokens, prefix ? sessions * query_tokens : query_tokens,
        kHeadDim, &one, pointers + offset, CUDA_R_16BF, kKvWidth,
        pointers + matrices + offset, CUDA_R_16BF, kQueryWidth, &zero,
        pointers + 2 * matrices + offset, CUDA_R_16BF, key_stride,
        prefix ? 16 : sessions * 16, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "segmented decision QK");
  };
  if (prefix_tokens) qk(true);
  qk(false);
  decision_segmented_softmax_kernel<<<sessions * 16 * query_tokens, 256, 0, stream>>>(
      scores, query_tokens, key_stride, prefix_tokens, storage_prefix, sliding);
  auto pv = [&](bool prefix) {
    const int offset = prefix ? 0 : 16;
    check(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        kHeadDim, prefix ? sessions * query_tokens : query_tokens,
        prefix ? prefix_tokens : query_tokens, &one,
        pointers + 3 * matrices + offset, CUDA_R_16BF, kKvWidth,
        pointers + 4 * matrices + offset, CUDA_R_16BF, key_stride,
        prefix || !prefix_tokens ? &zero : &one,
        pointers + 5 * matrices + offset, CUDA_R_32F, kHeadDim,
        prefix ? 16 : sessions * 16, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        prefix ? "segmented decision prefix PV" : "segmented decision suffix PV");
  };
  if (prefix_tokens) pv(true);
  pv(false);
  const std::size_t values = static_cast<std::size_t>(sessions) * query_tokens * kQueryWidth;
  decision_float_to_bf16_kernel<kHeadDim><<<std::min<std::size_t>(65535, (values + 255) / 256), 256, 0, stream>>>(
      accumulation, output, values, sessions * query_tokens);
  check(cudaGetLastError(), "segmented decision attention");
}

__global__ void gather_decision_embeddings_kernel(const __nv_bfloat16* embedding,
    const std::uint32_t* aliases, __nv_bfloat16* selected, int count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count * 2816; i += blockDim.x * gridDim.x)
    selected[i] = embedding[static_cast<std::size_t>(aliases[i / 2816]) * 2816 + i % 2816];
}
__global__ void summarize_decision_options_kernel(const float* input, float* summary, int count) {
  if (threadIdx.x == 0) { summary[blockIdx.x * (count + 2)] = NAN; summary[blockIdx.x * (count + 2) + 1] = 0; }
  if (threadIdx.x < count)
    summary[blockIdx.x * (count + 2) + 2 + threadIdx.x] = 30.0F * tanhf(input[blockIdx.x * count + threadIdx.x] / 30.0F);
}
