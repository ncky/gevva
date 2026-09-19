// Private implementation fragment; included only by src/gpu.cu.
__global__ void rmsnorm_1024_kernel(const __nv_bfloat16* input,
                                    const __nv_bfloat16* weight,
                                    __nv_bfloat16* output) {
  constexpr int kWidth = 1024;
  const int row = blockIdx.x;
  input += static_cast<std::size_t>(row) * kWidth;
  output += static_cast<std::size_t>(row) * kWidth;
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const float x = __bfloat162float(input[col]);
    sum = fmaf(x, x, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kWidth + 1e-6F);
  }
  __syncthreads();
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    output[col] = __float2bfloat16_rn(
        __bfloat162float(input[col]) * inverse_rms * __bfloat162float(weight[col]));
  }
}

__global__ void rmsnorm_add_scale_1024_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* residual, __nv_bfloat16* output,
    const __nv_bfloat16* output_scale = nullptr) {
  constexpr int kWidth = 1024;
  const int row = blockIdx.x;
  input += static_cast<std::size_t>(row) * kWidth;
  residual += static_cast<std::size_t>(row) * kWidth;
  output += static_cast<std::size_t>(row) * kWidth;
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  float sum = 0.0F;
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const float value = __bfloat162float(input[column]);
    sum = fmaf(value, value, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kWidth + 1e-6F);
  }
  __syncthreads();
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const auto normalized = __float2bfloat16_rn(
        __bfloat162float(input[column]) * inverse_rms *
        __bfloat162float(weight[column]));
    auto added = __float2bfloat16_rn(
        __bfloat162float(residual[column]) + __bfloat162float(normalized));
    if (output_scale)
      added = __float2bfloat16_rn(
          __bfloat162float(added) * __bfloat162float(*output_scale));
    output[column] = added;
  }
}

__global__ void argmax_bf16_partial_kernel(const __nv_bfloat16* values, int count,
                                           float* block_maxima,
                                           int* block_indices) {
  __shared__ float maxima[256];
  __shared__ int indices[256];
  float maximum = -INFINITY;
  int selected = 0;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += gridDim.x * blockDim.x) {
    const float value = __bfloat162float(values[index]);
    if (value > maximum || (value == maximum && index < selected)) {
      maximum = value;
      selected = index;
    }
  }
  maxima[threadIdx.x] = maximum;
  indices[threadIdx.x] = selected;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = maxima[threadIdx.x + stride];
      const int other_index = indices[threadIdx.x + stride];
      if (other > maxima[threadIdx.x] ||
          (other == maxima[threadIdx.x] && other_index < indices[threadIdx.x])) {
        maxima[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_maxima[blockIdx.x] = maxima[0];
    block_indices[blockIdx.x] = indices[0];
  }
}

__global__ void argmax_finish_kernel(const float* block_maxima,
                                     const int* block_indices, int blocks,
                                     int* result) {
  __shared__ float maxima[256];
  __shared__ int indices[256];
  const int lane = threadIdx.x;
  maxima[lane] = lane < blocks ? block_maxima[lane] : -INFINITY;
  indices[lane] = lane < blocks ? block_indices[lane] : 0;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (lane < stride) {
      const float other = maxima[lane + stride];
      const int other_index = indices[lane + stride];
      if (other > maxima[lane] ||
          (other == maxima[lane] && other_index < indices[lane])) {
        maxima[lane] = other;
        indices[lane] = other_index;
      }
    }
    __syncthreads();
  }
  if (lane == 0) *result = indices[0];
}

__global__ void quantize_int8_vector_kernel(const __nv_bfloat16* input,
                                            std::int8_t* output,
                                            float* output_scale, int count) {
  __shared__ float maxima[256];
  float maximum = 0.0F;
  for (int index = threadIdx.x; index < count; index += blockDim.x)
    maximum = fmaxf(maximum, fabsf(__bfloat162float(input[index])));
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const float scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0) *output_scale = scale;
  for (int index = threadIdx.x; index < count; index += blockDim.x)
    output[index] = static_cast<std::int8_t>(
        __float2int_rn(__bfloat162float(input[index]) / scale));
}

__global__ void quantize_int8_rows_kernel(const __nv_bfloat16* input,
                                          std::int8_t* output,
                                          float* output_scales, int columns,
                                          int rows) {
  __shared__ float maxima[256];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const auto* source = input + static_cast<std::size_t>(row) * columns;
  auto* destination = output + static_cast<std::size_t>(row) * columns;
  float maximum = 0.0F;
  for (int column = threadIdx.x; column < columns; column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(__bfloat162float(source[column])));
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const float scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0) output_scales[row] = scale;
  for (int column = threadIdx.x; column < columns; column += blockDim.x)
    destination[column] = static_cast<std::int8_t>(
        __float2int_rn(__bfloat162float(source[column]) / scale));
}

__global__ void scale_int32_logits_kernel(const std::int32_t* accumulated,
                                          const float* row_scales,
                                          const float* input_scale,
                                          __nv_bfloat16* logits, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    logits[index] = __float2bfloat16_rn(
        static_cast<float>(accumulated[index]) * row_scales[index] * *input_scale);
}

__global__ void local_top4_kernel(const __nv_bfloat16* logits,
                                  int* candidate_ids) {
  constexpr int kItems = 4;
  using Sort = cub::BlockRadixSort<float, 256, kItems, int>;
  __shared__ typename Sort::TempStorage storage;
  float keys[kItems];
  int values[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int index = blockIdx.x * 1024 + threadIdx.x * kItems + item;
    keys[item] = __bfloat162float(logits[index]);
    values[item] = index;
  }
  Sort(storage).SortDescending(keys, values);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int item = 0; item < kItems; ++item)
      candidate_ids[blockIdx.x * kItems + item] = values[item];
  }
}

// Select the per-tile candidates directly from the INT32 GEMM accumulator.
// The explicit BF16 round is intentional: candidate selection must have the
// same ordering as the former scale-to-BF16-logits followed by top-k path.
// Avoiding that intermediate removes a full rows x 262144 write and read from
// every verifier pass.
template <int kKeep>
__global__ void local_top_accumulated_rows_kernel(
    const std::int32_t* accumulated, const float* vocabulary_scales,
    const float* input_scales, int* candidate_ids, int rows) {
  constexpr int kItems = 4;
  constexpr int kBlocksPerRow = 256;
  constexpr int kVocabulary = 262144;
  using Sort = cub::BlockRadixSort<float, 256, kItems, int>;
  __shared__ typename Sort::TempStorage storage;
  const int row = blockIdx.x / kBlocksPerRow;
  const int tile = blockIdx.x % kBlocksPerRow;
  if (row >= rows) return;
  const float input_scale = input_scales[row];
  const std::size_t row_offset =
      static_cast<std::size_t>(row) * kVocabulary;
  float keys[kItems];
  int values[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int index = tile * 1024 + threadIdx.x * kItems + item;
    const float scaled =
        static_cast<float>(accumulated[row_offset + index]) *
        vocabulary_scales[index] * input_scale;
    keys[item] = __bfloat162float(__float2bfloat16_rn(scaled));
    values[item] = index;
  }
  Sort(storage).SortDescending(keys, values);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int item = 0; item < kKeep; ++item)
      candidate_ids[(static_cast<std::size_t>(row) * kBlocksPerRow + tile) *
                        kKeep +
                    item] = values[item];
  }
}

__global__ void correct_vocab_candidates_kernel(
    const __nv_bfloat16* weight, const __nv_bfloat16* activation, int columns,
    const int* candidate_ids, float* corrected) {
  __shared__ float warp_sums[8];
  float sum = 0.0F;
  const int row = candidate_ids[blockIdx.x];
  for (int col = threadIdx.x; col < columns; col += blockDim.x)
    sum = fmaf(__bfloat162float(weight[static_cast<std::size_t>(row) * columns + col]),
               __bfloat162float(activation[col]), sum);
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      corrected[blockIdx.x] =
          __bfloat162float(__float2bfloat16_rn(sum));
  }
}

__global__ void correct_vocab_candidates_rows_kernel(
    const __nv_bfloat16* weight, const __nv_bfloat16* activations, int columns,
    const int* candidate_ids, float* corrected, int rows, int candidates) {
  __shared__ float warp_sums[8];
  const int token = blockIdx.x / candidates;
  const int candidate = blockIdx.x % candidates;
  if (token >= rows) return;
  const int vocabulary_row =
      candidate_ids[static_cast<std::size_t>(token) * candidates + candidate];
  const auto* activation =
      activations + static_cast<std::size_t>(token) * columns;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < columns; col += blockDim.x)
    sum = fmaf(__bfloat162float(
                   weight[static_cast<std::size_t>(vocabulary_row) * columns +
                          col]),
               __bfloat162float(activation[col]), sum);
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      corrected[static_cast<std::size_t>(token) * candidates + candidate] =
          __bfloat162float(__float2bfloat16_rn(sum));
  }
}

__global__ void argmax_corrected_kernel(const float* corrected,
                                        const int* candidate_ids,
                                        int* result) {
  __shared__ float maxima[256];
  __shared__ int indices[256];
  float maximum = -INFINITY;
  int selected = 0;
  for (int item = 0; item < 4; ++item) {
    const int candidate = threadIdx.x * 4 + item;
    const float value = corrected[candidate];
    const int index = candidate_ids[candidate];
    if (value > maximum || (value == maximum && index < selected)) {
      maximum = value;
      selected = index;
    }
  }
  maxima[threadIdx.x] = maximum;
  indices[threadIdx.x] = selected;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = maxima[threadIdx.x + stride];
      const int other_index = indices[threadIdx.x + stride];
      if (other > maxima[threadIdx.x] ||
          (other == maxima[threadIdx.x] && other_index < indices[threadIdx.x])) {
        maxima[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = indices[0];
}

__global__ void argmax_corrected_rows_kernel(const float* corrected,
                                             const int* candidate_ids,
                                             int* result, int rows,
                                             int candidates) {
  __shared__ float maxima[256];
  __shared__ int indices[256];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const auto* row_values =
      corrected + static_cast<std::size_t>(row) * candidates;
  const auto* row_ids =
      candidate_ids + static_cast<std::size_t>(row) * candidates;
  float maximum = -INFINITY;
  int selected = 0;
  for (int candidate = threadIdx.x; candidate < candidates;
       candidate += blockDim.x) {
    const float value = row_values[candidate];
    const int index = row_ids[candidate];
    if (value > maximum || (value == maximum && index < selected)) {
      maximum = value;
      selected = index;
    }
  }
  maxima[threadIdx.x] = maximum;
  indices[threadIdx.x] = selected;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = maxima[threadIdx.x + stride];
      const int other_index = indices[threadIdx.x + stride];
      if (other > maxima[threadIdx.x] ||
          (other == maxima[threadIdx.x] && other_index < indices[threadIdx.x])) {
        maxima[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) result[row] = indices[0];
}

__global__ void prepare_next_assistant_input_kernel(
    const __nv_bfloat16* target_embedding, const int* token,
    const __nv_bfloat16* projected_state, __nv_bfloat16* combined) {
  constexpr int kHidden = 2816;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= kHidden) return;
  const auto scale = __float2bfloat16_rn(sqrtf(static_cast<float>(kHidden)));
  const auto embedding = target_embedding[
      static_cast<std::size_t>(*token) * kHidden + col];
  combined[col] = __float2bfloat16_rn(
      __bfloat162float(embedding) * __bfloat162float(scale));
  combined[kHidden + col] = projected_state[col];
}

__global__ void route_128_top8_kernel(const __nv_bfloat16* logits,
                                      const __nv_bfloat16* expert_scale,
                                      float* weights, int* ids, int rows,
                                      const float* activation_scales = nullptr,
                                      const float* projection_weight_scale = nullptr) {
  constexpr int kExperts = 128;
  using Sort = cub::BlockRadixSort<float, kExperts, 1, int>;
  __shared__ typename Sort::TempStorage sort_storage;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  float key_value = __bfloat162float(logits[row * kExperts + lane]);
  if (activation_scales) {
    const float scale = activation_scales[row] * projection_weight_scale[0];
    key_value = __bfloat162float(__float2bfloat16_rn(key_value * scale));
  }
  float keys[1] = {key_value};
  int values[1] = {lane};
  Sort(sort_storage).SortDescending(keys, values);
  __syncthreads();

  const float key = keys[0];
  const int value = values[0];
  // Stable softmax: sorted lane zero is the maximum.
  const float maximum = __shfl_sync(0xffffffff, key, 0);
  float exponential = lane < 8 ? expf(key - maximum) : 0.0F;
  for (int offset = 16; offset; offset >>= 1) {
    exponential += __shfl_down_sync(0xffffffff, exponential, offset);
  }
  const float denominator = __shfl_sync(0xffffffff, exponential, 0);
  if (lane < 8) {
    ids[row * 8 + lane] = value;
    weights[row * 8 + lane] = expf(key - maximum) / denominator *
                              __bfloat162float(expert_scale[value]);
  }
}

