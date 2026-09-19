// Private implementation fragment; included only by src/gpu.cu.
template <int kHeadDim>
__global__ void norm_rope_ragged_kernel(
    __nv_bfloat16* vectors, const __nv_bfloat16* weight, int rows, int heads,
    const int* context_lengths, int tokens_per_session, bool global,
    const float* activation_scales = nullptr,
    const float* weight_scale = nullptr) {
  __shared__ float reductions[512];
  __shared__ __nv_bfloat16 normed[512];
  const int vector = blockIdx.x;
  const int row = vector / heads;
  if (row >= rows) return;
  const int col = threadIdx.x;
  float x = __bfloat162float(vectors[vector * kHeadDim + col]);
  if (activation_scales) {
    const float scale = activation_scales[row] * weight_scale[0];
    x = __bfloat162float(__float2bfloat16_rn(
        x * scale));
  }
  reductions[col] = x * x;
  __syncthreads();
  for (int stride = kHeadDim / 2; stride; stride >>= 1) {
    if (col < stride) reductions[col] += reductions[col + stride];
    __syncthreads();
  }
  x *= rsqrtf(reductions[0] / kHeadDim + 1e-6F);
  x *= __bfloat162float(weight[col]);
  normed[col] = __float2bfloat16_rn(x);
  __syncthreads();
  const int session = row / tokens_per_session;
  const int step = row % tokens_per_session;
  const int position = context_lengths[session] + step;
  const int half = kHeadDim / 2;
  const int half_col = col & (half - 1);
  const float frequency = global
      ? (half_col < 64
             ? powf(1000000.0F, -2.0F * half_col / kHeadDim)
             : 0.0F)
      : powf(10000.0F, -2.0F * half_col / kHeadDim);
  const auto cosine = __float2bfloat16_rn(cosf(position * frequency));
  const auto sine = __float2bfloat16_rn(sinf(position * frequency));
  const int paired = col < half ? col + half : col - half;
  const auto left = __float2bfloat16_rn(
      __bfloat162float(normed[col]) * __bfloat162float(cosine));
  const float rotated = col < half ? -__bfloat162float(normed[paired])
                                   : __bfloat162float(normed[paired]);
  const auto right = __float2bfloat16_rn(
      rotated * __bfloat162float(sine));
  vectors[vector * kHeadDim + col] = __float2bfloat16_rn(
      __bfloat162float(left) + __bfloat162float(right));
}

namespace {
constexpr int kRopePositions = 262144;

template <int Dim, bool Global>
__device__ __nv_bfloat162 compute_target_rope(int position, int half_col) {
  const float frequency = Global
      ? (half_col < 64 ? powf(1000000.0F, -2.0F * half_col / Dim) : 0.0F)
      : powf(10000.0F, -2.0F * half_col / Dim);
  return __halves2bfloat162(__float2bfloat16_rn(cosf(position * frequency)),
                           __float2bfloat16_rn(sinf(position * frequency)));
}

template <int Dim, bool Global>
__global__ void initialize_target_rope(__nv_bfloat162* table) {
  constexpr int width = Global ? 64 : 128;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kRopePositions * width)
    table[index] = compute_target_rope<Dim, Global>(index / width, index % width);
}

const __nv_bfloat162* precomputed_target_rope() {
  struct Table {
    __nv_bfloat162* data{};
    Table() {
      check(cudaMalloc(&data, static_cast<std::size_t>(kRopePositions) * 192 * 4),
            "allocate native-context rotary table");
      initialize_target_rope<256, false><<<(kRopePositions * 128 + 255) / 256, 256>>>(data);
      initialize_target_rope<512, true><<<(kRopePositions * 64 + 255) / 256, 256>>>(
          data + static_cast<std::size_t>(kRopePositions) * 128);
      check(cudaGetLastError(), "initialize rotary table");
      check(cudaStreamSynchronize(nullptr), "finish rotary table initialization");
    }
    ~Table() { if (data) cudaFree(data); }
  };
  static const Table table;
  return table.data;
}
}  // namespace

// Decode and prefill project Q/K/V concurrently, after which all three head transforms
// have the same row geometry and no dependency on one another. Combining them
// removes two launches per layer (and the global K->V copy) while retaining
// the reduction tree and BF16 rounding of the separate kernels.
template <int kHeadDim, int kKvHeads, bool kGlobal>
__global__ void transform_qkv_ragged_kernel(
    __nv_bfloat16* queries, __nv_bfloat16* keys, __nv_bfloat16* values,
    const __nv_bfloat16* query_weight, const __nv_bfloat16* key_weight,
    int rows, const int* context_lengths, int tokens_per_session,
    const float* query_activation_scales, const float* query_weight_scale,
    const float* key_activation_scales, const float* key_weight_scale,
    const float* value_activation_scales, const float* value_weight_scale,
    int fixed_position_base, const __nv_bfloat162* rope_table = nullptr) {
  constexpr int kQueryHeads = 16;
  __shared__ float key_reductions[512];
  __shared__ float value_reductions[512];
  __shared__ __nv_bfloat16 key_normed[512];
  const int vector = blockIdx.x;
  const int query_vectors = rows * kQueryHeads;
  const bool is_query = vector < query_vectors;
  const int local_vector = is_query ? vector : vector - query_vectors;
  const int row = is_query ? vector / kQueryHeads : local_vector / kKvHeads;
  if (row >= rows) return;
  const int col = threadIdx.x;

  auto* source = is_query ? queries : keys;
  const int source_vector = is_query ? vector : local_vector;
  float x = __bfloat162float(
      source[static_cast<std::size_t>(source_vector) * kHeadDim + col]);
  const float* activation_scales =
      is_query ? query_activation_scales : key_activation_scales;
  const float* output_weight_scale =
      is_query ? query_weight_scale : key_weight_scale;
  if (activation_scales) {
    const float scale = activation_scales[row] * output_weight_scale[0];
    x = __bfloat162float(__float2bfloat16_rn(
        x * scale));
  }
  key_reductions[col] = x * x;

  float value_x = 0.0F;
  if (!is_query) {
    // Global attention's V projection is tied to K. Read it before K is
    // overwritten and write its independently normalized result to values.
    value_x = kGlobal
        ? __bfloat162float(
              keys[static_cast<std::size_t>(local_vector) * kHeadDim + col])
        : __bfloat162float(
              values[static_cast<std::size_t>(local_vector) * kHeadDim + col]);
    const float* scales = kGlobal ? key_activation_scales
                                  : value_activation_scales;
    const float* weight_scale = kGlobal ? key_weight_scale
                                        : value_weight_scale;
    if (scales) {
      const float scale = scales[row] * weight_scale[0];
      value_x = __bfloat162float(__float2bfloat16_rn(
          value_x * scale));
    }
    value_reductions[col] = value_x * value_x;
  }
  __syncthreads();
  for (int stride = kHeadDim / 2; stride; stride >>= 1) {
    if (col < stride) {
      key_reductions[col] += key_reductions[col + stride];
      if (!is_query)
        value_reductions[col] += value_reductions[col + stride];
    }
    __syncthreads();
  }

  x *= rsqrtf(key_reductions[0] / kHeadDim + 1e-6F);
  x *= __bfloat162float((is_query ? query_weight : key_weight)[col]);
  key_normed[col] = __float2bfloat16_rn(x);
  if (!is_query) {
    values[static_cast<std::size_t>(local_vector) * kHeadDim + col] =
        __float2bfloat16_rn(
            value_x * rsqrtf(value_reductions[0] / kHeadDim + 1e-6F));
  }
  __syncthreads();

  const int session = row / tokens_per_session;
  const int step = row % tokens_per_session;
  const int position = fixed_position_base >= 0
      ? fixed_position_base + row
      : context_lengths[session] + step;
  const int half = kHeadDim / 2;
  const int half_col = col & (half - 1);
  __nv_bfloat162 rope;
  if (rope_table && position >= 0 && position < kRopePositions) {
    if constexpr (kGlobal) {
      rope = half_col < 64
          ? rope_table[static_cast<std::size_t>(kRopePositions) * 128 +
                       static_cast<std::size_t>(position) * 64 + half_col]
          : __halves2bfloat162(__float2bfloat16_rn(1.0F), __float2bfloat16_rn(0.0F));
    } else {
      rope = rope_table[static_cast<std::size_t>(position) * 128 + half_col];
    }
  } else {
    rope = compute_target_rope<kHeadDim, kGlobal>(position, half_col);
  }
  const auto cosine = rope.x;
  const auto sine = rope.y;
  const int paired = col < half ? col + half : col - half;
  const auto left = __float2bfloat16_rn(
      __bfloat162float(key_normed[col]) * __bfloat162float(cosine));
  const float rotated = col < half ? -__bfloat162float(key_normed[paired])
                                   : __bfloat162float(key_normed[paired]);
  const auto right = __float2bfloat16_rn(
      rotated * __bfloat162float(sine));
  source[static_cast<std::size_t>(source_vector) * kHeadDim + col] =
      __float2bfloat16_rn(__bfloat162float(left) + __bfloat162float(right));
}

