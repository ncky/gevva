// Private implementation fragment; included only by src/gpu.cu.
__global__ void quantize_fp8_rows_kernel(const __nv_bfloat16* input,
                                         __nv_fp8_e4m3* output,
                                         float* scales, int rows, int width) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  float maximum = 0.0F;
  for (int column = threadIdx.x; column < width; column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(__bfloat162float(input[row * width + column])));
  __shared__ float values[256];
  values[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      values[threadIdx.x] = fmaxf(values[threadIdx.x],
                                  values[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scales[row] = fmaxf(values[0] / 448.0F, 1.17549435e-38F);
  __syncthreads();
  const float inverse = 1.0F / scales[row];
  for (int column = threadIdx.x; column < width; column += blockDim.x)
    output[row * width + column] =
        __nv_fp8_e4m3(__bfloat162float(input[row * width + column]) * inverse);
}

template<int D, int Heads, bool RestoreRing>
__global__ void unpack_quantize_attention_kernel(const __nv_bfloat16* packed,
    __nv_bfloat16* unpacked, __nv_fp8_e4m3* quantized, float* scales,
    const g4::DeviceKvView* views, const int* contexts, int tokens, int rows) {
  constexpr int width = 16 * D, group = 16 / Heads;
  if (blockIdx.x < rows) {
  const int row = blockIdx.x, session = row / tokens, step = row % tokens;
  auto load = [&](int column) {
    const int head = column / D, within = head % group;
    return packed[((static_cast<std::size_t>(session) * Heads + head / group) * tokens * group +
                   step * group + within) * D + column % D];
  };
  float maximum = 0.0F;
  for (int col = threadIdx.x; col < width; col += blockDim.x)
    maximum = fmaxf(maximum, fabsf(__bfloat162float(load(col))));
  __shared__ float reduction[256];
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int s = 128; s >= 32; s >>= 1) {
    if (threadIdx.x < s) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float value = reduction[threadIdx.x];
    for (int offset = 16; offset; offset >>= 1)
      value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    if (threadIdx.x == 0) reduction[0] = value;
  }
  __syncthreads();
  const float scale = fmaxf(reduction[0] / 448.0F, 1.17549435e-38F);
  if (threadIdx.x == 0) scales[row] = scale;
  const float inverse = 1.0F / scale;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const auto value = load(col);
    unpacked[row * width + col] = value;
    quantized[row * width + col] = __nv_fp8_e4m3(__bfloat162float(value) * inverse);
  }
  }
  if constexpr (RestoreRing) {
    constexpr int plane = Heads * D;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < rows * plane;
         index += gridDim.x * blockDim.x) {
    const int col = index % plane, row = index / plane;
    const int session = row / tokens, step = row % tokens;
    const int cached = contexts[session];
    const int first = cached > 1023 ? (cached - 1023) & 1023 : 0;
    const int tail = 1024 + first + min(cached, 1023) + step;
    auto* k = reinterpret_cast<__nv_bfloat16*>(views[session].keys);
    auto* v = reinterpret_cast<__nv_bfloat16*>(views[session].values);
      k[tail * plane + col] = k[(tail & 1023) * plane + col];
      v[tail * plane + col] = v[(tail & 1023) * plane + col];
    }
  }
}

__global__ void rmsnorm_quantize_fp8_2816_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_fp8_e4m3* output, float* scales, int rows) {
  constexpr int kWidth = 2816;
  __shared__ float reductions[256];
  __shared__ float inverse_rms;
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0F;
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const float value = __bfloat162float(input[row * kWidth + column]);
    sum = fmaf(value, value, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) reductions[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? reductions[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kWidth + 1e-6F);
  }
  __syncthreads();
  float maximum = 0.0F;
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const auto normalized = __float2bfloat16_rn(
        __bfloat162float(input[row * kWidth + column]) * inverse_rms *
        __bfloat162float(weight[column]));
    maximum = fmaxf(maximum, fabsf(__bfloat162float(normalized)));
  }
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x],
                                      reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scales[row] = fmaxf(reductions[0] / 448.0F, 1.17549435e-38F);
  __syncthreads();
  const float inverse_scale = 1.0F / scales[row];
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const auto normalized = __float2bfloat16_rn(
        __bfloat162float(input[row * kWidth + column]) * inverse_rms *
        __bfloat162float(weight[column]));
    output[row * kWidth + column] = __nv_fp8_e4m3(
        __bfloat162float(normalized) * inverse_scale);
  }
}

template <bool FuseAttention = false>
__global__ void dual_rmsnorm_router_quantize_fp8_2816_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* dense_weight,
    const __nv_bfloat16* expert_weight, const __nv_bfloat16* router_weight,
    __nv_bfloat16* dense_output, __nv_bfloat16* expert_output,
    __nv_fp8_e4m3* router_output, float* scales,
    __nv_fp8_e4m3* dense_fp8_output, float* dense_scales, int rows,
    const __nv_bfloat16* attention = nullptr,
    const __nv_bfloat16* attention_weight = nullptr,
    __nv_bfloat16* residual_output = nullptr,
    const float* attention_scales = nullptr,
    const float* attention_projection_scale = nullptr) {
  constexpr int kWidth = 2816;
  __shared__ float reductions[512];
  __shared__ float dense_reductions[512];
  __shared__ float inverse_rms;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int reduction_threads = min(static_cast<int>(blockDim.x), 256);
  __shared__ __nv_bfloat16 residual_row[FuseAttention ? kWidth : 1];
  if constexpr (FuseAttention) {
    // Preserve the two original BF16 rounding boundaries and reduction order.
    float attention_sum = 0.0F;
    for (int column = threadIdx.x;
         threadIdx.x < reduction_threads && column < kWidth;
         column += reduction_threads) {
      float x = __bfloat162float(attention[row * kWidth + column]);
      if (attention_scales)
        x = __bfloat162float(__float2bfloat16_rn(
            x * (attention_scales[row] * attention_projection_scale[0])));
      attention_sum = fmaf(x, x, attention_sum);
    }
    for (int offset = 16; offset; offset >>= 1)
      attention_sum += __shfl_down_sync(0xffffffff, attention_sum, offset);
    if ((threadIdx.x & 31) == 0)
      reductions[threadIdx.x >> 5] = attention_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
      attention_sum = threadIdx.x < reduction_threads / 32
          ? reductions[threadIdx.x] : 0.0F;
      for (int offset = 16; offset; offset >>= 1)
        attention_sum += __shfl_down_sync(0xffffffff, attention_sum, offset);
      if (threadIdx.x == 0)
        inverse_rms = rsqrtf(attention_sum / kWidth + 1e-6F);
    }
    __syncthreads();
    for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
      const int index = row * kWidth + column;
      float x = __bfloat162float(attention[index]);
      if (attention_scales)
        x = __bfloat162float(__float2bfloat16_rn(
            x * (attention_scales[row] * attention_projection_scale[0])));
      const auto normalized = __float2bfloat16_rn(
          x * inverse_rms * __bfloat162float(attention_weight[column]));
      const auto value = __float2bfloat16_rn(
          __bfloat162float(input[index]) + __bfloat162float(normalized));
      residual_row[column] = value;
      residual_output[index] = value;
    }
    __syncthreads();
  }
  float sum = 0.0F;
  for (int column = threadIdx.x;
       threadIdx.x < reduction_threads && column < kWidth;
       column += reduction_threads) {
    const float value = __bfloat162float(FuseAttention
        ? residual_row[column] : input[row * kWidth + column]);
    sum = fmaf(value, value, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) reductions[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < reduction_threads / 32
        ? reductions[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kWidth + 1e-6F);
  }
  __syncthreads();
  const float root = rsqrtf(static_cast<float>(kWidth));
  float maximum = 0.0F;
  float dense_maximum = 0.0F;
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const int index = row * kWidth + column;
    const float normalized = __bfloat162float(FuseAttention
        ? residual_row[column] : input[index]) * inverse_rms;
    const auto dense_value = __float2bfloat16_rn(
        normalized * __bfloat162float(dense_weight[column]));
    dense_output[index] = dense_value;
    dense_maximum = fmaxf(
        dense_maximum, fabsf(__bfloat162float(dense_value)));
    expert_output[index] = __float2bfloat16_rn(
        normalized * __bfloat162float(expert_weight[column]));
    auto routed = __float2bfloat16_rn(normalized);
    routed = __float2bfloat16_rn(
        __bfloat162float(routed) * __bfloat162float(router_weight[column]));
    const auto value = __float2bfloat16_rn(__bfloat162float(routed) * root);
    maximum = fmaxf(maximum, fabsf(__bfloat162float(value)));
  }
  if constexpr (FuseAttention) {
    for (int offset = 16; offset; offset >>= 1) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
      dense_maximum = fmaxf(dense_maximum,
          __shfl_down_sync(0xffffffff, dense_maximum, offset));
    }
    if ((threadIdx.x & 31) == 0) {
      reductions[threadIdx.x >> 5] = maximum;
      dense_reductions[threadIdx.x >> 5] = dense_maximum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      maximum = threadIdx.x < blockDim.x / 32 ? reductions[threadIdx.x] : 0.0F;
      dense_maximum = threadIdx.x < blockDim.x / 32
          ? dense_reductions[threadIdx.x] : 0.0F;
      for (int offset = 16; offset; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        dense_maximum = fmaxf(dense_maximum,
            __shfl_down_sync(0xffffffff, dense_maximum, offset));
      }
      if (threadIdx.x == 0) {
        scales[row] = fmaxf(maximum / 448.0F, 1.17549435e-38F);
        dense_scales[row] = fmaxf(dense_maximum / 448.0F, 1.17549435e-38F);
      }
    }
  } else {
    reductions[threadIdx.x] = maximum;
    dense_reductions[threadIdx.x] = dense_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x],
                                        reductions[threadIdx.x + stride]);
        dense_reductions[threadIdx.x] = fmaxf(dense_reductions[threadIdx.x],
                                            dense_reductions[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scales[row] = fmaxf(reductions[0] / 448.0F, 1.17549435e-38F);
      dense_scales[row] = fmaxf(dense_reductions[0] / 448.0F, 1.17549435e-38F);
    }
  }
  __syncthreads();
  const float inverse_scale = 1.0F / scales[row];
  const float dense_inverse_scale = 1.0F / dense_scales[row];
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const int index = row * kWidth + column;
    const float normalized = __bfloat162float(FuseAttention
        ? residual_row[column] : input[index]) * inverse_rms;
    auto routed = __float2bfloat16_rn(normalized);
    routed = __float2bfloat16_rn(
        __bfloat162float(routed) * __bfloat162float(router_weight[column]));
    const auto value = __float2bfloat16_rn(__bfloat162float(routed) * root);
    router_output[index] =
        __nv_fp8_e4m3(__bfloat162float(value) * inverse_scale);
    const auto dense_value = __float2bfloat16_rn(
        normalized * __bfloat162float(dense_weight[column]));
    dense_fp8_output[index] = __nv_fp8_e4m3(
        __bfloat162float(dense_value) * dense_inverse_scale);
  }
}

__device__ __forceinline__ __nv_bfloat16 gelu_product_bf16(
    __nv_bfloat16 gate, __nv_bfloat16 up) {
  const float x = __bfloat162float(gate);
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  const float gelu = 0.5F * x *
      (1.0F + tanhf(kSqrtTwoOverPi * (x + 0.044715F * x * x * x)));
  return __float2bfloat16_rn(gelu * __bfloat162float(up));
}

__global__ void gelu_quantize_fp8_2112_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_fp8_e4m3* output, float* scales, int rows,
    const float* gate_activation_scales,
    const float* gate_projection_weight_scale,
    const float* up_activation_scales,
    const float* up_projection_weight_scale, int input_stride = 2112) {
  constexpr int kWidth = 2112;
  __shared__ float reductions[256];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float maximum = 0.0F;
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    auto gate_value = gate[row * input_stride + column];
    auto up_value = up[row * input_stride + column];
    if (gate_activation_scales) {
      const float gate_scale = gate_activation_scales[row] *
                               gate_projection_weight_scale[0];
      const float up_scale = up_activation_scales[row] *
                             up_projection_weight_scale[0];
      gate_value = __float2bfloat16_rn(
          __bfloat162float(gate_value) * gate_scale);
      up_value = __float2bfloat16_rn(__bfloat162float(up_value) * up_scale);
    }
    const auto value = gelu_product_bf16(gate_value, up_value);
    maximum = fmaxf(maximum, fabsf(__bfloat162float(value)));
  }
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x],
                                      reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scales[row] = fmaxf(reductions[0] / 448.0F, 1.17549435e-38F);
  __syncthreads();
  const float inverse_scale = 1.0F / scales[row];
  for (int column = threadIdx.x; column < kWidth; column += blockDim.x) {
    const int index = row * kWidth + column;
    auto gate_value = gate[row * input_stride + column];
    auto up_value = up[row * input_stride + column];
    if (gate_activation_scales) {
      const float gate_scale = gate_activation_scales[row] *
                               gate_projection_weight_scale[0];
      const float up_scale = up_activation_scales[row] *
                             up_projection_weight_scale[0];
      gate_value = __float2bfloat16_rn(
          __bfloat162float(gate_value) * gate_scale);
      up_value = __float2bfloat16_rn(__bfloat162float(up_value) * up_scale);
    }
    const auto value = gelu_product_bf16(gate_value, up_value);
    output[index] =
        __nv_fp8_e4m3(__bfloat162float(value) * inverse_scale);
  }
}

__global__ void scale_fp8_gemm_output_kernel(__nv_bfloat16* output,
                                              const float* activation_scales,
                                              const float* weight_scale,
                                              int rows, int width,
                                              bool per_output_scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * width) return;
  const float scale = activation_scales[index / width] *
                      weight_scale[per_output_scale ? index % width : 0];
  output[index] = __float2bfloat16_rn(__bfloat162float(output[index]) * scale);
}

__global__ void scale_fp8_qkv_output_kernel(
    __nv_bfloat16* query, int query_width, const float* query_weight_scale,
    __nv_bfloat16* key, int key_width, const float* key_weight_scale,
    __nv_bfloat16* value, const float* value_weight_scale,
    const float* activation_scales, int rows) {
  const int value_width = value ? key_width : 0;
  const int row_width = query_width + key_width + value_width;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * row_width) return;
  const int row = index / row_width;
  const int column = index % row_width;
  __nv_bfloat16* destination;
  const float* weight_scale;
  int destination_index;
  if (column < query_width) {
    destination = query;
    weight_scale = query_weight_scale;
    destination_index = row * query_width + column;
  } else if (column < query_width + key_width) {
    destination = key;
    weight_scale = key_weight_scale;
    destination_index = row * key_width + column - query_width;
  } else {
    destination = value;
    weight_scale = value_weight_scale;
    destination_index = row * value_width + column - query_width - key_width;
  }
  const float scale = activation_scales[row] * weight_scale[0];
  destination[destination_index] = __float2bfloat16_rn(
      __bfloat162float(destination[destination_index]) * scale);
}

