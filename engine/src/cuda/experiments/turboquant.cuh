// Private implementation fragment; included only by src/gpu.cu.
// TurboQuant feasibility path for Gemma 4's D=512 global heads.  The cache
// stores a signed randomized Hadamard transform followed by 4-bit Gaussian
// Lloyd-Max scalar quantization.  Scales are kept separately so packed rows
// remain a dense 256 bytes and attention consumes them without materializing
// a BF16 cache.  Constants follow the public TurboQuant reference algorithm;
// the kernels and layout here are purpose-built for this engine's SM120 shape.
__device__ __constant__ unsigned char tq_signs_512[64] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85};

__device__ __constant__ float tq4_centroids_512[16] = {
    -2.7326F,-2.0690F,-1.6180F,-1.2562F,-0.9424F,-0.6568F,-0.3881F,-0.1284F,
     0.1284F, 0.3881F, 0.6568F, 0.9424F, 1.2562F, 1.6180F, 2.0690F, 2.7326F};

__device__ __forceinline__ unsigned tq4_index(float value) {
  constexpr float boundaries[15] = {
      -2.4008F,-1.8435F,-1.4371F,-1.0993F,-0.7996F,-0.5225F,-0.2583F,
       0.0F, 0.2583F, 0.5225F, 0.7996F, 1.0993F, 1.4371F, 1.8435F, 2.4008F};
  unsigned index = 0;
#pragma unroll
  for (int boundary = 0; boundary < 15; ++boundary)
    index += value >= boundaries[boundary];
  return index;
}

__device__ __forceinline__ float tq_sign(int column) {
  return (tq_signs_512[column >> 3] >> (column & 7)) & 1 ? -1.0F : 1.0F;
}

__device__ void tq_fwht_512(float* values) {
  const int lane = threadIdx.x;
#pragma unroll
  for (int span = 1; span < 512; span <<= 1) {
    __syncthreads();
    for (int operation = lane; operation < 256; operation += blockDim.x) {
      const int group = operation / span;
      const int offset = operation - group * span;
      const int left = group * span * 2 + offset;
      const float a = values[left];
      const float b = values[left + span];
      values[left] = a + b;
      values[left + span] = a - b;
    }
  }
  __syncthreads();
}

__global__ void tq4_quantize_512_kernel(
    const __nv_bfloat16* source, unsigned char* packed,
    __nv_bfloat16* scales, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float transformed[512];
  __shared__ float sums[256];
  const int lane = threadIdx.x;
  transformed[lane] = __bfloat162float(source[row * 512 + lane]) * tq_sign(lane);
  transformed[lane + 256] =
      __bfloat162float(source[row * 512 + lane + 256]) * tq_sign(lane + 256);
  tq_fwht_512(transformed);
  for (int half = 0; half < 2; ++half) {
    const float value = transformed[lane + half * 256];
    sums[lane] = value * value;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
      if (lane < stride) sums[lane] += sums[lane + stride];
      __syncthreads();
    }
    const float scale = sqrtf(sums[0] / 256.0F);
    if (lane == 0) scales[row * 2 + half] = __float2bfloat16_rn(scale);
    const unsigned index = tq4_index(scale > 1e-10F ? value / scale : 0.0F);
    const int element = lane + half * 256;
    if ((element & 1) == 0) {
      const unsigned other = tq4_index(
          scale > 1e-10F ? transformed[element + 1] / scale : 0.0F);
      packed[row * 256 + element / 2] =
          static_cast<unsigned char>(index | (other << 4));
    }
    __syncthreads();
  }
}

__global__ void tq_transform_queries_512_kernel(
    const __nv_bfloat16* query, float* transformed) {
  const int head = blockIdx.x;
  __shared__ float values[512];
  const int lane = threadIdx.x;
  values[lane] = __bfloat162float(query[head * 512 + lane]) * tq_sign(lane);
  values[lane + 256] =
      __bfloat162float(query[head * 512 + lane + 256]) * tq_sign(lane + 256);
  tq_fwht_512(values);
  transformed[head * 512 + lane] = values[lane] / 512.0F;
  transformed[head * 512 + lane + 256] = values[lane + 256] / 512.0F;
}

__global__ void tq4_attention_scores_512_kernel(
    const float* query, const unsigned char* keys,
    const __nv_bfloat16* scales, float* scores, int context_tokens) {
  constexpr int kWarps = 8;
  const int local_warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x * kWarps + local_warp;
  const int head = blockIdx.y;
  if (token >= context_tokens) return;
  const int kv_head = head >> 3;
  const int row = token * 2 + kv_head;
  __shared__ float centroids[16];
  if (threadIdx.x < 16) centroids[threadIdx.x] = tq4_centroids_512[threadIdx.x];
  __syncthreads();
  float sum = 0.0F;
#pragma unroll
  for (int pair = lane; pair < 256; pair += 32) {
    const unsigned byte = keys[row * 256 + pair];
    const int column = pair * 2;
    const float scale0 = __bfloat162float(scales[row * 2 + (column >= 256)]);
    const float scale1 = __bfloat162float(scales[row * 2 + (column + 1 >= 256)]);
    sum = fmaf(query[head * 512 + column], centroids[byte & 15] * scale0, sum);
    sum = fmaf(query[head * 512 + column + 1], centroids[byte >> 4] * scale1, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (lane == 0) scores[head * context_tokens + token] = sum;
}

__global__ void tq4_attention_values_512_kernel(
    const float* probabilities, const unsigned char* values,
    const __nv_bfloat16* scales, float* transformed_output,
    int context_tokens, int partitions) {
  const int head = blockIdx.x;
  const int partition = blockIdx.y;
  const int kv_head = head >> 3;
  const int chunk = (context_tokens + partitions - 1) / partitions;
  const int begin = partition * chunk;
  const int end = min(begin + chunk, context_tokens);
  __shared__ float centroids[16];
  if (threadIdx.x < 16) centroids[threadIdx.x] = tq4_centroids_512[threadIdx.x];
  __syncthreads();
  for (int column = threadIdx.x; column < 512; column += blockDim.x) {
    float sum = 0.0F;
    for (int token = begin; token < end; ++token) {
      const int row = token * 2 + kv_head;
      const unsigned byte = values[row * 256 + column / 2];
      const unsigned index = column & 1 ? byte >> 4 : byte & 15;
      const float scale = __bfloat162float(scales[row * 2 + (column >= 256)]);
      sum = fmaf(probabilities[head * context_tokens + token],
                 centroids[index] * scale, sum);
    }
    transformed_output[(partition * 16 + head) * 512 + column] = sum;
  }
}

__global__ void tq_inverse_outputs_512_kernel(
    const float* partials, __nv_bfloat16* output, int partitions) {
  const int head = blockIdx.x;
  __shared__ float values[512];
  const int lane = threadIdx.x;
  float first = 0.0F;
  float second = 0.0F;
  for (int partition = 0; partition < partitions; ++partition) {
    first += partials[(partition * 16 + head) * 512 + lane];
    second += partials[(partition * 16 + head) * 512 + lane + 256];
  }
  values[lane] = first;
  values[lane + 256] = second;
  tq_fwht_512(values);
  output[head * 512 + lane] =
      __float2bfloat16_rn(values[lane] * tq_sign(lane) / 512.0F);
  output[head * 512 + lane + 256] = __float2bfloat16_rn(
      values[lane + 256] * tq_sign(lane + 256) / 512.0F);
}

__global__ void initialize_tq_benchmark_kernel(
    __nv_bfloat16* query, __nv_bfloat16* keys, __nv_bfloat16* values,
    int context_tokens) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t cache_elements =
      static_cast<std::size_t>(context_tokens) * 2 * 512;
  if (index < cache_elements) {
    const float x = sinf(static_cast<float>(index % 8191) * 0.0137F) +
                    cosf(static_cast<float>(index % 4093) * 0.0071F);
    keys[index] = __float2bfloat16_rn(x * 0.55F);
    values[index] = __float2bfloat16_rn(
        sinf(static_cast<float>(index % 6151) * 0.0093F) * 0.8F);
  }
  if (index < 16 * 512)
    query[index] = __float2bfloat16_rn(
        cosf(static_cast<float>(index) * 0.011F) * 0.09F);
}


