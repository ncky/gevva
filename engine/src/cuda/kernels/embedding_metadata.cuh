// Private implementation fragment; included only by src/gpu.cu.
__global__ void prepare_vision_patches_kernel(const float* pixels,
                                               __nv_bfloat16* patches,
                                               int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements)
    patches[index] = __float2bfloat16_rn(2.0F * (pixels[index] - 0.5F));
}

__global__ void compose_multimodal_embeddings_kernel(
    const __nv_bfloat16* vocabulary, const std::uint32_t* input_ids,
    const int* image_rows, const __nv_bfloat16* image_features,
    __nv_bfloat16* output, int tokens) {
  constexpr int kHidden = 2816;
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= tokens * kHidden) return;
  const int token = element / kHidden;
  const int column = element % kHidden;
  if (image_rows[token] >= 0) {
    output[element] = image_features[image_rows[token] * kHidden + column];
  } else {
    const auto scale = __float2bfloat16_rn(sqrtf(static_cast<float>(kHidden)));
    output[element] = __float2bfloat16_rn(
        __bfloat162float(vocabulary[
            static_cast<std::size_t>(input_ids[token]) * kHidden + column]) *
        __bfloat162float(scale));
  }
}

__global__ void gather_scaled_embeddings_kernel(
    const __nv_bfloat16* vocabulary, const std::uint32_t* input_ids,
    __nv_bfloat16* output, int tokens, int hidden) {
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= tokens * hidden) return;
  const int token = element / hidden;
  const int column = element % hidden;
  const auto scale = __float2bfloat16_rn(sqrtf(static_cast<float>(hidden)));
  output[element] = __float2bfloat16_rn(
      __bfloat162float(vocabulary[
          static_cast<std::size_t>(input_ids[token]) * hidden + column]) *
      __bfloat162float(scale));
}

__global__ void copy_sliding_prefix_arena_kernel(
    const uint4* source, uint4* destination, int tokens) {
  constexpr int kSlidingLayers = 25;
  constexpr int kPlanesPerLayer = 2;
  constexpr int kStorageTokens = 3 * 1024 + 4;
  constexpr int kPlaneElements = 8 * 256;
  constexpr int kElementsPerVector = sizeof(uint4) / sizeof(std::uint16_t);
  constexpr int kVectorsPerToken = kPlaneElements / kElementsPerVector;
  const std::size_t vectors = static_cast<std::size_t>(kSlidingLayers) *
      kPlanesPerLayer * tokens * kVectorsPerToken;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vectors;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t value = index;
    const int vector = value % kVectorsPerToken;
    value /= kVectorsPerToken;
    const int token = value % tokens;
    const int plane = value / tokens;
    const std::size_t plane_first =
        static_cast<std::size_t>(plane) * kStorageTokens * kVectorsPerToken;
    const std::size_t canonical =
        plane_first + static_cast<std::size_t>(token) * kVectorsPerToken +
        vector;
    const uint4 bits = source[canonical];
    destination[canonical] = bits;
    destination[canonical + 1024ULL * kVectorsPerToken] = bits;
    destination[canonical + 2048ULL * kVectorsPerToken] = bits;
    if (token < 4)
      destination[canonical + 3072ULL * kVectorsPerToken] = bits;
  }
}

__global__ void prepare_verifier_candidate_ids_kernel(
    int previous_token, const int* drafts, std::uint32_t* candidates,
    int tokens) {
  const int index = threadIdx.x;
  if (index == 0)
    candidates[0] = static_cast<std::uint32_t>(previous_token);
  else if (index < tokens)
    candidates[index] = static_cast<std::uint32_t>(drafts[index - 1]);
}

struct DeviceMtpResult {
  int target_tokens[5];
  int draft_tokens[4];
  int output_tokens[5];
  int output_count;
  int matched_drafts;
};

