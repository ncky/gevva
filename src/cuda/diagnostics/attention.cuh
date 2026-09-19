// Private implementation fragment; included only by src/gpu.cu.
int test_attention_unpack() {
  struct Buffer {
    void* p{};
    explicit Buffer(std::size_t n) { check(cudaMallocManaged(&p, n), "unpack test allocation"); }
    ~Buffer() { if (p) cudaFree(p); }
  };
  constexpr int capacity = 8 * 5 * 8192;
  Buffer src(capacity * 2), old(capacity * 2), fused(capacity * 2),
      qa(capacity), qb(capacity), sa(40 * 4), sb(40 * 4);
  auto* packed = static_cast<__nv_bfloat16*>(src.p);
  for (int i = 0; i < capacity; ++i)
    packed[i] = __float2bfloat16_rn((static_cast<int>((static_cast<unsigned>(i) * 1664525U + 1013904223U) >> 16) - 32768) / 1024.0F);
  int count = 0;
  for (int sessions : {1, 2, 8}) for (int tokens : {1, 4, 5}) {
    auto run = [&]<int D, int Heads>() {
      const int rows = sessions * tokens, elements = rows * 16 * D;
      unpack_ragged_gemm_attention_kernel<D, Heads><<<(elements + 255) / 256, 256>>>(
          packed, static_cast<__nv_bfloat16*>(old.p), sessions, tokens);
      quantize_fp8_rows_kernel<<<rows, 256>>>(static_cast<const __nv_bfloat16*>(old.p),
          static_cast<__nv_fp8_e4m3*>(qa.p), static_cast<float*>(sa.p), rows, 16 * D);
      unpack_quantize_attention_kernel<D, Heads, false><<<rows, 256>>>(packed,
          static_cast<__nv_bfloat16*>(fused.p), static_cast<__nv_fp8_e4m3*>(qb.p),
          static_cast<float*>(sb.p), nullptr, nullptr, tokens, rows);
      check(cudaDeviceSynchronize(), "attention unpack differential");
      if (std::memcmp(old.p, fused.p, elements * 2) || std::memcmp(qa.p, qb.p, elements) ||
          std::memcmp(sa.p, sb.p, rows * sizeof(float)))
        throw std::runtime_error("attention unpack/quantization mismatch");
      ++count;
    };
    run.operator()<256, 8>(); run.operator()<512, 2>();
  }
  // Compare complete mirrored caches as well as activations at wrap boundaries.
  constexpr int plane = 8 * 256, ring_rows = 10;
  constexpr std::size_t cache_elements = 2ULL * 2 * 4096 * plane;
  Buffer ca(cache_elements * 2), cb(cache_elements * 2), view_buffer(4 * sizeof(DeviceKvView)), context_buffer(2 * sizeof(int));
  auto* old_cache = static_cast<__nv_bfloat16*>(ca.p);
  auto* new_cache = static_cast<__nv_bfloat16*>(cb.p);
  for (std::size_t i = 0; i < cache_elements; ++i)
    old_cache[i] = __float2bfloat16_rn(static_cast<int>(i % 257) / 16.0F);
  check(cudaMemcpy(new_cache, old_cache, cache_elements * 2, cudaMemcpyDefault), "clone ring test cache");
  auto* views = static_cast<DeviceKvView*>(view_buffer.p);
  auto* contexts = static_cast<int*>(context_buffer.p);
  contexts[0] = 2046; contexts[1] = 4096;
  for (int state = 0; state < 2; ++state) for (int session = 0; session < 2; ++session) {
    auto* base = state ? new_cache : old_cache;
    views[state * 2 + session] = {
        reinterpret_cast<std::byte*>(base + static_cast<std::size_t>(session * 2) * 4096 * plane),
        reinterpret_cast<std::byte*>(base + static_cast<std::size_t>(session * 2 + 1) * 4096 * plane), 1024, 8, 256};
  }
  unpack_ragged_gemm_attention_kernel<256, 8, true><<<ring_rows * 16, 256>>>(
      packed, static_cast<__nv_bfloat16*>(old.p), 2, 5, views, contexts);
  quantize_fp8_rows_kernel<<<ring_rows, 256>>>(static_cast<const __nv_bfloat16*>(old.p),
      static_cast<__nv_fp8_e4m3*>(qa.p), static_cast<float*>(sa.p), ring_rows, 4096);
  unpack_quantize_attention_kernel<256, 8, true><<<ring_rows * 8, 256>>>(packed,
      static_cast<__nv_bfloat16*>(fused.p), static_cast<__nv_fp8_e4m3*>(qb.p),
      static_cast<float*>(sb.p), views + 2, contexts, 5, ring_rows);
  check(cudaDeviceSynchronize(), "ring unpack differential");
  if (std::memcmp(ca.p, cb.p, cache_elements * 2) || std::memcmp(old.p, fused.p, ring_rows * 4096 * 2) ||
      std::memcmp(qa.p, qb.p, ring_rows * 4096) || std::memcmp(sa.p, sb.p, ring_rows * 4))
    throw std::runtime_error("fused ring restoration mismatch");
  return count + 1;
}

int test_fast_attention_softmax() {
  struct Buffer {
    void* p{};
    explicit Buffer(std::size_t size) { check(cudaMallocManaged(&p, size), "fast softmax test allocation"); }
    ~Buffer() { if (p) cudaFree(p); }
  };
  constexpr int rows = 8 * 5 * 16, stride = 4112;
  Buffer sc(static_cast<std::size_t>(rows) * stride * 4), a(static_cast<std::size_t>(rows) * stride * 2),
      b(static_cast<std::size_t>(rows) * stride * 2), c(8 * sizeof(int));
  auto* scores = static_cast<float*>(sc.p);
  auto* original = static_cast<__nv_bfloat16*>(a.p);
  auto* fast = static_cast<__nv_bfloat16*>(b.p);
  auto* contexts = static_cast<int*>(c.p);
  for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * stride; ++i)
    scores[i] = static_cast<int>((static_cast<unsigned>(i) * 1664525U + 1013904223U) >> 16) / 4096.0F - 8.0F;
  const std::array lengths{1, 17, 63, 129, 255, 257, 505, 4090};
  int count = 0;
  for (int keys : {17, 255, 256, 257, 505, 1028, 1280, 1281, 4096}) {
    for (int i = 0; i < 8; ++i) contexts[i] = std::min(lengths[i], keys - 5);
    auto run = [&]<bool Ring, bool Candidates>() {
      ragged_gemm_softmax_kernel<Ring, Candidates><<<rows, 256>>>(scores, original, contexts, 5, 2, 8, keys, stride);
      if (keys <= 256)
        warp_gemm_softmax_kernel<Ring, Candidates><<<rows, 32>>>(scores, fast, contexts, 5, 2, 8, keys, stride);
      else if (keys <= 512)
        ragged_gemm_softmax_kernel<Ring, Candidates, 2><<<rows, 256>>>(scores, fast, contexts, 5, 2, 8, keys, stride);
      else if (keys <= 1024)
        ragged_gemm_softmax_kernel<Ring, Candidates, 4><<<rows, 256>>>(scores, fast, contexts, 5, 2, 8, keys, stride);
      else if (keys <= 1280)
        ragged_gemm_softmax_kernel<Ring, Candidates, 5><<<rows, 256>>>(scores, fast, contexts, 5, 2, 8, keys, stride);
      else
        ragged_gemm_softmax_kernel<Ring, Candidates, -1><<<rows, 256>>>(scores, fast, contexts, 5, 2, 8, keys, stride);
      check(cudaDeviceSynchronize(), "fast softmax differential");
      for (int row = 0; row < rows; ++row)
        if (std::memcmp(original + row * stride, fast + row * stride, keys * 2) != 0)
          throw std::runtime_error("fast softmax mismatch at keys=" + std::to_string(keys));
      ++count;
    };
    run.operator()<false, false>(); run.operator()<false, true>();
    run.operator()<true, false>(); run.operator()<true, true>();
  }
  return count;
}

int test_attention_softmax() {
  // Zero logits give an exact, simple oracle and expose the maximum/sum
  // shared-memory race under instrumentation, including underfilled warps.
  struct Buffer {
    void* pointer{};
    explicit Buffer(std::size_t bytes) {
      check(cudaMallocManaged(&pointer, bytes), "allocate softmax test");
    }
    ~Buffer() { if (pointer) cudaFree(pointer); }
  };
  constexpr int sessions = 8, tokens = 5, heads = 2, group = 8;
  constexpr int rows = sessions * tokens * 16, stride = 1105;
  Buffer context_buffer(sessions * sizeof(int));
  Buffer scores_buffer(static_cast<std::size_t>(rows) * stride * sizeof(float));
  Buffer probs_buffer(static_cast<std::size_t>(rows) * stride * sizeof(__nv_bfloat16));
  auto* contexts = static_cast<int*>(context_buffer.pointer);
  auto* scores = static_cast<float*>(scores_buffer.pointer);
  auto* probs = static_cast<__nv_bfloat16*>(probs_buffer.pointer);
  const std::array lengths{1, 17, 63, 129, 255, 257, 505, 1100};
  std::copy(lengths.begin(), lengths.end(), contexts);
  int cases = 0;
  auto run = [&]<bool Ring, bool Candidates>() {
    check(cudaMemset(scores, 0, static_cast<std::size_t>(rows) * stride * sizeof(float)), "clear softmax test");
    ragged_gemm_softmax_kernel<Ring, Candidates><<<rows, 256>>>(
        scores, probs, contexts, tokens, heads, group, stride);
    check(cudaDeviceSynchronize(), "test GEMM softmax");
    for (int row = 0; row < rows; ++row) {
      const int cached = contexts[row / (heads * tokens * group)];
      const int step = row % (tokens * group) / group;
      const int end = cached + step + (Candidates ? 1 : 0);
      const int first = Ring ? std::max(0, end - 1024) : 0;
      const int pool_first = Ring ? std::max(0, cached - (Candidates ? 1023 : 1024)) : 0;
      for (int k = 0; k < stride; ++k) {
        const bool allowed = pool_first + k >= first && pool_first + k < end;
        const float expected = __bfloat162float(__float2bfloat16_rn(allowed ? 1.0F / (end - first) : 0.0F));
        if (__bfloat162float(probs[static_cast<std::size_t>(row) * stride + k]) != expected)
          throw std::runtime_error("GEMM softmax uniform oracle mismatch");
      }
    }
    ++cases;
    ragged_attention_softmax_kernel<Ring, Candidates><<<dim3(16, sessions * tokens), 256>>>(
        scores, contexts, sessions * tokens, tokens, stride);
    check(cudaDeviceSynchronize(), "test ragged softmax");
    for (int row = 0; row < rows; ++row) {
      const int end = contexts[row / (tokens * 16)] + (Candidates ? (row / 16) % tokens + 1 : 0);
      const int attended = Ring ? std::min(end, 1024) : end;
      for (int k = 0; k < attended; ++k) {
        const float value = scores[static_cast<std::size_t>(row) * stride + k];
        if (!std::isfinite(value) || std::abs(value - 1.0F / attended) > 1e-7F)
          throw std::runtime_error("ragged softmax uniform oracle mismatch");
      }
    }
    ++cases;
  };
  run.operator()<false, false>(); run.operator()<false, true>();
  run.operator()<true, false>(); run.operator()<true, true>();
  return cases;
}

