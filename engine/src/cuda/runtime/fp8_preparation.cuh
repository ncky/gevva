// Private implementation fragment; included only by src/gpu.cu.
void Fp8LinearRunner::prepare_dual_rmsnorm_router_2816(
    const void* input, const void* dense_weight, const void* expert_weight,
    const void* router_weight, void* dense_output, void* expert_output,
    const void* activation_cache_key, int tokens, cudaStream_t stream,
    int threads, const void* attention, const void* attention_weight,
    void* residual_output, const float* attention_scales,
    const float* attention_projection_scale) const {
  impl_->prepare_dual_rmsnorm_router_2816(
      input, dense_weight, expert_weight, router_weight, dense_output,
      expert_output, activation_cache_key, tokens, stream, threads, attention,
      attention_weight, residual_output, attention_scales,
      attention_projection_scale);
}

void Fp8LinearRunner::use_secondary_activation_2816(
    const void* activation_cache_key, int tokens,
    cudaStream_t dependency_ordered_stream) const {
  impl_->use_secondary_activation_2816(
      activation_cache_key, tokens, dependency_ordered_stream);
}

void Fp8LinearRunner::assume_activation_ready_on(
    cudaStream_t dependency_ordered_stream) const {
  impl_->activation_stream = dependency_ordered_stream;
}

void Fp8LinearRunner::prepare_gelu_2112(
    const void* gate, const void* up, Fp8OutputScale gate_scale,
    Fp8OutputScale up_scale, const void* activation_cache_key, int tokens,
    cudaStream_t stream, bool packed) const {
  impl_->prepare_gelu_2112(
      gate, up, gate_scale, up_scale, activation_cache_key, tokens, stream, packed);
}

bool Fp8LinearRunner::launch_fused_gate_up(const void* gate_weight, const void* up_weight,
    const void* input, void* output, int tokens, cudaStream_t stream,
    void* blas_handle, Fp8OutputScale& gate_scale, Fp8OutputScale& up_scale) const {
  const auto found = impl_->paired_keys.find(gate_weight);
  if (found == impl_->paired_keys.end()) return false;
  if (!impl_->launch(found->second, 4224, 2816, input, output, tokens, stream,
                    static_cast<cublasHandle_t>(blas_handle), false, 1, true)) return false;
  gate_scale = {impl_->activation_scales, impl_->weights.at(gate_weight).scale};
  up_scale = {impl_->activation_scales, impl_->weights.at(up_weight).scale};
  return true;
}

void Fp8LinearRunner::prepare_packed_attention(const void* packed, void* output,
    const void* views, const int* contexts, int sessions, int tokens,
    int dimension, bool restore_ring, cudaStream_t stream) const {
  const int rows = sessions * tokens;
  if (rows < 1 || rows > impl_->maximum_tokens || 16 * dimension > impl_->maximum_width)
    throw std::runtime_error("packed attention FP8 geometry mismatch");
  auto launch = [&]<int D, int Heads, bool Ring>() {
    const int blocks = Ring ? std::max(rows, (rows * Heads * D + 255) / 256) : rows;
    unpack_quantize_attention_kernel<D, Heads, Ring><<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed), static_cast<__nv_bfloat16*>(output),
        impl_->activation, impl_->activation_scales, static_cast<const DeviceKvView*>(views), contexts, tokens, rows);
  };
  if (dimension == 512) launch.operator()<512, 2, false>();
  else if (restore_ring) launch.operator()<256, 8, true>();
  else launch.operator()<256, 8, false>();
  check(cudaEventRecord(impl_->activation_ready, stream), "record packed attention activation readiness");
  impl_->activation_stream = stream;
  impl_->last_source = output;
  impl_->last_tokens = rows;
  impl_->last_inputs = 16 * dimension;
}

