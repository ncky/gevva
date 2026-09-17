// Private implementation fragment; included only by src/gpu.cu.
float event_benchmark(cudaStream_t stream, const std::function<void()>& launch,
                      int warmups, int iterations) {
  for (int i = 0; i < warmups; ++i) launch();
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "cudaEventCreate");
  check(cudaEventCreate(&end), "cudaEventCreate");
  check(cudaEventRecord(begin, stream), "cudaEventRecord");
  for (int i = 0; i < iterations; ++i) launch();
  check(cudaEventRecord(end, stream), "cudaEventRecord");
  check(cudaEventSynchronize(end), "cudaEventSynchronize");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds * 1000.0F / iterations;
}

// The target prefill reuses per-layer grouped-GEMM metadata and scratch across
// requests. Measure complete requests independently, matching the runtime
// contract that a request's result is consumed before its scratch is reused.
float isolated_event_benchmark(cudaStream_t stream,
                               const std::function<void()>& launch,
                               int warmups, int iterations) {
  for (int i = 0; i < warmups; ++i) {
    launch();
    check(cudaStreamSynchronize(stream), "synchronize isolated warmup");
  }
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "cudaEventCreate");
  check(cudaEventCreate(&end), "cudaEventCreate");
  float total_milliseconds = 0.0F;
  for (int i = 0; i < iterations; ++i) {
    check(cudaEventRecord(begin, stream), "cudaEventRecord");
    launch();
    check(cudaEventRecord(end, stream), "cudaEventRecord");
    check(cudaEventSynchronize(end), "cudaEventSynchronize");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin, end),
          "cudaEventElapsedTime");
    total_milliseconds += milliseconds;
  }
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return total_milliseconds * 1000.0F / iterations;
}

