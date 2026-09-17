#include "g4/gpu.hpp"
#include "cuda_errors.hpp"

#include <functional>
#include <unordered_map>
#include <utility>

namespace g4 {
using detail::check;

struct GpuExecutionContext::Impl {
  cudaStream_t stream{};
  cudaStream_t auxiliary_stream{};
  cudaStream_t tertiary_stream{};
  cudaEvent_t auxiliary_begin{};
  cudaEvent_t auxiliary_end{};
  cudaEvent_t tertiary_begin{};
  cudaEvent_t tertiary_end{};
  cublasHandle_t handle{};
  std::unordered_map<std::uint64_t,
                     std::pair<cudaGraph_t, cudaGraphExec_t>> graphs;
  explicit Impl(bool high_priority) {
    int least_priority = 0, greatest_priority = 0;
    check(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
          "query CUDA stream priority range");
    const int priority = high_priority ? greatest_priority : least_priority;
    check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority),
          "cudaStreamCreate(execution context)");
    check(cudaStreamCreateWithPriority(&auxiliary_stream, cudaStreamNonBlocking,
                                       priority),
          "cudaStreamCreate(auxiliary execution context)");
    check(cudaStreamCreateWithPriority(&tertiary_stream, cudaStreamNonBlocking,
                                       priority),
          "cudaStreamCreate(tertiary execution context)");
    check(cudaEventCreateWithFlags(&auxiliary_begin, cudaEventDisableTiming),
          "cudaEventCreate(auxiliary begin)");
    check(cudaEventCreateWithFlags(&auxiliary_end, cudaEventDisableTiming),
          "cudaEventCreate(auxiliary end)");
    check(cudaEventCreateWithFlags(&tertiary_begin, cudaEventDisableTiming),
          "cudaEventCreate(tertiary begin)");
    check(cudaEventCreateWithFlags(&tertiary_end, cudaEventDisableTiming),
          "cudaEventCreate(tertiary end)");
    check(cublasCreate(&handle), "cublasCreate(execution context)");
    check(cublasSetStream(handle, stream),
          "cublasSetStream(execution context)");
    check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
          "cublasSetAtomicsMode(execution context)");
  }
  ~Impl() {
    for (auto& [_, graph] : graphs) {
      if (graph.second) cudaGraphExecDestroy(graph.second);
      if (graph.first) cudaGraphDestroy(graph.first);
    }
    if (handle) cublasDestroy(handle);
    if (tertiary_end) cudaEventDestroy(tertiary_end);
    if (tertiary_begin) cudaEventDestroy(tertiary_begin);
    if (auxiliary_end) cudaEventDestroy(auxiliary_end);
    if (auxiliary_begin) cudaEventDestroy(auxiliary_begin);
    if (auxiliary_stream) cudaStreamDestroy(auxiliary_stream);
    if (tertiary_stream) cudaStreamDestroy(tertiary_stream);
    if (stream) cudaStreamDestroy(stream);
  }
};

GpuExecutionContext::GpuExecutionContext(bool high_priority)
    : impl_(std::make_unique<Impl>(high_priority)) {}
GpuExecutionContext::~GpuExecutionContext() = default;
cudaStream_t GpuExecutionContext::stream() const { return impl_->stream; }
cudaStream_t GpuExecutionContext::auxiliary_stream() const {
  return impl_->auxiliary_stream;
}
cudaStream_t GpuExecutionContext::tertiary_stream() const {
  return impl_->tertiary_stream;
}
void* GpuExecutionContext::blas_handle() const { return impl_->handle; }
void GpuExecutionContext::begin_auxiliary() const {
  check(cudaEventRecord(impl_->auxiliary_begin, impl_->stream),
        "record auxiliary fork");
  check(cudaStreamWaitEvent(impl_->auxiliary_stream, impl_->auxiliary_begin),
        "wait auxiliary fork");
}
void GpuExecutionContext::end_auxiliary() const {
  check(cudaEventRecord(impl_->auxiliary_end, impl_->auxiliary_stream),
        "record auxiliary join");
  check(cudaStreamWaitEvent(impl_->stream, impl_->auxiliary_end),
        "wait auxiliary join");
}
void GpuExecutionContext::begin_tertiary() const {
  check(cudaEventRecord(impl_->tertiary_begin, impl_->stream),
        "record tertiary fork");
  check(cudaStreamWaitEvent(impl_->tertiary_stream, impl_->tertiary_begin),
        "wait tertiary fork");
}
void GpuExecutionContext::end_tertiary() const {
  check(cudaEventRecord(impl_->tertiary_end, impl_->tertiary_stream),
        "record tertiary join");
  check(cudaStreamWaitEvent(impl_->stream, impl_->tertiary_end),
        "wait tertiary join");
}
void GpuExecutionContext::join_tertiary_to_auxiliary() const {
  check(cudaEventRecord(impl_->tertiary_end, impl_->tertiary_stream),
        "record tertiary-to-auxiliary join");
  check(cudaStreamWaitEvent(impl_->auxiliary_stream, impl_->tertiary_end),
        "wait tertiary work on auxiliary stream");
}
bool GpuExecutionContext::has_graph(std::uint64_t key) const {
  return impl_->graphs.contains(key);
}
void GpuExecutionContext::capture_graph(
    std::uint64_t key, const std::function<void()>& enqueue) const {
  if (impl_->graphs.contains(key))
    throw std::runtime_error("duplicate execution graph key");
  // Decode shapes change as context grows.  Keeping every historical graph
  // would otherwise retain tens of thousands of kernel nodes during a
  // native-length generation.  Capture sites synchronize before reaching
  // this function, so an arbitrary cold entry can be reclaimed safely.
  constexpr std::size_t kMaximumResidentGraphs = 64;
  if (impl_->graphs.size() >= kMaximumResidentGraphs) {
    auto expired = impl_->graphs.begin();
    if (expired->second.second)
      check(cudaGraphExecDestroy(expired->second.second),
            "destroy expired execution graph");
    if (expired->second.first)
      check(cudaGraphDestroy(expired->second.first),
            "destroy expired captured graph");
    impl_->graphs.erase(expired);
  }
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal),
        "begin persistent execution graph");
  try {
    enqueue();
    check(cudaStreamEndCapture(impl_->stream, &graph),
          "end persistent execution graph");
    check(cudaGraphInstantiate(&executable, graph),
          "instantiate persistent execution graph");
    check(cudaGraphUpload(executable, impl_->stream),
          "upload persistent execution graph");
    impl_->graphs.emplace(key, std::pair{graph, executable});
  } catch (...) {
    cudaStreamEndCapture(impl_->stream, &graph);
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
}
void GpuExecutionContext::launch_graph(std::uint64_t key) const {
  const auto found = impl_->graphs.find(key);
  if (found == impl_->graphs.end())
    throw std::runtime_error("execution graph key is not cached");
  check(cudaGraphLaunch(found->second.second, impl_->stream),
        "launch persistent execution graph");
}

}  // namespace g4
