#include "gevva/tiled_attention.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mma.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace gevva {
namespace {
using BF = __nv_bfloat16;
namespace wm = nvcuda::wmma;
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("attention benchmark cuBLAS error " +
                             std::to_string(status));
}
struct Arena {
  std::vector<void *> buffers;
  ~Arena() {
    for (void *ptr : buffers)
      cudaFree(ptr);
  }
  template <class T> T *allocate(std::size_t count) {
    T *ptr{};
    check(cudaMalloc(&ptr, count * sizeof(T)));
    buffers.push_back(ptr);
    return ptr;
  }
  template <class T> T *upload(const std::vector<T> &host) {
    T *ptr = allocate<T>(host.size());
    check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice));
    return ptr;
  }
};
struct Header {
  std::uint32_t magic = 0x31415447;
  int batch, tokens, dim, heads, keys, ring;
};

template <int D, int S>
__device__ void stage_kv(BF *destination, const BF *source, int stride,
                         int valid) {
  // Eight BF16 values per asynchronous transaction; padded shared rows avoid
  // the power-of-two stride of the token-interleaved global cache.
  for (int i = threadIdx.x; i < S * (D / 8); i += blockDim.x) {
    const int row = i / (D / 8), col = i % (D / 8) * 8;
    const auto address = static_cast<unsigned>(
        __cvta_generic_to_shared(destination + row * (D + 16) + col));
    const BF *input = source + (row < valid ? row : 0) * stride + col;
    const int bytes = row < valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address),
                 "l"(input), "r"(bytes)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ void wait_kv() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
}

template <int D, int S, int Chunks = 1, bool Staged = false,
          bool Compact = false, int QTiles = 1>
__global__ void
partition_attention(const BF *const *queries, const BF *const *keys,
                    const BF *const *values, const int *contexts,
                    float *partial, int sessions, int tokens, int heads,
                    bool ring) {
  const int group = 16 / heads, rows = tokens * group;
  constexpr int QRows = 16 * QTiles;
  static_assert(QTiles == 1 || Compact);
  const int batch = blockIdx.y, qbase = blockIdx.x * QRows;
  const int row_stride = heads * D;
  const int cached = contexts[batch / heads];
  const int past = ring ? min(cached, 1023) : cached;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const std::size_t base =
      (static_cast<std::size_t>(blockIdx.z) * sessions * heads + batch) * rows;
  extern __shared__ __align__(32) unsigned char smem[];
  auto *tail = reinterpret_cast<BF *>(smem);
  constexpr int StageWidth = Compact ? 64 : D;
  constexpr int ScoreStride = S + (Staged ? 8 : 0),
                ProbabilityStride = S + (Staged ? 16 : 0);
  auto *scores = reinterpret_cast<float *>(
      tail + (Staged ? S * (StageWidth + 16) : 16 * D));
  auto *probability = reinterpret_cast<BF *>(scores + QRows * ScoreStride);
  auto *output_tiles =
      reinterpret_cast<float *>(probability + QRows * ProbabilityStride);
  auto *old_factor = output_tiles + 1024 * QTiles;
  auto *new_factor = old_factor + QRows;
  auto *staged_query = reinterpret_cast<BF *>(new_factor + QRows);
  if constexpr (Staged && !Compact) {
    stage_kv<D, 16>(staged_query, queries[batch] + qbase * D, D,
                    min(16, rows - qbase));
    wait_kv();
  }
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    const int start = (blockIdx.z * Chunks + chunk) * S;
    const int valid = min(S, max(0, past + tokens - start));
    if (!valid) {
      if (chunk)
        break;
      for (int i = threadIdx.x; i < QRows * (D + 2); i += blockDim.x) {
        const int row = i / (D + 2), col = i % (D + 2);
        if (qbase + row < rows)
          partial[(base + qbase + row) * (D + 2) + col] =
              col == D ? -INFINITY : 0.0F;
      }
      return;
    }
    const int tail_start = valid / 16 * 16;
    if constexpr (Compact) {
      static_assert(S == 64 && Staged);
      wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
      wm::fill_fragment(acc, 0.0F);
      for (int feature = 0; feature < D; feature += 64) {
        stage_kv<64, QRows>(staged_query, queries[batch] + qbase * D + feature,
                            D, min(QRows, rows - qbase));
        stage_kv<64, S>(tail, keys[batch] + start * row_stride + feature,
                        row_stride, valid);
        wait_kv();
        for (int k = 0; k < 64; k += 16) {
          wm::fragment<wm::matrix_a, 16, 16, 16, BF, wm::row_major> a;
          wm::fragment<wm::matrix_b, 16, 16, 16, BF, wm::col_major> b;
          wm::load_matrix_sync(a, staged_query + (warp / 4) * 16 * 80 + k, 80);
          wm::load_matrix_sync(b, tail + (warp % 4) * 16 * 80 + k, 80);
          wm::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
      }
      wm::store_matrix_sync(scores + (warp / 4) * 16 * ScoreStride +
                                (warp % 4) * 16,
                            acc, ScoreStride, wm::mem_row_major);
    } else {
      if constexpr (Staged) {
        stage_kv<D, S>(tail, keys[batch] + start * row_stride, row_stride,
                       valid);
        wait_kv();
      } else {
        for (int i = threadIdx.x; i < 16 * D; i += blockDim.x) {
          const int row = i / D, col = i % D;
          if (valid % 16)
            tail[i] =
                tail_start + row < valid
                    ? keys[batch][(start + tail_start + row) * row_stride + col]
                    : __float2bfloat16(0.0F);
        }
        __syncthreads();
      }
      for (int tile = warp * 16; tile < S; tile += 64) {
        wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
        wm::fill_fragment(acc, 0.0F);
        if (tile < valid) {
          for (int k = 0; k < D; k += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, BF, wm::row_major> a;
            wm::fragment<wm::matrix_b, 16, 16, 16, BF, wm::col_major> b;
            // Query storage includes a 16-row guard at the allocation's end.
            // Padded query rows are never stored or merged into valid output.
            if constexpr (Staged)
              wm::load_matrix_sync(a, staged_query + k, D + 16);
            else
              wm::load_matrix_sync(a, queries[batch] + qbase * D + k, D);
            const bool last = tile + 16 > valid;
            if constexpr (Staged)
              wm::load_matrix_sync(b, tail + tile * (D + 16) + k, D + 16);
            else
              wm::load_matrix_sync(b,
                                   last ? tail + k
                                        : keys[batch] +
                                              (start + tile) * row_stride + k,
                                   last ? D : row_stride);
            wm::mma_sync(acc, a, b, acc);
          }
        }
        wm::store_matrix_sync(scores + tile, acc, ScoreStride,
                              wm::mem_row_major);
      }
    }
    __syncthreads();
    if constexpr (Staged && !Compact)
      stage_kv<D, S>(tail, values[batch] + start * row_stride, row_stride,
                     valid);
    const int qr = threadIdx.x / 8, sublane = threadIdx.x % 8;
    const int step = (qbase + qr) / group;
    const int end = past + step + 1, first = ring ? max(0, end - 1024) : 0;
    float maximum = -INFINITY;
    for (int k = sublane; k < S; k += 8)
      if (qbase + qr < rows && k < valid && start + k >= first &&
          start + k < end)
        maximum = fmaxf(maximum, scores[qr * ScoreStride + k]);
    for (int offset = 4; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 8));
    float denominator = 0.0F;
    for (int k = sublane; k < S; k += 8) {
      const float p = qbase + qr < rows && k < valid && start + k >= first &&
                              start + k < end
                          ? expf(scores[qr * ScoreStride + k] - maximum)
                          : 0.0F;
      probability[qr * ProbabilityStride + k] = __float2bfloat16_rn(p);
      denominator += p;
    }
    for (int offset = 4; offset; offset >>= 1)
      denominator += __shfl_xor_sync(0xffffffff, denominator, offset, 8);
    if (sublane == 0 && qbase + qr < rows) {
      const auto offset = (base + qbase + qr) * (D + 2);
      const float old_m = chunk ? partial[offset + D] : -INFINITY;
      const float new_m = fmaxf(old_m, maximum);
      old_factor[qr] = isfinite(old_m) ? expf(old_m - new_m) : 0.0F;
      new_factor[qr] = isfinite(maximum) ? expf(maximum - new_m) : 0.0F;
      const float old_l = chunk ? partial[offset + D + 1] : 0.0F;
      partial[offset + D] = new_m;
      partial[offset + D + 1] =
          old_l * old_factor[qr] + denominator * new_factor[qr];
    }
    if (!Staged && valid % 16)
      for (int i = threadIdx.x; i < 16 * D; i += blockDim.x) {
        const int row = i / D, col = i % D;
        tail[i] =
            tail_start + row < valid
                ? values[batch][(start + tail_start + row) * row_stride + col]
                : __float2bfloat16(0.0F);
      }
    if constexpr (Staged)
      wait_kv();
    else
      __syncthreads();
    auto *tile_output = output_tiles + warp * 256;
    for (int column = (warp % 4) * 16; column < D; column += 64) {
      if constexpr (Compact) {
        stage_kv<64, S>(tail,
                        values[batch] + start * row_stride + (column / 64) * 64,
                        row_stride, valid);
        wait_kv();
      }
      wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
      wm::fill_fragment(acc, 0.0F);
      for (int k = 0; k < valid; k += 16) {
        wm::fragment<wm::matrix_a, 16, 16, 16, BF, wm::row_major> a;
        wm::fragment<wm::matrix_b, 16, 16, 16, BF, wm::row_major> b;
        wm::load_matrix_sync(
            a, probability + (warp / 4) * 16 * ProbabilityStride + k,
            ProbabilityStride);
        const bool last = k + 16 > valid;
        if constexpr (Compact)
          wm::load_matrix_sync(b, tail + k * 80 + (warp % 4) * 16, 80);
        else if constexpr (Staged)
          wm::load_matrix_sync(b, tail + k * (D + 16) + column, D + 16);
        else
          wm::load_matrix_sync(b,
                               last ? tail + column
                                    : values[batch] + (start + k) * row_stride +
                                          column,
                               last ? D : row_stride);
        wm::mma_sync(acc, a, b, acc);
      }
      wm::store_matrix_sync(tile_output, acc, 16, wm::mem_row_major);
      __syncwarp();
      for (int i = lane; i < 256; i += 32) {
        const int row = (warp / 4) * 16 + i / 16;
        if (qbase + row < rows) {
          const auto offset = (base + qbase + row) * (D + 2) + column + i % 16;
          const float old = chunk ? partial[offset] : 0.0F;
          partial[offset] =
              old * old_factor[row] + tile_output[i] * new_factor[row];
        }
      }
      __syncwarp();
      if constexpr (Compact)
        __syncthreads();
    }
    __syncthreads();
  }
}

__global__ void merge_attention(const float *partial, BF *output, int rows,
                                int dim, int partitions) {
  const int row = blockIdx.x;
  extern __shared__ float weights[];
  __shared__ float reduction[256];
  __shared__ float denominator;
  float maximum = -INFINITY;
  for (int p = threadIdx.x; p < partitions; p += blockDim.x)
    maximum = fmaxf(
        maximum,
        partial[(static_cast<std::size_t>(p) * rows + row) * (dim + 2) + dim]);
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + s]);
    __syncthreads();
  }
  maximum = reduction[0];
  float sum = 0.0F;
  for (int p = threadIdx.x; p < partitions; p += blockDim.x) {
    const auto index = (static_cast<std::size_t>(p) * rows + row) * (dim + 2);
    const float w = isfinite(partial[index + dim])
                        ? expf(partial[index + dim] - maximum)
                        : 0.0F;
    weights[p] = w;
    sum += w * partial[index + dim + 1];
  }
  __syncthreads();
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      reduction[threadIdx.x] += reduction[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    denominator = reduction[0];
  __syncthreads();
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    float sum = 0.0F;
    for (int p = 0; p < partitions; ++p)
      sum = fmaf(
          weights[p],
          partial[(static_cast<std::size_t>(p) * rows + row) * (dim + 2) + col],
          sum);
    output[static_cast<std::size_t>(row) * dim + col] =
        __float2bfloat16_rn(sum / denominator);
  }
}

__global__ void reference_softmax(const float *scores, BF *probs,
                                  const int *contexts, int tokens, int heads,
                                  int keys, bool ring) {
  __shared__ float reduction[256];
  const int row = blockIdx.x, group = 16 / heads;
  const int cached = contexts[row / (heads * tokens * group)];
  const int step = row % (tokens * group) / group;
  const int end = (ring ? min(cached, 1023) : cached) + step + 1;
  const int first = ring ? max(0, end - 1024) : 0;
  scores += static_cast<std::size_t>(row) * keys;
  probs += static_cast<std::size_t>(row) * keys;
  float maximum = -INFINITY;
  for (int k = threadIdx.x; k < keys; k += 256)
    if (k >= first && k < end)
      maximum = fmaxf(maximum, scores[k]);
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + s]);
    __syncthreads();
  }
  maximum = reduction[0];
  float sum = 0.0F;
  for (int k = threadIdx.x; k < keys; k += 256) {
    const float p = k >= first && k < end ? expf(scores[k] - maximum) : 0.0F;
    probs[k] = __float2bfloat16_rn(p);
    sum += p;
  }
  // All warps must consume the maximum before lane zero reuses its slot.
  __syncthreads();
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      reduction[threadIdx.x] += reduction[threadIdx.x + s];
    __syncthreads();
  }
  for (int k = threadIdx.x; k < keys; k += 256)
    probs[k] = __float2bfloat16_rn(__bfloat162float(probs[k]) / reduction[0]);
}
template <class F> float time_kernel(F fn) {
  if (std::getenv("GEVVA_ATTENTION_CHECK_ONLY")) {
    fn();
    check(cudaDeviceSynchronize());
    return 0.0F;
  }
  for (int i = 0; i < 5; ++i)
    fn();
  cudaEvent_t a{}, b{};
  check(cudaEventCreate(&a));
  check(cudaEventCreate(&b));
  if (std::getenv("GEVVA_PROFILE_ATTENTION"))
    check(cudaProfilerStart());
  check(cudaEventRecord(a));
  for (int i = 0; i < 30; ++i)
    fn();
  check(cudaEventRecord(b));
  check(cudaEventSynchronize(b));
  float ms{};
  check(cudaEventElapsedTime(&ms, a, b));
  if (std::getenv("GEVVA_PROFILE_ATTENTION"))
    check(cudaProfilerStop());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms * 1000 / 30;
}
} // namespace

void capture_attention(const std::string &path, const void *const *queries,
                       const void *const *keys, const void *const *values,
                       const int *contexts, const void *reference, int sessions,
                       int tokens, int dimension, int heads, int key_tokens,
                       bool ring, cudaStream_t stream) {
  if (std::filesystem::exists(path))
    return;
  check(cudaStreamSynchronize(stream));
  std::vector<int> host_contexts(sessions);
  check(cudaMemcpy(host_contexts.data(), contexts, sessions * sizeof(int),
                   cudaMemcpyDeviceToHost));
  if (const char *minimum = std::getenv("GEVVA_CAPTURE_ATTENTION_MIN_CONTEXT");
      minimum && *std::max_element(host_contexts.begin(), host_contexts.end()) <
                     std::atoi(minimum))
    return;
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot write attention capture");
  Header header{0x31415447, sessions,   tokens, dimension,
                heads,      key_tokens, ring};
  auto write = [&](const void *p, std::size_t bytes) {
    file.write(static_cast<const char *>(p), bytes);
  };
  write(&header, sizeof(header));
  write(host_contexts.data(), sessions * sizeof(int));
  for (int kind = 0; kind < 3; ++kind) {
    std::vector<const void *> pointers(sessions * heads);
    check(cudaMemcpy(pointers.data(),
                     kind == 0   ? queries
                     : kind == 1 ? keys
                                 : values,
                     pointers.size() * sizeof(void *), cudaMemcpyDeviceToHost));
    const int rows = kind == 0 ? tokens * (16 / heads) : key_tokens;
    for (int batch = 0; batch < sessions * heads; ++batch) {
      std::vector<BF> host(static_cast<std::size_t>(rows) * dimension,
                           __float2bfloat16(0.0F));
      const int valid =
          kind == 0 ? rows
                    : (ring ? std::min(host_contexts[batch / heads], 1023)
                            : host_contexts[batch / heads]) +
                          tokens;
      check(cudaMemcpy2D(host.data(), dimension * 2, pointers[batch],
                         (kind == 0 ? dimension : heads * dimension) * 2,
                         dimension * 2, std::min(rows, valid),
                         cudaMemcpyDeviceToHost));
      write(host.data(), host.size() * 2);
    }
  }
  std::vector<BF> out(static_cast<std::size_t>(sessions) * tokens * 16 *
                      dimension);
  check(cudaMemcpy(out.data(), reference, out.size() * 2,
                   cudaMemcpyDeviceToHost));
  write(out.data(), out.size() * 2);
  if (!file)
    throw std::runtime_error("attention capture write failed");
}

void benchmark_tiled_attention(const std::string &path, int batch, bool ragged,
                               int query_tokens) {
  std::ifstream file(path, std::ios::binary);
  Header h{};
  file.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (!file || h.magic != 0x31415447 || (h.dim != 256 && h.dim != 512) ||
      h.batch < 1 || h.batch > 8 || batch < 1 || batch > h.batch ||
      h.tokens < 1 || h.tokens > 5 || h.keys < 1 || h.keys > 262144 ||
      h.heads != (h.dim == 256 ? 8 : 2))
    throw std::runtime_error("invalid attention fixture");
  auto read = [&]<class T>(std::size_t n) {
    std::vector<T> out(n);
    file.read(reinterpret_cast<char *>(out.data()), n * sizeof(T));
    if (!file)
      throw std::runtime_error("truncated attention fixture");
    return out;
  };
  auto contexts = read.operator()<int>(h.batch);
  int grouped = h.tokens * 16 / h.heads;
  auto q = read.operator()<BF>(static_cast<std::size_t>(h.batch) * h.heads *
                               grouped * h.dim);
  auto kh = read.operator()<BF>(static_cast<std::size_t>(h.batch) * h.heads *
                                h.keys * h.dim);
  auto vh = read.operator()<BF>(kh.size());
  auto captured = read.operator()<BF>(q.size());
  if (query_tokens < 0 || query_tokens > h.tokens)
    throw std::runtime_error("query token count exceeds captured positions");
  if (query_tokens && query_tokens != h.tokens) {
    const int new_grouped = query_tokens * 16 / h.heads;
    auto trim = [&](std::vector<BF> &data) {
      std::vector<BF> trimmed(static_cast<std::size_t>(batch) * h.heads *
                              new_grouped * h.dim);
      for (int i = 0; i < batch * h.heads; ++i)
        std::copy_n(data.data() + static_cast<std::size_t>(i) * grouped * h.dim,
                    new_grouped * h.dim,
                    trimmed.data() +
                        static_cast<std::size_t>(i) * new_grouped * h.dim);
      data = std::move(trimmed);
    };
    trim(q);
    trim(captured);
    grouped = new_grouped;
    h.tokens = query_tokens;
  }
  contexts.resize(batch);
  q.resize(static_cast<std::size_t>(batch) * h.heads * grouped * h.dim);
  if (ragged)
    for (int i = 0; i < batch; ++i)
      contexts[i] = std::max(1, contexts[i] - i * 17);
  std::vector<BF> k(static_cast<std::size_t>(batch) * h.keys * h.heads * h.dim),
      v(k.size());
  for (int b = 0; b < batch; ++b)
    for (int head = 0; head < h.heads; ++head)
      for (int t = 0; t < h.keys; ++t)
        for (int d = 0; d < h.dim; ++d) {
          auto dst =
              ((static_cast<std::size_t>(b) * h.keys + t) * h.heads + head) *
                  h.dim +
              d;
          auto src =
              ((static_cast<std::size_t>(b) * h.heads + head) * h.keys + t) *
                  h.dim +
              d;
          k[dst] = kh[src];
          v[dst] = vh[src];
        }
  Arena arena;
  auto *dq = arena.allocate<BF>(q.size() + 16 * h.dim);
  check(cudaMemset(dq, 0, (q.size() + 16 * h.dim) * 2));
  check(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
  auto *dk = arena.upload(k);
  auto *dv = arena.upload(v);
  auto *dc = arena.upload(contexts);
  const int batches = batch * h.heads, rows = batches * grouped;
  std::vector<const BF *> qp(batches), kp(batches), vp(batches), pp(batches);
  std::vector<float *> sp(batches);
  std::vector<BF *> op(batches);
  auto *scores = arena.allocate<float>(static_cast<std::size_t>(rows) * h.keys);
  auto *probs = arena.allocate<BF>(static_cast<std::size_t>(rows) * h.keys);
  auto *baseline = arena.allocate<BF>(q.size());
  auto *tiled = arena.allocate<BF>(q.size());
  for (int i = 0; i < batches; ++i) {
    qp[i] = dq + static_cast<std::size_t>(i) * grouped * h.dim;
    const auto offset =
        (static_cast<std::size_t>(i / h.heads) * h.keys * h.heads +
         i % h.heads) *
        h.dim;
    kp[i] = dk + offset;
    vp[i] = dv + offset;
    pp[i] = probs + static_cast<std::size_t>(i) * grouped * h.keys;
    sp[i] = scores + static_cast<std::size_t>(i) * grouped * h.keys;
    op[i] = baseline + static_cast<std::size_t>(i) * grouped * h.dim;
  }
  auto *dqp = arena.upload(qp);
  auto *dkp = arena.upload(kp);
  auto *dvp = arena.upload(vp);
  auto *dpp = arena.upload(pp);
  auto *dsp = arena.upload(sp);
  auto *dop = arena.upload(op);
  cublasHandle_t handle{};
  blas(cublasCreate(&handle));
  const float one = 1, zero = 0;
  auto reference = [&] {
    blas(cublasGemmBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, h.keys, grouped, h.dim, &one,
        reinterpret_cast<const void *const *>(dkp), CUDA_R_16BF,
        h.heads * h.dim, reinterpret_cast<const void *const *>(dqp),
        CUDA_R_16BF, h.dim, &zero, reinterpret_cast<void *const *>(dsp),
        CUDA_R_32F, h.keys, batches, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    reference_softmax<<<rows, 256>>>(scores, probs, dc, h.tokens, h.heads,
                                     h.keys, h.ring);
    blas(cublasGemmBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, h.dim, grouped, h.keys, &one,
        reinterpret_cast<const void *const *>(dvp), CUDA_R_16BF,
        h.heads * h.dim, reinterpret_cast<const void *const *>(dpp),
        CUDA_R_16BF, h.keys, &zero, reinterpret_cast<void *const *>(dop),
        CUDA_R_16BF, h.dim, batches, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  const float reference_us = time_kernel(reference);
  std::vector<BF> ref(q.size()), observed(q.size());
  check(
      cudaMemcpy(ref.data(), baseline, ref.size() * 2, cudaMemcpyDeviceToHost));
  std::size_t capture_mismatches = 0;
  if (!ragged)
    for (std::size_t i = 0; i < ref.size(); ++i)
      capture_mismatches +=
          __bfloat162float(ref[i]) != __bfloat162float(captured[i]);
  std::vector<BF> direct64;
  auto run = [&]<int D, int S, int Chunks = 1, bool Staged = false,
                 bool Compact = false, int QTiles = 1>() {
    const int partitions = (h.keys + S * Chunks - 1) / (S * Chunks);
    auto *partial = arena.allocate<float>(static_cast<std::size_t>(partitions) *
                                          rows * (D + 2));
    const int shared =
        (Staged ? (S + 16 * QTiles) * ((Compact ? 64 : D) + 16) : 16 * D) * 2 +
        QTiles * (16 * (S + (Staged ? 8 : 0)) * 4 +
                  16 * (S + (Staged ? 16 : 0)) * 2 + 4096 + 128);
    check(cudaFuncSetAttribute(
        partition_attention<D, S, Chunks, Staged, Compact, QTiles>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    auto launch = [&] {
      partition_attention<D, S, Chunks, Staged, Compact, QTiles>
          <<<dim3((grouped + 16 * QTiles - 1) / (16 * QTiles), batches,
                  partitions),
             128 * QTiles, shared>>>(dqp, dkp, dvp, dc, partial, batch,
                                     h.tokens, h.heads, h.ring);
      merge_attention<<<rows, 256, partitions * sizeof(float)>>>(
          partial, tiled, rows, D, partitions);
    };
    const float us = time_kernel(launch);
    check(cudaMemcpy(observed.data(), tiled, observed.size() * 2,
                     cudaMemcpyDeviceToHost));
    if constexpr (!Staged && S == 64 && Chunks == 1)
      direct64 = observed;
    constexpr bool check_staging = Staged && S == 64 && Chunks == 1;
    std::size_t staging_mismatches = 0;
    if constexpr (check_staging)
      for (std::size_t i = 0; i < observed.size(); ++i)
        staging_mismatches +=
            __bfloat162float(observed[i]) != __bfloat162float(direct64.at(i));
    double sq = 0, ref_sq = 0;
    float maximum = 0;
    std::size_t nonfinite = 0, different = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const float x = __bfloat162float(observed[i]),
                  y = __bfloat162float(ref[i]);
      nonfinite += !std::isfinite(x);
      different += x != y;
      maximum = std::max(maximum, std::abs(x - y));
      sq += (x - y) * (x - y);
      ref_sq += y * y;
    }
    std::cout << nlohmann::json{{"fixture", path},
                                {"batch", batch},
                                {"dim", D},
                                {"keys", h.keys},
                                {"tokens", h.tokens},
                                {"ragged", ragged},
                                {"split", S},
                                {"chunks", Chunks},
                                {"staged", Staged},
                                {"compact", Compact},
                                {"query_tiles", QTiles},
                                {"baseline_us", reference_us},
                                {"tiled_us", us},
                                {"speedup", reference_us / us},
                                {"max_abs_error", maximum},
                                {"relative_rms_error",
                                 std::sqrt(sq / std::max(1e-30, ref_sq))},
                                {"different_elements", different},
                                {"nonfinite", nonfinite},
                                {"check_only",
                                 std::getenv("GEVVA_ATTENTION_CHECK_ONLY") !=
                                     nullptr},
                                {"staging_checked", check_staging},
                                {"staging_mismatches", staging_mismatches},
                                {"capture_checked", !ragged},
                                {"capture_mismatches", capture_mismatches}}
                     .dump()
              << std::endl;
    if (nonfinite)
      throw std::runtime_error("nonfinite tiled attention");
    if (staging_mismatches)
      throw std::runtime_error("staged attention changed same-tile output");
  };
  if (h.dim == 256) {
    run.operator()<256, 64>();
    run.operator()<256, 128>();
    run.operator()<256, 256>();
    run.operator()<256, 128, 2>();
    run.operator()<256, 128, 4>();
    run.operator()<256, 128, 8>();
    run.operator()<256, 32, 1, true>();
    run.operator()<256, 64, 1, true>();
    run.operator()<256, 64, 4, true>();
    run.operator()<256, 64, 1, true, true>();
  } else {
    run.operator()<512, 64>();
    run.operator()<512, 128>();
    run.operator()<512, 256>();
    run.operator()<512, 128, 2>();
    run.operator()<512, 128, 4>();
    run.operator()<512, 128, 8>();
    run.operator()<512, 32, 1, true>();
    run.operator()<512, 64, 1, true>();
    run.operator()<512, 64, 4, true>();
    run.operator()<512, 64, 1, true, true>();
    run.operator()<512, 64, 1, true, true, 2>();
    run.operator()<512, 64, 1, true, true, 3>();
  }
  cublasDestroy(handle);
}
void prepare_serving_tiled_attention() {
  static const bool ready = [] {
    auto prepare = [&]<int D, int S, bool Compact, int QTiles>() {
      const int shared = (S + 16 * QTiles) * ((Compact ? 64 : D) + 16) * 2 +
          QTiles * (16 * (S + 8) * 4 + 16 * (S + 16) * 2 + 4096 + 128);
      check(cudaFuncSetAttribute(partition_attention<D, S, 1, true, Compact, QTiles>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    };
    prepare.operator()<256, 32, false, 1>();
    prepare.operator()<256, 64, false, 1>();
    prepare.operator()<256, 64, true, 1>();
    prepare.operator()<512, 64, true, 2>();
    return true;
  }();
  (void)ready;
}

void launch_serving_tiled_attention(const void* const* queries,
    const void* const* keys, const void* const* values, const int* contexts,
    void* output, float* partial, int batch, int tokens, int dim, int key_tokens,
    cudaStream_t stream) {
  auto launch = [&]<int D, int S, bool Compact, int QTiles>() {
    constexpr int heads = D == 256 ? 8 : 2;
    const int grouped = tokens * 16 / heads, rows = batch * tokens * 16;
    const int partitions = (key_tokens + S - 1) / S;
    const int shared = (S + 16 * QTiles) * ((Compact ? 64 : D) + 16) * 2 +
        QTiles * (16 * (S + 8) * 4 + 16 * (S + 16) * 2 + 4096 + 128);
    partition_attention<D, S, 1, true, Compact, QTiles><<<
        dim3((grouped + 16 * QTiles - 1) / (16 * QTiles), batch * heads, partitions),
        128 * QTiles, shared, stream>>>(reinterpret_cast<const BF* const*>(queries),
        reinterpret_cast<const BF* const*>(keys), reinterpret_cast<const BF* const*>(values),
        contexts, partial, batch, tokens, heads, D == 256);
    merge_attention<<<rows, 256, partitions * sizeof(float), stream>>>(
        partial, static_cast<BF*>(output), rows, D, partitions);
    check(cudaGetLastError());
  };
  if (dim == 512) launch.operator()<512, 64, true, 2>();
  else if (batch == 1 && key_tokens <= 512) launch.operator()<256, 32, false, 1>();
  else if (batch == 1) launch.operator()<256, 64, false, 1>();
  else launch.operator()<256, 64, true, 1>();
}
} // namespace gevva
