#include "g4/gpu.hpp"
#include "cuda_errors.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace {
std::string uuid_string(const cudaUUID_t& uuid) {
  char out[37];
  const auto* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
  std::snprintf(out, sizeof(out),
                "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
                b[10], b[11], b[12], b[13], b[14], b[15]);
  return out;
}

}  // namespace

namespace g4 {
using detail::check;

GpuInfo select_target_gpu() {
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");

  for (int ordinal = 0; ordinal < count; ++ordinal) {
    char pci[32]{};
    check(cudaDeviceGetPCIBusId(pci, sizeof(pci), ordinal), "cudaDeviceGetPCIBusId");
    std::string normalized_pci = pci;
    std::transform(normalized_pci.begin(), normalized_pci.end(), normalized_pci.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized_pci != kTargetPciBusId) continue;

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, ordinal), "cudaGetDeviceProperties");
    if (prop.major != kTargetComputeMajor || prop.minor != kTargetComputeMinor) {
      throw std::runtime_error("GPU at " + std::string(pci) + " is not SM 12.0");
    }
    if (prop.totalGlobalMem < kMinimumTargetVram) {
      throw std::runtime_error("GPU at " + std::string(pci) + " has less than 90 GiB VRAM");
    }
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    check(cudaFree(nullptr), "CUDA context initialization");
    return {ordinal, prop.name, normalized_pci, uuid_string(prop.uuid), prop.major, prop.minor,
            prop.totalGlobalMem};
  }

  throw std::runtime_error(
      "target RTX PRO 6000 Blackwell not visible at PCI 0000:0f:00.0; refusing to use another GPU");
}

}  // namespace g4
