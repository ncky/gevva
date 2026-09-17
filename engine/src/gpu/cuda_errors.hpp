#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace g4::detail {

inline void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

inline void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
  }
}

}  // namespace g4::detail
