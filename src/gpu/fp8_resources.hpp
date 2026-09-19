#pragma once

#include <cstddef>
#include <vector>

namespace gevva::detail {

// Own the allocations independently of the runner's mutable/swapped views.
// Runtime is injectable so every partial-initialization path can be tested
// without a GPU or model weights.
template <class Runtime>
struct Fp8Resources {
  using Event = typename Runtime::Event;
  void* activation{};
  void* activation_scales{};
  void* padded_output{};
  void* secondary_activation{};
  void* secondary_activation_scales{};
  void* unit_scale{};
  Event activation_ready{};
  Event secondary_activation_ready{};

  template <class InitializeWorkspace>
  Fp8Resources(std::size_t activation_bytes, std::size_t tokens,
               std::size_t output_bytes, InitializeWorkspace initialize_workspace) {
    try {
      Runtime::allocate(&activation, activation_bytes, "cudaMalloc(FP8 activation)");
      Runtime::allocate(&activation_scales, tokens * sizeof(float), "cudaMalloc(FP8 activation scales)");
      std::vector<float> initial_scales(tokens, 1.0F);
      Runtime::copy(activation_scales, initial_scales.data(), tokens * sizeof(float),
                    "initialize FP8 activation scales");
      Runtime::allocate(&padded_output, output_bytes, "cudaMalloc(FP8 padded output)");
      Runtime::allocate(&secondary_activation, activation_bytes, "cudaMalloc(secondary FP8 activation)");
      Runtime::allocate(&secondary_activation_scales, tokens * sizeof(float),
                        "cudaMalloc(secondary FP8 activation scales)");
      initialize_workspace();
      Runtime::allocate(&unit_scale, sizeof(float), "cudaMalloc(FP8 unit scale)");
      const float one = 1.0F;
      Runtime::copy(unit_scale, &one, sizeof(one), "copy FP8 unit scale");
      Runtime::create_event(&activation_ready, "create FP8 activation ready event");
      Runtime::create_event(&secondary_activation_ready, "create secondary FP8 activation ready event");
    } catch (...) {
      release();
      throw;
    }
  }

  Fp8Resources(const Fp8Resources&) = delete;
  Fp8Resources& operator=(const Fp8Resources&) = delete;
  ~Fp8Resources() { release(); }

 private:
  void release() noexcept {
    if (secondary_activation_ready) Runtime::destroy_event(secondary_activation_ready);
    if (activation_ready) Runtime::destroy_event(activation_ready);
    for (void* pointer : {unit_scale, secondary_activation_scales, secondary_activation,
                          padded_output, activation_scales, activation})
      if (pointer) Runtime::free(pointer);
  }
};

}  // namespace gevva::detail
